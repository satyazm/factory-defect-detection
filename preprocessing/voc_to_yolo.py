"""
Convert a manually-downloaded NEU-DET dataset (PASCAL VOC XML
annotations, per-class image folders) into this repo's YOLO layout:
dataset/{train,validation,test}/{images,labels}.

Use this instead of utils/download_dataset.py's Roboflow path when you
downloaded the Kaggle "NEU Surface Defect Database" zip by hand — that
version ships as:

    <source_root>/train/images/<class>/*.jpg
    <source_root>/train/annotations/<stem>.xml
    <source_root>/validation/images/<class>/*.jpg
    <source_root>/validation/annotations/<stem>.xml

(PASCAL VOC XML, one <object> per defect instance, filename stem shared
between an image and its annotation.) The dataset ships with no test
split, so this script carves one out of "validation" — 1 in 4 images per
class, deterministic by sorted filename, so validation and test stay
disjoint.

Images with no matching annotation (or annotations with no matching
image — both occur in this dataset) are skipped with a warning rather
than failing the whole conversion.

Usage:
    python preprocessing/voc_to_yolo.py --source ~/Downloads/NEU-DET
"""
import argparse
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

CLASS_NAMES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}

DATASET_ROOT = Path(__file__).resolve().parent.parent / "dataset"
TEST_HOLDOUT_EVERY = 4  # 1 in 4 validation images per class -> test


def convert_annotation(xml_path: Path) -> list[str]:
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    width = float(size.findtext("width"))
    height = float(size.findtext("height"))

    lines = []
    for obj in root.findall("object"):
        name = obj.findtext("name")
        if name not in CLASS_TO_ID:
            print(f"  skip unknown class '{name}' in {xml_path.name}")
            continue
        box = obj.find("bndbox")
        xmin, ymin, xmax, ymax = (float(box.findtext(tag)) for tag in ("xmin", "ymin", "xmax", "ymax"))
        cx = ((xmin + xmax) / 2) / width
        cy = ((ymin + ymax) / 2) / height
        w = (xmax - xmin) / width
        h = (ymax - ymin) / height
        lines.append(f"{CLASS_TO_ID[name]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


def write_example(image_path: Path, xml_path: Path, images_dir: Path, labels_dir: Path) -> None:
    lines = convert_annotation(xml_path)
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_path, images_dir / image_path.name)
    (labels_dir / f"{image_path.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))


def collect_examples(source_split_dir: Path) -> list[tuple[Path, Path]]:
    images_root = source_split_dir / "images"
    annotations_root = source_split_dir / "annotations"
    examples = []
    for class_dir in sorted(p for p in images_root.iterdir() if p.is_dir()):
        for image_path in sorted(class_dir.glob("*.jpg")):
            xml_path = annotations_root / f"{image_path.stem}.xml"
            if not xml_path.exists():
                print(f"  skip {image_path.name} — no matching annotation")
                continue
            examples.append((image_path, xml_path))
    return examples


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, type=Path, help="path to the extracted NEU-DET folder")
    args = parser.parse_args()
    source = args.source.expanduser()

    print("Converting train split...")
    train_examples = collect_examples(source / "train")
    for image_path, xml_path in train_examples:
        write_example(image_path, xml_path, DATASET_ROOT / "train" / "images", DATASET_ROOT / "train" / "labels")
    print(f"  {len(train_examples)} train examples")

    print(f"Converting validation split (holding out 1-in-{TEST_HOLDOUT_EVERY} per class for test)...")
    val_examples_by_class: dict[str, list[tuple[Path, Path]]] = {}
    for image_path, xml_path in collect_examples(source / "validation"):
        val_examples_by_class.setdefault(image_path.parent.name, []).append((image_path, xml_path))

    val_count = test_count = 0
    for class_name, examples in val_examples_by_class.items():
        for i, (image_path, xml_path) in enumerate(sorted(examples)):
            if i % TEST_HOLDOUT_EVERY == 0:
                write_example(image_path, xml_path, DATASET_ROOT / "test" / "images", DATASET_ROOT / "test" / "labels")
                test_count += 1
            else:
                write_example(image_path, xml_path, DATASET_ROOT / "validation" / "images", DATASET_ROOT / "validation" / "labels")
                val_count += 1
    print(f"  {val_count} validation examples, {test_count} test examples")
    print(f"\nDone. Populated {DATASET_ROOT}/{{train,validation,test}}/{{images,labels}}")


if __name__ == "__main__":
    main()
