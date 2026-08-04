"""
Offline augmentation for a YOLO-format dataset (images/ + labels/ with
matching filenames, YOLO txt: class cx cy w h, all normalized 0-1).

Generates N augmented copies per source image and writes them alongside
the originals so ultralytics' own train-time augmentation (mosaic, flip,
HSV jitter, etc.) has more base variety to work with — useful when a
defect class is underrepresented (e.g. rare defect types in NEU-DET).
"""
import argparse
from pathlib import Path

import albumentations as A
import cv2

TRANSFORM = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomBrightnessContrast(p=0.4),
        A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, p=0.4),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.GaussNoise(p=0.2),
    ],
    bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=0.3),
)


def read_yolo_labels(label_path: Path):
    boxes, classes = [], []
    if not label_path.exists():
        return boxes, classes
    for line in label_path.read_text().strip().splitlines():
        if not line.strip():
            continue
        cls, cx, cy, w, h = line.split()
        boxes.append([float(cx), float(cy), float(w), float(h)])
        classes.append(int(cls))
    return boxes, classes


def write_yolo_labels(label_path: Path, boxes, classes) -> None:
    lines = [f"{c} {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}" for c, b in zip(classes, boxes)]
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""))


def augment_split(images_dir: Path, labels_dir: Path, copies: int) -> None:
    image_paths = [p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    for img_path in image_paths:
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        label_path = labels_dir / f"{img_path.stem}.txt"
        boxes, classes = read_yolo_labels(label_path)

        for i in range(copies):
            result = TRANSFORM(image=image, bboxes=boxes, class_labels=classes)
            out_img = img_path.with_name(f"{img_path.stem}_aug{i}{img_path.suffix}")
            out_lbl = labels_dir / f"{img_path.stem}_aug{i}.txt"
            cv2.imwrite(str(out_img), result["image"])
            write_yolo_labels(out_lbl, result["bboxes"], result["class_labels"])

    print(f"Augmented {len(image_paths)} images x{copies} copies in {images_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True, type=Path, help="e.g. dataset/train/images")
    parser.add_argument("--labels", required=True, type=Path, help="e.g. dataset/train/labels")
    parser.add_argument("--copies", type=int, default=2, help="augmented copies per source image")
    args = parser.parse_args()
    augment_split(args.images, args.labels, args.copies)


if __name__ == "__main__":
    main()
