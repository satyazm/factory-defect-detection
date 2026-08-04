"""
Fetch the NEU-DET (steel surface defect) dataset in YOLO format and lay it
out under dataset/{train,validation,test}/{images,labels}.

NEU-DET has no single stable public download URL, so this script supports
two sources you point it at explicitly rather than guessing one for you:

1. Roboflow Universe (recommended — already in YOLOv8 txt format)
   - Create a free account at https://roboflow.com, search "NEU-DET",
     and grab your workspace API key from Settings > API Keys.
   - Run:
       export ROBOFLOW_API_KEY=your_key_here
       python utils/download_dataset.py --source roboflow --project neu-det --version 1

2. Kaggle (raw NEU-DET, PASCAL VOC XML annotations — needs conversion)
   - Set up the Kaggle CLI (~/.kaggle/kaggle.json), then run:
       python utils/download_dataset.py --source kaggle --dataset <kaggle-dataset-slug>
   - You will still need to convert VOC XML to YOLO txt; see
     preprocessing/voc_to_yolo.py (not included by default — ask if you need it).

Either way, the end result should be:
  dataset/train/images/*.jpg      dataset/train/labels/*.txt
  dataset/validation/images/*.jpg dataset/validation/labels/*.txt
  dataset/test/images/*.jpg       dataset/test/labels/*.txt
"""
import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path

DATASET_ROOT = Path(__file__).resolve().parent.parent / "dataset"


def download_roboflow(project: str, version: int, api_key: str) -> None:
    try:
        from roboflow import Roboflow
    except ImportError:
        sys.exit("Missing dependency. Run: pip install roboflow")

    rf = Roboflow(api_key=api_key)
    ws = rf.workspace()
    proj = ws.project(project)
    dataset = proj.version(version).download("yolov8", location=str(DATASET_ROOT / "_roboflow_download"))

    for split, target in [("train", "train"), ("valid", "validation"), ("test", "test")]:
        src_images = Path(dataset.location) / split / "images"
        src_labels = Path(dataset.location) / split / "labels"
        if not src_images.exists():
            continue
        dst_images = DATASET_ROOT / target / "images"
        dst_labels = DATASET_ROOT / target / "labels"
        dst_images.mkdir(parents=True, exist_ok=True)
        dst_labels.mkdir(parents=True, exist_ok=True)
        for f in src_images.glob("*"):
            shutil.copy(f, dst_images / f.name)
        for f in src_labels.glob("*"):
            shutil.copy(f, dst_labels / f.name)

    print(f"Done. Images/labels copied into {DATASET_ROOT}")


def download_kaggle(dataset_slug: str) -> None:
    try:
        import kaggle
    except ImportError:
        sys.exit("Missing dependency. Run: pip install kaggle, and configure ~/.kaggle/kaggle.json")

    raw_dir = DATASET_ROOT / "_kaggle_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    kaggle.api.dataset_download_files(dataset_slug, path=str(raw_dir), unzip=True)
    print(f"Downloaded raw NEU-DET files to {raw_dir}.")
    print("These are PASCAL VOC XML annotations — convert them to YOLO txt "
          "before training (see docstring at top of this file).")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=["roboflow", "kaggle"], required=True)
    parser.add_argument("--project", default="neu-det", help="Roboflow project slug")
    parser.add_argument("--version", type=int, default=1, help="Roboflow dataset version")
    parser.add_argument("--dataset", help="Kaggle dataset slug, e.g. user/neu-det")
    args = parser.parse_args()

    if args.source == "roboflow":
        api_key = os.environ.get("ROBOFLOW_API_KEY")
        if not api_key:
            sys.exit("Set ROBOFLOW_API_KEY in your environment first.")
        download_roboflow(args.project, args.version, api_key)
    else:
        if not args.dataset:
            sys.exit("--dataset <kaggle-slug> is required for --source kaggle")
        download_kaggle(args.dataset)


if __name__ == "__main__":
    main()
