"""
Fetch the NEU-DET (steel surface defect) dataset in YOLO format and lay it
out under dataset/{train,validation,test}/{images,labels}.

NEU-DET has no single stable public download URL, so pick whichever of
these fits how you already have (or want) access:

1. Manual Kaggle download (no API token needed — verified working)
   - Download the "NEU Surface Defect Database" zip by hand from
     https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database
     (just needs you logged into Kaggle in the browser) and unzip it.
   - It ships as PASCAL VOC XML annotations under train/validation only
     (no test split). Convert + carve out a test split with:
       python preprocessing/voc_to_yolo.py --source ~/Downloads/NEU-DET

2. Roboflow Universe (already in YOLOv8 txt format, needs an API key)
   - Create a free account at https://roboflow.com, search "NEU-DET",
     and grab your workspace API key from Settings > API Keys.
   - Run:
       export ROBOFLOW_API_KEY=your_key_here
       python utils/download_dataset.py --source roboflow --project neu-det --version 1

3. Kaggle CLI (scripted download of the same dataset as option 1)
   - Set up the Kaggle CLI (~/.kaggle/kaggle.json), then run:
       python utils/download_dataset.py --source kaggle --dataset kaustubhdikshit/neu-surface-defect-database
   - Then convert with preprocessing/voc_to_yolo.py as in option 1.

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
    print(f"These are PASCAL VOC XML annotations — convert them with:\n"
          f"  python preprocessing/voc_to_yolo.py --source {raw_dir}")


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
