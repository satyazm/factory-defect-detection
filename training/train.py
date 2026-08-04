"""
Train a YOLOv8 defect detector on dataset/data.yaml.

Usage:
    python training/train.py --model yolov8n.pt --epochs 100 --imgsz 640
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

from _data_config import build_absolute_data_yaml

RUNS_DIR = Path(__file__).resolve().parent.parent / "models" / "yolov8"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="yolov8n.pt", help="base checkpoint (n/s/m/l/x) or path to resume from")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None, help="e.g. 0 for GPU 0, or 'cpu'")
    parser.add_argument("--name", default="defect_detector", help="run name under models/yolov8/")
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(
        data=build_absolute_data_yaml(),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(RUNS_DIR),
        name=args.name,
    )


if __name__ == "__main__":
    main()
