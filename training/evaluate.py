"""
Evaluate a trained checkpoint on the validation/test split and print
mAP50, mAP50-95, precision/recall per class.

Usage:
    python training/evaluate.py --weights models/yolov8/defect_detector/weights/best.pt --split test
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

DATA_YAML = Path(__file__).resolve().parent.parent / "dataset" / "data.yaml"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, help="path to trained .pt checkpoint")
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    model = YOLO(args.weights)
    metrics = model.val(data=str(DATA_YAML), split=args.split, imgsz=args.imgsz)

    print("\n--- Summary ---")
    print(f"mAP50:    {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision (mean): {metrics.box.mp:.4f}")
    print(f"Recall (mean):    {metrics.box.mr:.4f}")


if __name__ == "__main__":
    main()
