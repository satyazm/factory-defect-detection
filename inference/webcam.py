"""Convenience wrapper: run real-time detection on the default webcam.

Usage:
    python inference/webcam.py --weights models/yolov8/defect_detector/weights/best.pt
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from inference.realtime import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--no-alert", action="store_true")
    args = parser.parse_args()
    run(args.weights, source="0", conf=args.conf, camera_id="webcam", alert=not args.no_alert)


if __name__ == "__main__":
    main()
