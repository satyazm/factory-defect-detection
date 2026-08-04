"""
Known-defect detection: thin wrapper around a trained YOLO model.

inference/realtime.py and inference/combined_pipeline.py both import
KnownDefectDetector so the prediction logic lives in exactly one place.

Standalone usage (single image):
    python inference/detect_known.py --weights models/yolov8/defect_detector/weights/best.pt --image path/to/image.jpg
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from ultralytics import YOLO


class KnownDefectDetector:
    def __init__(self, weights_path: str | Path, conf: float = 0.5):
        self.model = YOLO(str(weights_path))
        self.conf = conf

    def predict(self, frame_bgr: np.ndarray) -> dict:
        results = self.model.predict(frame_bgr, conf=self.conf, verbose=False)[0]
        annotated = results.plot()

        detections = []
        for box in results.boxes:
            cls_name = self.model.names[int(box.cls.item())]
            confidence = float(box.conf.item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append({"class_name": cls_name, "confidence": confidence, "bbox": (x1, y1, x2, y2)})

        return {"annotated": annotated, "detections": detections}


def main():
    import argparse

    import cv2

    parser = argparse.ArgumentParser(description="Run known-defect (YOLO) detection on a single image file.")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--out", default="known_defects.jpg")
    args = parser.parse_args()

    frame = cv2.imread(args.image)
    if frame is None:
        raise SystemExit(f"Could not read image: {args.image}")

    detector = KnownDefectDetector(args.weights, conf=args.conf)
    result = detector.predict(frame)
    for det in result["detections"]:
        print(f"{det['class_name']}: {det['confidence']:.0%} @ {det['bbox']}")
    cv2.imwrite(args.out, result["annotated"])
    print(f"Annotated image saved to {args.out}")


if __name__ == "__main__":
    main()
