"""
Real-time known-defect detection over a webcam, video file, or RTSP
stream (YOLO only — for both known-defect + anomaly detection together,
use inference/combined_pipeline.py).

Usage:
    python inference/realtime.py --weights models/yolov8/defect_detector/weights/best.pt --source 0
    python inference/realtime.py --weights ... --source path/to/video.mp4
    python inference/realtime.py --weights ... --source rtsp://user:pass@camera-ip/stream

Each detection above --conf is drawn on-screen and logged to the SQLite
database; defect counts and recent detections show up live on the
Streamlit dashboard (dashboard/app.py). Press 'q' to quit.
"""
import argparse
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from database.db import init_db, log_detection
from inference.detect_known import KnownDefectDetector


def parse_source(source: str):
    """Webcam indices arrive as strings ('0'); everything else (file path, RTSP URL) stays a string."""
    return int(source) if source.isdigit() else source


def run(weights: str, source: str, conf: float, camera_id: str) -> None:
    init_db()

    detector = KnownDefectDetector(weights, conf=conf)
    cap = cv2.VideoCapture(parse_source(source))
    if not cap.isOpened():
        sys.exit(f"Could not open source: {source}")

    prev_frame_time = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Stream ended or frame grab failed.")
            break

        result = detector.predict(frame)
        annotated = result["annotated"]

        now = time.time()
        fps = 1.0 / max(now - prev_frame_time, 1e-6)
        prev_frame_time = now
        cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        for det in result["detections"]:
            log_detection(
                class_name=det["class_name"],
                confidence=det["confidence"],
                bbox_xyxy=det["bbox"],
                camera_id=camera_id,
            )

        cv2.imshow("Factory Defect Detection", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", required=True, help="path to trained .pt checkpoint")
    parser.add_argument("--source", default="0", help="webcam index, video file path, or RTSP URL")
    parser.add_argument("--conf", type=float, default=0.5, help="confidence threshold")
    parser.add_argument("--camera-id", default="default", help="label for this camera in the database")
    args = parser.parse_args()

    run(args.weights, args.source, args.conf, args.camera_id)


if __name__ == "__main__":
    main()
