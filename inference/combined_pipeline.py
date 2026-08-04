"""
Combined real-time pipeline: YOLO known-defect detection (Module 1) and
PatchCore anomaly detection (Module 2) run on the same camera/video feed
side by side.

    Camera → OpenCV frame
               │
    ┌──────────┴──────────┐
    ▼                      ▼
  YOLO                  PatchCore
  (known defects)       (unknown anomalies)
    │                      │
  boxes                 heatmap + score
    └──────────┬───────────┘
               ▼
        SQLite (detections + anomalies tables)

Usage:
    python inference/combined_pipeline.py \
        --yolo-weights models/yolov8/defect_detector/weights/best.pt \
        --patchcore-weights models/patchcore/bottle/weights/torch/model.pt \
        --source 0

--source also accepts a folder of images, which loops through them
indefinitely — useful for demoing continuous detection without a real
camera:
    python inference/combined_pipeline.py --yolo-weights ... --patchcore-weights ... --source dataset/test/images

Displays known-defect boxes on the left and the anomaly heatmap on the
right. Press 'q' to quit.
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from database.db import init_db, log_anomaly, log_detection
from inference.detect_anomaly import AnomalyDetector
from inference.detect_known import KnownDefectDetector
from inference.video_source import open_capture


def run(
    yolo_weights: str,
    patchcore_weights: str,
    source: str,
    conf: float,
    camera_id: str,
    image_delay: float = 1.5,
) -> None:
    init_db()

    known_detector = KnownDefectDetector(yolo_weights, conf=conf)
    anomaly_detector = AnomalyDetector(patchcore_weights)

    cap = open_capture(source, image_folder_delay_seconds=image_delay)
    if not cap.isOpened():
        sys.exit(f"Could not open source: {source}")

    prev_frame_time = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Stream ended or frame grab failed.")
            break

        known_result = known_detector.predict(frame)
        anomaly_result = anomaly_detector.predict(frame)

        for det in known_result["detections"]:
            log_detection(
                class_name=det["class_name"],
                confidence=det["confidence"],
                bbox_xyxy=det["bbox"],
                camera_id=camera_id,
            )

        log_anomaly(
            score=anomaly_result["score"],
            is_anomalous=anomaly_result["is_anomalous"],
            camera_id=camera_id,
        )

        left = known_result["annotated"]
        right = cv2.resize(anomaly_result["heatmap"], (left.shape[1], left.shape[0]))

        status_color = (0, 0, 255) if anomaly_result["is_anomalous"] else (0, 200, 0)
        status_text = (
            f"Anomaly score: {anomaly_result['score']:.2f} "
            f"({'ABNORMAL' if anomaly_result['is_anomalous'] else 'NORMAL'})"
        )
        cv2.putText(right, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        now = time.time()
        fps = 1.0 / max(now - prev_frame_time, 1e-6)
        prev_frame_time = now
        cv2.putText(left, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        combined = np.hstack([left, right])
        cv2.imshow("Known Defects (left) | Anomaly Heatmap (right)", combined)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--yolo-weights", required=True, help="path to trained YOLO .pt checkpoint")
    parser.add_argument("--patchcore-weights", required=True, help="path to exported PatchCore model.pt")
    parser.add_argument("--source", default="0", help="webcam index, video file path, RTSP URL, or folder of images")
    parser.add_argument("--conf", type=float, default=0.5, help="YOLO confidence threshold")
    parser.add_argument("--camera-id", default="default")
    parser.add_argument("--image-delay", type=float, default=1.5, help="seconds between frames when --source is a folder of images")
    args = parser.parse_args()

    run(args.yolo_weights, args.patchcore_weights, args.source, args.conf, args.camera_id, args.image_delay)


if __name__ == "__main__":
    main()
