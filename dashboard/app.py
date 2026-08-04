"""
Streamlit dashboard: live detection feed + production stats.

Panel 1 — known defects (YOLO, Module 1): "what type of defect is this?"
Panel 2 — unknown anomalies (PatchCore, Module 2): "does this look abnormal at all?"

Each panel has its own video source, since the two models were trained
on completely different image domains (NEU-DET steel surfaces vs.
MVTec bottle photos) — a source that suits one panel is meaningless (or
actively misleading) for the other.

Usage:
    streamlit run dashboard/app.py
"""
import sys
import time
from pathlib import Path

import cv2
import pandas as pd
import plotly.express as px
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from database.db import (
    anomaly_stats_today,
    defect_counts_today,
    init_db,
    log_anomaly,
    log_detection,
    recent_anomalies,
    recent_detections,
)
from inference.video_source import guess_ground_truth_label, open_capture

# Ground-truth labels are guessed from filenames of this repo's own demo
# folders (preprocessing/voc_to_yolo.py's NEU-DET output; the flattened
# MVTec bottle test split) — meaningless for other sources, in which
# case guess_ground_truth_label just returns None and nothing shows.
NEU_DET_CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
MVTEC_BOTTLE_CLASSES = ["good", "broken_small", "broken_large", "contamination"]

st.set_page_config(page_title="Factory Defect Detection", layout="wide")
init_db()

st.title("Factory Defect Detection — Live Dashboard")

with st.sidebar:
    st.header("Configuration")

    enable_known = st.checkbox("Panel 1: known-defect detection (YOLO)", value=True)
    yolo_weights = st.text_input(
        "YOLO weights", str(REPO_ROOT / "models/yolov8/defect_detector/weights/best.pt")
    )
    known_source = st.text_input(
        "Panel 1 video source (webcam / file / RTSP / folder)",
        str(REPO_ROOT / "dataset/test/images"),
    )

    enable_anomaly = st.checkbox("Panel 2: unknown-anomaly detection (PatchCore)", value=True)
    patchcore_weights = st.text_input(
        "PatchCore weights", str(REPO_ROOT / "models/patchcore/bottle/weights/torch/model.pt")
    )
    anomaly_source = st.text_input(
        "Panel 2 video source (webcam / file / RTSP / folder)",
        str(REPO_ROOT / "dataset/mvtec_ad/bottle/test_flat"),
    )

    image_delay = st.number_input(
        "Seconds per image (folder sources only)", min_value=0.1, max_value=10.0, value=1.5, step=0.1
    )
    conf_threshold = st.slider("YOLO confidence threshold", 0.1, 0.95, 0.5, 0.05)
    camera_id = st.text_input("Camera ID", "default")
    run_stream = st.toggle("Start live feed")

col_known, col_anomaly = st.columns(2)

with col_known:
    st.subheader("Panel 1 — Known Defects (YOLO)")
    known_alert_placeholder = st.empty()
    known_gt_placeholder = st.empty()
    known_frame_placeholder = st.empty()

with col_anomaly:
    st.subheader("Panel 2 — Unknown Anomalies (PatchCore)")
    anomaly_alert_placeholder = st.empty()
    anomaly_gt_placeholder = st.empty()
    anomaly_original_col, anomaly_heatmap_col = st.columns(2)
    with anomaly_original_col:
        st.caption("Original")
        anomaly_original_placeholder = st.empty()
    with anomaly_heatmap_col:
        st.caption("Anomaly Heatmap")
        anomaly_frame_placeholder = st.empty()

fps_placeholder = st.empty()

if run_stream:
    known_detector = None
    anomaly_detector = None

    if enable_known:
        try:
            from inference.detect_known import KnownDefectDetector

            known_detector = KnownDefectDetector(yolo_weights, conf=conf_threshold)
        except ImportError:
            st.error("Panel 1 needs `ultralytics` installed (`pip install ultralytics`).")
        except Exception as exc:
            st.error(f"Could not load YOLO weights at '{yolo_weights}': {exc}")

    if enable_anomaly:
        try:
            from inference.detect_anomaly import AnomalyDetector

            anomaly_detector = AnomalyDetector(patchcore_weights)
        except ImportError:
            st.error("Panel 2 needs `anomalib` installed (`pip install anomalib`) — or uncheck it in the sidebar to run Panel 1 only.")
        except Exception as exc:
            st.error(f"Could not load PatchCore weights at '{patchcore_weights}': {exc}")

    if not known_detector and not anomaly_detector:
        st.warning("Enable at least one panel in the sidebar to start streaming.")
        st.stop()

    known_cap = None
    anomaly_cap = None

    if known_detector:
        known_cap = open_capture(known_source, image_folder_delay_seconds=image_delay)
        if not known_cap.isOpened():
            st.error(f"Could not open Panel 1 video source: {known_source}")
            st.stop()

    if anomaly_detector:
        anomaly_cap = open_capture(anomaly_source, image_folder_delay_seconds=image_delay)
        if not anomaly_cap.isOpened():
            st.error(f"Could not open Panel 2 video source: {anomaly_source}")
            st.stop()

    prev_time = time.time()
    while run_stream:
        known_ok = anomaly_ok = False

        if known_cap:
            known_ok, known_frame = known_cap.read()
            if known_ok:
                known_result = known_detector.predict(known_frame)
                for det in known_result["detections"]:
                    log_detection(det["class_name"], det["confidence"], det["bbox"], camera_id=camera_id)

                gt = guess_ground_truth_label(getattr(known_cap, "last_path", None), NEU_DET_CLASSES)
                known_gt_placeholder.caption(f"Ground truth: {gt}" if gt else "")

                if known_result["detections"]:
                    labels = ", ".join(f"{d['class_name']} ({d['confidence']:.0%})" for d in known_result["detections"])
                    known_alert_placeholder.error(f"⚠️ Defect detected: {labels}")
                else:
                    known_alert_placeholder.success("✅ No known defects in current frame")

                known_frame_placeholder.image(cv2.cvtColor(known_result["annotated"], cv2.COLOR_BGR2RGB), channels="RGB")
            else:
                known_alert_placeholder.warning("Panel 1 stream ended or frame grab failed.")

        if anomaly_cap:
            anomaly_ok, anomaly_frame = anomaly_cap.read()
            if anomaly_ok:
                anomaly_result = anomaly_detector.predict(anomaly_frame)
                log_anomaly(anomaly_result["score"], anomaly_result["is_anomalous"], camera_id=camera_id)

                gt = guess_ground_truth_label(getattr(anomaly_cap, "last_path", None), MVTEC_BOTTLE_CLASSES)
                anomaly_gt_placeholder.caption(f"Ground truth: {gt}" if gt else "")

                if anomaly_result["is_anomalous"]:
                    anomaly_alert_placeholder.error(f"🔴 Abnormal — score {anomaly_result['score']:.2f}")
                else:
                    anomaly_alert_placeholder.success(f"🟢 Normal — score {anomaly_result['score']:.2f}")

                anomaly_original_placeholder.image(cv2.cvtColor(anomaly_frame, cv2.COLOR_BGR2RGB), channels="RGB")
                anomaly_frame_placeholder.image(cv2.cvtColor(anomaly_result["heatmap"], cv2.COLOR_BGR2RGB), channels="RGB")
            else:
                anomaly_alert_placeholder.warning("Panel 2 stream ended or frame grab failed.")

        active_results = [ok for cap, ok in ((known_cap, known_ok), (anomaly_cap, anomaly_ok)) if cap]
        if active_results and not any(active_results):
            st.warning("Video source(s) ended.")
            break

        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now
        fps_placeholder.caption(f"FPS: {fps:.1f}")

    if known_cap:
        known_cap.release()
    if anomaly_cap:
        anomaly_cap.release()
else:
    st.info("Toggle 'Start live feed' in the sidebar to begin streaming.")

st.divider()
col_known_stats, col_anomaly_stats = st.columns(2)

with col_known_stats:
    st.subheader("Today's Known-Defect Stats")
    counts = defect_counts_today()
    st.metric("Total Known Defects Today", sum(counts.values()))

    if counts:
        df_counts = pd.DataFrame({"defect_type": list(counts.keys()), "count": list(counts.values())})
        fig = px.bar(df_counts, x="defect_type", y="count", title="Defects by Type (Today)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No known-defect detections logged yet today.")

    rows = recent_detections(limit=25)
    if rows:
        df_recent = pd.DataFrame(
            [
                {"time": r.timestamp, "camera": r.camera_id, "class": r.class_name, "confidence": f"{r.confidence:.0%}"}
                for r in rows
            ]
        )
        st.dataframe(df_recent, use_container_width=True, hide_index=True)

with col_anomaly_stats:
    st.subheader("Today's Anomaly Stats")
    stats = anomaly_stats_today()
    m1, m2, m3 = st.columns(3)
    m1.metric("Frames Scanned", stats["total"])
    m2.metric("Abnormal", stats["abnormal"])
    m3.metric("Avg Score", f"{stats['avg_score']:.2f}")

    anomaly_rows = recent_anomalies(limit=100)
    if anomaly_rows:
        df_anomaly = pd.DataFrame(
            [
                {"time": r.timestamp, "camera": r.camera_id, "score": r.anomaly_score, "status": "Abnormal" if r.is_anomalous else "Normal"}
                for r in reversed(anomaly_rows)
            ]
        )
        fig = px.line(df_anomaly, x="time", y="score", color="status", title="Anomaly Score Trend")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df_anomaly.iloc[::-1], use_container_width=True, hide_index=True)
    else:
        st.info("No anomaly detections logged yet today.")
