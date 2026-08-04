"""
Streamlit dashboard: live detection feed + production stats.

Panel 1 — known defects (YOLO, Module 1): "what type of defect is this?"
Panel 2 — unknown anomalies (PatchCore, Module 2): "does this look abnormal at all?"

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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from database.db import (
    anomaly_stats_today,
    defect_counts_today,
    init_db,
    log_anomaly,
    log_detection,
    recent_anomalies,
    recent_detections,
)
from inference.video_source import open_capture

st.set_page_config(page_title="Factory Defect Detection", layout="wide")
init_db()

st.title("Factory Defect Detection — Live Dashboard")

with st.sidebar:
    st.header("Configuration")
    enable_known = st.checkbox("Panel 1: known-defect detection (YOLO)", value=True)
    yolo_weights = st.text_input("YOLO weights", "models/yolov8/defect_detector/weights/best.pt")
    enable_anomaly = st.checkbox("Panel 2: unknown-anomaly detection (PatchCore)", value=True)
    patchcore_weights = st.text_input("PatchCore weights", "models/patchcore/bottle/weights/torch/model.pt")

    source = st.text_input("Video source (webcam index / file / RTSP URL / folder of images)", "0")
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
    known_frame_placeholder = st.empty()

with col_anomaly:
    st.subheader("Panel 2 — Unknown Anomalies (PatchCore)")
    anomaly_alert_placeholder = st.empty()
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

    cap = open_capture(source, image_folder_delay_seconds=image_delay)
    if not cap.isOpened():
        st.error(f"Could not open video source: {source}")
        st.stop()

    prev_time = time.time()
    while run_stream:
        ok, frame = cap.read()
        if not ok:
            st.warning("Stream ended or frame grab failed.")
            break

        if known_detector:
            known_result = known_detector.predict(frame)
            for det in known_result["detections"]:
                log_detection(det["class_name"], det["confidence"], det["bbox"], camera_id=camera_id)

            if known_result["detections"]:
                labels = ", ".join(f"{d['class_name']} ({d['confidence']:.0%})" for d in known_result["detections"])
                known_alert_placeholder.error(f"⚠️ Defect detected: {labels}")
            else:
                known_alert_placeholder.success("✅ No known defects in current frame")

            known_frame_placeholder.image(cv2.cvtColor(known_result["annotated"], cv2.COLOR_BGR2RGB), channels="RGB")

        if anomaly_detector:
            anomaly_result = anomaly_detector.predict(frame)
            log_anomaly(anomaly_result["score"], anomaly_result["is_anomalous"], camera_id=camera_id)

            if anomaly_result["is_anomalous"]:
                anomaly_alert_placeholder.error(f"🔴 Abnormal — score {anomaly_result['score']:.2f}")
            else:
                anomaly_alert_placeholder.success(f"🟢 Normal — score {anomaly_result['score']:.2f}")

            anomaly_frame_placeholder.image(cv2.cvtColor(anomaly_result["heatmap"], cv2.COLOR_BGR2RGB), channels="RGB")

        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now
        fps_placeholder.caption(f"FPS: {fps:.1f}")

    cap.release()
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
