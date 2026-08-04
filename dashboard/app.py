"""
Streamlit dashboard: live detection feed + production stats.

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
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from database.db import defect_counts_today, init_db, log_detection, recent_detections

st.set_page_config(page_title="Factory Defect Detection", layout="wide")
init_db()

st.title("Factory Defect Detection — Live Dashboard")

with st.sidebar:
    st.header("Configuration")
    weights_path = st.text_input("Model weights", "models/yolov8/defect_detector/weights/best.pt")
    source = st.text_input("Video source (webcam index / file / RTSP URL)", "0")
    conf_threshold = st.slider("Confidence threshold", 0.1, 0.95, 0.5, 0.05)
    camera_id = st.text_input("Camera ID", "default")
    run_stream = st.toggle("Start live feed")

col_video, col_stats = st.columns([2, 1])

with col_stats:
    alert_placeholder = st.empty()

    st.subheader("Today's Production Stats")
    counts = defect_counts_today()
    total_defects = sum(counts.values())
    st.metric("Total Defects Today", total_defects)

    if counts:
        df_counts = pd.DataFrame({"defect_type": list(counts.keys()), "count": list(counts.values())})
        fig = px.bar(df_counts, x="defect_type", y="count", title="Defects by Type (Today)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No detections logged yet today.")

    st.subheader("Recent Detections")
    rows = recent_detections(limit=25)
    if rows:
        df_recent = pd.DataFrame(
            [
                {
                    "time": r.timestamp,
                    "camera": r.camera_id,
                    "class": r.class_name,
                    "confidence": f"{r.confidence:.0%}",
                }
                for r in rows
            ]
        )
        st.dataframe(df_recent, use_container_width=True, hide_index=True)
    else:
        st.info("No detections yet.")

with col_video:
    st.subheader("Live Feed")
    frame_placeholder = st.empty()
    fps_placeholder = st.empty()

    if run_stream:
        try:
            model = YOLO(weights_path)
        except Exception as exc:
            st.error(f"Could not load model weights at '{weights_path}': {exc}")
            st.stop()

        cap_source = int(source) if source.isdigit() else source
        cap = cv2.VideoCapture(cap_source)
        if not cap.isOpened():
            st.error(f"Could not open video source: {source}")
            st.stop()

        prev_time = time.time()
        while run_stream:
            ok, frame = cap.read()
            if not ok:
                st.warning("Stream ended or frame grab failed.")
                break

            results = model.predict(frame, conf=conf_threshold, verbose=False)[0]
            annotated = results.plot()

            frame_defects = []
            for box in results.boxes:
                cls_name = model.names[int(box.cls.item())]
                confidence = float(box.conf.item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                log_detection(cls_name, confidence, (x1, y1, x2, y2), camera_id=camera_id)
                frame_defects.append(f"{cls_name} ({confidence:.0%})")

            if frame_defects:
                alert_placeholder.error(f"⚠️ Defect detected: {', '.join(frame_defects)}")
            else:
                alert_placeholder.success("✅ No defects in current frame")

            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            frame_placeholder.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB")
            fps_placeholder.caption(f"FPS: {fps:.1f}")

        cap.release()
    else:
        st.info("Toggle 'Start live feed' in the sidebar to begin streaming.")
