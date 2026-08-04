# Factory Defect Detection

Real-time industrial defect detection: a camera feed is run through a
YOLOv8 detector trained on steel surface defects (NEU-DET), detections
are logged to a database, defects trigger Telegram/email alerts with a
screenshot, and a Streamlit dashboard shows the live feed plus
production stats.

```
Camera → OpenCV → YOLOv8 → {bounding boxes, labels, confidence}
                              │
              ┌───────────────┼────────────────┐
              ▼               ▼                ▼
          SQLite DB      Alerts (TG/email)   Streamlit dashboard
```

## Status

Scaffold + working pipeline code. Not yet trained — you need to pull the
dataset and run training before `best.pt` exists.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 1. Get the dataset

NEU-DET (6 steel surface defect classes: crazing, inclusion, patches,
pitted_surface, rolled-in_scale, scratches) has no single canonical
download URL. Use one of:

```bash
# Option A: Roboflow Universe, already in YOLO format
export ROBOFLOW_API_KEY=your_key
python utils/download_dataset.py --source roboflow --project neu-det --version 1

# Option B: Kaggle raw NEU-DET (VOC XML — needs conversion, see the
# docstring in utils/download_dataset.py)
python utils/download_dataset.py --source kaggle --dataset <kaggle-slug>
```

This populates `dataset/{train,validation,test}/{images,labels}`, matching
the layout declared in `dataset/data.yaml`.

Optionally expand a thin class with offline augmentation:

```bash
python preprocessing/augment.py --images dataset/train/images --labels dataset/train/labels --copies 2
```

## 2. Train

```bash
python training/train.py --model yolov8n.pt --epochs 100 --imgsz 640
```

Weights land in `models/yolov8/defect_detector/weights/best.pt`.

## 3. Evaluate

```bash
python training/evaluate.py --weights models/yolov8/defect_detector/weights/best.pt --split test
```

## 4. Run real-time inference

```bash
# Webcam
python inference/webcam.py --weights models/yolov8/defect_detector/weights/best.pt

# Video file or RTSP stream
python inference/realtime.py --weights .../best.pt --source path/to/video.mp4
python inference/realtime.py --weights .../best.pt --source rtsp://user:pass@camera-ip/stream
```

Every detection is logged to `database/detections.db`. Defects also
trigger a rate-limited Telegram/email alert with a saved screenshot —
copy `alerts/.env.example` to `alerts/.env` and fill in credentials to
enable that (skipped silently if unset).

## 5. Dashboard

```bash
streamlit run dashboard/app.py
```

Shows the live annotated feed, today's defect counts/charts, and a table
of recent detections.

## Known limitations (fine for a portfolio MVP, worth calling out if asked)

- The Streamlit live-feed loop blocks Streamlit's normal rerun model
  while streaming; for a production dashboard you'd move capture to a
  background thread or use `streamlit-webrtc`.
- Alerting is polling/cooldown-based per class (30s), not a full
  event queue — fine for a demo, not for high-throughput lines.
- SQLite is used for simplicity; swap `database/db.py`'s
  `SQLALCHEMY_URL` for a Postgres DSN to scale beyond single-writer.

## Possible extensions (for a stronger M.Tech writeup)

- Compare CNN vs Vision Transformer backbones on the same dataset.
- Add PatchCore/FastFlow for one-class anomaly detection on classes
  with very few labeled defect examples.
- Grad-CAM visualization for explainability.
- Edge deployment benchmark on Jetson (latency vs accuracy tradeoff).

## Folder structure

```
factory-defect-detection/
├── dataset/            # train/validation/test images+labels, data.yaml
├── preprocessing/       # resize.py, normalize.py, augment.py
├── models/yolov8/        # training run outputs land here
├── training/            train.py, evaluate.py
├── inference/           realtime.py, webcam.py
├── dashboard/           app.py (Streamlit)
├── alerts/              telegram.py, email.py, .env.example
├── database/            db.py (SQLite/SQLAlchemy)
├── reports/              generated analytics output
└── utils/               download_dataset.py
```
