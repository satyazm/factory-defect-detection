# Real-Time Industrial Defect Detection and Anomaly Detection System

Two complementary detection paradigms running on the same camera feed:

- **Module 1 — known-defect detection (supervised).** A YOLOv8 detector
  trained on labeled steel surface defects (NEU-DET). Answers *"what type
  of defect is this?"* — scratch, crazing, inclusion, etc., with a
  bounding box and confidence.
- **Module 2 — unknown-anomaly detection (unsupervised).** A PatchCore
  model (via [anomalib](https://github.com/open-edge-platform/anomalib))
  trained only on normal product images from MVTec AD. Answers *"does
  this look abnormal at all?"* — including defect types the model was
  never explicitly shown, with an anomaly score and heatmap.

They are two separately trained, separately deployed models — not one
model trained on both datasets. That's a standard pattern in industrial
inspection: a supervised classifier for defects you already know about,
plus an unsupervised anomaly detector as a catch-all for the ones you
don't.

```
                 Camera
                    │
                    ▼
              OpenCV frames
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
     YOLOv8                 PatchCore
 (known defects)       (unknown anomalies)
        │                       │
        ▼                       ▼
  Bounding boxes          Heatmap + score
        │                       │
        └───────────┬───────────┘
                    ▼
              SQLite (detections + anomalies)
                    │
                    ▼
            Streamlit dashboard
      ┌─────────────┴─────────────┐
      ▼                           ▼
 Panel 1: Known Defects   Panel 2: Unknown Anomalies
```

## Status

Scaffold + working pipeline code for both modules. Nothing is trained
yet — you need to pull each dataset and run its training script before
either `best.pt` (YOLO) or `model.pt` (PatchCore) exists.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`anomalib` pulls in `torch` and `lightning` as dependencies. If you have
a GPU, install the CUDA build of PyTorch first per the
[official instructions](https://pytorch.org/get-started/locally/) before
running `pip install -r requirements.txt`, or the generic CPU wheel will
be installed instead.

---

## Module 1: Known Defect Detection (YOLOv8 + NEU-DET)

### 1. Get the dataset

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

### 2. Train

```bash
python training/train.py --model yolov8n.pt --epochs 100 --imgsz 640
```

Weights land in `models/yolov8/defect_detector/weights/best.pt`.

### 3. Evaluate

```bash
python training/evaluate.py --weights models/yolov8/defect_detector/weights/best.pt --split test
```

### 4. Run standalone

```bash
# Single image
python inference/detect_known.py --weights .../best.pt --image path/to/image.jpg

# Webcam
python inference/webcam.py --weights .../best.pt

# Video file or RTSP stream
python inference/realtime.py --weights .../best.pt --source path/to/video.mp4
python inference/realtime.py --weights .../best.pt --source rtsp://user:pass@camera-ip/stream
```

Every detection is logged to `database/detections.db`.

---

## Module 2: Unknown Anomaly Detection (PatchCore + MVTec AD)

### 1. Get the dataset

Unlike NEU-DET, MVTec AD doesn't need a manual download step — anomalib
fetches the category archive automatically from the official mirror the
first time you train, into `dataset/mvtec_ad/<category>/`.

Pick one of the 15 MVTec AD categories (`bottle`, `cable`, `capsule`,
`carpet`, `grid`, `hazelnut`, `leather`, `metal_nut`, `pill`, `screw`,
`tile`, `toothbrush`, `transistor`, `wood`, `zipper`). `bottle` is the
default and matches a conveyor-belt bottle-inspection story well.

### 2. Train

```bash
python training/train_patchcore.py --category bottle
```

PatchCore trains only on "good" images — it learns what normal looks
like, then flags deviations at inference time. There's no iterative loss
to converge (it builds a coreset feature memory bank in one pass), so
this finishes fast even on CPU, unlike YOLO training. The exported model
lands at `models/patchcore/bottle/weights/torch/model.pt`.

### 3. Evaluate

```bash
python training/evaluate_patchcore.py --category bottle
```

Reports image-level AUROC and F1 on the full MVTec AD test split (good +
every defect type for that category).

### 4. Run standalone

```bash
python inference/detect_anomaly.py --weights models/patchcore/bottle/weights/torch/model.pt --image path/to/image.jpg
```

Prints the anomaly score + normal/abnormal status and saves a heatmap
overlay.

---

## Combined pipeline

```bash
python inference/combined_pipeline.py \
    --yolo-weights models/yolov8/defect_detector/weights/best.pt \
    --patchcore-weights models/patchcore/bottle/weights/torch/model.pt \
    --source 0
```

Runs both models on every frame from one source, shows known-defect
boxes on the left and the anomaly heatmap on the right, and logs to both
the `detections` and `anomalies` tables.

## Dashboard

```bash
streamlit run dashboard/app.py
```

Runs its own live inference loop (independent of the CLI scripts above)
with two panels, each independently toggleable from the sidebar:

- **Panel 1 — Known Defects:** annotated feed, red/green alert banner,
  today's defect counts by type, recent detections table.
- **Panel 2 — Unknown Anomalies:** heatmap feed, 🔴/🟢 status banner with
  live score, today's scan/abnormal counts + average score, anomaly
  score trend chart.

## Known limitations (fine for a portfolio MVP, worth calling out if asked)

- The Streamlit live-feed loop blocks Streamlit's normal rerun model
  while streaming; for a production dashboard you'd move capture to a
  background thread or use `streamlit-webrtc`.
- Alerting is in-dashboard only (banner per frame) — no notification
  fires if nobody has the dashboard open. Fine for a demo/portfolio
  setting; a real line would still want push alerts (Telegram/email/SMS).
- PatchCore's anomaly threshold (`pred_label`) is whatever anomalib
  calibrated during training — there's no manual threshold slider yet.
- SQLite is used for simplicity; swap `database/db.py`'s
  `SQLALCHEMY_URL` for a Postgres DSN to scale beyond single-writer.

## Possible extensions (for a stronger M.Tech writeup)

- Compare PatchCore against EfficientAD or FastFlow on the same MVTec
  category (accuracy vs inference latency tradeoff).
- Compare CNN vs Vision Transformer backbones for Module 1.
- Grad-CAM visualization for the YOLO detector's explainability.
- Active learning: route low-confidence YOLO detections and
  borderline PatchCore scores to a human-labeling queue.
- Edge deployment benchmark on Jetson (latency vs accuracy tradeoff);
  anomalib's OpenVINO export path is a natural fit here.

## Folder structure

```
factory-defect-detection/
├── dataset/
│   ├── train/ validation/ test/   # NEU-DET images+labels, data.yaml
│   └── mvtec_ad/                  # MVTec AD, auto-downloaded per category
├── preprocessing/                 # resize.py, normalize.py, augment.py
├── models/
│   ├── yolov8/                    # YOLO training run outputs
│   └── patchcore/                 # PatchCore checkpoints + exported models
├── training/
│   ├── train.py, evaluate.py              # Module 1 (YOLO)
│   └── train_patchcore.py, evaluate_patchcore.py  # Module 2 (PatchCore)
├── inference/
│   ├── detect_known.py            # KnownDefectDetector (YOLO wrapper)
│   ├── detect_anomaly.py          # AnomalyDetector (PatchCore wrapper)
│   ├── realtime.py, webcam.py     # Module 1 only, live
│   └── combined_pipeline.py       # both modules, live
├── dashboard/                     # app.py (Streamlit, two panels)
├── database/                      # db.py (SQLite/SQLAlchemy)
├── reports/                       # generated analytics output
└── utils/                         # download_dataset.py (NEU-DET)
```
