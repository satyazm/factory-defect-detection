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

Both modules trained and verified end to end (Kaggle GPU for training,
local CPU for inference/dashboard) — see Results below.

## Results

**Module 1 — YOLOv8n on NEU-DET, 100 epochs, held-out test split (90 images):**

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| crazing | 0.667 | 0.154 | 0.439 | 0.190 |
| inclusion | 0.862 | 0.800 | 0.850 | 0.448 |
| patches | 0.843 | 0.868 | 0.941 | 0.590 |
| pitted_surface | 0.877 | 0.842 | 0.893 | 0.447 |
| rolled-in_scale | 0.541 | 0.517 | 0.574 | 0.231 |
| scratches | 0.548 | 0.778 | 0.716 | 0.359 |
| **all (mean)** | **0.723** | **0.660** | **0.736** | **0.378** |

`crazing` is the clear weak point — recall of 0.154 means it's missing
most crazing instances on test. Not a bug: crazing defects are diffuse,
low-contrast crack networks rather than a compact shape, and they're the
hardest class in every published NEU-DET benchmark, not just this run.
Worth naming directly in a writeup rather than only reporting the mean.

**Module 2 — PatchCore on MVTec AD `bottle`, single-pass coreset training:**

| Metric | Score |
|---|---|
| image_AUROC | 1.000 |
| image_F1Score | 0.992 |
| pixel_AUROC | 0.986 |
| pixel_F1Score | 0.725 |

`image_AUROC = 1.0` looks like overfitting at first glance but isn't —
`bottle` is one of MVTec AD's easiest categories and PatchCore's
original paper reports ~100% image-AUROC on it too, so this matches the
published benchmark rather than being specific to this run. The
pixel-level F1 being lower than the image-level scores is the normal
pattern: localizing *which pixels* are anomalous is harder than just
flagging *whether* the image is anomalous at all.

Reproduce with `training/evaluate.py --split test` and
`training/evaluate_patchcore.py --category bottle` (see below).

## Setup

**Requires Python 3.10+** — `anomalib>=2.6.0` needs it, but silently
degrades to an ancient, API-incompatible `anomalib 0.7.0` on older
Python instead of erroring, so a plain `python -m venv` on an old system
Python can look like it worked and then fail confusingly later. Check
`python3 --version` first; on macOS, `brew install python@3.11` if
needed.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`anomalib` pulls in `torch` and `lightning` as dependencies. If you have
a GPU, install the CUDA build of PyTorch first per the
[official instructions](https://pytorch.org/get-started/locally/) before
running `pip install -r requirements.txt`, or the generic CPU wheel will
be installed instead.

Loading a trained PatchCore checkpoint requires setting
`TRUST_REMOTE_CODE=1` (handled automatically in `inference/detect_anomaly.py`) —
anomalib guards `.pt` loading behind this because it uses `pickle` under
the hood, which can execute arbitrary code. Safe for checkpoints this
repo trained itself (Kaggle notebook or `training/train_patchcore.py`);
don't point `AnomalyDetector` at a `.pt` file from an untrusted source.

---

## Module 1: Known Defect Detection (YOLOv8 + NEU-DET)

### 1. Get the dataset

NEU-DET (6 steel surface defect classes: crazing, inclusion, patches,
pitted_surface, rolled-in_scale, scratches) has no single canonical
download URL. **Verified working path:** download the
["NEU Surface Defect Database"](https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database)
zip by hand from Kaggle (just needs you logged into Kaggle in the
browser, no API token) and convert its PASCAL VOC XML annotations:

```bash
python preprocessing/voc_to_yolo.py --source ~/Downloads/NEU-DET
```

This ships as `train`/`validation` only (no test split) with per-class
image folders and a flat `annotations/` folder of matching XML files.
The script converts VOC bounding boxes to YOLO txt and carves a test
split out of validation (1 in 4 images per class, so validation and test
stay disjoint), skipping any image/annotation that doesn't have a match
on the other side. See `utils/download_dataset.py`'s docstring for two
alternative sources (Roboflow with an API key, or the Kaggle CLI) if you
want to script the download step too.

Either way, the result populates `dataset/{train,validation,test}/{images,labels}`,
matching the layout declared in `dataset/data.yaml`.

Optionally expand a thin class with offline augmentation:

```bash
python preprocessing/augment.py --images dataset/train/images --labels dataset/train/labels --copies 2
```

### 2. Train

Locally (CPU will work but is slow):

```bash
python training/train.py --model yolov8n.pt --epochs 100 --imgsz 640
```

Weights land in `models/yolov8/defect_detector/weights/best.pt`.

**Or on Kaggle (free GPU):** upload
[kaggle/train_yolo_kaggle.ipynb](kaggle/train_yolo_kaggle.ipynb) as a new
Kaggle notebook, attach the same
["NEU Surface Defect Database"](https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database)
as an input dataset, set the accelerator to a GPU, and run all cells. It's
self-contained (no repo cloning, no credentials) — it re-does the VOC→YOLO
conversion and training inside the notebook. Download `best.pt` from the
notebook's Output pane afterward and drop it into
`models/yolov8/defect_detector/weights/best.pt` locally (it's gitignored,
so this is a manual copy).

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

# Folder of images — loops through them indefinitely (sorted by
# filename, one every --image-delay seconds), for demoing continuous
# detection without a real camera
python inference/realtime.py --weights .../best.pt --source dataset/test/images --image-delay 1.5
```

Every detection is logged to `database/detections.db`. The folder-of-images
source works the same way in `combined_pipeline.py` and the dashboard
(sidebar → Video source), and is implemented once in
`inference/video_source.py` so all three share it.

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

**Or on Kaggle (free GPU):** upload
[kaggle/train_patchcore_kaggle.ipynb](kaggle/train_patchcore_kaggle.ipynb)
as a new Kaggle notebook. It clones the repo with a `GITHUB_TOKEN` Kaggle
Secret (same pattern as the YOLO notebook) and runs the actual
`training/train_patchcore.py`/`evaluate_patchcore.py` scripts — no
duplicated logic to drift out of sync. Unlike NEU-DET, MVTec AD needs no
manual download or attached Input: anomalib fetches the category archive
directly from the official mirror at train time, given internet access.
Enable the `GITHUB_TOKEN` secret, set accelerator to GPU, internet on,
run all cells, then download `model.pt` from the Output pane and drop it
into `models/patchcore/bottle/weights/torch/model.pt` locally (gitignored).

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
- **Panel 2 — Unknown Anomalies:** original frame next to the heatmap,
  🔴/🟢 status banner with live score, today's scan/abnormal counts +
  average score, anomaly score trend chart.

The dashboard's default "Video source" is `dataset/test/images`
(NEU-DET) — great for demoing Panel 1, but every frame will read as
"Abnormal, score ~1.0" on Panel 2, since PatchCore only knows what a
normal *bottle* looks like and steel surface photos are wildly outside
that domain (not a bug — the model is correctly saying "this isn't a
bottle at all"). For a Panel 2 demo that actually varies between normal
and abnormal, point "Video source" at a folder of real bottle images
instead:

```bash
# Official per-category download (148 MB), no login required — verified
# working 2026-08-04; if it 404s, get the current link from
# https://www.mvtec.com/company/research/datasets/mvtec-ad/downloads
curl -L -o dataset/mvtec_ad/bottle.tar.xz "https://www.mydrive.ch/shares/150452/132a93367fb17cdf968dfb5c4013f6e7/download/420937370-1629958698/bottle.tar.xz"
tar -xf dataset/mvtec_ad/bottle.tar.xz -C dataset/mvtec_ad
chmod -R u+rwX dataset/mvtec_ad/bottle  # archive ships with restrictive permissions

# Flatten test/{good,broken_small,broken_large,contamination} into one
# folder so it works as a single --source (ImageFolderCapture reads
# files directly in a folder, not subfolders)
cd dataset/mvtec_ad/bottle && mkdir -p test_flat
for d in test/good test/broken_small test/broken_large test/contamination; do
  label=$(basename "$d")
  for f in "$d"/*; do cp "$f" "test_flat/${label}_$(basename "$f")"; done
done
```

Then set "Video source" to `dataset/mvtec_ad/bottle/test_flat` — 83
images (20 normal, 63 defective across 3 defect types), scores actually
vary (`good` images score ~0.3, `broken_large` scores ~1.0), and the
heatmaps localize onto the real visible damage.

There's no single source that demos both panels meaningfully at once —
Module 1 only knows steel defects, Module 2 only knows bottles. Pick
whichever source matches the panel you're showing off.

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

## Possible extensions (to make the portfolio story stronger)

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
├── preprocessing/                 # resize.py, normalize.py, augment.py, voc_to_yolo.py
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
├── kaggle/                        # train_yolo_kaggle.ipynb, train_patchcore_kaggle.ipynb (GPU training)
├── reports/                       # generated analytics output
└── utils/                         # download_dataset.py (NEU-DET)
```
