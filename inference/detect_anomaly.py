"""
Unknown-anomaly detection: thin wrapper around a trained PatchCore model,
exported via training/train_patchcore.py to a TorchInferencer-loadable
model.pt.

Where inference/detect_known.py answers "what type of defect is this?",
this answers "does this look abnormal at all?" — including defect types
the model was never explicitly trained on, since PatchCore only ever
learned what a normal product looks like.

Standalone usage (single image):
    python inference/detect_anomaly.py --weights models/patchcore/bottle/weights/torch/model.pt --image path/to/image.jpg
"""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

# anomalib's TorchInferencer loads .pt checkpoints via torch.load/pickle,
# which can execute arbitrary code, so it refuses to load unless this is
# set. Safe here specifically because every model.pt this wrapper is
# pointed at comes from training/train_patchcore.py's own
# engine.export(..., export_type=ExportType.TORCH) in this repo, not a
# checkpoint downloaded from an untrusted source.
os.environ.setdefault("TRUST_REMOTE_CODE", "1")

from anomalib.deploy import TorchInferencer  # noqa: E402


def _to_numpy(value):
    """anomalib prediction fields may be torch tensors or already numpy — normalize either."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


class AnomalyDetector:
    def __init__(self, weights_path: str | Path):
        self.inferencer = TorchInferencer(path=str(weights_path))

    def predict(self, frame_bgr: np.ndarray) -> dict:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        # A single image still comes back as a batch of 1 (ImageBatch), so
        # pred_score/pred_label are shape (1,) rather than scalars.
        result = self.inferencer.predict(image=frame_rgb)

        score = float(_to_numpy(result.pred_score)[0])
        is_anomalous = bool(_to_numpy(result.pred_label)[0])
        heatmap = self._overlay_heatmap(frame_bgr, _to_numpy(result.anomaly_map))

        return {"score": score, "is_anomalous": is_anomalous, "heatmap": heatmap}

    @staticmethod
    def _overlay_heatmap(frame_bgr: np.ndarray, anomaly_map: np.ndarray) -> np.ndarray:
        amap = np.squeeze(anomaly_map.astype(np.float32))
        amap = cv2.resize(amap, (frame_bgr.shape[1], frame_bgr.shape[0]))
        amap_norm = cv2.normalize(amap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(amap_norm, cv2.COLORMAP_JET)
        return cv2.addWeighted(frame_bgr, 0.6, heatmap_color, 0.4, 0)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run PatchCore anomaly detection on a single image file.")
    parser.add_argument("--weights", required=True, help="path to exported model.pt")
    parser.add_argument("--image", required=True, help="path to an image file")
    parser.add_argument("--out", default="anomaly_overlay.jpg", help="where to save the heatmap overlay")
    args = parser.parse_args()

    frame = cv2.imread(args.image)
    if frame is None:
        raise SystemExit(f"Could not read image: {args.image}")

    detector = AnomalyDetector(args.weights)
    result = detector.predict(frame)
    status = "ABNORMAL" if result["is_anomalous"] else "NORMAL"
    print(f"Anomaly score: {result['score']:.4f} — {status}")
    cv2.imwrite(args.out, result["heatmap"])
    print(f"Heatmap overlay saved to {args.out}")


if __name__ == "__main__":
    main()
