"""
Train a PatchCore anomaly-detection model on one MVTec AD category.

Unlike the YOLO detector (Module 1, trained on labeled defects), PatchCore
trains only on "good" (normal) product images and learns what normal
looks like — at inference time, anything that deviates enough from that
learned distribution is flagged as an anomaly. This is what lets it catch
defect types it was never explicitly shown.

The MVTec AD category archive is auto-downloaded by anomalib into
dataset/mvtec_ad/<category>/ the first time it's missing.

Usage:
    python training/train_patchcore.py --category bottle
"""
import argparse
from pathlib import Path

from anomalib.data import MVTecAD
from anomalib.deploy import ExportType
from anomalib.engine import Engine
from anomalib.models import Patchcore

DATASET_ROOT = Path(__file__).resolve().parent.parent / "dataset" / "mvtec_ad"
EXPORT_ROOT = Path(__file__).resolve().parent.parent / "models" / "patchcore"

MVTEC_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
    "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--category", default="bottle", choices=MVTEC_CATEGORIES)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--coreset-sampling-ratio", type=float, default=0.1)
    args = parser.parse_args()

    datamodule = MVTecAD(
        root=str(DATASET_ROOT),
        category=args.category,
        train_batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
    )

    model = Patchcore(
        backbone="wide_resnet50_2",
        layers=["layer2", "layer3"],
        coreset_sampling_ratio=args.coreset_sampling_ratio,
    )

    run_dir = EXPORT_ROOT / args.category
    # PatchCore has no iterative loss to converge — one pass extracts features
    # and builds the coreset memory bank, so max_epochs=1 is correct here,
    # not a placeholder.
    engine = Engine(max_epochs=1, default_root_dir=str(run_dir))
    engine.fit(datamodule=datamodule, model=model)

    engine.export(model=model, export_type=ExportType.TORCH, export_root=str(run_dir))
    print(f"Exported PatchCore model for '{args.category}' to {run_dir}/weights/torch/model.pt")


if __name__ == "__main__":
    main()
