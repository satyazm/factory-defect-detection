"""
Evaluate a trained PatchCore checkpoint on the MVTec AD test split (good
+ every defect type) for one category. Reports image-level AUROC and F1.

Usage:
    python training/evaluate_patchcore.py --category bottle
"""
import argparse
from pathlib import Path

from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Patchcore

DATASET_ROOT = Path(__file__).resolve().parent.parent / "dataset" / "mvtec_ad"
RESULTS_ROOT = Path(__file__).resolve().parent.parent / "models" / "patchcore"


def find_latest_checkpoint(category: str) -> Path:
    candidates = sorted((RESULTS_ROOT / category).rglob("weights/lightning/model.ckpt"))
    if not candidates:
        raise SystemExit(
            f"No checkpoint found under {RESULTS_ROOT / category} — run train_patchcore.py first."
        )
    return candidates[-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", default="bottle")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    ckpt_path = find_latest_checkpoint(args.category)
    datamodule = MVTecAD(root=str(DATASET_ROOT), category=args.category, eval_batch_size=args.batch_size)
    model = Patchcore()
    engine = Engine()

    results = engine.test(model=model, datamodule=datamodule, ckpt_path=str(ckpt_path))
    print(f"\n--- Summary ({args.category}, checkpoint: {ckpt_path}) ---")
    print(results)


if __name__ == "__main__":
    main()
