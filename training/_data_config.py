"""
Shared helper: materialize dataset/data.yaml as a temp yaml with fully
absolute image paths.

Ultralytics resolves a relative `path:` in a dataset yaml against the
current working directory at *train/val time*, not against the yaml
file's own location — a long-standing quirk
(https://github.com/ultralytics/ultralytics/issues/9503) that breaks
dataset/data.yaml's checked-in relative paths unless the script happens
to be invoked with cwd == repo root. Absolute paths sidestep the
ambiguity entirely regardless of cwd, so train.py and evaluate.py both
go through this instead of pointing ultralytics at the static file.
"""
import tempfile
from pathlib import Path

import yaml

DATASET_DIR = Path(__file__).resolve().parent.parent / "dataset"


def build_absolute_data_yaml() -> str:
    with open(DATASET_DIR / "data.yaml") as f:
        data = yaml.safe_load(f)

    data.pop("path", None)
    data["train"] = str(DATASET_DIR / "train" / "images")
    data["val"] = str(DATASET_DIR / "validation" / "images")
    data["test"] = str(DATASET_DIR / "test" / "images")

    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    yaml.safe_dump(data, tmp)
    tmp.close()
    return tmp.name
