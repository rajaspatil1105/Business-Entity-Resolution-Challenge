"""All paths and parameters in one place. Paths can be overridden with env vars (Kaggle)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("BER_DATA_DIR", ROOT.parent / "dataset"))
WORK_DIR = Path(os.environ.get("BER_WORK_DIR", ROOT / "artifacts"))
OUT_DIR = Path(os.environ.get("BER_OUT_DIR", ROOT / "output"))

SEED = 42
SOURCES = (1, 2, 3)
GT_PATH = DATA_DIR / "train" / "train_ground_truth.tsv"


def source_path(split, s):
    return DATA_DIR / split / f"{split}_source{s}.tsv"
