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

# ---- Step 2: blocking
BLOCK_K = {"name": 30, "addr": 30, "combo": 30}   # top-k per blocker, per source
BLOCK_CAP = 30        # candidates kept per S1 per source after the union
MAX_DF = 0.2         # drop tokens/3-grams present in more than this share of a country pool
MIN_SIM = 0.05        # ignore retrieval scores below this
Q_CHUNK = 100_000     # S1 queries per matrix product
N_THREADS = os.cpu_count() or 4
