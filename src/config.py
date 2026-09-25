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

NAME_NGRAM = (3, 3)   # char n-gram range of the name blocker
REV_K = 5             # reverse name blocker: top S1 kept per S2/S3 record (0 = off)
FEAT_CHUNK = 1_000_000   # candidate pairs per feature chunk (memory bound)
N_FOLDS = 5              # GroupKFold by S1
HOLDOUT_PCT = 15         # % of S1 groups locked for the final score
LGB_SEED = 42
LGB_ROUNDS = 2000
LGB_EARLY_STOP = 50
STATE_BUCKETS = True     # search same-state + no-state records only; no-state queries search the whole country
