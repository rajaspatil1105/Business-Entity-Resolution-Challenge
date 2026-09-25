"""Step 4: Stage A LightGBM (local pair score).

Split by S1 id hash: HOLDOUT_PCT % of S1 groups are locked (scored once, at the end); the rest
use N_FOLDS-fold GroupKFold by S1. Dev rows get strictly out-of-fold scores; holdout rows get the
average of the fold models (same as test). No sklearn here (blocked DLL on the laptop).
"""
import lightgbm as lgb
import numpy as np
import polars as pl

from src import config as C
from src import data as D
from src import features as F
from src import normalize as N

KEEP = ["s1", "src", "id", "cty", "label"]


def fold_of(s1):
    """-1 = locked holdout, else fold 0..N_FOLDS-1 (deterministic, independent of query_mask)."""
    h = (np.asarray(s1).astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15)) >> np.uint64(32)
    b = (h % np.uint64(1000)).astype(np.int64)
    return np.where(b < C.HOLDOUT_PCT * 10, -1, b % C.N_FOLDS).astype(np.int8)


def params():
    return dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=100,
                feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                seed=C.LGB_SEED, num_threads=C.N_THREADS, verbose=-1)


def tag(frac, qfrac):
    return (f"f{frac:g}_q{qfrac:g}_cap{C.BLOCK_CAP}_n{N.VERSION}_df{C.MAX_DF:g}"
            f"_ng{C.NAME_NGRAM[0]}{C.NAME_NGRAM[1]}_rev{C.REV_K}_sb{int(C.STATE_BUCKETS)}_v{F.VERSION}")


def matrix(df, cols):
    return df.select([pl.col(c).cast(pl.Float32) for c in cols]).to_numpy()


def stage_a(frac=1.0, qfrac=1.0):
    """Train fold models on train features; return s1/src/id/cty/label/p/fold. Cached."""
    t = tag(frac, qfrac)
    d = C.WORK_DIR / "model"
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"stageA_{t}.parquet"
    if f.exists():
        return pl.read_parquet(f)
    df = F.build("train", frac, qfrac)
    cols = F.feature_cols(df)
    X, y = matrix(df, cols), df["label"].to_numpy()
    fold = fold_of(df["s1"].to_numpy())
    dev = fold >= 0
    p = np.zeros(len(y), np.float32)
    for k in range(C.N_FOLDS):
        tr, va = dev & (fold != k), fold == k
        with D.step(f"stageA fold {k} train={int(tr.sum()):,} valid={int(va.sum()):,}"):
            dtr = lgb.Dataset(X[tr], y[tr], feature_name=cols)
            dva = lgb.Dataset(X[va], y[va], reference=dtr)
            m = lgb.train(params(), dtr, C.LGB_ROUNDS, valid_sets=[dva],
                          callbacks=[lgb.early_stopping(C.LGB_EARLY_STOP, verbose=False)])
            p[va] = m.predict(X[va], num_iteration=m.best_iteration)
            if (~dev).any():
                p[~dev] += m.predict(X[~dev], num_iteration=m.best_iteration) / C.N_FOLDS
            m.save_model(str(d / f"stageA_{t}_f{k}.txt"), num_iteration=m.best_iteration)
        print(f"  fold {k}: best_iter {m.best_iteration}")
    out = df.select(KEEP).with_columns(pl.Series("p", p), pl.Series("fold", fold))
    out.write_parquet(f)
    return out


def importance(frac=1.0, qfrac=1.0, top=25):
    """Gain importance summed over fold models, in %."""
    d, t, g, m = C.WORK_DIR / "model", tag(frac, qfrac), None, None
    for k in range(C.N_FOLDS):
        m = lgb.Booster(model_file=str(d / f"stageA_{t}_f{k}.txt"))
        v = m.feature_importance("gain")
        g = v if g is None else g + v
    return (pl.DataFrame({"feature": m.feature_name(), "gain_pct": np.round(100 * g / g.sum(), 2)})
            .sort("gain_pct", descending=True).head(top))
