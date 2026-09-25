"""Step 8: score test candidates with the Stage A fold models, apply the decision rule,
write output/matching_results.tsv + output/candidate_pairs.tsv, then run the validator.

Streams test in S1 chunks (features -> predict -> keep s1/src/id/p only) so ~100M
candidate rows never sit in RAM as one feature matrix.

  full:  python -m src.predict --qfrac 0.05 --ts 0.7 --ta 0.7 --tr 0.0 --mode hard
  smoke: python -m src.predict --frac 0.01 --train-frac 0.01 --qfrac 0.25
"""
import argparse
import subprocess
import sys
import time

import lightgbm as lgb
import numpy as np
import polars as pl

from src import blocking as B
from src import config as C
from src import data as D
from src import decide as K
from src import features as F
from src import model as M
from src import normalize as N


def _ids(s, frac):
    df = pl.read_csv(C.source_path("test", s), **D.READ_OPTS).select(
        pl.col("entity_id").alias("eid"),
        pl.col("entity_id").str.slice(3).cast(pl.Int64).alias("id"),
        pl.col("country").fill_null("").alias("country"))
    return df.filter(D._keep("id", frac)) if frac < 1 else df


def score(frac, train_frac, qfrac, cap, nmodels, s1_chunk):
    t = M.tag(train_frac, qfrac)
    f = C.WORK_DIR / "pred" / f"test_f{frac:g}_{t}_cap{cap}_m{nmodels}.parquet"
    f.parent.mkdir(parents=True, exist_ok=True)
    if f.exists():
        print(f"cached scores: {f.name}")
        return pl.read_parquet(f)

    paths = [C.WORK_DIR / "model" / f"stageA_{t}_f{k}.txt" for k in range(nmodels)]
    miss = [p.name for p in paths if not p.exists()]
    if miss:
        raise FileNotFoundError(f"missing Stage A models {miss} -> run: "
                                f"python -m src.run stagea --frac {train_frac:g} --qfrac {qfrac:g}")
    models = [lgb.Booster(model_file=str(p)) for p in paths]
    cols = models[0].feature_name()

    cand_f = C.WORK_DIR / "cand" / f"{B.tag('test', frac, 1.0)}.parquet"
    if not cand_f.exists():
        cand_f.parent.mkdir(parents=True, exist_ok=True)
        B.block_split("test", frac, 1.0).write_parquet(cand_f)

    norm = N.normalize_split("test", frac)
    s1 = F._prep(norm[1]).rename({**{c: f"{c}_1" for c in F.COLS}, "id": "s1"})
    oth = (pl.concat([F._prep(norm[s]).drop("cty").with_columns(pl.lit(s, pl.Int8).alias("src"))
                      for s in (2, 3)]).rename({c: f"{c}_2" for c in F.COLS}))
    idf = {c: F._idf(norm, c) for c in ("name_core", "addr_norm")}
    q = np.sort(s1["s1"].unique().to_numpy())
    lf = pl.scan_parquet(cand_f).filter(pl.col("pos") < cap)
    nch = (len(q) + s1_chunk - 1) // s1_chunk
    parts, t0 = [], time.time()
    with D.step(f"predict test f{frac:g} S1={len(q):,} chunks={nch} models={nmodels}"):
        for i, a in enumerate(range(0, len(q), s1_chunk)):
            lo, hi = int(q[a]), int(q[min(a + s1_chunk, len(q)) - 1])
            cand = lf.filter(pl.col("s1").is_between(lo, hi)).collect()
            if cand.height:
                P = cand.join(s1, on="s1").join(oth, on=["src", "id"]).sort(["s1", "src", "pos"])
                X = F._features(P, idf)
                if X.height != P.height:
                    raise RuntimeError(f"_features changed row count {P.height} -> {X.height}")
                for k in ("s1", "id"):
                    if k in X.columns and not (X[k] == P[k]).all():
                        raise RuntimeError("_features changed row order")
                lost = [c for c in cols if c not in X.columns]
                if lost:
                    raise KeyError(f"features missing for the model: {lost}")
                Xm = M.matrix(X, cols)
                p = np.mean([m.predict(Xm) for m in models], axis=0).astype(np.float32)
                parts.append(P.select("s1", "src", "id").with_columns(pl.Series("p", p)))
            if (i + 1) % 10 == 0 or i + 1 == nch:
                el = time.time() - t0
                print(f"  chunk {i + 1}/{nch} | {el / 60:.1f} min | eta {el / (i + 1) * (nch - i - 1) / 60:.1f} min",
                      flush=True)
    S = pl.concat(parts)
    S.write_parquet(f)
    return S


def write(S, frac, ts, ta, tr, mode):
    i1 = _ids(1, frac)
    oth = pl.concat([_ids(s, frac).select("eid", "id", pl.lit(s, pl.Int8).alias("src")) for s in (2, 3)])
    base = i1.select(pl.col("eid").alias("source1_entity_id"), pl.col("id").alias("s1"), "country")
    S = S.with_columns(pl.col("s1").cast(pl.Int64), pl.col("src").cast(pl.Int8), pl.col("id").cast(pl.Int64))
    sel = K.decide(S, mode, ts, ta, tr)

    def lists(df, col):
        g = (df.join(oth, on=["src", "id"]).sort(["s1", "p"], descending=[False, True])
             .group_by("s1", maintain_order=True)
             .agg(pl.col("eid").unique(maintain_order=True).str.join(",").alias(col)))
        return base.join(g, on="s1", how="left").with_columns(pl.col(col).fill_null(""))

    out = C.OUT_DIR if frac >= 1 else C.OUT_DIR / "smoke"
    out.mkdir(parents=True, exist_ok=True)
    mf, cf = out / "matching_results.tsv", out / "candidate_pairs.tsv"
    lists(S, "candidate_entity_ids").select("source1_entity_id", "candidate_entity_ids") \
        .write_csv(cf, separator="\t", quote_style="never")
    m = lists(sel, "matched_entity_ids")
    m.select("source1_entity_id", "matched_entity_ids").write_csv(mf, separator="\t", quote_style="never")
    print(f"wrote {mf} ({m.height:,} rows) and {cf}")

    n = pl.col("matched_entity_ids").str.split(",").list.eval(pl.element().filter(pl.element() != "")).list.len()
    print(m.with_columns(n.alias("n")).group_by("country").agg(
        pl.len().alias("s1s"), (pl.col("n") == 0).mean().round(4).alias("empty_share"),
        pl.col("n").mean().round(3).alias("mean_matches")).sort("country"))

    v = C.DATA_DIR.parent / "utils" / "validate_submission.py"
    if frac < 1:
        print("smoke test: validator skipped (the sample leaves out S1s by design)")
    elif v.exists():
        r = subprocess.run([sys.executable, str(v), "--matching", str(mf), "--candidate", str(cf),
                            "--test-dir", str(C.DATA_DIR / "test")])
        print("validator exit code", r.returncode, "(0 = PASS)")
    else:
        print(f"validator not found at {v} - run it by hand")


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--frac", type=float, default=1.0, help="test record sample (1.0 = real submission)")
    a.add_argument("--train-frac", type=float, default=1.0, help="frac the Stage A models were trained with")
    a.add_argument("--qfrac", type=float, default=0.05, help="qfrac the Stage A models were trained with")
    a.add_argument("--cap", type=int, default=C.BLOCK_CAP)
    a.add_argument("--nmodels", type=int, default=C.N_FOLDS)
    a.add_argument("--s1-chunk", type=int, default=15_000)
    a.add_argument("--ts", type=float, default=0.7)
    a.add_argument("--ta", type=float, default=0.7)
    a.add_argument("--tr", type=float, default=0.0)
    a.add_argument("--mode", default="hard", choices=["none", "hard"])
    x = a.parse_args()
    S = score(x.frac, x.train_frac, x.qfrac, x.cap, x.nmodels, x.s1_chunk)
    write(S, x.frac, x.ts, x.ta, x.tr, x.mode)


if __name__ == "__main__":
    main()
