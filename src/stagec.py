"""Stage C: re-rank on Stage B scores + 3-seed ensemble + blend + per-country decision rule.

  python -m src.stagec oracle    # where points are lost: decision vs ranking vs blocking
  python -m src.stagec train     # Stage C models, blend weight w, per-country strictness v
  python -m src.stagec predict   # test -> matching_results.tsv + candidate_pairs.tsv -> validator
Safe: w=0 means "Stage B only", so the tuner falls back to Stage B if Stage C does not help.
"""
import argparse
import json

import lightgbm as lgb
import numpy as np
import polars as pl

from src import config as C
from src import model as M
from src import normalize as N
from src import stageb as SB

SB.PRUNE_P, SB.PRUNE_TOP = 0.01, 3          # must match the saved Stage B features/models
VERSION = 1
SEEDS = (11, 22, 33)
WGRID = (0.0, 0.5, 0.7, 1.0)
VGRID = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0)
OUT = SB.OUT
BEST = OUT / f"stagec_best_v{VERSION}.json"
K = SB.KEYS


# ---------------------------------------------------------------- data
def cty_map(split):
    f = OUT / f"rec_{split}_n{N.VERSION}_v{SB.VERSION}.parquet"
    R = pl.scan_parquet(f) if f.exists() else SB.records(split).lazy()
    return (R.filter(pl.col("src") == 1)
            .select(pl.col("id").cast(pl.Int64).alias("s1"), pl.col("cty").cast(pl.Utf8))
            .collect())


def pb_feats(X):
    q = pl.col("pb")
    X = X.sort(["s1", "pb"], descending=[False, True]).with_columns(
        pl.int_range(pl.len()).over("s1").alias("c_rk"),
        q.rank("ordinal", descending=True).over(["s1", "src"]).alias("c_rk_src"),
        q.max().over("s1").alias("c_max"),
        q.sum().over("s1").alias("c_sum"),
        q.sum().over(["s1", "src"]).alias("c_sum_src"),
        q.max().over(["s1", "src"]).alias("c_max_src"),
        (q > 0.5).sum().over("s1").alias("c_n50"),
        (q > 0.9).sum().over("s1").alias("c_n90"),
        q.shift(1).over("s1").alias("c_up"),
        q.shift(-1).over("s1").alias("c_dn"))
    qc = q.clip(1e-6, 1 - 1e-6)
    return X.with_columns(
        (q / pl.col("c_max").clip(1e-6)).alias("c_rel"),
        (pl.col("c_up") - q).alias("c_gap_up"),
        (q - pl.col("c_dn")).alias("c_gap_dn"),
        (pl.col("c_sum") - q).alias("c_rest"),
        (pl.col("c_max") - pl.col("c_max_src")).alias("c_src_gap"),
        (qc / (1 - qc)).log().alias("c_logit"),
        (pl.col("p") - q).alias("c_ab_diff")).drop("c_up", "c_dn")


def load_train():
    X = SB.build("train", SB.train_scores())
    O = pl.read_parquet(OUT / f"oof_v{SB.VERSION}.parquet").select(*K, "pb")
    n = X.height
    X = X.join(O, on=K, how="inner")
    if X.height != n:
        print(f"WARNING: oof join kept {X.height:,} of {n:,} rows")
    if "cty" in X.columns:
        X = X.drop("cty")
    X = X.join(cty_map("train"), on="s1", how="left")
    return pb_feats(X)


# ---------------------------------------------------------------- models
def params(seed):
    p = SB.params()
    p.update(seed=seed)
    return p


def fit(X, cols):
    A, y, fold = M.matrix(X, cols), X["label"].to_numpy(), X["fold"].to_numpy()
    dev, pc = fold >= 0, np.zeros(len(y), np.float32)
    for s in SEEDS:
        for k in range(C.N_FOLDS):
            tr, va = dev & (fold != k), fold == k
            dtr = lgb.Dataset(A[tr], y[tr], feature_name=cols)
            m = lgb.train(params(s), dtr, 5000, valid_sets=[lgb.Dataset(A[va], y[va], reference=dtr)],
                          callbacks=[lgb.early_stopping(100, verbose=False)])
            pc[va] += m.predict(A[va], num_iteration=m.best_iteration) / len(SEEDS)
            if (~dev).any():
                pc[~dev] += m.predict(A[~dev], num_iteration=m.best_iteration) / (len(SEEDS) * C.N_FOLDS)
            m.save_model(str(OUT / f"stageC_v{VERSION}_s{s}_f{k}.txt"), num_iteration=m.best_iteration)
            print(f"  seed {s} fold {k}: best_iter {m.best_iteration}", flush=True)
    return pc


def avg(models, X):
    cols = models[0].feature_name()
    lost = [c for c in cols if c not in X.columns]
    if lost:
        raise KeyError(f"missing features {lost}")
    out = np.zeros(X.height, np.float32)
    for a in range(0, X.height, 2_000_000):
        A = M.matrix(X.slice(a, 2_000_000), cols)
        out[a:a + len(A)] = np.mean([m.predict(A) for m in models], axis=0)
    return out


# ---------------------------------------------------------------- decision
def rule(O, c, vs, default):
    V = pl.DataFrame({"cty": list(vs), "g": [float(x) for x in vs.values()]},
                     schema={"cty": pl.Utf8, "g": pl.Float64})
    S = (SB.own(O, c).with_columns(pl.col("cty").cast(pl.Utf8))
         .join(V, on="cty", how="left").with_columns(pl.col("g").fill_null(default))
         .with_columns(pl.col(c).cast(pl.Float64).clip(0.0, 0.999999).pow(pl.col("g")).alias("q"))
         .sort(["s1", "q"], descending=[False, True]))
    S = S.with_columns(pl.int_range(1, pl.len() + 1).over("s1").alias("k"),
                       pl.col("q").cum_sum().over("s1").alias("A"),
                       pl.col("q").sum().over("s1").alias("T"),
                       (1 - pl.col("q")).log().sum().over("s1").exp().alias("E0"))
    S = S.with_columns((1.25 * pl.col("A") / (pl.col("k") + 0.25 * pl.col("T"))).alias("EF"))
    S = S.with_columns(pl.col("EF").max().over("s1").alias("EFm"))
    kb = S.filter(pl.col("EF") == pl.col("EFm")).group_by("s1").agg(pl.col("k").min().alias("kb"))
    return S.join(kb, on="s1").filter((pl.col("k") <= pl.col("kb")) & (pl.col("EFm") > pl.col("E0")))


# ---------------------------------------------------------------- commands
def cmd_oracle():
    O = pl.read_parquet(OUT / f"oof_v{SB.VERSION}.parquet")
    U, gt = SB.truth()
    cfg = json.loads((OUT / f"best_v{SB.VERSION}.json").read_text())
    cur = SB.f05(SB.rule(O, "pb", cfg["kind"], cfg["v"]).select(K), U, gt)
    cur = cur.filter(pl.col("fold") >= 0)["f"].mean()
    t = gt.group_by("s1").agg(pl.len().alias("t"))
    S = (SB.own(O, "pb").sort(["s1", "pb"], descending=[False, True])
         .with_columns(pl.int_range(1, pl.len() + 1).over("s1").alias("k"),
                       pl.col("label").cast(pl.Int32).cum_sum().over("s1").alias("a"))
         .join(t, on="s1", how="left").with_columns(pl.col("t").fill_null(0))
         .with_columns((1.25 * pl.col("a") / (pl.col("k") + 0.25 * pl.col("t"))).alias("f"))
         .group_by("s1").agg(pl.col("f").max().alias("fk")))
    hit = O.filter(pl.col("label") == 1).group_by("s1").agg(pl.len().alias("a"))
    E = (U.join(t, on="s1", how="left").join(S, on="s1", how="left").join(hit, on="s1", how="left")
         .with_columns(pl.col("t").fill_null(0), pl.col("fk").fill_null(0.0), pl.col("a").fill_null(0)))
    E = E.with_columns(
        pl.max_horizontal("fk", (pl.col("t") == 0).cast(pl.Float64)).alias("f_dec"),
        pl.when(pl.col("t") == 0).then(1.0)
          .otherwise(1.25 * pl.col("a") / (pl.col("a") + 0.25 * pl.col("t"))).alias("f_all"))
    E = E.filter(pl.col("fold") >= 0)
    dec, al = E["f_dec"].mean(), E["f_all"].mean()
    print(f"ORACLE current Stage B rule        : {cur:.4f}")
    print(f"ORACLE perfect cut per S1 (pb order): {dec:.4f}  -> decision loss {dec - cur:.4f}")
    print(f"ORACLE perfect pick in candidates   : {al:.4f}  -> ranking loss  {al - dec:.4f}")
    print(f"ORACLE blocking loss                : {1 - al:.4f}")
    print("VERDICT:", "DECISION is the main loss" if dec - cur > al - dec
          else "RANKING is the main loss -> record-level model next")


def cmd_train():
    X = load_train()
    cols = SB.feat_cols(X)
    print(f"stage C rows {X.height:,} features {len(cols)}")
    pc = fit(X, cols)
    O = X.select(*K, "cty", "label", "fold", "p", "pb").with_columns(pl.Series("pc", pc))
    O.write_parquet(OUT / f"stagec_oof_v{VERSION}.parquet")
    U, gt = SB.truth()
    cm = cty_map("train")
    rows = []
    for w in WGRID:
        Ow = O.with_columns((w * pl.col("pc") + (1 - w) * pl.col("pb")).alias("sc"))
        for v in VGRID:
            E = SB.f05(rule(Ow, "sc", {}, v).select(K), U, gt).drop("cty").join(cm, on="s1", how="left")
            rows += (E.group_by("cty", (pl.col("fold") >= 0).alias("dev"))
                     .agg(pl.col("f").sum().alias("fs"), pl.len().alias("n"))
                     .with_columns(pl.lit(w).alias("w"), pl.lit(v).alias("v")).to_dicts())
    T = pl.DataFrame(rows).with_columns(pl.col("cty").fill_null(""))
    D, H = T.filter(pl.col("dev")), T.filter(~pl.col("dev"))
    g = D.group_by("w", "v").agg((pl.col("fs").sum() / pl.col("n").sum()).alias("dev")).sort("dev", descending=True)
    base = g.filter(pl.col("w") == 0).row(0, named=True)
    res = []
    for w in WGRID:
        pick = (D.filter(pl.col("w") == w).with_columns((pl.col("fs") / pl.col("n")).alias("m"))
                .sort("m", descending=True).unique("cty", keep="first", maintain_order=True))
        vs = dict(zip(pick["cty"].to_list(), pick["v"].to_list()))
        dev = pick["fs"].sum() / pick["n"].sum()
        h = H.filter(pl.col("w") == w).join(pick.select("cty", "v"), on=["cty", "v"])
        hold = h["fs"].sum() / max(h["n"].sum(), 1)
        res.append(dict(w=w, vs=vs, default=base["v"], dev=dev, holdout=hold))
        print(f"w={w}: dev {dev:.5f} holdout {hold:.5f} v per country {vs}")
    b = max(res, key=lambda r: r["dev"])
    BEST.write_text(json.dumps(b))
    print(f"\nSTAGE B (one v={base['v']}) dev {base['dev']:.5f}")
    print(f"BEST w={b['w']} dev {b['dev']:.5f} holdout {b['holdout']:.5f} | gain {b['dev'] - base['dev']:+.5f}")


def cmd_predict():
    b = json.loads(BEST.read_text())
    print("config:", b)
    X = SB.build("test", SB.test_scores())
    mb = [lgb.Booster(model_file=str(OUT / f"stageB_v{SB.VERSION}_f{k}.txt")) for k in range(C.N_FOLDS)]
    X = X.with_columns(pl.Series("pb", avg(mb, X)))
    if "cty" in X.columns:
        X = X.drop("cty")
    X = X.join(cty_map("test"), on="s1", how="left")
    X = pb_feats(X)
    pb = X["pb"].to_numpy()
    if b["w"] > 0:
        mc = [lgb.Booster(model_file=str(OUT / f"stageC_v{VERSION}_s{s}_f{k}.txt"))
              for s in SEEDS for k in range(C.N_FOLDS)]
        sc = b["w"] * avg(mc, X) + (1 - b["w"]) * pb
    else:
        sc = pb
    O = X.select(*K, "cty", "p").with_columns(pl.Series("pb", sc.astype(np.float32)))
    O.write_parquet(OUT / f"stagec_test_v{VERSION}.parquet")
    SB.write(O, rule(O, "pb", b["vs"], b["default"]), None)


def main():
    a = argparse.ArgumentParser()
    a.add_argument("cmd", choices=["oracle", "train", "predict"])
    x = a.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"oracle": cmd_oracle, "train": cmd_train, "predict": cmd_predict}[x.cmd]()


if __name__ == "__main__":
    main()
