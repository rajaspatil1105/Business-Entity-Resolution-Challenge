"""Stage B: context model on top of Stage A scores + expected-F0.5 decision rule.

  python -m src.stageb check      # prune recall + orphan check
  python -m src.stageb train      # features -> 5-fold LightGBM -> OOF/holdout F0.5 -> best rule saved
  python -m src.stageb predict    # test features -> scores -> both TSVs -> validator
"""
import argparse
import gc
import json
import re
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import Levenshtein

from src import blocking as B
from src import config as C
from src import data as D
from src import features as F
from src import model as M
from src import normalize as N

VERSION = 1
TRAIN_FRAC, QFRAC = 1.0, 0.05
PRUNE_P, PRUNE_TOP = 0.01, 3
RIVAL_MAX = 20
CHUNK = 1_000_000
KEYS = ["s1", "src", "id"]
META = ["s1", "id", "label", "fold", "cty", "a_src", "a_id"]
OUT = C.WORK_DIR / "stageb"
LEGAL_MAP = {"private": "pvt", "limited": "ltd", "corporation": "corp",
             "incorporated": "inc", "company": "co"}


# ---------------------------------------------------------------- scores
def _cast(S):
    return S.with_columns(pl.col("s1").cast(pl.Int64), pl.col("src").cast(pl.Int8),
                          pl.col("id").cast(pl.Int64), pl.col("p").cast(pl.Float32))


def train_scores():
    f = C.WORK_DIR / "model" / f"stageA_{M.tag(TRAIN_FRAC, QFRAC)}.parquet"
    if not f.exists():
        raise FileNotFoundError(f"missing {f} - add the Stage A output, do not retrain")
    S = _cast(pl.read_parquet(f))
    return S.select(*KEYS, "p", pl.col("label").cast(pl.Int8), pl.col("fold").cast(pl.Int8))


def test_scores():
    from src import predict as P
    S = P.score(1.0, TRAIN_FRAC, QFRAC, C.BLOCK_CAP, C.N_FOLDS, 15_000)
    return _cast(S).select(*KEYS, "p")


def prune(S):
    r = pl.col("p").rank("ordinal", descending=True).over(["s1", "src"])
    return S.filter((pl.col("p") >= PRUNE_P) | (r <= PRUNE_TOP))


# ---------------------------------------------------------------- records
def records(split):
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / f"rec_{split}_n{N.VERSION}_v{VERSION}.parquet"
    if f.exists():
        return pl.read_parquet(f)
    norm = N.normalize_split(split, 1.0)
    R = pl.concat([F._prep(norm[s]).with_columns(pl.lit(s, pl.Int8).alias("src")) for s in (1, 2, 3)])
    del norm
    gc.collect()
    R = R.select(
        "src", pl.col("id").cast(pl.Int64), "cty", "name_core", "addr_norm",
        pl.col("legal").str.split(" ").list.eval(pl.element().replace(LEGAL_MAP))
          .list.unique().list.sort().list.join(" ").alias("lg"),
        pl.col("addr_num1").str.replace_all(r"\D", "").str.strip_chars_start("0").str.slice(0, 9).alias("n1"),
        (pl.col("addr_nums").list.len() == 0).cast(pl.Int8).alias("nonum"),
        pl.col("addr_state").list.first().fill_null("").alias("st"))
    is1, nm = pl.col("src") == 1, pl.col("name_core") != ""
    R = R.with_columns(
        pl.when(nm).then(is1.sum().over(["cty", "name_core"])).alias("fn1"),
        pl.when(nm).then((~is1).sum().over(["cty", "name_core"])).alias("fno"),
        pl.when(nm).then(is1.sum().over(["cty", "st", "name_core"])).alias("fn1st"),
        pl.when(pl.col("addr_norm") != "").then(is1.sum().over(["cty", "addr_norm"])).alias("fa1"),
        pl.col("n1").cast(pl.Int64, strict=False).alias("n1i"))
    R.write_parquet(f)
    return R


def sib_words(R):
    """How much more often a name token appears in S2/S3 than in S1 (label-free sibling signal)."""
    t = (R.select("cty", (pl.col("src") == 1).alias("is1"),
                  pl.col("name_core").str.split(" ").list.unique().alias("t"))
         .explode("t").filter(pl.col("t").is_not_null() & (pl.col("t") != "")))
    n = R.group_by("cty").agg((pl.col("src") == 1).sum().alias("n1"), (pl.col("src") != 1).sum().alias("no"))
    return (t.group_by("cty", "t").agg(pl.col("is1").sum().alias("d1"), (~pl.col("is1")).sum().alias("do"))
            .join(n, on="cty")
            .select("cty", "t",
                    (((pl.col("do") + 1) / pl.col("no")) / ((pl.col("d1") + 1) / pl.col("n1")))
                    .log().cast(pl.Float32).alias("sib"),
                    (pl.col("d1") + pl.col("do")).cast(pl.Float32).alias("tdf")))


def rival_index(R):
    return (R.filter((pl.col("src") == 1) & (pl.col("name_core") != "") & (pl.col("fn1") <= RIVAL_MAX))
            .select("cty", "name_core", pl.col("id").alias("s1r"),
                    pl.col("addr_norm").alias("addr_norm_r"), pl.col("n1").alias("n1_r")))


# ---------------------------------------------------------------- features
def group_feats(S):
    p = pl.col("p")
    S = S.sort(["s1", "p"], descending=[False, True]).with_columns(
        pl.int_range(pl.len()).over("s1").alias("rk"),
        p.rank("ordinal", descending=True).over(["s1", "src"]).alias("rk_src"),
        p.max().over("s1").alias("p_max"),
        p.sum().over("s1").alias("p_sum"),
        p.sum().over(["s1", "src"]).alias("p_sum_src"),
        (p > 0.5).sum().over("s1").alias("n50"),
        (p > 0.9).sum().over("s1").alias("n90"),
        (p > 0.5).sum().over(["s1", "src"]).alias("n50_src"),
        pl.len().over("s1").alias("n_c"),
        p.shift(1).over("s1").alias("p_up"),
        p.shift(-1).over("s1").alias("p_dn"))
    t0 = S.filter(pl.col("rk") == 0).select("s1", pl.col("src").alias("t0_src"), pl.col("id").alias("t0_id"))
    t1 = S.filter(pl.col("rk") == 1).select("s1", pl.col("src").alias("t1_src"),
                                            pl.col("id").alias("t1_id"), pl.col("p").alias("p_2"))
    S = S.join(t0, on="s1", how="left").join(t1, on="s1", how="left")
    top, pc = pl.col("rk") == 0, p.clip(1e-6, 1 - 1e-6)
    return S.with_columns(
        pl.when(top).then(pl.col("t1_src")).otherwise(pl.col("t0_src")).alias("a_src"),
        pl.when(top).then(pl.col("t1_id")).otherwise(pl.col("t0_id")).alias("a_id"),
        pl.when(top).then(pl.col("p_2")).otherwise(pl.col("p_max")).alias("p_anc"),
        (p / pl.col("p_max")).alias("p_rel"),
        (pl.col("p_max") - p).alias("gap_top"),
        (pl.col("p_up") - p).alias("gap_up"),
        (p - pl.col("p_dn")).alias("gap_dn"),
        (pl.col("p_max") - pl.col("p_2")).alias("gap12"),
        (pc / (1 - pc)).log().alias("logit"),
    ).drop("t0_src", "t0_id", "t1_src", "t1_id")


@lru_cache(maxsize=None)
def skel(s):
    """Consonant-class skeleton: 'abheghs' and 'apex' -> 'PKS'."""
    s = re.sub(r"[^a-z]", "", s.replace("x", "ks"))
    s = re.sub(r"ph|bh|b|p|f", "P", s)
    s = re.sub(r"kh|gh|ch|k|g|q|c", "K", s)
    s = re.sub(r"th|dh|t|d", "T", s)
    s = re.sub(r"sh|s|z|j", "S", s)
    s = re.sub(r"[vw]", "V", s)
    s = re.sub(r"[aeiouyh]", "", s)
    return re.sub(r"(.)\1+", r"\1", s)


def _pair_feats(X, sw, R1i):
    X = X.with_row_index("i")
    f = {}

    def col(c):
        return X[c].fill_null("")

    def sim(key, a, b, scorer, scale=100.0):
        x, y = col(a), col(b)
        v = process.cpdist(x.to_list(), y.to_list(), scorer=scorer, workers=C.N_THREADS,
                           dtype=np.float32) / np.float32(scale)
        v[((x == "") | (y == "")).to_numpy()] = np.nan
        f[key] = v.astype(np.float32)

    sim("b_nm_tset", "name_core_1", "name_core_2", fuzz.token_set_ratio)
    sim("b_ad_tset", "addr_norm_1", "addr_norm_2", fuzz.token_set_ratio)
    sim("b_nm_anc", "name_core_2", "name_core_a", fuzz.token_set_ratio)
    sim("b_ad_anc", "addr_norm_2", "addr_norm_a", fuzz.token_set_ratio)
    sim("b_num_edit", "n1_1", "n1_2", Levenshtein.distance, 1.0)
    k1 = [skel(s) for s in col("name_core_1").to_list()]
    k2 = [skel(s) for s in col("name_core_2").to_list()]
    v = process.cpdist(k1, k2, scorer=fuzz.ratio, workers=C.N_THREADS, dtype=np.float32) / np.float32(100)
    v[np.array([not a or not b for a, b in zip(k1, k2)])] = np.nan
    f["b_skel"] = v.astype(np.float32)
    skel.cache_clear()

    lg1, lg2 = pl.col("lg_1").fill_null(""), pl.col("lg_2").fill_null("")
    legal = (pl.when((lg1 == "") & (lg2 == "")).then(0).when((lg1 == "") | (lg2 == "")).then(1)
             .when(lg1 == lg2).then(2).otherwise(3)).cast(pl.Int8)
    core_eq = (pl.col("name_core_1") == pl.col("name_core_2")) & (pl.col("name_core_1") != "")
    d = pl.col("n1i_2") - pl.col("n1i_1")
    nn = (pl.col("n1_1").fill_null("") != "") & (pl.col("n1_2").fill_null("") != "")
    base = X.select(
        "i",
        legal.alias("b_legal"),
        core_eq.cast(pl.Int8).alias("b_core_eq"),
        (core_eq & (legal == 3)).cast(pl.Int8).alias("b_core_eq_legal_conf"),
        d.cast(pl.Float32).alias("b_dnum"),
        d.abs().cast(pl.Float32).alias("b_dnum_abs"),
        pl.when(nn).then((pl.col("n1_1").str.len_chars() == pl.col("n1_2").str.len_chars())
                         .cast(pl.Int8)).alias("b_num_samelen"),
        pl.col("nonum_1").alias("b_nonum_1"), pl.col("nonum_2").alias("b_nonum_2"),
        (pl.col("a_src") == pl.col("src")).cast(pl.Int8).alias("b_anc_same_src"),
        *[pl.col(f"{c}_{k}").cast(pl.Float32).alias(f"b_{c}_{k}")
          for c in ("fn1", "fno", "fn1st", "fa1") for k in (1, 2)])
    base = base.with_columns([pl.Series(k, v) for k, v in f.items()])

    t1, t2 = pl.col("name_core_1").str.split(" "), pl.col("name_core_2").str.split(" ")

    def diff(a, b, pre):
        return (X.select("i", "cty", a.list.set_difference(b).alias("t")).explode("t")
                .filter(pl.col("t").is_not_null() & (pl.col("t") != ""))
                .join(sw, on=["cty", "t"], how="left")
                .group_by("i").agg(pl.len().cast(pl.Float32).alias(f"b_n_{pre}"),
                                   pl.col("sib").max().alias(f"b_{pre}_sib"),
                                   pl.col("tdf").min().alias(f"b_{pre}_mindf")))

    q = (X.filter(pl.col("name_core_2") != "")
         .select("i", "s1", "cty", pl.col("name_core_2").alias("name_core"), "addr_norm_2", "n1_2"))
    x = q.join(R1i, on=["cty", "name_core"]).filter(pl.col("s1r") != pl.col("s1"))
    if x.height:
        a, b = x["addr_norm_2"].fill_null(""), x["addr_norm_r"].fill_null("")
        v = process.cpdist(a.to_list(), b.to_list(), scorer=fuzz.token_set_ratio, workers=C.N_THREADS,
                           dtype=np.float32) / np.float32(100)
        v[((a == "") | (b == "")).to_numpy()] = np.nan
        rv = (x.with_columns(pl.Series("v", v.astype(np.float32)).fill_nan(None))
              .group_by("i").agg(pl.len().cast(pl.Float32).alias("b_rv_n"),
                                 pl.col("v").max().alias("b_rv_ad"),
                                 ((pl.col("n1_2") == pl.col("n1_r")) & (pl.col("n1_2") != ""))
                                 .any().cast(pl.Int8).alias("b_rv_num")))
    else:
        rv = pl.DataFrame(schema={"i": pl.UInt32, "b_rv_n": pl.Float32, "b_rv_ad": pl.Float32, "b_rv_num": pl.Int8})

    for e in (diff(t2, t1, "extra"), diff(t1, t2, "miss"), rv):
        base = base.join(e, on="i", how="left")
    return (base.with_columns(pl.col("b_n_extra").fill_null(0), pl.col("b_n_miss").fill_null(0),
                              pl.col("b_rv_n").fill_null(0),
                              (pl.col("b_ad_tset") - pl.col("b_rv_ad")).alias("b_rv_gap"))
            .sort("i").drop("i"))


def build(split, S):
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / f"feat_{split}_{M.tag(TRAIN_FRAC, QFRAC)}_p{PRUNE_P:g}_t{PRUNE_TOP}_v{VERSION}.parquet"
    if f.exists():
        return pl.read_parquet(f)
    R = records(split)
    sw, R1i = sib_words(R), rival_index(R)
    keep = ["name_core", "addr_norm", "lg", "n1", "n1i", "nonum", "st", "fn1", "fno", "fn1st", "fa1"]
    R1 = R.filter(pl.col("src") == 1).select(pl.col("id").alias("s1"), "cty",
                                             *[pl.col(c).alias(f"{c}_1") for c in keep])
    R2 = R.filter(pl.col("src") != 1).select("src", "id", *[pl.col(c).alias(f"{c}_2") for c in keep])
    RA = R2.select(pl.col("src").alias("a_src"), pl.col("id").alias("a_id"),
                   pl.col("name_core_2").alias("name_core_a"), pl.col("addr_norm_2").alias("addr_norm_a"))
    del R
    gc.collect()
    G = group_feats(prune(S)).sort(KEYS)
    parts, n = [], (G.height + CHUNK - 1) // CHUNK
    with D.step(f"stageB features {split} rows={G.height:,} chunks={n}"):
        for j, a in enumerate(range(0, G.height, CHUNK)):
            g = G.slice(a, CHUNK)
            X = (g.with_row_index("r").join(R1, on="s1", how="left").join(R2, on=["src", "id"], how="left")
                 .join(RA, on=["a_src", "a_id"], how="left").sort("r").drop("r"))
            parts.append(pl.concat([g, _pair_feats(X, sw, R1i)], how="horizontal"))
            del X
            gc.collect()
            print(f"  chunk {j + 1}/{n}", flush=True)
    out = pl.concat(parts)
    out.write_parquet(f)
    return out


def feat_cols(df):
    return [c for c in df.columns if c not in META and df.schema[c].is_numeric()]


# ---------------------------------------------------------------- model
def params():
    return dict(objective="binary", learning_rate=0.03, num_leaves=63, min_data_in_leaf=50,
                feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                seed=C.LGB_SEED, num_threads=C.N_THREADS, verbose=-1)


def fit(X):
    cols = feat_cols(X)
    A, y, fold = M.matrix(X, cols), X["label"].to_numpy(), X["fold"].to_numpy()
    dev, pb = fold >= 0, np.zeros(len(y), np.float32)
    gain = np.zeros(len(cols))
    for k in range(C.N_FOLDS):
        tr, va = dev & (fold != k), fold == k
        dtr = lgb.Dataset(A[tr], y[tr], feature_name=cols)
        m = lgb.train(params(), dtr, 5000, valid_sets=[lgb.Dataset(A[va], y[va], reference=dtr)],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        pb[va] = m.predict(A[va], num_iteration=m.best_iteration)
        if (~dev).any():
            pb[~dev] += m.predict(A[~dev], num_iteration=m.best_iteration) / C.N_FOLDS
        m.save_model(str(OUT / f"stageB_v{VERSION}_f{k}.txt"), num_iteration=m.best_iteration)
        gain += m.feature_importance("gain")
        print(f"  fold {k}: best_iter {m.best_iteration}", flush=True)
    imp = pl.DataFrame({"feature": cols, "gain_pct": np.round(100 * gain / gain.sum(), 2)})
    print(imp.sort("gain_pct", descending=True).head(20))
    return X.select(*KEYS, "label", "fold", "p").with_columns(pl.Series("pb", pb))


# ---------------------------------------------------------------- decision + metric
def own(S, c):
    return S.filter(pl.col(c).rank("ordinal", descending=True).over(["src", "id"]) == 1)


def rule_thr(S, c, t):
    return own(S, c).filter(pl.col(c) >= t)


def rule_ef(S, c, g):
    """Per S1 pick the list length k (0 allowed) with the highest expected F0.5."""
    S = (own(S, c).with_columns(pl.col(c).clip(0.0, 0.999999).pow(g).alias("q"))
         .sort(["s1", "q"], descending=[False, True]))
    S = S.with_columns(pl.int_range(1, pl.len() + 1).over("s1").alias("k"),
                       pl.col("q").cum_sum().over("s1").alias("A"),
                       pl.col("q").sum().over("s1").alias("T"),
                       (1 - pl.col("q")).log().sum().over("s1").exp().alias("E0"))
    S = S.with_columns((1.25 * pl.col("A") / (pl.col("k") + 0.25 * pl.col("T"))).alias("EF"))
    S = S.with_columns(pl.col("EF").max().over("s1").alias("EFm"))
    kb = S.filter(pl.col("EF") == pl.col("EFm")).group_by("s1").agg(pl.col("k").min().alias("kb"))
    return S.join(kb, on="s1").filter((pl.col("k") <= pl.col("kb")) & (pl.col("EFm") > pl.col("E0")))


def rule(S, c, kind, v):
    return rule_thr(S, c, v) if kind == "thr" else rule_ef(S, c, v)


def truth():
    q = B.query_s1("train", 1.0, QFRAC).select(pl.col("s1").cast(pl.Int64), "cty")
    gt = D.load_split("train", 1.0)["pairs"].select(pl.col("s1").cast(pl.Int64), pl.col("src").cast(pl.Int8),
                                                    pl.col("id").cast(pl.Int64))
    gt = gt.join(q.select("s1"), on="s1", how="semi")
    return q.with_columns(pl.Series("fold", M.fold_of(q["s1"].to_numpy()))), gt


def f05(sel, U, gt):
    k = sel.group_by("s1").agg(pl.len().alias("k"))
    t = gt.group_by("s1").agg(pl.len().alias("t"))
    a = sel.join(gt, on=KEYS, how="inner").group_by("s1").agg(pl.len().alias("a"))
    E = (U.join(k, on="s1", how="left").join(t, on="s1", how="left").join(a, on="s1", how="left")
         .with_columns(pl.col("k").fill_null(0), pl.col("t").fill_null(0), pl.col("a").fill_null(0)))
    return E.with_columns(pl.when((pl.col("k") == 0) & (pl.col("t") == 0)).then(1.0)
                          .otherwise(1.25 * pl.col("a") / (pl.col("k") + 0.25 * pl.col("t"))).alias("f"))


GRID = {"thr": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], "ef": [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]}


def tune(O, U, gt, c):
    rows = []
    for kind, vals in GRID.items():
        for v in vals:
            E = f05(rule(O, c, kind, v).select(KEYS), U, gt)
            rows.append({"kind": kind, "v": v, "dev": E.filter(pl.col("fold") >= 0)["f"].mean(),
                         "holdout": E.filter(pl.col("fold") < 0)["f"].mean()})
    T = pl.DataFrame(rows).sort("dev", descending=True)
    print(T)
    best = T.row(0, named=True)
    E = f05(rule(O, c, best["kind"], best["v"]).select(KEYS), U, gt)
    print(E.group_by("cty").agg(pl.col("f").mean().round(4)).sort("cty"))
    return best


# ---------------------------------------------------------------- commands
def cmd_check():
    S = train_scores()
    Sp = prune(S)
    npos, npos_p = S["label"].sum(), Sp["label"].sum()
    print(f"prune: rows {S.height:,} -> {Sp.height:,} | true pairs kept {npos_p / npos:.5f}")
    tr, te = D.load_split("train", 1.0), D.load_split("test", 1.0)
    rate = tr["pairs"].height / (tr["src"][2].height + tr["src"][3].height)
    hit = float((S.filter(pl.col("label") == 1)["p"] > 0.5).mean())
    T = test_scores()
    best = T.group_by(["src", "id"]).agg(pl.col("p").max())
    obs = best.filter(pl.col("p") > 0.5).height / (te["src"][2].height + te["src"][3].height)
    exp = rate * hit
    print(f"orphan check: expected owned share {exp:.3f} | test observed {obs:.3f}")
    print("VERDICT:", "OK (no extra orphans)" if obs >= 0.9 * exp else "ORPHANS (test has ownerless records -> be stricter)")


def cmd_train():
    S = train_scores()
    X = build("train", S)
    print(f"train rows {X.height:,}, features {len(feat_cols(X))}")
    O = fit(X)
    O.write_parquet(OUT / f"oof_v{VERSION}.parquet")
    U, gt = truth()
    print(f"ceiling (true pairs inside pruned candidates): {O['label'].sum() / gt.height:.4f}")
    print("\n=== Stage A score (baseline) ===")
    a = tune(O, U, gt, "p")
    print("\n=== Stage B score ===")
    b = tune(O, U, gt, "pb")
    (OUT / f"best_v{VERSION}.json").write_text(json.dumps(b))
    print(f"\nSTAGE A best dev {a['dev']:.4f} holdout {a['holdout']:.4f}")
    print(f"STAGE B best dev {b['dev']:.4f} holdout {b['holdout']:.4f}  rule={b['kind']} v={b['v']}")
    print("USE STAGE B" if b["dev"] > a["dev"] else "STAGE B NOT BETTER -> keep the Stage A submission")


def _validator(v):
    c = ([Path(v)] if v else []) + [C.DATA_DIR.parent / "utils" / "validate_submission.py"]
    if Path("/kaggle/input").exists():
        c += list(Path("/kaggle/input").rglob("validate_submission.py"))
    return next((p for p in c if p.exists()), None)


def write(O, sel, validator):
    from src import predict as P
    i1 = P._ids(1, 1.0)
    oth = pl.concat([P._ids(s, 1.0).select("eid", "id", pl.lit(s, pl.Int8).alias("src")) for s in (2, 3)])
    base = i1.select(pl.col("eid").alias("source1_entity_id"), pl.col("id").alias("s1"), "country")

    def lists(df, col):
        g = (df.join(oth, on=["src", "id"]).sort(["s1", "pb"], descending=[False, True])
             .group_by("s1", maintain_order=True)
             .agg(pl.col("eid").unique(maintain_order=True).str.join(",").alias(col)))
        return base.join(g, on="s1", how="left").with_columns(pl.col(col).fill_null(""))

    C.OUT_DIR.mkdir(parents=True, exist_ok=True)
    mf, cf = C.OUT_DIR / "matching_results.tsv", C.OUT_DIR / "candidate_pairs.tsv"
    lists(O, "candidate_entity_ids").select("source1_entity_id", "candidate_entity_ids") \
        .write_csv(cf, separator="\t", quote_style="never")
    m = lists(sel, "matched_entity_ids")
    m.select("source1_entity_id", "matched_entity_ids").write_csv(mf, separator="\t", quote_style="never")
    n = pl.col("matched_entity_ids").str.split(",").list.eval(pl.element().filter(pl.element() != "")).list.len()
    print(m.with_columns(n.alias("n")).group_by("country").agg(
        pl.len().alias("s1s"), (pl.col("n") == 0).mean().round(4).alias("empty_share"),
        pl.col("n").mean().round(3).alias("mean_matches")).sort("country"))
    v = _validator(validator)
    if v:
        r = subprocess.run([sys.executable, str(v), "--matching", str(mf), "--candidate", str(cf),
                            "--test-dir", str(C.DATA_DIR / "test")])
        print("validator exit code", r.returncode, "(0 = PASS)")
    else:
        print("validator not found - run it on the laptop")


def cmd_predict(kind, v, validator):
    cfg = {"kind": kind, "v": v} if kind else json.loads((OUT / f"best_v{VERSION}.json").read_text())
    print("rule:", cfg)
    X = build("test", test_scores())
    models = [lgb.Booster(model_file=str(OUT / f"stageB_v{VERSION}_f{k}.txt")) for k in range(C.N_FOLDS)]
    cols = models[0].feature_name()
    lost = [c for c in cols if c not in X.columns]
    if lost:
        raise KeyError(f"test lacks features {lost}")
    pb = np.zeros(X.height, np.float32)
    with D.step(f"stageB predict rows={X.height:,}"):
        for a in range(0, X.height, 2_000_000):
            A = M.matrix(X.slice(a, 2_000_000), cols)
            pb[a:a + len(A)] = np.mean([m.predict(A) for m in models], axis=0)
    O = X.select(*KEYS, "p").with_columns(pl.Series("pb", pb))
    O.write_parquet(OUT / f"test_scores_v{VERSION}.parquet")
    write(O, rule(O, "pb", cfg["kind"], cfg["v"]), validator)


def main():
    a = argparse.ArgumentParser()
    a.add_argument("cmd", choices=["check", "train", "predict"])
    a.add_argument("--kind", choices=["thr", "ef"], default=None)
    a.add_argument("--v", type=float, default=None)
    a.add_argument("--validator", default=None)
    x = a.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if x.cmd == "check":
        cmd_check()
    elif x.cmd == "train":
        cmd_train()
    else:
        cmd_predict(x.kind, x.v, x.validator)


if __name__ == "__main__":
    main()