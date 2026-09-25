"""Step 3: local pair features (Stage A input).

For every blocked candidate (pos < cap) of every S1 query, compute features that look only at
the pair itself: fuzzy name/address scores, IDF-weighted token overlap (rare-token signal; IDF
fit on the split's own pool, so test/France gets its own IDF), legal suffix, house numbers,
state, flags, lengths, plus blocker ranks/scores. Context over all candidates is Stage B.
Missing comparisons stay null/NaN (LightGBM handles them). `cty` is metadata, never a feature.
"""
import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

from src import blocking as B
from src import config as C
from src import data as D
from src import normalize as N

VERSION = 1
STR = ["name_core", "name_sq", "name_alias", "legal", "addr_norm", "addr_num1"]
LST = ["addr_state", "addr_nums"]
FLG = ["name_nonlatin", "is_handle", "name_phone", "addr_nonlatin", "addr_empty", "has_landmark"]
COLS = STR + LST + FLG
META = ["s1", "id", "cty", "label"]


def feature_cols(df):
    return [c for c in df.columns if c not in META]


def _prep(df):
    miss = [c for c in COLS if c not in df.columns]
    if miss:
        raise KeyError(f"normalized frame lacks {miss}; available: {df.columns}")
    ex = [pl.col(c).cast(pl.Utf8).fill_null("").str.strip_chars().alias(c) for c in STR]
    for c in LST:
        e = (pl.col(c).cast(pl.List(pl.Utf8)) if isinstance(df.schema[c], pl.List)
             else pl.col(c).cast(pl.Utf8).fill_null("").str.split(" "))
        ex.append(e.list.eval(pl.element().filter(pl.element().is_not_null() & (pl.element() != "")))
                  .alias(c))
    ex += [pl.col(c).cast(pl.Int8).fill_null(0).alias(c) for c in FLG]
    return df.select("id", "cty", *ex)


def _idf(norm, col):
    """Per-country IDF over the split's S1+S2+S3 pool; idfp = share of token occurrences that
    are more common (0 = very common, ~1 = rare)."""
    t = pl.concat([norm[s].select("cty", pl.col(col).cast(pl.Utf8).fill_null("")
                                  .str.split(" ").list.unique().alias("t")) for s in (1, 2, 3)])
    n = t.group_by("cty").agg(pl.len().alias("n"))
    return (t.explode("t").filter(pl.col("t").is_not_null() & (pl.col("t") != ""))
            .group_by("cty", "t").agg(pl.len().alias("df")).join(n, on="cty")
            .with_columns(((((pl.col("n") + 1) / (pl.col("df") + 1)).log()) + 1).cast(pl.Float32).alias("idf"))
            .sort("df", descending=True)
            .with_columns((pl.col("df").cum_sum().over("cty") / pl.col("df").sum().over("cty"))
                          .cast(pl.Float32).alias("idfp"))
            .select("cty", "t", "idf", "idfp"))


def _wjac(P, col, pre, idf):
    imax = idf["idf"].max()

    def toks(k):
        return (P.select("i", "cty", pl.col(f"{col}_{k}").str.split(" ").list.unique().alias("t"))
                .explode("t").filter(pl.col("t").is_not_null() & (pl.col("t") != ""))
                .join(idf, on=["cty", "t"], how="left")
                .with_columns(pl.col("idf").fill_null(imax), pl.col("idfp").fill_null(1.0)))

    a, b = toks(1), toks(2)
    parts = [a.group_by("i").agg(pl.col("idf").sum().alias("wa")),
             b.group_by("i").agg(pl.col("idf").sum().alias("wb")),
             a.join(b.select("i", "t"), on=["i", "t"]).group_by("i")
              .agg(pl.col("idf").sum().alias("wi"), pl.col("idfp").max().alias("pi")),
             a.join(b.select("i", "t"), on=["i", "t"], how="anti").group_by("i")
              .agg(pl.col("idfp").max().alias("m1")),
             b.join(a.select("i", "t"), on=["i", "t"], how="anti").group_by("i")
              .agg(pl.col("idfp").max().alias("m2"))]
    out = P.select("i")
    for x in parts:
        out = out.join(x, on="i", how="left")
    ok = pl.col("wa").is_not_null() & pl.col("wb").is_not_null()
    wi = pl.col("wi").fill_null(0)
    return out.select(
        "i",
        pl.when(ok).then(wi / (pl.col("wa") + pl.col("wb") - wi)).alias(f"{pre}_wjac"),
        pl.when(ok).then(pl.col("pi").fill_null(0)).alias(f"{pre}_share_p"),
        pl.when(ok).then(pl.max_horizontal(pl.col("m1").fill_null(0), pl.col("m2").fill_null(0)))
          .alias(f"{pre}_miss_p"))


def _features(P, idf):
    P = P.with_row_index("i")
    f = {}

    def sim(key, c1, c2, scorer, scale=100.0):
        v = process.cpdist(P[c1].to_list(), P[c2].to_list(), scorer=scorer,
                           workers=C.N_THREADS, dtype=np.float32) / np.float32(scale)
        v[((P[c1] == "") | (P[c2] == "")).to_numpy()] = np.nan
        f[key] = v.astype(np.float32)
        return f[key]

    for key, sc, s in (("ratio", fuzz.ratio, 100.0), ("tset", fuzz.token_set_ratio, 100.0),
                       ("tsort", fuzz.token_sort_ratio, 100.0), ("part", fuzz.partial_ratio, 100.0),
                       ("jw", JaroWinkler.normalized_similarity, 1.0)):
        sim(f"nm_{key}", "name_core_1", "name_core_2", sc, s)
    sim("sq_ratio", "name_sq_1", "name_sq_2", fuzz.ratio)
    sim("sq_part", "name_sq_1", "name_sq_2", fuzz.partial_ratio)
    for key, sc in (("ratio", fuzz.ratio), ("tset", fuzz.token_set_ratio), ("part", fuzz.partial_ratio)):
        sim(f"ad_{key}", "addr_norm_1", "addr_norm_2", sc)
    a1 = sim("_a1", "name_alias_1", "name_core_2", fuzz.token_set_ratio)
    a2 = sim("_a2", "name_core_1", "name_alias_2", fuzz.token_set_ratio)
    f["nm_alias"] = np.fmax(a1, a2)
    del f["_a1"], f["_a2"]

    n1, n2 = (pl.col(f"addr_nums_{k}").list.len().fill_null(0) for k in (1, 2))
    ni = pl.col("addr_nums_1").list.set_intersection(pl.col("addr_nums_2")).list.len().fill_null(0)
    t1, t2 = (pl.col(f"addr_state_{k}").list.len().fill_null(0) for k in (1, 2))
    ti = pl.col("addr_state_1").list.set_intersection(pl.col("addr_state_2")).list.len().fill_null(0)

    def both(c):
        return (pl.col(f"{c}_1") != "") & (pl.col(f"{c}_2") != "")

    blk = [c for c in P.columns if c.startswith(("r_", "s_", "q_"))] + ["min_rank", "max_score", "nb", "pos"]
    X = P.select(
        "i", "s1", "src", "id", "cty", *blk,
        pl.when(both("legal")).then((pl.col("legal_1") == pl.col("legal_2")).cast(pl.Int8)).alias("legal_eq"),
        (pl.col("legal_1") != "").cast(pl.Int8).alias("legal_has1"),
        (pl.col("legal_2") != "").cast(pl.Int8).alias("legal_has2"),
        ni.alias("num_inter"), n1.alias("num_n1"), n2.alias("num_n2"),
        pl.when((n1 > 0) & (n2 > 0)).then(ni / (n1 + n2 - ni)).alias("num_jac"),
        ((n1 > 0) & (n2 > 0) & (ni == 0)).cast(pl.Int8).alias("num_conflict"),
        pl.when(both("addr_num1")).then((pl.col("addr_num1_1") == pl.col("addr_num1_2")).cast(pl.Int8))
          .alias("num1_eq"),
        pl.when((t1 > 0) & (t2 > 0)).then((ti > 0).cast(pl.Int8)).alias("state_eq"),
        *[pl.col(f"{c}_{k}").str.len_chars().alias(f"len_{c}_{k}") for c in ("name_core", "addr_norm") for k in (1, 2)],
        *[pl.col(f"{c}_{k}") for c in FLG for k in (1, 2)],
    ).with_columns([pl.Series(k, v) for k, v in f.items()])
    X = (X.join(_wjac(P, "name_core", "nm", idf["name_core"]), on="i", how="left")
          .join(_wjac(P, "addr_norm", "ad", idf["addr_norm"]), on="i", how="left")
          .sort("i").drop("i"))
    return X.with_columns(pl.col(pl.Float64).cast(pl.Float32))


def build(split, frac=1.0, qfrac=1.0, cap=None):
    """Features for all candidates with pos < cap; label column for train. Cached as parquet."""
    cap = cap or C.BLOCK_CAP
    tag = (f"{split}_f{frac:g}_q{qfrac:g}_cap{cap}_n{N.VERSION}_df{C.MAX_DF:g}"
           f"_ng{C.NAME_NGRAM[0]}{C.NAME_NGRAM[1]}_rev{C.REV_K}_v{VERSION}")
    d = C.WORK_DIR / "feat"
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"{tag}.parquet"
    if f.exists():
        return pl.read_parquet(f)

    cand = B.block_split(split, frac, qfrac).filter(pl.col("pos") < cap)
    norm = N.normalize_split(split, frac)
    s1 = _prep(norm[1]).rename({**{c: f"{c}_1" for c in COLS}, "id": "s1"})
    oth = (pl.concat([_prep(norm[s]).drop("cty").with_columns(pl.lit(s, pl.Int8).alias("src"))
                      for s in (2, 3)])
           .rename({c: f"{c}_2" for c in COLS}))
    idf = {c: _idf(norm, c) for c in ("name_core", "addr_norm")}
    P = cand.join(s1, on="s1").join(oth, on=["src", "id"]).sort(["s1", "src", "pos"])

    parts = []
    with D.step(f"features {tag} pairs={P.height:,}"):
        for a in range(0, P.height, C.FEAT_CHUNK):
            parts.append(_features(P.slice(a, C.FEAT_CHUNK), idf))
    out = pl.concat(parts)
    if split == "train":
        gt = D.load_split(split, frac)["pairs"]
        gt = gt.select([pl.col(k).cast(out.schema[k]) for k in B.KEYS]).with_columns(pl.lit(1, pl.Int8).alias("label"))
        out = out.join(gt, on=B.KEYS, how="left").with_columns(pl.col("label").fill_null(0))
    out.write_parquet(f)
    return out
