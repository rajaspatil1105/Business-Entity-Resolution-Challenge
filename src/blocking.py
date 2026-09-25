"""Step 2: candidate generation (blocking).

Per country and per source (S2, S3), three TF-IDF blockers retrieve the top-k records for
every Source 1 query:
  name  : char 3-grams of the core name (typos, spacing, suffix noise)
  addr  : address tokens incl. house numbers (pairs whose names differ completely)
  combo : 50/50 blend of both (weak name + good address, or the reverse)
TF-IDF is fit on each split's own pool (unsupervised, no labels). Tokens present in more than
MAX_DF of a country pool are dropped: they carry little signal and dominate compute.
The union is ordered per (S1, source) by best rank across blockers -> column `pos`.
"""
import numpy as np
import polars as pl
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from src import config as C
from src import data as D
from src import normalize as N

try:
    from sparse_dot_topn import sp_matmul_topn
except ImportError:          # exact fallback: fine for small local samples only
    sp_matmul_topn = None

KEYS = ["s1", "src", "id"]
BLOCKERS = list(C.BLOCK_K)


def query_mask(ids, qfrac):
    """Deterministic S1 query subsample, independent of the frac used in data.load_split."""
    if qfrac >= 1:
        return np.ones(len(ids), bool)
    return ((ids // 10_000) % 10_000) < int(round(qfrac * 10_000))


def query_s1(split, frac, qfrac):
    s1 = N.normalize_split(split, frac)[1].select(pl.col("id").alias("s1"), "cty")
    return s1.filter(pl.Series(query_mask(s1["s1"].to_numpy(), qfrac)))


def _tfidf(texts, kind):
    kw = dict(sublinear_tf=True, dtype=np.float32, max_df=max(int(C.MAX_DF * len(texts)), 2))
    if kind == "name":
        v = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), **kw)
    else:
        v = TfidfVectorizer(token_pattern=r"\S+", **kw)
    try:
        return v.fit_transform(texts).tocsr()
    except ValueError:       # empty vocabulary (tiny or empty pool)
        return None


def _topk(Q, PT, k):
    """Top-k columns per row of Q @ PT -> (row, col, score, rank)."""
    if sp_matmul_topn is not None:
        S = sp_matmul_topn(Q, PT, top_n=k, threshold=C.MIN_SIM, n_threads=C.N_THREADS)
    else:
        S = Q @ PT
    S = S.tocoo()
    m = S.data >= C.MIN_SIM
    r, c, v = S.row[m], S.col[m], S.data[m].astype(np.float32)
    o = np.lexsort((-v, r))
    r, c, v = r[o], c[o], v[o]
    rank = np.arange(len(r)) - np.searchsorted(r, r)
    keep = rank < k
    return r[keep], c[keep], v[keep], rank[keep].astype(np.int16)


def _block_country(s1, qmask, pools):
    frames = [s1, pools[2], pools[3]]
    bounds = np.cumsum([0] + [f.height for f in frames])
    names = pl.concat([f["name_core"] for f in frames]).to_list()
    addrs = pl.concat([f["addr_norm"] for f in frames]).to_list()
    mats = {"name": _tfidf(names, "name"), "addr": _tfidf(addrs, "addr")}
    if mats["name"] is not None and mats["addr"] is not None:
        h = np.float32(np.sqrt(0.5))
        mats["combo"] = sp.hstack([mats["name"] * h, mats["addr"] * h], format="csr", dtype=np.float32)
    qids = s1["id"].to_numpy()[qmask]

    res = None
    for b in BLOCKERS:
        X = mats.get(b)
        if X is None or len(qids) == 0:
            continue
        Q = X[bounds[0]:bounds[1]][qmask]
        parts = []
        for j, s in ((1, 2), (2, 3)):
            if bounds[j + 1] == bounds[j]:
                continue
            PT = X[bounds[j]:bounds[j + 1]].T.tocsr()
            pids = frames[j]["id"].to_numpy()
            for a in range(0, Q.shape[0], C.Q_CHUNK):
                r, c, v, k = _topk(Q[a:a + C.Q_CHUNK], PT, C.BLOCK_K[b])
                parts.append(pl.DataFrame({"s1": qids[a + r], "src": np.full(len(r), s, np.int8),
                                           "id": pids[c], f"r_{b}": k, f"s_{b}": v}))
        if parts:
            df = pl.concat(parts)
            res = df if res is None else res.join(df, on=KEYS, how="full", coalesce=True)

    if res is None:
        res = pl.DataFrame(schema={"s1": pl.Int64, "src": pl.Int8, "id": pl.Int64})
    for b in BLOCKERS:
        if f"r_{b}" not in res.columns:
            res = res.with_columns(pl.lit(None, pl.Int16).alias(f"r_{b}"),
                                   pl.lit(None, pl.Float32).alias(f"s_{b}"))
    return res.select(KEYS + [x for b in BLOCKERS for x in (f"r_{b}", f"s_{b}")])


def block_split(split, frac=1.0, qfrac=1.0):
    """All candidates (uncapped) for the S1 queries of a split; cached as parquet."""
    tag = (f"{split}_f{frac:g}_q{qfrac:g}_n{N.VERSION}"
           f"_k{max(C.BLOCK_K.values())}_df{C.MAX_DF:g}")
    d = C.WORK_DIR / "cand"
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"{tag}.parquet"
    if f.exists():
        return pl.read_parquet(f)

    norm = N.normalize_split(split, frac)
    parts = []
    for cty in sorted(norm[1]["cty"].unique().to_list()):
        s1 = norm[1].filter(pl.col("cty") == cty)
        qmask = query_mask(s1["id"].to_numpy(), qfrac)
        if not qmask.any():
            continue
        pools = {s: norm[s].filter(pl.col("cty") == cty) for s in (2, 3)}
        with D.step(f"block {tag} [{cty}] queries={int(qmask.sum()):,}"):
            parts.append(_block_country(s1, qmask, pools))

    rc = [f"r_{b}" for b in BLOCKERS]
    sc = [f"s_{b}" for b in BLOCKERS]
    cand = (pl.concat(parts)
            .with_columns(pl.min_horizontal(rc).alias("min_rank"),
                          pl.max_horizontal(sc).alias("max_score"),
                          pl.sum_horizontal([pl.col(c).is_not_null() for c in rc]).cast(pl.Int8).alias("nb"))
            .sort(["s1", "src", "min_rank", "max_score"], descending=[False, False, False, True])
            .with_columns(pl.int_range(pl.len()).over(["s1", "src"]).cast(pl.Int16).alias("pos")))
    cand.write_parquet(f)
    return cand
