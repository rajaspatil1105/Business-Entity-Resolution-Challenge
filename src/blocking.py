"""Step 2: candidate generation (blocking).

Per country and per source (S2, S3), TF-IDF blockers retrieve candidates for every Source 1 query:
  name  : char n-grams (NAME_NGRAM) of the core name (typos, spacing, suffix noise)
  addr  : address tokens incl. house numbers (pairs whose names differ completely)
  combo : 50/50 blend of both (weak name + good address, or the reverse)
  rev   : reverse name search. Every S2/S3 record searches ALL S1 of its country and keeps its
          top REV_K. Each S2/S3 has at most one true S1, so this rescues typo names that sink
          below many look-alikes in the forward search. Only pairs whose S1 is a query are kept.
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
K = {**C.BLOCK_K, **({"rev": C.REV_K} if C.REV_K else {})}
BLOCKERS = list(K)


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
        v = TfidfVectorizer(analyzer="char_wb", ngram_range=tuple(C.NAME_NGRAM), **kw)
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


def _reverse(X, bounds, frames, qmask):
    """Each S2/S3 record -> its top REV_K S1 (full S1 pool); keep pairs whose S1 is a query."""
    PT = X[bounds[0]:bounds[1]].T.tocsr()
    s1ids = frames[0]["id"].to_numpy()
    parts = []
    for j, s in ((1, 2), (2, 3)):
        P = X[bounds[j]:bounds[j + 1]]
        pids = frames[j]["id"].to_numpy()
        for a in range(0, P.shape[0], C.Q_CHUNK):
            r, c, v, k = _topk(P[a:a + C.Q_CHUNK], PT, C.REV_K)
            m = qmask[c]
            parts.append(pl.DataFrame({"s1": s1ids[c[m]], "src": np.full(int(m.sum()), s, np.int8),
                                       "id": pids[a + r[m]], "r_rev": k[m], "s_rev": v[m]}))
    return pl.concat(parts) if parts else None


def _block_country(s1, qmask, pools, label):
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
    for b in C.BLOCK_K:
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

    if C.REV_K and mats["name"] is not None and len(qids):
        with D.step(f"  rev {label}: {int(bounds[-1] - bounds[1]):,} records -> {s1.height:,} S1"):
            df = _reverse(mats["name"], bounds, frames, qmask)
        if df is not None:
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
    tag = (f"{split}_f{frac:g}_q{qfrac:g}_n{N.VERSION}_k{max(C.BLOCK_K.values())}_df{C.MAX_DF:g}"
           f"_ng{C.NAME_NGRAM[0]}{C.NAME_NGRAM[1]}_rev{C.REV_K}q")
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
            parts.append(_block_country(s1, qmask, pools, cty))

    cand = pl.concat(parts)
    rc = [f"r_{b}" for b in C.BLOCK_K]
    if C.REV_K:   # re-rank reverse hits per (S1, source) so hub S1s cannot flood the low positions
        cand = cand.with_columns((pl.col("s_rev").rank("ordinal", descending=True).over(["s1", "src"]) - 1)
                                 .cast(pl.Int16).alias("q_rev"))
        rc.append("q_rev")
    sc = [f"s_{b}" for b in BLOCKERS]
    cand = (cand
            .with_columns(pl.min_horizontal(rc).alias("min_rank"),
                          pl.max_horizontal(sc).alias("max_score"),
                          pl.sum_horizontal([pl.col(c).is_not_null() for c in rc]).cast(pl.Int8).alias("nb"))
            .sort(["s1", "src", "min_rank", "max_score"], descending=[False, False, False, True])
            .with_columns(pl.int_range(pl.len()).over(["s1", "src"]).cast(pl.Int16).alias("pos")))
    cand.write_parquet(f)
    return cand
