"""Step 2: candidate generation (blocking).

Per country and per source (S2, S3), TF-IDF blockers retrieve candidates for every Source 1 query:
  name  : char n-grams (NAME_NGRAM) of the core name (typos, spacing, suffix noise)
  addr  : address tokens incl. house numbers (pairs whose names differ completely)
  combo : 50/50 blend of both (weak name + good address, or the reverse)
  rev   : reverse name search. Every S2/S3 record searches all S1 of its country and keeps its
          top REV_K (each S2/S3 has at most one true S1). Only pairs whose S1 is a query are kept.
Speed (STATE_BUCKETS): TF-IDF is fit once per country (scores stay comparable), but a query only
searches records that share a state with it or have no state; a query without a state searches
the whole country. True pairs share the state whenever both records have one (norm report), so
missing states never drop a pair. Per-bucket top-k lists are merged and re-ranked.
TF-IDF is fit on each split's own pool (unsupervised, no labels). Tokens present in more than
MAX_DF of a country pool are dropped. Each (country, blocker) result is checkpointed, so a rerun
after a crash resumes. The union is ordered per (S1, source) by best rank -> column `pos`.
"""
import shutil

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
EMPTY = np.empty(0, np.int64)


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


def _mats(frames):
    names = pl.concat([f["name_core"] for f in frames]).to_list()
    addrs = pl.concat([f["addr_norm"] for f in frames]).to_list()
    m = {"name": _tfidf(names, "name"), "addr": _tfidf(addrs, "addr")}
    if m["name"] is not None and m["addr"] is not None:
        h = np.float32(np.sqrt(0.5))
        m["combo"] = sp.hstack([m["name"] * h, m["addr"] * h], format="csr", dtype=np.float32)
    return m


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


def _states(df):
    """addr_state as a list of state codes per row (empty list = no state)."""
    c = df["addr_state"]
    s = (c.cast(pl.List(pl.Utf8)) if isinstance(c.dtype, pl.List)
         else c.cast(pl.Utf8).fill_null("").str.split(" "))
    return s.list.eval(pl.element().filter(pl.element().is_not_null() & (pl.element() != "")))


def _members(states):
    """state -> sorted row indices, plus rows without a state. Buckets off: all rows 'no state'."""
    n = len(states)
    if not C.STATE_BUCKETS:
        return {}, np.arange(n)
    d = pl.DataFrame({"row": np.arange(n), "st": states}).explode("st")
    none = np.unique(d.filter(pl.col("st").is_null())["row"].to_numpy())
    groups = {}
    for key, g in d.filter(pl.col("st").is_not_null()).group_by("st"):
        groups[key[0] if isinstance(key, tuple) else key] = np.unique(g["row"].to_numpy())
    return groups, none


def _search(Xq, qst, Xp, pst, k):
    """Top-k pool rows per query row -> (q, p, score, rank). A query searches pool rows of its own
    state(s) plus pool rows without a state; a query without a state searches the whole pool."""
    qg, qn = _members(qst)
    pg, pn = _members(pst)
    jobs = [(qi, np.union1d(pg.get(s, EMPTY), pn)) for s, qi in qg.items()]
    if len(qn):
        jobs.append((qn, np.arange(Xp.shape[0])))
    R, P, V = [], [], []
    for qi, pi in jobs:
        if len(qi) == 0 or len(pi) == 0:
            continue
        PT = Xp[pi].T.tocsr()
        Q = Xq[qi]
        for a in range(0, len(qi), C.Q_CHUNK):
            r, c, v, _ = _topk(Q[a:a + C.Q_CHUNK], PT, k)
            R.append(qi[a + r])
            P.append(pi[c])
            V.append(v)
    if not R:
        return EMPTY, EMPTY, np.empty(0, np.float32), np.empty(0, np.int16)
    df = (pl.DataFrame({"q": np.concatenate(R), "p": np.concatenate(P), "v": np.concatenate(V)})
          .group_by("q", "p").agg(pl.col("v").max())
          .with_columns(pl.col("v").rank("ordinal", descending=True).over("q").alias("k"))
          .filter(pl.col("k") <= k))
    return (df["q"].to_numpy(), df["p"].to_numpy(), df["v"].to_numpy().astype(np.float32),
            (df["k"].to_numpy() - 1).astype(np.int16))


def _block_country(s1, qmask, pools, cty, pdir):
    frames = [s1, pools[2], pools[3]]
    bounds = np.cumsum([0] + [f.height for f in frames])
    sts = [_states(f) for f in frames]
    ns = [float((s.list.len().fill_null(0) == 0).mean() or 0) for s in sts]
    print(f"  [{cty}] no-state share  S1 {ns[0]:.1%} | S2 {ns[1]:.1%} | S3 {ns[2]:.1%} | "
          f"state buckets {'on' if C.STATE_BUCKETS else 'off'}")
    qidx = np.flatnonzero(qmask)
    s1ids = s1["id"].to_numpy()
    mats = {}

    res = None
    for b in BLOCKERS:
        cp = pdir / f"{cty}_{b}.parquet"
        if cp.exists():
            df = pl.read_parquet(cp)
        else:
            if not mats:
                with D.step(f"  tfidf [{cty}]"):
                    mats.update(_mats(frames))
            X = mats.get("name" if b == "rev" else b)
            parts = []
            if X is not None and len(qidx):
                with D.step(f"  {b} [{cty}]"):
                    S1X = X[bounds[0]:bounds[1]]
                    for j, s in ((1, 2), (2, 3)):
                        if bounds[j + 1] == bounds[j]:
                            continue
                        PX = X[bounds[j]:bounds[j + 1]]
                        pids = frames[j]["id"].to_numpy()
                        if b == "rev":
                            q, p, v, k = _search(PX, sts[j], S1X, sts[0], C.REV_K)
                            m = qmask[p]
                            parts.append(pl.DataFrame({"s1": s1ids[p[m]], "src": np.full(int(m.sum()), s, np.int8),
                                                       "id": pids[q[m]], "r_rev": k[m], "s_rev": v[m]}))
                        else:
                            q, p, v, k = _search(S1X[qidx], sts[0].gather(qidx), PX, sts[j], K[b])
                            parts.append(pl.DataFrame({"s1": s1ids[qidx[q]], "src": np.full(len(q), s, np.int8),
                                                       "id": pids[p], f"r_{b}": k, f"s_{b}": v}))
            df = pl.concat(parts) if parts else pl.DataFrame(
                schema={"s1": pl.Int64, "src": pl.Int8, "id": pl.Int64, f"r_{b}": pl.Int16, f"s_{b}": pl.Float32})
            df.write_parquet(cp)
        if df.height:
            res = df if res is None else res.join(df, on=KEYS, how="full", coalesce=True)

    if res is None:
        res = pl.DataFrame(schema={"s1": pl.Int64, "src": pl.Int8, "id": pl.Int64})
    for b in BLOCKERS:
        if f"r_{b}" not in res.columns:
            res = res.with_columns(pl.lit(None, pl.Int16).alias(f"r_{b}"),
                                   pl.lit(None, pl.Float32).alias(f"s_{b}"))
    return res.select(KEYS + [x for b in BLOCKERS for x in (f"r_{b}", f"s_{b}")])


def tag(split, frac, qfrac):
    return (f"{split}_f{frac:g}_q{qfrac:g}_n{N.VERSION}_k{max(C.BLOCK_K.values())}_df{C.MAX_DF:g}"
            f"_ng{C.NAME_NGRAM[0]}{C.NAME_NGRAM[1]}_rev{C.REV_K}q_sb{int(C.STATE_BUCKETS)}")


def _rank(cand, cap=None):
    """Order candidates per (S1, source) by best rank -> `pos`; optional cap keeps pos < cap."""
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
    return cand.filter(pl.col("pos") < cap) if cap else cand


def block_split(split, frac=1.0, qfrac=1.0):
    """Candidates for the S1 queries of a split; cached as parquet.
    Train: uncapped (report needs the cap table). Test: capped at BLOCK_CAP per country before concat.
    Ranking is per (S1, source) and each S1 has one country, so this equals capping at the end."""
    import gc
    base = tag(split, frac, qfrac)
    cap = C.BLOCK_CAP if (split == "test" and C.CAP_TEST) else None
    t = base + (f"_cap{cap}" if cap else "")
    d = C.WORK_DIR / "cand"
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"{t}.parquet"
    if f.exists():
        return pl.read_parquet(f)
    pdir = d / f"{base}_parts"      # per-blocker checkpoints shared by capped/uncapped runs
    pdir.mkdir(exist_ok=True)

    norm = N.normalize_split(split, frac)
    parts = []
    for cty in sorted(norm[1]["cty"].unique().to_list()):
        cf = pdir / (f"{cty}_final" + (f"_cap{cap}" if cap else "") + ".parquet")
        if cf.exists():                      # country already finished -> resume
            parts.append(pl.read_parquet(cf))
            continue
        s1 = norm[1].filter(pl.col("cty") == cty)
        qmask = query_mask(s1["id"].to_numpy(), qfrac)
        if not qmask.any():
            continue
        pools = {s: norm[s].filter(pl.col("cty") == cty) for s in (2, 3)}
        with D.step(f"block {t} [{cty}] queries={int(qmask.sum()):,}"):
            res = _rank(_block_country(s1, qmask, pools, cty, pdir), cap)
            res.write_parquet(cf)
        parts.append(res)
        del s1, pools, res
        gc.collect()

    cand = pl.concat(parts)
    cand.write_parquet(f)
    return cand
    shutil.rmtree(pdir, ignore_errors=True)
    return cand
