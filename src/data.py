"""Data loading, deterministic sampling, parquet caching, step timing."""
import time
from contextlib import contextmanager

import polars as pl
import psutil

from src import config as C

READ_OPTS = dict(separator="\t", quote_char=None, infer_schema_length=0, encoding="utf8-lossy")


@contextmanager
def step(name):
    t0 = time.time()
    print(f"[{name}] ...", flush=True)
    yield
    rss = psutil.Process().memory_info().rss / 1e9
    print(f"[{name}] {time.time() - t0:.1f}s | RSS {rss:.2f} GB", flush=True)


def _scan_source(split, s):
    return pl.scan_csv(C.source_path(split, s), **READ_OPTS).select(
        pl.col("entity_id").str.slice(3).cast(pl.Int64).alias("id"),
        pl.lit(s, dtype=pl.Int8).alias("src"),
        pl.col("business_name").fill_null("").alias("name"),
        pl.col("business_address").fill_null("").alias("addr"),
        pl.col("country").fill_null("").alias("country"),
    )


def load_gt():
    """Returns (gt_s1[s1], pairs[s1, src, id]). Key for S2/S3 records is (src, id)."""
    gt = pl.read_csv(C.GT_PATH, **READ_OPTS).select(
        pl.col("source1_entity_id").str.slice(3).cast(pl.Int64).alias("s1"),
        pl.col("matched_entity_ids").fill_null("").alias("m"),
    )
    pairs = (
        gt.with_columns(pl.col("m").str.split(","))
        .explode("m")
        .with_columns(pl.col("m").str.strip_chars())
        .filter(pl.col("m").str.len_chars() > 0)
        .select(
            "s1",
            pl.col("m").str.slice(1, 1).cast(pl.Int8).alias("src"),
            pl.col("m").str.slice(3).cast(pl.Int64).alias("id"),
        )
    )
    return gt.select("s1"), pairs


def _keep(col, frac):
    return (pl.col(col) % 10_000) < int(round(frac * 10_000))


def load_split(split, frac=1.0):
    """Train sample keeps whole S1 groups (S1 + all its matches) plus the same share of decoys."""
    tag = f"{split}_f{frac:g}"
    cache = C.WORK_DIR / "raw"
    cache.mkdir(parents=True, exist_ok=True)
    files = {s: cache / f"{tag}_s{s}.parquet" for s in C.SOURCES}
    gt_files = (cache / f"{tag}_gt_s1.parquet", cache / f"{tag}_gt_pairs.parquet")
    out = {"split": split, "frac": frac}

    need = list(files.values()) + (list(gt_files) if split == "train" else [])
    if all(f.exists() for f in need):
        out["src"] = {s: pl.read_parquet(f) for s, f in files.items()}
        if split == "train":
            out["gt_s1"], out["pairs"] = (pl.read_parquet(f) for f in gt_files)
        return out

    with step(f"load {tag}"):
        srcs = {}
        if split == "train":
            gt_s1, pairs = load_gt()
            kept = pairs
            lf1 = _scan_source(split, 1)
            if frac < 1:
                gt_s1 = gt_s1.filter(_keep("s1", frac))
                kept = pairs.filter(_keep("s1", frac))
                lf1 = lf1.filter(_keep("id", frac))
            srcs[1] = lf1.collect()
            for s in (2, 3):
                lf = _scan_source(split, s)
                if frac < 1:
                    m_all = pairs.filter(pl.col("src") == s)["id"]
                    m_keep = kept.filter(pl.col("src") == s)["id"]
                    lf = lf.filter(pl.col("id").is_in(m_keep) | (~pl.col("id").is_in(m_all) & _keep("id", frac)))
                srcs[s] = lf.collect()
            out["gt_s1"], out["pairs"] = gt_s1, kept
            gt_s1.write_parquet(gt_files[0])
            kept.write_parquet(gt_files[1])
        else:
            for s in C.SOURCES:
                lf = _scan_source(split, s)
                if frac < 1:
                    lf = lf.filter(_keep("id", frac))
                srcs[s] = lf.collect()
        for s, df in srcs.items():
            df.write_parquet(files[s])
        out["src"] = srcs
    return out
