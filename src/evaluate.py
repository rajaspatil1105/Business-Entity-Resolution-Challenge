"""Metrics, reports and data checks."""
import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz import utils as rfu

from src import config as C
from src import data as D

POSTAL = r"\b\d{5,6}\b"
POSTAL_END = r"\b\d{5,6}\s*$"


def _hdr(t):
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}", flush=True)


def _sample(df, n):
    return df.sample(n=min(n, df.height), seed=C.SEED) if df.height else df


def data_check(frac=1.0):
    pl.Config.set_tbl_rows(40)
    pl.Config.set_tbl_cols(20)
    pl.Config.set_fmt_str_lengths(90)
    pl.Config.set_tbl_width_chars(220)

    tr, te = D.load_split("train", frac), D.load_split("test", frac)

    _hdr("1. ROWS AND COUNTRY LABELS (repr shows hidden spaces/case)")
    for d in (tr, te):
        for s, df in d["src"].items():
            vc = df.group_by("country").len().sort("len", descending=True)
            labels = ", ".join(f"{c!r}:{n:,}" for c, n in vc.iter_rows())
            print(f"{d['split']} S{s}: {df.height:,} rows | {labels}")

    _hdr("2. TEXT QUALITY PER SOURCE/COUNTRY (% and mean lengths)")
    for d in (tr, te):
        for s, df in d["src"].items():
            e = df.group_by("country").agg(
                (pl.col("name").str.strip_chars() == "").mean().mul(100).round(3).alias("name_empty%"),
                (pl.col("addr").str.strip_chars() == "").mean().mul(100).round(3).alias("addr_empty%"),
                pl.col("name").str.len_chars().mean().round(1).alias("name_len"),
                pl.col("addr").str.len_chars().mean().round(1).alias("addr_len"),
                pl.col("addr").str.contains(POSTAL).mean().mul(100).round(1).alias("postal%"),
                pl.col("addr").str.contains(POSTAL_END).mean().mul(100).round(1).alias("postal_end%"),
            ).sort("country")
            print(f"\n{d['split']} S{s}")
            print(e)

    _hdr("3. GROUND TRUTH: JOIN SANITY + CROSS-COUNTRY PAIRS")
    s1 = tr["src"][1].select(
        pl.col("id").alias("s1"), pl.col("name").alias("name1"),
        pl.col("addr").alias("addr1"), pl.col("country").alias("c1"),
    )
    oth = pl.concat([tr["src"][2], tr["src"][3]])
    j = tr["pairs"].join(s1, on="s1", how="left").join(oth, on=["src", "id"], how="left")
    print(f"pairs: {j.height:,} | S1 not found: {j['c1'].null_count():,} | S2/S3 not found: {j['country'].null_count():,}")
    j = j.drop_nulls(["c1", "country"])
    cross = j.filter(pl.col("c1") != pl.col("country"))
    print(f"cross-country true pairs: {cross.height:,}")
    if cross.height:
        print(cross.head(10))

    _hdr("4. MATCHES PER S1 BY SOURCE")
    cnt = (
        tr["gt_s1"]
        .join(
            tr["pairs"].group_by("s1").agg(
                (pl.col("src") == 2).sum().alias("n2"), (pl.col("src") == 3).sum().alias("n3")
            ),
            on="s1", how="left",
        )
        .fill_null(0)
        .with_columns((pl.col("n2") + pl.col("n3")).alias("n"))
    )
    for c in ("n2", "n3", "n"):
        q = cnt[c]
        print(f"{c}: mean {q.mean():.2f} | p99 {q.quantile(0.99)} | max {q.max()} | dist {dict(cnt.group_by(c).len().sort(c).iter_rows())}")
    per_c = cnt.join(s1.select("s1", "c1"), on="s1").group_by("c1").agg(
        (pl.col("n") == 0).mean().mul(100).round(2).alias("singleton%"),
        pl.col("n2").mean().round(2), pl.col("n3").mean().round(2), pl.col("n").mean().round(2),
    )
    print(per_c)

    _hdr("5. SIMILARITY: TRUE PAIRS vs SHUFFLED SAME-COUNTRY PAIRS (token_set_ratio)")
    smp = _sample(j, 20_000)
    parts = []
    for _, g in smp.group_by("c1"):
        sh = g.select(pl.col("name").alias("name_r"), pl.col("addr").alias("addr_r")).sample(
            fraction=1.0, shuffle=True, seed=C.SEED
        )
        parts.append(pl.concat([g, sh], how="horizontal"))
    smp = pl.concat(parts)
    tests = {
        "name true    ": ("name1", "name"), "name shuffled": ("name1", "name_r"),
        "addr true    ": ("addr1", "addr"), "addr shuffled": ("addr1", "addr_r"),
    }
    print("percentiles:   p1    p5    p25   p50   p75")
    for label, (a, b) in tests.items():
        v = process.cpdist(smp[a].to_list(), smp[b].to_list(), scorer=fuzz.token_set_ratio,
                           processor=rfu.default_process, workers=-1)
        print(label, np.percentile(v, [1, 5, 25, 50, 75]).round(1))

    _hdr("6. EXAMPLE CLUSTERS (S1 then its matches)")
    for c in sorted(j["c1"].unique().to_list()):
        ids = j.filter(pl.col("c1") == c)["s1"].unique()
        for sid in ids.sample(n=min(4, len(ids)), seed=C.SEED).to_list():
            g = j.filter(pl.col("s1") == sid).sort("src")
            print(f"\n[{c}] S1 | {g['name1'][0]} | {g['addr1'][0]}")
            for r in g.iter_rows(named=True):
                print(f"      S{r['src']} | {r['name']} | {r['addr']}")

    _hdr("7. SINGLETON S1 EXAMPLES + DECOY (UNMATCHED) S2/S3 EXAMPLES")
    single = cnt.filter(pl.col("n") == 0).select("s1").join(s1, on="s1")
    print(_sample(single, 6).select("name1", "addr1", "c1"))
    decoy = oth.join(tr["pairs"].select("src", "id"), on=["src", "id"], how="anti")
    print(f"decoy share S2: {decoy.filter(pl.col('src') == 2).height / tr['src'][2].height:.3f} | "
          f"S3: {decoy.filter(pl.col('src') == 3).height / tr['src'][3].height:.3f}")
    print(_sample(decoy, 8).select("src", "name", "addr", "country"))

    _hdr("8. TEST RECORDS FROM COUNTRIES NOT IN TRAIN")
    train_c = tr["src"][1]["country"].unique().to_list()
    for s, df in te["src"].items():
        new = df.filter(~pl.col("country").is_in(train_c))
        print(f"\ntest S{s}: {new.height:,} rows from unseen countries")
        print(_sample(new, 8).select("name", "addr", "country"))
