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


# ============================================================ Step 1 report
def _pct(e, name):
    return e.cast(pl.Float64).mean().mul(100).round(2).alias(name)


def _top_tokens(s, n=40, skip_digits=True):
    t = s.str.split(" ").explode()
    t = t.filter(t.str.len_chars() > 0)
    if skip_digits:
        t = t.filter(~t.str.contains(r"^\d+$"))
    return ", ".join(f"{a}({b})" for a, b in t.value_counts(sort=True).head(n).iter_rows())


def norm_report(frac=1.0):
    from src import normalize as N
    pl.Config.set_tbl_rows(40)
    pl.Config.set_tbl_cols(20)
    pl.Config.set_fmt_str_lengths(90)
    pl.Config.set_tbl_width_chars(220)
    norm = {sp: N.normalize_split(sp, frac) for sp in ("train", "test")}
    raw = {sp: D.load_split(sp, frac) for sp in ("train", "test")}

    _hdr("N1. FLAGS PER SOURCE/COUNTRY (%)")
    for sp in norm:
        for s, df in norm[sp].items():
            print(f"\n{sp} S{s}")
            print(df.group_by("cty").agg(
                _pct(pl.col("name_nonlatin"), "name_nonlat"),
                _pct(pl.col("addr_nonlatin"), "addr_nonlat"),
                _pct(pl.col("addr_state") != "", "state"),
                _pct(pl.col("addr_empty"), "addr_empty"),
                _pct(pl.col("addr_nums").list.len() > 0, "has_num"),
                _pct(pl.col("has_landmark"), "landmark"),
                _pct(pl.col("is_handle"), "handle"),
                _pct(pl.col("name_phone"), "phone"),
                _pct(pl.col("name_alias").is_not_null(), "alias"),
                _pct(pl.col("legal") != "", "legal"),
            ).sort("cty"))

    _hdr("N2. EXAMPLES: raw => normalized")
    for sp in norm:
        for s, df in norm[sp].items():
            j = df.join(raw[sp]["src"][s].select("src", "id", "name", "addr"), on=["src", "id"])
            for c in sorted(j["cty"].unique().to_list()):
                print(f"\n--- {sp} S{s} [{c}]")
                for r in _sample(j.filter(pl.col("cty") == c), 3).iter_rows(named=True):
                    print(f"  {r['name']}  =>  core='{r['name_core']}' legal='{r['legal']}'")
                    print(f"  {r['addr']}  =>  '{r['addr_norm']}' | state={r['addr_state']} | nums={r['addr_nums']}")

    _hdr("N3. MOST FREQUENT TOKENS (look for missed junk / abbreviations)")
    for sp in norm:
        allr = pl.concat(list(norm[sp].values()))
        for c in sorted(allr["cty"].unique().to_list()):
            g = allr.filter(pl.col("cty") == c)
            print(f"\n[{sp} {c}] name_core: {_top_tokens(g['name_core'])}")
            print(f"[{sp} {c}] addr_norm: {_top_tokens(g['addr_norm'])}")
            print(f"[{sp} {c}] state    : {_top_tokens(g['addr_state'], 20, False)}")

    _hdr("N4. TRUE PAIRS: RAW vs NORMALIZED (train)")
    tr, nt = raw["train"], norm["train"]
    ncols = ["name_sq", "name_core", "addr_norm", "addr_state", "addr_nums", "addr_empty", "name_nonlatin", "is_handle"]

    def full(s):
        return nt[s].join(tr["src"][s].select("src", "id", "name", "addr"), on=["src", "id"])

    s1 = full(1).select(pl.col("id").alias("s1"), *[pl.col(c).alias(c + "_1") for c in ["name", "addr", *ncols]])
    oth = pl.concat([full(2), full(3)]).select("src", "id", "name", "addr", *ncols)
    j = _sample(tr["pairs"].join(s1, on="s1").join(oth, on=["src", "id"]), 30_000)

    def sim(a, b):
        return process.cpdist(j[a].to_list(), j[b].to_list(), scorer=fuzz.token_set_ratio,
                              processor=rfu.default_process, workers=-1)

    print("percentiles:   p1    p5    p10   p25   p50")
    res = {}
    for label, a, b in (("name raw ", "name_1", "name"), ("name core", "name_core_1", "name_core"), ("name sq  ", "name_sq_1", "name_sq"),
                        ("addr raw ", "addr_1", "addr"), ("addr norm", "addr_norm_1", "addr_norm")):
        res[label] = sim(a, b)
        print(label, np.percentile(res[label], [1, 5, 10, 25, 50]).round(1))

    nl = j["name_nonlatin"].to_numpy()
    if nl.any():
        print("name core, native-script S2/S3 only:", np.percentile(res["name core"][nl], [5, 25, 50]).round(1), f"(n={nl.sum()})")
    b = j.filter((pl.col("addr_nums_1").list.len() > 0) & (pl.col("addr_nums").list.len() > 0))
    sh = b.select((pl.col("addr_nums_1").list.set_intersection(pl.col("addr_nums")).list.len() > 0)
                  .cast(pl.Float64).mean()).item()
    b2 = j.filter((pl.col("addr_state_1") != "") & (pl.col("addr_state") != ""))
    ss = b2.select((pl.col("addr_state_1").str.split(" ").list.set_intersection(pl.col("addr_state").str.split(" ")).list.len() > 0).cast(pl.Float64).mean()).item()
    print(f"\nboth have numbers: {b.height / j.height:.1%} | of those share >=1 number: {sh:.1%}")
    print(f"both have state  : {b2.height / j.height:.1%} | of those same state: {ss:.1%}")
    print(f"S2/S3 side: empty addr {j['addr_empty'].mean():.1%} | non-latin name "
          f"{j['name_nonlatin'].mean():.1%} | handle {j['is_handle'].mean():.1%}")
    print("\nstate mismatches (check gazetteer):")
    print(b2.filter(pl.col("addr_state_1").str.split(" ").list.set_intersection(pl.col("addr_state").str.split(" ")).list.len() == 0).head(8).select("addr_1", "addr"))

    ns, asim = res["name core"], res["addr norm"]
    low = ns < 50
    print(f"\nname_core sim < 50: {low.mean():.1%} of true pairs | of those addr_norm sim >= 80: "
          f"{(low & (asim >= 80)).sum() / max(low.sum(), 1):.1%}")
    print(j.with_columns(pl.Series("ns", ns)).filter(pl.col("ns") < 50).head(12)
          .select("name_1", "name", "addr_1", "addr"))
