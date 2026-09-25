"""Step 7 (v1): decision layer = one-to-one owner rule + per-S1 threshold rule.

one_to_one "hard": each S2/S3 record is kept only for the S1 that scores it highest
(EDA: every S2/S3 matches at most one S1). The margin-gated version is a later ablation.
Threshold rule: empty if max p < t_single, else keep p >= t_abs and p >= t_rel * max p.
Thresholds come from a grid on dev OOF scores; the pick is the best neighbourhood average
(plateau centre), not the single best cell.
"""
import itertools

import numpy as np
import polars as pl

T_SINGLE = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
T_ABS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
T_REL = [0.0, 0.2, 0.4, 0.6]


def f05(t, tp, n):
    """Per-S1 F0.5 = 1.25 tp / (0.25 t + n); 1.0 when nothing is true and nothing predicted."""
    t, tp, n = (np.asarray(a, np.float64) for a in (t, tp, n))
    return np.where(t + n == 0, 1.0, 1.25 * tp / np.maximum(0.25 * t + n, 1e-9))


def one_to_one(P, mode):
    if mode == "none":
        return P
    return P.filter(pl.col("p") == pl.col("p").max().over(["src", "id"]))


def select(P, ts, ta, tr):
    mx = pl.col("p").max().over("s1")
    return P.filter((mx >= ts) & (pl.col("p") >= ta) & (pl.col("p") >= tr * mx))


def decide(P, mode, ts, ta, tr):
    # dropping p < ta before one-to-one never changes an owner that could survive ta
    return select(one_to_one(P.filter(pl.col("p") >= ta), mode), ts, ta, tr)


def per_s1(sel, base):
    """base: s1, cty, t (all queried S1, incl. t = 0). Returns base + n, tp, f."""
    g = sel.group_by("s1").agg(pl.len().alias("n"), pl.col("label").cast(pl.Int32).sum().alias("tp"))
    e = base.join(g, on="s1", how="left").with_columns(pl.col("n").fill_null(0), pl.col("tp").fill_null(0))
    return e.with_columns(pl.Series("f", f05(e["t"], e["tp"], e["n"])))


def grid(P, base, mode):
    P = one_to_one(P.filter(pl.col("p") >= min(T_ABS)), mode)
    rows = []
    for (i, ts), (j, ta), (k, tr) in itertools.product(enumerate(T_SINGLE), enumerate(T_ABS), enumerate(T_REL)):
        if ta > ts:
            continue
        rows.append({"i": i, "j": j, "k": k, "t_single": ts, "t_abs": ta, "t_rel": tr,
                     "f05": float(per_s1(select(P, ts, ta, tr), base)["f"].mean())})
    g = pl.DataFrame(rows)
    o = g.select(pl.col("i").alias("i2"), pl.col("j").alias("j2"), pl.col("k").alias("k2"), pl.col("f05").alias("f2"))
    nb = (g.join(o, how="cross")
          .filter(((pl.col("i") - pl.col("i2")).abs() <= 1) & ((pl.col("j") - pl.col("j2")).abs() <= 1)
                  & ((pl.col("k") - pl.col("k2")).abs() <= 1))
          .group_by("i", "j", "k").agg(pl.col("f2").mean().alias("plateau")))
    return g.join(nb, on=["i", "j", "k"]).drop("i", "j", "k").sort("plateau", descending=True)
