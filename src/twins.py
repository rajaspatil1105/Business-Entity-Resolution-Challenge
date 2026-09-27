"""Twin rule: S2/S3 records with the same name+address tokens always share one owner and are
never orphans (train check: precision 1.0). Pool their scores, give the whole group to one S1.

  python -m src.twins eval                        # train OOF: gain vs Stage B
  python -m src.twins predict [--mode max|nor] [--v 1.0]   # test: rewrite both TSVs
"""
import argparse
import json

import polars as pl

from src import config as C
from src import data as D
from src import stageb as SB


def norm(e):
    return e.fill_null("").str.to_lowercase().str.replace_all(r"[^\p{L}\p{N}]+", " ").str.strip_chars()


def keys(split):
    f = SB.OUT / f"twins_{split}.parquet"
    if f.exists():
        return pl.read_parquet(f)
    parts = []
    for s in (2, 3):
        df = pl.read_csv(C.source_path(split, s), **D.READ_OPTS).select(
            pl.lit(s, pl.Int8).alias("src"),
            pl.col("entity_id").str.slice(3).cast(pl.Int64).alias("id"),
            norm(pl.col("business_name")).alias("n"),
            norm(pl.col("business_address")).alias("a"),
            pl.col("country").fill_null("").alias("cty"))
        df = df.filter((pl.col("n") != "") & (pl.col("a") != ""))
        parts.append(df.select("src", "id", (
            pl.col("cty") + "|" + pl.col("n").str.split(" ").list.sort().list.join(" ") + "|"
            + pl.col("a").str.split(" ").list.sort().list.join(" ")).hash().alias("g")))
        del df
    K = pl.concat(parts)
    K = K.filter(pl.len().over("g") >= 2)
    SB.OUT.mkdir(parents=True, exist_ok=True)
    K.write_parquet(f)
    print(f"twins {split}: {K.height:,} records in {K['g'].n_unique():,} groups")
    return K


def apply(O, K, mode):
    X = O.select("s1", "src", "id", "p", "pb").join(K, on=["src", "id"], how="left")
    solo = X.filter(pl.col("g").is_null()).drop("g")
    grp = X.filter(pl.col("g").is_not_null())
    if mode == "max":
        agg = pl.col("pb").max()
    else:
        agg = 1 - (1 - pl.col("pb").cast(pl.Float64).clip(0, 0.999999)).log().sum().exp()
    G = grp.group_by("g", "s1").agg(agg.cast(pl.Float32).alias("gp"), pl.col("p").max().alias("gpa"))
    best = G.filter(pl.col("gp") == pl.col("gp").max().over("g")).unique("g", keep="any")
    out = best.join(K, on="g").select(
        pl.col("s1").cast(pl.Int64), pl.col("src").cast(pl.Int8), pl.col("id").cast(pl.Int64),
        pl.col("gpa").cast(pl.Float32).alias("p"), pl.col("gp").cast(pl.Float32).alias("pb"))
    solo = solo.with_columns(pl.col("p").cast(pl.Float32), pl.col("pb").cast(pl.Float32))
    return pl.concat([solo, out])


def cmd_eval():
    O = pl.read_parquet(SB.OUT / f"oof_v{SB.VERSION}.parquet")
    U, gt = SB.truth()
    K = keys("train")
    rows = []
    for mode in ("none", "max", "nor"):
        X = O if mode == "none" else apply(O, K, mode)
        for v in (0.75, 1.0, 1.25, 1.5, 2.0):
            E = SB.f05(SB.rule(X, "pb", "ef", v).select(SB.KEYS), U, gt)
            rows.append(dict(mode=mode, v=v, dev=E.filter(pl.col("fold") >= 0)["f"].mean(),
                             holdout=E.filter(pl.col("fold") < 0)["f"].mean()))
    T = pl.DataFrame(rows).sort("dev", descending=True)
    print(T)
    b = T.row(0, named=True)
    base = T.filter(pl.col("mode") == "none")["dev"].max()
    print(f"\nBEST mode={b['mode']} v={b['v']} dev {b['dev']:.4f} holdout {b['holdout']:.4f} | "
          f"Stage B alone {base:.4f} | gain {b['dev'] - base:+.4f}")
    (SB.OUT / "twins_best.json").write_text(json.dumps(b))


def cmd_predict(mode, v):
    f = SB.OUT / "twins_best.json"
    b = json.loads(f.read_text()) if f.exists() else {"mode": "max", "v": 1.0}
    mode, v = mode or b["mode"], v if v is not None else b["v"]
    print(f"twin rule mode={mode} v={v}")
    O = pl.read_parquet(SB.OUT / f"test_scores_v{SB.VERSION}.parquet")
    X = O if mode == "none" else apply(O, keys("test"), mode)
    SB.write(X, SB.rule(X, "pb", "ef", v), None)


def main():
    a = argparse.ArgumentParser()
    a.add_argument("cmd", choices=["eval", "predict"])
    a.add_argument("--mode", choices=["none", "max", "nor"], default=None)
    a.add_argument("--v", type=float, default=None)
    x = a.parse_args()
    cmd_eval() if x.cmd == "eval" else cmd_predict(x.mode, x.v)


if __name__ == "__main__":
    main()
