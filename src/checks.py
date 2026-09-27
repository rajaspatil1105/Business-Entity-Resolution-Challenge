"""Quick data checks for Stage B v2 (train only, labels used only to measure).

  python -m src.checks

Check 1: are S1 "siblings" (same core name with a different legal form, or an extra
         family word, or the same street) much more often singletons?
Check 3: do exact duplicate S2/S3 records always share the same owner?
"""
import polars as pl

from src import config as C
from src import data as D

pl.Config(tbl_rows=60, tbl_cols=20, tbl_width_chars=200)

LEGAL = ["inc", "incorporated", "llc", "pllc", "ltd", "limited", "pvt", "private", "corp",
         "corporation", "co", "company", "llp", "lp", "plc", "sarl", "sas", "sasu", "eurl",
         "sa", "sci", "gmbh", "ag", "bv", "nv", "ets"]
LMAP = {"incorporated": "inc", "limited": "ltd", "private": "pvt", "corporation": "corp",
        "company": "co"}
FAMILY = ["group", "groupe", "holdings", "holding", "partners", "enterprises", "enterprise",
          "industries", "overseas", "ventures", "exports", "infratech", "public",
          "international", "intl", "global", "solutions", "services", "associates",
          "participations", "developpement", "distribution"]


def norm(e):
    return e.fill_null("").str.to_lowercase().str.replace_all(r"[^\p{L}\p{N}]+", " ").str.strip_chars()


def rate(df, flag, title):
    t = (df.group_by("cty", flag)
         .agg(pl.len().alias("s1s"),
              (pl.col("m") == 0).mean().round(4).alias("singleton"),
              (pl.col("m") == 1).mean().round(4).alias("one_match"),
              pl.col("m").mean().round(2).alias("mean_m"))
         .sort("cty", flag))
    print(f"\n== {title} ==")
    print(t)
    return t


def check1(pairs):
    s1 = pl.read_csv(C.source_path("train", 1), **D.READ_OPTS).select(
        pl.col("entity_id").str.slice(3).cast(pl.Int64).alias("s1"),
        norm(pl.col("business_name")).alias("n"),
        norm(pl.col("business_address")).alias("a"),
        pl.col("country").fill_null("").alias("cty"))
    m = pairs.group_by("s1").agg(pl.len().alias("m"))
    s1 = s1.join(m, on="s1", how="left").with_columns(pl.col("m").fill_null(0))

    tok = pl.col("n").str.split(" ")
    s1 = s1.with_columns(
        tok.list.eval(pl.element().filter(~pl.element().is_in(LEGAL))).list.join(" ").alias("core"),
        tok.list.eval(pl.element().filter(pl.element().is_in(LEGAL)).replace(LMAP))
           .list.unique().list.sort().list.join(" ").alias("lg"),
        tok.list.eval(pl.element().filter(~pl.element().is_in(LEGAL + FAMILY))).list.join(" ").alias("base"),
        tok.list.eval(pl.element().filter(pl.element().is_in(FAMILY))).list.len().alias("nfam"),
        pl.col("a").str.replace_all(r"\b\d+\w*\b", " ").str.replace_all(r"\s+", " ")
          .str.strip_chars().alias("street"),
        pl.col("a").str.extract(r"\b(\d+)\b", 1).alias("hn"))

    s1 = s1.with_columns(
        pl.when(pl.col("core") != "").then(pl.len().over(["cty", "core"])).otherwise(1).alias("n_core"),
        pl.when(pl.col("core") != "").then(pl.col("lg").n_unique().over(["cty", "core"])).otherwise(1).alias("n_lg"),
        pl.when(pl.col("street") != "").then(pl.len().over(["cty", "street"])).otherwise(1).alias("n_street"),
        pl.when(pl.col("street") != "").then(pl.col("core").n_unique().over(["cty", "street"])).otherwise(1).alias("n_street_names"))

    plain = s1.filter((pl.col("nfam") == 0) & (pl.col("base") != "")).select(
        "cty", pl.col("base").alias("b")).unique().with_columns(pl.lit(True).alias("has_plain"))
    fam = s1.filter(pl.col("nfam") > 0).select("cty", pl.col("base").alias("b")).unique() \
        .with_columns(pl.lit(True).alias("has_fam"))
    s1 = (s1.join(plain, left_on=["cty", "base"], right_on=["cty", "b"], how="left")
            .join(fam, left_on=["cty", "base"], right_on=["cty", "b"], how="left")
            .with_columns(pl.col("has_plain").fill_null(False), pl.col("has_fam").fill_null(False)))

    s1 = s1.with_columns(
        ((pl.col("n_core") > 1) & (pl.col("n_lg") > 1)).alias("legal_sib"),
        (pl.col("core") == "").alias("core_empty"),
        pl.when((pl.col("nfam") > 0) & pl.col("has_plain")).then(pl.lit("1 I have EXTRA family word"))
          .when((pl.col("nfam") == 0) & pl.col("has_fam")).then(pl.lit("2 I am the PLAIN base"))
          .otherwise(pl.lit("3 none")).alias("fam_sib"),
        ((pl.col("n_street") > 1) & (pl.col("n_street_names") > 1)).alias("street_sib"),
        pl.when(pl.col("n_core") == 1).then(pl.lit("1"))
          .when(pl.col("n_core") == 2).then(pl.lit("2"))
          .when(pl.col("n_core") <= 5).then(pl.lit("3-5")).otherwise(pl.lit("6+")).alias("core_group"))
    s1 = s1.with_columns(
        (pl.col("legal_sib") | (pl.col("fam_sib") != "3 none") | pl.col("street_sib")).alias("any_sib"),
        pl.when(~pl.col("legal_sib")).then(pl.lit("0 no legal sibling"))
          .when(pl.col("lg") == "").then(pl.lit("1 sibling, I have NO legal"))
          .otherwise(pl.lit("2 sibling, I HAVE legal")).alias("legal_side"))

    base = (s1["m"] == 0).mean()
    print(f"\nCHECK 1 base singleton rate: {base:.4f}")
    rate(s1, "core_group", "S1s sharing the same core name")
    rate(s1, "legal_sib", "legal sibling (same core, different legal form)")
    rate(s1, "legal_side", "legal sibling: which side am I")
    rate(s1, "fam_sib", "family-word sibling")
    rate(s1, "street_sib", "street sibling (same street, different name)")
    t = rate(s1, "any_sib", "ANY sibling")

    best = t.filter(pl.col("any_sib"))["singleton"].max()
    print(f"\nVERDICT 1: singleton rate with a sibling {best} vs base {base:.4f} ->",
          "BIG signal, build the sibling features" if best >= 2 * base else "weak, skip")


def check3(pairs):
    def load(s):
        return pl.read_csv(C.source_path("train", s), **D.READ_OPTS).select(
            pl.lit(s, pl.Int8).alias("src"),
            pl.col("entity_id").str.slice(3).cast(pl.Int64).alias("id"),
            norm(pl.col("business_name")).alias("n"),
            norm(pl.col("business_address")).alias("a"),
            pl.col("country").fill_null("").alias("cty"))

    O = pl.concat([load(2), load(3)]).join(pairs, on=["src", "id"], how="left")
    print(f"\nCHECK 3 train S2/S3 rows {O.height:,} | orphan share (no owner) {O['s1'].is_null().mean():.4f}")
    O = O.with_columns(
        (pl.col("n") + "|" + pl.col("a")).alias("k_exact"),
        (pl.col("n").str.split(" ").list.sort().list.join(" ") + "|" +
         pl.col("a").str.split(" ").list.sort().list.join(" ")).alias("k_sorted"),
        pl.col("n").alias("k_name"))

    def dup(key, filt, title):
        G = (O.filter(filt).group_by("cty", key)
             .agg(pl.len().alias("sz"),
                  pl.col("s1").null_count().alias("norph"),
                  pl.col("s1").drop_nulls().n_unique().alias("nown"),
                  pl.col("src").n_unique().alias("nsrc"))
             .filter(pl.col("sz") >= 2))
        t = (G.group_by("cty").agg(
                pl.len().alias("groups"),
                pl.col("sz").sum().alias("rows"),
                (pl.col("nsrc") == 2).mean().round(4).alias("cross_src"),
                ((pl.col("nown") == 1) & (pl.col("norph") == 0)).mean().round(4).alias("all_same_owner"),
                ((pl.col("nown") >= 1) & (pl.col("norph") > 0)).mean().round(4).alias("owner+orphan_mix"),
                (pl.col("nown") == 0).mean().round(4).alias("all_orphan"),
                (pl.col("nown") > 1).mean().round(4).alias("diff_owners"),
                (((pl.col("nown") == 1) & (pl.col("norph") == 0)).sum()
                 / (pl.col("nown") >= 1).sum()).round(4).alias("RULE_PRECISION"))
             .sort("cty"))
        print(f"\n== duplicates: {title} ==")
        print(t)
        return t

    t = dup("k_exact", (pl.col("n") != "") & (pl.col("a") != ""), "exact name + address")
    dup("k_sorted", (pl.col("n") != "") & (pl.col("a") != ""), "same tokens, any order")
    dup("k_name", (pl.col("n") != "") & (pl.col("a") == ""), "same name, both addresses empty")

    p = t["RULE_PRECISION"].min()
    print(f"\nVERDICT 3: 'twins follow each other' precision {p} ->",
          "BIG, add the twin rule" if p >= 0.95 else "not safe as a hard rule, use as a feature only")


def main():
    _, pairs = D.load_gt()
    pairs = pairs.select(pl.col("s1").cast(pl.Int64), pl.col("src").cast(pl.Int8), pl.col("id").cast(pl.Int64))
    check1(pairs)
    check3(pairs)


if __name__ == "__main__":
    main()
