"""Step 1: text normalization.

Every record gets several text versions plus address components.
Rules are hand-written from patterns seen in the training data; nothing external is used.
Variants are mapped to ONE canonical short form (street/str/saint -> st), so both sides of a
pair are transformed identically and ambiguous tokens cannot create false mismatches.
Indic-script text is transliterated to Latin with rule-based indic-transliteration (MIT).
This file is pure ASCII: non-ASCII characters are written as escapes.
"""
import re

import polars as pl
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

from src import config as C
from src import data as D

VERSION = 5          # bump when rules change -> old cache is ignored
CHUNK = 1_000_000

NONLATIN = r"[^\x00-\x7f]"                 # after accent stripping: anything non-ASCII
NONLATIN_LETTER = r"[^\p{Latin}\P{L}]"      # raw text: a letter from a non-Latin script
PUNCT = r"[^\p{L}\p{N}]+"
ACCENTS = "[\u0300-\u036f]+"                # Latin combining accents only (keeps Indic intact)
APOS = "['`\u2018\u2019]"
SPECIAL = {"\u0153": "oe", "\u00e6": "ae", "\u00df": "ss", "\u00f8": "o",
           "\u0142": "l", "\u0111": "d", "\u0131": "i"}
NULL_TOK = ["null", "none", "nan", "nil", "unknown"]
LANDMARK = r"\b(near|nr|opp|opposite|behind|beside|besides|next to|pres de|en face)\b"
ALIAS_RE = r"^(.*?)\s+(?:dba|d b a|aka|a k a|fka|f k a|trading as|formerly)\s+(.+)$"
DOMAIN = r"\.(com|net|org|biz|info|co\.in|in|fr|us|io)\s*$"

PVT_RE = r"\b[pb]h?i?ra?i?[vb]h?e[td]h?\b"   # transliterated 'private' spellings
LTD_RE = r"\blimi[td]h?e?[td]h?\b"           # transliterated 'limited' spellings
INDIC = [("\u0900", "\u097f", sanscript.DEVANAGARI), ("\u0980", "\u09ff", sanscript.BENGALI),
         ("\u0a00", "\u0a7f", sanscript.GURMUKHI), ("\u0a80", "\u0aff", sanscript.GUJARATI),
         ("\u0b00", "\u0b7f", sanscript.ORIYA), ("\u0b80", "\u0bff", sanscript.TAMIL),
         ("\u0c00", "\u0c7f", sanscript.TELUGU), ("\u0c80", "\u0cff", sanscript.KANNADA),
         ("\u0d00", "\u0d7f", sanscript.MALAYALAM)]
_INDIC_RX = [(re.compile(f"[{a}-{b}]"), s) for a, b, s in INDIC]
INDIC_ANY = "[\u0900-\u0d7f]"
# drop the inherent final short 'a' (limiteda -> limited); long vowels are kept
_SCHWA = re.compile("(?<=\\w\\w)(?<=[^aeiou\u0101\u012b\u016b\u1e5b\\W])a\\b")


def _inv(spec):
    """{"canon": "var1 var2"} -> {var: canon}"""
    return {v: k for k, vs in spec.items() for v in vs.split()}


def _gaz(spec, with_codes=True):
    """{"code": "name1|name2"} -> {name: code}"""
    m = {}
    for code, names in spec.items():
        if with_codes:
            m[code] = code
        for nm in names.split("|"):
            m[nm] = code
    return m


# ---------------------------------------------------------------- names
NAME_MAP = _inv({
    "corp": "corporation corpn", "inc": "incorporated", "co": "company cmpny",
    "ltd": "limited limitet limitad limidhedh", "pvt": "private pvte praivet piraivet praibhet",
    "cie": "compagnie", "bros": "brothers",
    "intl": "international internationale", "mfg": "manufacturing",
    "assoc": "associates association associes", "tech": "technologies technology technologie",
    "ctr": "center centre", "st": "saint sainte ste", "mt": "mount", "grp": "group groupe",
    "ent": "enterprises enterprise entreprise entreprises", "svc": "services service",
    "sys": "systems system systemes", "ind": "industries industry industrie",
    "sol": "solutions solution", "soc": "societe society",
    "ets": "etablissements etablissement",
})
LEGAL = ("inc corp co ltd pvt llc lp llp lllp pllc plc pc opc sarl sas sasu sa eurl sci snc "
         "scop scs selarl ei eirl cie gmbh ag bv srl spa pty pte").split()
WEAK = "shri sri shree sree smt the".split()
NAME_STOP = "and of de des du la le les et da".split()
SQ_DROP = LEGAL + WEAK + NAME_STOP + ("limited private corporation incorporated company "
                                      "compagnie praivet piraivet limitet praibhet limidhedh").split()

# -------------------------------------------------------------- address
ADDR_MAP = _inv({
    "rd": "road", "st": "street str saint", "ste": "suite sainte", "av": "avenue ave aven",
    "bd": "boulevard blvd bld boul bvd", "dr": "drive drv", "ct": "court crt", "ln": "lane",
    "pl": "place", "sq": "square", "hwy": "highway", "pkwy": "parkway pky", "cir": "circle",
    "ter": "terrace", "trl": "trail", "apt": "apartment", "fl": "floor flr",
    "bldg": "building", "n": "north", "s": "south", "e": "east", "w": "west",
    "ne": "northeast", "nw": "northwest", "se": "southeast", "sw": "southwest",
    "sec": "sector", "rue": "r", "all": "allee", "imp": "impasse", "chem": "chemin",
    "chaus": "chaussee", "fg": "faubourg fbg", "quai": "qu", "rte": "route",
    "mt": "mount", "ft": "fort", "hts": "heights", "jct": "junction jn", "nagar": "ngr",
})
ADDR_STOP = ("no number num of the de du des la le les d l et and near nr opp opposite "
             "behind beside besides next to tehsil teh dist district po ps cedex via at").split()

_US = _gaz({
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas", "ca": "california",
    "co": "colorado", "ct": "connecticut", "de": "delaware",
    "dc": "district of columbia|washington dc|d c", "fl": "florida", "ga": "georgia",
    "hi": "hawaii", "id": "idaho", "il": "illinois", "in": "indiana", "ia": "iowa",
    "ks": "kansas", "ky": "kentucky", "la": "louisiana", "me": "maine", "md": "maryland",
    "ma": "massachusetts", "mi": "michigan", "mn": "minnesota", "ms": "mississippi",
    "mo": "missouri", "mt": "montana", "ne": "nebraska", "nv": "nevada",
    "nh": "new hampshire", "nj": "new jersey", "nm": "new mexico", "ny": "new york",
    "nc": "north carolina", "nd": "north dakota", "oh": "ohio", "ok": "oklahoma",
    "or": "oregon", "pa": "pennsylvania", "ri": "rhode island", "sc": "south carolina",
    "sd": "south dakota", "tn": "tennessee", "tx": "texas", "ut": "utah", "vt": "vermont",
    "va": "virginia", "wa": "washington", "wv": "west virginia", "wi": "wisconsin",
    "wy": "wyoming", "pr": "puerto rico",
})
_IN = _gaz({  # telangana is merged into ap: the data labels Telangana addresses as AP
    "ap": "andhra pradesh|telangana|tg|ts", "ar": "arunachal pradesh", "as": "assam",
    "br": "bihar", "cg": "chhattisgarh|chattisgarh|ct", "ga": "goa", "gj": "gujarat|gujrat",
    "hr": "haryana", "hp": "himachal pradesh", "jh": "jharkhand", "ka": "karnataka",
    "kl": "kerala", "mp": "madhya pradesh", "mh": "maharashtra|maharastra", "mn": "manipur",
    "ml": "meghalaya", "mz": "mizoram", "nl": "nagaland", "od": "odisha|orissa|or",
    "pb": "punjab", "rj": "rajasthan", "sk": "sikkim", "tn": "tamil nadu|tamilnadu",
    "tr": "tripura", "up": "uttar pradesh",
    "uk": "uttarakhand|uttaranchal|ut", "wb": "west bengal",
    "dl": "delhi|nct of delhi|nct delhi", "jk": "jammu and kashmir", "la": "ladakh",
    "py": "puducherry|pondicherry", "ch": "chandigarh",
    "an": "andaman and nicobar|andaman and nicobar islands",
    "dn": "dadra and nagar haveli|daman and diu|dadra and nagar haveli and daman and diu",
    "ld": "lakshadweep",
})
_FR = _gaz({  # region + its departments -> region code
    "ara": "auvergne rhone alpes|ain|allier|ardeche|cantal|drome|isere|loire|haute loire|"
           "puy de dome|rhone|savoie|haute savoie",
    "bfc": "bourgogne franche comte|cote d or|doubs|jura|nievre|haute saone|saone et loire|"
           "yonne|territoire de belfort",
    "bre": "bretagne|cotes d armor|finistere|ille et vilaine|morbihan",
    "cvl": "centre val de loire|centre|cher|eure et loir|indre|indre et loire|loir et cher|loiret",
    "cor": "corse|corse du sud|haute corse",
    "ges": "grand est|alsace|lorraine|ardennes|aube|marne|haute marne|meurthe et moselle|"
           "meuse|moselle|bas rhin|haut rhin|vosges",
    "hdf": "hauts de france|aisne|nord|oise|pas de calais|somme",
    "idf": "ile de france|idf|paris|seine et marne|yvelines|essonne|hauts de seine|"
           "seine saint denis|val de marne|val d oise",
    "nor": "normandie|calvados|eure|manche|orne|seine maritime",
    "naq": "nouvelle aquitaine|charente|charente maritime|correze|creuse|dordogne|gironde|"
           "landes|lot et garonne|pyrenees atlantiques|deux sevres|vienne|haute vienne",
    "occ": "occitanie|ariege|aude|aveyron|gard|haute garonne|gers|herault|lot|lozere|"
           "hautes pyrenees|pyrenees orientales|tarn|tarn et garonne",
    "pdl": "pays de la loire|loire atlantique|maine et loire|mayenne|sarthe|vendee",
    "pac": "provence alpes cote d azur|paca|region sud|alpes de haute provence|hautes alpes|"
           "alpes maritimes|bouches du rhone|var|vaucluse",
}, with_codes=False)
# Gazetteers are looked up by country label; unknown countries simply get no state mapping.
STATE_MAP = {"us": _US, "usa": _US, "united states": _US, "india": _IN,
             "france": _FR, "fr": _FR}

OUT_COLS = ["id", "src", "cty", "name_norm", "name_exp", "name_core", "name_sq", "name_alias",
            "legal", "name_nonlatin", "is_handle", "name_phone", "addr_norm", "addr_state",
            "addr_nums", "addr_num1", "addr_nonlatin", "addr_empty", "has_landmark"]


# ------------------------------------------------------- transliteration
def _translit_one(t):
    try:
        for rx, scheme in _INDIC_RX:
            if rx.search(t):
                t = transliterate(t, scheme, sanscript.IAST)
        return re.sub("pr[a\u0101]\\s*\\.?\\s*li\\b\\.?", "pvt ltd", _SCHWA.sub("", t))
    except Exception:
        return t


def _translit(s, latin_free_only=False):
    """Transliterate Indic-script strings; each unique string is converted once."""
    mask = s.str.contains(INDIC_ANY)
    if latin_free_only:
        mask = mask & ~s.str.contains("[A-Za-z]")
    if not mask.any():
        return s
    uniq = s.filter(mask).unique().to_list()
    return s.replace({u: _translit_one(u) for u in uniq})


# ---------------------------------------------------------------- helpers
def _base(e):
    """lowercase, N-degree -> no, strip Latin accents and apostrophes, n/a m/s, & + -> and"""
    return (e.fill_null("").str.to_lowercase()
            .str.replace_all("n\\s*[\u00b0\u00ba]\\s*", " no ")
            .str.replace_many(list(SPECIAL), list(SPECIAL.values()))
            .str.normalize("NFKD").str.replace_all(ACCENTS, "")
            .str.replace_all(APOS, "")
            .str.replace_all(r"\bn/a\b|\bm/s\b", " ")
            .str.replace_all(r"[&+]", " and "))


def _toks(e):
    return e.str.replace_all(r"\s+", " ").str.strip_chars().str.split(" ")


def _drop(lst, words):
    return lst.list.eval(pl.element().filter((pl.element() != "") & ~pl.element().is_in(words)))


# ---------------------------------------------------------------- core
def _normalize_country(df, cty):
    smap = {k: "~" + v for k, v in STATE_MAP.get(cty, {}).items()}
    n = _base(pl.col("name"))
    a = (_base(pl.col("addr"))
         .str.replace_all(r"\b[cswd]\s*/\s*o\b", " ")
         .str.replace_all(r"\bh\s*\.?\s*no\b", " no "))

    df = df.with_columns(
        (pl.col("name").str.strip_chars().str.contains(r"^[@#]")
         | n.str.contains(DOMAIN) | n.str.contains(r"^\s*www\b")).alias("is_handle"),
        n.str.contains(r"\d{7,}").alias("name_phone"),
        n.str.replace_all(r"\d{7,}", " ").str.replace_all(r"^\s*www\s*\.", " ")
         .str.replace_all(DOMAIN, " ").str.replace_all(r"\.", "").str.replace_all(PUNCT, " ")
         .str.replace_all(r"([a-z])0([a-z])", "${1}o${2}")
         .str.replace_all(r"([a-z])1([a-z])", "${1}l${2}")
         .str.replace_all(r"\b5([a-z]{3,})", "s${1}")
         .str.replace_all(r"\b0([a-z]{2,})", "o${1}")
         .str.replace_all(r"\s+", " ").str.strip_chars().alias("_n"),
        a.str.contains(LANDMARK).alias("has_landmark"),
        a.str.split(",").list.eval(
            pl.element().str.replace_all(PUNCT, " ").str.replace_all(r"\s+", " ").str.strip_chars()
        ).alias("_segs"),
    )
    df = df.with_columns(
        pl.when(pl.col("name_nonlatin"))
          .then(pl.col("_n").str.replace_all(PVT_RE, "pvt").str.replace_all(LTD_RE, "ltd"))
          .otherwise(pl.col("_n")).alias("_n"))
    if smap:
        df = df.with_columns(pl.col("_segs").list.eval(pl.element().replace(smap)))

    df = df.with_columns(
        _drop(_toks(pl.col("_n")), NULL_TOK).alias("_nt"),
        pl.col("_segs").list.eval(pl.element().filter(pl.element().str.starts_with("~")))
          .list.eval(pl.element().str.slice(1)).list.unique().list.sort().list.join(" ").alias("addr_state"),
        pl.col("_segs").list.eval(pl.element().filter(
            ~pl.element().str.starts_with("~") & ~pl.element().str.contains(NONLATIN)
            & (pl.element() != ""))).list.join(" ").alias("_a"),
        pl.col("_n").str.extract(ALIAS_RE, 2).alias("name_alias"),
    )
    df = df.with_columns(
        pl.col("_nt").list.unique(maintain_order=True).list.join(" ").alias("name_norm"),
        pl.col("_nt").list.eval(pl.element().replace(NAME_MAP))
          .list.unique(maintain_order=True).alias("_ne"),
        _drop(pl.col("_nt"), SQ_DROP).list.join("").alias("_sq"),
        _drop(_toks(pl.col("_a")).list.eval(
            pl.element().str.replace(r"^0+(\d)", "${1}").replace(ADDR_MAP)), ADDR_STOP + NULL_TOK)
          .list.unique(maintain_order=True).alias("_at"),
    )
    df = df.with_columns(
        pl.col("_ne").list.join(" ").alias("name_exp"),
        _drop(pl.col("_ne"), LEGAL + WEAK + NAME_STOP).list.join(" ").alias("_core"),
        pl.col("_ne").list.eval(pl.element().filter(pl.element().is_in(LEGAL)))
          .list.sort().list.join(" ").alias("legal"),
        pl.col("_at").list.join(" ").alias("addr_norm"),
        pl.col("_at").list.eval(pl.element().str.extract(r"^(\d+)", 1))
          .list.drop_nulls().list.unique(maintain_order=True).alias("addr_nums"),
    )
    df = df.with_columns(
        pl.when(pl.col("_core") == "").then(pl.col("name_exp")).otherwise(pl.col("_core"))
          .alias("name_core"),
        pl.when(pl.col("_sq") == "").then(pl.col("name_norm").str.replace_all(" ", ""))
          .otherwise(pl.col("_sq")).alias("name_sq"),
        (pl.col("addr_norm") == "").alias("addr_empty"),
        pl.col("addr_nums").list.first().alias("addr_num1"),
    )
    return df.select(OUT_COLS)


def normalize_frame(df):
    df = df.with_columns(
        pl.col("country").str.strip_chars().str.to_lowercase().alias("cty"),
        pl.col("name").str.contains(NONLATIN_LETTER).alias("name_nonlatin"),
        pl.col("addr").str.contains(NONLATIN_LETTER).alias("addr_nonlatin"),
    )
    # native-script fixes (after flags so name_nonlatin/addr_nonlatin keep their meaning)
    from src.native import fix_states, apply_word_map, load_map
    df = df.with_columns(fix_states("addr"))
    df = df.with_columns(apply_word_map(df["name"], load_map()).alias("name"))
    df = df.with_columns(_translit(df["name"]).alias("name"),
                         _translit(df["addr"], latin_free_only=True).alias("addr"))
    return pl.concat([_normalize_country(g, g["cty"][0]) for g in df.partition_by("cty")])


def normalize_split(split, frac=1.0):
    """Returns {source: normalized frame}; cached as parquet per source."""
    tag = f"{split}_f{frac:g}_v{VERSION}"
    d = C.WORK_DIR / "norm"
    d.mkdir(parents=True, exist_ok=True)
    files = {s: d / f"{tag}_s{s}.parquet" for s in C.SOURCES}
    if all(f.exists() for f in files.values()):
        return {s: pl.read_parquet(f) for s, f in files.items()}

    raw = D.load_split(split, frac)["src"]
    out = {}
    for s, df in raw.items():
        if files[s].exists():
            out[s] = pl.read_parquet(files[s])
            continue
        with D.step(f"normalize {tag} S{s}"):
            res = pl.concat([normalize_frame(df.slice(i, CHUNK)) for i in range(0, df.height, CHUNK)])
            res.write_parquet(files[s])
        out[s] = res
    return out
