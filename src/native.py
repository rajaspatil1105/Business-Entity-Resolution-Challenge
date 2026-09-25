"""Native-script fixes: (2) native state names -> English, (3) learned native->English name-word map."""
import json, re, sys, unicodedata
from collections import Counter, defaultdict
import polars as pl

# ---------- Fix 2: native-script state names ----------
_STATES = {
    "Maharashtra": ["महाराष्ट्र"], "Delhi": ["नई दिल्ली", "दिल्ली"], "Haryana": ["हरियाणा"],
    "Uttar Pradesh": ["उत्तर प्रदेश"], "Uttarakhand": ["उत्तराखंड", "उत्तराखण्ड"], "Bihar": ["बिहार"],
    "Jharkhand": ["झारखंड", "झारखण्ड"], "Rajasthan": ["राजस्थान"],
    "Madhya Pradesh": ["मध्य प्रदेश", "मध्यप्रदेश"], "Chhattisgarh": ["छत्तीसगढ़"],
    "Himachal Pradesh": ["हिमाचल प्रदेश"], "Chandigarh": ["चंडीगढ़", "ਚੰਡੀਗੜ੍ਹ"], "Goa": ["गोवा"],
    "Gujarat": ["ગુજરાત", "गुजरात"], "Punjab": ["ਪੰਜਾਬ", "पंजाब"],
    "West Bengal": ["পশ্চিমবঙ্গ", "পশ্চিম বঙ্গ", "पश्चिम बंगाल"], "Assam": ["অসম", "আসাম"],
    "Tamil Nadu": ["தமிழ்நாடு", "தமிழ் நாடு", "तमिलनाडु"], "Karnataka": ["ಕರ್ನಾಟಕ", "कर्नाटक"],
    "Kerala": ["കേരളം", "കേരള", "केरल"], "Telangana": ["తెలంగాణ", "तेलंगाना"],
    "Andhra Pradesh": ["ఆంధ్రప్రదేశ్", "ఆంధ్ర ప్రదేశ్", "आंध्र प्रदेश"], "Odisha": ["ଓଡ଼ିଶା", "ଓଡିଶା", "ओडिशा"],
}
STATE_MAP = {}
for en, natives in _STATES.items():
    for n in natives:
        for form in ("NFC", "NFD"):
            STATE_MAP[unicodedata.normalize(form, n)] = en
_KEYS = sorted(STATE_MAP, key=len, reverse=True)
_VALS = [STATE_MAP[k] for k in _KEYS]


def fix_states(col: str) -> pl.Expr:
    return pl.col(col).str.replace_many(_KEYS, _VALS)


# ---------- Fix 3: learned native -> English name words ----------
_NONLAT = r"[^\x00-\x7F]"
_NL = re.compile(_NONLAT)
_SPLIT = re.compile(r"[\s,.\-()\[\]/&]+")


def _toks(s):
    return [t for t in _SPLIT.split(s.lower()) if t]


def learn_word_map(native, latin, min_count=5, min_purity=0.6):
    co = defaultdict(Counter)
    for a, b in zip(native, latin):
        if not a or not b or _NL.search(b):
            continue
        ta, tb = _toks(a), _toks(b)
        if len(ta) != len(tb):
            continue
        for x, y in zip(ta, tb):
            if _NL.search(x) and not _NL.search(y):
                co[x][y] += 1
    out = {}
    for x, c in co.items():
        y, n = c.most_common(1)[0]
        if n >= min_count and n / sum(c.values()) >= min_purity:
            out[x] = y
    return out


def apply_word_map(s: pl.Series, wmap: dict) -> pl.Series:
    """Rewrite native tokens; each unique native string is processed once."""
    if not wmap:
        return s
    rep = {}
    for u in s.filter(s.str.contains(_NONLAT)).drop_nulls().unique():
        t = _toks(u)
        if any(w in wmap for w in t):
            rep[u] = " ".join(wmap.get(w, w) for w in t)
    return s.replace(rep) if rep else s


def map_path():
    from src import config as C
    return C.WORK_DIR / "word_map.json"


def load_map():
    p = map_path()
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _scan(path):
    lf = pl.scan_csv(path, separator="\t", quote_char=None, infer_schema=False)
    cols = lf.collect_schema().names()
    idc = next(c for c in cols if "id" in c.lower())
    nmc = next(c for c in cols if "name" in c.lower())
    return lf.select(pl.col(idc).alias("id"), pl.col(nmc).alias("name"))


def build():
    from src import config as C
    d = C.DATA_DIR / "train"
    gt = (pl.scan_csv(d / "train_ground_truth.tsv", separator="\t", quote_char=None, infer_schema=False)
          .select(pl.col("source1_entity_id").alias("s1"),
                  pl.col("matched_entity_ids").str.split(",").alias("id"))
          .explode("id").with_columns(pl.col("id").str.strip_chars()))
    s1 = _scan(d / "train_source1.tsv").rename({"id": "s1", "name": "name1"})
    s23 = pl.concat([_scan(d / "train_source2.tsv"), _scan(d / "train_source3.tsv")]) \
            .filter(pl.col("name").str.contains(_NONLAT))
    pairs = gt.join(s23, on="id").join(s1, on="s1").select("name", "name1").collect()
    wmap = learn_word_map(pairs["name"].to_list(), pairs["name1"].to_list())
    p = map_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(wmap, ensure_ascii=False), encoding="utf-8")
    print(f"native pairs {len(pairs):,} | word map size {len(wmap):,} -> {p}")
    print(list(wmap.items())[:25])


if __name__ == "__main__" and sys.argv[1:] == ["learn"]:
    build()
