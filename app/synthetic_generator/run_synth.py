
"""
synthetic_string_generator.py
─────────────────────────────
Header-less CSV ➜ Synthetic CSV using real-word replacements,
while **leaving common identifier/column names unchanged**.

Run demo   : python synthetic_string_generator.py
Process CSV: python synthetic_string_generator.py --input raw.csv
"""

from __future__ import annotations
import argparse, hashlib, random, re, sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np, pandas as pd
from app.synthetic_generator.light_weight_synth import SimpleTabularSynth

# ════════════════════════════════════════════════════════════════════════════
# 0 – Identifiers to keep AS-IS  (case-insensitive)
# ════════════════════════════════════════════════════════════════════════════
IDENTIFIER_WORDS = {
    # personal / contact
    "name", "first_name", "last_name", "full_name", "middle_name",
    "age", "dob", "date_of_birth", "gender", "sex",
    "phone", "phone_number", "contact_number",
    "email", "email_address",

    # addresses / geo
    "country", "state", "province", "county", "region", "city", "town",
    "village", "location", "location_id", "location_code", "location_name",
    "address", "street", "street_address", "house_number", "zip", "zipcode",
    "postal_code", "postcode", "lat", "latitude", "lon", "lng", "longitude",

    # identifiers / keys
    "id", "identifier", "uid", "guid", "slug",
    "user_id", "customer_id", "client_id", "employee_id",
    "order_id", "invoice_id", "product_id", "item_id",
    "sku", "code", "ref", "reference",

    # dates / times
    "date", "time", "timestamp", "created_at", "updated_at",
    "year", "month", "day", "week", "quarter",

    # money / numbers
    "cost", "price", "amount", "value", "total", "subtotal",
    "quantity", "qty", "count", "score", "grade", "rating",

    # categorical / misc
    "category", "type", "status", "flag", "active",
    "is_active", "enabled", "disabled", "true", "false",
    "currency", "language", "url", "link", "website",
    "company", "organization", "department", "description",
    "comment", "comments", "note", "notes", "title", "label", "tag",
}

# ════════════════════════════════════════════════════════════════════════════
# (Rest of the script unchanged except IDENTIFIER_WORDS expanded)
# ════════════════════════════════════════════════════════════════════════════
_RNG = random.Random(42)

# wordfreq for noun check + common word pool
try:
    from wordfreq import zipf_frequency as _zipf, top_n_list
    def _likely_noun(tok: str) -> bool:
        return tok.isalpha() and _zipf(tok.lower(), "en") >= 3
    COMMON_WORDS = [w for w in top_n_list("en", 50000) if w.isalpha()]
    _noun_strategy = "wordfreq"
except ImportError:
    COMMON_WORDS = [
        "apple","anchor","artist","basic","beacon","candle","circle",
        "dragon","driver","eagle","engine","filter","garden","globe",
        "honey","island","jungle","kernel","laptop","market","memory",
        "number","office","planet","queen","rocket","signal","ticket",
        "unicorn","valley","window","xenon","yonder","zephyr",
    ]
    _noun_strategy = "wordnet_or_none"

# WordNet bootstrap
def _load_wordnet():
    try:
        from nltk.corpus import wordnet as wn
        wn.synsets("dog"); return wn
    except LookupError:
        try:
            import nltk; nltk.download("wordnet", quiet=True)
            from nltk.corpus import wordnet as wn
            wn.synsets("dog"); print("[INFO] WordNet downloaded", file=sys.stderr)
            return wn
        except Exception as exc:
            print("[WARN] WordNet unavailable:", exc, file=sys.stderr); return None
    except Exception as exc:
        print("[WARN] WordNet load failed:", exc, file=sys.stderr); return None

wn = _load_wordnet()
if _noun_strategy == "wordnet_or_none":
    def _likely_noun(tok: str) -> bool:  # type: ignore
        return bool(wn) and tok.isalpha() and bool(wn.synsets(tok.lower(), pos=wn.NOUN))

# caches
_SYNONYM_CACHE: Dict[str,str] = {}
_USED_WORDS: set[str] = set()
_TOKEN_MAP: Dict[str,str] = {}

def _wordnet_synonym(noun: str) -> Optional[str]:
    if wn is None: return None
    if noun in _SYNONYM_CACHE: return _SYNONYM_CACHE[noun]
    lemmas={l.name().replace("_"," ") for s in wn.synsets(noun.lower(),pos=wn.NOUN)
            for l in s.lemmas() if l.name().isalpha()}
    for cand in sorted(lemmas, key=len):
        if cand.lower()!=noun.lower() and cand not in _USED_WORDS:
            _SYNONYM_CACHE[noun]=cand; _USED_WORDS.add(cand); return cand
    return None

def _other_word(src: str) -> str:
    """Pick a deterministic English word ≈ same length, unused so far."""
    length = len(src)

    # 1) exact-length pool
    pool = [w for w in COMMON_WORDS if len(w) == length and w not in _USED_WORDS]

    # 2) widen ±1, ±2 … until we find something
    if not pool:
        for delta in range(1, 10):
            pool = [
                w for w in COMMON_WORDS
                if len(w) in (length - delta, length + delta) and w not in _USED_WORDS
            ]
            if pool:
                break

    # 3) **new guard** – if still empty, allow reuse
    if not pool:                                  # ← add this block
        pool = [w for w in COMMON_WORDS if len(w)]  # reuse is okay

    # deterministic pick
    idx = int(hashlib.sha1(src.encode()).hexdigest(), 16) % len(pool)
    word = pool[idx]
    _USED_WORDS.add(word)
    return word

def _replace_token(tok:str|float)->str|float:
    if pd.isna(tok): return np.nan
    s=str(tok)
    if s in _TOKEN_MAP: return _TOKEN_MAP[s]
    if s.lower() in IDENTIFIER_WORDS or s.isdigit():
        _TOKEN_MAP[s]=s; return s
    if _likely_noun(s):
        syn=_wordnet_synonym(s)
        if syn: _TOKEN_MAP[s]=syn; return syn
    word=_other_word(s); _TOKEN_MAP[s]=word; return word

# def _replace_cell(cell:str|float)->str|float:
#     if pd.isna(cell): return np.nan
#     return " ".join(_replace_token(t) for t in re.findall(r"\S+", str(cell)))
_TOKEN_SPLIT_RE = re.compile(r"(\w+|\s+|[^\w\s]+)")

# ─── REPLACE the old _replace_cell with this version ───────────────────────
def _replace_cell(cell: str | float) -> str | float:
    """
    Replace every *word* token while **preserving all whitespace and
    non-word separators** exactly as they appear in the original string.
    Examples preserved intact:
        "Consumer | edga |simon"
        "abc-123 / xyz"
        "foo,bar;baz"
    """
    if pd.isna(cell):
        return np.nan

    out_parts: list[str] = []
    for part in _TOKEN_SPLIT_RE.findall(str(cell)):
        # keep whitespace & pure punctuation untouched
        if part.isspace() or not part or not part.isalnum():
            out_parts.append(part)
        else:
            # alphanumeric “word” → noun / identifier logic
            out_parts.append(str(_replace_token(part)))

    return "".join(out_parts)

def overwrite_strings(df:pd.DataFrame)->pd.DataFrame:
    synth=SimpleTabularSynth(seed=42).fit(df)
    df_out=synth.sample(len(df))
    for col in df.select_dtypes(include="object").columns:
        df_out[col]=df[col].apply(_replace_cell)
    return df_out

def load_csv_no_header(path:Path,d=",")->pd.DataFrame:
    return pd.read_csv(path,header=None,sep=d,keep_default_na=True)
def save_csv_no_header(df:pd.DataFrame,path:Path,d=",")->None:
    df.to_csv(path,index=False,header=False,sep=d)

_DEMO_TEXT="""
Name | Hussein | Hakeem Address Number 22 Fioye Crescent Surulere Lagos Age 17 Gender Male
Name Arojoye Samuel Address 11 Omolade Close Omole Estate Lagos Age 16 Gender Male
Name Alex Ezurum Address 1 Adamu Lane, Abuja Age 14 Gender Male
Name Susan Nwaimo Address Number 58 Yaba Street, Kaduna State Age 16 Gender Female
Name Ajao Opeyemi Address No12 Olubunmi Street, Abeokuta Age 18 Gender Female
Name Banjoko Adebusola Address 34 Ngige Street, Ugheli, Delta Age 14 Gender Female
Name Muhammed Olabisi Address 13, ICAN road, Enugu Age 12 Gender Female
Name Oluwagbemi Mojisola Address ACCA Lane, Onitsha Age 13 Gender Female
"""

def main()->None:
    ap=argparse.ArgumentParser(description="Header-less CSV → synthetic CSV")
    ap.add_argument("--input",type=Path); ap.add_argument("--output",type=Path)
    ap.add_argument("--delimiter",default=","); args=ap.parse_args()
    print(f"[INFO] Noun detection: {_noun_strategy} (WordNet {'✔' if wn else '✘'})")
    print(f"[INFO] Unchanged identifiers: {len(IDENTIFIER_WORDS)} words\n")
    if args.input is None:
        df_demo=pd.DataFrame(_DEMO_TEXT.splitlines(),columns=[0])
        print("Original ↓\n",df_demo,"\n")
        df_demo_synth=overwrite_strings(df_demo)
        print("Synthetic ↓\n",df_demo_synth,"\n")
        save_csv_no_header(df_demo_synth,Path("demo_synth.csv"),args.delimiter)
        print("Demo file saved → demo_synth.csv"); return
    out=args.output or args.input.with_name(f"{args.input.stem}_synth.csv")
    df_real=load_csv_no_header(args.input,args.delimiter)
    save_csv_no_header(overwrite_strings(df_real),out,args.delimiter)
    print("Synthetic CSV written →",out.resolve())

if __name__=="__main__":
    main()
