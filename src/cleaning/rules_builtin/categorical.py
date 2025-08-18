from __future__ import annotations
from typing import Iterable, Sequence, Dict, Tuple, List, Set
import re
import unicodedata
import pandas as pd
from pandas.api.types import CategoricalDtype

__all__ = [
    "consolidate_rare_categories",
    "map_category_synonyms",
    "extract_category_anchor",
    "fuzzy_map_categories",
    "extract_category_key",
]

# =============================================================================
# Utilities
# =============================================================================

# A small, conservative stopword list for labels (won't strip brand names).
_DEFAULT_STOPWORDS: Set[str] = {
    "inc", "llc", "l.l.c", "corp", "co", "company", "ltd",
    "restaurant", "restaurants", "store", "stores",
    "the", "group", "intl", "international",
    "sa", "ag", "gmbh", "bv", "plc", "pte",
    "llp", "l.l.p", "pc", "p.c.",
    "and", "&",
    "rd", "st", "ave", "blvd", "dr", "hwy", "ctr", "ct", "pl", "ln",
    "suite", "ste", "unit",
    "us", "usa",
}

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+", flags=re.UNICODE)

def _nfkc(s: str) -> str:
    return unicodedata.normalize("NFKC", s)

def _casefold(s: str) -> str:
    return s.casefold()

def _tokenize_label(text: str, *, allow_digits: bool = True) -> List[str]:
    """
    Unicode-normalize, then extract alnum tokens; optionally drop pure-digit tokens.
    """
    s = _nfkc(text)
    toks = _TOKEN_RE.findall(s)
    if not allow_digits:
        toks = [t for t in toks if not t.isdigit()]
    return toks

def _clean_token(t: str) -> str:
    # lower/casefold and strip trivial trailing/leading dots/hyphens/underscores
    tt = _casefold(_nfkc(t)).strip("._- ")
    return tt

def _prepare_stopwords(stopwords: Iterable[str] | None) -> Set[str]:
    if not stopwords:
        return set(_DEFAULT_STOPWORDS)
    return {_clean_token(w) for w in (set(stopwords) | _DEFAULT_STOPWORDS)}

# -----------------------------------------------------------------------------
# Levenshtein distance (no external deps)
# -----------------------------------------------------------------------------
def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            ins = cur[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            cur.append(min(ins, dele, sub))
        prev = cur
    return prev[-1]

def _best_fuzzy_match(
    value: str,
    candidates: Sequence[str],
    *,
    max_distance: int = 2,
    max_rel_distance: float = 0.25,
) -> Tuple[str | None, int]:
    """
    Return (best_match, distance) if within thresholds; else (None, large).
    Relative distance threshold guards long strings.
    """
    if not candidates:
        return None, 10**9
    v = _clean_token(value)
    best = None
    best_d = 10**9
    for c in candidates:
        cc = _clean_token(c)
        d = _levenshtein(v, cc)
        max_rel = max(1, int(round(max(len(v), len(cc)) * max_rel_distance)))
        if d < best_d and d <= max(max_distance, max_rel):
            best, best_d = c, d
    return best, best_d

# =============================================================================
# Core transforms
# =============================================================================

def consolidate_rare_categories(
    s: pd.Series,
    *,
    min_freq: float = 0.01,
    other_label: str = "Other",
) -> pd.Series:
    """
    Collapse categories whose relative frequency < min_freq into `other_label`.
    Works on object/string/categorical; preserves NA as NA.
    If final cardinality is small (<=256), cast to 'category'.
    """
    if not (
        pd.api.types.is_object_dtype(s.dtype)
        or pd.api.types.is_string_dtype(s.dtype)
        or isinstance(s.dtype, CategoricalDtype)
    ):
        return s.copy(deep=True)

    x = s.astype("string")
    vc = x.value_counts(dropna=True, normalize=True)
    rare_idx = vc[vc < float(min_freq)].index
    if len(rare_idx) == 0:
        return s.copy(deep=True)

    out = x.where(~x.isin(rare_idx), other_label)
    try:
        if int(out.nunique(dropna=True)) <= 256:
            return out.astype("category")
    except Exception:
        pass
    return out

def map_category_synonyms(
    s: pd.Series,
    *,
    mapping: Dict[str, str],
    case_insensitive: bool = True,
    strip: bool = True,
) -> pd.Series:
    """
    Normalize label variants via an explicit mapping, e.g. {"us":"USA","u.s.":"USA"}.
    - case_insensitive: if True, uses lowercased keys
    - strip: trims whitespace before mapping
    """
    if not (
        pd.api.types.is_object_dtype(s.dtype)
        or pd.api.types.is_string_dtype(s.dtype)
        or isinstance(s.dtype, CategoricalDtype)
    ):
        return s.copy(deep=True)

    x = s.astype("string")
    if strip:
        x = x.str.strip()

    if case_insensitive:
        lower_map = {k.lower(): v for k, v in mapping.items()}
        out = x.map(lambda v: lower_map.get(v.lower(), v) if isinstance(v, str) else v)
    else:
        out = x.map(lambda v: mapping.get(v, v))
    return out.astype("string")

def extract_category_key(
    s: pd.Series,
    *,
    # regex extraction (optional; if provided, used first)
    patterns: Sequence[str] | None = None,
    capture_group: int | None = None,   # 1-based; if None, use whole match
    default: str | None = None,
    case_insensitive: bool = True,

    # anchor options (used if patterns is None, or to post-process later)
    allow_digits: bool = False,
    min_token_len: int = 3,
    stopwords: Iterable[str] | None = None,
    prefer_global_frequency: bool = True,

    # fuzzy options (only used if candidates provided)
    candidates: Sequence[str] | None = None,
    max_distance: int = 2,
    max_rel_distance: float = 0.25,
) -> pd.Series:
    """
    Canonical category key extractor.

    Modes:
    1) Regex mode (if `patterns` provided):
       - Search each cell with patterns (first match wins).
       - If `capture_group` is given (1-based), return that group's text **exactly as matched**.
       - Else return the whole matched substring **exactly as matched** (original case preserved).
       - If no match: <NA> unless `default` is provided.

    2) Anchor mode (if `patterns` not provided):
       - Use `extract_category_anchor` to pick a single brand-like token per row.

    Optionally, after either mode, pass through fuzzy normalization if `candidates`
    are provided.

    Returns a pandas 'string' dtype series; NA preserved as <NA> unless `default` used.
    """
    if not (
        pd.api.types.is_object_dtype(s.dtype)
        or pd.api.types.is_string_dtype(s.dtype)
        or isinstance(s.dtype, CategoricalDtype)
    ):
        return s.copy(deep=True)

    x = s.astype("string")

    # --------------------------
    # 1) Regex extraction (if any)
    # --------------------------
    out: pd.Series
    if patterns:
        flags = re.UNICODE | (re.IGNORECASE if case_insensitive else 0)
        compiled = [re.compile(p, flags) for p in patterns if p]

        def _extract_one(val: object) -> object:
            if pd.isna(val):
                return pd.NA
            text = str(val)
            for rx in compiled:
                m = rx.search(text)
                if m:
                    if capture_group is not None:
                        # return the captured group exactly as matched (no trimming/case-change)
                        try:
                            grp = m.group(capture_group)
                        except IndexError:
                            continue
                        return grp if grp != "" else (default if default is not None else pd.NA)
                    else:
                        # return the whole match exactly as matched
                        return m.group(0)
            return default if default is not None else pd.NA

        out = x.map(_extract_one).astype("string")

    else:
        # --------------------------
        # 2) Anchor extraction
        # --------------------------
        out = extract_category_anchor(
            x,
            allow_digits=allow_digits,
            min_token_len=min_token_len,
            stopwords=stopwords,
            prefer_global_frequency=prefer_global_frequency,
        ).astype("string")

    # --------------------------
    # Optional fuzzy normalization
    # --------------------------
    if candidates:
        out = fuzzy_map_categories(
            out,
            candidates=candidates,
            case_insensitive=case_insensitive,
            max_distance=max_distance,
            max_rel_distance=max_rel_distance,
        )

    return out.astype("string")

def extract_category_anchor(
    s: pd.Series,
    *,
    allow_digits: bool = False,
    min_token_len: int = 3,
    stopwords: Iterable[str] | None = None,
    prefer_global_frequency: bool = True,
) -> pd.Series:
    """
    Heuristic, regex-free extractor that picks a single 'anchor' token per row:
      1) tokenize each row to alnum tokens
      2) drop stopwords/short tokens
      3) compute column-wise token frequencies
      4) for each row, choose the row-token with:
            - highest global frequency
            - then **closest to the primary (most frequent) token** by Levenshtein
            - then longer length
            - then lexicographic
    This reliably prefers brand-like tokens over locations when frequencies tie.
    """
    if not (
        pd.api.types.is_object_dtype(s.dtype)
        or pd.api.types.is_string_dtype(s.dtype)
        or isinstance(s.dtype, CategoricalDtype)
    ):
        return s.copy(deep=True)

    sw = _prepare_stopwords(stopwords)

    # Tokenize all rows
    rows_tokens: List[List[str]] = []
    for v in s.astype("string"):
        if pd.isna(v):
            rows_tokens.append([])
            continue
        toks = [_clean_token(t) for t in _tokenize_label(v, allow_digits=allow_digits)]
        toks = [t for t in toks if len(t) >= int(min_token_len) and t not in sw]
        rows_tokens.append(toks)

    # Global frequency model
    freq: Dict[str, int] = {}
    for toks in rows_tokens:
        for t in toks:
            freq[t] = freq.get(t, 0) + 1

    # Determine primary anchor across the column (most frequent; tie → longer token)
    primary = None
    if freq:
        primary = max(freq.items(), key=lambda it: (it[1], len(it[0]), it[0]))[0]

    # Choose the best token per row with improved tie-breaking
    anchors: List[object] = []
    for toks in rows_tokens:
        if not toks:
            anchors.append(pd.NA)
            continue

        if not prefer_global_frequency or primary is None:
            # fall back to frequency, then length, then lexicographic
            toks_sorted = sorted(toks, key=lambda t: (-freq.get(t, 0), -len(t), t))
            anchors.append(toks_sorted[0])
            continue

        # sort by: global freq desc, distance to primary asc, length desc, lexicographic
        toks_sorted = sorted(
            toks,
            key=lambda t: (
                -freq.get(t, 0),
                _levenshtein(t, primary),
                -len(t),
                t,
            ),
        )
        anchors.append(toks_sorted[0])

    return pd.Series(anchors, index=s.index, dtype="string")

def fuzzy_map_categories(
    s: pd.Series,
    *,
    candidates: Sequence[str],
    case_insensitive: bool = True,
    max_distance: int = 2,
    max_rel_distance: float = 0.25,
) -> pd.Series:
    """
    Map each value to the nearest candidate using Levenshtein distance, if close enough.
    Values with no acceptable match are left as-is (string).
    Intended to be used on already-extracted anchors.
    """
    if not (
        pd.api.types.is_object_dtype(s.dtype)
        or pd.api.types.is_string_dtype(s.dtype)
        or isinstance(s.dtype, CategoricalDtype)
    ):
        return s.copy(deep=True)

    # Normalize candidate casing once
    if case_insensitive:
        canon_map = {_clean_token(c): c for c in candidates}
        canon_keys = list(canon_map.keys())
    else:
        canon_map = {c: c for c in candidates}
        canon_keys = list(canon_map.keys())

    def _one(v: object) -> object:
        if pd.isna(v):
            return pd.NA
        txt = str(v)
        key = _clean_token(txt) if case_insensitive else txt
        best_key, _ = _best_fuzzy_match(
            key, canon_keys, max_distance=max_distance, max_rel_distance=max_rel_distance
        )
        if best_key is None:
            return txt
        return canon_map[best_key]

    out = s.map(_one)
    return out.astype("string")
