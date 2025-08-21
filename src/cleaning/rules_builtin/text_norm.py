from __future__ import annotations
from typing import Any, Iterable, Optional, Set
import re
import unicodedata
import pandas as pd
import numpy as np

_CONTROL_RE = re.compile(r"[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]")
_WHITESPACE_RE = re.compile(r"\s+")

_FANCY_TRANSLATE = {
    ord("\u2018"): "'",
    ord("\u2019"): "'",
    ord("\u201C"): '"',
    ord("\u201D"): '"',
    ord("\u2013"): "-",
    ord("\u2014"): "-",
    ord("\u2212"): "-",
    ord("\u00A0"): " ",
}

_DEFAULT_NULL_TOKENS: Set[str] = {
    "", "-", "—", "–", "n/a", "na", "none", "null", "nil", "nan",
    "N/A", "NA", "NONE", "NULL", "NIL", "NAN", "NaT",
    "<NA>", "<na>", "<null>", "<none>"
}

def _to_str_or_none(x: Any) -> Optional[str]:
    # ✅ treat any pandas missing as None (handles pd.NA, NaT, numpy NaN cleanly)
    if pd.isna(x):
        return None
    return str(x)

def _normalize_one(
    x: Any,
    *,
    strip: bool = True,
    lower: bool = False,
) -> Any:
    s = _to_str_or_none(x)
    if s is None:
        return None
    s = unicodedata.normalize("NFKC", s)
    s = s.translate(_FANCY_TRANSLATE)
    s = _CONTROL_RE.sub("", s)
    s = _WHITESPACE_RE.sub(" ", s)
    if strip:
        s = s.strip()
    if lower:
        s = s.lower()
    return s

def text_normalize(
    s: pd.Series,
    *,
    strip: bool = True,
    lower: bool = False,
) -> pd.Series:
    # Always return a new object (purity), even for non-strings
    if not (pd.api.types.is_object_dtype(s.dtype) or pd.api.types.is_string_dtype(s.dtype)):
        return s.copy(deep=True)

    # Remember which entries were literally `None` so we can preserve that sentinel
    none_mask = s.map(lambda v: v is None)

    # Do fast, NA-safe vectorized ops using pandas' "string" accessor
    out = s.astype("string")
    out = out.str.normalize("NFKC")
    out = out.str.translate(_FANCY_TRANSLATE)
    out = out.str.replace(_CONTROL_RE, "", regex=True)
    out = out.str.replace(_WHITESPACE_RE, " ", regex=True)
    if strip:
        out = out.str.strip()
    if lower:
        out = out.str.lower()

    # Optional: treat empty-string as missing (keeps your cleaning logic effective)
    out = out.replace("", pd.NA)

    # Convert back to object dtype and restore literal None where it existed
    out = out.astype(object)
    out[none_mask] = None
    return out

def normalize_null_tokens(
    s: pd.Series,
    *,
    null_tokens: Optional[Iterable[str]] = None,
    case_insensitive: bool = True,
    apply_text_normalize_first: bool = True,
) -> pd.Series:
    if not (pd.api.types.is_object_dtype(s.dtype) or pd.api.types.is_string_dtype(s.dtype)):
        return s.copy(deep=True)

    base = set(_DEFAULT_NULL_TOKENS)
    if null_tokens is not None:
        base |= set(null_tokens)

    norm_tokens: Set[str] = set()
    for t in base:
        tt = _normalize_one(t, strip=True, lower=case_insensitive)
        if tt is not None:
            norm_tokens.add(tt)

    x = (
        text_normalize(s, strip=True, lower=case_insensitive)
        if apply_text_normalize_first
        else s.astype("string")
    )

    # NA or equals any normalized token → NA
    mask = x.isna() | x.isin(norm_tokens)
    return x.mask(mask, other=pd.NA)
