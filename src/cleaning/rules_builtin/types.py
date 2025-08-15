from __future__ import annotations
from typing import Any, Iterable
import re
import pandas as pd
import numpy as np

_CURRENCY_RE = re.compile(r'^\s*[$€£¥₩₹]')
_PERCENT_SYM_RE = re.compile(r'\s*%$')
_PERCENT_WORD_RE = re.compile(r'\b(percent|pct)\b', re.I)
_PARENS_NEG_RE = re.compile(r'^\((.*)\)$')
_UNICODE_MINUS_RE = re.compile(r'^[\u2212\u2010-\u2015]\s*')  # − ‐–— etc. at start

_K_SUFFIX_RE = re.compile(r'(?i)\s*(k|thousand)\s*$')
_M_SUFFIX_RE = re.compile(r'(?i)\s*(m|mm|million)\s*$')

_THOUSAND_SEP_RE = re.compile(r'[,_\s]')  # remove commas/underscores/spaces inside number

def _to_float(val: Any) -> float | np.floating | np.nan:
    """
    Best-effort numeric parse with support for:
      - currency prefix: $ € £ ¥ ₩ ₹
      - percent suffix/synonyms: %, percent, pct (Divide by 100)
      - parentheses for negatives: "(123)" -> -123
      - unicode minus at start
      - magnitude suffixes: k/thousand, m/mm/million
      - thousand separators: ',', '_', spaces
    Returns np.nan if it can't parse.
    """
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return np.nan
    if isinstance(val, (int, float, np.integer, np.floating)):
        # pass numeric through
        return float(val)

    s = str(val).strip()
    if s == "":
        return np.nan

    neg = False
    # Parentheses negative
    m = _PARENS_NEG_RE.match(s)
    if m:
        s = m.group(1).strip()
        neg = True

    # Unicode minus at start (normalize to sign)
    if _UNICODE_MINUS_RE.match(s):
        s = _UNICODE_MINUS_RE.sub("", s, count=1)
        neg = True

    # ASCII sign at start
    if s.startswith("-"):
        neg = True
        s = s[1:].strip()
    elif s.startswith("+"):
        s = s[1:].strip()

    # Currency at start
    if _CURRENCY_RE.match(s):
        s = _CURRENCY_RE.sub("", s, count=1).strip()

    # Percent?
    had_percent = False
    if _PERCENT_SYM_RE.search(s) or _PERCENT_WORD_RE.search(s):
        had_percent = True
        # strip trailing % and any percent words
        s = _PERCENT_SYM_RE.sub("", s).strip()
        s = _PERCENT_WORD_RE.sub("", s).strip()

    # Magnitude suffix?
    mult = 1.0
    if _K_SUFFIX_RE.search(s):
        mult = 1_000.0
        s = _K_SUFFIX_RE.sub("", s).strip()
    elif _M_SUFFIX_RE.search(s):
        mult = 1_000_000.0
        s = _M_SUFFIX_RE.sub("", s).strip()

    # Remove thousand separators and spaces inside
    s = _THOUSAND_SEP_RE.sub("", s)

    # Final numeric coercion
    try:
        x = float(s)
    except Exception:
        try:
            # last-ditch via pandas
            x = pd.to_numeric(s, errors="coerce")
            if pd.isna(x):
                return np.nan
            x = float(x)
        except Exception:
            return np.nan

    if neg:
        x = -x
    if had_percent:
        x = x / 100.0
    x = x * mult
    return x

def coerce_numeric_from_string(s: pd.Series, unit_hint: str | None = None) -> pd.Series:
    """
    Convert strings like "$1,234", "12k", "(3.5)", "15%", "2 million" into numeric.
    If a '%' (or percent/pct word) is present, divide by 100. `unit_hint` is advisory;
    we don't force division without an explicit percent signal to avoid accidental scaling.
    Pure: does not mutate the input series.
    """
    # Work on object view to keep apply predictable
    out = s.copy(deep=True)
    # Fast path: if already numeric dtype, just return float
    if pd.api.types.is_numeric_dtype(out.dtype):
        return pd.to_numeric(out, errors="coerce").astype(float)

    # Map elementwise
    coerced = out.map(_to_float)
    # Ensure float dtype
    return pd.to_numeric(coerced, errors="coerce").astype(float)

def parse_datetime_from_string(s: pd.Series, formats: list[str]) -> pd.Series:
    """
    Try a list of explicit formats first (strict), then fall back to pandas'
    general parser. Pure: no mutation.
    """
    # Already datetime? just coerce to datetime64[ns] (idempotent)
    if pd.api.types.is_datetime64_any_dtype(s.dtype):
        return pd.to_datetime(s, errors="coerce")

    fmts = list(formats or [])

    def _parse_one(x: Any):
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return pd.NaT
        # try explicit formats first
        for f in fmts:
            try:
                return pd.to_datetime(x, format=f, errors="raise")
            except Exception:
                continue
        # fallback: general parser
        try:
            return pd.to_datetime(x, errors="coerce")
        except Exception:
            return pd.NaT

    out = s.map(_parse_one)
    # ensure dtype is datetime64[ns]
    return pd.to_datetime(out, errors="coerce")

def cast_category_if_small(s: pd.Series, max_card: int) -> pd.Series:
    """
    If cardinality <= max_card, cast to pandas 'category'. Otherwise return a new
    Series object with the original dtype (purity).
    """
    try:
        nunique = int(s.nunique(dropna=True))
    except TypeError:
        nunique = int(s.astype(str).nunique(dropna=True))
    if nunique <= int(max_card):
        return s.astype("category")
    # return a new object even if dtype unchanged
    return s.copy(deep=False)

def cast_string_dtype(s: pd.Series) -> pd.Series:
    """
    Ensure text-like columns use Pandas' nullable string dtype.
    No-op for non-text dtypes.
    """
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        return s.astype("string")
    return s
