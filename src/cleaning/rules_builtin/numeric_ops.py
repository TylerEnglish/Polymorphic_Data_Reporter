from __future__ import annotations
from typing import Optional
import math
import re
import unicodedata
import pandas as pd


__all__ = ["clip_range", "enforce_sign", "zero_as_missing"]


# ---------------------------------------------------------------------------
# Helpers: robust numeric coercion
# ---------------------------------------------------------------------------

# Match a numeric token with optional sign and exponent (after cleanup)
_NUM_TOKEN_RE = re.compile(r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\Z")

# Match any numeric-looking token anywhere (fallback extractor)
_NUM_FALLBACK_RE = re.compile(r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")

_CURRENCY_CHARS = set("$€£¥₩₹₽₫₪₴₦₱")  # common currency symbols
# Note: we also strip commas, underscores, spaces, NBSPs (via normalization/regex)
_PAREN_NUM_RE = re.compile(
    r"^\s*\(\s*[+-−]?\s*\d+(?:[,\s\u00A0_]*\d{3})*(?:\.\d+)?\s*%?\s*\)\s*$"
)
def _is_paren_number(val: object) -> bool:
    if pd.isna(val):
        return False
    txt = _nfkc(str(val))
    return bool(_PAREN_NUM_RE.match(txt))

def _nfkc(s: str) -> str:
    return unicodedata.normalize("NFKC", s)


def _clean_numeric_text(txt: str) -> tuple[str, bool, bool]:
    """
    Normalize a raw string into a clean numeric-ish token string.

    Returns (clean_text, is_percent, paren_negative)

    - Converts Unicode minus (U+2212) to ASCII '-'
    - Removes thousands separators: commas, spaces, NBSPs, underscores
    - Removes currency symbols
    - Detects surrounding parentheses "(...)" → paren_negative=True
    - Detects trailing % anywhere → is_percent=True (single % removed)
    """
    t = _nfkc(txt).strip()
    if not t:
        return "", False, False

    # Parentheses-as-negative: allow outermost (...) only
    paren_negative = False
    if len(t) >= 2 and t[0] == "(" and t[-1] == ")":
        paren_negative = True
        t = t[1:-1].strip()

    # Replace Unicode minus with ASCII hyphen-minus
    t = t.replace("\u2212", "-")

    # Remove currency symbols
    t = "".join(ch for ch in t if ch not in _CURRENCY_CHARS)

    # Handle percent
    is_percent = False
    if "%" in t:
        is_percent = True
        t = t.replace("%", "")

    # Remove thousands-like separators (commas, spaces, NBSPs, underscores)
    # Keep dots for decimals and signs
    t = re.sub(r"[,\s\u00A0_]+", "", t)

    return t, is_percent, paren_negative


def _coerce_numeric_robust(s: pd.Series) -> pd.Series:
    """
    Coerce a Series to numeric (float64/Float64), robust to:
      - currency symbols ($€£…)
      - thousands separators (, _ space)
      - Unicode minus (−)
      - parentheses for negative values: (123) -> -123
      - percent values: 5% -> 0.05 ; (10%) -> -0.10
      - exponents (1.2e3)
      - common text "inf", "-inf", "nan" (case-insensitive)
    """
    # Fast path for already-numeric dtypes
    if pd.api.types.is_numeric_dtype(s.dtype):
        return pd.to_numeric(s, errors="coerce")

    x = s.astype("string")

    def _parse_one(val: object) -> Optional[float]:
        if pd.isna(val):
            return math.nan
        raw = str(val)
        t = raw.strip()
        if not t:
            return math.nan

        t = _nfkc(t)

        # Common text tokens
        low = t.lower()
        if low in {"nan", "na", "null"}:
            return math.nan
        if low in {"inf", "+inf", "infinity", "+infinity"}:
            return math.inf
        if low in {"-inf", "-infinity"}:
            return -math.inf

        cleaned, is_percent, paren_neg = _clean_numeric_text(t)

        if not cleaned:
            return math.nan

        # If whole string matches numeric token, great
        if _NUM_TOKEN_RE.fullmatch(cleaned):
            num_txt = cleaned
        else:
            # Fallback: extract the first numeric-looking token
            m = _NUM_FALLBACK_RE.search(cleaned)
            if not m:
                return math.nan
            num_txt = m.group(0)

        try:
            valf = float(num_txt)
        except Exception:
            return math.nan

        if paren_neg:
            valf = -valf
        if is_percent:
            valf = valf / 100.0
        return valf

    # Map and return float dtype (nullable via NaN)
    out = x.map(_parse_one)
    return pd.to_numeric(out, errors="coerce")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def clip_range(
    s: pd.Series,
    *,
    lo: float | None = None,
    hi: float | None = None,
) -> pd.Series:
    """
    Clip numeric series to [lo, hi] (inclusive). Non-numeric coerced robustly.

    Supports strings with:
      - currency/thousands (e.g., "$1,234.50")
      - unicode minus (e.g., "−5")
      - parentheses negative (e.g., "(100)")
      - percents (e.g., "12.5%")
      - exponents ("1.2e3")
    """
    x = _coerce_numeric_robust(s)

    if lo is not None:
        lo = float(lo)
        x = x.mask(x < lo, lo)
    if hi is not None:
        hi = float(hi)
        x = x.mask(x > hi, hi)
    # keep plain float dtype (NaN for missing)
    return x.astype(float)


def enforce_sign(
    s: pd.Series,
    *,
    sign: str = "nonnegative",  # "positive" | "nonnegative" | "negative"
) -> pd.Series:
    """
    Enforce sign constraints after robust coercion.

    - "positive": keep x > 0; set x <= 0 → NA.
      Accounting negatives like "(5)" are treated as +5.0 first.
    - "nonnegative": negatives become 0; accounting negatives "(5)" become +5.0.
    - "negative": keep true negatives (<0). Accounting negatives "(5)" → NA.
    """
    # Robust numeric parse (handles currency, parentheses as -x, unicode minus, percents, etc.)
    x = _coerce_numeric_robust(s)

    # Detect accounting-style negatives from the original values
    try:
        orig_str = s.astype("string")
    except Exception:
        orig_str = s.astype(object).astype("string")
    paren_mask = orig_str.map(_is_paren_number)

    # Start from parsed values and adjust accounting negatives
    res = x.copy()
    # For positive/nonnegative we take absolute value for accounting negatives
    sig = (sign or "nonnegative").lower()
    if sig in {"positive", "nonnegative"}:
        mask_abs = paren_mask & res.notna()
        if mask_abs.any():
            res.loc[mask_abs] = res.loc[mask_abs].abs()

    if sig == "positive":
        res = res.mask(~(res > 0), pd.NA)

    elif sig == "nonnegative":
        res = res.mask(res < 0, 0)

    elif sig == "negative":
        # Accounting negatives should not count as "true" negatives here
        if paren_mask.any():
            res = res.mask(paren_mask, pd.NA)
        res = res.mask(~(res < 0), pd.NA)

    # Unknown policy → no-op
    return res.astype(float)


def zero_as_missing(
    s: pd.Series,
    *,
    eps: float = 0.0,   # treat |x| <= eps as zero
) -> pd.Series:
    """
    Convert (near-)zero values to missing.

    eps = 0.0 → exactly zero only.
    eps > 0  → anything with |x| <= eps becomes NA.
    """
    x = _coerce_numeric_robust(s).astype(float)
    if eps and eps > 0:
        mask = x.abs() <= float(eps)
    else:
        mask = x == 0.0
    x.loc[mask] = pd.NA
    return x.astype(float)
