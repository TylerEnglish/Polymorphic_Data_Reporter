from __future__ import annotations
from typing import Iterable, Sequence
import re
import unicodedata
import pandas as pd

# ---- Defaults (exact-token matches) -----------------------------------------

_DEFAULT_TRUE = {
    "true", "t", "y", "yes", "1",
    "on", "enabled", "active" 
}
_DEFAULT_FALSE = {
    "false", "f", "n", "no", "0",
    "off", "disabled", "inactive"
}

# ---- Helpers ----------------------------------------------------------------

def _casefold(s: str) -> str:
    return s.casefold()

_NUMLIKE_RE = re.compile(r"[+-]?\d+(?:\.\d+)?\Z")

def _normalize_str(x: object, *, strip_nonword_edges: bool) -> str | None:
    """
    Normalize to a clean string (or None for NA).
    - Unicode NFKC, strip whitespace.
    - If the value looks numeric (e.g., "0", "1", "-1", "0.0"), we **do not**
      strip non-word edges (to preserve sign/decimal).
    - Otherwise, we trim leading/trailing non-word chars so tokens like "(Yes!)" match.
    """
    if pd.isna(x):
        return None
    s = str(x)
    s = unicodedata.normalize("NFKC", s).strip()
    if not s:
        return s  # empty string stays empty

    if strip_nonword_edges:
        # Preserve numeric semantics (don't turn "-1" into "1")
        if not _NUMLIKE_RE.fullmatch(s):
            s = re.sub(r"^\W+|\W+$", "", s, flags=re.UNICODE)

    return s

def _compile_union_regex(patterns: Sequence[str] | None, *, case_insensitive: bool) -> re.Pattern | None:
    """
    Join multiple user patterns into a single anchored regex.
    If a pattern already has ^ or $, we respect it; otherwise we anchor.
    """
    if not patterns:
        return None
    def _anchor(p: str) -> str:
        p = p.strip()
        if p.startswith("^") or p.endswith("$"):
            return p
        return rf"(?:{p})"
    body = "|".join(_anchor(p) for p in patterns if p)
    if not body:
        return None
    if not any(p.strip().startswith("^") or p.strip().endswith("$") for p in patterns):
        body = rf"^(?:{body})$"
    flags = re.UNICODE | (re.IGNORECASE if case_insensitive else 0)
    return re.compile(body, flags)

def _normalize_token_set(tokens: Iterable[str] | None, *, case_insensitive: bool) -> set[str]:
    if not tokens:
        return set()
    return {_casefold(t.strip()) if case_insensitive else t.strip() for t in tokens if isinstance(t, str) and t.strip()}

# ---- Public API --------------------------------------------------------------

def coerce_bool_from_tokens(
    s: pd.Series,
    *,
    true_tokens: set[str] | None = None,
    false_tokens: set[str] | None = None,
    true_regex: list[str] | None = None,
    false_regex: list[str] | None = None,
    numeric_01_only: bool = True,
    allow_numeric_truthy: bool = False,
    case_insensitive: bool = True,
    strip_nonword_edges: bool = True,
    drop_invalid: bool = True,
) -> pd.Series:
    """
    Vectorized coercion of boolean-like columns to:
      - Pandas 'boolean' dtype (nullable) by default, OR
      - A mixed object series (if drop_invalid=False) preserving invalids.

    Recognition sources:
      1) Exact tokens (e.g., "yes", "no", "on", "off") with Unicode-safe casefold.
      2) Regex patterns (full-string match by default; supply ^/$ to customize).
      3) Numeric:
         - by default, only 0/1 (any int/float form or "0"/"1") are recognized.
         - if allow_numeric_truthy=True and numeric_01_only=False:
               nonzero => True, zero => False.

    Parameters are safe to pass from a rule registry via kwargs.
    """

    if pd.api.types.is_numeric_dtype(s.dtype):
        x = pd.to_numeric(s, errors="coerce")
        out = pd.Series(pd.NA, index=s.index, dtype="boolean")
        if numeric_01_only or not allow_numeric_truthy:
            m01 = x.isin([0, 1])
            out.loc[m01] = x.loc[m01].astype(int).astype(bool)
        else:
            nz = x.notna()
            out.loc[nz] = (x.loc[nz] != 0)
        return out

    x = s.astype("string")
    norm = x.map(lambda v: _normalize_str(v, strip_nonword_edges=strip_nonword_edges))
    cf = norm.map(lambda v: (_casefold(v) if (v is not None and case_insensitive) else v))

    tset = _normalize_token_set(true_tokens or _DEFAULT_TRUE, case_insensitive=case_insensitive)
    fset = _normalize_token_set(false_tokens or _DEFAULT_FALSE, case_insensitive=case_insensitive)

    mask_true = cf.isin(tset)
    mask_false = cf.isin(fset)

    # Numeric strings (preserved with sign/decimal by the normalizer)
    is_numlike = cf.fillna("").str.fullmatch(_NUMLIKE_RE, na=False)
    if is_numlike.any():
        as_num = pd.to_numeric(cf.where(is_numlike), errors="coerce")
        if numeric_01_only or not allow_numeric_truthy:
            mask_true |= (as_num == 1)
            mask_false |= (as_num == 0)
        else:
            mask_true |= (as_num.notna() & (as_num != 0))
            mask_false |= (as_num == 0)

    rx_true = _compile_union_regex(true_regex, case_insensitive=case_insensitive)
    rx_false = _compile_union_regex(false_regex, case_insensitive=case_insensitive)
    if rx_true is not None:
        mask_true |= norm.fillna("").str.match(rx_true, na=False)
    if rx_false is not None:
        mask_false |= norm.fillna("").str.match(rx_false, na=False)

    conflict = mask_true & mask_false
    mask_true = mask_true & ~conflict
    mask_false = mask_false & ~conflict

    if drop_invalid:
        out = pd.Series(pd.NA, index=s.index, dtype="boolean")
        out.loc[mask_true] = True
        out.loc[mask_false] = False
        return out

    mixed = s.copy(deep=True).astype(object)
    mixed.loc[mask_true] = True
    mixed.loc[mask_false] = False
    return mixed
