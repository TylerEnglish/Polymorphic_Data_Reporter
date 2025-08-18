from __future__ import annotations
from typing import Iterable, Optional
import re
import unicodedata
import pandas as pd


__all__ = ["zero_pad", "keep_alnum"]


def _to_string_series(s: pd.Series) -> pd.Series:
    """Convert to pandas StringDtype, preserving NA."""
    try:
        return s.astype("string")
    except Exception:
        return s.astype(object).astype("string")


def _nfkc(s: str) -> str:
    # Normalize for consistency; don't aggressively strip accents since some IDs are non-ASCII.
    return unicodedata.normalize("NFKC", s)


def zero_pad(
    s: pd.Series,
    *,
    width: int = 5,
    fillchar: str = "0",
    strip_whitespace: bool = True,
    numeric_only: bool = False,
) -> pd.Series:
    """
    Left-pad values to a minimum width while preserving a leading sign (+/-).

    - Works for string, integer, and mixed object columns; returns StringDtype.
    - `fillchar` must be a single character; if longer, first char is used.
    - If `numeric_only=True`, extract the **rightmost** numeric token and preserve
      a sign only when it truly acts as a sign (start-of-string or preceded by another sign).
      E.g. " INV-007 " -> "007", "A12B" -> "12", "++-3" -> "-3".
    """
    # Sanitize inputs
    try:
        w = max(0, int(width))
    except Exception:
        w = 0

    fc = str(fillchar or "0")
    if len(fc) != 1:
        fc = fc[0]

    x = _to_string_series(s)
    if strip_whitespace:
        x = x.str.replace(r"\s+", "", regex=True)

    if numeric_only:
        # Extract rightmost numeric token; preserve sign only if it's truly a sign
        # (at string start or preceded by another sign).
        num_pat = re.compile(r"[+-]?\d+")

        def _extract_numeric(v: object) -> object:
            if pd.isna(v):
                return pd.NA
            txt = _nfkc(str(v))
            matches = list(num_pat.finditer(txt))
            if not matches:
                return ""  # let padding handle it

            def preservable(m: re.Match) -> bool:
                # sign present and either at start or preceded by another sign
                if txt[m.start()] not in "+-":
                    return False
                if m.start() == 0:
                    return True
                return txt[m.start() - 1] in "+-"

            # Prefer the rightmost preservable sign+digits; otherwise the rightmost digits (dropping any hyphen-like)
            chosen = None
            for mm in reversed(matches):
                if preservable(mm):
                    chosen = mm
                    break
            if chosen is None:
                chosen = matches[-1]
                # If it has a leading '+' or '-' but isn't preservable, drop that sign
                token = chosen.group(0)
                if token and token[0] in "+-":
                    token = token[1:]
                return token
            return chosen.group(0)

        x = x.map(_extract_numeric).astype("string")

    # Custom zfill that respects sign with arbitrary fillchar
    def _pad_one(v: object) -> object:
        if pd.isna(v):
            return pd.NA
        txt = str(v)
        if txt == "":
            return fc * w if w > 0 else ""
        # Preserve leading sign
        sign = ""
        rest = txt
        if rest[:1] in {"+", "-"}:
            sign, rest = rest[0], rest[1:]

        if numeric_only:
            # Width applies to digit count (exclude sign)
            if len(rest) >= w:
                return f"{sign}{rest}"
            need = w - len(rest)
            return f"{sign}{fc * need}{rest}"
        else:
            # Width applies to total length (including sign) like zfill
            if len(sign) + len(rest) >= w:
                return f"{sign}{rest}"
            need = w - (len(sign) + len(rest))
            return f"{sign}{fc * need}{rest}"
    out = x.map(_pad_one)
    return out.astype("string")


def keep_alnum(
    s: pd.Series,
    *,
    allowed: Optional[Iterable[str]] = None,
    collapse_allowed_runs: bool = True,
    case: Optional[str] = None,  # "upper" | "lower" | None
    empty_to_na: bool = False,
    strip: bool = False,
) -> pd.Series:
    """
    Keep only alphanumeric characters (Unicode-aware) and optionally some allowed punctuation.

    Parameters
    ----------
    allowed : iterable of one-char strings, optional
        Additional characters to keep (e.g., {'-','_'}). By default, None → keep only alnum.
    collapse_allowed_runs : bool, default True
        If True, collapse repeated runs of the SAME allowed char, e.g., "A--B---C" -> "A-B-C".
    case : {"upper","lower",None}, default None
        Force case. (IDs often prefer uppercase.)
    empty_to_na : bool, default False
        If the cleaned string ends up empty, return <NA> instead of "".
    strip : bool, default False
        Trim whitespace before processing.

    Notes
    -----
    - Uses `str.isalnum()` to preserve non-ASCII alphanumeric characters (e.g., "Åland" stays "Åland").
    - Returns StringDtype.
    """
    x = _to_string_series(s)
    if strip:
        x = x.str.strip()

    allowed_set = set(allowed) if allowed else set()

    def _clean_one(v: object) -> object:
        if pd.isna(v):
            return pd.NA
        txt = _nfkc(str(v))
        # Filter to alnum or allowed
        kept = "".join(ch for ch in txt if (ch.isalnum() or ch in allowed_set))
        # Optional: collapse repeated occurrences of the same allowed char
        if kept and collapse_allowed_runs and allowed_set:
            for ch in sorted(allowed_set, key=lambda c: (len(c), c), reverse=True):
                # collapse ch{2,} to a single ch
                kept = re.sub(re.escape(ch) + r"{2,}", ch, kept)
        if case == "upper":
            kept = kept.upper()
        elif case == "lower":
            kept = kept.lower()
        if empty_to_na and kept == "":
            return pd.NA
        return kept

    out = x.map(_clean_one)
    return out.astype("string")
