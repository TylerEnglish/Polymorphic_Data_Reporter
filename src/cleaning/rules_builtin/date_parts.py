from __future__ import annotations
from typing import Iterable, List, Sequence, Optional
import re
import pandas as pd


__all__ = [
    "to_datetime_robust",
    "dt_round",
    "dt_floor",
    "dt_ceil",
    "dt_part",
    "extract_datetime_from_text",
]


# =============================================================================
# Hard-coded regex patterns for finding date/time blobs inside messy strings
# =============================================================================

# ISO-like: 2024-07-01 or 2024-07-01 15:20[:30][.123456] with optional Z or ±hh[:mm]
_ISO_DATETIME_RE = re.compile(
    r"\b"
    r"\d{4}-\d{2}-\d{2}"
    r"(?:[ T]"
    r"\d{2}:\d{2}"
    r"(?::\d{2})?"
    r"(?:\.\d{1,9})?"
    r"(?:Z|[+-]\d{2}:?\d{2})?"
    r")?"
    r"\b",
    flags=re.UNICODE,
)

# yyyy/mm/dd or yyyy-mm-dd with flexible month/day widths
_YMD_SLASH_DASH_RE = re.compile(r"\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b", re.UNICODE)

# mm/dd/yyyy or dd/mm/yyyy (ambiguous, caller can set dayfirst=True)
_MDY_DMY_RE = re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", re.UNICODE)

# Month name forms: "Jan 2 2024", "January 2, 2024", "2 January 2024"
_MONTHS = r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|" \
          r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
_MON_DY_YR_RE = re.compile(rf"\b{_MONTHS}\s+\d{{1,2}},?\s+\d{{2,4}}\b", re.IGNORECASE | re.UNICODE)
_DY_MON_YR_RE = re.compile(rf"\b\d{{1,2}}\s+{_MONTHS}\s+\d{{2,4}}\b", re.IGNORECASE | re.UNICODE)

# Compact YYYYMMDD (ensure plausibility before using)
_COMPACT_YYYYMMDD_RE = re.compile(r"\b(\d{4})(\d{2})(\d{2})\b", re.UNICODE)

# Digits-only epoch detection
_DIGITS_ONLY_RE = re.compile(r"^\s*[+-]?\d{9,19}\s*$")


_DEFAULT_PATTERNS: List[re.Pattern] = [
    _ISO_DATETIME_RE,
    _YMD_SLASH_DASH_RE,
    _MDY_DMY_RE,
    _MON_DY_YR_RE,
    _DY_MON_YR_RE,
    _COMPACT_YYYYMMDD_RE,  # handled specially
]


# =============================================================================
# Helpers
# =============================================================================

def _as_string_series(s: pd.Series) -> pd.Series:
    """Convert to pandas StringDtype, preserving NA."""
    try:
        return s.astype("string")
    except Exception:
        # Fallback: ensure we still return something string-like
        return s.astype(object).astype("string")


def _choose_epoch_unit(text: str) -> Optional[str]:
    """
    Heuristically guess epoch unit from digit length:
    seconds(~10), ms(13), µs(16), ns(19).
    Return one of {"s","ms","us","ns"} or None if not confident.
    """
    t = text.strip().lstrip("+-")
    n = len(t)
    if n in (9, 10, 11):   # allow 9/11 for leading sign or slight variations
        return "s"
    if n == 13:
        return "ms"
    if n == 16:
        return "us"
    if n == 19:
        return "ns"
    return None


def _maybe_parse_compact(match: re.Match) -> Optional[str]:
    """Convert a compact YYYYMMDD match to 'YYYY-MM-DD' if plausible; else None."""
    try:
        y, m, d = match.group(1), match.group(2), match.group(3)
        yy, mm, dd = int(y), int(m), int(d)
        if 1 <= mm <= 12 and 1 <= dd <= 31:
            return f"{yy:04d}-{mm:02d}-{dd:02d}"
    except Exception:
        pass
    return None


def extract_datetime_from_text(
    s: pd.Series,
    *,
    patterns: Sequence[re.Pattern] | None = None,
    case_insensitive: bool = False,
) -> pd.Series:
    """
    Extract the first date/time-like substring per row using hard-coded regexes.
    Returns a pandas 'string' series with the matched substring (original casing),
    or <NA> when no match.

    This does **not** parse to datetime; see `to_datetime_robust` if you want
    parsing too (it can call this as a first step).
    """
    if patterns is None:
        patterns = _DEFAULT_PATTERNS

    x = _as_string_series(s)
    # Optionally toggle casefold on month-name patterns without recompiling all
    compiled = []
    for p in patterns:
        if case_insensitive and p.flags & re.IGNORECASE == 0:
            compiled.append(re.compile(p.pattern, p.flags | re.IGNORECASE))
        else:
            compiled.append(p)

    def _one(val: object) -> object:
        if pd.isna(val):
            return pd.NA
        text = str(val)
        for rx in compiled:
            m = rx.search(text)
            if not m:
                continue
            if rx is _COMPACT_YYYYMMDD_RE:
                as_norm = _maybe_parse_compact(m)
                if as_norm:
                    return as_norm  # normalized YYYY-MM-DD
                # If implausible, fall through to next pattern
                continue
            return m.group(0)
        return pd.NA

    return x.map(_one).astype("string")


def to_datetime_robust(
    s: pd.Series,
    *,
    utc: bool = False,
    dayfirst: bool = False,
    yearfirst: bool = False,
    assume_utc_if_tz_naive: bool = True,
) -> pd.Series:
    # Step 1: vectorized parse
    try:
        out = pd.to_datetime(
            s, errors="coerce",
            dayfirst=dayfirst, yearfirst=yearfirst,
            utc=(utc if assume_utc_if_tz_naive else False),
        )
    except Exception:
        out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")

    ss = s.astype("string")

    def _patch_epoch(mask: pd.Series, unit: str):
        if not mask.any():
            return
        vals = pd.to_numeric(ss.where(mask), errors="coerce")
        valid = mask & vals.notna()
        if valid.any():
            dt = pd.to_datetime(vals.loc[valid].astype("int64"), unit=unit, utc=utc, errors="coerce")
            # 🔧 keep tz/dtype by assigning the Series (NOT `.values`)
            out.loc[valid] = dt

    m_sec = ss.str.match(r"^[+-]?\d{10}$", na=False)
    m_ms  = ss.str.match(r"^[+-]?\d{13}$", na=False)
    m_us  = ss.str.match(r"^[+-]?\d{16}$", na=False)
    m_ns  = ss.str.match(r"^[+-]?\d{19}$", na=False)

    nat_mask = out.isna()
    _patch_epoch(nat_mask & m_sec, "s")
    _patch_epoch(nat_mask & m_ms,  "ms")
    _patch_epoch(nat_mask & m_us,  "us")
    _patch_epoch(nat_mask & m_ns,  "ns")

    still_nat = out.isna()
    if still_nat.any():
        extracted = extract_datetime_from_text(ss.loc[still_nat])
        filled = pd.to_datetime(
            extracted, errors="coerce",
            dayfirst=dayfirst, yearfirst=yearfirst,
            utc=(utc if assume_utc_if_tz_naive else False),
        )
        # 🔧 same here: assign the Series to preserve dtype/tz
        out.loc[still_nat] = filled

    return out


# =============================================================================
# Public API
# =============================================================================

def dt_round(s: pd.Series, *, freq: str = "D") -> pd.Series:
    """
    Round datetimes to the given `freq` (e.g., "D", "H", "15min").
    Coerces input to datetime; returns NaT where parsing fails.
    """
    x = to_datetime_robust(s)
    try:
        return x.dt.round(freq)
    except Exception:
        # If freq invalid or tz issues, just return parsed datetimes unrounded
        return x


def dt_floor(s: pd.Series, *, freq: str = "D") -> pd.Series:
    """Floor datetimes to the given frequency."""
    x = to_datetime_robust(s)
    try:
        return x.dt.floor(freq)
    except Exception:
        return x


def dt_ceil(s: pd.Series, *, freq: str = "D") -> pd.Series:
    """Ceil datetimes to the given frequency."""
    x = to_datetime_robust(s)
    try:
        return x.dt.ceil(freq)
    except Exception:
        return x


def dt_part(
    s: pd.Series,
    *,
    part: str = "date",
    dayfirst: bool = False,
    yearfirst: bool = False,
) -> pd.Series:
    """
    Extract a date/time component with sensible dtypes.

    Supported `part` values:
      - "date" (midnight-normalized datetime, preserves tz)
      - "time" (string, "HH:MM:SS" or "HH:MM:SS.ssssss" if needed)
      - "year","quarter","month","day","hour","minute","second"
      - "week","weekday" (Mon=0..Sun=6), "dayofyear"
      - "iso_year","iso_week","iso_day" (ISO-8601)
      - "month_name","day_name"

    Returns nullable integer dtype for numeric parts (Int64), string for names,
    and datetime for "date".
    """
    x = to_datetime_robust(s, dayfirst=dayfirst, yearfirst=yearfirst)

    part_cf = str(part).strip().lower()

    try:
        if part_cf == "date":
            return x.dt.normalize()

        if part_cf == "time":
            # keep as string to avoid Python time objects mixed with NA
            return x.dt.strftime("%H:%M:%S").astype("string")

        if part_cf == "year":
            return x.dt.year.astype("Int64")

        if part_cf == "quarter":
            return x.dt.quarter.astype("Int64")

        if part_cf == "month":
            return x.dt.month.astype("Int64")

        if part_cf in ("month_name", "monthname"):
            return x.dt.month_name().astype("string")

        if part_cf in ("day", "dayofmonth"):
            return x.dt.day.astype("Int64")

        if part_cf in ("day_name", "dayname"):
            return x.dt.day_name().astype("string")

        if part_cf in ("dayofyear", "doy"):
            return x.dt.day_of_year.astype("Int64")

        if part_cf == "hour":
            return x.dt.hour.astype("Int64")

        if part_cf == "minute":
            return x.dt.minute.astype("Int64")

        if part_cf == "second":
            return x.dt.second.astype("Int64")

        if part_cf == "weekday":
            # Monday=0..Sunday=6
            return x.dt.weekday.astype("Int64")

        if part_cf in ("week", "weekofyear", "isoweek"):
            # Prefer ISO week for stability across pandas versions
            try:
                return x.dt.isocalendar().week.astype("Int64")
            except Exception:
                # Fallback (older pandas)
                return x.dt.week.astype("Int64")

        if part_cf in ("iso_year", "isoyear"):
            return x.dt.isocalendar().year.astype("Int64")

        if part_cf in ("iso_week", "isoweek"):
            return x.dt.isocalendar().week.astype("Int64")

        if part_cf in ("iso_day", "isoday"):
            return x.dt.isocalendar().day.astype("Int64")

        # Unknown part: return parsed datetimes (better than raising)
        return x

    except Exception:
        # If something goes wrong on extraction, just return the parsed datetimes.
        return x
