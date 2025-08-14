from __future__ import annotations
from typing import Any, Dict, Tuple
import pandas as pd
import numpy as np

from ..nlp.roles import _dtype_str as _nlp_dtype_str  # keep dtype mapping consistent

# ---- Public API ----

def profile_columns(
    df: pd.DataFrame,
    schema_roles: dict[str, str],
    datetime_formats: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """
    Build a metrics context per column used by the DSL and reports.
    Pure: no side effects.

    Returns: { col -> metrics dict }
    Keys expected by DSL:
      - name, type, role
      - missing_pct, non_null_ratio
      - nunique, unique_ratio, cardinality
      - avg_len (strings only; else None)
      - has_time_index (bool)
      - mean, std, iqr (numeric only; else None)
      - bool_token_ratio (strings only)
      - unit_hint (optional; may be filled by upstream schema hints)

    Additional safe goodies (may be useful later):
      - min, max (numeric only; else None)
      - datetime_parse_ratio (strings w/ role 'time'; else 0.0)
      - is_monotonic_increasing (datetime or numeric)
    """
    out: dict[str, dict[str, Any]] = {}
    n = int(len(df))

    # simple heuristic; refine later if needed
    time_cols = [c for c, r in schema_roles.items() if r == "time"]
    has_time_index = len(time_cols) == 1

    for c in df.columns:
        s = df[c]
        t = _series_dtype_str(s)
        role = schema_roles.get(c, "text")

        # basic counts
        try:
            missing_pct = float(s.isna().mean()) if n else 0.0
        except Exception:
            # highly exotic dtypes – fall back to treating nothing as missing
            missing_pct = 0.0
        non_null_ratio = 1.0 - missing_pct

        nunique = _safe_nunique(s)
        unique_ratio = float(nunique) / float(n or 1)

        # type-specific metrics
        if t in {"int", "float"}:
            stats = _numeric_stats(s)
            avg_len = None
            btr = 0.0
            dt_ratio = 0.0
            is_mono = _is_monotonic_numeric(s)
        elif t == "string":
            stats = {"mean": None, "std": None, "iqr": None, "min": None, "max": None}
            avg_len = _string_avg_len(s)
            btr = _bool_token_ratio(s)
            # If schema hints this column is time-like, estimate how parseable it is
            dt_ratio = _datetime_parse_ratio(s, datetime_formats) if role == "time" else 0.0
            is_mono = False
        elif t in {"datetime", "date"}:
            stats = {"mean": None, "std": None, "iqr": None, "min": None, "max": None}
            avg_len = None
            btr = 0.0
            dt_ratio = 1.0  # already datetime-like
            is_mono = bool(getattr(s, "is_monotonic_increasing", False))
        else:
            stats = {"mean": None, "std": None, "iqr": None, "min": None, "max": None}
            avg_len = None
            btr = 0.0
            dt_ratio = 0.0
            is_mono = False

        out[c] = {
            "name": c,
            "type": t,
            "role": role,
            "missing_pct": float(missing_pct),
            "non_null_ratio": float(non_null_ratio),
            "nunique": int(nunique),
            "unique_ratio": float(unique_ratio),
            "cardinality": int(nunique),
            "avg_len": avg_len,
            "has_time_index": bool(has_time_index),
            "mean": stats["mean"],
            "std": stats["std"],
            "iqr": stats["iqr"],
            "min": stats.get("min"),
            "max": stats.get("max"),
            "bool_token_ratio": float(btr),
            "datetime_parse_ratio": float(dt_ratio),
            "is_monotonic_increasing": bool(is_mono),
            # "unit_hint": None,  # caller can enrich from schema.hints if desired
        }
    return out

# ---- Helpers (pure) ----

def _series_dtype_str(s: pd.Series) -> str:
    # use same mapping as NLP for consistency
    return _nlp_dtype_str(s)

def _safe_nunique(s: pd.Series) -> int:
    """Robust nunique that handles unhashables and treats nulls correctly."""
    try:
        # Fast path for hashables
        return int(s.nunique(dropna=True))
    except TypeError:
        # Drop real nulls first so None/pd.NA/nan aren't turned into "None"/"nan"
        non_null = s[~s.isna()]
        try:
            # Many unhashables (lists, dicts) can be compared via repr safely enough for counts
            return int(non_null.astype(str).nunique())
        except Exception:
            # Last resort: map to repr explicitly, then count
            return int(pd.Series(non_null.map(repr)).nunique())

def _string_avg_len(s: pd.Series) -> float | None:
    try:
        vals = s.dropna().astype(str)
        return float(vals.map(len).mean()) if not vals.empty else 0.0
    except Exception:
        return None

def _numeric_stats(s: pd.Series) -> Dict[str, float | None]:
    """
    Return mean, std (ddof=1), iqr, min, max for numeric-ish series.
    Coerces safely; returns None for empty results.
    """
    try:
        x = pd.to_numeric(s, errors="coerce").dropna()
        if x.empty:
            return {"mean": None, "std": None, "iqr": None, "min": None, "max": None}
        q1 = float(x.quantile(0.25))
        q3 = float(x.quantile(0.75))
        return {
            "mean": float(x.mean()),
            "std": float(x.std(ddof=1)),
            "iqr": float(q3 - q1),
            "min": float(x.min()),
            "max": float(x.max()),
        }
    except Exception:
        return {"mean": None, "std": None, "iqr": None, "min": None, "max": None}

def _bool_token_ratio(s: pd.Series) -> float:
    """
    Share of values that look like boolean tokens (true/false/yes/no/1/0…),
    after trimming & lowercasing; non-strings are coerced to str first.
    """
    try:
        vals = s.dropna().astype(str).str.strip().str.lower()
        if vals.empty:
            return 0.0
        tokens = {"true", "false", "t", "f", "y", "n", "yes", "no", "1", "0"}
        return float(vals.isin(tokens).mean())
    except Exception:
        return 0.0

def _datetime_parse_ratio(s: pd.Series, fmts: list[str] | None) -> float:
    """
    For string series, attempt parse using provided formats, falling back to pandas'
    general parser. Returns fraction of non-null values that parse to a timestamp.
    Pure; does not mutate.
    """
    try:
        x = s.dropna().astype(str)
        if x.empty:
            return 0.0
        parsed = None
        for f in (fmts or []):
            try:
                cand = pd.to_datetime(x, format=f, errors="coerce")
            except Exception:
                continue
            parsed = cand if parsed is None else parsed.fillna(cand)
        if parsed is None:
            parsed = pd.to_datetime(x, errors="coerce")
        non_null = int(x.shape[0])
        ok = int(parsed.notna().sum())
        return float(ok) / float(non_null or 1)
    except Exception:
        return 0.0

def _is_monotonic_numeric(s: pd.Series) -> bool:
    try:
        x = pd.to_numeric(s, errors="coerce")
        return bool(x.is_monotonic_increasing)
    except Exception:
        return False
