from __future__ import annotations
from typing import Any, Dict
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
    """
    # TODO: compute unit_hint by ingesting schema hints at call-site if needed
    out: dict[str, dict[str, Any]] = {}
    n = len(df)

    time_cols = [c for c, r in schema_roles.items() if r == "time"]
    has_time_index = len(time_cols) == 1  # simple heuristic; refine later if needed

    for c in df.columns:
        s = df[c]
        t = _series_dtype_str(s)
        role = schema_roles.get(c, "text")

        missing_pct = float(s.isna().mean()) if n else 0.0
        non_null_ratio = 1.0 - missing_pct
        nunique = _safe_nunique(s)
        unique_ratio = float(nunique) / float(n or 1)

        avg_len = _string_avg_len(s) if t == "string" else None
        stats = _numeric_basic_stats(s) if t in {"int", "float"} else {"mean": None, "std": None, "iqr": None}
        btr = _bool_token_ratio(s) if t == "string" else 0.0

        out[c] = {
            "name": c,
            "type": t,
            "role": role,
            "missing_pct": missing_pct,
            "non_null_ratio": non_null_ratio,
            "nunique": int(nunique),
            "unique_ratio": unique_ratio,
            "cardinality": int(nunique),
            "avg_len": avg_len,
            "has_time_index": has_time_index,
            "mean": stats["mean"],
            "std": stats["std"],
            "iqr": stats["iqr"],
            "bool_token_ratio": btr,
            # "unit_hint": None,  # caller can enrich from schema.hints if desired
        }
    return out

# ---- Helpers (pure) ----

def _series_dtype_str(s: pd.Series) -> str:
    # use same mapping as NLP for consistency
    return _nlp_dtype_str(s)

def _safe_nunique(s: pd.Series) -> int:
    try:
        return int(s.nunique(dropna=True))
    except TypeError:
        return int(s.astype(str).nunique(dropna=True))

def _string_avg_len(s: pd.Series) -> float | None:
    try:
        vals = s.dropna().astype(str)
        return float(vals.map(len).mean()) if not vals.empty else 0.0
    except Exception:
        return None

def _numeric_basic_stats(s: pd.Series) -> dict[str, float | None]:
    try:
        x = pd.to_numeric(s, errors="coerce").dropna()
        if x.empty:
            return {"mean": None, "std": None, "iqr": None}
        q1 = float(x.quantile(0.25))
        q3 = float(x.quantile(0.75))
        return {"mean": float(x.mean()), "std": float(x.std(ddof=1)), "iqr": float(q3 - q1)}
    except Exception:
        return {"mean": None, "std": None, "iqr": None}

def _bool_token_ratio(s: pd.Series) -> float:
    try:
        vals = s.dropna().astype(str).str.strip().str.lower()
        if vals.empty:
            return 0.0
        tokens = {"true","false","t","f","y","n","yes","no","1","0"}
        return float(vals.isin(tokens).mean())
    except Exception:
        return 0.0
