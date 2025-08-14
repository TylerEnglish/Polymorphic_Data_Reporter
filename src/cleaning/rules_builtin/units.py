from __future__ import annotations
from typing import Dict, Tuple, Optional
import pandas as pd
import numpy as np

from .types import coerce_numeric_from_string  # reuse our robust string→float

PercentTarget = str  # "fraction" (0..1) or "percent" (0..100)

def _maybe_copy(s: pd.Series) -> pd.Series:
    # Keep purity guarantees even for numeric series
    return s.copy(deep=True)

def _coerce_any_to_float(s: pd.Series) -> pd.Series:
    """
    Accepts numeric or object-like series; returns float dtype with NaNs for
    unparseable values. Pure.
    """
    if pd.api.types.is_numeric_dtype(s.dtype):
        return pd.to_numeric(_maybe_copy(s), errors="coerce").astype(float)
    # object/strings: go through our rich coercer
    return coerce_numeric_from_string(s)

def _normalize_percent(
    x: pd.Series,
    *,
    target: PercentTarget = "fraction",
) -> Tuple[pd.Series, Dict[str, object]]:
    """
    Normalize a numeric float series `x` that *may* be percent-encoded in two ways:
      - already fraction (0..1),
      - whole-number percent (0..100).
    Heuristic: values > 1 and <= 100 likely denote whole-number percent.
    We only rescale those >1 when target == "fraction".
    """
    out = x.copy(deep=True)

    meta: Dict[str, object] = {"unit_in": "mixed/unknown", "unit_out": target, "rescaled": False}

    if target not in {"fraction", "percent"}:
        target = "fraction"

    non_null = out.dropna()
    if non_null.empty:
        meta["unit_in"] = "unknown"
        return out, meta

    # Simple heuristic: if most values are <= 1, treat as fraction; otherwise treat as percent.
    share_le1 = float((non_null <= 1.0).mean())
    unit_in = "fraction" if share_le1 >= 0.8 else "percent"
    meta["unit_in"] = unit_in

    if unit_in == target:
        return out, meta

    if unit_in == "percent" and target == "fraction":
        mask = out.notna() & (out > 1.0)
        if mask.any():
            out.loc[mask] = out.loc[mask] / 100.0
            meta["rescaled"] = True
        return out, meta

    if unit_in == "fraction" and target == "percent":
        mask = out.notna() & (out <= 1.0)
        if mask.any():
            out.loc[mask] = out.loc[mask] * 100.0
            meta["rescaled"] = True
        return out, meta

    return out, meta  # fallback (shouldn't hit)

def standardize_numeric_units(
    s: pd.Series,
    *,
    unit_hint: Optional[str] = None,
    percent_target: PercentTarget = "fraction",
) -> Tuple[pd.Series, Dict[str, object]]:
    """
    Unified entrypoint for unit normalization.

    - Always returns a float series (pure) and a small meta dict.
    - If `unit_hint == "percent"`, normalize mixed encodings to `percent_target`
      (default: "fraction" → 0..1).
    - Currency and k/m magnitudes are already handled in coerce step.
    """
    # First, coerce to float using our robust parser (handles $, %, k/m, parens neg, etc.)
    numeric = _coerce_any_to_float(s)

    meta: Dict[str, object] = {
        "hint": unit_hint or "",
        "percent_target": percent_target,
        "unit_in": None,
        "unit_out": None,
        "rescaled": False,
    }

    if (unit_hint or "").lower() == "percent":
        norm, pm = _normalize_percent(numeric, target=percent_target)
        meta.update(pm)
        return norm.astype(float), meta

    # For non-percent hints (currency/magnitude), coercion already standardized to base units.
    meta["unit_in"] = "base"
    meta["unit_out"] = "base"
    return numeric.astype(float), meta
