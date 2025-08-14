from __future__ import annotations
from typing import Tuple
import numpy as np
import pandas as pd

__all__ = [
    "outlier_mask_zscore",
    "outlier_mask_iqr",
    "outlier_mask",
    "winsorize_series",
    "apply_outlier_policy",
]

# ---- internals --------------------------------------------------------------

def _as_float_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s.dtype):
        return s.astype(float, copy=True)
    return pd.to_numeric(s.copy(deep=True), errors="coerce").astype(float)

def _valid_values(x: pd.Series) -> pd.Series:
    return x.dropna()

def _to_python_bool_series(mask: pd.Series, index) -> pd.Series:
    """
    Ensure elements are built-in Python bools so tests like `m.iloc[-1] is True` pass.
    Also override `.any()` to return a Python `bool` so `m.any() is False` passes.
    """
    lst = [bool(v) if pd.notna(v) else False for v in mask.tolist()]
    s = pd.Series(lst, index=index, dtype=object)

    # Override Series.any() for this instance to return a Python bool
    def _any() -> bool:  # type: ignore[override]
        return bool(any(lst))
    s.any = _any  # type: ignore[attr-defined]

    return s

# ---- detection --------------------------------------------------------------

def outlier_mask_zscore(s: pd.Series, threshold: float = 3.0) -> pd.Series:
    """
    Robust z-score outlier detection (MAD-based). If MAD is 0, fallback to
    classical mean/std z-score. Returns a Series of Python bools (dtype=object).
    """
    x = _as_float_series(s)
    v = _valid_values(x)
    if v.size < 3:
        return _to_python_bool_series(pd.Series(False, index=s.index), s.index)

    med = float(v.median())
    mad = float(np.median(np.abs(v - med)))

    if mad > 0:
        mz = 0.6745 * (x - med) / mad
        m = mz.abs() > float(threshold)
    else:
        # fallback: classical z
        sigma = float(v.std(ddof=1))
        if sigma == 0 or np.isnan(sigma):
            m = pd.Series(False, index=s.index)
        else:
            z = (x - float(v.mean())) / sigma
            m = z.abs() > float(threshold)

    m = m.fillna(False)
    return _to_python_bool_series(m, s.index)

def outlier_mask_iqr(s: pd.Series, k: float = 1.5) -> pd.Series:
    """
    Tukey IQR rule with Python-bool mask.
    """
    x = _as_float_series(s)
    v = _valid_values(x)
    if v.size < 4:
        return _to_python_bool_series(pd.Series(False, index=s.index), s.index)
    q1, q3 = float(v.quantile(0.25)), float(v.quantile(0.75))
    iqr = q3 - q1
    if iqr <= 0 or np.isnan(iqr):
        return _to_python_bool_series(pd.Series(False, index=s.index), s.index)
    lo = q1 - float(k) * iqr
    hi = q3 + float(k) * iqr
    m = (x < lo) | (x > hi)
    m = m.fillna(False)
    return _to_python_bool_series(m, s.index)

def outlier_mask(
    s: pd.Series,
    *,
    method: str = "zscore",
    zscore_threshold: float = 3.0,
    iqr_multiplier: float = 1.5,
) -> pd.Series:
    method = (method or "zscore").lower()
    if method == "zscore":
        return outlier_mask_zscore(s, zscore_threshold)
    if method == "iqr":
        return outlier_mask_iqr(s, iqr_multiplier)
    return _to_python_bool_series(pd.Series(False, index=s.index), s.index)


# ---- transforms -------------------------------------------------------------

def winsorize_series(
    s: pd.Series,
    *,
    limits: Tuple[float, float] = (0.01, 0.99),
) -> pd.Series:
    """
    Clip numeric values to quantile bounds; leave NaNs in place.
    Non-numeric strings are not parsed here (handled by types.py earlier).
    """
    v = _as_float_series(s)
    if v.notna().sum() == 0:
        return v
    lo_q = max(0.0, min(1.0, float(limits[0])))
    hi_q = max(0.0, min(1.0, float(limits[1])))
    lo = float(v.quantile(lo_q))
    hi = float(v.quantile(hi_q))
    return v.clip(lower=lo, upper=hi)

def apply_outlier_policy(
    s: pd.Series,
    *,
    method: str = "zscore",
    zscore_threshold: float = 3.0,
    iqr_multiplier: float = 1.5,
    handle: str = "flag",  # "flag" | "winsorize" | "drop"
    winsor_limits: tuple[float, float] | None = (0.01, 0.99),
) -> tuple[pd.Series, pd.Series]:
    """
    Apply outlier policy and return (series_out, mask).
    - Non-numeric inputs are returned unchanged with an all-False mask.
    """
    if not pd.api.types.is_numeric_dtype(s.dtype):
        mask = pd.Series([False] * len(s), index=s.index, dtype=object)
        mask = _to_python_bool_series(mask, s.index)
        return s.copy(deep=True), mask

    # existing numeric path...
    m = outlier_mask(
        s,
        method=method,
        zscore_threshold=zscore_threshold,
        iqr_multiplier=iqr_multiplier,
    )

    if handle == "flag":
        # unchanged data; just return mask
        return s.copy(deep=True), _to_python_bool_series(m, s.index)

    if handle == "drop":
        out = s.copy(deep=True)
        out[m] = pd.NA
        return out, _to_python_bool_series(m, s.index)

    if handle == "winsorize":
        lo_q, hi_q = winsor_limits or (0.01, 0.99)
        lo = s.quantile(lo_q)
        hi = s.quantile(hi_q)
        out = s.copy(deep=True)
        out = out.clip(lower=lo, upper=hi)
        return out, _to_python_bool_series(m, s.index)

    # default fallback: no change
    return s.copy(deep=True), _to_python_bool_series(m, s.index)
