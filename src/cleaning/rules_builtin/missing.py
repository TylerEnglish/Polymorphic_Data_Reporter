from __future__ import annotations
from typing import Any, Optional
import pandas as pd
import numpy as np
from pandas.api.types import CategoricalDtype as _CatDtype

_NULL_TOKENS_LOWER = {
    "", "-", "—", "–", "n/a", "na", "none", "null", "nil", "nan", "nat",
    "<na>", "<null>", "<none>", "<na>", "<null>", "<none>",
}

def _is_datetime_index(idx: pd.Index) -> bool:
    try:
        return pd.api.types.is_datetime64_any_dtype(idx)
    except Exception:
        return False


def _numeric_view(s: pd.Series) -> pd.Series:
    """
    Return a float view suitable for numeric imputation.
    Does NOT mutate the input; preserves index.
    """
    if pd.api.types.is_numeric_dtype(s.dtype):
        # Ensure float so NaN is representable for ints/bools
        return s.astype(float).copy(deep=True)
    # Best-effort coercion
    return pd.to_numeric(s, errors="coerce").astype(float)


def impute_numeric(
    s: pd.Series,
    method: str = "median",  # "mean" | "median" | "ffill" | "bfill" | "interpolate"
    *,
    time_aware: bool = False,
    limit_direction: Optional[str] = None,
) -> pd.Series:
    """
    Impute numeric series according to method. Pure (returns a new Series).

    - mean/median: fill NaNs with the aggregate (ignoring NaNs).
    - ffill/bfill: forward/backward fill.
    - interpolate: linear by default; if time_aware and index is datetime-like, use method="time".
    """
    x = _numeric_view(s)

    if method == "mean":
        val = float(np.nanmean(x.values)) if x.notna().any() else np.nan
        return x.fillna(val)

    if method == "median":
        val = float(np.nanmedian(x.values)) if x.notna().any() else np.nan
        return x.fillna(val)

    if method == "ffill":
        return x.ffill()

    if method == "bfill":
        return x.bfill()

    if method == "interpolate":
        if time_aware and _is_datetime_index(x.index):
            return x.interpolate(method="time", limit_direction=limit_direction or "both")
        return x.interpolate(method="linear", limit_direction=limit_direction or "both")

    raise ValueError("method must be one of: mean, median, ffill, bfill, interpolate")

def impute_datetime(s: pd.Series, method: str = "ffill", value: str | None = None) -> pd.Series:
    """
    Impute missing datetimes.
      method: "ffill" | "bfill" | "median" | "mode" | "constant"
      value:  used when method="constant" (e.g., "2000-01-01")
    """
    x = pd.to_datetime(s, errors="coerce")

    if method in ("ffill", "bfill"):
        return x.fillna(method=method)

    if method == "median":
        # handle tz-aware and tz-naive safely, without .view()
        if hasattr(x.dt, "tz") and x.dt.tz is not None:
            tz = x.dt.tz
            vals_i8 = x.dropna().dt.tz_convert("UTC").astype("int64")  # ns since epoch UTC
            if vals_i8.empty:
                return x
            med_ns = int(vals_i8.median())
            med = pd.to_datetime(med_ns, utc=True).tz_convert(tz)
        else:
            vals_i8 = x.dropna().astype("int64")
            if vals_i8.empty:
                return x
            med_ns = int(vals_i8.median())
            med = pd.to_datetime(med_ns)  # tz-naive
        return x.fillna(med)

    if method == "mode":
        m = x.dropna().mode()
        return x.fillna(m.iloc[0]) if not m.empty else x

    if method == "constant":
        target = pd.to_datetime("1970-01-01") if value is None else pd.to_datetime(value, errors="coerce")
        return x.fillna(target)

    return x

def _ensure_category_contains(s: pd.Series, value: Any) -> pd.Series:
    """
    If s is categorical and value not in categories, add it. Return a NEW series.
    """
    if not isinstance(s.dtype, _CatDtype):
        return s.copy(deep=True)
    cat = s.copy(deep=True)
    if value not in cat.cat.categories:
        cat = cat.cat.add_categories([value])
    return cat


def _fillna_no_warn(s: pd.Series, value: Any) -> pd.Series:
    """
    Fill NaNs without triggering Pandas' future downcasting warning by avoiding
    Series.fillna on object dtype. Pure: returns a new Series.
    """
    out = s.copy(deep=True)
    m = out.isna()
    if not m.any():
        return out
    out = out.where(~m, value)
    # Align with prior intent to 'infer' object dtypes explicitly
    return out.infer_objects(copy=False)


def impute_value(s: pd.Series, value: Any, *, force: bool = False) -> pd.Series:
    """
    Generic impute by constant value (works for text/categorical).
    Always fill with the provided literal (except when value is None and force=False).
    """
    if (value is None) and (not force):
        # keep true nulls; ensure pandas "string" if texty
        if isinstance(s.dtype, _CatDtype):
            return s.copy(deep=True)
        if pd.api.types.is_string_dtype(s) or pd.api.types.is_object_dtype(s):
            return s.astype("string")
        return s.copy(deep=True)

    # Categorical: ensure category exists, then fill
    if isinstance(s.dtype, _CatDtype):
        cat = _ensure_category_contains(s, value)
        return cat.fillna(value)

    # Text-like: keep nullable string dtype
    if pd.api.types.is_string_dtype(s) or pd.api.types.is_object_dtype(s):
        return s.astype("string").fillna(value).astype("string")

    # Numerics / others: fill directly
    return s.fillna(value)


def impute_mode(s: pd.Series) -> pd.Series:
    """
    Impute with the most frequent non-null value.
    - Ties: pick the first by value_counts order.
    - Preserves categorical dtype if possible.
    """
    vc = s.dropna().value_counts()
    fill = vc.index[0] if len(vc) else None

    if fill is None:
        return s.copy(deep=True)

    if isinstance(s.dtype, _CatDtype):
        cat = _ensure_category_contains(s, fill)
        return cat.fillna(fill)

    return _fillna_no_warn(s, fill)


def impute_bool_mode(s: pd.Series) -> pd.Series:
    """
    Specialized convenience: impute boolean-like series by its mode.
    Returns a boolean dtype if possible, otherwise leaves dtype as-is.
    """
    out = impute_mode(s)
    try:
        # Only cast if there are no NaNs and values are bool-ish
        if not out.isna().any():
            unique_vals = set(out.unique().tolist())
            if unique_vals.issubset({True, False}):
                return out.astype(bool)
    except Exception:
        pass
    return out
