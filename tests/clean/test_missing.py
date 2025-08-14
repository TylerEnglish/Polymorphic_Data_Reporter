from __future__ import annotations
import pandas as pd
import numpy as np

from src.cleaning.rules_builtin.missing import (
    impute_numeric,
    impute_value,
    impute_mode,
    impute_bool_mode,
)


def test_impute_numeric_mean_and_median():
    s = pd.Series([1.0, np.nan, 3.0, np.nan])
    out_mean = impute_numeric(s, "mean")
    out_median = impute_numeric(s, "median")
    assert out_mean.isna().sum() == 0
    assert out_median.isna().sum() == 0
    # mean should be (1+3)/2 = 2
    assert out_mean.iloc[1] == 2.0 and out_mean.iloc[3] == 2.0
    # median of [1,3] = 2
    assert out_median.iloc[1] == 2.0 and out_median.iloc[3] == 2.0
    # purity
    assert out_mean is not s and out_median is not s


def test_impute_numeric_ffill_bfill():
    s = pd.Series([1.0, np.nan, np.nan, 4.0])
    out_ffill = impute_numeric(s, "ffill")
    out_bfill = impute_numeric(s, "bfill")
    assert list(out_ffill) == [1.0, 1.0, 1.0, 4.0]
    assert list(out_bfill) == [1.0, 4.0, 4.0, 4.0]


def test_impute_numeric_interpolate_linear_and_time():
    # Linear
    s = pd.Series([1.0, np.nan, 3.0], index=[0, 1, 2])
    out_lin = impute_numeric(s, "interpolate", time_aware=False)
    assert np.isclose(out_lin.iloc[1], 2.0)

    # Time-aware (datetime index)
    idx = pd.to_datetime(["2024-01-01", "2024-01-03", "2024-01-05"])
    s2 = pd.Series([10.0, np.nan, 30.0], index=idx)
    out_time = impute_numeric(s2, "interpolate", time_aware=True)
    # The midpoint in time between 1st and 3rd is 2nd; value should be ~20
    assert np.isclose(out_time.iloc[1], 20.0)


def test_impute_value_for_text_and_categorical():
    s_text = pd.Series(["a", None, "b", None], dtype="object")
    out_text = impute_value(s_text, "N/A")
    assert list(out_text) == ["a", "N/A", "b", "N/A"]

    s_cat = pd.Series(["red", None, "blue", None], dtype="category")
    out_cat = impute_value(s_cat, "Unknown")
    assert str(out_cat.dtype) == "category"
    # New category should be present
    assert "Unknown" in list(out_cat.cat.categories)
    assert list(out_cat.astype("object")) == ["red", "Unknown", "blue", "Unknown"]


def test_impute_mode_and_bool_mode():
    s = pd.Series([None, "x", "y", "x", None])
    out = impute_mode(s)
    assert list(out) == ["x", "x", "y", "x", "x"]

    b = pd.Series([True, None, True, False, None], dtype="object")
    out_b = impute_bool_mode(b)
    # Mode is True; cast back to bool
    assert list(out_b.astype("object")) == [True, True, True, False, True]
    assert pd.api.types.is_bool_dtype(out_b.dtype) or set(out_b.unique()).issubset({True, False})
