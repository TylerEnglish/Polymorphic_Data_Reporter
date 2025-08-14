from __future__ import annotations
import numpy as np
import pandas as pd

from src.cleaning.rules_builtin.outliers import (
    outlier_mask_zscore,
    outlier_mask_iqr,
    outlier_mask,
    winsorize_series,
    apply_outlier_policy,
)

def test_zscore_outlier_mask_basic():
    s = pd.Series([0.0, 1.0, 2.0, 100.0])
    m = outlier_mask_zscore(s, threshold=3.0)
    assert m.tolist() == [False, False, False, True]

def test_iqr_outlier_mask_basic():
    s = pd.Series([1, 1, 2, 2, 2, 3, 3, 100])
    m = outlier_mask_iqr(s, k=1.5)
    assert m.tolist()[-1] is True
    assert m.sum() == 1

def test_unified_mask_method_switch():
    s = pd.Series([1, 2, 3, 100])
    m1 = outlier_mask(s, method="zscore", zscore_threshold=2.5)
    m2 = outlier_mask(s, method="iqr", iqr_multiplier=1.5)
    assert m1.any()
    assert m2.any()

def test_winsorize_clips_extremes():
    s = pd.Series([1, 2, 3, 100, np.nan, -999])
    w = winsorize_series(s, limits=(0.05, 0.95))
    assert w.notna().sum() == s.notna().sum()
    # extremes should be within original inner range
    assert w.max() <= 100
    assert w.min() >= -999
    # but clipped closer to central range
    assert w.max() < 100 or w.min() > -999

def test_apply_policy_flag_returns_original_and_mask():
    s = pd.Series([0, 1, 2, 100])
    out, m = apply_outlier_policy(s, method="zscore", zscore_threshold=3.0, handle="flag")
    pd.testing.assert_series_equal(out, s)  # unchanged
    assert m.sum() == 1 and bool(m.iloc[-1]) is True

def test_apply_policy_winsorize_replaces_extreme():
    s = pd.Series([10, 11, 9, 1000])
    out, m = apply_outlier_policy(s, method="iqr", iqr_multiplier=1.5, handle="winsorize", winsor_limits=(0.10, 0.90))
    assert m.iloc[-1] is True
    assert out.iloc[-1] < 1000  # clipped down

def test_apply_policy_drop_sets_nan_on_outliers():
    s = pd.Series([1, 2, 3, 100])
    out, m = apply_outlier_policy(s, method="zscore", zscore_threshold=2.5, handle="drop")
    assert m.iloc[-1] is True
    assert pd.isna(out.iloc[-1])
    # non-outliers stay
    assert out.iloc[0] == 1

def test_non_numeric_series_no_outliers_and_no_crash():
    s = pd.Series(["a", "b", "c", "d"])
    m = outlier_mask(s, method="zscore")
    assert m.any() is False
    out, m2 = apply_outlier_policy(s, handle="winsorize")
    # unchanged for all-strings
    pd.testing.assert_series_equal(out.astype(object), s.astype(object))
    assert m2.any() is False

def test_small_or_constant_series_has_no_outliers():
    s = pd.Series([1, 1, 1])
    assert outlier_mask_zscore(s).any() is False
    assert outlier_mask_iqr(s).any() is False
