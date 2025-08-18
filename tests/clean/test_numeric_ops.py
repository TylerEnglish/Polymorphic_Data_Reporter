import math
import pandas as pd
from src.cleaning.rules_builtin.numeric_ops import (
    clip_range,
    enforce_sign,
    zero_as_missing,
)

def assert_float_series_equal(a: pd.Series, b: pd.Series, atol=1e-12):
    # Same length, ignore name/dtype, compare with tolerance and NaN-equality
    pd.testing.assert_series_equal(
        a.reset_index(drop=True),
        b.reset_index(drop=True),
        check_names=False,
        check_dtype=False,
        check_index_type=False,
        check_categorical=False,
        check_exact=False,
        rtol=0,
        atol=atol,
        obj="Series",
    )

def test_clip_range_basic_and_strings():
    s = pd.Series([
        -5,
        "10",
        "15",
        "$1,234.50",
        "(100)",
        "−20",     # unicode minus
        None,
    ])
    out = clip_range(s, lo=0, hi=100)
    exp = pd.Series([
        0.0,       # clipped up to lo
        10.0,
        15.0,
        100.0,     # clipped down to hi
        0.0,       # (100) -> -100 -> clipped to 0
        0.0,       # −20 -> -20 -> clipped to 0
        math.nan,
    ])
    assert_float_series_equal(out, exp)

def test_enforce_sign_modes():
    s = pd.Series(["-3", "0", "4", "(5)", "−2", None])
    pos = enforce_sign(s, sign="positive")
    nonneg = enforce_sign(s, sign="nonnegative")
    neg = enforce_sign(s, sign="negative")

    exp_pos = pd.Series([math.nan, math.nan, 4.0, 5.0, math.nan, math.nan])
    exp_nonneg = pd.Series([0.0, 0.0, 4.0, 5.0, 0.0, math.nan])
    exp_neg = pd.Series([-3.0, math.nan, math.nan, math.nan, -2.0, math.nan])

    assert_float_series_equal(pos, exp_pos)
    assert_float_series_equal(nonneg, exp_nonneg)
    assert_float_series_equal(neg, exp_neg)

def test_zero_as_missing_eps():
    s = pd.Series(["0", "0.0001", "1e-6", "-0.00004", "0.1", None])
    out = zero_as_missing(s, eps=5e-5)
    exp = pd.Series([math.nan, 0.0001, math.nan, math.nan, 0.1, math.nan])
    assert_float_series_equal(out, exp)

def test_percent_and_currency_parsing_through_clip():
    s = pd.Series(["5%", "12.5 %", "(10%)", "$100", "€2,500.50"])
    out = clip_range(s)  # just parse
    # 5% -> 0.05 ; 12.5% -> 0.125 ; (10%) -> -0.10 ; $100 -> 100 ; €2,500.50 -> 2500.50
    exp = pd.Series([0.05, 0.125, -0.10, 100.0, 2500.50])
    assert_float_series_equal(out, exp, atol=1e-9)

def test_exponent_and_mixed_junk():
    s = pd.Series(["1.2e3", "foo -3.4E-1 bar", "abc", " ( 2.5e2 ) "])
    out = clip_range(s)
    exp = pd.Series([1200.0, -0.34, math.nan, -250.0])
    assert_float_series_equal(out, exp, atol=1e-12)

def test_inf_nan_strings_sign_policy():
    s = pd.Series(["NaN", "inf", "-Inf", "Infinity", "-infinity"])
    neg = enforce_sign(s, sign="negative")
    # keep only negatives; everything else -> NA
    exp = pd.Series([math.nan, math.nan, -math.inf, math.nan, -math.inf])
    # Special handling for infinities: compare manually
    assert len(neg) == len(exp)
    for a, b in zip(neg, exp):
        if pd.isna(b):
            assert pd.isna(a)
        else:
            assert math.isinf(a) and math.isinf(b) and (a < 0) == (b < 0)
