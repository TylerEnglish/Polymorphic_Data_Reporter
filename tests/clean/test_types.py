from __future__ import annotations
import numpy as np
import pandas as pd
import pytest

from src.cleaning.rules_builtin.types import (
    coerce_numeric_from_string,
    parse_datetime_from_string,
    cast_category_if_small,
)

def _nan_equal(a, b) -> bool:
    # helper: treats NaN == NaN
    if isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b):
        return True
    return a == b

def test_coerce_numeric_currency_percent_parens_k_million_and_edge_tokens():
    s = pd.Series([
        "$1,234",      # currency + comma
        "€2,000",      # other currency
        "15%",         # percent sign
        "(3.5)",       # parentheses negative
        "12k",         # k suffix
        "2 million",   # million word
        "1mm",         # mm suffix
        "−42",         # unicode minus
        "-1_234",      # underscore sep + minus
        "  +5  ",      # plus sign + spaces
        None, "", "abc"  # missing / bad
    ])
    out = coerce_numeric_from_string(s)
    expected = [
        1234.0,
        2000.0,
        0.15,
        -3.5,
        12000.0,
        2_000_000.0,
        1_000_000.0,
        -42.0,
        -1234.0,
        5.0,
        np.nan, np.nan, np.nan
    ]
    assert list(map(type, out.tolist()))  # sanity: convertible
    assert all(_nan_equal(a, b) for a, b in zip(out.tolist(), expected))
    # dtype should be float
    assert pd.api.types.is_float_dtype(out.dtype)
    # pure: ensure a new object is returned
    assert out is not s

@pytest.mark.parametrize(
    "vals,expected",
    [
        (["12K", "7 thousand", "3M", "4 mm", "5 Million"], [12000.0, 7000.0, 3_000_000.0, 4_000_000.0, 5_000_000.0]),
        (["10 pct", "25 PERCENT"], [0.10, 0.25]),
    ],
)
def test_coerce_numeric_magnitude_and_percent_words(vals, expected):
    s = pd.Series(vals)
    out = coerce_numeric_from_string(s)
    assert all(_nan_equal(a, b) for a, b in zip(out.tolist(), expected))

def test_coerce_numeric_pass_through_numeric_dtype():
    s = pd.Series([1, 2.5, np.nan], dtype=float)
    out = coerce_numeric_from_string(s)
    assert out.tolist()[0] == 1.0 and out.tolist()[1] == 2.5 and np.isnan(out.tolist()[2])
    assert pd.api.types.is_float_dtype(out.dtype)

def test_parse_datetime_with_explicit_formats_then_fallback():
    s = pd.Series(["01/02/2024", "2024-03-05 12:34:56", "not a date"])
    formats = ["%m/%d/%Y"]  # first value requires this; others use fallback
    out = parse_datetime_from_string(s, formats)
    assert pd.api.types.is_datetime64_any_dtype(out.dtype)
    assert out.iloc[0] == pd.Timestamp(2024, 1, 2)
    assert out.iloc[1] == pd.Timestamp(2024, 3, 5, 12, 34, 56)
    assert pd.isna(out.iloc[2])
    # pure: returns a new series
    assert out is not s

def test_parse_datetime_idempotent_on_datetime_dtype():
    base = pd.Series(pd.to_datetime(["2024-01-01", "2024-01-02"]))
    out = parse_datetime_from_string(base, ["%Y-%m-%d"])
    # values preserved
    pd.testing.assert_series_equal(out.reset_index(drop=True), base.reset_index(drop=True))

def test_cast_category_if_small_thresholds():
    small = pd.Series(["a"] * 10 + ["b"] * 5 + ["c"] * 2)
    big = pd.Series([f"v{i}" for i in range(300)])

    cast_small = cast_category_if_small(small, max_card=200)
    cast_big = cast_category_if_small(big, max_card=200)

    assert str(cast_small.dtype) == "category"
    # big should remain object (or its original dtype) because cardinality > max
    assert str(cast_big.dtype) != "category"
    # purity: return new objects
    assert cast_small is not small
    assert cast_big is not big

def test_cast_category_exact_boundary():
    # exactly max_card distinct values → should cast
    vals = [f"x{i}" for i in range(20)]
    s = pd.Series(vals)
    out = cast_category_if_small(s, max_card=20)
    assert str(out.dtype) == "category"
