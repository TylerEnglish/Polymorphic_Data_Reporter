from __future__ import annotations
import numpy as np
import pandas as pd

from src.cleaning.rules_builtin.units import standardize_numeric_units

def test_standardize_percent_strings_and_numbers_to_fraction():
    s = pd.Series(["10%", "25 %", "0.5", None, "NaN"])
    out, meta = standardize_numeric_units(s, unit_hint="percent", percent_target="fraction")
    assert meta["unit_out"] == "fraction"
    assert np.isclose(out.iloc[0], 0.10, equal_nan=False)
    assert np.isclose(out.iloc[1], 0.25, equal_nan=False)
    assert np.isclose(out.iloc[2], 0.50, equal_nan=False)
    assert pd.isna(out.iloc[3]) and pd.isna(out.iloc[4])

def test_standardize_percent_numeric_whole_numbers_to_fraction():
    s = pd.Series([10, 20, 55, np.nan])
    out, meta = standardize_numeric_units(s, unit_hint="percent", percent_target="fraction")
    assert meta["unit_out"] == "fraction" and meta["rescaled"] is True
    assert np.isclose(out.iloc[0], 0.10)
    assert np.isclose(out.iloc[1], 0.20)
    assert np.isclose(out.iloc[2], 0.55)
    assert pd.isna(out.iloc[3])

def test_standardize_percent_fraction_to_percent_target():
    s = pd.Series([0.1, 0.5, np.nan])
    out, meta = standardize_numeric_units(s, unit_hint="percent", percent_target="percent")
    assert meta["unit_out"] == "percent" and meta["rescaled"] is True
    assert np.isclose(out.iloc[0], 10.0)
    assert np.isclose(out.iloc[1], 50.0)
    assert pd.isna(out.iloc[2])

def test_currency_and_magnitude_are_parsed_but_not_rescaled():
    s = pd.Series(["$1,200", "3k", "(4,500)", "2 million", "1mm", None, "bogus"])
    out, meta = standardize_numeric_units(s, unit_hint=None)
    # currency/k/million normalized to base units by coercer
    assert np.isclose(out.iloc[0], 1200.0)
    assert np.isclose(out.iloc[1], 3000.0)
    assert np.isclose(out.iloc[2], -4500.0)
    assert np.isclose(out.iloc[3], 2_000_000.0)
    assert np.isclose(out.iloc[4], 1_000_000.0)
    assert pd.isna(out.iloc[5]) and pd.isna(out.iloc[6])
    assert meta["unit_in"] == meta["unit_out"] == "base"
    # purity
    assert out.dtype == float

def test_mixed_percent_strings_and_numeric_normalize_cleanly():
    s = pd.Series(["15%", 0.2, 50, "0.33", "pct", "percent", None])
    out, meta = standardize_numeric_units(s, unit_hint="percent", percent_target="fraction")
    # "pct"/"percent" are unparseable alone → NaN; others normalized
    assert np.isclose(out.iloc[0], 0.15)
    assert np.isclose(out.iloc[1], 0.20)
    assert np.isclose(out.iloc[2], 0.50)
    assert np.isclose(out.iloc[3], 0.33)
    assert pd.isna(out.iloc[4]) and pd.isna(out.iloc[5]) and pd.isna(out.iloc[6])
