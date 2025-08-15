from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.testing as tm

from src.cleaning.rules_builtin.datetime_ops import parse_epoch_auto


def test_parse_epoch_auto_infers_seconds():
    # 2023-01-01 00:00:00 and +1s in epoch SECONDS
    s = pd.Series([1672531200, None, "1672531201", "abc"], dtype=object)
    out = parse_epoch_auto(s)
    exp = pd.to_datetime(
        pd.Series([1672531200, np.nan, 1672531201, np.nan]),
        unit="s",
        errors="coerce",
    )
    tm.assert_series_equal(out, exp)


def test_parse_epoch_auto_infers_milliseconds():
    # 2023-01-01 00:00:00 and +1s in epoch MILLISECONDS
    s = pd.Series([1672531200000, 1672531201000], dtype="int64")
    out = parse_epoch_auto(s)
    exp = pd.to_datetime(pd.Series([1672531200000, 1672531201000]), unit="ms", errors="coerce")
    tm.assert_series_equal(out, exp)


def test_parse_epoch_auto_infers_nanoseconds():
    # 2023-01-01 00:00:00 in epoch NANOSECONDS
    s = pd.Series([1672531200000000000], dtype="int64")
    out = parse_epoch_auto(s)
    exp = pd.to_datetime(pd.Series([1672531200000000000]), unit="ns", errors="coerce")
    tm.assert_series_equal(out, exp)


def test_parse_epoch_auto_respects_explicit_unit_override():
    # Provide explicit unit so heuristic is bypassed
    ms_vals = pd.Series([1672531200000, 1672531200500], dtype="int64")
    out = parse_epoch_auto(ms_vals, unit="ms")
    exp = pd.to_datetime(pd.Series([1672531200000, 1672531200500]), unit="ms", errors="coerce")
    tm.assert_series_equal(out, exp)


def test_parse_epoch_auto_invalid_values_and_dtype():
    s = pd.Series(["not_a_number", "", None])
    out = parse_epoch_auto(s)
    # All invalid => all NaT, dtype must be datetime64[ns]
    assert out.dtype == "datetime64[ns]"
    assert out.isna().all()


def test_parse_epoch_auto_empty_series():
    s = pd.Series([], dtype="float64")
    out = parse_epoch_auto(s)
    # Empty but should still be datetime dtype
    assert str(out.dtype) == "datetime64[ns]"
    assert len(out) == 0
