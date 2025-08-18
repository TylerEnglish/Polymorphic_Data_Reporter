import pandas as pd
import numpy as np
import pytest

from src.cleaning.rules_builtin.date_parts import (
    extract_datetime_from_text,
    to_datetime_robust,
    dt_round,
    dt_floor,
    dt_ceil,
    dt_part,
)


# -----------------------
# Helper assertion utils
# -----------------------

def assert_series_equal_no_name(a: pd.Series, b: pd.Series):
    pd.testing.assert_series_equal(
        a.reset_index(drop=True),
        b.reset_index(drop=True),
        check_names=False,
    )


def assert_string_series_equal(a: pd.Series, b: pd.Series):
    assert a.dtype.name == "string"
    assert b.dtype.name == "string"
    assert_series_equal_no_name(a, b)


def assert_dt_series_equal(a: pd.Series, b: pd.Series):
    # Allow tz-aware or tz-naive depending on expectation; values must match.
    assert pd.api.types.is_datetime64_any_dtype(a.dtype)
    assert pd.api.types.is_datetime64_any_dtype(b.dtype)
    assert_series_equal_no_name(a, b)


# =========================
# extract_datetime_from_text
# =========================

def test_extract_datetime_from_text_various_patterns():
    s = pd.Series([
        "Order placed at 2021-07-01T12:34:56Z by user.",
        "Meeting on 2021/7/1 (rescheduled)",
        "Appt 7/01/2021 at noon",
        "Event Jan 2 2024 confirmed",
        "On 2 January 2024 we met",
        "Compact date: 20240102",
        "nothing here",
        None,
        ""
    ])

    out = extract_datetime_from_text(s)

    exp = pd.Series([
        "2021-07-01T12:34:56Z",  # ISO with Z
        "2021/7/1",              # Y/M/D
        "7/01/2021",             # M/D/Y (ambiguous)
        "Jan 2 2024",            # Month-name
        "2 January 2024",        # Day Month-name
        "2024-01-02",            # compact normalized to YYYY-MM-DD
        pd.NA,                   # no match
        pd.NA,                   # None -> NA
        pd.NA,                   # empty -> NA
    ], dtype="string")

    assert_string_series_equal(out, exp)


# ===============
# to_datetime_robust
# ===============

def test_to_datetime_robust_parses_normal_and_embedded():
    s = pd.Series([
        "2021-01-02 03:04:05",
        "foo 2021-03-04 bar",     # embedded ISO date only
        None,
        "not a date",
    ])

    out = to_datetime_robust(s)

    # Build expected explicitly (avoids per-element parsing heuristics)
    exp = pd.Series([
        pd.Timestamp("2021-01-02 03:04:05"),
        pd.Timestamp("2021-03-04"),
        pd.NaT,
        pd.NaT
    ], dtype="datetime64[ns]")

    assert_dt_series_equal(out, exp)


def test_to_datetime_robust_epoch_units_with_utc():
    # 2021-01-01 00:00:00 UTC in various epoch units
    sec = "1609459200"
    ms  = "1609459200000"
    us  = "1609459200000000"
    ns  = "1609459200000000000"

    s = pd.Series([sec, ms, us, ns])
    out = to_datetime_robust(s, utc=True)

    # Cast to integers for unit-based parsing (strings+unit are deprecated and may fail)
    exp = pd.Series([
        pd.to_datetime(int(sec), unit="s",  utc=True),
        pd.to_datetime(int(ms),  unit="ms", utc=True),
        pd.to_datetime(int(us),  unit="us", utc=True),
        pd.to_datetime(int(ns),  unit="ns", utc=True),
    ])

    assert_dt_series_equal(out, exp)


def test_to_datetime_robust_dayfirst_behavior():
    s = pd.Series(["03/04/2021", "04/03/2021"])
    # dayfirst=False (default): interpret as mm/dd/yyyy
    out_mdy = to_datetime_robust(s, dayfirst=False)
    exp_mdy = pd.to_datetime(pd.Series(["2021-03-04", "2021-04-03"]))
    assert_dt_series_equal(out_mdy, exp_mdy)

    # dayfirst=True: interpret as dd/mm/yyyy
    out_dmy = to_datetime_robust(s, dayfirst=True)
    exp_dmy = pd.to_datetime(pd.Series(["2021-04-03", "2021-03-04"]))
    assert_dt_series_equal(out_dmy, exp_dmy)


def test_to_datetime_robust_assume_utc_localize_naive():
    s = pd.Series(["2021-01-02 03:04:05", "2021-06-01 00:00:00"])
    out = to_datetime_robust(s, utc=True, assume_utc_if_tz_naive=True)
    # Both should be tz-aware UTC
    exp = pd.to_datetime(s, utc=True)
    assert_dt_series_equal(out, exp)


# ============
# Rounding APIs
# ============

def test_round_floor_ceil_basic():
    s = pd.Series(["2021-01-01 10:07:00", "2021-01-01 10:02:29"])
    out_round = dt_round(s, freq="5min")
    out_floor = dt_floor(s, freq="5min")
    out_ceil  = dt_ceil(s,  freq="5min")

    exp_round = pd.to_datetime(pd.Series(["2021-01-01 10:05:00", "2021-01-01 10:00:00"]))
    exp_floor = pd.to_datetime(pd.Series(["2021-01-01 10:05:00", "2021-01-01 10:00:00"]))
    exp_ceil  = pd.to_datetime(pd.Series(["2021-01-01 10:10:00", "2021-01-01 10:05:00"]))

    assert_dt_series_equal(out_round, exp_round)
    assert_dt_series_equal(out_floor, exp_floor)
    assert_dt_series_equal(out_ceil,  exp_ceil)


def test_round_invalid_freq_graceful():
    s = pd.Series(["2021-01-01 10:07:00", "not a date"])
    out = dt_round(s, freq="NOTAFREQ")
    # Should just return parsed datetimes unchanged (NaT for invalid)
    exp = to_datetime_robust(s)
    assert_dt_series_equal(out, exp)


# =========
# dt_part()
# =========

def test_dt_part_date_and_time_and_numeric_parts():
    s = pd.Series(["2021-03-05 06:07:08", "2020-12-31 23:59:59", None])
    # date (normalize to midnight)
    out_date = dt_part(s, part="date")
    exp_date = pd.to_datetime(pd.Series(["2021-03-05 00:00:00", "2020-12-31 00:00:00", pd.NaT]))
    assert_dt_series_equal(out_date, exp_date)

    # time (HH:MM:SS string)
    out_time = dt_part(s, part="time")
    exp_time = pd.Series(["06:07:08", "23:59:59", pd.NA], dtype="string")
    assert_string_series_equal(out_time, exp_time)

    # numeric parts
    assert_series_equal_no_name(dt_part(s, part="year"),      pd.Series([2021, 2020, pd.NA], dtype="Int64"))
    assert_series_equal_no_name(dt_part(s, part="quarter"),   pd.Series([1, 4, pd.NA], dtype="Int64"))
    assert_series_equal_no_name(dt_part(s, part="month"),     pd.Series([3, 12, pd.NA], dtype="Int64"))
    assert_series_equal_no_name(dt_part(s, part="day"),       pd.Series([5, 31, pd.NA], dtype="Int64"))
    assert_series_equal_no_name(dt_part(s, part="hour"),      pd.Series([6, 23, pd.NA], dtype="Int64"))
    assert_series_equal_no_name(dt_part(s, part="minute"),    pd.Series([7, 59, pd.NA], dtype="Int64"))
    assert_series_equal_no_name(dt_part(s, part="second"),    pd.Series([8, 59, pd.NA], dtype="Int64"))
    assert_series_equal_no_name(dt_part(s, part="dayofyear"), pd.Series([64, 366, pd.NA], dtype="Int64"))


def test_dt_part_names_and_weeklike():
    s = pd.Series(["2021-01-04", "2021-01-10", None])  # 2021-01-04 is Monday; 2021-01-10 is Sunday

    # Month & day names
    assert_string_series_equal(
        dt_part(s, part="month_name"),
        pd.Series(["January", "January", pd.NA], dtype="string"),
    )
    assert_string_series_equal(
        dt_part(s, part="day_name"),
        pd.Series(["Monday", "Sunday", pd.NA], dtype="string"),
    )

    # Weekday: Monday=0..Sunday=6
    assert_series_equal_no_name(
        dt_part(s, part="weekday"),
        pd.Series([0, 6, pd.NA], dtype="Int64"),
    )

    # ISO calendar components
    isocal = pd.to_datetime(s).dt.isocalendar()
    exp_iso_year = isocal.year.astype("Int64")
    exp_iso_week = isocal.week.astype("Int64")
    exp_iso_day  = isocal.day.astype("Int64")

    assert_series_equal_no_name(dt_part(s, part="iso_year"), exp_iso_year)
    assert_series_equal_no_name(dt_part(s, part="iso_week"), exp_iso_week)
    assert_series_equal_no_name(dt_part(s, part="iso_day"),  exp_iso_day)

    # "week" should align with ISO week in implementation (as coded)
    assert_series_equal_no_name(dt_part(s, part="week"), exp_iso_week)


def test_dt_part_unknown_part_returns_parsed_datetimes():
    s = pd.Series(["2021-03-05 06:07:08", "not a date"])
    out = dt_part(s, part="not_a_real_part")
    exp = to_datetime_robust(s)
    assert_dt_series_equal(out, exp)


def test_to_datetime_robust_handles_empty_and_none():
    s = pd.Series([None, "", "   "])
    out = to_datetime_robust(s)
    exp = pd.to_datetime(pd.Series([pd.NaT, pd.NaT, pd.NaT]))
    assert_dt_series_equal(out, exp)
