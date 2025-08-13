import datetime as dt
import pandas as pd
from src.utils.time import parse_any_datetime, to_timezone, infer_freq, is_regular_frequency, floor_period, ceil_period

def test_parse_any_datetime_formats_and_fallback():
    assert parse_any_datetime("2024-01-31").date().isoformat() == "2024-01-31"
    assert parse_any_datetime("01/31/2024").date().isoformat() == "2024-01-31"
    assert parse_any_datetime("2024-01-31 12:34:56").isoformat().startswith("2024-01-31T12:34:56")
    assert parse_any_datetime("not a date") is None

def test_timezone_conversion_localize_and_convert():
    naive = dt.datetime(2024, 1, 1, 12, 0, 0)
    central = to_timezone(naive, "America/Chicago")
    assert central.tzinfo is not None
    already = central.astimezone()
    # round-trip convert
    back = to_timezone(already, "UTC")
    assert back.tzinfo is not None

def test_infer_and_regular_frequency():
    s = pd.Series(pd.date_range("2024-01-01", periods=10, freq="D"))
    assert infer_freq(s) in ("D", "C")  # pandas might return business day for some cases
    assert is_regular_frequency(s) is True

    s2 = pd.Series(pd.to_datetime(["2024-01-01","2024-01-02","2024-01-04"]))
    assert is_regular_frequency(s2) is False

def test_floor_ceil_period():
    t = dt.datetime(2024, 1, 1, 12, 34, 56)
    assert floor_period(t, "h").minute == 0   # lower-case h avoids deprecation
    assert ceil_period(t, "h").minute in (0,)