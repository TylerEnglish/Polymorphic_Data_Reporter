from __future__ import annotations
from typing import Iterable, List, Optional, Tuple
from datetime import datetime
import pandas as pd
import pytz

def parse_any_datetime(
    s: str,
    formats: Iterable[str] = ("%Y-%m-%d", "%m/%d/%Y", "%Y-%m-%d %H:%M:%S")
) -> Optional[datetime]:
    for f in formats:
        try:
            return datetime.strptime(s, f)
        except Exception:
            continue
    # pandas to_datetime as fallback
    try:
        dt = pd.to_datetime(s, errors="raise")
        if pd.isna(dt):
            return None
        return dt.to_pydatetime()
    except Exception:
        return None

def to_timezone(dt: datetime, tz: str) -> datetime:
    tzinfo = pytz.timezone(tz)
    if dt.tzinfo is None:
        return tzinfo.localize(dt)
    return dt.astimezone(tzinfo)

def infer_freq(ts: pd.Series) -> Optional[str]:
    # expects datetime64 series
    try:
        return pd.infer_freq(ts.sort_values())
    except Exception:
        return None

def is_regular_frequency(ts: pd.Series) -> bool:
    f = infer_freq(ts)
    return f is not None

def floor_period(dt: datetime, freq: str = "D") -> datetime:
    return pd.Timestamp(dt).floor(freq).to_pydatetime()

def ceil_period(dt: datetime, freq: str = "D") -> datetime:
    return pd.Timestamp(dt).ceil(freq).to_pydatetime()