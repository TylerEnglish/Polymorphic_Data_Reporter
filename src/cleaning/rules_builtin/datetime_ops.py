from __future__ import annotations
import pandas as pd
import numpy as np

def parse_epoch_auto(s: pd.Series, unit: str | None = None) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    med = float(x.dropna().abs().median()) if x.notna().any() else np.nan

    # crude but effective unit guess
    if unit is None:
        if med > 1e16: u = "ns"   # 1.67e18 ~ 2023 in ns
        elif med > 1e12: u = "ms"
        else: u = "s"
    else:
        u = unit

    dt = pd.to_datetime(x, unit=u, errors="coerce")
    return dt