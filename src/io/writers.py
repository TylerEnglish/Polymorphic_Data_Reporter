from __future__ import annotations
from io import BytesIO
from typing import Any, Dict, Optional
import pandas as pd

from .storage import Storage

def write_any(
    storage: Storage,
    df: Any,
    path: str,
    *,
    fmt: Optional[str] = None,
    backend: str = "pandas",  # or "polars"
    csv_options: Optional[Dict] = None,
    json_lines: bool = False,
) -> None:
    """
    Write a DataFrame-like to the given path using the chosen backend.
    """
    csv_options = csv_options or {}
    fmt = (fmt or path.split(".")[-1]).lower()

    if backend == "polars":
        import polars as pl
        if not isinstance(df, pl.DataFrame):
            # Accept pandas and convert
            if isinstance(df, pd.DataFrame):
                df = pl.from_pandas(df)
            else:
                raise TypeError("Expected polars.DataFrame or pandas.DataFrame for backend='polars'.")

        if fmt in {"parquet", "pq"}:
            buf = BytesIO()
            df.write_parquet(buf)
            storage.write_bytes(path, buf.getvalue())
            return
        if fmt == "feather":
            buf = BytesIO()
            df.write_ipc(buf)
            storage.write_bytes(path, buf.getvalue())
            return
        if fmt in {"json", "ndjson"}:
            buf = BytesIO()
            if json_lines:
                df.write_ndjson(buf)
            else:
                df.write_json(buf)
            storage.write_bytes(path, buf.getvalue())
            return
        # csv default
        buf = BytesIO()
        df.write_csv(buf, **csv_options)
        storage.write_bytes(path, buf.getvalue())
        return

    # pandas default
    if not isinstance(df, pd.DataFrame):
        # try best-effort convert if it's polars
        try:
            import polars as pl
            if isinstance(df, pl.DataFrame):
                df = df.to_pandas()
        except Exception:
            pass
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected pandas.DataFrame (or convertible).")

    if fmt in {"parquet", "pq"}:
        buf = BytesIO()
        df.to_parquet(buf, index=False)
        storage.write_bytes(path, buf.getvalue())
        return
    if fmt == "feather":
        buf = BytesIO()
        df.to_feather(buf)
        storage.write_bytes(path, buf.getvalue())
        return
    if fmt in {"json", "ndjson"}:
        buf = BytesIO()
        df.to_json(buf, orient="records", lines=json_lines)
        storage.write_bytes(path, buf.getvalue())
        return
    # csv default
    buf = BytesIO()
    df.to_csv(buf, index=False, **csv_options)
    storage.write_bytes(path, buf.getvalue())
