from __future__ import annotations
from io import BytesIO, StringIO
from typing import Any, Dict, Optional, Tuple
import json
import pandas as pd
import pyarrow.parquet as pq  # fast path for parquet
# polars is optional at call-site; we keep pandas default to reduce surprises

from .storage import Storage

_EXTS = (".csv", ".json", ".ndjson", ".parquet", ".pq", ".feather", ".arrows", ".txt")

def _infer_ext(path: str) -> str:
    p = path.lower()
    for e in _EXTS:
        if p.endswith(e):
            return e
    # default to csv if unknown
    return ".csv"

def read_any(
    storage: Storage,
    path: str,
    *,
    fmt: Optional[str] = None,
    backend: str = "pandas",  # or "polars"
    csv_options: Optional[Dict[str, Any]] = None,
    json_lines: bool = True,
) -> Any:
    """
    Read a dataset into a DataFrame-like object using the selected backend.
    - storage: Storage facade
    - path: local path or s3://bucket/key
    - fmt: optional override ("csv","json","parquet","feather")
    - backend: "pandas" (default) or "polars"
    """
    csv_options = csv_options or {}
    fmt = (fmt or _infer_ext(path)).lstrip(".").lower()

    if backend not in {"pandas", "polars"}:
        raise ValueError("backend must be 'pandas' or 'polars'")

    data = storage.read_bytes(path)

    if backend == "polars":
        import polars as pl  # local import
        if fmt in {"parquet", "pq"}:
            return pl.read_parquet(BytesIO(data))
        if fmt == "feather":
            return pl.read_ipc(BytesIO(data))
        if fmt in {"json", "ndjson"}:
            # Heuristic: if first non-space is '[' or '{', treat as JSON array/object, not NDJSON.
            head = data.lstrip()[:1]
            looks_like_array_or_object = head in (b"[", b"{")
            if json_lines and not looks_like_array_or_object:
                # Try native NDJSON; on failure fall back to pandas->polars.
                try:
                    return pl.read_ndjson(BytesIO(data))
                except Exception:
                    pass
            # JSON array/object fallback via pandas -> polars for robustness across versions
            import pandas as _pd  # alias to avoid shadowing
            pdf = _pd.read_json(BytesIO(data))
            return pl.from_pandas(pdf)
        # csv / txt default
        return pl.read_csv(BytesIO(data), **csv_options)

    # pandas
    if fmt in {"parquet", "pq"}:
        # Faster via pyarrow directly from bytes
        return pq.read_table(BytesIO(data)).to_pandas()
    if fmt == "feather":
        return pd.read_feather(BytesIO(data))
    if fmt in {"json", "ndjson"}:
        if json_lines:
            return pd.read_json(BytesIO(data), lines=True)
        return pd.read_json(BytesIO(data))
    # csv / txt default
    return pd.read_csv(BytesIO(data), **csv_options)
