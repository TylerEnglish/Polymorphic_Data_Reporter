from io import BytesIO
from pathlib import Path
import json
import pandas as pd
import pytest

from src.io.storage import StoreConfig, Storage
from src.io.writers import write_any
from src.io.readers import read_any

def _mk_storage(tmp_path: Path) -> Storage:
    return Storage(StoreConfig(
        local_raw_root=str(tmp_path / "data" / "raw"),
        local_gold_root=str(tmp_path / "data" / "gold"),
        s3_enabled=False,
        s3_endpoint=None,
        s3_bucket=None,
        s3_access_key=None,
        s3_secret_key=None,
        s3_secure=False,
        s3_raw_prefix=None,
        s3_gold_prefix=None,
    ))

def test_write_read_pandas_all_formats(tmp_path: Path):
    st = _mk_storage(tmp_path)
    df = pd.DataFrame({"a":[1,2], "b":["x","y"]})

    # CSV
    p_csv = st.gold_path("demo","t.csv")
    write_any(st, df, p_csv, fmt="csv")
    got_csv = read_any(st, p_csv, backend="pandas")
    pd.testing.assert_frame_equal(df, got_csv)

    # JSON array (json_lines=False)
    p_json = st.gold_path("demo","t.json")
    write_any(st, df, p_json, fmt="json", json_lines=False)
    got_json = read_any(st, p_json, backend="pandas", fmt="json", json_lines=False)
    pd.testing.assert_frame_equal(df, got_json)

    # NDJSON (json_lines=True)
    p_ndjson = st.gold_path("demo","t.ndjson")
    write_any(st, df, p_ndjson, fmt="json", json_lines=True)
    got_nd = read_any(st, p_ndjson, backend="pandas", fmt="json", json_lines=True)
    pd.testing.assert_frame_equal(df, got_nd)

    # Parquet
    p_parq = st.gold_path("demo","t.parquet")
    write_any(st, df, p_parq, fmt="parquet")
    got_parq = read_any(st, p_parq, backend="pandas")
    pd.testing.assert_frame_equal(df, got_parq)

    # Feather
    p_feather = st.gold_path("demo","t.feather")
    write_any(st, df, p_feather, fmt="feather")
    got_feat = read_any(st, p_feather, backend="pandas")
    pd.testing.assert_frame_equal(df, got_feat)

def test_write_polars_from_pandas_and_read_back(tmp_path: Path):
    st = _mk_storage(tmp_path)
    df = pd.DataFrame({"k":[1,2,3], "v":[9,8,7]})

    # Use polars backend while passing pandas DataFrame -> auto-convert
    p_parq = st.gold_path("demo","p.parquet")
    write_any(st, df, p_parq, fmt="parquet", backend="polars")
    got = read_any(st, p_parq, backend="pandas")
    pd.testing.assert_frame_equal(df, got)

def test_write_polars_strict_type(tmp_path: Path):
    st = _mk_storage(tmp_path)
    # Passing an unsupported object for pandas backend should error
    with pytest.raises(TypeError):
        write_any(st, {"not": "a dataframe"}, st.gold_path("x","bad.csv"), fmt="csv")

def test_json_lines_writes_newline_delimited(tmp_path: Path):
    st = _mk_storage(tmp_path)
    df = pd.DataFrame({"a":[1,2]})
    p = st.gold_path("demo","lines.ndjson")
    write_any(st, df, p, fmt="json", json_lines=True)
    raw = st.read_bytes(p).decode("utf-8").strip().splitlines()
    assert all(line.startswith("{") and line.endswith("}") for line in raw)
    assert len(raw) == 2

def test_writers_respect_csv_options(tmp_path: Path):
    st = _mk_storage(tmp_path)
    df = pd.DataFrame({"a":[1,2], "b":[3,4]})
    p = st.gold_path("demo","opts.csv")
    write_any(st, df, p, fmt="csv", csv_options={"sep":";","header":True})
    txt = st.read_bytes(p).decode("utf-8").splitlines()
    assert txt[0] == "a;b"
    assert "1;3" in txt[1]
