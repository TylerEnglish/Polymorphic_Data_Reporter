from io import BytesIO
from pathlib import Path
import json
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.io.storage import StoreConfig, Storage
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

def test_read_csv_and_unknown_extension_defaults_to_csv(tmp_path: Path):
    st = _mk_storage(tmp_path)
    p_csv = st.raw_path("demo", "t.csv")
    st.write_bytes(p_csv, b"a,b\n1,2\n")
    df = read_any(st, p_csv, backend="pandas")
    assert list(df.columns) == ["a","b"]
    assert df.iloc[0].tolist() == [1,2]

    # Unknown extension -> defaults to CSV
    p_weird = st.raw_path("demo", "u.data")
    st.write_bytes(p_weird, b"x,y\n9,8\n")
    df2 = read_any(st, p_weird, backend="pandas")  # fmt inferred -> .csv
    assert df2.iloc[0].tolist() == [9,8]

def test_read_json_lines_and_normal_json(tmp_path: Path):
    st = _mk_storage(tmp_path)

    # NDJSON
    nd_path = st.raw_path("demo", "j.ndjson")
    nd = b'{"a":1}\n{"a":2}\n{"a":3}\n'
    st.write_bytes(nd_path, nd)
    df_lines = read_any(st, nd_path, backend="pandas", fmt="json", json_lines=True)
    assert df_lines.shape == (3,1)
    assert df_lines["a"].tolist() == [1,2,3]

    # JSON array
    arr_path = st.raw_path("demo", "j.json")
    arr = json.dumps([{"x": 10}, {"x": 20}]).encode()
    st.write_bytes(arr_path, arr)
    df_json = read_any(st, arr_path, backend="pandas", fmt="json", json_lines=False)
    assert df_json["x"].tolist() == [10,20]

def test_read_parquet_and_feather_pandas(tmp_path: Path):
    st = _mk_storage(tmp_path)
    df = pd.DataFrame({"a":[1,2,3], "b":["x","y","z"]})

    # parquet via pyarrow to bytes
    buf = BytesIO(); table = pa.Table.from_pandas(df); pq.write_table(table, buf)
    p_parq = st.raw_path("demo", "d.parquet")
    st.write_bytes(p_parq, buf.getvalue())
    got_parq = read_any(st, p_parq, backend="pandas")
    pd.testing.assert_frame_equal(df, got_parq)

    # feather
    fbuf = BytesIO(); df.to_feather(fbuf)
    p_feat = st.raw_path("demo", "d.feather")
    st.write_bytes(p_feat, fbuf.getvalue())
    got_feat = read_any(st, p_feat, backend="pandas")
    pd.testing.assert_frame_equal(df, got_feat)

@pytest.mark.parametrize("fmt", ["csv","json","parquet","feather"])
def test_polars_backend_reads(tmp_path: Path, fmt: str):
    st = _mk_storage(tmp_path)
    df = pd.DataFrame({"k":[1,2], "v":[3,4]})

    # write bytes per fmt
    if fmt == "csv":
        st.write_bytes(st.raw_path("demo","p.csv"), df.to_csv(index=False).encode())
        path = st.raw_path("demo","p.csv")
    elif fmt == "json":
        st.write_bytes(st.raw_path("demo","p.json"), df.to_json(orient="records").encode())
        path = st.raw_path("demo","p.json")
    elif fmt == "parquet":
        buf = BytesIO(); df.to_parquet(buf, index=False)
        path = st.raw_path("demo","p.parquet"); st.write_bytes(path, buf.getvalue())
    else:  # feather
        buf = BytesIO(); df.to_feather(buf)
        path = st.raw_path("demo","p.feather"); st.write_bytes(path, buf.getvalue())

    out = read_any(st, path, backend="polars")
    # convert to pandas for equality check
    try:
        pdf = out.to_pandas()
    except Exception:
        import polars as pl
        assert isinstance(out, pl.DataFrame)
        pdf = out.to_pandas()
    # JSON roundtrip may reorder columns; sort
    pd.testing.assert_frame_equal(
        df.sort_index(axis=1), pdf.sort_index(axis=1), check_dtype=False
    )

def test_invalid_backend_raises(tmp_path: Path):
    st = _mk_storage(tmp_path)
    p = st.raw_path("demo","x.csv")
    st.write_bytes(p, b"a\n1\n")
    with pytest.raises(ValueError):
        read_any(st, p, backend="duck")  # unsupported
