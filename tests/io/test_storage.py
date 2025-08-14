import os
from pathlib import Path
import io
import pytest

from src.io.storage import StoreConfig, Storage

def _mk_storage(tmp_path: Path) -> Storage:
    cfg = StoreConfig(
        local_raw_root=str(tmp_path / "data" / "raw"),
        local_gold_root=str(tmp_path / "data" / "gold"),
        s3_enabled=False,           # keep disabled; we’ll fake S3 below
        s3_endpoint=None,
        s3_bucket=None,
        s3_access_key=None,
        s3_secret_key=None,
        s3_secure=False,
        s3_raw_prefix=None,
        s3_gold_prefix=None,
    )
    return Storage(cfg)

def test_path_helpers_and_roundtrip_bytes(tmp_path: Path):
    st = _mk_storage(tmp_path)

    # raw_path / gold_path join correctly
    raw_csv = st.raw_path("sales_demo", "folder", "sample.csv")
    gold_parquet = st.gold_path("sales_demo", "tables", "data.parquet")
    assert raw_csv.endswith(os.path.join("sales_demo", "folder", "sample.csv"))
    assert gold_parquet.endswith(os.path.join("sales_demo", "tables", "data.parquet"))

    # write/read/exists
    payload = b"a,b\n1,2\n"
    st.write_bytes(raw_csv, payload)
    assert st.exists(raw_csv)
    assert st.read_bytes(raw_csv) == payload

def test_local_open_context_manager(tmp_path: Path):
    st = _mk_storage(tmp_path)
    p = st.raw_path("demo", "file.txt")

    with st.open(p, "wb") as f:
        f.write(b"hello")

    with st.open(p, "rb") as f:
        data = f.read()
    assert data == b"hello"

def test_ls_variants(tmp_path: Path):
    st = _mk_storage(tmp_path)

    # non-existent returns []
    base = st.raw_path("demo")
    assert st.ls(base) == []

    # nested files
    p1 = st.raw_path("demo", "a", "x.csv")
    p2 = st.raw_path("demo", "b", "y.json")
    p3 = st.raw_path("demo", "b", "deep", "z.parquet")
    for p in (p1, p2, p3):
        st.write_bytes(p, b"data")

    out = st.ls(base)
    assert set(map(Path, out)) == {Path(p1), Path(p2), Path(p3)}

    # ls on a file returns [file]
    assert st.ls(p1) == [p1]

def test_file_scheme_local(tmp_path: Path):
    st = _mk_storage(tmp_path)
    # “file://” scheme should be accepted by LocalStore
    p = "file://" + st.raw_path("scheme", "ok.csv")
    st.write_bytes(p, b"x\n1\n")
    assert st.exists(p)
    assert st.read_bytes(p).startswith(b"x")

def test_s3_backend_routing_with_fake_store(tmp_path: Path):
    """
    We don't rely on real s3fs. Instead inject a fake backend and assert routing.
    """
    st = _mk_storage(tmp_path)

    class FakeS3:
        def __init__(self):
            self.written = {}
            self.reads = {}
            self.listed = set()
            self._opened = {}

        def exists(self, path: str) -> bool:
            return path in self.written

        def ls(self, prefix: str):
            # Return all paths starting with prefix
            self.listed.add(prefix)
            return [k for k in self.written if k.startswith(prefix)]

        def read_bytes(self, path: str) -> bytes:
            self.reads[path] = self.reads.get(path, 0) + 1
            return self.written[path]

        def write_bytes(self, path: str, data: bytes) -> None:
            self.written[path] = data

        def open(self, path: str, mode: str = "rb"):
            if "w" in mode:
                buf = io.BytesIO()
                def _close_write():
                    self.written[path] = buf.getvalue()
                return _ClosableBuffer(buf, _close_write)
            else:
                return io.BytesIO(self.written.get(path, b""))

    class _ClosableBuffer:
        def __init__(self, buf, on_close):
            self._b = buf
            self._on_close = on_close
        def write(self, *a, **k): return self._b.write(*a, **k)
        def read(self, *a, **k): return self._b.read(*a, **k)
        def __enter__(self): return self
        def __exit__(self, exc_type, exc, tb): self._on_close()

    # Inject fake backend and call with s3:// URI
    fake = FakeS3()
    # bypass constructor S3 checks by setting the attribute directly
    st._s3 = fake  # type: ignore[attr-defined]

    s3_uri = "s3://bucket/raw/demo/file.csv"
    st.write_bytes(s3_uri, b"a,b\n1,2\n")
    assert st.exists(s3_uri)
    assert st.read_bytes(s3_uri).startswith(b"a,b")
    assert s3_uri in fake.written

def test_s3_path_helpers_raise_when_disabled(tmp_path: Path):
    st = _mk_storage(tmp_path)
    with pytest.raises(RuntimeError, match="S3 raw root requested"):
        st.raw_path("x", s3=True)
    with pytest.raises(RuntimeError, match="S3 gold root requested"):
        st.gold_path("x", s3=True)
