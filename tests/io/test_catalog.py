from pathlib import Path
from src.io.storage import StoreConfig, Storage
from src.io.catalog import Catalog, DATA_FILE_RE

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

def test_data_file_regex_covers_expected_extensions():
    ok = [
        "a.csv","b.json","c.ndjson","d.parquet","e.pq","f.feather","g.txt",
        "ANY.PARQUET","WeIrD.FeAtHeR"
    ]
    bad = ["h.xlsx","i.md","j.bin","k.png","l.parq", "noext"]
    for p in ok:
        assert DATA_FILE_RE.match(p)
    for p in bad:
        assert not DATA_FILE_RE.match(p)

def test_local_inventory_full_and_scoped(tmp_path: Path):
    st = _mk_storage(tmp_path)
    cat = Catalog(st)

    # Create nested structure
    files = [
        st.raw_path("sales_demo", "part1", "a.csv"),
        st.raw_path("sales_demo", "part2", "b.json"),
        st.raw_path("iot_sensors", "2024", "c.parquet"),
        st.raw_path("iot_sensors", "ignore", "readme.md"),  # should be ignored
    ]
    for p in files:
        if p.endswith(".md"):
            st.write_bytes(p, b"# ignore")
        else:
            st.write_bytes(p, b"data")

    # Full inventory (infers slug by folder under raw/)
    inv = cat.inventory()
    got = {(e.slug, Path(e.uri).suffix.lower()) for e in inv}
    assert ("sales_demo", ".csv") in got
    assert ("sales_demo", ".json") in got
    assert ("iot_sensors", ".parquet") in got
    # markdown ignored
    assert all(".md" not in suff for _, suff in got)

    # Scoped to specific dataset
    inv_sales = cat.inventory(dataset_slug="sales_demo")
    slugs = {e.slug for e in inv_sales}
    assert slugs == {"sales_demo"}
    kinds = {e.kind for e in inv_sales}
    assert kinds == {"csv","json"}
