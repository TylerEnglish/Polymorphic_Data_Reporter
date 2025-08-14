from __future__ import annotations
import json
from io import BytesIO
from pathlib import Path
import pandas as pd

from src.io.storage import Storage, StoreConfig
from src.nlp.bootstrap import run_nlp_bootstrap, propose_schema_for_df
from src.nlp.schema import ProposedSchema

def _mk_storage(tmp_path: Path) -> Storage:
    cfg = StoreConfig(
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
    )
    st = Storage(cfg)
    (tmp_path / "data" / "raw").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data" / "gold").mkdir(parents=True, exist_ok=True)
    return st

def test_propose_schema_for_df_shapes_and_confidence():
    df = pd.DataFrame({
        "order_id":[1,2,3],
        "amount":[10.0,20.0,30.5],
        "when": pd.to_datetime(["2024-01-01","2024-01-02","2024-01-03"]),
        "note": ["Alpha","alpha ","ALPHA"]
    })
    ps = propose_schema_for_df("sales_demo", df)
    assert isinstance(ps, ProposedSchema)
    assert ps.dataset_slug == "sales_demo"
    assert len(ps.columns) == 4
    # confidence within bounds
    assert 0.0 <= ps.schema_confidence <= 1.0

def test_run_nlp_bootstrap_end_to_end_csv_json_parquet(tmp_path: Path):
    st = _mk_storage(tmp_path)
    slug = "sales_demo"
    # Prepare raw files
    raw_dir = Path(st.raw_path(slug))
    (raw_dir / "sub").mkdir(parents=True, exist_ok=True)

    # CSV
    pd.DataFrame({"id":[1,2,3],"amount":[10,20,30]}).to_csv(raw_dir / "a.csv", index=False)

    # JSON (array form)
    pd.DataFrame({"flag":[True, False, True]}).to_json(raw_dir / "sub" / "b.json", orient="records")

    # Parquet
    buf = BytesIO()
    pd.DataFrame({"ts": pd.to_datetime(["2024-01-01","2024-01-02"])}).to_parquet(buf, index=False)
    (raw_dir / "sub" / "c.parquet").write_bytes(buf.getvalue())

    out = run_nlp_bootstrap(st, slug, project_root=tmp_path)

    # IO paths
    schema_path = Path(out["io"]["schema_toml_path"])
    assert schema_path.parent.name == "schemas"
    assert schema_path.name == f"{slug}.schema.toml"
    assert "bootstrap_json_path" in out["io"]

    # bootstrap content sanity
    bs = out["bootstrap"]
    assert bs["dataset"] == slug
    assert bs["row_count"] > 0
    assert bs["column_count"] >= 1
    assert isinstance(bs["schema_confidence"], float)

    # columns have metrics + suggestions
    assert isinstance(bs["columns"], list) and len(bs["columns"]) >= 1
    c0 = bs["columns"][0]
    assert "metrics" in c0 and isinstance(c0["metrics"], dict)
    assert "non_null_ratio" in c0["metrics"]
    assert 0.0 <= c0["metrics"]["non_null_ratio"] <= 1.0
    assert "suggestions" in c0 and isinstance(c0["suggestions"], dict)

    # entries match discovered files (md ignored)
    uris = [e["uri"] for e in out["entries"]]
    assert any(u.endswith("a.csv") for u in uris)
    assert any(u.endswith("b.json") for u in uris)
    assert any(u.endswith("c.parquet") for u in uris)

def test_run_nlp_bootstrap_empty_dataset(tmp_path: Path):
    st = _mk_storage(tmp_path)
    slug = "empty_slug"
    (Path(st.raw_path(slug))).mkdir(parents=True, exist_ok=True)

    out = run_nlp_bootstrap(st, slug, project_root=tmp_path)
    assert out["bootstrap"]["row_count"] == 0
    assert out["bootstrap"]["column_count"] == 0
    assert out["bootstrap"]["schema_confidence"] == 0.0
    assert out["proposed_schema"].columns == []

def test_run_nlp_bootstrap_respects_sample_rows(tmp_path: Path):
    st = _mk_storage(tmp_path)
    slug = "sampled"
    raw_dir = Path(st.raw_path(slug))
    raw_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"k": list(range(100)), "v": list(range(100))})
    df.to_csv(raw_dir / "big.csv", index=False)

    out = run_nlp_bootstrap(st, slug, project_root=tmp_path, nlp_cfg={"sample_rows": 10})
    assert out["bootstrap"]["row_count"] <= 10
