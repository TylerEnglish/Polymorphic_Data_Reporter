import json
from types import SimpleNamespace

import pandas as pd
import numpy as np
import pytest

from src.gold.materialize import build_gold_from_silver


def _dummy_cfg():
    # Minimal duck-typed config with the fields materialize uses
    return SimpleNamespace(
        env=SimpleNamespace(theme="dark_blue"),
        charts=SimpleNamespace(export_static_png=False, max_per_topic=2),
        reports=SimpleNamespace(
            enabled_generators=SimpleNamespace(charts=False),  # disable chart rendering in tests
            html=SimpleNamespace(title_prefix="Report:")
        ),
    )


@pytest.fixture
def silver_fixture(tmp_path, monkeypatch):
    """
    Sets up a temp working dir with a silver parquet and stubs all external
    deps used by build_gold_from_silver so the test stays fast and deterministic.
    """
    # --- Arrange: working dir and dataset
    slug = "toy"
    monkeypatch.chdir(tmp_path)

    # Simple dataset with date + metric; also include a category string col to ensure no surprises
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    df = pd.DataFrame({
        "date": dates.repeat(2),     # duplicate each day twice to test grouping
        "y":   np.tile([1, 2], 5),   # mean per day = 1.5
        "grp": ["A", "B"] * 5,
    })

    silver_dir = tmp_path / "data" / "silver" / slug
    silver_dir.mkdir(parents=True, exist_ok=True)
    (silver_dir / "dataset.parquet").write_bytes(b"")  # placeholder to ensure path exists
    df.to_parquet(silver_dir / "dataset.parquet", index=False)

    # --- Stub storage to write into tmp_path/gold
    storage = SimpleNamespace(gold_path=lambda s: str((tmp_path / "gold" / s)))

    def _build_storage_from_config(_cfg):
        return storage

    monkeypatch.setattr("src.gold.materialize.build_storage_from_config", _build_storage_from_config)

    # --- Stub writers.write_any to write directly to filesystem (parquet/json already handled in code)
    def _write_any(_storage, obj, path, fmt: str):
        if fmt == "parquet":
            assert isinstance(obj, pd.DataFrame)
            pd.DataFrame(obj).to_parquet(path, index=False)
        else:
            raise AssertionError("This test only expects parquet writes")

    monkeypatch.setattr("src.gold.materialize.write_any", _write_any)

    # --- Stub schema + recheck to avoid I/O and keep df intact
    def _load_schema(_slug, _root):
        return SimpleNamespace()

    def _recheck_silver(_cfg, _slug, write_dataset=False):
        return df.copy(), {}

    monkeypatch.setattr("src.gold.materialize._load_frozen_schema", _load_schema)
    monkeypatch.setattr("src.gold.materialize.recheck_silver", _recheck_silver)

    # --- Stub topics pipeline to produce two simple topics (trend + kpi)
    topics = pd.DataFrame([
        dict(
            topic_id="t_trend",
            family="trend",
            primary_fields=json.dumps(["y"]),
            secondary_fields=json.dumps([]),
            time_field="date",
            proposed_charts="[]",
            coverage_pct=1.0,
            effect_size=0.0,
            significance=json.dumps({}),
            n_obs=len(df),
            score_total=0.9,
        ),
        dict(
            topic_id="t_kpi",
            family="kpi",
            primary_fields=json.dumps(["y"]),
            secondary_fields=json.dumps([]),
            time_field=None,
            proposed_charts=None,
            coverage_pct=1.0,
            effect_size=0.0,
            significance=None,
            n_obs=len(df),
            score_total=0.5,
        ),
    ])

    # candidates -> score -> select are all stubbed to pass this through
    monkeypatch.setattr("src.gold.materialize.build_candidates", lambda _df, _schema, _cfg: topics.copy())
    monkeypatch.setattr("src.gold.materialize.score_topics", lambda t, _cfg: t)
    monkeypatch.setattr("src.gold.materialize.select_topics", lambda t, _cfg: t)

    # Stub layout planner to a minimal plan; build_gold_from_silver writes it as JSON
    def _make_layout_plan(selected, cfg, *, dataset_slug: str):
        return {
            "report_title": f"{cfg.reports.html.title_prefix} {dataset_slug}",
            "dataset_slug": dataset_slug,
            "generated_at": pd.Timestamp.utcnow().isoformat(),
            "sections": [
                {"topic_ref": r["topic_id"], "components": [{"export": {"html": True, "png": True}}]}
                for _, r in selected.sort_values("score_total", ascending=False).iterrows()
            ],
        }

    monkeypatch.setattr("src.gold.materialize.make_layout_plan", _make_layout_plan)

    return slug, df


def test_build_gold_from_silver_writes_artifacts_and_components(tmp_path, silver_fixture):
    slug, df = silver_fixture
    cfg = _dummy_cfg()

    # Act
    paths = build_gold_from_silver(cfg, slug, recheck=True)

    # Assert: top-level returned paths exist
    for k in ("topics", "topics_selected", "layout_plan", "manifest"):
        p = paths[k]
        assert isinstance(p, str) and (tmp_path / p).exists()

    # Load manifest to inspect component outputs
    manifest_path = tmp_path / paths["manifest"]
    with open(manifest_path, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)

    assert manifest["dataset_slug"] == slug
    comps = manifest["components"]
    assert isinstance(comps, list) and len(comps) == 2

    # No charts should be generated because charts are disabled in cfg
    assert manifest.get("charts") == []

    # Each component path should exist; inspect shapes/content
    comp_paths = [tmp_path / c["path"] for c in comps]
    for cp in comp_paths:
        assert cp.exists()

    # Identify which is trend vs kpi by family in manifest
    by_topic = {c["topic_id"]: c for c in comps}
    trend_cp = tmp_path / by_topic["t_trend"]["path"]
    kpi_cp = tmp_path / by_topic["t_kpi"]["path"]

    trend_df = pd.read_parquet(trend_cp)
    kpi_df = pd.read_parquet(kpi_cp)

    # Trend: grouped by day with a "value" column equal to daily mean of y (=1.5)
    assert "date" in trend_df.columns and "value" in trend_df.columns
    assert len(trend_df) == df["date"].nunique()
    assert pytest.approx(trend_df["value"].iloc[0], rel=1e-6) == 1.5

    # KPI: single-row with mean(y)
    assert list(kpi_df.columns) == ["metric_name", "value"]
    assert len(kpi_df) == 1
    assert kpi_df["metric_name"].iloc[0] == "y"
    assert pytest.approx(kpi_df["value"].iloc[0], rel=1e-6) == float(df["y"].mean())

    # Layout plan exists and is valid JSON with expected title
    plan_path = tmp_path / paths["layout_plan"]
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert plan["report_title"].startswith("Report:")
    assert plan["dataset_slug"] == slug
    assert isinstance(plan["sections"], list) and len(plan["sections"]) == 2
