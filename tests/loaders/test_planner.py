import json
from types import SimpleNamespace

import pandas as pd

from src.layout.planner import make_layout_plan


def _cfg(title_prefix="Report:"):
    # Minimal duck-typed config object for planner
    return SimpleNamespace(
        reports=SimpleNamespace(
            html=SimpleNamespace(
                title_prefix=title_prefix
            )
        )
    )


def _selected_df():
    rows = [
        # High-score correlation with p-value -> "significant" badge, uses first proposed chart
        dict(
            topic_id="t1",
            family="correlation",
            primary_fields=json.dumps(["y1", "y2"]),
            secondary_fields=json.dumps([]),
            time_field=None,
            coverage_pct=0.80,
            effect_size=0.62,
            significance=json.dumps({"test": "pearson", "p_value": 0.01}),
            proposed_charts='["scatter", "xy_heatmap"]',
            n_obs=100,
            score_total=0.90,
        ),
        # Low-coverage trend with empty proposed charts -> fallback to "line"
        dict(
            topic_id="t2",
            family="trend",
            primary_fields=json.dumps(["y1"]),
            secondary_fields=json.dumps([]),
            time_field="date",
            coverage_pct=0.30,  # triggers "low-coverage"
            effect_size=0.05,
            significance=json.dumps({}),
            proposed_charts="[]",
            n_obs=50,
            score_total=0.80,
        ),
        # KPI -> key-metric badge, fallback to "kpi" if no chart proposals
        dict(
            topic_id="t3",
            family="kpi",
            primary_fields="y2",  # plain string should coerce to ["y2"]
            secondary_fields=None,
            time_field=None,
            coverage_pct=1.00,
            effect_size=0.0,
            significance=None,
            proposed_charts=None,
            n_obs=120,
            score_total=0.50,
        ),
        # Causal -> "causal" badge
        dict(
            topic_id="t4",
            family="causal",
            primary_fields=json.dumps(["y3"]),
            secondary_fields=json.dumps(["treat"]),
            time_field=None,
            coverage_pct=0.95,
            effect_size=0.2,
            significance=json.dumps({"test": "did", "p_value": 0.2}),
            proposed_charts='["bar", "column"]',
            n_obs=120,
            score_total=0.60,
        ),
    ]
    return pd.DataFrame(rows)


def test_plan_basic_structure_and_order():
    cfg = _cfg("Report for")
    selected = _selected_df()

    plan = make_layout_plan(selected, cfg, dataset_slug="toy")

    # top-level fields
    assert plan["report_title"] == "Report for toy"
    assert plan["dataset_slug"] == "toy"
    assert pd.to_datetime(plan["generated_at"], errors="coerce") is not pd.NaT

    # sections sorted by score_total desc
    sections = plan["sections"]
    assert len(sections) == 4
    assert [s["topic_ref"] for s in sections] == ["t1", "t2", "t4", "t3"]

    # each section has one component with export hints
    for s in sections:
        assert "title" in s and s["title"]
        assert isinstance(s["components"], list) and len(s["components"]) == 1
        comp = s["components"][0]
        assert comp["export"] == {"html": True, "png": True}
        assert isinstance(comp.get("encodings"), dict)


def test_chart_selection_badges_and_dataset_spec():
    cfg = _cfg()
    selected = _selected_df()
    plan = make_layout_plan(selected, cfg, dataset_slug="toy")

    s1, s2, s3, s4 = plan["sections"]

    # correlation: first proposed chart used; badges include "significant"
    c1 = s1["components"][0]
    assert c1["kind"] == "scatter"
    assert c1["proposed_charts"] == ["scatter", "xy_heatmap"]
    assert "significant" in c1["badges"]
    assert "low-coverage" not in c1["badges"]
    assert c1["dataset_spec"]["family"] == "correlation"
    assert c1["dataset_spec"]["primary_fields"] == ["y1", "y2"]
    assert c1["dataset_spec"]["topic_id"] == "t1"
    assert s1["title"].startswith("Correlation – ")

    # trend: empty proposals -> fallback to "line"; low coverage badge present
    c2 = s2["components"][0]
    assert c2["kind"] == "line"
    assert c2["proposed_charts"] == []
    assert "low-coverage" in c2["badges"]
    assert c2["dataset_spec"]["family"] == "trend"
    assert c2["dataset_spec"]["time_field"] == "date"
    assert s2["title"] == "Trend – y1"

    # causal (third section after sorting by score_total)
    c3 = s3["components"][0]
    assert "causal" in c3["badges"]
    assert c3["kind"] == "bar"  # first proposed in the row
    assert c3["proposed_charts"] == ["bar", "column"]
    assert c3["dataset_spec"]["secondary_fields"] == ["treat"]

    # kpi (fourth section after sorting by score_total)
    c4 = s4["components"][0]
    assert "key-metric" in c4["badges"]
    assert c4["kind"] == "kpi"
    assert c4["proposed_charts"] == []
    assert c4["dataset_spec"]["primary_fields"] == ["y2"]
