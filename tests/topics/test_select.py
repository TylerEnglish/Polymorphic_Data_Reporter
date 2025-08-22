import json
import pandas as pd
from types import SimpleNamespace as NS

from src.topics.select import select_topics


def _cfg(
    min_corr_for_scatter=0.35,
    min_slope_for_trend=0.03,
    max_charts_total=12,
):
    thresholds = NS(
        min_corr_for_scatter=min_corr_for_scatter,
        min_slope_for_trend=min_slope_for_trend,
        max_charts_total=max_charts_total,
    )
    return NS(topics=NS(thresholds=thresholds))


def _row(**kwargs):
    """Minimal topic row builder for select() tests."""
    defaults = dict(
        family="kpi",
        effect_size=0.0,
        effect_norm=0.0,
        coverage_pct=1.0,
        score_total=0.5,
        primary_fields=("x",),
        significance=json.dumps({}),
        complexity_penalty=0.0,
    )
    defaults.update(kwargs)
    return defaults


def test_filters_weak_pearson_only():
    cfg = _cfg(min_corr_for_scatter=0.35)

    strong = _row(
        family="correlation",
        effect_size=0.60,
        score_total=0.9,
        primary_fields=("y1",),
        significance=json.dumps({"test": "pearson", "p_value": 0.01}),
    )
    weak = _row(
        family="correlation",
        effect_size=0.20,  # below threshold
        score_total=0.95,  # even with higher score, should be filtered
        primary_fields=("y2",),
        significance=json.dumps({"test": "pearson", "p_value": 0.20}),
    )
    df = pd.DataFrame([strong, weak])

    out = select_topics(df, cfg)
    assert len(out) == 1
    assert out.iloc[0]["primary_fields"] == ("y1",)


def test_does_not_gate_non_pearson_correlation():
    cfg = _cfg(min_corr_for_scatter=0.35)

    anova_low = _row(
        family="correlation",
        effect_size=0.10,  # below corr threshold, but ANOVA shouldn't be gated by it
        score_total=0.6,
        primary_fields=("m_vs_c",),
        significance=json.dumps({"test": "anova"}),
    )
    df = pd.DataFrame([anova_low])

    out = select_topics(df, cfg)
    assert len(out) == 1
    assert out.iloc[0]["primary_fields"] == ("m_vs_c",)


def test_trend_gate_uses_normalized_threshold():
    # min_slope=0.03 -> denom=0.09 -> min_norm = 0.03/0.09 = 1/3 ≈ 0.3333
    cfg = _cfg(min_slope_for_trend=0.03)

    below = _row(
        family="trend",
        effect_norm=0.20,
        score_total=0.7,
        primary_fields=("y1",),
    )
    above = _row(
        family="trend",
        effect_norm=0.50,
        score_total=0.8,
        primary_fields=("y2",),
    )
    df = pd.DataFrame([below, above])

    out = select_topics(df, cfg)
    assert len(out) == 1
    assert out.iloc[0]["primary_fields"] == ("y2",)


def test_causal_coverage_guard():
    cfg = _cfg()

    low_cov = _row(
        family="causal",
        coverage_pct=0.55,
        score_total=0.9,
        primary_fields=("y1",),
    )
    ok_cov = _row(
        family="causal",
        coverage_pct=0.80,
        score_total=0.7,
        primary_fields=("y2",),
    )
    df = pd.DataFrame([low_cov, ok_cov])

    out = select_topics(df, cfg)
    assert len(out) == 1
    assert out.iloc[0]["primary_fields"] == ("y2",)


def test_dedup_keeps_highest_score_for_same_family_and_primary_fields():
    cfg = _cfg()

    a = _row(
        family="distribution",
        score_total=0.60,
        primary_fields=("y1",),
    )
    b = _row(
        family="distribution",
        score_total=0.95,  # higher score, same family & primary_fields
        primary_fields=("y1",),
    )
    df = pd.DataFrame([a, b])

    out = select_topics(df, cfg)
    assert len(out) == 1
    assert out.iloc[0]["score_total"] == 0.95
    assert out.iloc[0]["primary_fields"] == ("y1",)


def test_caps_to_max_charts_total():
    cfg = _cfg(max_charts_total=5)
    rows = [
        _row(family="kpi", score_total=1.0 - i * 0.01, primary_fields=(f"m{i}",))
        for i in range(10)
    ]
    df = pd.DataFrame(rows)

    out = select_topics(df, cfg)
    assert len(out) == 5
    # ensure sorted by score_total desc
    assert all(out["score_total"].values[i] >= out["score_total"].values[i + 1] for i in range(len(out) - 1))
