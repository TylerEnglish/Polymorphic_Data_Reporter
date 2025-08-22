import json
from types import SimpleNamespace

import pandas as pd
import numpy as np

from src.topics.scoring import (
    score_topics,
    _family_effect_norm,
)


def _cfg():
    # minimal config with thresholds + weights
    return SimpleNamespace(
        topics=SimpleNamespace(
            thresholds=SimpleNamespace(
                min_slope_for_trend=0.02
            )
        ),
        weights=SimpleNamespace(
            suitability=0.25,
            effect_size=0.35,
            signal_quality=0.2,
            readability=0.15,
            complexity=0.15,
        ),
    )


def test_empty_dataframe_returns_expected_columns():
    cfg = _cfg()
    empty = pd.DataFrame(columns=["family", "coverage_pct", "complexity_penalty"])
    out = score_topics(empty, cfg)
    for col in ["effect_norm", "suitability", "readability_penalty",
                "complexity", "signal_quality", "score_total"]:
        assert col in out.columns
    assert out.empty


def test_signal_quality_uses_p_value_when_available():
    cfg = _cfg()
    base = dict(
        family="correlation",
        effect_size=0.5,
        coverage_pct=1.0,
        complexity_penalty=0.0,
        n_obs=100,
        effect_detail=json.dumps({}),
    )
    good = {**base, "significance": json.dumps({"test": "pearson", "p_value": 0.01})}
    bad  = {**base, "significance": json.dumps({"test": "pearson", "p_value": 0.90})}
    df = pd.DataFrame([good, bad])

    scored = score_topics(df, cfg)
    # lower p should yield higher signal and higher score
    sg = scored["score_total"].values
    assert sg[0] > sg[1]


def test_signal_quality_falls_back_to_n_obs_when_no_p():
    cfg = _cfg()
    lo_n = dict(
        family="correlation",
        effect_size=0.3,
        coverage_pct=1.0,
        complexity_penalty=0.0,
        n_obs=10,
        significance=json.dumps({"test": "anova"}),  # no p_value
        effect_detail=json.dumps({}),
    )
    hi_n = {**lo_n, "n_obs": 100}
    df = pd.DataFrame([lo_n, hi_n])

    scored = score_topics(df, cfg)
    assert scored.loc[1, "score_total"] > scored.loc[0, "score_total"]


def test_family_effect_norm_trend_respects_threshold():
    cfg = _cfg()
    # slope exactly at 1.5× min threshold should have higher norm than tiny slope
    row_hi = pd.Series({
        "family": "trend",
        "effect_size": cfg.topics.thresholds.min_slope_for_trend * 1.5
    })
    row_lo = pd.Series({
        "family": "trend",
        "effect_size": cfg.topics.thresholds.min_slope_for_trend * 0.1
    })
    assert _family_effect_norm(row_hi, cfg) > _family_effect_norm(row_lo, cfg)


def test_readability_penalty_reduces_score_for_long_trends():
    cfg = _cfg()
    base = dict(
        family="trend",
        effect_size=cfg.topics.thresholds.min_slope_for_trend * 1.5,
        coverage_pct=1.0,
        complexity_penalty=0.0,
        n_obs=200,
    )
    short = {
        **base,
        "effect_detail": json.dumps({"n_points": 60}),
        "significance": json.dumps({}),
    }
    long = {
        **base,
        "effect_detail": json.dumps({"n_points": 200}),
        "significance": json.dumps({}),
    }
    df = pd.DataFrame([short, long])

    scored = score_topics(df, cfg)
    assert scored.loc[0, "score_total"] > scored.loc[1, "score_total"]


def test_components_columns_present_and_reasonable_ranges():
    cfg = _cfg()
    row = dict(
        family="distribution",
        effect_size=2.0,              # will be squashed
        coverage_pct=0.6,
        complexity_penalty=0.05,
        n_obs=120,
        significance=json.dumps({}),  # no p
        effect_detail=json.dumps({}),
    )
    df = pd.DataFrame([row])
    scored = score_topics(df, cfg).iloc[0]

    # component columns exist and are 0..1-ish where expected
    assert 0.0 <= scored["effect_norm"] <= 1.0
    assert 0.0 <= scored["suitability"] <= 1.0
    assert 0.0 <= scored["signal_quality"] <= 1.0
    assert scored["complexity"] >= 0.0
