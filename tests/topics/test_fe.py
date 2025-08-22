import math
from collections import Counter

import numpy as np
import pandas as pd
import pytest

from src.topics.fe import (
    build_engineered_candidates,
    good_metric_col,
    good_category_col,
    find_time_fallback,
)


# ----------------------------- fixtures ----------------------------- #

@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(7)


@pytest.fixture
def df(rng):
    """Synthetic dataset that lights up numerics, cats, bools, text, time, causal."""
    n = 300
    ts = pd.date_range("2023-01-01", periods=n, freq="D")
    t_idx = np.arange(n)

    # ids & groups
    user_ids = [f"u{(i % 10) + 1}" for i in range(n)]
    groups = rng.choice(["A", "B", "C"], size=n, p=[0.4, 0.4, 0.2])
    region_map = {"A": "North", "B": "South", "C": "North"}
    regions = np.array([region_map[g] for g in groups])

    # boolean/string treatment correlated with group
    is_treatment = np.array([g in ("A", "C") for g in groups])  # True for A/C
    variant_str = np.where(is_treatment, "Yes", "No")

    # numeric with trend and lift
    num1 = 0.1 * t_idx + (0.5 * is_treatment.astype(float)) + rng.normal(0, 0.6, size=n)

    # correlated numeric to num1
    num_corr = 0.9 * num1 + rng.normal(0, 0.2, size=n)

    # skewed numeric
    num2 = rng.lognormal(mean=0.0, sigma=1.0, size=n)

    # rate parts
    num_total = np.maximum(5.0 + rng.gamma(2.0, 3.0, size=n), 1.0)
    raw = 0.35 * num_total + rng.normal(0, 1.2, size=n)
    num_success = np.clip(raw, 0, num_total * 0.9)

    # extra categorical for frequency tables
    country = rng.choice(["US", "CA", "MX", "GB", "DE"], size=n, p=[0.5, 0.15, 0.15, 0.1, 0.1])

    # another boolean for bool-bool association
    is_promo = np.array([g == "A" for g in groups])

    # text with repeated tokens
    reviews = np.where(
        is_treatment,
        rng.choice(
            ["good value", "excellent service", "very good experience", "good good good", "fast and good"],
            size=n,
        ),
        rng.choice(
            ["bad UX", "slow and bad", "quite bad", "not good", "poor docs, bad support"],
            size=n,
        ),
    )

    # id-like monotone numeric (should be rejected as metric)
    row_index = np.arange(n)

    df = pd.DataFrame(
        {
            "ts": ts,
            "user_id": user_ids,
            "group": groups,
            "region": regions,
            "variant": variant_str,        # string boolean-like
            "is_promo": is_promo,          # boolean dtype
            "num1": num1,
            "num_corr": num_corr,
            "num2": num2,
            "num_total": num_total,
            "num_success": num_success,
            "country": country,
            "review": reviews,
            "row_index": row_index,
        }
    )

    # sprinkle NaNs
    nan_idx = np.linspace(0, n - 1, 15, dtype=int)
    df.loc[nan_idx, "num1"] = np.nan
    df.loc[nan_idx[::2], "review"] = None

    return df


@pytest.fixture
def cols():
    nums = ["num1", "num_corr", "num2", "num_total", "num_success", "row_index"]
    cats = ["group", "region", "country", "variant"]
    return nums, cats


# ----------------------------- unit tests ----------------------------- #

def test_good_metric_and_category_heuristics(df):
    # numeric: real metric
    assert good_metric_col(df, "num1") is True
    # id-like monotone near-unique
    assert good_metric_col(df, "row_index") is False
    # categories
    assert good_category_col(df, "group", max_card=60) is True
    # too-high cardinality created on the fly
    df["high_card"] = [f"id{i}" for i in range(len(df))]
    assert good_category_col(df, "high_card", max_card=60) is False


def test_find_time_fallback_detects_ts(df):
    assert find_time_fallback(df) == "ts"


def test_build_engineered_candidates_core_presence(df, cols):
    nums, cats = cols
    rows = build_engineered_candidates(df, nums, cats, time_col="ts", id_col="user_id")

    assert isinstance(rows, list) and len(rows) > 0
    families = Counter(r.family for r in rows)
    # broad families should appear
    assert families["distribution"] > 0
    assert families["trend"] > 0
    assert families["ranking"] > 0
    assert families["correlation"] > 0
    # causal screens may appear if treatment-like names are detected
    assert families["causal"] >= 1


def test_distribution_transform_plans_and_trend_plans_exist(df):
    nums = ["num1", "num2", "num_corr", "num_total", "num_success"]
    cats = ["group", "variant", "country"]
    rows = build_engineered_candidates(df, nums, cats, time_col="ts", id_col="user_id")

    # at least one distribution_fe with feature_plan (log1p/sqrt/zscore_wins)
    dist_fe = [r for r in rows if r.family == "distribution" and "feature_plan" in r.effect_detail]
    assert any(r.effect_detail.get("feature_plan") for r in dist_fe)

    # trend_fe or ts_windows present
    trend_like = [r for r in rows if r.family == "trend"]
    assert len(trend_like) > 0
    assert any("feature_plan" in r.effect_detail for r in trend_like)


def test_rate_feature_is_planned(df):
    nums = ["num1", "num2", "num_total", "num_success"]
    cats = ["group"]
    rows = build_engineered_candidates(df, nums, cats, time_col="ts", id_col=None)

    has_ratio = False
    for r in rows:
        fp = r.effect_detail.get("feature_plan")
        if not fp:
            continue
        for step in fp:
            if step.get("kind") == "binary_op" and step.get("params", {}).get("op") == "ratio":
                bases = tuple(step.get("base_cols", []))
                if "num_success" in bases and "num_total" in bases:
                    has_ratio = True
                    break
        if has_ratio:
            break
    assert has_ratio, "expected a rate(num_success/num_total) plan"


def test_calendar_breakdown_and_interarrival_present(df, cols):
    nums, cats = cols
    rows = build_engineered_candidates(df, nums, cats, time_col="ts", id_col="user_id")

    cal = [r for r in rows if r.topic_id.startswith("calendar_breakdown-")]
    assert len(cal) == 1
    plan = cal[0].effect_detail.get("feature_plan", [])
    units = {step.get("params", {}).get("unit") for step in plan}
    assert {"dow", "hour", "week", "month", "quarter", "year", "is_weekend"} <= units

    inter = [r for r in rows if r.topic_id.startswith("interarrival-")]
    assert len(inter) == 1
    iparams = [step.get("params", {}) for step in inter[0].effect_detail.get("feature_plan", [])]
    assert any(p.get("by") is None for p in iparams)
    assert any(p.get("by") == "user_id" for p in iparams)


def test_time_window_features_exist_global_entity_category(df):
    nums = ["num1", "num2", "num_corr"]
    cats = ["group", "country"]
    rows = build_engineered_candidates(df, nums, cats, time_col="ts", id_col="user_id")

    assert any(r.topic_id.startswith("ts_windows-") for r in rows)
    assert any(r.topic_id.startswith("entity_ts_windows-") for r in rows)
    assert any(r.topic_id.startswith("cat_ts_windows-") for r in rows)


def test_text_feature_planning(df):
    nums = ["num1"]
    cats = ["group"]
    rows = build_engineered_candidates(df, nums, cats, time_col="ts", id_col=None)

    text_rows = [r for r in rows if r.topic_id.startswith("text_fe-")]
    assert len(text_rows) >= 1
    plan = text_rows[0].effect_detail.get("feature_plan", [])
    kinds = {step.get("kind") for step in plan}
    assert {"transform", "text_tokens", "text_summary"} <= kinds
    token_steps = [s for s in plan if s.get("kind") == "text_tokens"]
    assert token_steps and isinstance(token_steps[0].get("params", {}).get("token_list"), list)


def test_pairwise_screens_num_num_num_cat_cat_cat_bool_bool_num_bool(df):
    nums = ["num1", "num_corr", "num2"]
    cats = ["group", "region", "country", "variant"]
    rows = build_engineered_candidates(df, nums, cats, time_col="ts", id_col=None)

    # num-num correlation present (num1 vs num_corr should be strong)
    assert any(r.topic_id.startswith("corr_num_num-") for r in rows)

    # num-cat (ANOVA)
    assert any(r.topic_id.startswith("assoc_num_cat-") for r in rows)

    # cat-cat association
    assert any(r.topic_id.startswith("assoc_cat_cat-") for r in rows)

    # bool-bool association (variant vs is_promo)
    assert any(r.topic_id.startswith("assoc_bool_bool-") for r in rows)

    # num-bool Welch screens
    assert any(r.topic_id.startswith("assoc_num_bool-") for r in rows)


def test_ranking_and_part_to_whole_and_counts(df):
    nums = ["num1", "num2"]
    cats = ["group", "country"]
    rows = build_engineered_candidates(df, nums, cats, time_col="ts", id_col=None)

    assert any(r.topic_id.startswith("ranking_fe-") for r in rows)
    assert any(r.topic_id.startswith("part_to_whole_fe-") for r in rows)
    assert any(r.topic_id.startswith("count_table-") for r in rows)


def test_causal_screens_present_with_treatment_like_name(df):
    nums = ["num1", "num2"]
    cats = ["group", "variant"]
    rows = build_engineered_candidates(df, nums, cats, time_col="ts", id_col=None)

    # ab screens (variant detected)
    assert any(r.topic_id.startswith("causal_ab_fe-") for r in rows)
    # did screens (needs time)
    assert any(r.topic_id.startswith("causal_did_fe-") for r in rows)


def test_caps_limit_pairwise_blowup(df):
    # create many numeric columns to exceed caps; function should bound work
    for i in range(20):
        df[f"noise_{i}"] = np.random.default_rng(123 + i).normal(size=len(df))

    nums = ["num1", "num2", "num_corr"] + [f"noise_{i}" for i in range(20)]
    cats = ["group", "region", "country", "variant"]
    rows = build_engineered_candidates(df, nums, cats, time_col="ts", id_col=None)

    # corr_num_num topics should not exceed (16 choose 2) due to cap (NUM_MAX=16)
    nn = [r for r in rows if r.topic_id.startswith("corr_num_num-")]
    assert len(nn) <= (16 * 15) // 2


def test_graceful_without_time_columns(df):
    # build view without any parseable time column
    d2 = df.drop(columns=["ts"])
    nums = ["num1", "num2", "num_corr"]
    cats = ["group", "country", "variant"]
    rows = build_engineered_candidates(d2, nums, cats, time_col=None, id_col=None)

    # No calendar breakdown or time-window topics when no time is present
    assert not any(r.topic_id.startswith("calendar_breakdown-") for r in rows)
    assert not any(r.topic_id.startswith("ts_windows-") for r in rows)
    # Still should have plenty of non-time topics
    assert any(r.family == "distribution" for r in rows)
    assert any(r.family == "correlation" for r in rows)
