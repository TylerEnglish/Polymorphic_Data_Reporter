from __future__ import annotations

import math
import numpy as np
import pandas as pd
import pytest

import src.topics.stats_basic as sb


def test_to_num_basic_and_invalid():
    s = pd.Series(["1", "2.5", "x", None, " 3 "])
    out = sb.to_num(s)
    assert out.tolist()[:2] == [1.0, 2.5]
    assert math.isnan(out.iloc[2])
    assert math.isnan(out.iloc[3])
    assert out.iloc[4] == 3.0


# ---------------- safe_pearson ----------------

def test_safe_pearson_perfect_positive():
    x = pd.Series([0, 1, 2, 3, 4])
    y = pd.Series([0, 2, 4, 6, 8])
    r, n, p = sb.safe_pearson(x, y)
    assert n == 5
    assert pytest.approx(r, abs=1e-12) == 1.0
    assert p == 0.0

def test_safe_pearson_perfect_negative():
    x = pd.Series([0, 1, 2, 3, 4])
    y = pd.Series([0, -1, -2, -3, -4])
    r, n, p = sb.safe_pearson(x, y)
    assert n == 5
    assert pytest.approx(r, abs=1e-12) == -1.0
    assert p == 0.0

def test_safe_pearson_with_nans_and_low_n():
    x = pd.Series([1.0, np.nan, 3.0])
    y = pd.Series([2.0, 4.0, np.nan])
    r, n, p = sb.safe_pearson(x, y)
    # after dropna there are 1 or 0 pairs, so n<3
    assert n < 3
    assert r == 0.0
    assert p == 1.0

def test_safe_pearson_zero_variance():
    x = pd.Series([5.0, 5.0, 5.0, 5.0])
    y = pd.Series([1.0, 2.0, 3.0, 4.0])
    r, n, p = sb.safe_pearson(x, y)
    assert n == 4
    # undefined, we return r=0, p=1
    assert r == 0.0
    assert p == 1.0

def test_safe_pearson_strong_but_not_perfect():
    rng = np.random.default_rng(0)
    x = pd.Series(rng.normal(size=200))
    y = 0.8 * x + pd.Series(rng.normal(scale=0.2, size=200))
    r, n, p = sb.safe_pearson(x, y)
    assert n == 200
    assert r > 0.7
    assert p < 1e-10


# ---------------- anova_oneway ----------------

def test_anova_oneway_clear_separation():
    # Three groups with different means
    g = np.repeat(["A", "B", "C"], 50)
    y = np.concatenate([
        np.random.normal(loc=0.0, scale=0.5, size=50),
        np.random.normal(loc=2.0, scale=0.5, size=50),
        np.random.normal(loc=4.0, scale=0.5, size=50),
    ])
    eta2, k, n, F, p = sb.anova_oneway(pd.Series(y), pd.Series(g))
    assert k == 3 and n == 150
    assert F > 100.0
    assert p < 1e-20
    assert 0.8 < eta2 < 1.0

def test_anova_oneway_degenerate_single_group():
    y = pd.Series([1, 2, 3, 4])
    g = pd.Series(["A", "A", "A", "A"])
    eta2, k, n, F, p = sb.anova_oneway(y, g)
    assert k == 1
    assert p == 1.0
    assert F == 0.0
    assert eta2 == 0.0

def test_anova_oneway_all_equal_values():
    y = pd.Series([5, 5, 5, 5, 5, 5])
    g = pd.Series(["A", "A", "B", "B", "C", "C"])
    eta2, k, n, F, p = sb.anova_oneway(y, g)
    # No variability → ss_total ~ 0, eta2 ~ 0, F ~ 0, p ~ 1
    assert k == 3 and n == 6
    assert F == 0.0
    assert p == 1.0
    assert eta2 == 0.0

def test_anova_oneway_dfw_zero_case():
    # n == k (each group has a single value)
    y = pd.Series([0.0, 1.0, 2.0])
    g = pd.Series(["A", "B", "C"])
    eta2, k, n, F, p = sb.anova_oneway(y, g)
    assert k == 3 and n == 3
    # With dfw=0, implementation yields F = inf if msb>0 → p ~ 0
    assert (math.isinf(F) and p == 0.0) or (F > 1e6 and p < 1e-6)
    assert 0.0 <= eta2 <= 1.0


# ---------------- cramers_v_with_p ----------------

def test_cramers_v_with_p_strong_association_2x2():
    # Contingency with strong diagonal
    # [[30, 1],
    #  [ 1,30]]
    a = pd.Series(["X"] * 31 + ["Y"] * 31)
    b = pd.Series(["M"] * 30 + ["N"] * 1 + ["M"] * 1 + ["N"] * 30)
    v, n, dof, p, chi2 = sb.cramers_v_with_p(a, b)
    assert n == 62 and dof == 1
    assert v > 0.8
    assert p < 1e-10
    assert chi2 > 50.0

def test_cramers_v_with_p_degenerate_1xN():
    a = pd.Series(["X", "X", "X", "X"])
    b = pd.Series(["M", "N", "M", "N"])
    v, n, dof, p, chi2 = sb.cramers_v_with_p(a, b)
    assert n == 4 and dof == 0
    assert v == 0.0
    assert p == 1.0
    assert chi2 == 0.0

def test_cramers_v_with_p_random_independence():
    rng = np.random.default_rng(1)
    a = pd.Series(rng.choice(["A", "B", "C"], size=500, replace=True))
    b = pd.Series(rng.choice(["X", "Y"], size=500, replace=True))
    v, n, dof, p, chi2 = sb.cramers_v_with_p(a, b)
    assert n == 500 and dof == (3 - 1) * (2 - 1) == 2
    # likely weak association, p not very small
    assert v < 0.2
    assert p > 0.01


# ---------------- trend_slope ----------------

def test_trend_slope_linear_increase():
    y = pd.Series([0, 1, 2, 3, 4])
    slope, n = sb.trend_slope(y)
    # slope normalized by mean(|y|)=2 → expect ~0.5
    assert n == 5
    assert pytest.approx(slope, rel=1e-6, abs=1e-6) == 0.5

def test_trend_slope_constant_and_nans():
    y = pd.Series([5, 5, 5, np.nan, 5, 5])
    slope, n = sb.trend_slope(y)
    # After dropna, constant → slope ~ 0
    assert n == 5
    assert abs(slope) < 1e-12

def test_trend_slope_small_n():
    y = pd.Series([1.0, np.nan])
    slope, n = sb.trend_slope(y)
    assert n == 1
    assert slope == 0.0


# ---------------- gini ----------------

def test_gini_uniform_and_empty():
    assert sb.gini(np.array([])) == 0.0
    assert sb.gini(np.array([1, 1, 1, 1])) == 0.0

def test_gini_two_point_mass():
    # [0, 2] should yield 0.5 for equal weights
    assert pytest.approx(sb.gini(np.array([0.0, 2.0])), rel=1e-12, abs=1e-12) == 0.5

def test_gini_general_bounds():
    g = sb.gini(np.array([1, 2, 3, 4]))
    assert 0.0 < g < 1.0


# ---------------- shares ----------------

def test_shares_basic_and_sum_to_one():
    s = pd.Series([2.0, 3.0, 5.0])
    sh = sb.shares(s)
    assert sh.shape == (3,)
    assert (sh >= 0).all()
    assert pytest.approx(sh.sum(), rel=1e-12, abs=1e-12) == 1.0

def test_shares_nonpositive_sum_and_nans():
    s = pd.Series([np.nan, -1.0, 1.0, 0.0])
    # sum of numeric (ignoring NaN) is 0 → zeros
    sh = sb.shares(s)
    assert np.allclose(sh, 0.0)


# ---------------- cohort_table ----------------

def test_cohort_table_happy_path():
    df = pd.DataFrame(
        {
            "user_id": ["A", "A", "B", "B", "C"],
            "event_time": [
                "2024-01-10",  # A first seen Jan
                "2024-02-05",  # A again in Feb (age 1 for Jan cohort)
                "2024-02-03",  # B first seen Feb
                "2024-03-01",  # B in Mar (age 1 for Feb cohort)
                "2024-02-15",  # C first seen Feb
            ],
        }
    )
    tab = sb.cohort_table(df, "event_time", "user_id")
    assert isinstance(tab, pd.DataFrame)
    # Expect rows for:
    # - Jan cohort age 0 (A), age 1 (A present in Feb)
    # - Feb cohort age 0 (B,C), age 1 (B present in Mar)
    # Check retention percentages:
    # Jan cohort size=1 → age0 retained=1 ⇒ 1.0; age1 retained=1 ⇒ 1.0
    jan = pd.to_datetime("2024-01-01")
    feb = pd.to_datetime("2024-02-01")
    mar = pd.to_datetime("2024-03-01")

    jan_rows = tab[tab["_first_seen"] == jan]
    feb_rows = tab[tab["_first_seen"] == feb]

    # Jan cohort
    assert any((jan_rows["_age"] == 0) & (np.isclose(jan_rows["retention_pct"], 1.0)))
    assert any((jan_rows["_age"] == 1) & (np.isclose(jan_rows["retention_pct"], 1.0)))

    # Feb cohort
    # Age 0: both B and C in Feb → cohort_size=2, retained=2 → 1.0
    assert any((feb_rows["_age"] == 0) & (np.isclose(feb_rows["retention_pct"], 1.0)))
    # Age 1: only B appears in Mar → retained=1, size=2 → 0.5
    assert any((feb_rows["_age"] == 1) & (np.isclose(feb_rows["retention_pct"], 0.5)))

def test_cohort_table_missing_time_or_id():
    df = pd.DataFrame({"user_id": ["A", "B"], "when": ["2024-01-01", "2024-01-02"]})
    # wrong time col → None
    assert sb.cohort_table(df, "event_time", "user_id") is None

    df2 = pd.DataFrame({"event_time": ["not-a-date", None]})
    # unparsable time → None
    assert sb.cohort_table(df2, "event_time", "user_id") is None
