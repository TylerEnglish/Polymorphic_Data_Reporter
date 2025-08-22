from __future__ import annotations

from typing import Optional, Tuple
import math

import numpy as np
import pandas as pd

# Prefer SciPy for exact distributions when installed
try:  # pragma: no cover - availability depends on environment
    from scipy import stats as _st
    _SCIPY_OK = True
except Exception:  # pragma: no cover
    _st = None
    _SCIPY_OK = False

# Lightweight fallbacks (already in your repo)
from .stats_pvalues import z_two_tailed, t_approx_p, chi2_sf, f_sf


# ------------------------------ small helpers ------------------------------

def to_num(s: pd.Series) -> pd.Series:
    """
    Coerce to numeric with NaN on failure. Stable and vectorized.

    Time: O(n). Space: O(1) beyond the returned Series.
    """
    # Avoid triggering pandas SettingWithCopy or dtype warnings
    return pd.to_numeric(s, errors="coerce")


# ------------------------------ correlations ------------------------------

def safe_pearson(x: pd.Series, y: pd.Series) -> Tuple[float, int, float]:
    """
    Pearson correlation r, sample size n, two-tailed p-value.

    - Drops NaNs pairwise.
    - If n<3 or variance is zero → r=0, p=1.
    - Uses SciPy's exact p-value when available, otherwise a robust approximation.

    Time: O(n). Space: O(1) beyond copies for dropna.
    """
    xx = to_num(x)
    yy = to_num(y)
    m = pd.concat([xx, yy], axis=1).dropna()
    n = int(m.shape[0])
    if n < 3:
        return 0.0, n, 1.0

    # Fast path if SciPy is available
    if _SCIPY_OK:  # pragma: no cover (exercised in full env)
        r, p = _st.pearsonr(m.iloc[:, 0].to_numpy(), m.iloc[:, 1].to_numpy())
        if not (np.isfinite(r) and np.isfinite(p)):
            return 0.0, n, 1.0
        return float(r), n, float(max(0.0, min(1.0, p)))

    # Fallback: r via pandas/numpy + approximate p
    r = float(m.iloc[:, 0].corr(m.iloc[:, 1]))
    if not np.isfinite(r):
        return 0.0, n, 1.0
    # Handle perfect correlation numerically
    if abs(r) >= 1.0 - 1e-12:
        return float(np.sign(r)), n, 0.0
    # t-stat under H0: r*sqrt((n-2)/(1-r^2))
    denom = max(1e-12, 1.0 - r * r)
    t = abs(r) * math.sqrt(max(n - 2, 1) / denom)
    p = t_approx_p(t, n - 2)
    p = float(max(0.0, min(1.0, p)))
    return float(r), n, p


# ------------------------------ ANOVA / effect sizes ------------------------------

def anova_oneway(vals: pd.Series, groups: pd.Series) -> Tuple[float, int, int, float, float]:
    """
    One-way ANOVA on (vals ~ groups) returning:
        (eta2, k_groups, n, F, p_value)

    - Drops NaN pairs.
    - If k<2 or n<=k, returns zeros-ish and p=1.
    - eta² = SS_between / SS_total.

    Time: O(n). Space: O(k) for per-group aggregates.
    """
    m = pd.concat([to_num(vals), groups.astype("string")], axis=1).dropna()
    if m.empty:
        return 0.0, 0, 0, 0.0, 1.0

    y = m.iloc[:, 0].to_numpy(dtype=float, copy=False)
    g = m.iloc[:, 1].to_numpy()
    n = int(y.size)
    if n == 0:
        return 0.0, 0, 0, 0.0, 1.0

    # Group indices (stable order is not required)
    levels = np.unique(g)
    k = int(levels.size)
    if k < 2:
        return 0.0, k, n, 0.0, 1.0

    overall = float(np.mean(y))
    ss_total = float(np.square(y - overall).sum())

    # Compute SS_between and SS_within without materializing large subframes
    ss_between = 0.0
    ss_within = 0.0
    for lvl in levels:
        mask = (g == lvl)
        yi = y[mask]
        if yi.size == 0:
            continue
        mu_i = float(np.mean(yi))
        # between: n_i * (mu_i - mu)^2
        ss_between += float(yi.size) * (mu_i - overall) ** 2
        # within: sum (y - mu_i)^2
        ss_within += float(np.square(yi - mu_i).sum())

    # Degrees of freedom
    dfb = max(k - 1, 1)
    dfw = max(n - k, 1)

    # Mean squares
    msb = ss_between / dfb
    # If dfw==0, treat MS_within as +inf (degenerate); results in F=0, p=1 unless ss_between==0
    msw = (ss_within / dfw) if dfw > 0 else float("inf")

    # F-statistic
    if not np.isfinite(msw) or msw <= 0.0:
        F = float("inf") if msb > 0.0 else 0.0
    else:
        F = float(msb / max(msw, 1e-12))

    # p-value
    if _SCIPY_OK and np.isfinite(F):  # pragma: no cover
        p = float(_st.f.sf(F, dfb, dfw))
    else:
        p = 0.0 if not np.isfinite(F) and msb > 0.0 else float(f_sf(F, dfb, dfw))

    # eta^2 (guard ss_total ~ 0)
    eta2 = float(ss_between / max(ss_total, 1e-12))
    eta2 = max(0.0, min(1.0, eta2))

    # Clamp p to [0,1]
    p = max(0.0, min(1.0, p))
    return eta2, k, n, F, p


def cramers_v_with_p(cat_a: pd.Series, cat_b: pd.Series) -> Tuple[float, int, int, float, float]:
    """
    Cramér's V (effect size) and chi-square p-value for association between two categoricals.

    Returns:
        (v, n, dof, p_value, chi2)

    - Drops NaN pairs.
    - Handles degenerate tables (1xN, Nx1) with dof=0 → p=1, v=0.
    - Uses SciPy's chi2_contingency when available, otherwise computes chi2 and p from expected counts.

    Time: O(n + r*c). Space: O(r*c) for the contingency table.
    """
    m = pd.concat([cat_a.astype("string"), cat_b.astype("string")], axis=1).dropna()
    if m.empty:
        return 0.0, 0, 0, 1.0, 0.0

    tab = pd.crosstab(m.iloc[:, 0], m.iloc[:, 1])
    r, c = tab.shape
    total = float(tab.values.sum())
    if total <= 0.0 or r == 0 or c == 0:
        return 0.0, 0, 0, 1.0, 0.0

    # Degenerate: <2 rows or <2 cols ⇒ dof=0, v=0, p=1
    if r < 2 or c < 2:
        return 0.0, int(total), 0, 1.0, 0.0

    if _SCIPY_OK:  # pragma: no cover
        chi2, p, dof, _ = _st.chi2_contingency(tab.values, correction=False)
    else:
        rowsum = tab.sum(axis=1).to_numpy(dtype=float, copy=False)
        colsum = tab.sum(axis=0).to_numpy(dtype=float, copy=False)
        expected = np.outer(rowsum, colsum) / max(total, 1e-12)
        with np.errstate(divide="ignore", invalid="ignore"):
            chi2 = float(np.where(expected > 0.0,
                                  np.square(tab.values - expected) / expected,
                                  0.0).sum())
        dof = (r - 1) * (c - 1)
        # Use our survival function for stability
        p = chi2_sf(chi2, dof)

    # Cramér's V: sqrt(chi2 / (n*(k-1))) where k=min(r,c)
    k = min(r, c)
    denom = max(total * (k - 1), 1e-12)
    v = math.sqrt(max(chi2 / denom, 0.0))

    # Clamp
    p = float(max(0.0, min(1.0, p)))
    v = float(max(0.0, min(1.0, v)))
    return v, int(total), int(dof), p, float(chi2)


# ------------------------------ time-series helper ------------------------------

def trend_slope(y: pd.Series) -> Tuple[float, int]:
    """
    Linear trend slope normalized by mean(|y|). Useful for scale-free ranking.

    - Drops NaNs.
    - If n<3 or y is (near) constant → slope 0.

    Time: O(n). Space: O(1).
    """
    s = to_num(y).dropna()
    n = int(s.size)
    if n < 3:
        return 0.0, n
    # x = 0..n-1 to avoid datetime fitting overhead, O(n)
    x = np.arange(n, dtype=float)
    with np.errstate(all="ignore"):
        try:
            m = float(np.polyfit(x, s.to_numpy(dtype=float, copy=False), 1)[0])
        except Exception:
            m = 0.0
    denom = float(np.mean(np.abs(s.to_numpy(dtype=float, copy=False))) + 1e-12)
    return float(m / denom), n


# ------------------------------ simple inequality / shares ------------------------------

def gini(x: np.ndarray) -> float:
    """
    Gini coefficient for non-negative data.
    - Negative entries are ignored (clamped out).
    - Returns 0 if array is empty or sums to 0.

    Time: O(n log n) due to sort. Space: O(1) besides a copy for sorting.
    """
    arr = np.asarray(x, dtype=float)
    arr = arr[arr >= 0.0]
    if arr.size == 0:
        return 0.0
    s = float(arr.sum())
    if s <= 0.0:
        return 0.0
    # Sort in-place
    arr.sort()
    n = float(arr.size)
    cum = np.cumsum(arr)
    # Using 1-based indices Gini closed form
    return float((n + 1.0 - 2.0 * (cum.sum() / cum[-1])) / n)


def shares(series: pd.Series) -> np.ndarray:
    """
    Convert a numeric Series to share weights that sum to 1 (or all-zeros if sum<=0).

    Time: O(n). Space: O(1).
    """
    vals = to_num(series).to_numpy(dtype=float, copy=False)
    s = float(np.nansum(vals))
    if not np.isfinite(s) or s <= 0.0:
        return np.zeros_like(vals, dtype=float)
    return (vals / s).astype(float, copy=False)


# ------------------------------ cohorts ------------------------------

def cohort_table(df: pd.DataFrame, time_col: str, id_col: str) -> Optional[pd.DataFrame]:
    """
    Monthly cohort retention table:
        columns: ["_first_seen", "_age", "retained", "cohort_size", "retention_pct"]

    - Snaps timestamps to the first day of month.
    - Computes cohort by first-seen month per id.
    - _age is month difference from first-seen.
    - Returns None if time column cannot be parsed to >=1 non-null values.

    Time: O(n log n) worst-case for groupby; Space: O(n) for intermediate columns.
    """
    if id_col not in df.columns or time_col not in df.columns:
        return None

    t = pd.to_datetime(df[time_col], errors="coerce")
    if t.isna().all():
        return None

    d = pd.DataFrame({id_col: df[id_col].copy(), "_t": t})
    # Normalize to month starts
    d["_t"] = pd.to_datetime(d["_t"].dt.to_period("M").dt.to_timestamp())

    # First-seen per id (vectorized transform)
    try:
        d["_first_seen"] = d.groupby(id_col, sort=False)["_t"].transform("min")
    except Exception:
        # Fallback: agg then join (slower)
        firsts = d.groupby(id_col, sort=False)["_t"].min()
        d = d.join(firsts, on=id_col, rsuffix="_first_seen")
        d["_first_seen"] = d["_t_first_seen"]
        d.drop(columns=["_t_first_seen"], inplace=True)

    # Month age = (year*12 + month) diff (avoid Period arithmetic pitfalls)
    ty, tm = d["_t"].dt.year.to_numpy(), d["_t"].dt.month.to_numpy()
    fy, fm = d["_first_seen"].dt.year.to_numpy(), d["_first_seen"].dt.month.to_numpy()
    with np.errstate(invalid="ignore"):
        age = (ty - fy) * 12 + (tm - fm)
    d["_age"] = pd.Series(age, index=d.index, dtype="Int64")

    # Count unique ids per (cohort, age)
    g = (
        d.groupby(["_first_seen", "_age"], dropna=True, sort=True)[id_col]
         .nunique()
         .reset_index(name="retained")
    )

    if g.empty:
        return g  # empty DataFrame, consistent type

    # Cohort sizes at age 0
    sizes = (
        g[g["_age"] == 0][["_first_seen", "retained"]]
          .rename(columns={"retained": "cohort_size"})
    )
    out = g.merge(sizes, on="_first_seen", how="left")
    out["cohort_size"] = out["cohort_size"].astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["retention_pct"] = (out["retained"] / out["cohort_size"]).astype(float)
    out["retention_pct"] = out["retention_pct"].fillna(0.0).clip(lower=0.0, upper=1.0)
    return out.sort_values(["_first_seen", "_age"]).reset_index(drop=True)
