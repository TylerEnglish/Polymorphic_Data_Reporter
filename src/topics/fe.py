# src/topics/fe.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Optional, Sequence, Tuple, Dict, Any
import itertools as it
import re

import numpy as np
import pandas as pd

from .model import TopicRow, topic_id
from .stats_basic import (
    to_num,
    safe_pearson,
    anova_oneway,
    cramers_v_with_p,
    trend_slope,
    gini,
    shares,
)
from .stats_pvalues import z_two_tailed

# --------------------------------------------------------------------------------------
# Tunables / caps (keep the planner fast & memory-safe; materializer does the heavy work)
# --------------------------------------------------------------------------------------

MAX_NUMS_FOR_PAIRWISE   = 16            # cap numeric columns for pairwise scans
MAX_CATS_FOR_PAIRWISE   = 12            # cap categorical columns for pairwise scans
MAX_BOOL_FOR_PAIRWISE   = 12            # cap booleans considered as categorical
MAX_CATS_CARDINALITY    = 60            # skip categories above this
MAX_TEXT_SAMPLE         = 25_000        # sample rows for token frequency planning
MAX_TOKENS_PER_TEXTCOL  = 50            # plan at most top-K tokens per text column
MAX_TOP_LEVELS_FOR_TIME = 6             # time group windows per top category
MIN_NON_NULL            = 10            # minimum non-null values to consider a feature
EPS                     = 1e-12

_ID_PAT  = re.compile(r"(?:^id$|_id$|^id_|uuid|guid|hash|checksum|^row_?id$|^entity_?id$|_index$)", re.I)
_URI_PAT = re.compile(r"(?:uri|url|^path$|file(path|name)$)", re.I)

BOOL_TRUE  = {"1", "true", "t", "yes", "y"}
BOOL_FALSE = {"0", "false", "f", "no", "n"}


# --------------------------------------------------------------------------------------
# Helpers: column type heuristics
# --------------------------------------------------------------------------------------

def is_id_like(name: str) -> bool:
    return bool(_ID_PAT.search(str(name)))

def is_uri_like(name: str) -> bool:
    return bool(_URI_PAT.search(str(name)))

def _unique_ratio(s: pd.Series) -> float:
    n = int(s.notna().sum())
    return 1.0 if n == 0 else float(s.nunique(dropna=True) / n)

def is_index_like_numeric(s: pd.Series) -> bool:
    ss = to_num(s).dropna()
    if ss.empty:
        return False
    ur = _unique_ratio(ss)
    return ur >= 0.98 and (ss.is_monotonic_increasing or ss.is_monotonic_decreasing)

def good_metric_col(df: pd.DataFrame, col: str) -> bool:
    if is_id_like(col) or is_uri_like(col):
        return False
    s = to_num(df[col])
    if _unique_ratio(s) >= 0.98 or is_index_like_numeric(s):
        return False
    return int(s.notna().sum()) >= MIN_NON_NULL

def good_category_col(df: pd.DataFrame, col: str, max_card: int = MAX_CATS_CARDINALITY) -> bool:
    if is_id_like(col) or is_uri_like(col):
        return False
    k = int(df[col].astype("string").nunique(dropna=True))
    return 2 <= k <= max_card

def _is_bool_series(s: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(s):
        return True
    if pd.api.types.is_numeric_dtype(s):
        # numeric 0/1
        vals = pd.Series(s.dropna().unique())
        return set(pd.unique(vals.clip(0, 1))) <= {0, 1}
    # string-like booleans
    vals = s.astype("string").dropna().str.lower().str.strip().unique()
    if len(vals) == 0:
        return False
    if len(vals) <= 3 and set(vals) <= (BOOL_TRUE | BOOL_FALSE | {"", "nan"}):
        return True
    return False

def _is_text_series(df: pd.DataFrame, col: str) -> bool:
    if is_id_like(col) or is_uri_like(col):
        return False
    s = df[col]
    if pd.api.types.is_string_dtype(s) or s.dtype == "object":
        # distinguish low-card categorical from "free text"
        k = int(s.astype("string").nunique(dropna=True))
        if k <= MAX_CATS_CARDINALITY:
            return False
        # long-ish strings or high unique ratio -> consider text
        avg_len = float(s.dropna().astype("string").str.len().mean() or 0.0)
        return avg_len >= 12 or _unique_ratio(s) >= 0.6
    return False

def _parseable_datetime(s: pd.Series, min_valid=0.60) -> bool:
    t = pd.to_datetime(s, errors="coerce")
    return float(t.notna().mean()) >= min_valid

def find_time_fallback(df: pd.DataFrame) -> Optional[str]:
    # look for common names
    hints = [
        c for c in df.columns
        if any(k in str(c).lower() for k in ("date", "time", "timestamp", "event_time", "eventtime", "dt", "ts"))
    ]
    for c in hints:
        if _parseable_datetime(df[c]):
            return c
    # dtype already datetime?
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            return c
    return None

def best_metric(df: pd.DataFrame, cols: List[str]) -> Optional[str]:
    if not cols:
        return None
    def _score(col: str) -> tuple[float, float]:
        s = to_num(df[col])
        return (float(s.notna().mean()), float(s.std(skipna=True) or 0.0))
    return sorted(cols, key=_score, reverse=True)[0]

def _top_k_categories(s: pd.Series, k: int) -> List[str]:
    vc = s.astype("string").value_counts(dropna=True)
    return vc.index[:k].tolist()


# --------------------------------------------------------------------------------------
# A small Welch test helper for bool↔numeric screens (normal approx for speed)
# --------------------------------------------------------------------------------------

def _welch_p(num: pd.Series, grp: pd.Series) -> float:
    # grp expected to be boolean-like 0/1
    x = to_num(num)
    g = grp.astype("string").str.lower().str.strip().map(lambda v: v in BOOL_TRUE or v == "1").astype(int)
    m = pd.concat([x, g], axis=1).dropna()
    if m.empty:
        return 1.0
    t = m.iloc[:, 1] == 1
    c = ~t
    x_t = m.loc[t, m.columns[0]].values.astype(float)
    x_c = m.loc[c, m.columns[0]].values.astype(float)
    if x_t.size < 3 or x_c.size < 3:
        return 1.0
    mu_t, mu_c = float(np.mean(x_t)), float(np.mean(x_c))
    var_t, var_c = float(np.var(x_t, ddof=1)), float(np.var(x_c, ddof=1))
    se = float(np.sqrt(max(var_t / max(x_t.size, 1) + var_c / max(x_c.size, 1), EPS)))
    z = 0.0 if se == 0.0 else (mu_t - mu_c) / se
    return float(z_two_tailed(z))


# --------------------------------------------------------------------------------------
# FeatureSpec — JSON-serializable plans for the materializer
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class FeatureSpec:
    name: str
    kind: str
    base_cols: Tuple[str, ...]
    params: Dict[str, Any]
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["base_cols"] = list(self.base_cols)
        return d


# --------------------------------------------------------------------------------------
# Engineered topic builder (planner)
# --------------------------------------------------------------------------------------

def build_engineered_candidates(
    df: pd.DataFrame,
    nums: List[str],
    cats: List[str],
    time_col: Optional[str],
    *,
    id_col: Optional[str] = None,
) -> List[TopicRow]:
    """
    Returns TopicRow plans with effect_detail.feature_plan describing *what to build*.
    The materializer will compute the actual tables/series and KPIs.

    This planner:
      - Handles numeric, categorical, boolean, text/string, datetime
      - Proposes transforms, ratios, deltas, interactions
      - Plans time-derived features, inter-arrival, rolling & ewm
      - Screens correlation/association across pairings (num/num, num/cat, num/bool, cat/cat, bool/bool)
      - Emits lightweight causal screens (A/B, DiD)
      - Keeps time/memory bounded via caps and sampling

    NOTE: All heavy work (tokenization, windows, groupbys) is deferred.
    """
    out: List[TopicRow] = []

    # ---- Detect roles beyond provided nums/cats
    # booleans
    bools_all = [c for c in df.columns if _is_bool_series(df[c])]
    bools = [c for c in bools_all if c not in cats]  # treat separately but also usable as cats
    if len(bools) > MAX_BOOL_FOR_PAIRWISE:
        bools = bools[:MAX_BOOL_FOR_PAIRWISE]

    # text columns
    text_cols = [c for c in df.columns if _is_text_series(df, c)]

    # datetime columns (beyond time_col)
    dt_candidates = [c for c in df.columns if _parseable_datetime(df[c])]
    if not time_col:
        time_col = find_time_fallback(df)

    # filter numeric & categorical to safe sets
    nums_f = [c for c in nums if c in df.columns and good_metric_col(df, c)]
    cats_f = [c for c in cats if c in df.columns and good_category_col(df, c)]
    if len(nums_f) > MAX_NUMS_FOR_PAIRWISE:
        nums_f = sorted(
            nums_f,
            key=lambda c: (float(to_num(df[c]).notna().mean()), float(to_num(df[c]).std(skipna=True) or 0.0)),
            reverse=True,
        )[:MAX_NUMS_FOR_PAIRWISE]
    if len(cats_f) > MAX_CATS_FOR_PAIRWISE:
        cats_f = sorted(cats_f, key=lambda c: int(df[c].astype("string").nunique(dropna=True)))[:MAX_CATS_FOR_PAIRWISE]

    # ----------------------------------------------------------------------------------
    # 1) Numeric transforms (log1p/sqrt/winsor-zscore) + trend variants
    # ----------------------------------------------------------------------------------
    for m in nums_f[:8]:
        s = to_num(df[m])
        if s.notna().sum() < MIN_NON_NULL:
            continue
        skew = float(s.skew(skipna=True))
        fe_plan = []

        if skew > 1.0:
            fe_plan.append(FeatureSpec(f"log1p({m})", "transform", (m,), {"op": "log1p"}).to_dict())
        if 0.7 < skew <= 1.5:
            fe_plan.append(FeatureSpec(f"sqrt({m})", "transform", (m,), {"op": "sqrt", "clip_min": 0.0}).to_dict())
        fe_plan.append(
            FeatureSpec(f"zscore_wins({m})", "transform", (m,), {"op": "zscore_wins", "lower_q": 0.01, "upper_q": 0.99}).to_dict()
        )

        if fe_plan:
            out.append(
                TopicRow(
                    topic_id=topic_id("distribution_fe", [m]),
                    family="distribution",
                    primary_fields=(m,),
                    secondary_fields=(),
                    time_field=None,
                    n_obs=int(s.notna().sum()),
                    coverage_pct=float(s.notna().mean()),
                    effect_size=abs(skew),
                    effect_detail={"feature_plan": fe_plan, "skew": skew, "scenario_tags": ["dist_summary"]},
                    significance={},
                    causal_design=None,
                    assumptions_met=None,
                    readability=1.0,
                    complexity_penalty=0.06,
                    proposed_charts=("histogram", "boxplot", "violin"),
                )
            )
            if time_col:
                out.append(
                    TopicRow(
                        topic_id=topic_id("trend_fe", [time_col, m]),
                        family="trend",
                        primary_fields=(m,),
                        secondary_fields=(),
                        time_field=time_col,
                        n_obs=int(len(df)),
                        coverage_pct=float(s.notna().mean()),
                        effect_size=abs(skew),
                        effect_detail={"feature_plan": fe_plan, "scenario_tags": ["trend"]},
                        significance={},
                        causal_design=None,
                        assumptions_met=None,
                        readability=1.0,
                        complexity_penalty=0.12,
                        proposed_charts=("line", "column", "area"),
                    )
                )

    # ----------------------------------------------------------------------------------
    # 2) Ratios, deltas, and simple interactions
    # ----------------------------------------------------------------------------------
    denom_cands = [c for c in nums_f if re.search(r"(total|overall|base|denom|_all$|_total$)", str(c), re.I)]
    num_cands   = [c for c in nums_f if re.search(r"(success|win|click|conv|count|num|passed|ok)", str(c), re.I)]

    for numc in num_cands[:5]:
        s_num = to_num(df[numc])
        for denc in denom_cands[:5]:
            if denc == numc:
                continue
            s_den = to_num(df[denc])
            if s_den.notna().sum() < MIN_NON_NULL:
                continue
            if float((s_den > 0).mean()) < 0.5:
                continue
            if float(s_num.mean() or 0) > float(s_den.mean() or 0):
                continue
            name = f"rate_{numc}_over_{denc}"
            fe_plan = [FeatureSpec(name, "binary_op", (numc, denc), {"op": "ratio", "epsilon": EPS}).to_dict()]
            fam = "trend" if time_col else "distribution"
            out.append(
                TopicRow(
                    topic_id=topic_id("rate", [numc, denc]),
                    family=fam,
                    primary_fields=(numc,),
                    secondary_fields=(denc,),
                    time_field=time_col,
                    n_obs=int(min(s_num.notna().sum(), s_den.notna().sum())),
                    coverage_pct=float(min(s_num.notna().mean(), s_den.notna().mean())),
                    effect_size=0.0,
                    effect_detail={"feature_plan": fe_plan, "scenario_tags": ["rate", "kpi"]},
                    significance={},
                    causal_design=None,
                    assumptions_met=None,
                    readability=1.0,
                    complexity_penalty=0.08,
                    proposed_charts=("line", "column") if time_col else ("histogram", "boxplot"),
                )
            )

    # deltas and simple products (limited)
    for x, y in it.combinations(nums_f[:6], 2):
        if int(to_num(df[x]).notna().sum()) < MIN_NON_NULL or int(to_num(df[y]).notna().sum()) < MIN_NON_NULL:
            continue
        fe_plan = [
            FeatureSpec(f"delta_{x}_minus_{y}", "binary_op", (x, y), {"op": "diff"}).to_dict(),
            FeatureSpec(f"prod_{x}_times_{y}", "binary_op", (x, y), {"op": "prod"}).to_dict(),
        ]
        out.append(
            TopicRow(
                topic_id=topic_id("num_interactions", [x, y]),
                family="distribution" if not time_col else "trend",
                primary_fields=(x, y),
                secondary_fields=(),
                time_field=time_col,
                n_obs=int(len(df)),
                coverage_pct=float(min(to_num(df[x]).notna().mean(), to_num(df[y]).notna().mean())),
                effect_size=0.0,
                effect_detail={"feature_plan": fe_plan, "scenario_tags": ["feature_interaction"]},
                significance={},
                causal_design=None,
                assumptions_met=None,
                readability=1.0,
                complexity_penalty=0.07,
                proposed_charts=("line", "column") if time_col else ("histogram",),
            )
        )

    # ----------------------------------------------------------------------------------
    # 3) Time-derived features: calendar breakdown, inter-arrival
    # ----------------------------------------------------------------------------------
    if time_col:
        fe_time = [
            FeatureSpec("day_of_week", "time_derive", (time_col,), {"unit": "dow"}).to_dict(),
            FeatureSpec("hour_of_day", "time_derive", (time_col,), {"unit": "hour"}).to_dict(),
            FeatureSpec("week", "time_derive", (time_col,), {"unit": "week"}).to_dict(),
            FeatureSpec("month", "time_derive", (time_col,), {"unit": "month"}).to_dict(),
            FeatureSpec("quarter", "time_derive", (time_col,), {"unit": "quarter"}).to_dict(),
            FeatureSpec("year", "time_derive", (time_col,), {"unit": "year"}).to_dict(),
            FeatureSpec("is_weekend", "time_derive", (time_col,), {"unit": "is_weekend"}).to_dict(),
        ]
        out.append(
            TopicRow(
                topic_id=topic_id("calendar_breakdown", [time_col]),
                family="ranking",
                primary_fields=(time_col,),
                secondary_fields=(),
                time_field=time_col,
                n_obs=int(len(df)),
                coverage_pct=1.0,
                effect_size=0.0,
                effect_detail={"feature_plan": fe_time, "scenario_tags": ["calendar", "segmentation"]},
                significance={},
                causal_design=None,
                assumptions_met=None,
                readability=1.0,
                complexity_penalty=0.04,
                proposed_charts=("ordered_bar", "lollipop"),
            )
        )

        # inter-arrival (global and by id if available)
        inter_plan = [FeatureSpec("interarrival_seconds", "time_delta", (time_col,), {"by": None}).to_dict()]
        if id_col and id_col in df.columns:
            inter_plan.append(
                FeatureSpec("interarrival_seconds_by_id", "time_delta", (time_col, id_col), {"by": id_col}).to_dict()
            )
        out.append(
            TopicRow(
                topic_id=topic_id("interarrival", [time_col] + ([id_col] if id_col else [])),
                family="distribution",
                primary_fields=(time_col,) if not id_col else (time_col, id_col),
                secondary_fields=(),
                time_field=time_col,
                n_obs=int(len(df)),
                coverage_pct=1.0,
                effect_size=0.0,
                effect_detail={"feature_plan": inter_plan, "scenario_tags": ["flow", "ops"]},
                significance={},
                causal_design=None,
                assumptions_met=None,
                readability=1.0,
                complexity_penalty=0.06,
                proposed_charts=("histogram", "boxplot", "violin"),
            )
        )

    # ----------------------------------------------------------------------------------
    # 4) Windows (global, per-entity, per-top-category)
    # ----------------------------------------------------------------------------------
    if time_col:
        # global windows
        for m in nums_f[:8]:
            s = to_num(df[m])
            if s.notna().sum() < MIN_NON_NULL:
                continue
            slope, npts = trend_slope(df.sort_values(by=time_col, kind="mergesort")[m])
            fe_plan = [
                FeatureSpec(f"lag1_{m}", "window", (m,), {"op": "lag", "by": None, "sort": time_col, "periods": 1}).to_dict(),
                FeatureSpec(f"lead1_{m}", "window", (m,), {"op": "lead", "by": None, "sort": time_col, "periods": 1}).to_dict(),
                FeatureSpec(f"roll7_mean_{m}", "window", (m,), {"op": "rolling_mean", "by": None, "sort": time_col, "window": 7, "min_periods": 3}).to_dict(),
                FeatureSpec(f"ewm_a0.2_{m}", "window", (m,), {"op": "ewm_mean", "by": None, "sort": time_col, "alpha": 0.2, "adjust": False}).to_dict(),
                FeatureSpec(f"acorr_{m}", "window", (m,), {"op": "autocorr_lags", "by": None, "sort": time_col, "lags": [7, 30]}).to_dict(),
            ]
            out.append(
                TopicRow(
                    topic_id=topic_id("ts_windows", [time_col, m]),
                    family="trend",
                    primary_fields=(m,),
                    secondary_fields=(),
                    time_field=time_col,
                    n_obs=int(npts),
                    coverage_pct=float(s.notna().mean()),
                    effect_size=float(abs(slope)),
                    effect_detail={"feature_plan": fe_plan, "slope_norm": slope, "scenario_tags": ["trend", "seasonality"]},
                    significance={},
                    causal_design=None,
                    assumptions_met=None,
                    readability=1.0,
                    complexity_penalty=0.10,
                    proposed_charts=("line", "column", "area"),
                )
            )

        # per entity
        if id_col and id_col in df.columns:
            for m in nums_f[:6]:
                s = to_num(df[m])
                if s.notna().sum() < MIN_NON_NULL:
                    continue
                fe_plan = [
                    FeatureSpec(f"lag1_{m}_by_{id_col}", "window", (m, id_col), {"op": "lag", "by": id_col, "sort": time_col, "periods": 1}).to_dict(),
                    FeatureSpec(f"roll3_mean_{m}_by_{id_col}", "window", (m, id_col), {"op": "rolling_mean", "by": id_col, "sort": time_col, "window": 3, "min_periods": 2}).to_dict(),
                    FeatureSpec(f"roll3_sum_{m}_by_{id_col}", "window", (m, id_col), {"op": "rolling_sum", "by": id_col, "sort": time_col, "window": 3, "min_periods": 2}).to_dict(),
                    FeatureSpec(f"prev_event_dt_by_{id_col}", "time_delta", (time_col, id_col), {"by": id_col}).to_dict(),
                ]
                out.append(
                    TopicRow(
                        topic_id=topic_id("entity_ts_windows", [id_col, time_col, m]),
                        family="trend",
                        primary_fields=(m,),
                        secondary_fields=(id_col,),
                        time_field=time_col,
                        n_obs=int(len(df)),
                        coverage_pct=float(s.notna().mean()),
                        effect_size=0.0,
                        effect_detail={"feature_plan": fe_plan, "scenario_tags": ["entity_trend"]},
                        significance={},
                        causal_design=None,
                        assumptions_met=None,
                        readability=0.95,
                        complexity_penalty=0.12,
                        proposed_charts=("line", "column"),
                    )
                )

        # per top category levels
        for c in cats_f[:MAX_TOP_LEVELS_FOR_TIME]:
            lvls = _top_k_categories(df[c], k=min(5, MAX_TOP_LEVELS_FOR_TIME))
            if not lvls:
                continue
            for m in nums_f[:4]:
                s = to_num(df[m])
                if s.notna().sum() < MIN_NON_NULL:
                    continue
                fe_plan = [
                    FeatureSpec(f"lag1_{m}_by_{c}", "window", (m, c), {"op": "lag", "by": c, "sort": time_col, "periods": 1, "keep_levels": lvls}).to_dict(),
                    FeatureSpec(f"roll7_mean_{m}_by_{c}", "window", (m, c), {"op": "rolling_mean", "by": c, "sort": time_col, "window": 7, "min_periods": 3, "keep_levels": lvls}).to_dict(),
                ]
                out.append(
                    TopicRow(
                        topic_id=topic_id("cat_ts_windows", [c, time_col, m]),
                        family="trend",
                        primary_fields=(m,),
                        secondary_fields=(c,),
                        time_field=time_col,
                        n_obs=int(len(df)),
                        coverage_pct=float(s.notna().mean()),
                        effect_size=0.0,
                        effect_detail={"feature_plan": fe_plan, "levels": lvls, "scenario_tags": ["segmented_trend"]},
                        significance={},
                        causal_design=None,
                        assumptions_met=None,
                        readability=0.95,
                        complexity_penalty=0.10,
                        proposed_charts=("line", "column"),
                    )
                )

    # ----------------------------------------------------------------------------------
    # 5) Text features: length, word count, top tokens (plan via sampling)
    # ----------------------------------------------------------------------------------
    for tcol in text_cols:
        s = df[tcol].astype("string")
        # sample at most MAX_TEXT_SAMPLE to estimate top tokens
        if len(s) > MAX_TEXT_SAMPLE:
            s_samp = s.sample(MAX_TEXT_SAMPLE, random_state=0, ignore_index=True)
        else:
            s_samp = s

        # cheap tokenization by whitespace; plan only (materializer will compute fully)
        # get rough top tokens from sample to build the plan
        toks = (
            s_samp.dropna()
            .str.lower()
            .str.replace(r"[^\w\s]", " ", regex=True)
            .str.split()
        )
        # frequency via explode (bounded by sample)
        try:
            vc = toks.explode().value_counts()
            top_tokens = [str(tok) for tok in vc.index[:MAX_TOKENS_PER_TEXTCOL].tolist() if isinstance(tok, str)]
        except Exception:
            top_tokens = []

        fe_plan = [
            FeatureSpec(f"len({tcol})", "transform", (tcol,), {"op": "text_len"}).to_dict(),
            FeatureSpec(f"words({tcol})", "transform", (tcol,), {"op": "text_word_count"}).to_dict(),
            FeatureSpec(f"tokens_topk({tcol})", "text_tokens", (tcol,), {"token_list": top_tokens, "lower": True}).to_dict(),
            FeatureSpec(f"unique_token_ratio({tcol})", "text_summary", (tcol,), {"op": "unique_token_ratio"}).to_dict(),
        ]
        out.append(
            TopicRow(
                topic_id=topic_id("text_fe", [tcol]),
                family="distribution",
                primary_fields=(tcol,),
                secondary_fields=(),
                time_field=time_col,
                n_obs=int(len(df)),
                coverage_pct=float(s.notna().mean()),
                effect_size=float(len(top_tokens)),
                effect_detail={"feature_plan": fe_plan, "scenario_tags": ["text_summary", "nlp_light"]},
                significance={},
                causal_design=None,
                assumptions_met=None,
                readability=0.95,
                complexity_penalty=0.08,
                proposed_charts=("ordered_bar", "lollipop"),
            )
        )

    # ----------------------------------------------------------------------------------
    # 6) Pairwise screens across ALL types
    # ----------------------------------------------------------------------------------
    # num-num (pearson)
    for x, y in it.combinations(nums_f, 2):
        r, n, p = safe_pearson(df[x], df[y])
        if n >= MIN_NON_NULL and (abs(r) >= 0.25 or p < 0.05):
            out.append(
                TopicRow(
                    topic_id=topic_id("corr_num_num", [x, y]),
                    family="correlation",
                    primary_fields=(x, y),
                    secondary_fields=(),
                    time_field=None,
                    n_obs=n,
                    coverage_pct=float(n / max(len(df), 1)),
                    effect_size=float(abs(r)),
                    effect_detail={"r": r, "scenario_tags": ["correlation"]},
                    significance={"p_value": float(p), "test": "pearson"},
                    causal_design=None,
                    assumptions_met=None,
                    readability=1.0,
                    complexity_penalty=0.14,
                    proposed_charts=("scatter", "xy_heatmap"),
                )
            )

    # num-cat (anova)
    for m in nums_f[:8]:
        for c in cats_f[:10]:
            eta2, k, n, F, p = anova_oneway(df[m], df[c])
            if n >= MIN_NON_NULL and k >= 2 and (eta2 >= 0.05 or p < 0.05):
                out.append(
                    TopicRow(
                        topic_id=topic_id("assoc_num_cat", [m, c]),
                        family="correlation",
                        primary_fields=(m, c),
                        secondary_fields=(),
                        time_field=None,
                        n_obs=n,
                        coverage_pct=float(n / max(len(df), 1)),
                        effect_size=float(eta2),
                        effect_detail={"eta2": eta2, "F": F, "scenario_tags": ["segmentation"]},
                        significance={"p_value": float(p), "test": "anova"},
                        causal_design=None,
                        assumptions_met=None,
                        readability=1.0,
                        complexity_penalty=0.12,
                        proposed_charts=("ordered_bar", "lollipop", "boxplot"),
                    )
                )

    # cat-cat (Cramér’s V)
    for a, b in it.combinations(cats_f, 2):
        v, n, dof, p, chi2 = cramers_v_with_p(df[a], df[b])
        if n >= MIN_NON_NULL and dof > 0 and (v >= 0.15 or p < 0.05):
            out.append(
                TopicRow(
                    topic_id=topic_id("assoc_cat_cat", [a, b]),
                    family="correlation",
                    primary_fields=(a, b),
                    secondary_fields=(),
                    time_field=None,
                    n_obs=n,
                    coverage_pct=float(n / max(len(df), 1)),
                    effect_size=float(v),
                    effect_detail={"cramers_v": v, "chi2": chi2, "dof": dof, "scenario_tags": ["association"]},
                    significance={"p_value": float(p), "test": "chi2"},
                    causal_design=None,
                    assumptions_met=None,
                    readability=1.0,
                    complexity_penalty=0.10,
                    proposed_charts=("table",),
                )
            )

    # bool-bool (chi2 on 2x2)
    bool_as_cat = [b for b in bools if b in df.columns]
    for a, b in it.combinations(bool_as_cat, 2):
        v, n, dof, p, chi2 = cramers_v_with_p(df[a], df[b])
        if n >= MIN_NON_NULL and dof > 0 and (v >= 0.1 or p < 0.05):
            out.append(
                TopicRow(
                    topic_id=topic_id("assoc_bool_bool", [a, b]),
                    family="correlation",
                    primary_fields=(a, b),
                    secondary_fields=(),
                    time_field=None,
                    n_obs=n,
                    coverage_pct=float(n / max(len(df), 1)),
                    effect_size=float(v),
                    effect_detail={"cramers_v": v, "chi2": chi2, "dof": dof, "scenario_tags": ["association"]},
                    significance={"p_value": float(p), "test": "chi2"},
                    causal_design=None,
                    assumptions_met=None,
                    readability=1.0,
                    complexity_penalty=0.09,
                    proposed_charts=("table",),
                )
            )

    # num-bool (Welch screen)
    for m in nums_f[:10]:
        for b in bool_as_cat[:10]:
            p = _welch_p(df[m], df[b])
            if p < 0.05:
                out.append(
                    TopicRow(
                        topic_id=topic_id("assoc_num_bool", [m, b]),
                        family="correlation",
                        primary_fields=(m, b),
                        secondary_fields=(),
                        time_field=None,
                        n_obs=int(len(df)),
                        coverage_pct=float(to_num(df[m]).notna().mean()),
                        effect_size=0.0,
                        effect_detail={"scenario_tags": ["segmentation"]},
                        significance={"p_value": float(p), "test": "welch_normal_approx"},
                        causal_design=None,
                        assumptions_met=None,
                        readability=1.0,
                        complexity_penalty=0.08,
                        proposed_charts=("boxplot", "violin", "bar"),
                    )
                )

    # ----------------------------------------------------------------------------------
    # 7) Ranking & Part-to-Whole (sum-by-cat; counts)
    # ----------------------------------------------------------------------------------
    m_rank = best_metric(df, nums_f)
    if m_rank:
        for c in cats_f[:12]:
            grp = pd.concat([df[[c]].astype("string"), to_num(df[m_rank])], axis=1).dropna()
            if grp.empty:
                continue
            s = grp.groupby(c)[m_rank].sum().sort_values(ascending=False)
            pct = shares(s)
            out.append(
                TopicRow(
                    topic_id=topic_id("ranking_fe", [c, m_rank]),
                    family="ranking",
                    primary_fields=(c, m_rank),
                    secondary_fields=(),
                    time_field=None,
                    n_obs=int(len(s)),
                    coverage_pct=float(len(grp) / max(len(df), 1)),
                    effect_size=float(np.sum(pct ** 2)),
                    effect_detail={"top": list(s.index[:5]), "scenario_tags": ["ranking"]},
                    significance={},
                    causal_design=None,
                    assumptions_met=None,
                    readability=1.0,
                    complexity_penalty=0.04,
                    proposed_charts=("ordered_bar", "ordered_column", "lollipop"),
                )
            )
            out.append(
                TopicRow(
                    topic_id=topic_id("part_to_whole_fe", [c, m_rank]),
                    family="part_to_whole",
                    primary_fields=(c, m_rank),
                    secondary_fields=(),
                    time_field=None,
                    n_obs=int(len(s)),
                    coverage_pct=float(len(grp) / max(len(df), 1)),
                    effect_size=float(max(pct) if len(pct) else 0.0),
                    effect_detail={"gini": float(gini(pct)), "scenario_tags": ["composition"]},
                    significance={},
                    causal_design=None,
                    assumptions_met=None,
                    readability=1.0,
                    complexity_penalty=0.04,
                    proposed_charts=("treemap", "pie_guarded", "donut_guarded"),
                )
            )

    # simple counts for cats & bools (frequency tables)
    for c in (cats_f + bool_as_cat)[:16]:
        out.append(
            TopicRow(
                topic_id=topic_id("count_table", [c, "__row_count__"]),
                family="ranking",
                primary_fields=(c, "__row_count__"),
                secondary_fields=(),
                time_field=None,
                n_obs=int(len(df)),
                coverage_pct=1.0,
                effect_size=0.0,
                effect_detail={"metric": "count", "scenario_tags": ["frequency"]},
                significance={},
                causal_design=None,
                assumptions_met=None,
                readability=1.0,
                complexity_penalty=0.03,
                proposed_charts=("ordered_bar", "lollipop"),
            )
        )

    # ----------------------------------------------------------------------------------
    # 8) Lightweight causal screens (A/B & DiD hints) using bool/categorical treatment
    # ----------------------------------------------------------------------------------
    treat_like: List[str] = []
    for c in cats_f + bool_as_cat:
        name = str(c).lower()
        if any(k in name for k in ("treat", "variant", "group", "ab", "bucket", "experiment", "arm")):
            treat_like.append(c)

    for tr in treat_like[:4]:
        # A/B for each numeric
        for y in nums_f[:6]:
            out.append(
                TopicRow(
                    topic_id=topic_id("causal_ab_fe", [tr, y]),
                    family="causal",
                    primary_fields=(y,),
                    secondary_fields=(tr,),
                    time_field=None,
                    n_obs=int(len(df)),
                    coverage_pct=float(to_num(df[y]).notna().mean()),
                    effect_size=0.0,
                    effect_detail={
                        "feature_plan": [
                            FeatureSpec(f"ab_diff_{y}_by_{tr}", "causal_ab", (y, tr), {"assess": "diff_in_means"}).to_dict()
                        ],
                        "scenario_tags": ["causal_screen_ab"],
                    },
                    significance={},
                    causal_design="ab",
                    assumptions_met=["randomization (heuristic)"],
                    readability=0.95,
                    complexity_penalty=0.15,
                    proposed_charts=("bar", "column"),
                )
            )
        if time_col:
            for y in nums_f[:4]:
                out.append(
                    TopicRow(
                        topic_id=topic_id("causal_did_fe", [tr, time_col, y]),
                        family="causal",
                        primary_fields=(y,),
                        secondary_fields=(tr, time_col),
                        time_field=time_col,
                        n_obs=int(len(df)),
                        coverage_pct=float(to_num(df[y]).notna().mean()),
                        effect_size=0.0,
                        effect_detail={
                            "feature_plan": [
                                FeatureSpec(
                                    f"did_{y}_by_{tr}_over_{time_col}",
                                    "causal_did",
                                    (y, tr, time_col),
                                    {"pre_post": "median_time_split"},
                                ).to_dict()
                            ],
                            "scenario_tags": ["causal_screen_did"],
                        },
                        significance={},
                        causal_design="did",
                        assumptions_met=["parallel trends (heuristic)"],
                        readability=0.9,
                        complexity_penalty=0.18,
                        proposed_charts=("bar", "column"),
                    )
                )

    return out

