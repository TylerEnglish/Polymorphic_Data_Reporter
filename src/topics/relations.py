from __future__ import annotations
from typing import List, Optional, Dict, Any
import itertools as it
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
    cohort_table,
)

def build_base_candidates(
    df: pd.DataFrame,
    *,
    nums: List[str],
    cats: List[str],
    bools: List[str],
    time_col: Optional[str],
    id_col: Optional[str],
    max_bar_cats: int,
) -> List[TopicRow]:
    out: List[TopicRow] = []

    # ---------------- KPI ----------------
    for m in nums:
        s = to_num(df[m]); n = int(s.notna().sum())
        if n < 10: continue
        out.append(TopicRow(
            topic_id=topic_id("kpi", [m]),
            family="kpi",
            primary_fields=(m,), secondary_fields=(), time_field=None,
            n_obs=n, coverage_pct=float(n / max(len(df), 1)), effect_size=0.0,
            effect_detail={"mean": float(s.mean(skipna=True)), "sum": float(s.sum(skipna=True))},
            significance={}, causal_design=None, assumptions_met=None,
            readability=1.0, complexity_penalty=0.0,
            proposed_charts=("kpi","table")
        ))

    # --------------- Trend ---------------
    if time_col:
        t = pd.to_datetime(df[time_col], errors="coerce")
        g = df.copy(); g[time_col] = pd.to_datetime(t.dt.to_period("D").dt.to_timestamp())
        for m in nums:
            ser = to_num(g[m])
            if ser.notna().sum() < 10: continue
            agg = g.groupby(time_col, dropna=True, sort=True)[m].mean()
            slope, npts = trend_slope(agg.reset_index(drop=True))
            out.append(TopicRow(
                topic_id=topic_id("trend", [time_col, m]),
                family="trend",
                primary_fields=(m,), secondary_fields=(), time_field=time_col,
                n_obs=int(len(agg)), coverage_pct=float(ser.notna().mean()),
                effect_size=float(abs(slope)),
                effect_detail={"slope_norm": float(slope), "n_points": int(npts)},
                significance={}, causal_design=None, assumptions_met=None,
                readability=1.0, complexity_penalty=0.1 if int(len(agg))>180 else 0.0,
                proposed_charts=("line","column","area","calendar_heatmap")
            ))

    # --------- Correlation: num-num ---------
    for x, y in it.combinations(nums, 2):
        r, n, p = safe_pearson(df[x], df[y])
        if n < 10: continue
        out.append(TopicRow(
            topic_id=topic_id("correlation", [x, y]),
            family="correlation",
            primary_fields=(x, y), secondary_fields=(), time_field=None,
            n_obs=n, coverage_pct=float(n / max(len(df),1)),
            effect_size=float(abs(r)), effect_detail={"r_pearson": float(r)},
            significance={"p_value": float(p), "test": "pearson"},
            causal_design=None, assumptions_met=None,
            readability=1.0, complexity_penalty=0.2,
            proposed_charts=("scatter","bubble","xy_heatmap")
        ))

    # -------- Correlation: num-cat (ANOVA) --------
    for m in nums:
        for c in cats:
            if df[c].nunique(dropna=True) > max_bar_cats:
                continue
            eta2, k, n, F, p = anova_oneway(df[m], df[c])
            if n < 10: continue
            out.append(TopicRow(
                topic_id=topic_id("assoc_num_cat", [m, c]),
                family="correlation",
                primary_fields=(m, c), secondary_fields=(), time_field=None,
                n_obs=n, coverage_pct=float(n / max(len(df),1)),
                effect_size=float(eta2),
                effect_detail={"eta2": float(eta2), "groups": int(k), "F": float(F)},
                significance={"p_value": float(p), "test": "anova"},
                causal_design=None, assumptions_met=None,
                readability=1.0, complexity_penalty=0.15,
                proposed_charts=("ordered_bar","ordered_column","lollipop","boxplot","violin")
            ))

    # -------- Correlation: cat-cat (Cramér’s V) --------
    for a, b in it.combinations(cats, 2):
        if df[a].nunique(dropna=True) * df[b].nunique(dropna=True) > max_bar_cats ** 2:
            continue
        v, n, dof, pval, chi2 = cramers_v_with_p(df[a], df[b])
        if n < 10: continue
        out.append(TopicRow(
            topic_id=topic_id("assoc_cat_cat", [a, b]),
            family="correlation",
            primary_fields=(a, b), secondary_fields=(), time_field=None,
            n_obs=n, coverage_pct=float(n / max(len(df), 1)),
            effect_size=float(v),
            effect_detail={"cramers_v": float(v), "chi2": float(chi2), "dof": int(dof)},
            significance={"p_value": float(pval), "test": "chi2"},
            causal_design=None, assumptions_met=None,
            readability=1.0, complexity_penalty=0.2,
            proposed_charts=("table",)
        ))

    # --------------- Ranking ---------------
    if nums and cats:
        m = nums[0]
        for c in cats:
            if df[c].nunique(dropna=True) > max_bar_cats: continue
            grp = pd.concat([df[[c]].astype(str), to_num(df[m])], axis=1).dropna()
            if grp.empty: continue
            s = grp.groupby(c)[m].sum().sort_values(ascending=False)
            sh = shares(s)
            hhi = float(np.sum(sh ** 2))
            out.append(TopicRow(
                topic_id=topic_id("ranking", [c, m]),
                family="ranking",
                primary_fields=(c, m), secondary_fields=(), time_field=None,
                n_obs=int(len(s)), coverage_pct=float(len(grp)/max(len(df),1)),
                effect_size=float(hhi), effect_detail={"hhi": float(hhi), "top": list(s.index[:5])},
                significance={}, causal_design=None, assumptions_met=None,
                readability=1.0, complexity_penalty=0.05 + 0.02 * max(0, int(len(s)) - 10),
                proposed_charts=("ordered_bar","ordered_column","lollipop","dot_strip_plot")
            ))

    # --------- Part-to-whole ---------
    if nums and cats:
        m = nums[0]
        for c in cats:
            if df[c].nunique(dropna=True) > max_bar_cats: continue
            grp = pd.concat([df[[c]].astype(str), to_num(df[m])], axis=1).dropna()
            if grp.empty: continue
            totals = grp.groupby(c)[m].sum()
            pct = shares(totals); G = gini(pct)
            out.append(TopicRow(
                topic_id=topic_id("part_to_whole", [c, m]),
                family="part_to_whole",
                primary_fields=(c, m), secondary_fields=(), time_field=None,
                n_obs=int(len(pct)), coverage_pct=float(len(grp)/max(len(df),1)),
                effect_size=float(max(pct) if len(pct) else 0.0),
                effect_detail={"gini": float(G)},
                significance={}, causal_design=None, assumptions_met=None,
                readability=1.0, complexity_penalty=0.05,
                proposed_charts=("treemap","pie_guarded","donut_guarded")
            ))

    # --------------- Distribution ---------------
    for m in nums:
        s = to_num(df[m])
        if s.notna().sum() < 10: continue
        q = s.quantile([0.25, 0.5, 0.75]).to_dict()
        skew = float(s.skew(skipna=True)); kurt = float(s.kurtosis(skipna=True))
        iqr = float(q.get(0.75, 0.0) - q.get(0.25, 0.0))
        out.append(TopicRow(
            topic_id=topic_id("distribution", [m]),
            family="distribution",
            primary_fields=(m,), secondary_fields=(), time_field=None,
            n_obs=int(s.notna().sum()), coverage_pct=float(s.notna().mean()),
            effect_size=float(iqr / (abs(float(s.mean() or 0.0)) + 1e-12)),
            effect_detail={"iqr": iqr, "skew": skew, "kurtosis": kurt},
            significance={}, causal_design=None, assumptions_met=None,
            readability=1.0, complexity_penalty=0.05,
            proposed_charts=("histogram","boxplot","violin","dot_strip_plot","beeswarm","barcode_plot","frequency_polygon","cumulative_curve")
        ))

    # --------------- Deviation ---------------
    for m in nums:
        s = to_num(df[m]); mu, sd = float(s.mean(skipna=True) or 0.0), float(s.std(skipna=True) or 0.0)
        if sd <= 0 or s.notna().sum() < 10: continue
        zmax = float(((s - mu) / sd).abs().max(skipna=True))
        out.append(TopicRow(
            topic_id=topic_id("deviation", [m]),
            family="deviation",
            primary_fields=(m,), secondary_fields=(), time_field=None,
            n_obs=int(s.notna().sum()), coverage_pct=float(s.notna().mean()),
            effect_size=float(zmax), effect_detail={"zmax": float(zmax)},
            significance={}, causal_design=None, assumptions_met=None,
            readability=1.0, complexity_penalty=0.05,
            proposed_charts=("diverging_bar",)
        ))

    # --------------- Cohort ---------------
    if time_col and id_col and id_col in df.columns:
        tab = cohort_table(df, time_col, id_col)
        if tab is not None and not tab.empty:
            area = float((tab["retention_pct"].fillna(0.0)).mean())
            out.append(TopicRow(
                topic_id=topic_id("cohort", [id_col, time_col]),
                family="cohort",
                primary_fields=(id_col, time_col), secondary_fields=(), time_field=time_col,
                n_obs=int(tab.shape[0]), coverage_pct=1.0,
                effect_size=float(area), effect_detail={"cohorts": int(tab["_first_seen"].nunique())},
                significance={}, causal_design=None, assumptions_met=None,
                readability=0.9, complexity_penalty=0.1,
                proposed_charts=("table",)
            ))

    # (Light causation can live here too if you want; omitted to keep diff focused)
    return out
