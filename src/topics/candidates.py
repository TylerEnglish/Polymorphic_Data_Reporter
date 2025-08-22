from __future__ import annotations
from typing import Any, Dict, List, Optional
import json
import pandas as pd

from src.config_model.model import RootCfg
from src.nlp.schema import ProposedSchema

from .model import TopicRow
from .fe import (
    is_id_like, is_uri_like,
    good_metric_col, good_category_col,
    find_time_fallback, best_metric,
    build_engineered_candidates,
)
from .relations import build_base_candidates

def _roles(schema: ProposedSchema) -> Dict[str, List[str]]:
    rmap = {"time": [], "numeric": [], "categorical": [], "boolean": [], "id": [], "text": []}
    for c in schema.columns:
        role = (c.role_confidence.role or "").lower()
        if role in rmap:
            rmap[role].append(c.name)
    return rmap

def build_candidates(df: pd.DataFrame, schema: ProposedSchema, cfg: RootCfg) -> pd.DataFrame:
    roles = _roles(schema)
    th = cfg.topics.thresholds
    max_bar_cats = int(th.max_categories_bar)

    # roles → concrete columns
    time_col = next((c for c in roles.get("time", []) if c in df.columns), None) or find_time_fallback(df)
    id_col   = next((c for c in roles.get("id",   []) if c in df.columns), None)
    nums_raw = [c for c in roles.get("numeric", []) if c in df.columns]
    cats_raw = [c for c in roles.get("categorical", []) if c in df.columns]
    bools    = [c for c in roles.get("boolean", []) if c in df.columns]

    # select useful metrics/categories
    cats = [c for c in cats_raw if good_category_col(df, c, max_bar_cats)]
    if time_col in cats:
        cats = [c for c in cats if c != time_col]
    if not cats:
        # fallback: scan for low-card strings
        for c in df.columns:
            if c in nums_raw or c in bools or is_id_like(c) or is_uri_like(c):
                continue
            s = df[c]
            if pd.api.types.is_numeric_dtype(s):  # already numeric
                continue
            try:
                if good_category_col(df, c, max_bar_cats):
                    cats.append(c)
            except Exception:
                continue
    nums = [c for c in nums_raw if good_metric_col(df, c)]

    # Base relations (KPI, trend, corr, ranking, p2w, distribution, deviation, cohort)
    base_rows: List[TopicRow] = build_base_candidates(
        df,
        nums=nums,
        cats=cats,
        bools=bools,
        time_col=time_col,
        id_col=id_col,
        max_bar_cats=max_bar_cats,
    )

    # Engineered topics (log1p, rate) – minimal set that current materializer can render
    fe_rows: List[TopicRow] = build_engineered_candidates(df, nums, cats, time_col)

    rows: List[Dict[str, Any]] = []
    for r in base_rows + fe_rows:
        rows.append({
            "topic_id": r.topic_id,
            "family": r.family,
            "primary_fields": list(r.primary_fields),
            "secondary_fields": list(r.secondary_fields),
            "time_field": r.time_field,
            "n_obs": r.n_obs,
            "coverage_pct": r.coverage_pct,
            "effect_size": r.effect_size,
            "effect_detail": json.dumps(r.effect_detail),
            "significance": json.dumps(r.significance),
            "causal_design": r.causal_design,
            "assumptions_met": json.dumps(list(r.assumptions_met) if r.assumptions_met else []),
            "readability": r.readability,
            "complexity_penalty": r.complexity_penalty,
            "proposed_charts": json.dumps(list(r.proposed_charts)),
        })
    return pd.DataFrame(rows)
