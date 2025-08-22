from __future__ import annotations
import json
from typing import Any, Dict
import pandas as pd

from src.config_model.model import RootCfg


# --------------------------- small helpers -----------------------------------

def _cap01(x: float) -> float:
    try:
        return max(0.0, min(1.0, float(x)))
    except Exception:
        return 0.0

def _json_get(row: pd.Series, col: str) -> Dict[str, Any]:
    """Load a JSON dict from a dataframe cell; tolerate dict/None/garbage."""
    v = row.get(col, None)
    if isinstance(v, dict):
        return v
    if v is None:
        return {}
    try:
        return json.loads(v)
    except Exception:
        return {}


# --------------------------- family effect scaling ---------------------------

def _family_effect_norm(row: pd.Series, cfg: RootCfg) -> float:
    """
    Normalize effect sizes to ~[0,1] so cross-family scores are comparable.
    """
    fam = row["family"]
    es = float(row.get("effect_size", 0.0) or 0.0)

    if fam == "correlation":
        # |r|, η², V are already on ~0..1 scales
        return _cap01(es)

    if fam == "trend":
        # Compare against configured slope threshold; cap around ~3× the min
        denom = max(cfg.topics.thresholds.min_slope_for_trend * 3.0, 1e-12)
        return _cap01(es / denom)

    if fam in {"ranking", "part_to_whole", "cohort"}:
        return _cap01(es)

    if fam == "distribution":
        # iqr/|mean| can exceed 1; squash
        return _cap01(es / (1.0 + es))

    if fam == "deviation":
        # zmax: soft-saturate near 1 after ~3 SDs
        return _cap01(es / (3.0 + es))

    if fam == "causal":
        # candidates normalized by outcome SD already; mild squash
        return _cap01(es / (0.5 + es))

    if fam == "kpi":
        # keep KPIs low so they don't crowd out analytic topics
        return 0.2

    return 0.0


# ---------------------------- readability/complexity -------------------------

def _readability_penalty(row: pd.Series, cfg: RootCfg) -> float:
    fam = row["family"]

    if fam == "correlation" and int(row.get("n_obs", 0)) < 30:
        # small-n scatter/contingency can be noisy to read
        return 0.3

    if fam == "trend":
        # penalize very long series (dense charts)
        try:
            pts = int(_json_get(row, "effect_detail").get("n_points", 0))
        except Exception:
            pts = 0
        return 0.1 if pts > 180 else 0.0

    if fam == "cohort":
        # cohort tables are busier than single charts
        return 0.1

    return 0.0


# ------------------------------- signal quality ------------------------------

def _signal_quality(row: pd.Series, cfg: RootCfg) -> float:
    """
    Convert 'significance' → [0,1] signal quality.

    Priority:
    1) If a p-value exists, map to 1-p (so smaller p ⇒ higher quality).
    2) Otherwise, fall back to a gentle n_obs heuristic.
    """
    sig = _json_get(row, "significance")
    p = sig.get("p_value", None)

    # 1) Use p-value if available
    if p is not None:
        try:
            return _cap01(1.0 - float(p))
        except Exception:
            # if parsing fails, fall through to n_obs fallback
            pass

    # 2) Fallback: scale by sample size (soft signal proxy)
    n = float(row.get("n_obs", 0) or 0)
    return _cap01(n / 100.0)


# --------------------------------- scoring -----------------------------------

def score_topics(topics: pd.DataFrame, cfg: RootCfg) -> pd.DataFrame:
    """
    Add a `score_total` column using configurable weights:
      score = w_suitability * suitability
            + w_effect     * effect_norm
            + w_signal     * signal_quality
            + w_readability*(1 - readability_penalty)
            - w_complexity * complexity

    Also returns intermediate columns to aid testing/inspection:
      - effect_norm
      - suitability
      - readability_penalty
      - complexity
      - signal_quality
    """
    if topics.empty:
        # keep shape but add the expected columns
        return topics.assign(
            effect_norm=pd.Series(dtype=float),
            suitability=pd.Series(dtype=float),
            readability_penalty=pd.Series(dtype=float),
            complexity=pd.Series(dtype=float),
            signal_quality=pd.Series(dtype=float),
            score_total=pd.Series(dtype=float),
        )

    t = topics.copy()
    w = cfg.weights

    t["effect_norm"] = t.apply(lambda r: _family_effect_norm(r, cfg), axis=1)
    t["suitability"] = t["coverage_pct"].fillna(0.0).clip(0.0, 1.0)
    t["readability_penalty"] = t.apply(lambda r: _readability_penalty(r, cfg), axis=1)
    t["signal_quality"] = t.apply(lambda r: _signal_quality(r, cfg), axis=1)
    t["complexity"] = t["complexity_penalty"].fillna(0.0) + t["readability_penalty"]

    t["score_total"] = (
        w.suitability * t["suitability"].astype(float)
        + w.effect_size * t["effect_norm"].astype(float)
        + w.signal_quality * t["signal_quality"].astype(float)
        + w.readability * (1.0 - t["readability_penalty"].astype(float)).clip(0.0, 1.0)
        - w.complexity * t["complexity"].astype(float)
    )

    return t
