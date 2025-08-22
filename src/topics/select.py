from __future__ import annotations
import json
import pandas as pd
from src.config_model.model import RootCfg


def _parse_sig(sig_val):
    """Return a dict from significance cell that may be a JSON string or dict."""
    if isinstance(sig_val, dict):
        return sig_val
    if isinstance(sig_val, str):
        try:
            return json.loads(sig_val)
        except Exception:
            return {}
    return {}


def select_topics(scored: pd.DataFrame, cfg: RootCfg) -> pd.DataFrame:
    if scored.empty:
        return scored

    th = cfg.topics.thresholds
    keep = scored.copy()

    # --- family-specific gates ------------------------------------------------
    # Correlation: only gate Pearson (num–num) by correlation threshold
    is_corr = keep["family"] == "correlation"
    is_pearson = keep.get("significance", pd.Series([None] * len(keep))).apply(
        lambda s: _parse_sig(s).get("test") == "pearson"
    )
    keep = keep[~(is_corr & is_pearson & (keep["effect_size"] < float(th.min_corr_for_scatter)))]

    # Trend: require at least minimal normalized slope (raw >= min_slope)
    # effect_norm was computed as (slope / (3 * min_slope)), so the gate is 1/3.
    try:
        min_slope = float(th.min_slope_for_trend)
    except Exception:
        min_slope = 0.0
    denom = max(3.0 * min_slope, 1e-12)
    min_norm = (min_slope / denom) if min_slope > 0 else 0.0  # = ~1/3 when min_slope > 0
    keep = keep[~((keep["family"] == "trend") & (keep["effect_norm"] < min_norm))]

    # Causal: ensure reasonable coverage
    keep = keep[~((keep["family"] == "causal") & (keep["coverage_pct"] < 0.7))]

    # --- remove near-duplicates: same family + same primary_fields ------------
    def _key(r: pd.Series):
        return (r["family"], tuple(r["primary_fields"]))

    seen = set()
    rows = []
    # prefer higher scores
    for _, r in keep.sort_values("score_total", ascending=False).iterrows():
        k = _key(r)
        if k in seen:
            continue
        seen.add(k)
        rows.append(r)
    keep = pd.DataFrame(rows)

    # --- cap to configured maximum -------------------------------------------
    keep = keep.sort_values("score_total", ascending=False).head(int(th.max_charts_total)).reset_index(drop=True)
    return keep
