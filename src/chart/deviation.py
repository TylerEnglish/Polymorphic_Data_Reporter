from __future__ import annotations
from typing import Optional, Sequence
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .common import apply_theme, theme_from_cfg

def _ensure_df(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")
    return df

def diverging_bar(
    df: pd.DataFrame,
    *,
    category: str,
    value: str,
    reference: Optional[float] = None,
    title: Optional[str] = None,
    theme_name: Optional[str] = "dark_blue",
) -> go.Figure:
    _ensure_df(df, [category, value])
    ref = float(reference if reference is not None else df[value].mean())
    dd = df[[category, value]].copy()
    dd["_dev"] = dd[value] - ref
    dd = dd.sort_values("_dev", key=lambda s: s.abs(), ascending=False)

    fig = go.Figure()
    fig.add_bar(
        x=dd["_dev"],
        y=dd[category],
        orientation="h",
        marker=dict(
            color=np.where(dd["_dev"] >= 0, theme_from_cfg(theme_name)["primary"], theme_from_cfg(theme_name)["accent"])
        ),
        hovertemplate=f"{category}: %{{y}}<br>{value}: %{{x:+,.2f}} vs ref {ref:,.2f}<extra></extra>",
    )
    fig.add_vline(x=0, line=dict(color=theme_from_cfg(theme_name)["grid"], width=1))
    fig.update_layout(
        title=title or f"Diverging Bar — deviation from {ref:,.2f}",
        xaxis_title=f"{value} − reference",
        yaxis_title=category,
    )
    return apply_theme(fig, theme_from_cfg(theme_name))

def diverging_stacked_bar(
    df: pd.DataFrame,
    *,
    category: str,
    subcategory: str,
    value: str,
    reference: Optional[float] = None,
    title: Optional[str] = None,
    theme_name: Optional[str] = "dark_blue",
) -> go.Figure:
    _ensure_df(df, [category, subcategory, value])
    ref = float(reference if reference is not None else df[value].mean())

    dd = df[[category, subcategory, value]].copy()
    dd["_dev"] = dd[value] - ref
    dd = dd.sort_values([category, subcategory])

    fig = go.Figure()
    theme = theme_from_cfg(theme_name)
    # stack each subcategory with sign preserving barmode='relative'
    for i, (sub, g) in enumerate(dd.groupby(subcategory)):
        fig.add_bar(
            x=g["_dev"],
            y=g[category],
            name=str(sub),
            orientation="h",
        )
    fig.update_layout(
        barmode="relative",
        title=title or f"Diverging Stacked Bar — deviation from {ref:,.2f}",
        xaxis_title=f"{value} − reference",
        yaxis_title=category,
    )
    return apply_theme(fig, theme)

def spine(
    df: pd.DataFrame,
    *,
    category: str,
    pos_col: str,
    neg_col: str,
    title: Optional[str] = None,
    theme_name: Optional[str] = "dark_blue",
) -> go.Figure:
    _ensure_df(df, [category, pos_col, neg_col])
    dd = df[[category, pos_col, neg_col]].copy()
    s = dd[pos_col] + dd[neg_col]
    # normalize to 100%
    dd[pos_col] = (dd[pos_col] / s) * 100.0
    dd[neg_col] = (dd[neg_col] / s) * -100.0  # negative to center on 0

    fig = go.Figure()
    theme = theme_from_cfg(theme_name)
    fig.add_bar(
        x=dd[neg_col],
        y=dd[category],
        orientation="h",
        name=str(neg_col),
    )
    fig.add_bar(
        x=dd[pos_col],
        y=dd[category],
        orientation="h",
        name=str(pos_col),
    )
    fig.add_vline(x=0, line=dict(color=theme["grid"], width=1))
    fig.update_layout(
        barmode="relative",
        title=title or "Spine chart (100% centered)",
        xaxis_title="Percent",
        yaxis_title=category,
    )
    return apply_theme(fig, theme)

def surplus_deficit_line(
    df: pd.DataFrame,
    *,
    time: str,
    value: str,
    target: float | str,  # float or column name for varying target
    title: Optional[str] = None,
    theme_name: Optional[str] = "dark_blue",
) -> go.Figure:
    cols = [time, value] + ([target] if isinstance(target, str) else [])
    _ensure_df(df, cols)
    theme = theme_from_cfg(theme_name)

    dd = df[[time, value] + ([target] if isinstance(target, str) else [])].copy()
    dd = dd.sort_values(time)
    tgt = dd[target].values if isinstance(target, str) else np.full(len(dd), float(target))
    val = dd[value].values
    t = dd[time]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=val, mode="lines", name=value))
    fig.add_trace(go.Scatter(x=t, y=tgt, mode="lines", name="target", line=dict(dash="dash")))

    # Shade above/below
    above = np.where(val > tgt, val, tgt)
    below = np.where(val > tgt, tgt, val)
    fig.add_traces([
        go.Scatter(
            x=np.concatenate([t, t[::-1]]),
            y=np.concatenate([above, below[::-1]]),
            fill="toself",
            fillcolor="rgba(50, 200, 120, 0.25)",
            line=dict(color="rgba(0,0,0,0)"),
            hoverinfo="skip",
            name="surplus",
            showlegend=False,
        )
    ])
    fig.update_layout(
        title=title or "Surplus / Deficit vs Target",
        xaxis_title=time,
        yaxis_title=value,
    )
    return apply_theme(fig, theme)
