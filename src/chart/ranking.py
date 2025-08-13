from __future__ import annotations
from typing import Optional, Sequence
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from .common import apply_theme, theme_from_cfg, _check

def ordered_bar(
    df: pd.DataFrame,
    *, category: str, value: str,
    topk: Optional[int] = None,
    ascending: bool = False,
    title: Optional[str] = None,
    theme_name: str = "dark_blue",
) -> go.Figure:
    _check(df, [category, value])
    dd = df[[category, value]].copy().sort_values(value, ascending=ascending)
    if topk:
        dd = dd.head(topk)
    # horizontal (bar)
    fig = px.bar(dd, y=category, x=value, orientation="h")
    # lock the visual order to the sorted order
    fig.update_layout(yaxis=dict(categoryorder="array", categoryarray=list(dd[category])))
    fig.update_layout(title=title or "Ordered Bar", xaxis_title=value, yaxis_title=category)
    return apply_theme(fig, theme_from_cfg(theme_name))

def ordered_column(
    df: pd.DataFrame,
    *, category: str, value: str,
    topk: Optional[int] = None,
    ascending: bool = False,
    title: Optional[str] = None,
    theme_name: str = "dark_blue",
) -> go.Figure:
    _check(df, [category, value])
    dd = df[[category, value]].copy().sort_values(value, ascending=ascending)
    if topk:
        dd = dd.head(topk)
    # vertical (column)
    fig = px.bar(dd, x=category, y=value)
    fig.update_layout(xaxis=dict(categoryorder="array", categoryarray=list(dd[category])))
    fig.update_layout(title=title or "Ordered Column", xaxis_title=category, yaxis_title=value)
    return apply_theme(fig, theme_from_cfg(theme_name))

def lollipop(
    df: pd.DataFrame,
    *, category: str, value: str,
    topk: Optional[int] = None,
    ascending: bool = False,
    title: Optional[str] = None,
    theme_name: str = "dark_blue",
    marker_size: int = 22,         # big head
    stem_width_frac: float = 0.18, # very thin stem (fraction of category band)
) -> go.Figure:
    _check(df, [category, value])

    theme = theme_from_cfg(theme_name)
    stem_color = theme.get("neutral", theme.get("grid", "#e5e7eb"))
    dot_color  = theme.get("primary", "#2563eb")

    dd = df[[category, value]].copy().sort_values(value, ascending=ascending)
    if topk:
        dd = dd.head(topk)

    fig = go.Figure()

    # Skinny stems from 0 → value (behind the dots)
    fig.add_bar(
        y=dd[category], x=dd[value],
        orientation="h",
        marker_color=stem_color,
        width=[stem_width_frac] * len(dd),
        hoverinfo="skip",
        showlegend=False,
        name="stem",
    )

    # Big dot heads at the end of each stem
    fig.add_scatter(
        y=dd[category], x=dd[value],
        mode="markers+text",
        marker=dict(
            size=marker_size,
            color=dot_color,
            line=dict(width=2, color=theme.get("plot_bg", "#ffffff")),
        ),
        text=[f"{v:,.0f}" for v in dd[value]],
        textposition="middle right",
        hovertemplate=f"{category}: %{{y}}<br>{value}: %{{x:,}}<extra></extra>",
        name=value,
        showlegend=False,
    )

    # Keep the visual order equal to the sort order
    fig.update_layout(
        barmode="overlay",
        yaxis=dict(categoryorder="array", categoryarray=list(dd[category])),
        title=title or "Lollipop",
        xaxis_title=value,
        yaxis_title=category,
        margin=dict(l=60, r=20, t=60, b=40),
    )
    return apply_theme(fig, theme)

def dot_strip_plot(
    df: pd.DataFrame,
    *, category: str, value: str,
    title: Optional[str] = None,
    theme_name: str = "dark_blue",
    jitter: float = 0.3,
) -> go.Figure:
    _check(df, [category, value])

    # Backward-compatible handling: older plotly.express.strip() had no 'jitter' kwarg
    try:
        fig = px.strip(df, x=category, y=value, jitter=jitter)
    except TypeError:
        fig = px.strip(df, x=category, y=value)
        if jitter and jitter > 0:
            # Apply jitter on the generated 'strip' traces
            fig.update_traces(jitter=jitter, selector={"type": "strip"})

    fig.update_layout(title=title or "Dot/Strip Plot", xaxis_title=category, yaxis_title=value)
    return apply_theme(fig, theme_from_cfg(theme_name))

def slope(
    df: pd.DataFrame,
    *, entity: str, time: str, value: str,
    title: Optional[str] = None,
    theme_name: str = "dark_blue",
) -> go.Figure:
    _check(df, [entity, time, value])
    dd = df[[entity, time, value]].copy()
    # pick exactly 2 time points: earliest & latest
    tvals = pd.Series(dd[time].unique())
    if len(tvals) < 2:
        raise ValueError("Slope chart needs at least two distinct time points.")
    t0, t1 = pd.to_datetime(tvals).sort_values().iloc[[0, -1]]
    dd = dd[dd[time].isin([t0, t1])]
    # wide form per entity
    pivot = (
        dd.pivot_table(index=entity, columns=time, values=value, aggfunc="mean")
          .dropna()
          .sort_values(by=t1, ascending=False)
    )
    fig = go.Figure()
    for ent, row in pivot.iterrows():
        fig.add_scatter(
            x=[t0, t1], y=[row[t0], row[t1]],
            mode="lines+markers", name=str(ent),
        )
    fig.update_layout(
        title=title or "Slope Chart",
        xaxis=dict(title=time, type="category", categoryorder="array", categoryarray=[t0, t1]),
        yaxis=dict(title=value),
    )
    return apply_theme(fig, theme_from_cfg(theme_name))

def ordered_proportional_symbol(
    df: pd.DataFrame,
    *,
    category: str,
    value: str,
    topk: Optional[int] = None,
    ascending: bool = False,
    title: Optional[str] = None,
    theme_name: Optional[str] = "dark_blue",
) -> go.Figure:
    _check(df, [category, value])
    dd = df[[category, value]].copy().sort_values(value, ascending=ascending)
    if topk:
        dd = dd.iloc[:topk]
    fig = go.Figure()
    fig.add_scatter(
        x=dd[category], y=[0]*len(dd), mode="markers+text",
        marker=dict(size=(dd[value] / dd[value].max() * 40).clip(lower=6)),
        text=[f"{v:,.0f}" for v in dd[value]], textposition="top center", name=value
    )
    fig.update_yaxes(visible=False)
    fig.update_layout(title=title or "Ordered Proportional Symbols", xaxis_title=category)
    return apply_theme(fig, theme_from_cfg(theme_name))

def bump(
    df: pd.DataFrame,
    *,
    entity: str,
    time: str,
    value: str,
    topk: int = 10,
    title: Optional[str] = None,
    theme_name: Optional[str] = "dark_blue",
) -> go.Figure:
    _check(df, [entity, time, value])
    # rank at each time, then keep top entities by overall prominence
    dd = df[[entity, time, value]].copy()
    dd = dd.sort_values([time, value], ascending=[True, False])
    dd["rank"] = dd.groupby(time)[value].rank(method="first", ascending=False)
    # pick entities that frequently appear in topk
    keep = (
        dd[dd["rank"] <= topk]
        .groupby(entity)["rank"].size()
        .sort_values(ascending=False)
        .head(topk)
        .index
    )
    dd = dd[dd[entity].isin(keep)]
    fig = go.Figure()
    for ent, g in dd.groupby(entity):
        g = g.sort_values(time)
        fig.add_scatter(x=g[time], y=g["rank"], mode="lines+markers", name=str(ent))
    fig.update_yaxes(autorange="reversed", title="rank (1=top)")
    fig.update_layout(title=title or f"Bump Chart (top {topk})", xaxis_title=time)
    return apply_theme(fig, theme_from_cfg(theme_name))
