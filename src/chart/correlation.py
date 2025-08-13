from __future__ import annotations
from typing import Optional
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from .common import apply_theme, theme_from_cfg, _check

def scatter(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    color: Optional[str] = None,
    size: Optional[str] = None,
    title: Optional[str] = None,
    theme_name: Optional[str] = "dark_blue",
) -> go.Figure:
    for c in [x, y] + ([color] if color else []) + ([size] if size else []):
        if c and c not in df.columns:
            raise KeyError(f"Missing column: {c}")
    fig = px.scatter(df, x=x, y=y, color=color, size=size, hover_data=[c for c in [color, size] if c])
    fig.update_layout(title=title or "Scatter")
    return apply_theme(fig, theme_from_cfg(theme_name))

def bubble(**kwargs) -> go.Figure:
    # alias to scatter (size provided)
    return scatter(**kwargs)

def connected_scatter(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    order_by: str,  # usually a time column
    color: Optional[str] = None,
    title: Optional[str] = None,
    theme_name: Optional[str] = "dark_blue",
) -> go.Figure:
    for c in [x, y, order_by] + ([color] if color else []):
        if c and c not in df.columns:
            raise KeyError(f"Missing column: {c}")
    dd = df[[x, y, order_by] + ([color] if color else [])].sort_values(order_by)
    fig = px.scatter(dd, x=x, y=y, color=color)
    fig.update_traces(mode="lines+markers")
    fig.update_layout(title=title or "Connected Scatter")
    return apply_theme(fig, theme_from_cfg(theme_name))

def xy_heatmap(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    nbinsx: int = 30,
    nbinsy: int = 30,
    title: Optional[str] = None,
    theme_name: Optional[str] = "dark_blue",
) -> go.Figure:
    for c in [x, y]:
        if c not in df.columns:
            raise KeyError(f"Missing column: {c}")
    fig = px.density_heatmap(df, x=x, y=y, nbinsx=nbinsx, nbinsy=nbinsy)
    fig.update_layout(title=title or "XY Density Heatmap", coloraxis_colorbar=dict(title="count"))
    return apply_theme(fig, theme_from_cfg(theme_name))

def column_line_timeline(
    df: pd.DataFrame,
    *,
    time: str,
    bar_value: str,
    line_value: str,
    title: Optional[str] = None,
    theme_name: Optional[str] = "dark_blue",
) -> go.Figure:
    for c in [time, bar_value, line_value]:
        if c not in df.columns:
            raise KeyError(f"Missing column: {c}")
    dd = df[[time, bar_value, line_value]].sort_values(time)
    fig = go.Figure()
    fig.add_bar(x=dd[time], y=dd[bar_value], name=bar_value, yaxis="y")
    fig.add_scatter(x=dd[time], y=dd[line_value], mode="lines+markers", name=line_value, yaxis="y2")
    fig.update_layout(
        title=title or "Column + Line Timeline",
        xaxis=dict(title=time),
        yaxis=dict(title=bar_value),
        yaxis2=dict(title=line_value, overlaying="y", side="right"),
        barmode="group",
    )
    return apply_theme(fig, theme_from_cfg(theme_name))
