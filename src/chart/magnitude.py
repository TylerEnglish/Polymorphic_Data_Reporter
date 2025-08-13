from __future__ import annotations
from typing import Iterable, Optional, Sequence
import math
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from .common import apply_theme, theme_from_cfg, colorway

def column(df: pd.DataFrame, *, category: str, value: str, title: Optional[str]=None, theme_name: str="dark_blue") -> go.Figure:
    for c in [category, value]:
        if c not in df.columns: raise KeyError(c)
    fig = px.bar(df, x=category, y=value)
    fig.update_layout(title=title or "Column")
    return apply_theme(fig, theme_from_cfg(theme_name))

def bar(df: pd.DataFrame, *, category: str, value: str, title: Optional[str]=None, theme_name: str="dark_blue") -> go.Figure:
    for c in [category, value]:
        if c not in df.columns: raise KeyError(c)
    fig = px.bar(df, y=category, x=value, orientation="h")
    fig.update_layout(title=title or "Bar")
    return apply_theme(fig, theme_from_cfg(theme_name))

def paired_column(df: pd.DataFrame, *, category: str, value_a: str, value_b: str, title: Optional[str]=None, theme_name: str="dark_blue") -> go.Figure:
    for c in [category, value_a, value_b]:
        if c not in df.columns: raise KeyError(c)
    dd = df[[category, value_a, value_b]]
    fig = go.Figure()
    fig.add_bar(x=dd[category], y=dd[value_a], name=value_a)
    fig.add_bar(x=dd[category], y=dd[value_b], name=value_b)
    fig.update_layout(barmode="group", title=title or "Paired Column")
    return apply_theme(fig, theme_from_cfg(theme_name))

def paired_bar(df: pd.DataFrame, *, category: str, value_a: str, value_b: str, title: Optional[str]=None, theme_name: str="dark_blue") -> go.Figure:
    for c in [category, value_a, value_b]:
        if c not in df.columns: raise KeyError(c)
    dd = df[[category, value_a, value_b]]
    fig = go.Figure()
    fig.add_bar(y=dd[category], x=dd[value_a], name=value_a, orientation="h")
    fig.add_bar(y=dd[category], x=dd[value_b], name=value_b, orientation="h")
    fig.update_layout(barmode="group", title=title or "Paired Bar")
    return apply_theme(fig, theme_from_cfg(theme_name))

def radar(
    df: pd.DataFrame,
    *,
    entity: str,
    metrics: Sequence[str],
    title: Optional[str]=None,
    theme_name: str="dark_blue",
) -> go.Figure:
    cols = [entity] + list(metrics)
    for c in cols:
        if c not in df.columns: raise KeyError(c)
    fig = go.Figure()
    for ent, g in df.groupby(entity):
        r = g[metrics].iloc[0].tolist()
        theta = list(metrics)
        fig.add_scatterpolar(r=r, theta=theta, name=str(ent), fill="toself")
    fig.update_layout(title=title or "Radar", polar=dict(radialaxis=dict(visible=True)))
    return apply_theme(fig, theme_from_cfg(theme_name))

def parallel_coordinates(
    df: pd.DataFrame,
    *,
    numeric_cols: Optional[Iterable[str]] = None,
    color_col: Optional[str] = None,
    title: Optional[str] = None,
    theme_name: str = "dark_blue",
) -> go.Figure:
    if numeric_cols is None:
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    for c in numeric_cols:
        if c not in df.columns: raise KeyError(c)
    fig = px.parallel_coordinates(df, dimensions=list(numeric_cols), color=color_col) if color_col else px.parallel_coordinates(df, dimensions=list(numeric_cols))
    fig.update_layout(title=title or "Parallel Coordinates")
    return apply_theme(fig, theme_from_cfg(theme_name))

# Advanced / TODOs:
def marimekko(
    df: pd.DataFrame,
    *,
    cat_x: str,         # top-level category (columns in a classic mekko)
    cat_y: str,         # subcategory within each column
    value: str,         # magnitude
    title: Optional[str] = None,
    theme_name: str = "dark_blue",
) -> go.Figure:
    for c in [cat_x, cat_y, value]:
        if c not in df.columns:
            raise KeyError(c)
    # Treemap approximates the mekko/mosaic idea (area encodes combined share)
    fig = px.treemap(df, path=[cat_x, cat_y], values=value)
    fig.update_layout(title=title or "Marimekko (Mosaic-style)")
    return apply_theme(fig, theme_from_cfg(theme_name))

def isotype(
    df: pd.DataFrame,
    *,
    category: str,
    value: str,
    unit: float = 1.0,          # one icon = `unit` of value
    per_row: int = 10,
    max_icons: int = 200,
    symbol: str = "square",     # try "square", "circle", "triangle-up", etc.
    title: Optional[str] = None,
    theme_name: str = "dark_blue",
) -> go.Figure:
    for c in [category, value]:
        if c not in df.columns:
            raise KeyError(c)
    rows = []
    for i, (cat, val) in enumerate(df[[category, value]].itertuples(index=False, name=None)):
        n = 0 if unit <= 0 else int(round(val / unit))
        n = max(0, min(n, max_icons))
        for k in range(n):
            rows.append({
                "cat": cat,
                "x": k % per_row,
                "y": - (i + (k // per_row)),  # stack rows downward per category
            })
    grid = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["cat","x","y"])
    fig = go.Figure()
    if not grid.empty:
        fig.add_scatter(
            x=grid["x"], y=grid["y"], mode="markers",
            marker=dict(symbol=symbol, size=14),
            text=grid["cat"], hoverinfo="text", name="isotype"
        )
    # Label categories on the left
    cats = df[category].unique().tolist()
    for i, cat in enumerate(cats):
        fig.add_annotation(x=-1.2, y=-(i), xref="x", yref="y",
                           text=str(cat), showarrow=False, align="right")
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(title=title or f"Isotype (1 icon = {unit:g} {value})")
    return apply_theme(fig, theme_from_cfg(theme_name))

def bullet_chart(
    df: pd.DataFrame,
    *,
    category: str,          # row label, one bullet per row
    value: str,             # actual value (bar)
    target: str | float,    # target marker (column name or constant)
    min_val: float | None = None,
    max_val: float | None = None,
    band1: float | str | None = None,  # upper edge for band 1 (e.g., poor)
    band2: float | str | None = None,  # upper edge for band 2 (e.g., avg)
    title: Optional[str] = None,
    theme_name: str = "dark_blue",
    show_delta: bool = True,
    valueformat: str = ",.0f",
    valuesuffix: str = "",
) -> go.Figure:
    """
    Multi-row bullet chart using Plotly indicators (gauge.shape='bullet').
    If bands are omitted, they’re derived heuristically from target/value.

    band1/band2 can be constants or column names (per-row).
    """
    cols = [category, value] + ([target] if isinstance(target, str) else []) \
           + ([band1] if isinstance(band1, str) else []) \
           + ([band2] if isinstance(band2, str) else [])
    for c in cols:
        if c and c not in df.columns:
            raise KeyError(c)

    theme = theme_from_cfg(theme_name)
    primary = theme.get("primary", "#2563eb")
    accent  = theme.get("accent",  "#ef4444")
    neutral = theme.get("neutral", "#6b7280")

    # light → dark background bands
    band_cols = [
        "rgba(107,114,128,0.20)",  # neutral @ 20%
        "rgba(107,114,128,0.35)",  # neutral @ 35%
        "rgba(107,114,128,0.55)",  # neutral @ 55%
    ]

    n = len(df)
    step = 1.0 / max(n, 1)
    pad = 0.04 * step  # vertical padding between bullets

    fig = go.Figure()
    for i, row in df.iterrows():
        v = float(row[value])
        t = float(row[target]) if isinstance(target, str) else float(target)

        # Axis range
        lo = 0.0 if min_val is None else float(min_val)
        hi = max(v, t)
        if isinstance(band1, (int, float)):
            hi = max(hi, float(band1))
        if isinstance(band2, (int, float)):
            hi = max(hi, float(band2))
        if isinstance(band1, str):
            hi = max(hi, float(row[band1]))
        if isinstance(band2, str):
            hi = max(hi, float(row[band2]))
        hi = float(max_val) if max_val is not None else hi * 1.1

        # Bands (0..b1..b2..hi)
        b1 = float(row[band1]) if isinstance(band1, str) else (float(band1) if band1 is not None else t * 0.6)
        b2 = float(row[band2]) if isinstance(band2, str) else (float(band2) if band2 is not None else t * 0.9)
        b1, b2 = sorted((b1, b2))
        steps = [
            {"range": [lo, b1], "color": band_cols[0]},
            {"range": [b1, b2], "color": band_cols[1]},
            {"range": [b2, hi], "color": band_cols[2]},
        ]

        # Vertical domain for this row
        y0 = 1.0 - (i + 1) * step + pad
        y1 = 1.0 - i * step - pad

        fig.add_trace(go.Indicator(
            mode="number+gauge+delta" if show_delta else "number+gauge",
            value=v,
            delta=dict(reference=t, valueformat=valueformat, increasing=dict(color=primary), decreasing=dict(color=accent)) if show_delta else None,
            number=dict(valueformat=valueformat, suffix=valuesuffix),
            title=dict(text=str(row[category]), font=dict(size=12), align="left"),
            gauge=dict(
                shape="bullet",
                axis=dict(range=[lo, hi]),
                bar=dict(color=primary),
                steps=steps,
                threshold=dict(line=dict(color=accent, width=2), value=t),
            ),
            domain=dict(x=[0.05, 0.98], y=[y0, y1]),
        ))

    fig.update_layout(height=max(120, 60 * n), margin=dict(l=60, r=30, t=60, b=40),
                      title=title or "Bullet Chart")
    return apply_theme(fig, theme)

def lollipop(
    df: pd.DataFrame,
    *,
    category: str,
    value: str,
    orientation: str = "v",      # "v" or "h"
    sort: Optional[str] = "desc", # "asc" | "desc" | None
    topk: Optional[int] = None,
    marker_size: int = 26,        # large head
    stem_width_frac: float = 0.18,# very thin stem (fraction of category band)
    title: Optional[str] = None,
    theme_name: str = "dark_blue",
) -> go.Figure:
    for c in [category, value]:
        if c not in df.columns:
            raise KeyError(c)

    theme = theme_from_cfg(theme_name)
    stem_color = theme.get("neutral", theme.get("grid", "#888"))
    dot_color  = theme.get("primary", "#2563eb")

    dd = df[[category, value]].copy()
    if sort == "asc":
        dd = dd.sort_values(value, ascending=True)
    elif sort == "desc":
        dd = dd.sort_values(value, ascending=False)
    if topk:
        dd = dd.head(topk)

    # One skinny bar (stem) per category + one big marker trace (head)
    fig = go.Figure()

    if orientation == "v":
        # stems as overlay bars from 0 → value (thin + muted)
        fig.add_bar(
            x=dd[category],
            y=dd[value],
            marker_color=stem_color,
            opacity=0.85,
            width=[stem_width_frac] * len(dd),
            hoverinfo="skip",
            showlegend=False,
            name="stem",
        )
        # heads as large circles on top
        fig.add_scatter(
            x=dd[category],
            y=dd[value],
            mode="markers",
            marker=dict(
                size=marker_size,
                color=dot_color,
                line=dict(width=2, color=theme.get("plot_bg", "#ffffff")),
            ),
            name=value,
            hovertemplate=f"{category}: %{{x}}<br>{value}: %{{y:,}}<extra></extra>",
        )
        fig.update_xaxes(title=category)
        fig.update_yaxes(title=value)
    else:
        fig.add_bar(
            y=dd[category],
            x=dd[value],
            orientation="h",
            marker_color=stem_color,
            opacity=0.85,
            width=[stem_width_frac] * len(dd),
            hoverinfo="skip",
            showlegend=False,
            name="stem",
        )
        fig.add_scatter(
            y=dd[category],
            x=dd[value],
            mode="markers",
            marker=dict(
                size=marker_size,
                color=dot_color,
                line=dict(width=2, color=theme.get("plot_bg", "#ffffff")),
            ),
            name=value,
            hovertemplate=f"{category}: %{{y}}<br>{value}: %{{x:,}}<extra></extra>",
        )
        fig.update_xaxes(title=value)
        fig.update_yaxes(title=category)

    # Make sure bars sit behind the dots
    fig.update_layout(
        barmode="overlay",
        title=title or "Lollipop",
        margin=dict(l=60, r=20, t=60, b=60),
    )
    return apply_theme(fig, theme)

def grouped_symbol(
    df: pd.DataFrame,
    *,
    category: str,     # groups across x (e.g., product)
    group: str,        # legend/color group (e.g., segment)
    value: str,        # magnitude per (category, group)
    unit: float = 1.0, # one icon = unit of value
    per_row: int = 10, # icons per row inside each category block
    max_icons_per_cat: int = 400,
    symbol: str = "square",
    title: Optional[str] = None,
    theme_name: str = "dark_blue",
) -> go.Figure:
    for c in [category, group, value]:
        if c not in df.columns:
            raise KeyError(c)

    theme = theme_from_cfg(theme_name)
    palette = colorway(theme)

    # Normalize data (ensure one row per (cat, grp))
    dd = (df[[category, group, value]]
          .groupby([category, group], as_index=False)[value].sum())

    cats = dd[category].unique().tolist()
    grp_order = dd[group].unique().tolist()
    grp_color = {g: palette[i % len(palette)] for i, g in enumerate(grp_order)}

    rows = []
    for ci, cat in enumerate(cats):
        block_x0 = ci * (per_row + 2)  # horizontal offset per category
        sub = dd[dd[category] == cat]
        for g, v in sub[[group, value]].itertuples(index=False, name=None):
            n_icons = 0 if unit <= 0 else int(round(float(v) / unit))
            n_icons = max(0, min(n_icons, max_icons_per_cat))
            for k in range(n_icons):
                rows.append({
                    "cat": cat,
                    "grp": g,
                    "x": block_x0 + (k % per_row),
                    "y": - (k // per_row),  # stack downward within block
                })

    grid = pd.DataFrame(rows, columns=["cat", "grp", "x", "y"])
    fig = go.Figure()
    if not grid.empty:
        for g, gdf in grid.groupby("grp"):
            fig.add_scatter(
                x=gdf["x"], y=gdf["y"], mode="markers", name=str(g),
                marker=dict(symbol=symbol, size=12, color=grp_color[g], line=dict(width=0.5, color=theme.get("plot_bg", "#fff"))),
                hovertemplate=f"{category}: %{{text}}<br>{group}: {g}<extra></extra>",
                text=gdf["cat"],
            )

    # Category labels centered under each block
    for ci, cat in enumerate(cats):
        cx = ci * (per_row + 2) + (per_row - 1) / 2
        fig.add_annotation(x=cx, y=-0.8, text=str(cat), showarrow=False, yanchor="top")

    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(title=title or f"Grouped Symbols (1 icon = {unit:g} {value})",
                      legend_title=group, margin=dict(l=40, r=20, t=60, b=60))
    return apply_theme(fig, theme)
