from __future__ import annotations
from typing import Optional
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from contextlib import suppress
import numpy as np
from .common import apply_theme, theme_from_cfg

def histogram(
    df: pd.DataFrame,
    *,
    value: str,
    bins: int = 30,
    title: Optional[str] = None,
    theme_name: str = "dark_blue",
    edge: bool = True,
    edge_width: float = 1.0,
    edge_color: Optional[str] = None,
    opacity: float = 0.9,
) -> go.Figure:
    if value not in df.columns:
        raise KeyError(value)

    theme = theme_from_cfg(theme_name)
    fig = px.histogram(df, x=value, nbins=bins)

    # Add borders to each bin (the “bound boxes”)
    if edge:
        fig.update_traces(
            marker_line_color=edge_color or theme.get("grid", "#e5e7eb"),
            marker_line_width=edge_width,
            opacity=opacity,
            selector={"type": "histogram"},
        )

    fig.update_layout(
        title=title or "Histogram",
        bargap=0.02,  # slight separation helps show edges
    )

    return apply_theme(fig, theme)

def violin(df: pd.DataFrame, *, value: str, by: Optional[str] = None, title: Optional[str] = None, theme_name: str="dark_blue") -> go.Figure:
    cols = [value] + ([by] if by else [])
    for c in cols:
        if c and c not in df.columns: raise KeyError(c)
    fig = px.violin(df, y=value, x=by, box=True, points="all" if by else "outliers")
    fig.update_layout(title=title or "Violin")
    return apply_theme(fig, theme_from_cfg(theme_name))

def line(df: pd.DataFrame, *, time: str, value: str, color: Optional[str]=None, title: Optional[str]=None, theme_name: str="dark_blue") -> go.Figure:
    for c in [time, value] + ([color] if color else []):
        if c and c not in df.columns: raise KeyError(c)
    dd = df[[time, value] + ([color] if color else [])].sort_values(time)
    fig = px.line(dd, x=time, y=value, color=color)
    fig.update_layout(title=title or "Line")
    return apply_theme(fig, theme_from_cfg(theme_name))

def dot_strip_plot(
    df: pd.DataFrame,
    *,
    value: str,
    by: Optional[str] = None,
    jitter: float = 0.0,     # 0 = true strip; >0 spreads points to reduce overlap
    orientation: str = "v",  # "v": categories on x, values on y; "h": categories on y, values on x
    title: Optional[str] = None,
    theme_name: str = "dark_blue",
) -> go.Figure:
    cols = [value] + ([by] if by else [])
    for c in cols:
        if c not in df.columns:
            raise KeyError(c)

    theme = theme_from_cfg(theme_name)
    rng = np.random.default_rng(42)  # deterministic jitter

    traces = []
    xaxis_conf = {}
    yaxis_conf = {}

    if by:
        # Map each category to an integer position
        cats = pd.Categorical(df[by])
        codes = cats.codes.astype(float)  # -1 marks NaN categories; drop them
        m = codes >= 0
        vals = df.loc[m, value].astype(float).values
        codes = codes[m]
        cat_labels = list(cats.categories)
        tickvals = list(range(len(cat_labels)))

        # Jitter along the categorical axis, values on the other axis
        if orientation == "v":
            x = codes + (rng.uniform(-jitter, jitter, size=len(codes)) if jitter > 0 else 0)
            y = vals
            xaxis_conf = dict(tickmode="array", tickvals=tickvals, ticktext=cat_labels, title=by)
            yaxis_conf = dict(title=value)
        else:
            x = vals
            y = codes + (rng.uniform(-jitter, jitter, size=len(codes)) if jitter > 0 else 0)
            yaxis_conf = dict(tickmode="array", tickvals=tickvals, ticktext=cat_labels, title=by)
            xaxis_conf = dict(title=value)

        traces.append(go.Scatter(
            x=x, y=y, mode="markers",
            marker=dict(opacity=0.85),
            showlegend=False,
            hovertemplate=(f"{by}: %{{customdata}}<br>{value}: %{{y}}<extra></extra>"
                           if orientation == "v" else
                           f"{by}: %{{customdata}}<br>{value}: %{{x}}<extra></extra>"),
            customdata=np.array(cat_labels)[codes.astype(int)]
        ))
    else:
        # Single strip: put all points on one categorical line and jitter perpendicular
        vals = df[value].astype(float).values
        if orientation == "v":
            x = rng.uniform(-jitter, jitter, size=len(vals)) if jitter > 0 else np.zeros(len(vals))
            y = vals
            xaxis_conf = dict(visible=False)
            yaxis_conf = dict(title=value)
        else:
            x = vals
            y = rng.uniform(-jitter, jitter, size=len(vals)) if jitter > 0 else np.zeros(len(vals))
            yaxis_conf = dict(visible=False)
            xaxis_conf = dict(title=value)

        traces.append(go.Scatter(
            x=x, y=y, mode="markers",
            marker=dict(opacity=0.85),
            showlegend=False,
            hovertemplate=(f"{value}: %{{y}}<extra></extra>" if orientation == "v"
                           else f"{value}: %{{x}}<extra></extra>")
        ))

    fig = go.Figure(traces)
    fig.update_layout(title=title or "Dot Strip Plot")
    fig.update_xaxes(**xaxis_conf)
    fig.update_yaxes(**yaxis_conf)
    return apply_theme(fig, theme)

def beeswarm(
    df: pd.DataFrame,
    *,
    value: str,
    by: Optional[str] = None,
    jitter: float = 0.35,     # more spread to reduce overlap
    orientation: str = "v",
    title: Optional[str] = None,
    theme_name: str = "dark_blue",
) -> go.Figure:
    # beeswarm ≈ strip plot with jitter
    return dot_strip_plot(
        df, value=value, by=by, jitter=jitter, orientation=orientation,
        title=title or "Beeswarm", theme_name=theme_name
    )

def barcode_plot(
    df: pd.DataFrame,
    *,
    value: str,
    title: Optional[str] = None,
    theme_name: str = "dark_blue",
    marker_size: int = 14,
) -> go.Figure:
    if value not in df.columns: raise KeyError(value)
    theme = theme_from_cfg(theme_name)
    y = np.zeros(len(df))
    # A clean “barcode” look using a vertical line glyph
    Trace = go.Scattergl if len(df) > 3000 else go.Scatter
    fig = go.Figure(Trace(
        x=df[value], y=y,
        mode="markers",
        marker=dict(symbol="line-ns-open", size=marker_size, color=theme.get("axis", "#111827")),
        hovertemplate=f"{value}: %{{x}}<extra></extra>",
        showlegend=False,
    ))
    fig.update_yaxes(visible=False)
    fig.update_layout(title=title or "Barcode Plot", xaxis_title=value)
    return apply_theme(fig, theme)

def boxplot(
    df: pd.DataFrame,
    *,
    value: str,
    by: Optional[str] = None,
    points: str = "outliers",   # "all" | "outliers" | False
    notched: bool = False,
    title: Optional[str] = None,
    theme_name: str = "dark_blue",
) -> go.Figure:
    cols = [value] + ([by] if by else [])
    for c in cols:
        if c not in df.columns: raise KeyError(c)
    fig = px.box(df, y=value, x=by, points=points if points else False, notched=notched)
    fig.update_layout(title=title or "Boxplot")
    return apply_theme(fig, theme_from_cfg(theme_name))

def population_pyramid(
    df: pd.DataFrame,
    *,
    category: str,      # e.g., age band
    left: str,          # e.g., male
    right: str,         # e.g., female
    normalize: bool = False,  # convert to % within each row
    title: Optional[str] = None,
    theme_name: str = "dark_blue",
) -> go.Figure:
    for c in [category, left, right]:
        if c not in df.columns: raise KeyError(c)
    theme = theme_from_cfg(theme_name)
    dd = df[[category, left, right]].copy()
    if normalize:
        s = (dd[left].astype(float).abs() + dd[right].astype(float).abs())
        s = s.replace(0, np.nan)
        dd[left] = dd[left] / s * 100.0
        dd[right] = dd[right] / s * 100.0

    # left goes negative so bars mirror around 0
    dd["_left_neg"] = -dd[left].astype(float)
    dd["_right_pos"] = dd[right].astype(float)

    fig = go.Figure()
    fig.add_bar(y=dd[category], x=dd["_left_neg"], orientation="h",
                name=str(left), marker_color=theme.get("accent", "#ef4444"))
    fig.add_bar(y=dd[category], x=dd["_right_pos"], orientation="h",
                name=str(right), marker_color=theme.get("primary", "#2563eb"))
    fig.update_layout(
        barmode="overlay",
        title=title or ("Population Pyramid (%)" if normalize else "Population Pyramid"),
        xaxis_title=("Percent" if normalize else "Count"),
        yaxis_title=category,
    )
    # nice symmetric ticks
    max_abs = max(abs(dd["_left_neg"].min()), dd["_right_pos"].max())
    fig.update_xaxes(range=[-max_abs * 1.1, max_abs * 1.1])
    return apply_theme(fig, theme)

def cumulative_curve(
    df: pd.DataFrame,
    *,
    value: str,
    by: Optional[str] = None,
    complementary: bool = False,   # True = 1 - ECDF (survival)
    title: Optional[str] = None,
    theme_name: str = "dark_blue",
) -> go.Figure:
    if value not in df.columns:
        raise KeyError(value)

    # Some plotly versions don't support `complementary`; some older ones accept
    # ecdfnorm="percent" instead of "probability". Handle both gracefully.
    fig = None
    # 1) Try modern API
    try:
        fig = px.ecdf(df, x=value, color=by, ecdfnorm="probability", complementary=complementary)
    except TypeError:
        # 2) Fallback: no complementary kwarg; call without and invert Y if requested
        try:
            fig = px.ecdf(df, x=value, color=by, ecdfnorm="probability")
        except Exception:
            # 3) Older fallback: use percent and convert to probability
            fig = px.ecdf(df, x=value, color=by, ecdfnorm="percent")
            # convert percent->probability
            for tr in fig.data:
                y = np.asarray(tr["y"], dtype=float)
                tr["y"] = y / 100.0

        if complementary:
            for tr in fig.data:
                y = np.asarray(tr["y"], dtype=float)
                tr["y"] = 1.0 - y

    # Lines look cleaner than step markers by default
    fig.update_traces(mode="lines")

    fig.update_layout(
        title=title or ("Survival Curve" if complementary else "Cumulative Distribution"),
        xaxis_title=value,
        yaxis_title=("Survival" if complementary else "Probability"),
    )
    return apply_theme(fig, theme_from_cfg(theme_name))

def frequency_polygon(
    df: pd.DataFrame,
    *,
    value: str,
    bins: int = 30,
    by: Optional[str] = None,
    density: bool = False,   # normalize area to 1
    title: Optional[str] = None,
    theme_name: str = "dark_blue",
) -> go.Figure:
    if value not in df.columns: raise KeyError(value)
    x = df[value].astype(float).values
    # stable bin edges across groups
    x_min, x_max = np.nanmin(x), np.nanmax(x)
    edges = np.linspace(x_min, x_max, bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0

    fig = go.Figure()
    if by and by in df.columns:
        for name, g in df[[value, by]].dropna().groupby(by):
            counts, _ = np.histogram(g[value].astype(float).values, bins=edges)
            y = counts / counts.sum() if density and counts.sum() else counts
            fig.add_scatter(x=centers, y=y, mode="lines+markers", name=str(name))
    else:
        counts, _ = np.histogram(x[~np.isnan(x)], bins=edges)
        y = counts / counts.sum() if density and counts.sum() else counts
        fig.add_scatter(x=centers, y=y, mode="lines+markers", name=value)

    fig.update_layout(title=title or ("Frequency Polygon (density)" if density else "Frequency Polygon"),
                      xaxis_title=value, yaxis_title=("Density" if density else "Count"))
    return apply_theme(fig, theme_from_cfg(theme_name))
