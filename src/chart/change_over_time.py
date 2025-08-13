from __future__ import annotations
from typing import Optional, Sequence, Iterable, Tuple, Dict, List
import math
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
from plotly.subplots import make_subplots
from .common import apply_theme, theme_from_cfg, colorway, _check


# ---------- small helpers ----------

def _to_datetime(s: pd.Series) -> pd.Series:
    """Coerce to datetime without crashing; preserves original tz if present."""
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    return pd.to_datetime(s, errors="coerce")

def _auto_markers_from_irregular(ts: pd.Series) -> bool:
    """Return True if time deltas look irregular (→ show markers)."""
    t = _to_datetime(ts).sort_values().dropna()
    if len(t) < 3:
        return True
    d = (t.diff().dt.total_seconds().dropna()).to_numpy()
    if len(d) < 2:
        return True
    q1, q3 = np.quantile(d, [0.25, 0.75])
    spread = q3 - q1
    return (spread > 0.15 * np.median(d))  # heuristic

def _scale_sizes(values: np.ndarray, size_max: float = 36, size_min: float = 6) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    v = np.nan_to_num(v, nan=0.0)
    if np.nanmax(v) <= 0:
        return np.full_like(v, size_min, dtype=float)
    s = np.sqrt(np.clip(v, 0, None)) / math.sqrt(float(np.nanmax(v)))
    s = size_min + (size_max - size_min) * s
    return s

def _rgba(col: str, alpha: float) -> str:
    """
    Convert a color to an rgba() string with the given alpha.
    Accepts '#RRGGBB' or 'rgb(r,g,b)'. Falls back to a neutral gray.
    """
    col = (col or "").strip()
    if col.startswith("rgb("):      # e.g. "rgb(37,99,235)"
        return "rgba(" + col[4:-1] + f",{alpha})"
    if col.startswith("#") and len(col) == 7:  # e.g. "#2563eb"
        r = int(col[1:3], 16)
        g = int(col[3:5], 16)
        b = int(col[5:7], 16)
        return f"rgba({r},{g},{b},{alpha})"
    # neutral fallback
    return f"rgba(107,114,128,{alpha})"

# ---------------------------------------------------------------------------------------
# Line
# ---------------------------------------------------------------------------------------

def line(
    df: pd.DataFrame,
    *,
    time: str,
    value: str,
    group: Optional[str] = None,
    markers: Optional[bool] = None,   # None -> auto if irregular
    title: Optional[str] = None,
    theme_name: str = "dark_blue",
) -> go.Figure:
    _check(df, [time, value] + ([group] if group else []))
    theme = theme_from_cfg(theme_name)

    dd = df[[time, value] + ([group] if group else [])].copy()
    dd[time] = _to_datetime(dd[time])
    dd = dd.sort_values(time)

    show_markers = _auto_markers_from_irregular(dd[time]) if markers is None else markers

    if group:
        pal = colorway(theme)
        fig = go.Figure()
        for i, (g, gdf) in enumerate(dd.groupby(group, sort=False)):
            fig.add_scatter(
                x=gdf[time], y=gdf[value], mode="lines+markers" if show_markers else "lines",
                name=str(g),
                line=dict(width=2),
                marker=dict(size=6, line=dict(width=0.5, color=theme.get("plot_bg", "#fff"))),
            )
    else:
        fig = px.line(dd, x=time, y=value)
        if show_markers:
            fig.update_traces(mode="lines+markers", marker=dict(size=6, line=dict(width=0.5, color=theme.get("plot_bg", "#fff"))))

    fig.update_layout(title=title or "Line", xaxis_title=time, yaxis_title=value)
    return apply_theme(fig, theme)


# ---------------------------------------------------------------------------------------
# Column (single series)
# ---------------------------------------------------------------------------------------

def column(
    df: pd.DataFrame,
    *,
    time: str,
    value: str,
    title: Optional[str] = None,
    theme_name: str = "dark_blue",
) -> go.Figure:
    _check(df, [time, value])
    theme = theme_from_cfg(theme_name)
    dd = df[[time, value]].copy()
    dd[time] = _to_datetime(dd[time])
    dd = dd.sort_values(time)
    fig = px.bar(dd, x=time, y=value)
    fig.update_layout(title=title or "Column", xaxis_title=time, yaxis_title=value)
    return apply_theme(fig, theme)


# ---------------------------------------------------------------------------------------
# Column + line timeline (amount + rate)
# ---------------------------------------------------------------------------------------

def column_line(
    df: pd.DataFrame,
    *,
    time: str,
    column_value: str,   # bars
    line_value: str,     # line (rate or index)
    line_on_secondary: bool = True,
    title: Optional[str] = None,
    theme_name: str = "dark_blue",
) -> go.Figure:
    _check(df, [time, column_value, line_value])
    theme = theme_from_cfg(theme_name)
    dd = df[[time, column_value, line_value]].copy()
    dd[time] = _to_datetime(dd[time])
    dd = dd.sort_values(time)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(x=dd[time], y=dd[column_value], name=column_value, opacity=0.85),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=dd[time], y=dd[line_value], mode="lines+markers",
                   name=line_value, line=dict(width=2)),
        secondary_y=line_on_secondary,
    )
    fig.update_layout(title=title or "Column + Line", barmode="overlay")
    fig.update_xaxes(title_text=time)
    fig.update_yaxes(title_text=column_value, secondary_y=False)
    fig.update_yaxes(title_text=line_value, secondary_y=True)
    return apply_theme(fig, theme)


# ---------------------------------------------------------------------------------------
# Slope (two or three points per entity)
# ---------------------------------------------------------------------------------------

def slope(
    df: pd.DataFrame,
    *,
    entity: str,
    time: str,
    value: str,
    points: int = 2,               # 2 or 3
    title: Optional[str] = None,
    theme_name: str = "dark_blue",
) -> go.Figure:
    _check(df, [entity, time, value])
    if points not in (2, 3):
        points = 2
    theme = theme_from_cfg(theme_name)

    dd = df[[entity, time, value]].copy()
    dd[time] = _to_datetime(dd[time])

    # pick earliest/median/latest if 3; else earliest/latest
    out = []
    for ent, g in dd.groupby(entity):
        g = g.sort_values(time).dropna(subset=[time, value])
        if len(g) < 2:
            continue
        idx = [0, -1] if points == 2 else [0, len(g)//2, -1]
        out.append(g.iloc[idx])
    if not out:
        raise ValueError("Not enough points per entity for a slope chart.")
    ss = pd.concat(out)
    order_times = sorted(ss[time].unique())

    fig = go.Figure()
    for ent, g in ss.groupby(entity):
        g = g.sort_values(time)
        fig.add_scatter(x=g[time], y=g[value], mode="lines+markers", name=str(ent))
    fig.update_layout(
        title=title or ("Slope (2-point)" if points == 2 else "Slope (3-point)"),
        xaxis=dict(type="category", categoryorder="array", categoryarray=order_times, title=time),
        yaxis=dict(title=value),
        showlegend=True,
    )
    return apply_theme(fig, theme)


# ---------------------------------------------------------------------------------------
# Area chart (stacked if group provided)
# ---------------------------------------------------------------------------------------

def area(
    df: pd.DataFrame,
    *,
    time: str,
    value: str,
    group: Optional[str] = None,
    stack: bool = True,
    title: Optional[str] = None,
    theme_name: str = "dark_blue",
) -> go.Figure:
    _check(df, [time, value] + ([group] if group else []))
    theme = theme_from_cfg(theme_name)
    dd = df[[time, value] + ([group] if group else [])].copy()
    dd[time] = _to_datetime(dd[time])
    dd = dd.sort_values(time)

    if group:
        pal = colorway(theme)
        fig = go.Figure()
        for i, (g, gdf) in enumerate(dd.groupby(group, sort=False)):
            fig.add_scatter(
                x=gdf[time], y=gdf[value],
                mode="lines",
                stackgroup="one" if stack else None,
                name=str(g),
                line=dict(width=1.5),
            )
    else:
        fig = go.Figure()
        fig.add_scatter(x=dd[time], y=dd[value], mode="lines", fill="tozeroy", name=value, line=dict(width=1.5))

    fig.update_layout(title=title or ("Stacked Area" if group and stack else "Area"),
                      xaxis_title=time, yaxis_title=value)
    return apply_theme(fig, theme)


# ---------------------------------------------------------------------------------------
# Candlestick
# ---------------------------------------------------------------------------------------

def candlestick(
    df: pd.DataFrame,
    *,
    time: str,
    open_: str,
    high: str,
    low: str,
    close: str,
    title: Optional[str] = None,
    theme_name: str = "dark_blue",
) -> go.Figure:
    _check(df, [time, open_, high, low, close])
    theme = theme_from_cfg(theme_name)

    dd = df[[time, open_, high, low, close]].copy()
    dd[time] = _to_datetime(dd[time])
    dd = dd.sort_values(time)

    fig = go.Figure(go.Candlestick(
        x=dd[time], open=dd[open_], high=dd[high], low=dd[low], close=dd[close],
        increasing_line_color=theme.get("up", "#16a34a"),
        decreasing_line_color=theme.get("down", "#dc2626"),
        name="candles",
    ))
    fig.update_layout(title=title or "Candlestick", xaxis_title=time, yaxis_title="price")
    return apply_theme(fig, theme)


# ---------------------------------------------------------------------------------------
# Fan chart (projection uncertainty bands)
# ---------------------------------------------------------------------------------------

def fan_chart(
    df: pd.DataFrame,
    *,
    time: str,
    central: str,                 # median/mean projection
    # Pairs from narrow → wide (e.g., [("q40","q60"), ("q30","q70"), ("q20","q80"), ("q10","q90")])
    bands: Sequence[Tuple[str, str]],
    title: Optional[str] = None,
    theme_name: str = "dark_blue",
    band_colorscale: str = "Blues",   # sequential palette for bands
) -> go.Figure:
    need = [time, central] + [c for pair in bands for c in pair]
    _check(df, need)
    theme = theme_from_cfg(theme_name)

    dd = df[need].copy()
    dd[time] = _to_datetime(dd[time])
    dd = dd.sort_values(time)

    # draw widest band first
    fig = go.Figure()
    n = len(bands)
    for i, (lo, hi) in enumerate(reversed(bands)):
        t = i / max(1, n - 1)
        col = sample_colorscale(band_colorscale, [t])[0]  # rgb(...)
        x = pd.concat([dd[time], dd[time][::-1]])
        y = pd.concat([dd[hi], dd[lo][::-1]])
        fig.add_scatter(
            x=x, y=y, mode="lines", line=dict(width=0), fill="toself",
            fillcolor=col, name=f"{lo}–{hi}", hoverinfo="skip", showlegend=True,
        )

    fig.add_scatter(x=dd[time], y=dd[central], mode="lines", line=dict(width=2),
                    name=central)

    fig.update_layout(title=title or "Fan Chart (projection bands)",
                      xaxis_title=time, yaxis_title=central)
    return apply_theme(fig, theme)


# ---------------------------------------------------------------------------------------
# Connected Scatterplot (x vs y over time, connected in order)
# ---------------------------------------------------------------------------------------

def connected_scatter(
    df: pd.DataFrame,
    *,
    time: str,
    xvar: str,
    yvar: str,
    annotate_ends: bool = True,
    title: Optional[str] = None,
    theme_name: str = "dark_blue",
) -> go.Figure:
    _check(df, [time, xvar, yvar])
    theme = theme_from_cfg(theme_name)

    dd = df[[time, xvar, yvar]].copy()
    dd[time] = _to_datetime(dd[time])
    dd = dd.sort_values(time).reset_index(drop=True)

    line_col = theme.get("primary", "#0ea5a8")
    arrow_col = theme.get("axis", "#0f172a")

    fig = go.Figure()

    # main path (slightly lighter so arrows pop)
    fig.add_scatter(
        x=dd[xvar], y=dd[yvar],
        mode="lines+markers+text",
        line=dict(width=2, color=_rgba(line_col, 0.75)),
        marker=dict(size=7, color=line_col,
                    line=dict(width=0.8, color=theme.get("plot_bg", "#fff")),
                    symbol="circle"),
        text=[t.strftime("%Y") for t in dd[time]],
        textposition=["top center" if i % 2 else "bottom right" for i in range(len(dd))],
        textfont=dict(size=10, color=theme.get("axis", "#374151")),
        hovertemplate=(
            f"{time}: %{{text}}<br>{xvar}: %{{x:,.2f}}<br>{yvar}: %{{y:,.2f}}<extra></extra>"
        ),
        name=f"{yvar} vs {xvar}",
        showlegend=False
    )

    # short arrows centered on each segment so they’re obvious
    for i in range(1, len(dd)):
        x0, y0 = float(dd[xvar].iloc[i-1]), float(dd[yvar].iloc[i-1])
        x1, y1 = float(dd[xvar].iloc[i]),   float(dd[yvar].iloc[i])

        # place a small arrow around the midpoint (45%->55%)
        xm, ym = x0 + 0.55*(x1-x0), y0 + 0.55*(y1-y0)
        axm, aym = x0 + 0.45*(x1-x0), y0 + 0.45*(y1-y0)

        fig.add_annotation(
            x=xm, y=ym, ax=axm, ay=aym,
            xref="x", yref="y", axref="x", ayref="y",
            arrowhead=4, arrowsize=1.4, arrowwidth=2.2,
            arrowcolor=arrow_col, opacity=1.0, standoff=0, showarrow=True
        )

    if annotate_ends and len(dd) >= 2:
        fig.add_annotation(
            x=dd[xvar].iloc[0], y=dd[yvar].iloc[0],
            text=dd[time].iloc[0].strftime("Start %Y"),
            showarrow=True, arrowhead=2, arrowcolor=arrow_col, bgcolor=_rgba("#ffffff", 0.7),
            ax=20, ay=20
        )
        fig.add_annotation(
            x=dd[xvar].iloc[-1], y=dd[yvar].iloc[-1],
            text=dd[time].iloc[-1].strftime("End %Y"),
            showarrow=True, arrowhead=2, arrowcolor=arrow_col, bgcolor=_rgba("#ffffff", 0.7),
            ax=-20, ay=-20
        )

    fig.update_layout(
        title=title or "Connected Scatter (time-ordered)",
        xaxis_title=xvar, yaxis_title=yvar,
        hovermode="closest", margin=dict(l=40, r=20, t=60, b=40)
    )
    return apply_theme(fig, theme)

# ---------------------------------------------------------------------------------------
# Calendar heatmap (daily)
# ---------------------------------------------------------------------------------------

def calendar_heatmap(
    df: pd.DataFrame,
    *,
    date: str,
    value: str,
    title: Optional[str] = None,
    theme_name: str = "dark_blue",
    colorscale: str = "Turbo",
) -> go.Figure:
    _check(df, [date, value])
    theme = theme_from_cfg(theme_name)

    dd = df[[date, value]].copy()
    dd[date] = pd.to_datetime(dd[date], errors="coerce")
    dd = dd.dropna(subset=[date]).sort_values(date)

    if dd.empty:
        fig = go.Figure()
        fig.update_layout(title=title or "Calendar Heatmap (no data)")
        return apply_theme(fig, theme)

    d0 = dd[date].min()
    # Align weeks to Monday
    week_index = ((dd[date] - (d0 - pd.to_timedelta((d0.weekday()), unit="D"))).dt.days // 7).astype(int)
    dow = dd[date].dt.weekday  # 0=Mon .. 6=Sun
    txt = dd[date].dt.strftime("%Y-%m-%d")

    fig = go.Figure(go.Heatmap(
        x=week_index, y=dow, z=dd[value],
        colorscale=colorscale, colorbar=dict(title=value),
        text=txt, hovertemplate="Date: %{text}<br>Value: %{z}<extra></extra>",
    ))
    fig.update_yaxes(
        tickmode="array",
        tickvals=list(range(7)),
        ticktext=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
        autorange="reversed",
        title="",
    )
    fig.update_xaxes(title="Week")
    fig.update_layout(title=title or "Calendar Heatmap")
    return apply_theme(fig, theme)


# ---------------------------------------------------------------------------------------
# Priestley timeline (durations)
# ---------------------------------------------------------------------------------------

def priestley_timeline(
    df: pd.DataFrame,
    *,
    label: str,
    start: str,
    end: str,
    group: Optional[str] = None,
    title: Optional[str] = None,
    theme_name: str = "dark_blue",
) -> go.Figure:
    need = [label, start, end] + ([group] if group else [])
    _check(df, need)
    theme = theme_from_cfg(theme_name)
    pal = colorway(theme)

    dd = df[need].copy()
    dd[start] = _to_datetime(dd[start])
    dd[end]   = _to_datetime(dd[end])

    # fix inversions and compute midpoints
    inv = dd[end] < dd[start]
    if inv.any():
        dd.loc[inv, [start, end]] = dd.loc[inv, [end, start]].values
    dd["mid"] = dd[start] + (dd[end] - dd[start]) / 2

    # order top→bottom by start ascending, then by duration desc
    dd = dd.assign(duration=(dd[end]-dd[start])).sort_values([start, "duration"], ascending=[True, False]).reset_index(drop=True)
    yvals = list(reversed(range(len(dd))))  # top at y max

    # colors
    if group:
        groups = list(pd.unique(dd[group]))
        cmap = {g: pal[i % len(pal)] for i, g in enumerate(groups)}
        cols = dd[group].map(cmap)
        legend_map_added = set()
    else:
        cols = pd.Series(theme.get("primary", "#2563eb"), index=dd.index)

    fig = go.Figure()

    # thick horizontal segments = the “bars”
    for i, row in dd.iterrows():
        col = cols.iloc[i]
        y = yvals[i]
        fig.add_scatter(
            x=[row[start], row[end]], y=[y, y],
            mode="lines",
            line=dict(width=12, color=col, shape="linear"),
            name=(str(row[group]) if group else ""),
            legendgroup=(str(row[group]) if group else ""),
            showlegend=(group and (row[group] not in legend_map_added)),
            hovertemplate=f"{label}: {row[label]}<br>{start}: {{x|%Y-%m-%d}}<br>{end}: {row[end]:%Y-%m-%d}<extra></extra>",
        )
        if group:
            legend_map_added.add(row[group])

        # rounded end caps
        fig.add_scatter(
            x=[row[start], row[end]], y=[y, y],
            mode="markers",
            marker=dict(size=12, color=col, line=dict(width=0.8, color=theme.get("plot_bg", "#fff"))),
            hoverinfo="skip", showlegend=False
        )

        # centered label on bar
        fig.add_annotation(
            x=row["mid"], y=y, text=str(row[label]),
            showarrow=False, font=dict(size=11, color="white"),
            align="center", yshift=0
        )

    # y as categorical positions with labels
    fig.update_yaxes(
        tickmode="array",
        tickvals=yvals,
        ticktext=list(dd[label].iloc[::-1]),
        showgrid=False,
        autorange="reversed"  # top→bottom
    )
    fig.update_xaxes(type="date", tickformat="%Y", title="Time", showline=True)

    fig.update_layout(
        title=title or "Priestley Timeline",
        margin=dict(l=110, r=40, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return apply_theme(fig, theme)


# ---------------------------------------------------------------------------------------
# Circle timeline (bubble events per category over time)
# ---------------------------------------------------------------------------------------

def circle_timeline(
    df: pd.DataFrame,
    *,
    time: str,
    category: str,
    size_value: str,
    title: Optional[str] = None,
    theme_name: str = "dark_blue",
) -> go.Figure:
    _check(df, [time, category, size_value])
    theme = theme_from_cfg(theme_name)
    pal = colorway(theme)

    dd = df[[time, category, size_value]].copy()
    dd[time] = _to_datetime(dd[time])
    dd = dd.sort_values(time)

    cats = list(pd.unique(dd[category]))
    cmap = {c: pal[i % len(pal)] for i, c in enumerate(cats)}

    fig = go.Figure()
    for c in cats:
        g = dd[dd[category] == c]
        fig.add_scatter(
            x=g[time], y=[c] * len(g),
            mode="markers",
            marker=dict(size=_scale_sizes(g[size_value].to_numpy(), size_max=46, size_min=8),
                        color=cmap[c],
                        line=dict(width=0.6, color=theme.get("plot_bg", "#fff"))),
            name=str(c),
            hovertemplate=f"{category}: {c}<br>{time}: %{{x|%Y-%m-%d}}<br>{size_value}: %{{customdata:,.0f}}<extra></extra>",
            customdata=g[size_value],
        )

    fig.update_layout(title=title or "Circle Timeline",
                      xaxis_title=time, yaxis_title=category)
    return apply_theme(fig, theme)


# ---------------------------------------------------------------------------------------
# Vertical Timeline (time on Y)
# ---------------------------------------------------------------------------------------

def vertical_timeline(
    df: pd.DataFrame,
    *,
    time: str,
    x: Optional[str] = None,            # numeric column for lateral drift; if None, auto-meander
    title: Optional[str] = None,
    theme_name: str = "dark_blue",
    guides: int = 6,                    # number of dotted vertical guide lines
    x_span: float = 1.0,                # half-width of the path extent in x units
    smooth: float = 0.85,               # 0..1: stronger smoothing => rounder curve
    line_width: float = 3.0,
) -> go.Figure:
    """
    Vertical timeline: time on Y (increasing downward), a single smoothed path across X,
    dotted vertical guides, and a clean triangular arrowhead at the end.
    """
    _check(df, [time] + ([x] if x else []))
    theme = theme_from_cfg(theme_name)

    def _rgba(col: str, a: float) -> str:
        col = (col or "").strip()
        if col.startswith("rgb("): return "rgba(" + col[4:-1] + f",{a})"
        if col.startswith("#") and len(col) == 7:
            r = int(col[1:3], 16); g = int(col[3:5], 16); b = int(col[5:7], 16)
            return f"rgba({r},{g},{b},{a})"
        return f"rgba(107,114,128,{a})"

    dd = df[[c for c in [time, x] if c]].copy()
    dd[time] = _to_datetime(dd[time])
    dd = dd.sort_values(time).reset_index(drop=True)

    # --- lateral path values ---
    if x:
        xv = pd.to_numeric(dd[x], errors="coerce").astype(float)
        xv = xv - np.nanmean(xv)
        rng = float(np.nanpercentile(xv, 97) - np.nanpercentile(xv, 3)) or 1.0
        xv = (xv / (rng / 2.0)) * (x_span * 0.9)
        xv = pd.Series(xv).interpolate(limit_direction="both").to_numpy()
    else:
        n = len(dd)
        t = np.linspace(0, 2.5*np.pi, n)
        xv = (np.sin(t) + 0.15*np.cos(3*t)) * (x_span * 0.7)

    yv = dd[time]

    # --- bounds & guides ---
    xmin, xmax = float(np.nanmin(xv)), float(np.nanmax(xv))
    xpad = (xmax - xmin or 1.0) * 0.25
    gxmin, gxmax = xmin - xpad, xmax + xpad
    y0, y1 = yv.min(), yv.max()

    fig = go.Figure()

    if guides > 0:
        xs = np.linspace(gxmin, gxmax, guides + 2)[1:-1]
        for xg in xs:
            fig.add_shape(
                type="line", x0=xg, x1=xg, y0=y0, y1=y1,
                line=dict(color=_rgba(theme.get("grid", "#9ca3af"), 0.45), width=1, dash="dot"),
                layer="below"
            )

    primary = theme.get("primary", "#2563eb")
    # glow
    fig.add_scatter(
        x=xv, y=yv,
        mode="lines",
        line=dict(width=line_width + 4, color=_rgba(primary, 0.20), shape="spline", smoothing=float(smooth)),
        hoverinfo="skip", showlegend=False,
    )
    # main path
    fig.add_scatter(
        x=xv, y=yv,
        mode="lines",
        line=dict(width=line_width, color=primary, shape="spline", smoothing=float(smooth)),
        hovertemplate=f"{time}: %{{y|%Y-%m-%d %H:%M}}<br>x: %{{x:.2f}}<extra></extra>",
        showlegend=False,
        name="path",
    )
    # start/end markers
    fig.add_scatter(
        x=[xv[0], xv[-1]], y=[yv.iloc[0], yv.iloc[-1]],
        mode="markers",
        marker=dict(size=8, color=primary, line=dict(width=1, color=theme.get("plot_bg", "#ffffff"))),
        hoverinfo="skip", showlegend=False,
    )

    # --- arrowhead polygon at end (FIX: handle datetimes by converting to seconds) ---
    if len(dd) >= 2:
        # Normalize x in unit space
        def _xu(xx: float) -> float:
            return (xx - gxmin) / (gxmax - gxmin + 1e-12)

        # Convert datetime y to a 0..1 scalar using total_seconds
        def _yu(yy: pd.Timestamp) -> float:
            if isinstance(y0, pd.Timestamp) or isinstance(y1, pd.Timestamp):
                num = (pd.to_datetime(yy) - pd.to_datetime(y0)).total_seconds()
                den = (pd.to_datetime(y1) - pd.to_datetime(y0)).total_seconds()
                den = den if abs(den) > 1e-12 else 1e-12
                return float(num / den)
            # numeric fallback
            den = float((y1 - y0) or 1e-12)
            return float((yy - y0) / den)

        # Back from unit y to datetime/numeric
        def _y_from_unit(yu: float):
            if isinstance(y0, pd.Timestamp) or isinstance(y1, pd.Timestamp):
                secs = (pd.to_datetime(y1) - pd.to_datetime(y0)).total_seconds()
                return y0 + pd.to_timedelta(yu * secs, unit="s")
            return y0 + yu * (y1 - y0)

        x_tip,  y_tip  = float(xv[-1]), yv.iloc[-1]
        x_prev, y_prev = float(xv[-2]), yv.iloc[-2]

        txu, tyu = _xu(x_tip), _yu(y_tip)
        pxu, pyu = _xu(x_prev), _yu(y_prev)

        vx, vy = (txu - pxu), (tyu - pyu)
        L = max(1e-6, (vx**2 + vy**2) ** 0.5)
        ux, uy = vx / L, vy / L

        head_len = 0.035  # 3.5% of axis span
        head_w   = 0.022
        bx, by = (txu - head_len * ux, tyu - head_len * uy)
        nx, ny = -uy, ux
        left  = (bx + head_w * nx, by + head_w * ny)
        right = (bx - head_w * nx, by - head_w * ny)

        # convert unit coords back to data coords
        def _x_from_unit(xu: float) -> float:
            return gxmin + xu * (gxmax - gxmin)

        hx, hy = _x_from_unit(txu), _y_from_unit(tyu)
        lx, ly = _x_from_unit(left[0]),  _y_from_unit(left[1])
        rx, ry = _x_from_unit(right[0]), _y_from_unit(right[1])

        fig.add_scatter(
            x=[lx, hx, rx, lx], y=[ly, hy, ry, ly],
            mode="lines", fill="toself",
            line=dict(width=0.5, color=primary),
            fillcolor=_rgba(primary, 0.95),
            hoverinfo="skip", showlegend=False,
        )

    # axes/layout
    fig.update_yaxes(type="date", autorange="reversed", title=time, showgrid=False, zeroline=False)
    fig.update_xaxes(title="", range=[gxmin, gxmax], showline=False, zeroline=False, showgrid=False)
    fig.update_layout(
        title=title or "Vertical Timeline",
        margin=dict(l=40, r=40, t=60, b=40),
        showlegend=False,
    )
    return apply_theme(fig, theme)

# ---------------------------------------------------------------------------------------
# Seismogram (impulse/stem plot over time)
# ---------------------------------------------------------------------------------------

def seismogram(
    df: pd.DataFrame,
    *,
    time: str,
    amplitude: str,
    title: Optional[str] = None,
    theme_name: str = "dark_blue",
    baseline: float = 0.0,
) -> go.Figure:
    _check(df, [time, amplitude])
    theme = theme_from_cfg(theme_name)
    dd = df[[time, amplitude]].copy()
    dd[time] = _to_datetime(dd[time])
    dd = dd.sort_values(time)

    # build a single polyline with vertical stems using NaN breaks
    xs: List[pd.Timestamp] = []
    ys: List[float] = []
    for t, a in zip(dd[time], dd[amplitude]):
        xs.extend([t, t, np.nan])   # vline + break
        ys.extend([baseline, a, np.nan])

    fig = go.Figure(go.Scatter(x=xs, y=ys, mode="lines",
                               line=dict(width=1.4),
                               name=amplitude))
    fig.add_hline(y=baseline, line=dict(width=1, dash="dot"))
    fig.update_layout(title=title or "Seismogram", xaxis_title=time, yaxis_title=amplitude, showlegend=False)
    return apply_theme(fig, theme)


# ---------------------------------------------------------------------------------------
# Streamgraph (centered stacked areas)
# ---------------------------------------------------------------------------------------

def streamgraph(
    df: pd.DataFrame,
    *,
    time: str,
    group: str,
    value: str,
    title: Optional[str] = None,
    theme_name: str = "dark_blue",
) -> go.Figure:
    """
    Streamgraph built by explicit polygons (centered baseline).
    Expects long-form data with (time, group, value).
    """
    _check(df, [time, group, value])
    theme = theme_from_cfg(theme_name)
    pal = colorway(theme)

    dd = df[[time, group, value]].copy()
    dd[time] = _to_datetime(dd[time])
    dd = dd.dropna(subset=[time]).groupby([time, group], as_index=False)[value].sum()
    dd = dd.sort_values([time, group])

    # wide matrix (T x G)
    W = dd.pivot_table(index=time, columns=group, values=value, aggfunc="sum").fillna(0)
    times = W.index
    G = list(W.columns)
    tot = W.sum(axis=1)

    # centered cumulative boundaries
    cum = W.cumsum(axis=1)
    lower = cum.shift(axis=1).fillna(0)
    # offset so that midline is at zero
    offset = (tot / 2.0)
    lower_c = lower.sub(offset, axis=0)
    upper_c = cum.sub(offset, axis=0)

    fig = go.Figure()
    for i, g in enumerate(G):
        x = list(times) + list(times[::-1])
        y = list(upper_c[g]) + list(lower_c[g][::-1])
        fig.add_scatter(
            x=x, y=y, mode="lines", fill="toself",
            line=dict(width=0), fillcolor=pal[i % len(pal)],
            name=str(g),
            hovertemplate=f"{group}: {g}<br>{time}: %{{x|%Y-%m-%d}}<br>{value}: %{{customdata:,.0f}}<extra></extra>",
            customdata=W[g].to_numpy(),
        )

    fig.update_layout(
        title=title or "Streamgraph",
        xaxis_title=time, yaxis_title=value,
        showlegend=True,
    )
    return apply_theme(fig, theme)
