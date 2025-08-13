from __future__ import annotations
from typing import Optional, Sequence
import plotly.graph_objects as go
import pandas as pd
from .common import apply_theme, theme_from_cfg, colorway

def sankey(
    df: pd.DataFrame,
    *,
    source: str,
    target: str,
    value: str,
    title: Optional[str] = None,
    theme_name: str = "dark_blue",
    pad: int = 12,
    thickness: int = 14,
    valueformat: str = ",.0f",
    valuesuffix: str = "",
    color_links_by: str = "source",  # "source" | "neutral"
) -> go.Figure:
    """
    Themed Sankey with sensible defaults.
    color_links_by:
      - "source": color links using a color per source node (readable!)
      - "neutral": single neutral link color from theme with alpha
    """
    for c in [source, target, value]:
        if c not in df.columns:
            raise KeyError(c)

    theme = theme_from_cfg(theme_name)

    labels = pd.Index(pd.unique(df[[source, target]].values.ravel())).tolist()
    idx = {k: i for i, k in enumerate(labels)}

    # build a per-source palette
    src_unique = pd.Index(pd.unique(df[source])).tolist()
    palette_list = colorway(theme)
    palette = {s: palette_list[i % len(palette_list)] for i, s in enumerate(src_unique)}

    if color_links_by == "source":
        link_colors = [palette[s] for s in df[source]]
    else:
        link_colors = [theme.get("neutral", "#6b7280")] * len(df)

    # color each node too (wrap palette)
    node_colors = [palette_list[i % len(palette_list)] for i in range(len(labels))]

    fmt = f"%{{value:{valueformat}}}{valuesuffix}" if valueformat else f"%{{value}}{valuesuffix}"
    link_hover = f"%{{source.label}} → %{{target.label}}<br>{fmt}<extra></extra>"

    fig = go.Figure(
        go.Sankey(
            # THESE belong on the trace, not on link:
            valueformat=valueformat,
            valuesuffix=valuesuffix,

            node=dict(
                label=labels,
                pad=pad,
                thickness=thickness,
                color=node_colors,
            ),
            link=dict(
                source=[idx[s] for s in df[source]],
                target=[idx[t] for t in df[target]],
                value=df[value].tolist(),
                color=link_colors,
                hovertemplate=link_hover,   # formatting lives here
            ),
            arrangement="snap",
        )
    )
    fig.update_layout(title=title or "Sankey")
    return apply_theme(fig, theme)

def chord(
    df: pd.DataFrame,
    *,
    source: str,
    target: str,
    value: str,
    title: Optional[str] = None,
    theme_name: str = "dark_blue",
    curvature: float = 0.55,  # 0..1 inward bow
    steps: int = 25,          # points along each curve
) -> go.Figure:
    """
    Chord-like circular layout using networkx + quadratic Bézier curves.
    NOTE: This is a "chord-style" plot (curved links), not a true ribbon chord.
    Requires: networkx
    """
    try:
        import networkx as nx
    except Exception as e:
        raise ImportError("networkx is required for chord()") from e

    for c in [source, target, value]:
        if c not in df.columns:
            raise KeyError(c)

    theme = theme_from_cfg(theme_name)

    # build directed graph with weights
    G = nx.DiGraph()
    for s, t, v in df[[source, target, value]].itertuples(index=False, name=None):
        G.add_edge(s, t, weight=float(v))

    nodes = list(G.nodes())
    if not nodes:
        fig = go.Figure()
        fig.update_layout(title=title or "Chord (empty)")
        return apply_theme(fig, theme)

    pos = nx.circular_layout(nodes, scale=1.0)  # node -> (x,y) on unit circle

    # Helper: quadratic Bézier points from a->b via control point c
    def _quad(a, c, b, n):
        import numpy as _np
        t = _np.linspace(0, 1, n)
        return ( (1-t)**2 * a[0] + 2*(1-t)*t * c[0] + t**2 * b[0],
                 (1-t)**2 * a[1] + 2*(1-t)*t * c[1] + t**2 * b[1] )

    # Curved edges
    edge_traces = []
    max_w = max((d["weight"] for *_, d in G.edges(data=True)), default=1.0)
    line_color = theme.get("neutral", "#6b7280")
    for u, v, d in G.edges(data=True):
        a = pos[u]; b = pos[v]
        mid = ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
        # inward control point (pull toward center)
        ctrl = (mid[0] * curvature, mid[1] * curvature)
        xs, ys = _quad(a, ctrl, b, steps)
        w = d.get("weight", 1.0)
        edge_traces.append(go.Scatter(
            x=xs, y=ys, mode="lines",
            line=dict(width=0.5 + 4.0 * (w / max_w), color=line_color),
            hoverinfo="text",
            text=f"{u} → {v}: {w:,.0f}",
            showlegend=False,
        ))

    # Nodes
    node_colors = colorway(theme)
    node_trace = go.Scatter(
        x=[pos[n][0] for n in nodes],
        y=[pos[n][1] for n in nodes],
        mode="markers+text",
        text=[str(n) for n in nodes],
        textposition="top center",
        marker=dict(
            size=14,
            color=[node_colors[i % len(node_colors)] for i, _ in enumerate(nodes)],
            line=dict(width=1, color=theme.get("plot_bg", "#ffffff")),
        ),
        hoverinfo="text",
        showlegend=False,
    )

    fig = go.Figure(edge_traces + [node_trace])
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, scaleanchor="x", scaleratio=1)
    fig.update_layout(title=title or "Chord (circular network style)")
    return apply_theme(fig, theme)

def network(
    df: pd.DataFrame,
    *,
    source: str,
    target: str,
    value: Optional[str] = None,
    layout: str = "spring",          # "spring" | "kamada_kawai" | "circular"
    title: Optional[str] = None,
    theme_name: str = "dark_blue",
    undirected: bool = False,
) -> go.Figure:
    """
    Themed network with degree-sized nodes and weighted edges.
    Requires: networkx
    """
    try:
        import networkx as nx
    except Exception as e:
        raise ImportError("networkx is required for network()") from e

    for c in [source, target] + ([value] if value else []):
        if c not in df.columns:
            raise KeyError(c)

    theme = theme_from_cfg(theme_name)

    G = nx.Graph() if undirected else nx.DiGraph()
    if value:
        for s, t, v in df[[source, target, value]].itertuples(index=False, name=None):
            G.add_edge(s, t, weight=float(v))
    else:
        for s, t in df[[source, target]].itertuples(index=False, name=None):
            G.add_edge(s, t, weight=1.0)

    nodes = list(G.nodes())
    if not nodes:
        fig = go.Figure()
        fig.update_layout(title=title or "Network (empty)")
        return apply_theme(fig, theme)

    if layout == "spring":
        pos = nx.spring_layout(G, k=None, seed=42)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    else:
        pos = nx.spring_layout(G, seed=42)

    # Edges
    edge_traces = []
    max_w = max((d["weight"] for *_, d in G.edges(data=True)), default=1.0)
    line_color = theme.get("neutral", "#6b7280")
    for u, v, d in G.edges(data=True):
        x0, y0 = pos[u]; x1, y1 = pos[v]
        w = d.get("weight", 1.0)
        edge_traces.append(go.Scatter(
            x=[x0, x1], y=[y0, y1],
            mode="lines",
            line=dict(width=0.5 + 4.0 * (w / max_w), color=line_color),
            hoverinfo="text",
            text=f"{u} → {v}: {w:,.0f}",
            showlegend=False,
        ))

    # Nodes sized by degree, colored by theme colorway
    deg = dict(G.degree())
    max_deg = max(deg.values()) if deg else 1
    node_colors = colorway(theme)
    node_trace = go.Scatter(
        x=[pos[n][0] for n in nodes],
        y=[pos[n][1] for n in nodes],
        mode="markers+text",
        text=[str(n) for n in nodes],
        textposition="top center",
        marker=dict(
            size=[8 + 12 * (deg[n] / max_deg) for n in nodes],
            color=[node_colors[i % len(node_colors)] for i, _ in enumerate(nodes)],
            line=dict(width=1, color=theme.get("plot_bg", "#ffffff")),
        ),
        hoverinfo="text",
        showlegend=False,
    )

    fig = go.Figure(edge_traces + [node_trace])
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, scaleanchor="x", scaleratio=1)
    fig.update_layout(title=title or "Network")
    return apply_theme(fig, theme)

def waterfall(
    df: pd.DataFrame,
    *,
    label: str,                 # category labels (order = draw order)
    value: str,                 # step deltas (relative) or totals if measure says so
    measure: Optional[str] = None,   # column with: "relative" | "total" | "subtotal"
    totals: Optional[Sequence[int]] = None,  # indices to mark as totals if no measure col
    orientation: str = "v",     # "v" (x=labels, y=values) or "h"
    title: Optional[str] = None,
    theme_name: str = "dark_blue",
    valueformat: str = ",.0f",
    valuesuffix: str = "",
    connector: bool = True,
) -> go.Figure:
    """
    Themed Waterfall (additive decomposition). If `measure` is absent, all steps
    are treated as 'relative' except indices in `totals`, which become 'total'.
    """
    cols = [label, value] + ([measure] if measure else [])
    for c in cols:
        if c not in df.columns:
            raise KeyError(c)

    theme = theme_from_cfg(theme_name)
    labels = df[label].astype(str).tolist()
    vals   = df[value].astype(float).tolist()

    if measure:
        meas = df[measure].astype(str).str.lower().replace(
            {"rel": "relative", "tot": "total", "sub": "subtotal"}
        ).tolist()
        # default any unknowns to relative
        meas = [m if m in {"relative", "total", "subtotal"} else "relative" for m in meas]
    else:
        meas = ["relative"] * len(df)
        if totals:
            for idx in totals:
                if 0 <= idx < len(meas):
                    meas[idx] = "total"

    # Colors
    inc_color = theme.get("good", "#16a34a")
    dec_color = theme.get("bad", "#dc2626")
    tot_color = theme.get("primary", "#2563eb")
    grid_col  = theme.get("grid", "#e5e7eb")

    # For hover formatting
    name_token = "%{x}" if orientation == "v" else "%{y}"
    hover = f"{name_token}<br>Δ: %{{customdata:{valueformat}}}{valuesuffix}<extra></extra>"

    trace = go.Waterfall(
        orientation=orientation,
        measure=meas,
        x=labels if orientation == "v" else None,
        y=vals   if orientation == "v" else None,
        y0=None,
        # Horizontal variant flips axes usage
        **({"y": labels, "x": vals} if orientation == "h" else {}),
        increasing=dict(marker=dict(color=inc_color)),
        decreasing=dict(marker=dict(color=dec_color)),
        totals=dict(marker=dict(color=tot_color)),
        connector=dict(visible=connector, line=dict(color=grid_col, width=1)),
        customdata=vals,
        hovertemplate=hover,
        textposition="outside",
        texttemplate=f"%{{customdata:{valueformat}}}{valuesuffix}",
    )

    fig = go.Figure(trace)
    fig.update_layout(
        title=title or "Waterfall",
        xaxis_title=label if orientation == "v" else None,
        yaxis_title=value if orientation == "v" else None,
    )
    return apply_theme(fig, theme)
