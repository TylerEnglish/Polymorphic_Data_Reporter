from __future__ import annotations
from typing import Optional
import pandas as pd
import math
import plotly.express as px
import plotly.graph_objects as go
from .common import apply_theme, theme_from_cfg, colorway

def stacked_bar(df: pd.DataFrame, *, category: str, part: str, value: str, title: Optional[str]=None, theme_name: str="dark_blue") -> go.Figure:
    for c in [category, part, value]:
        if c not in df.columns: raise KeyError(c)
    fig = px.bar(df, x=category, y=value, color=part)
    fig.update_layout(barmode="stack", title=title or "Stacked Bar")
    return apply_theme(fig, theme_from_cfg(theme_name))

def treemap(df: pd.DataFrame, *, labels: str, parents: Optional[str], value: str, title: Optional[str]=None, theme_name: str="dark_blue") -> go.Figure:
    cols = [labels, value] + ([parents] if parents else [])
    for c in cols:
        if c and c not in df.columns: raise KeyError(c)
    fig = px.treemap(df, path=[parents, labels] if parents else [labels], values=value)
    fig.update_layout(title=title or "Treemap")
    return apply_theme(fig, theme_from_cfg(theme_name))

def pie_guarded(df: pd.DataFrame, *, category: str, value: str, max_slices: int = 5, title: Optional[str]=None, theme_name: str="dark_blue") -> go.Figure:
    for c in [category, value]:
        if c not in df.columns: raise KeyError(c)
    dd = df[[category, value]].copy().sort_values(value, ascending=False)
    if len(dd) > max_slices:
        top = dd.head(max_slices - 1)
        other = pd.DataFrame({category: ["Other"], value: [dd[value].iloc[max_slices - 1 :].sum()]})
        dd = pd.concat([top, other], ignore_index=True)
    fig = px.pie(dd, names=category, values=value)
    fig.update_layout(title=title or "Pie (guarded)")
    return apply_theme(fig, theme_from_cfg(theme_name))

def donut_guarded(df: pd.DataFrame, *, category: str, value: str, max_slices: int = 5, title: Optional[str]=None, theme_name: str="dark_blue") -> go.Figure:
    for c in [category, value]:
        if c not in df.columns: raise KeyError(c)
    dd = df[[category, value]].copy().sort_values(value, ascending=False)
    if len(dd) > max_slices:
        top = dd.head(max_slices - 1)
        other = pd.DataFrame({category: ["Other"], value: [dd[value].iloc[max_slices - 1:].sum()]})
        dd = pd.concat([top, other], ignore_index=True)
    fig = px.pie(dd, names=category, values=value, hole=0.5)
    fig.update_layout(title=title or "Donut (guarded)")
    return apply_theme(fig, theme_from_cfg(theme_name))

def waterfall(
    df: pd.DataFrame,
    *,
    label: str,
    value: str,
    measure: Optional[str] = None,  # "relative" | "total"; if None -> all relative then final total
    title: Optional[str] = None,
    theme_name: str = "dark_blue",
) -> go.Figure:
    for c in [label, value] + ([measure] if measure else []):
        if c not in df.columns: raise KeyError(c)
    measures = df[measure].tolist() if measure else (["relative"] * (len(df) - 1) + ["total"])
    fig = go.Figure(go.Waterfall(
        x=df[label], y=df[value], measure=measures, connector={"line":{"color":"rgba(127,127,127,0.4)"}}
    ))
    fig.update_layout(title=title or "Waterfall")
    return apply_theme(fig, theme_from_cfg(theme_name))

def venn2(
    *,
    label_a: str,
    label_b: str,
    count_a: float,
    count_b: float,
    count_ab: float,
    title: Optional[str] = None,
    theme_name: str = "dark_blue",
) -> go.Figure:
    """Schematic 2-set Venn (not area-accurate by design)."""
    theme = theme_from_cfg(theme_name)
    colors = colorway(theme, 2)

    # circle centers & radius (fixed, nice overlap)
    r = 1.0
    cx_a, cy_a = -0.8, 0
    cx_b, cy_b = +0.8, 0

    fig = go.Figure()

    # filled circle-like patches via many points (for hover + legend)
    def _circle(x0, y0, r, name, fillcolor):
        steps = 100
        xs = [x0 + r * math.cos(t) for t in [i*2*math.pi/steps for i in range(steps+1)]]
        ys = [y0 + r * math.sin(t) for t in [i*2*math.pi/steps for i in range(steps+1)]]
        fig.add_scatter(
            x=xs, y=ys, mode="lines", line=dict(width=2, color=fillcolor),
            fill="toself", fillcolor=f"rgba(0,0,0,0.08)", name=name,
            hoverinfo="skip", showlegend=True
        )

    _circle(cx_a, cy_a, r, label_a, colors[0])
    _circle(cx_b, cy_b, r, label_b, colors[1])

    # labels for counts
    fig.add_annotation(x=cx_a - 0.35, y=0.05, text=f"{int(count_a - count_ab):,}", showarrow=False)
    fig.add_annotation(x=0, y=0.05, text=f"{int(count_ab):,}", showarrow=False)  # intersection
    fig.add_annotation(x=cx_b + 0.35, y=0.05, text=f"{int(count_b - count_ab):,}", showarrow=False)

    # set labels
    fig.add_annotation(x=cx_a, y= r+0.2, text=label_a, showarrow=False)
    fig.add_annotation(x=cx_b, y= r+0.2, text=label_b, showarrow=False)

    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, scaleanchor="x", scaleratio=1)
    fig.update_layout(title=title or "Venn (2-set, schematic)", margin=dict(l=20, r=20, t=60, b=20))
    return apply_theme(fig, theme)

def venn3(
    *,
    label_a: str, label_b: str, label_c: str,
    count_a: float, count_b: float, count_c: float,
    count_ab: float, count_ac: float, count_bc: float,
    count_abc: float,
    title: Optional[str] = None,
    theme_name: str = "dark_blue",
) -> go.Figure:
    """Schematic 3-set Venn. Positions form a triangle; intersection text shown."""
    theme = theme_from_cfg(theme_name)
    colors = colorway(theme, 3)

    r = 1.0
    # triangle layout
    cx_a, cy_a = 0, 1.0
    cx_b, cy_b = -0.9, -0.5
    cx_c, cy_c = +0.9, -0.5

    fig = go.Figure()
    def _circle(x0, y0, r, name, col):
        steps = 120
        xs = [x0 + r * math.cos(t) for t in [i*2*math.pi/steps for i in range(steps+1)]]
        ys = [y0 + r * math.sin(t) for t in [i*2*math.pi/steps for i in range(steps+1)]]
        fig.add_scatter(
            x=xs, y=ys, mode="lines", line=dict(width=2, color=col),
            fill="toself", fillcolor="rgba(0,0,0,0.08)", name=name, hoverinfo="skip"
        )

    _circle(cx_a, cy_a, r, label_a, colors[0])
    _circle(cx_b, cy_b, r, label_b, colors[1])
    _circle(cx_c, cy_c, r, label_c, colors[2])

    # Region labels (schematic placement)
    # Unique-only
    fig.add_annotation(x=cx_a, y=cy_a + 0.35, text=f"{int(count_a - count_ab - count_ac + count_abc):,}", showarrow=False)
    fig.add_annotation(x=cx_b - 0.35, y=cy_b - 0.1, text=f"{int(count_b - count_ab - count_bc + count_abc):,}", showarrow=False)
    fig.add_annotation(x=cx_c + 0.35, y=cy_c - 0.1, text=f"{int(count_c - count_ac - count_bc + count_abc):,}", showarrow=False)
    # Pair overlaps only
    fig.add_annotation(x=(cx_a+cx_b)/2 - 0.15, y=(cy_a+cy_b)/2, text=f"{int(count_ab - count_abc):,}", showarrow=False)
    fig.add_annotation(x=(cx_a+cx_c)/2 + 0.15, y=(cy_a+cy_c)/2, text=f"{int(count_ac - count_abc):,}", showarrow=False)
    fig.add_annotation(x=(cx_b+cx_c)/2, y=(cy_b+cy_c)/2 + 0.15, text=f"{int(count_bc - count_abc):,}", showarrow=False)
    # Triple
    fig.add_annotation(x=(cx_a+cx_b+cx_c)/3, y=(cy_a+cy_b+cy_c)/3, text=f"{int(count_abc):,}", showarrow=False)

    # Set labels
    fig.add_annotation(x=cx_a, y=cy_a + 1.1, text=label_a, showarrow=False)
    fig.add_annotation(x=cx_b - 0.9, y=cy_b - 0.9, text=label_b, showarrow=False)
    fig.add_annotation(x=cx_c + 0.9, y=cy_c - 0.9, text=label_c, showarrow=False)

    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, scaleanchor="x", scaleratio=1)
    fig.update_layout(title=title or "Venn (3-set, schematic)", margin=dict(l=20, r=20, t=60, b=20))
    return apply_theme(fig, theme)

def gridplot(
    df: pd.DataFrame,
    *,
    percent: str,                   # 0..1 or 0..100
    category: Optional[str] = None, # if provided, one grid per category
    total_cells: int = 100,
    ncols: int = 10,
    symbol: str = "square",
    title: Optional[str] = None,
    theme_name: str = "dark_blue",
) -> go.Figure:
    if percent not in df.columns:
        raise KeyError(percent)
    theme = theme_from_cfg(theme_name)
    filled_col = theme.get("primary", "#2563eb")
    empty_col  = theme.get("grid",    "#e5e7eb")

    def _make_grid(pct: float, xoff: float, yoff: float):
        # normalize to [0,1]
        p = pct / 100.0 if pct > 1.01 else pct
        p = max(0.0, min(1.0, p))
        nfill = int(round(p * total_cells))
        xs, ys, cs = [], [], []
        nrows = int(math.ceil(total_cells / ncols))
        for i in range(total_cells):
            r = i // ncols
            c = i % ncols
            xs.append(xoff + c)
            ys.append(yoff - r)
            cs.append(filled_col if i < nfill else empty_col)
        return xs, ys, cs

    fig = go.Figure()
    if category and category in df.columns:
        cats = df[category].tolist()
        for idx, (cat, pct) in enumerate(df[[category, percent]].itertuples(index=False, name=None)):
            xs, ys, cs = _make_grid(float(pct), xoff=0, yoff=-idx* (total_cells//ncols + 2))
            fig.add_scatter(
                x=xs, y=ys, mode="markers", name=str(cat),
                marker=dict(symbol=symbol, size=14, color=cs, line=dict(width=0.5, color=theme.get("plot_bg","#fff"))),
                hovertemplate=f"{category}: {cat}<br>{percent}: {pct:.1f}%<extra></extra>",
            )
            # label to the left
            fig.add_annotation(x=-1.2, y=ys[0]+0.2, text=str(cat), showarrow=False, xanchor="right")
    else:
        pct = float(df[percent].iloc[0])
        xs, ys, cs = _make_grid(pct, xoff=0, yoff=0)
        fig.add_scatter(
            x=xs, y=ys, mode="markers", name=f"{pct:.1f}%",
            marker=dict(symbol=symbol, size=14, color=cs, line=dict(width=0.5, color=theme.get("plot_bg","#fff"))),
            hovertemplate=f"{percent}: {pct:.1f}%<extra></extra>",
            showlegend=False,
        )

    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    ttl = title or "Gridplot (percent of whole)"
    return apply_theme(fig.update_layout(title=ttl, margin=dict(l=60, r=20, t=60, b=40)), theme)

def gridplot(
    df: pd.DataFrame,
    *,
    percent: str,                   # 0..1 or 0..100
    category: Optional[str] = None, # if provided, one grid per category
    total_cells: int = 100,
    ncols: int = 10,
    symbol: str = "square",
    title: Optional[str] = None,
    theme_name: str = "dark_blue",
) -> go.Figure:
    if percent not in df.columns:
        raise KeyError(percent)
    theme = theme_from_cfg(theme_name)
    filled_col = theme.get("primary", "#2563eb")
    empty_col  = theme.get("grid",    "#e5e7eb")

    def _make_grid(pct: float, xoff: float, yoff: float):
        # normalize to [0,1]
        p = pct / 100.0 if pct > 1.01 else pct
        p = max(0.0, min(1.0, p))
        nfill = int(round(p * total_cells))
        xs, ys, cs = [], [], []
        nrows = int(math.ceil(total_cells / ncols))
        for i in range(total_cells):
            r = i // ncols
            c = i % ncols
            xs.append(xoff + c)
            ys.append(yoff - r)
            cs.append(filled_col if i < nfill else empty_col)
        return xs, ys, cs

    fig = go.Figure()
    if category and category in df.columns:
        cats = df[category].tolist()
        for idx, (cat, pct) in enumerate(df[[category, percent]].itertuples(index=False, name=None)):
            xs, ys, cs = _make_grid(float(pct), xoff=0, yoff=-idx* (total_cells//ncols + 2))
            fig.add_scatter(
                x=xs, y=ys, mode="markers", name=str(cat),
                marker=dict(symbol=symbol, size=14, color=cs, line=dict(width=0.5, color=theme.get("plot_bg","#fff"))),
                hovertemplate=f"{category}: {cat}<br>{percent}: {pct:.1f}%<extra></extra>",
            )
            # label to the left
            fig.add_annotation(x=-1.2, y=ys[0]+0.2, text=str(cat), showarrow=False, xanchor="right")
    else:
        pct = float(df[percent].iloc[0])
        xs, ys, cs = _make_grid(pct, xoff=0, yoff=0)
        fig.add_scatter(
            x=xs, y=ys, mode="markers", name=f"{pct:.1f}%",
            marker=dict(symbol=symbol, size=14, color=cs, line=dict(width=0.5, color=theme.get("plot_bg","#fff"))),
            hovertemplate=f"{percent}: {pct:.1f}%<extra></extra>",
            showlegend=False,
        )

    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    ttl = title or "Gridplot (percent of whole)"
    return apply_theme(fig.update_layout(title=ttl, margin=dict(l=60, r=20, t=60, b=40)), theme)

def voronoi(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    color: Optional[str] = None,          # color points/polygons by category
    boundary: Optional[tuple[float,float,float,float]] = None,  # (xmin, xmax, ymin, ymax)
    title: Optional[str] = None,
    theme_name: str = "dark_blue",
) -> go.Figure:
    try:
        from scipy.spatial import Voronoi
        import numpy as np
    except Exception as e:
        raise ImportError("voronoi() requires scipy (scipy.spatial.Voronoi)") from e

    for c in [x, y] + ([color] if color else []):
        if c not in df.columns:
            raise KeyError(c)

    theme = theme_from_cfg(theme_name)
    pal = colorway(theme)

    pts = df[[x, y]].to_numpy(dtype=float)
    if pts.shape[0] < 2:
        fig = go.Figure()
        fig.update_layout(title=title or "Voronoi (not enough points)")
        return apply_theme(fig, theme)

    # bounding box
    xmin = float(np.nanmin(pts[:, 0])); xmax = float(np.nanmax(pts[:, 0]))
    ymin = float(np.nanmin(pts[:, 1])); ymax = float(np.nanmax(pts[:, 1]))
    if boundary:
        xmin, xmax, ymin, ymax = boundary
    pad_x = (xmax - xmin) * 0.05 or 1.0
    pad_y = (ymax - ymin) * 0.05 or 1.0
    xmin -= pad_x; xmax += pad_x; ymin -= pad_y; ymax += pad_y

    # Voronoi + region polygons (clip infinite regions to bbox)
    vor = Voronoi(pts)

    # NumPy 2.0 compatible range for “far” points
    radius = float(np.ptp(pts, axis=0).max()) * 2

    from collections import defaultdict
    new_regions = []
    new_vertices = vor.vertices.tolist()
    center = pts.mean(axis=0)

    all_ridges = defaultdict(list)
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges[p1].append((p2, v1, v2))
        all_ridges[p2].append((p1, v1, v2))

    for p, region_index in enumerate(vor.point_region):
        region = vor.regions[region_index]
        if -1 not in region and region:
            new_regions.append(region)
            continue

        ridges = all_ridges[p]
        new_region = [v for v in region if v != -1 and v is not None]
        for (p2, v1, v2) in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0 and v2 >= 0:
                continue
            t = pts[p] - pts[p2]                      # tangent (np.array)
            t /= (t**2).sum()**0.5
            n = np.array([-t[1], t[0]], dtype=float)  # normal as np.array

            midpoint = (pts[p] + pts[p2]) / 2
            direction = np.sign((midpoint - center).dot(n)) * n
            far = vor.vertices[v2] + direction * radius
            new_vertices.append(far.tolist())
            new_region.append(len(new_vertices) - 1)
        new_regions.append(new_region)

    def _clip_poly(poly):
        import numpy as _np
        def clip(subjectPolygon, edge):
            def inside(p):
                (x1, y1), (x2, y2) = edge
                return (x2 - x1) * (p[1] - y1) - (y2 - y1) * (p[0] - x1) >= 0
            def intersection(p1, p2):
                (x1, y1), (x2, y2) = edge
                x3, y3 = p1; x4, y4 = p2
                denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
                if denom == 0: return p2
                ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
                return [x1 + ua * (x2 - x1), y1 + ua * (y2 - y1)]
            outputList = subjectPolygon
            for clipEdge in [edge]:
                inputList = outputList
                outputList = []
                if not inputList: break
                S = inputList[-1]
                for E in inputList:
                    if inside(E):
                        if not inside(S):
                            outputList.append(intersection(S, E))
                        outputList.append(E)
                    elif inside(S):
                        outputList.append(intersection(S, E))
                    S = E
            return outputList

        box = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
        poly2 = poly[:]
        for i in range(4):
            poly2 = clip(poly2, (box[i], box[(i + 1) % 4]))
            if not poly2: break
        return poly2

    # Colors for categories (avoid pandas.unique deprecation)
    if color:
        cat_vals = df[color].to_numpy()
        cat_uni = list(np.unique(cat_vals))
        cmap = {c: pal[i % len(pal)] for i, c in enumerate(cat_uni)}
    else:
        cat_vals = None
        cmap = None

    def _hex_to_rgba(col: str, alpha: float = 0.2) -> str:
        """Convert '#RRGGBB' to 'rgba(r,g,b,a)'; fallback to a neutral rgba."""
        if isinstance(col, str) and col.startswith("#") and len(col) == 7:
            r = int(col[1:3], 16); g = int(col[3:5], 16); b = int(col[5:7], 16)
            return f"rgba({r},{g},{b},{alpha})"
        return f"rgba(107,114,128,{alpha})"

    # --- build the figure (ensure it's defined before adding traces) ---
    fig = go.Figure()

    for p_idx, reg in enumerate(new_regions):
        polygon = [new_vertices[v] for v in reg if v >= 0]
        if len(polygon) < 3:
            continue
        clipped = _clip_poly(polygon)
        if len(clipped) < 3:
            continue
        col = cmap[cat_vals[p_idx]] if color else theme.get("seq", ["#aaa"])[0]
        xs = [q[0] for q in clipped] + [clipped[0][0]]
        ys = [q[1] for q in clipped] + [clipped[0][1]]
        fig.add_scatter(
            x=xs, y=ys, mode="lines", fill="toself",
            line=dict(width=1, color=col),
            fillcolor=_hex_to_rgba(col, 0.2),
            name=str(cat_vals[p_idx]) if color else "cell",
            showlegend=False,
        )

    # points on top
    fig.add_scatter(
        x=pts[:, 0], y=pts[:, 1], mode="markers",
        marker=dict(
            size=6,
            color=[cmap[c] for c in cat_vals] if color else theme.get("primary", "#2563eb"),
            line=dict(width=1, color=theme.get("plot_bg", "#fff")),
        ),
        name="points",
    )

    fig.update_layout(title=title or "Voronoi (tessellation)", margin=dict(l=40, r=20, t=60, b=40))
    return apply_theme(fig, theme)

def marimekko(
    df: pd.DataFrame,
    *,
    cat_x: str,      # column categories
    cat_y: str,      # subcategories in each column
    value: str,      # magnitude
    title: Optional[str] = None,
    theme_name: str = "dark_blue",
) -> go.Figure:
    import plotly.express as px
    for c in [cat_x, cat_y, value]:
        if c not in df.columns:
            raise KeyError(c)
    theme = theme_from_cfg(theme_name)
    fig = px.treemap(df, path=[cat_x, cat_y], values=value)
    fig.update_layout(title=title or "Marimekko (Mosaic-style)")
    return apply_theme(fig, theme)

def arc_hemicycle(
    df: pd.DataFrame,
    *,
    group: str,        # party/segment
    seats: str,        # count per group
    rows: Optional[int] = None,   # auto if None
    start_angle_deg: float = 180, # left end
    end_angle_deg: float = 0,     # right end
    radius: float = 1.0,
    title: Optional[str] = None,
    theme_name: str = "dark_blue",
) -> go.Figure:
    for c in [group, seats]:
        if c not in df.columns:
            raise KeyError(c)

    theme = theme_from_cfg(theme_name)
    pal = colorway(theme)

    # expand seat list by group (left->right allocation)
    seat_labels = []
    for g, n in df[[group, seats]].itertuples(index=False, name=None):
        seat_labels.extend([g] * int(n))
    S = len(seat_labels)

    # choose rows: ~50 per row target
    if rows is None:
        rows = max(1, math.ceil(S / 50))
    # row capacities increasing outward ~ linear in row index
    coeff = S / (rows * (rows + 1) / 2.0)
    caps = [max(1, int(round(coeff * (i+1)))) for i in range(rows)]
    # adjust to sum S
    diff = S - sum(caps)
    i = rows - 1
    while diff > 0:
        caps[i] += 1; i = (i - 1) % rows; diff -= 1
    while diff < 0:
        j = next((j for j in range(rows-1, -1, -1) if caps[j] > 1), None)
        if j is None: break
        caps[j] -= 1; diff += 1

    # assign coordinates
    seats_out = []
    idx = 0
    for r, cap in enumerate(caps, start=1):
        rad = radius * (0.35 + 0.6 * r / rows)  # inner→outer
        angs = [math.radians(start_angle_deg + (end_angle_deg - start_angle_deg) * k/(max(cap-1,1)))
                for k in range(cap)]
        for a in angs:
            if idx >= S: break
            seats_out.append((rad*math.cos(a), rad*math.sin(a), seat_labels[idx]))
            idx += 1

    # color map by group
    uniq = list(dict.fromkeys(seat_labels))
    cmap = {g: pal[i % len(pal)] for i, g in enumerate(uniq)}

    fig = go.Figure()
    for g in uniq:
        pts = [(x,y) for (x,y,gg) in seats_out if gg == g]
        if not pts: continue
        fig.add_scatter(
            x=[p[0] for p in pts], y=[p[1] for p in pts], mode="markers", name=str(g),
            marker=dict(size=12, color=cmap[g], line=dict(width=1, color=theme.get("plot_bg","#fff"))),
            hovertemplate=f"{group}: {g}<extra></extra>",
        )

    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, scaleanchor="x", scaleratio=1)
    fig.update_layout(title=title or "Arc / Hemicycle (seats)", margin=dict(l=40, r=20, t=60, b=40))
    return apply_theme(fig, theme)
