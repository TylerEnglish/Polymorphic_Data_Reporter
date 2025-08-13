from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
from .common import apply_theme, theme_from_cfg, colorway

# ---------- helpers ----------

def _robust_min_max(a: np.ndarray, lo: float = 2.0, hi: float = 98.0) -> tuple[float, float]:
    a = a[np.isfinite(a)]
    if a.size == 0:
        return (0.0, 1.0)
    return (float(np.percentile(a, lo)), float(np.percentile(a, hi)))

def _try_gaussian_smooth(z: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    try:
        from scipy.ndimage import gaussian_filter  # optional smoothing if SciPy is present
        zz = z.copy()
        nan_mask = ~np.isfinite(zz)
        zz[nan_mask] = 0.0
        w = (~nan_mask).astype(float)
        z_s = gaussian_filter(zz, sigma=sigma, mode="nearest")
        w_s = gaussian_filter(w,  sigma=sigma, mode="nearest")
        with np.errstate(invalid="ignore", divide="ignore"):
            out = z_s / np.where(w_s == 0, np.nan, w_s)
        return out
    except Exception:
        return z

def _geo_bounds(lats: np.ndarray, lons: np.ndarray, pad_frac: float = 0.08) -> tuple[list[float], list[float], dict]:
    lats = lats[np.isfinite(lats)]
    lons = lons[np.isfinite(lons)]
    if lats.size == 0 or lons.size == 0:
        return [-180, 180], [-90, 90], dict(center=dict(lat=0, lon=0))
    lat_min, lat_max = float(lats.min()), float(lats.max())
    lon_min, lon_max = float(lons.min()), float(lons.max())
    dlat = max(1e-6, lat_max - lat_min)
    dlon = max(1e-6, lon_max - lon_min)
    lat_pad = dlat * pad_frac
    lon_pad = dlon * pad_frac
    lataxis = [lat_min - lat_pad, lat_max + lat_pad]
    lonaxis = [lon_min - lon_pad, lon_max + lon_pad]
    center = dict(lat=(lataxis[0] + lataxis[1]) / 2.0,
                  lon=(lonaxis[0] + lonaxis[1]) / 2.0)
    return lonaxis, lataxis, dict(center=center)

def _rgba(col: str, alpha: float) -> str:
    col = col.strip()
    if col.startswith("rgb("):     # "rgb(r,g,b)"
        return "rgba(" + col[4:-1] + f",{alpha})"
    if col.startswith("#") and len(col) == 7:
        r = int(col[1:3], 16); g = int(col[3:5], 16); b = int(col[5:7], 16)
        return f"rgba({r},{g},{b},{alpha})"
    return f"rgba(107,114,128,{alpha})"  # neutral fallback

def _gc_path(lon1: float, lat1: float, lon2: float, lat2: float, n: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Great-circle between points in degrees (robust for long distances)."""
    p1 = np.radians([lat1, lon1])
    p2 = np.radians([lat2, lon2])
    # convert to 3D
    def sph2cart(lat, lon):
        cl = np.cos(lat)
        return np.array([cl*np.cos(lon), cl*np.sin(lon), np.sin(lat)], dtype=float)
    v1, v2 = sph2cart(*p1), sph2cart(*p2)
    dot = float(np.clip(np.dot(v1, v2), -1.0, 1.0))
    omega = np.arccos(dot)
    if omega < 1e-9:
        return np.array([lon1, lon2]), np.array([lat1, lat2])
    t = np.linspace(0, 1, n)
    so = np.sin(omega)
    pts = (np.sin((1-t)*omega)/so)[:,None]*v1 + (np.sin(t*omega)/so)[:,None]*v2
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    lats = np.degrees(np.arcsin(pts[:,2]))
    lons = np.degrees(np.arctan2(pts[:,1], pts[:,0]))
    return lons, lats

def _project_xy(lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Crude equirectangular used only for layout/repulsion (not rendering)."""
    lat0 = float(np.nanmean(lat))
    k = float(np.cos(np.deg2rad(lat0)))
    X = lon * k
    Y = lat.copy()
    return X, Y, k

# ---------------------------------------------------------------------------------------------------

def choropleth(
    df: pd.DataFrame,
    *,
    locations: str,   # e.g., ISO-3 or country names
    value: str,
    locationmode: str = "country names",  # "ISO-3", "country names", "USA-states"
    title: Optional[str] = None,
    theme_name: str = "dark_blue",
) -> go.Figure:
    for c in [locations, value]:
        if c not in df.columns: raise KeyError(c)
    fig = px.choropleth(
        df, locations=locations, color=value, locationmode=locationmode,
        color_continuous_scale="Blues"
    )
    fig.update_layout(title=title or "Choropleth")
    return apply_theme(fig, theme_from_cfg(theme_name))

def proportional_symbol(
    df: pd.DataFrame,
    *,
    lat: str,
    lon: str,
    value: Optional[str] = None,
    title: Optional[str] = None,
    theme_name: str = "dark_blue",
    size_max: int = 40,
    size_min: int = 6,
) -> go.Figure:
    for c in [lat, lon] + ([value] if value else []):
        if c not in df.columns:
            raise KeyError(c)

    theme = theme_from_cfg(theme_name)
    lats = df[lat].to_numpy(dtype=float)
    lons = df[lon].to_numpy(dtype=float)

    if value:
        v = df[value].astype(float).to_numpy()
        sv = np.sqrt(np.clip(v, 0, None))
        sizes = np.where(sv.max() > 0, sv / sv.max() * size_max, 0)
        sizes = np.clip(sizes, size_min, None)
        hover = f"<b>{value}</b>: %{{customdata:,.2f}}<br>{lat}: %{{lat:.2f}}<br>{lon}: %{{lon:.2f}}<extra></extra>"
        customdata = v
    else:
        sizes = np.full(len(df), 10.0)
        hover = f"{lat}: %{{lat:.2f}}<br>{lon}: %{{lon:.2f}}<extra></extra>"
        customdata = None

    lonaxis, lataxis, geo_extra = _geo_bounds(lats, lons)

    fig = go.Figure(go.Scattergeo(
        lat=lats, lon=lons, mode="markers",
        marker=dict(
            size=sizes, sizemode="diameter",
            color=theme.get("primary", "#2563eb"),
            line=dict(width=1, color=theme.get("plot_bg", "#fff")),
            opacity=0.88
        ),
        customdata=customdata,
        hovertemplate=hover,
        showlegend=False,
    ))
    fig.update_layout(
        geo=dict(
            showland=True, showcountries=True, showcoastlines=True,
            lonaxis=dict(range=lonaxis), lataxis=dict(range=lataxis),
            projection=dict(type="natural earth"),
            **geo_extra
        ),
        title=title or ("Proportional Symbol Map" if value else "Point Map"),
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return apply_theme(fig, theme)

def flow_map_lines(
    df: pd.DataFrame,
    *,
    lat_from: str, lon_from: str,
    lat_to: str,   lon_to: str,
    value: Optional[str] = None,
    title: Optional[str] = None,
    theme_name: str = "dark_blue",
    samples: int = 96,
    width_min: float = 1.2,
    width_max: float = 7.0,
    head_deg: float = 0.9,       # arrowhead length in "planar degrees" (auto scales)
    head_angle: float = 25.0,    # arrowhead half-angle (deg)
    alpha: float = 0.85,
) -> go.Figure:
    """Great-circle flows with tapered stroke + true arrowheads at the destination."""
    for c in [lat_from, lon_from, lat_to, lon_to] + ([value] if value else []):
        if c not in df.columns:
            raise KeyError(c)

    theme = theme_from_cfg(theme_name)
    base = theme.get("primary", "#2563eb")

    # width scaling
    if value:
        vals = df[value].astype(float).to_numpy()
        vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))
        widths = np.interp(vals, [vmin, vmax] if vmax > vmin else [vmin-1, vmin+1], [width_min, width_max])
    else:
        widths = np.full(len(df), (width_min + width_max) / 2.0)

    # collect all arc points for autoscale
    all_lats, all_lons = [], []
    fig = go.Figure()

    def _arrowhead(lon_tip, lat_tip, lon_prev, lat_prev, size_deg, angle_deg):
        """Build a small triangle at the path end, oriented along last segment."""
        # local equirect projection around the tip
        lat0 = float(lat_tip); k = float(np.cos(np.deg2rad(lat0)))
        x_tip,  y_tip  = lon_tip * k,  lat_tip
        x_prev, y_prev = lon_prev * k, lat_prev
        dx, dy = x_tip - x_prev, y_tip - y_prev
        L = float(np.hypot(dx, dy)) + 1e-12
        ux, uy = dx / L, dy / L               # unit direction (toward tip)
        # back-vector for the triangle base
        bx, by = -ux * size_deg, -uy * size_deg
        # rotate ± angle around (0,0)
        a = np.deg2rad(angle_deg)
        ca, sa = np.cos(a), np.sin(a)
        lx = bx * ca - by * sa; ly = bx * sa + by * ca
        rx = bx * ca + by * sa; ry = -bx * sa + by * ca
        # three points: tip, left-base, right-base
        X = np.array([x_tip,  x_tip + lx,  x_tip + rx])
        Y = np.array([y_tip,  y_tip + ly,  y_tip + ry])
        # back to lon/lat
        return (X / k, Y)

    for i, r in df.reset_index(drop=True).iterrows():
        lons, lats = _gc_path(r[lon_from], r[lat_from], r[lon_to], r[lat_to], n=samples)
        all_lats.append(lats); all_lons.append(lons)

        # main arc (single stroke)
        fig.add_trace(go.Scattergeo(
            lon=lons, lat=lats, mode="lines",
            line=dict(width=widths[i], color=base),
            opacity=alpha,
            hoverinfo="text",
            text=(f"{value}: {df[value].iloc[i]:,.2f}" if value else "flow"),
            showlegend=False,
        ))

        # arrowhead at destination
        lon_tip, lat_tip = lons[-1], lats[-1]
        lon_prev, lat_prev = lons[-2], lats[-2]
        # scale arrowhead in a data-aware way
        dlon = float(np.nanmax(lons) - np.nanmin(lons)) or 1.0
        dlat = float(np.nanmax(lats) - np.nanmin(lats)) or 1.0
        scale = max(dlon, dlat)
        ah_len = head_deg * (0.015 * scale)  # ~1.5% of span by default
        ah_lon, ah_lat = _arrowhead(lon_tip, lat_tip, lon_prev, lat_prev, ah_len, head_angle)
        fig.add_trace(go.Scattergeo(
            lon=list(ah_lon) + [ah_lon[0]], lat=list(ah_lat) + [ah_lat[0]],
            mode="lines", fill="toself",
            line=dict(width=1, color=base),
            fillcolor=_rgba(base, 0.95),
            hoverinfo="skip", showlegend=False,
        ))

    all_lats = np.concatenate(all_lats); all_lons = np.concatenate(all_lons)
    lonaxis, lataxis, geo_extra = _geo_bounds(all_lats, all_lons)

    fig.update_layout(
        geo=dict(
            showland=True, showcountries=True, showcoastlines=True,
            lonaxis=dict(range=lonaxis), lataxis=dict(range=lataxis),
            projection=dict(type="natural earth"),
            **geo_extra
        ),
        title=title or "Flow Map (great-circle arrows)",
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return apply_theme(fig, theme)

def contour_map(
    df: pd.DataFrame,
    *,
    lat: str, lon: str, value: str,
    nbins_lat: int = 140, nbins_lon: int = 140,
    agg: str = "mean",              # "mean" | "sum" | "count"
    diverging: bool = False,
    smooth_sigma: float = 1.4,      # uses SciPy if available
    levels: int = 9,                # number of isoline levels
    n_fill: int = 11,               # number of filled bands (discrete!)
    quantile_fill: bool = True,     # True -> quantile bands (strong contrast)
    gamma: float = 0.85,            # <1 boosts midrange contrast for equal-interval bands
    colorscale: Optional[str] = None,  # None -> Turbo (seq) / RdBu_r (div)
    title: Optional[str] = None,
    theme_name: str = "dark_blue",
) -> go.Figure:
    """High-contrast filled contours on a geo map: opaque, discrete bands + haloed isolines."""
    for c in [lat, lon, value]:
        if c not in df.columns:
            raise KeyError(c)

    theme = theme_from_cfg(theme_name)
    lats = df[lat].to_numpy(dtype=float)
    lons = df[lon].to_numpy(dtype=float)
    vals = df[value].to_numpy(dtype=float)

    # grid
    lat_edges = np.linspace(np.nanmin(lats), np.nanmax(lats), nbins_lat + 1)
    lon_edges = np.linspace(np.nanmin(lons), np.nanmax(lons), nbins_lon + 1)
    lat_cent  = (lat_edges[:-1] + lat_edges[1:]) / 2.0
    lon_cent  = (lon_edges[:-1] + lon_edges[1:]) / 2.0

    counts, _, _ = np.histogram2d(lats, lons, bins=[lat_edges, lon_edges])
    if agg == "count":
        Z = counts
    else:
        sums, _, _ = np.histogram2d(lats, lons, bins=[lat_edges, lon_edges], weights=vals)
        if agg == "sum":
            Z = sums
        else:
            with np.errstate(invalid="ignore", divide="ignore"):
                Z = sums / np.where(counts == 0, np.nan, counts)

    # optional smoothing (NaN-aware via normalized weights)
    Z = _try_gaussian_smooth(Z, sigma=smooth_sigma)

    finite = Z[np.isfinite(Z)]
    if finite.size == 0:
        lonaxis, lataxis, geo_extra = _geo_bounds(lats, lons)
        fig = go.Figure()
        fig.update_layout(
            geo=dict(
                showland=True, showcountries=True, showcoastlines=True,
                lonaxis=dict(range=lonaxis), lataxis=dict(range=lataxis),
                projection=dict(type="natural earth"), **geo_extra
            ),
            title=title or "Contour Map (no data)",
        )
        return apply_theme(fig, theme)

    # robust limits
    vmin, vmax = np.percentile(finite, [3, 97])
    if diverging:
        vmax = max(abs(vmin), abs(vmax)); vmin = -vmax

    # palette
    cs = colorscale or ("RdBu_r" if diverging else "Turbo")

    # -------- DISCRETE BANDS (OPAQUE) --------
    if quantile_fill:
        # quantile bins -> guarantees full palette contrast even with skewed data
        qs = np.linspace(0, 1, n_fill + 1)
        band_edges = np.quantile(finite, qs)
        # enforce symmetry for diverging
        if diverging:
            m = max(abs(band_edges).max(), 1e-12)
            band_edges = np.linspace(-m, m, n_fill + 1)
    else:
        # equal-interval with gamma stretch for contrast
        ts = np.linspace(0, 1, n_fill + 1) ** gamma
        band_edges = vmin + ts * (vmax - vmin)

    # band colors: sample midpoints
    mids = (band_edges[:-1] + band_edges[1:]) / 2.0
    t_mid = (mids - vmin) / (vmax - vmin + 1e-12)
    band_colors = [sample_colorscale(cs, [float(np.clip(t, 0, 1))])[0] for t in t_mid]

    # assign each cell to a band
    with np.errstate(invalid="ignore"):
        band_idx = np.digitize(Z, band_edges, right=True) - 1
    band_idx = np.where(np.isfinite(Z), np.clip(band_idx, 0, n_fill - 1), -1)

    fig = go.Figure()

    # draw cells (opaque fill, zero line) — cap for perf but still dense
    R, C = Z.shape
    cap = min(14000, R * C)
    # prioritize cells with higher local contrast
    pad = np.pad(Z, 1, mode="edge")
    neigh = (
        pad[:-2, :-2] + pad[:-2, 1:-1] + pad[:-2, 2:] +
        pad[1:-1, :-2] + pad[1:-1, 1:-1] + pad[1:-1, 2:] +
        pad[2:, :-2] + pad[2:, 1:-1] + pad[2:, 2:]
    ) / 9.0
    score = np.abs(Z - neigh)
    score[~np.isfinite(Z)] = -np.inf
    order = np.argsort(score.ravel())[::-1][:cap]

    for k in order:
        r, c = divmod(k, C)
        bi = band_idx[r, c]
        if bi < 0:
            continue
        col = band_colors[bi]  # 'rgb(...)' string from sample_colorscale
        xs = [lon_edges[c], lon_edges[c+1], lon_edges[c+1], lon_edges[c], lon_edges[c]]
        ys = [lat_edges[r], lat_edges[r],   lat_edges[r+1], lat_edges[r+1], lat_edges[r]]
        fig.add_trace(go.Scattergeo(
            lon=xs, lat=ys, mode="lines", fill="toself",
            line=dict(width=0), fillcolor=col,   # OPAQUE fill
            hoverinfo="skip", showlegend=False,
        ))

    # -------- CRISP, HALOED ISOLINES --------
    def _iso_segments(Z, xs, ys, level):
        segs = []
        RR, CC = Z.shape
        for i in range(RR - 1):
            Zi0, Zi1 = Z[i], Z[i+1]
            for j in range(CC - 1):
                q00, q01, q11, q10 = Zi0[j], Zi0[j+1], Zi1[j+1], Zi1[j]
                if not np.isfinite([q00, q01, q11, q10]).all():
                    continue
                mn, mx = min(q00, q01, q11, q10), max(q00, q01, q11, q10)
                if not (mn < level < mx):
                    continue
                def lerp(a, b, va, vb):
                    t = (level - va) / (vb - va + 1e-12); return a + t * (b - a)
                x0, x1 = xs[j], xs[j+1]; y0, y1 = ys[i], ys[i+1]
                pts = []
                if (q00 - level) * (q10 - level) < 0: pts.append((x0, lerp(y0, y1, q00, q10)))
                if (q01 - level) * (q11 - level) < 0: pts.append((x1, lerp(y0, y1, q01, q11)))
                if (q00 - level) * (q01 - level) < 0: pts.append((lerp(x0, x1, q00, q01), y0))
                if (q10 - level) * (q11 - level) < 0: pts.append((lerp(x0, x1, q10, q11), y1))
                if len(pts) == 2: segs.append(pts)
        return segs

    Ls = np.linspace(vmin, vmax, max(3, int(levels)))
    halo = _rgba("#ffffff", 0.95)
    core = "#111111"

    for L in Ls:
        segs = _iso_segments(Z, lon_cent, lat_cent, L)
        for (x0, y0), (x1, y1) in segs:
            fig.add_trace(go.Scattergeo(  # halo
                lon=[x0, x1], lat=[y0, y1], mode="lines",
                line=dict(width=3.6, color=halo),
                hoverinfo="skip", showlegend=False,
            ))
        for (x0, y0), (x1, y1) in segs:
            fig.add_trace(go.Scattergeo(  # core
                lon=[x0, x1], lat=[y0, y1], mode="lines",
                line=dict(width=1.8, color=core),
                hoverinfo="skip", showlegend=False,
            ))

    # frame
    lonaxis, lataxis, geo_extra = _geo_bounds(lats, lons)
    fig.update_layout(
        geo=dict(
            showland=True, showcountries=True, showcoastlines=True,
            lonaxis=dict(range=lonaxis), lataxis=dict(range=lataxis),
            projection=dict(type="natural earth"),
            **geo_extra
        ),
        title=title or ("Contour Map (diverging)" if diverging else "Contour Map"),
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return apply_theme(fig, theme)

def dot_density(
    df: pd.DataFrame,
    *,
    lat: str, lon: str,
    color: Optional[str] = None,      # categorical column (legend)
    size: Optional[float] = None,     # override auto size
    annotate_hotspots: bool = True,
    bins_for_hotspots: Tuple[int, int] = (60, 60),
    topk_hotspots: int = 3,
    title: Optional[str] = None,
    theme_name: str = "dark_blue",
) -> go.Figure:
    """
    Crisp, readable dots (auto-sized by row count) + optional hotspot labels.
    """
    for c in [lat, lon] + ([color] if color else []):
        if c not in df.columns:
            raise KeyError(c)

    theme = theme_from_cfg(theme_name)
    pal = colorway(theme)

    lats = df[lat].to_numpy(dtype=float)
    lons = df[lon].to_numpy(dtype=float)

    # autosize dots by volume (but allow an explicit size)
    if size is None:
        n = len(df)
        if n < 500:   size = 7.0
        elif n < 2000: size = 5.0
        else:         size = 3.5

    lonaxis, lataxis, geo_extra = _geo_bounds(lats, lons)
    fig = go.Figure()

    if color:
        cats = pd.Index(pd.unique(df[color])).tolist()
        cmap = {g: pal[i % len(pal)] for i, g in enumerate(cats)}
        for g, gdf in df.groupby(color):
            fig.add_trace(go.Scattergeo(
                lon=gdf[lon], lat=gdf[lat], mode="markers",
                marker=dict(size=size, color=cmap[g], opacity=0.80,
                            line=dict(width=0.6, color=theme.get("plot_bg", "#fff"))),
                name=str(g),
                hovertemplate=f"{color}: {g}<br>{lat}: %{{lat:.2f}}<br>{lon}: %{{lon:.2f}}<extra></extra>",
            ))
    else:
        fig.add_trace(go.Scattergeo(
            lon=lons, lat=lats, mode="markers",
            marker=dict(size=size, color=theme.get("primary", "#2563eb"), opacity=0.80,
                        line=dict(width=0.6, color=theme.get("plot_bg", "#fff"))),
            name="points",
            hovertemplate=f"{lat}: %{{lat:.2f}}<br>{lon}: %{{lon:.2f}}<extra></extra>",
            showlegend=False,
        ))

    if annotate_hotspots and len(df) >= 30:
        H, lat_bins, lon_bins = np.histogram2d(lats, lons, bins=bins_for_hotspots)
        flat = H.ravel()
        if flat.max() > 0:
            k = min(topk_hotspots, int((flat > 0).sum()))
            idxs = np.argpartition(flat, -k)[-k:]
            idxs = idxs[np.argsort(flat[idxs])[::-1]]
            for idx in idxs:
                r = idx // H.shape[1]
                c = idx %  H.shape[1]
                lat_c = (lat_bins[r] + lat_bins[r+1]) / 2.0
                lon_c = (lon_bins[c] + lon_bins[c+1]) / 2.0
                count = int(H[r, c])
                # a faint ring + label
                fig.add_trace(go.Scattergeo(
                    lon=[lon_c], lat=[lat_c], mode="markers+text",
                    marker=dict(size=max(16, size*3.0), symbol="circle-open",
                                line=dict(width=2, color=_rgba(theme.get("accent", "#ef4444"), 0.8))),
                    text=[f"{count}"], textposition="top center",
                    textfont=dict(size=11, color=theme.get("axis", "#111827")),
                    hoverinfo="skip", showlegend=False,
                ))

    fig.update_layout(
        geo=dict(
            showland=True, showcountries=True, showcoastlines=True,
            lonaxis=dict(range=lonaxis), lataxis=dict(range=lataxis),
            projection=dict(type="natural earth"),
            **geo_extra
        ),
        title=title or "Dot Density",
        margin=dict(l=10, r=10, t=60, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    )
    return apply_theme(fig, theme)

def scaled_cartogram(
    df: pd.DataFrame,
    *,
    lat: str, lon: str, value: str,
    iterations: int = 160,
    k_repulse: float = 0.55,
    size_max: int = 46,
    size_min: int = 6,
    title: Optional[str] = None,
    theme_name: str = "dark_blue",
) -> go.Figure:
    for c in [lat, lon, value]:
        if c not in df.columns:
            raise KeyError(c)

    theme = theme_from_cfg(theme_name)

    lats = df[lat].to_numpy(dtype=float)
    lons = df[lon].to_numpy(dtype=float)
    vals = df[value].to_numpy(dtype=float)

    sv = np.sqrt(np.clip(vals, 0, None))
    sizes = np.where(sv.max() > 0, sv / sv.max() * size_max, 0)
    sizes = np.clip(sizes, size_min, None)

    X, Y, k = _project_xy(lats, lons)
    rad = sizes / 300.0

    n = len(X)
    for _ in range(iterations):
        moved = 0
        for i in range(n):
            for j in range(i + 1, n):
                dx = X[j] - X[i]; dy = Y[j] - Y[i]
                dist = float(np.hypot(dx, dy)) + 1e-12
                mind = rad[i] + rad[j]
                if dist < mind:
                    ov = (mind - dist)
                    ux, uy = dx / dist, dy / dist
                    shift = 0.5 * ov * k_repulse
                    X[i] -= ux * shift; Y[i] -= uy * shift
                    X[j] += ux * shift; Y[j] += uy * shift
                    moved += 1
        if moved == 0:
            break

    lon_new = X / k
    lat_new = Y

    lonaxis, lataxis, geo_extra = _geo_bounds(lat_new, lon_new)

    fig = go.Figure(go.Scattergeo(
        lon=lon_new, lat=lat_new, mode="markers",
        marker=dict(
            size=sizes, sizemode="diameter",
            color=theme.get("primary", "#2563eb"),
            line=dict(width=1, color=theme.get("plot_bg", "#fff")),
            opacity=0.92
        ),
        customdata=vals,
        hovertemplate=f"{value}: %{{customdata:,.0f}}<extra></extra>",
        showlegend=False,
        name=value,
    ))
    fig.update_layout(
        geo=dict(
            showland=True, showcountries=True, showcoastlines=True,
            lonaxis=dict(range=lonaxis), lataxis=dict(range=lataxis),
            projection=dict(type="natural earth"),
            **geo_extra
        ),
        title=title or "Scaled Cartogram (Dorling)",
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return apply_theme(fig, theme)
