"""
Quick smoke tests for spatial charts:
- choropleth
- proportional_symbol
- flow_map_lines
- contour_map
- dot_density
- scaled_cartogram
"""
from __future__ import annotations
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

from src.chart.common import export_html, export_png
from src.chart import spatial as sp


# ---------------------------
# Helpers
# ---------------------------

def _save(fig, out_html: Path, out_png: Path | None, title: str):
    # export_html supports a title kwarg in your project helper
    export_html(fig, out_html, title=title)
    if out_png:
        try:
            export_png(fig, out_png)
        except Exception as e:
            print(f"[WARN] PNG export failed for {out_png.name}: {e}")

def _seed(n: int) -> None:
    np.random.seed(n)


# ---------------------------
# Data generators
# ---------------------------

def df_choropleth() -> pd.DataFrame:
    countries = [
        "United States", "Canada", "Mexico",
        "Brazil", "Argentina",
        "United Kingdom", "France", "Germany", "Italy", "Spain",
        "India", "China", "Japan", "Australia", "South Africa"
    ]
    vals = np.abs(np.random.normal(100, 40, size=len(countries))).astype(int)
    return pd.DataFrame({"country": countries, "value": vals})

def df_points(n: int = 800, box=(-20, 55, -100, 30)) -> pd.DataFrame:
    """
    Random points in a broad world-ish bbox:
      lat in [box[0], box[1]], lon in [box[2], box[3]]
    Value ~ mixture of Gaussians to make nice contours/heatmaps.
    """
    lat_min, lat_max, lon_min, lon_max = box
    lat = np.random.uniform(lat_min, lat_max, size=n)
    lon = np.random.uniform(lon_min, lon_max, size=n)

    # Mixture centers
    centers = [
        (35, -5, +1.0),   # North Africa / Med-ish positive blob
        (10, -80, -0.9),  # Central America-ish negative blob
        (50, 10, +0.6),   # Europeish mild positive
    ]
    val = np.zeros(n)
    for (clat, clon, amp) in centers:
        val += amp * np.exp(-(((lat - clat) ** 2) / (2 * 10 ** 2) + ((lon - clon) ** 2) / (2 * 12 ** 2)))
    # add some noise, keep both +/-
    val += np.random.normal(0, 0.15, size=n)
    return pd.DataFrame({"lat": lat, "lon": lon, "value": val})

def df_flows(m: int = 20, box=(25, 50, -15, 30)) -> pd.DataFrame:
    """
    Generate synthetic flows within a smaller bbox (e.g., Europe/North Africa).
    """
    lat_min, lat_max, lon_min, lon_max = box
    lat_from = np.random.uniform(lat_min, lat_max, size=m)
    lon_from = np.random.uniform(lon_min, lon_max, size=m)
    lat_to   = np.random.uniform(lat_min, lat_max, size=m)
    lon_to   = np.random.uniform(lon_min, lon_max, size=m)
    val = np.abs(np.random.normal(50, 20, size=m)).astype(int)
    return pd.DataFrame({
        "lat_from": lat_from, "lon_from": lon_from,
        "lat_to": lat_to, "lon_to": lon_to,
        "value": val
    })

def df_dot_density(n: int = 600, box=(30, 52, -125, -65)) -> pd.DataFrame:
    """
    US-ish bbox points, colored into a few groups.
    """
    lat_min, lat_max, lon_min, lon_max = box
    lat = np.random.uniform(lat_min, lat_max, size=n)
    lon = np.random.uniform(lon_min, lon_max, size=n)
    groups = np.random.choice(["A", "B", "C"], size=n, p=[0.45, 0.35, 0.20])
    return pd.DataFrame({"lat": lat, "lon": lon, "grp": groups})


def df_scaled_cartogram(n: int = 18, box=(35, 60, -10, 30)) -> pd.DataFrame:
    """
    'Regions' with lat/lon + a magnitude to scale bubbles.
    """
    lat_min, lat_max, lon_min, lon_max = box
    lat = np.random.uniform(lat_min, lat_max, size=n)
    lon = np.random.uniform(lon_min, lon_max, size=n)
    val = np.abs(np.random.normal(100, 60, size=n)) + 10
    return pd.DataFrame({"lat": lat, "lon": lon, "value": val})


# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Spatial charts demo")
    ap.add_argument("-o", "--outdir", default="data/gold/_demo_charts/spatial")
    ap.add_argument("--png", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--theme", default=None, help="Theme name (defaults to config/env default if any)")
    args = ap.parse_args()

    _seed(args.seed)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    theme_name = args.theme or "dark_blue"

    # 1) Choropleth
    d1 = df_choropleth()
    fig1 = sp.choropleth(d1, locations="country", value="value",
                         locationmode="country names", title="Choropleth", theme_name=theme_name)
    _save(fig1, outdir / "choropleth.html", (outdir / "choropleth.png" if args.png else None), "Choropleth")

    # 2) Proportional Symbols
    d2 = df_points(n=250, box=(-10, 55, -100, 20))
    fig2 = sp.proportional_symbol(d2, lat="lat", lon="lon", value="value",
                                  title="Proportional Symbol Map", theme_name=theme_name)
    _save(fig2, outdir / "proportional_symbol.html", (outdir / "proportional_symbol.png" if args.png else None),
          "Proportional Symbols")

    # 3) Flow map (lines)
    d3 = df_flows(m=20, box=(25, 50, -15, 30))
    fig3 = sp.flow_map_lines(d3, lat_from="lat_from", lon_from="lon_from",
                             lat_to="lat_to", lon_to="lon_to", value="value",
                             title="Flow Map (lines)", theme_name=theme_name)
    _save(fig3, outdir / "flow_map_lines.html", (outdir / "flow_map_lines.png" if args.png else None),
          "Flow Map (lines)")

    # 4) Contour map (diverging)
    d4 = df_points(n=900, box=(-10, 55, -100, 30))
    fig4 = sp.contour_map(d4, lat="lat", lon="lon", value="value",
                          nbins_lat=70, nbins_lon=70, diverging=True,
                          title="Contour Map (diverging)", theme_name=theme_name)
    _save(fig4, outdir / "contour_map.html", (outdir / "contour_map.png" if args.png else None),
          "Contour Map")

    # 6) Dot density (+ hotspot annotations)
    d6 = df_dot_density(n=700, box=(30, 52, -125, -65))
    fig6 = sp.dot_density(d6, lat="lat", lon="lon", color="grp",
                          annotate_hotspots=True, bins_for_hotspots=(40, 40),
                          topk_hotspots=3, title="Dot Density (with hotspots)",
                          theme_name=theme_name)
    _save(fig6, outdir / "dot_density.html", (outdir / "dot_density.png" if args.png else None),
          "Dot Density")


    # 8) Scaled cartogram (Dorling-style bubbles)
    d8 = df_scaled_cartogram(n=18, box=(35, 60, -10, 30))
    fig8 = sp.scaled_cartogram(d8, lat="lat", lon="lon", value="value",
                               iterations=120, k_repulse=0.6,
                               title="Scaled Cartogram (Dorling-style)", theme_name=theme_name)
    _save(fig8, outdir / "scaled_cartogram.html", (outdir / "scaled_cartogram.png" if args.png else None),
          "Scaled Cartogram")

    print(f"Saved spatial charts to: {outdir.resolve()}")
    if args.png:
        print("Requested PNGs — if any export failed, see warnings above.")


if __name__ == "__main__":
    main()
