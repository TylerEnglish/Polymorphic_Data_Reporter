from __future__ import annotations
import argparse
from pathlib import Path
import sys
import math
import numpy as np
import pandas as pd

from src.chart import correlation as corr
from src.chart.common import export_html, export_png
try:
    from src.config_model.model import load_config
except Exception:
    load_config = None  # fallback later


def _ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def _save(fig, out_html: Path, out_png: Path | None, width: int, height: int, scale: float, engine: str):
    _ensure_dir(out_html)
    export_html(fig, str(out_html))
    if out_png is not None:
        try:
            _ensure_dir(out_png)
            export_png(fig, str(out_png), width=width, height=height, scale=scale, engine=engine)
        except Exception as e:
            # Keep going even if PNG fails (e.g., missing Chromium)
            print(f"[warn] PNG export failed for {out_png.name}: {e}\n"
                  f"       If using Playwright, run:  playwright install chromium", file=sys.stderr)


def _mk_corr_data(n=400, seed=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    y = 0.8 * x + rng.normal(0, 0.4, n)           # moderately strong positive correlation
    group = rng.choice(["A", "B", "C"], size=n, p=[0.5, 0.3, 0.2])
    size = np.abs(rng.normal(40, 15, n)).clip(5, 120)

    # a time index for connected trajectory & timeline
    # we'll make ~n unique timestamps at minute frequency
    start = pd.Timestamp("2023-01-01 00:00:00")
    time = pd.date_range(start, periods=n, freq="min")
    # synth "volume" & "price" for column+line timeline
    volume = np.abs(rng.normal(1000, 300, n)).astype(int)
    price = 50 + np.cumsum(rng.normal(0, 0.3, n))  # gentle random walk

    return pd.DataFrame({
        "x": x,
        "y": y,
        "group": group,
        "size": size,
        "time": time,
        "volume": volume,
        "price": price,
    })


def main():
    parser = argparse.ArgumentParser(description="Render correlation chart demos (HTML + optional PNG).")
    parser.add_argument("-o", "--outdir", default="data/gold/_demo_charts/correlation", help="Output directory")
    parser.add_argument("--n", type=int, default=400, help="Number of points")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--theme", default=None, help="Override theme name (defaults to cfg.env.theme or 'dark_blue')")
    parser.add_argument("--png", action="store_true", help="Also export PNG")
    parser.add_argument("--engine", default="playwright", choices=["playwright", "kaleido"], help="PNG engine")
    parser.add_argument("--width", type=int, default=None, help="PNG width (px)")
    parser.add_argument("--height", type=int, default=None, help="PNG height (px)")
    parser.add_argument("--scale", type=float, default=None, help="PNG device scale factor")

    args = parser.parse_args()
    outdir = Path(args.outdir).resolve()

    # Load config (theme + png sizing) if available
    theme_name = "dark_blue"
    png_width, png_height, png_scale = 1200, 700, 2.0
    if load_config:
        try:
            cfg = load_config()
            theme_name = args.theme or getattr(cfg.env, "theme", theme_name)
            png_width = args.width or getattr(cfg.charts, "png_width", png_width)
            png_height = args.height or getattr(cfg.charts, "png_height", png_height)
            png_scale = args.scale or getattr(cfg.charts, "png_scale", png_scale)
        except Exception as e:
            print(f"[warn] Could not load config/config.toml: {e}", file=sys.stderr)
            theme_name = args.theme or theme_name
            png_width = args.width or png_width
            png_height = args.height or png_height
            png_scale = args.scale or png_scale
    else:
        theme_name = args.theme or theme_name
        png_width = args.width or png_width
        png_height = args.height or png_height
        png_scale = args.scale or png_scale

    df = _mk_corr_data(n=args.n, seed=args.seed)

    # 1) Scatter
    fig = corr.scatter(df, x="x", y="y", color="group", title="Demo: Scatter (x vs y)", theme_name=theme_name)
    _save(fig, outdir / "scatter.html", (outdir / "scatter.png" if args.png else None),
          png_width, png_height, png_scale, args.engine)

    # 2) Bubble (alias; use 'size' as marker size)
    fig = corr.bubble(df=df, x="x", y="y", color="group", size="size",
                      title="Demo: Bubble (x vs y, size)", theme_name=theme_name)
    _save(fig, outdir / "bubble.html", (outdir / "bubble.png" if args.png else None),
          png_width, png_height, png_scale, args.engine)

    # 3) Connected Scatter (trajectory over time)
    # Use a small subset to keep path readable
    df_small = df.iloc[::max(1, len(df)//80)].copy()
    fig = corr.connected_scatter(
        df_small, x="x", y="y", order_by="time", color="group",
        title="Demo: Connected Scatter (trajectory over time)", theme_name=theme_name
    )
    _save(fig, outdir / "connected_scatter.html", (outdir / "connected_scatter.png" if args.png else None),
          png_width, png_height, png_scale, args.engine)

    # 4) XY Density Heatmap (large-ish sample to show bins)
    df_dense = _mk_corr_data(n=max(1000, args.n * 5), seed=args.seed + 1)
    fig = corr.xy_heatmap(df_dense, x="x", y="y", nbinsx=40, nbinsy=40,
                          title="Demo: XY Density Heatmap", theme_name=theme_name)
    _save(fig, outdir / "xy_heatmap.html", (outdir / "xy_heatmap.png" if args.png else None),
          png_width, png_height, png_scale, args.engine)

    # 5) Column + Line Timeline (volume bars + price line)
    # Aggregate by hour to smooth
    dd = (
        df.set_index("time")
          .assign(hour=lambda d: d.index.floor("h"))
          .groupby("hour")
          .agg(volume=("volume", "sum"), price=("price", "mean"))
          .reset_index(names="time")
    )
    fig = corr.column_line_timeline(
        dd, time="time", bar_value="volume", line_value="price",
        title="Demo: Column + Line Timeline (volume + price)", theme_name=theme_name
    )
    _save(fig, outdir / "column_line_timeline.html", (outdir / "column_line_timeline.png" if args.png else None),
          png_width, png_height, png_scale, args.engine)

    print(f"Done. Wrote HTML{ ' + PNG' if args.png else '' } to: {outdir}")


if __name__ == "__main__":
    main()
