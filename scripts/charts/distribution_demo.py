from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from src.chart import distribution as dist
from src.chart.common import export_html, export_png
from src.config_model.model import load_config


# ---------------------------
# Data generators
# ---------------------------

def _seed(n: int) -> None:
    np.random.seed(n)

def df_hist(n: int = 2000) -> pd.DataFrame:
    x = np.random.normal(loc=0.0, scale=1.0, size=n)
    return pd.DataFrame({"value": x})

def df_groups(n_per_group: int = 300, groups: int = 3, mu_step: float = 1.0) -> pd.DataFrame:
    rows = []
    for i in range(groups):
        mu = (i - (groups - 1) / 2) * mu_step
        sig = 0.7 + i * 0.15
        arr = np.random.normal(mu, sig, size=n_per_group)
        rows.extend([(f"G{i+1}", v) for v in arr])
    return pd.DataFrame(rows, columns=["group", "value"])

def df_line(n: int = 60, series: int = 2) -> pd.DataFrame:
    t = pd.date_range("2024-01-01", periods=n, freq="D")
    rows = []
    for s in range(series):
        base = 100 + 5 * np.sin(np.linspace(0, 2*np.pi, n) + s * 0.6)
        noise = np.random.normal(0, 1.0, size=n)
        y = base + np.cumsum(noise) * 0.3
        rows.extend([(t[i], y[i], f"S{s+1}") for i in range(n)])
    return pd.DataFrame(rows, columns=["time", "value", "series"])

def df_population_pyramid(bands: int = 9, base: int = 900) -> pd.DataFrame:
    ages = [f"{i*10}-{i*10+9}" for i in range(bands - 1)] + [f"{(bands-1)*10}+"]
    male = np.maximum(base + np.random.randint(-120, 120, size=len(ages)) - np.arange(len(ages))*60, 50)
    female = np.maximum(base + np.random.randint(-120, 120, size=len(ages)) - np.arange(len(ages))*55, 50)
    return pd.DataFrame({"age_band": ages, "male": male, "female": female})


# ---------------------------
# Save helpers
# ---------------------------

def _save(fig, out_html: Path, out_png: Path | None, cfg, title: str | None = None):
    # Write HTML (Plotly doesn't take title in write_html kwargs)
    if title:
        fig.update_layout(title=title)
    export_html(fig, str(out_html))
    if out_png:
        try:
            export_png(
                fig,
                str(out_png),
                width=getattr(cfg.charts, "png_width", 1200),
                height=getattr(cfg.charts, "png_height", 700),
                scale=getattr(cfg.charts, "png_scale", 2.0),
                engine="playwright",
            )
        except Exception as e:
            print(f"[WARN] PNG export failed: {e}")


# ---------------------------
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser(description="Distribution charts demo (current API)")
    parser.add_argument("-o", "--outdir", type=Path, required=True, help="Output directory")
    parser.add_argument("--theme", default=None, help="Theme name (defaults to config.env.theme)")
    parser.add_argument("--seed", type=int, default=11, help="Random seed")
    parser.add_argument("--png", action="store_true", help="Also export PNG via Playwright")

    # knobs for generators
    parser.add_argument("--n", type=int, default=60, help="Number of time points for line")
    parser.add_argument("--groups", type=int, default=3, help="Number of groups for grouped charts")
    args = parser.parse_args()

    cfg = load_config()
    theme_name = args.theme or getattr(cfg.env, "theme", "dark_blue")
    _seed(args.seed)

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Histogram (with bin borders)
    df1 = df_hist(2000)
    fig1 = dist.histogram(
        df1, value="value", bins=35,
        title="Histogram (normal ~ N(0,1))",
        theme_name=theme_name,
        edge=True, edge_width=1.0, opacity=0.9
    )
    _save(fig1, outdir / "histogram.html", (outdir / "histogram.png" if args.png else None), cfg)

    # 2) Violin (grouped)
    df2 = df_groups(n_per_group=300, groups=args.groups, mu_step=1.25)
    fig2 = dist.violin(df2, value="value", by="group",
                       title="Violin by Group (box + points)", theme_name=theme_name)
    _save(fig2, outdir / "violin.html", (outdir / "violin.png" if args.png else None), cfg)

    # 3) Line (single/multi series)
    df3 = df_line(n=args.n, series=2)
    fig3 = dist.line(df3, time="time", value="value", color="series",
                     title="Line (2 series)", theme_name=theme_name)
    _save(fig3, outdir / "line.html", (outdir / "line.png" if args.png else None), cfg)

    # 4) Dot strip plot (by group)
    fig_strip = dist.dot_strip_plot(df2, value="value", by="group", jitter=0.0,
                                    title="Dot Strip Plot (by group)", theme_name=theme_name)
    _save(fig_strip, outdir / "dot_strip_plot.html",
          (outdir / "dot_strip_plot.png" if args.png else None), cfg)

    # 5) Beeswarm (strip + jitter)
    fig_swarm = dist.beeswarm(df2, value="value", by="group", jitter=0.35,
                              title="Beeswarm (by group)", theme_name=theme_name)
    _save(fig_swarm, outdir / "beeswarm.html",
          (outdir / "beeswarm.png" if args.png else None), cfg)

    # 6) Barcode plot (lots of points to show GL path)
    df_barcode = df_hist(5000)
    fig_barcode = dist.barcode_plot(df_barcode, value="value",
                                    title="Barcode Plot", theme_name=theme_name)
    _save(fig_barcode, outdir / "barcode.html",
          (outdir / "barcode.png" if args.png else None), cfg)

    # 7) Boxplot (grouped)
    fig_box = dist.boxplot(df2, value="value", by="group", points="outliers",
                           title="Boxplot by Group", theme_name=theme_name)
    _save(fig_box, outdir / "boxplot_grouped.html",
          (outdir / "boxplot_grouped.png" if args.png else None), cfg)

    # 8) ECDF / Survival
    df_ecdf = df_groups(n_per_group=400, groups=args.groups, mu_step=0.8)
    fig_ecdf = dist.cumulative_curve(df_ecdf, value="value", by="group",
                                     title="ECDF (by group)", theme_name=theme_name)
    _save(fig_ecdf, outdir / "ecdf.html", (outdir / "ecdf.png" if args.png else None), cfg)

    fig_surv = dist.cumulative_curve(df_ecdf, value="value", by="group", complementary=True,
                                     title="Survival Curve (1 - ECDF)", theme_name=theme_name)
    _save(fig_surv, outdir / "survival_curve.html",
          (outdir / "survival_curve.png" if args.png else None), cfg)

    # 9) Frequency polygons (single / grouped / density)
    fig_fp_single = dist.frequency_polygon(df_ecdf, value="value", bins=40,
                                           title="Frequency Polygon", theme_name=theme_name)
    _save(fig_fp_single, outdir / "frequency_polygon.html",
          (outdir / "frequency_polygon.png" if args.png else None), cfg)

    fig_fp_grouped = dist.frequency_polygon(df_ecdf, value="value", bins=40, by="group",
                                            title="Frequency Polygon (grouped)", theme_name=theme_name)
    _save(fig_fp_grouped, outdir / "frequency_polygon_grouped.html",
          (outdir / "frequency_polygon_grouped.png" if args.png else None), cfg)

    fig_fp_density = dist.frequency_polygon(df_ecdf, value="value", bins=40, by="group", density=True,
                                            title="Frequency Polygon (density, grouped)", theme_name=theme_name)
    _save(fig_fp_density, outdir / "frequency_polygon_density.html",
          (outdir / "frequency_polygon_density.png" if args.png else None), cfg)

    print(f"Wrote charts to: {outdir.resolve()}")
    if args.png:
        print("Static PNGs created (Playwright). If any PNG failed, see warnings above.")


if __name__ == "__main__":
    main()