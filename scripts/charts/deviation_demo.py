from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from src.chart import deviation as dev
from src.chart.common import export_html, export_png
from src.config_model.model import load_config


def _seed(n: int) -> None:
    np.random.seed(n)


def _save(fig, out_html: Path, out_png: Path | None, cfg, title: str | None = None):
    export_html(fig, str(out_html), title=title)
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


def make_diverging_bar_df(ncat: int = 12) -> pd.DataFrame:
    cats = [f"Cat {i+1}" for i in range(ncat)]
    # values centered ~100 with noise
    vals = 100 + np.random.normal(0, 15, size=ncat)
    return pd.DataFrame({"category": cats, "value": vals})


def make_diverging_stacked_df(ncat: int = 10, sub_levels: int = 3) -> pd.DataFrame:
    cats = [f"Group {i+1}" for i in range(ncat)]
    subs = [f"S{i+1}" for i in range(sub_levels)]
    rows = []
    for c in cats:
        base = 100 + np.random.normal(0, 12)  # base per category
        for s in subs:
            # spread around base with sub bias
            v = base + np.random.normal(0, 10) + (subs.index(s) - (sub_levels - 1) / 2) * 4
            rows.append((c, s, v))
    return pd.DataFrame(rows, columns=["category", "subcategory", "value"])


def make_spine_df(ncat: int = 12) -> pd.DataFrame:
    cats = [f"Item {i+1}" for i in range(ncat)]
    pos = np.random.randint(10, 100, size=ncat)
    neg = np.random.randint(10, 100, size=ncat)
    # ensure no zero total (avoid division issues)
    pos = np.maximum(pos, 1)
    neg = np.maximum(neg, 1)
    return pd.DataFrame({"category": cats, "positive": pos, "negative": neg})


def make_surplus_deficit_df(n: int = 50, varying_target: bool = False) -> pd.DataFrame:
    t = pd.date_range("2024-01-01", periods=n, freq="D")
    # smooth signal around ~100 with seasonality + noise
    y = 100 + 10 * np.sin(np.linspace(0, 3 * np.pi, n)) + np.random.normal(0, 2, size=n)
    if varying_target:
        target = 100 + 4 * np.sin(np.linspace(0, 3 * np.pi, n) + np.pi / 6.0)
        df = pd.DataFrame({"time": t, "value": y, "target": target})
    else:
        df = pd.DataFrame({"time": t, "value": y})
    return df


def main():
    parser = argparse.ArgumentParser(description="Deviation charts demo")
    parser.add_argument("-o", "--outdir", type=Path, required=True, help="Output directory")
    parser.add_argument("--theme", default=None, help="Theme name (defaults to config.env.theme)")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument("--ncat", type=int, default=12, help="Number of categories")
    parser.add_argument("--png", action="store_true", help="Also export PNG via Playwright")
    parser.add_argument("--varying-target", action="store_true", help="Use varying target series for surplus/deficit")
    args = parser.parse_args()

    cfg = load_config()
    theme_name = args.theme or getattr(cfg.env, "theme", "dark_blue")
    _seed(args.seed)

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Diverging Bar
    df_div = make_diverging_bar_df(args.ncat)
    fig_div = dev.diverging_bar(
        df_div, category="category", value="value",
        title="Diverging Bar — deviation from mean", theme_name=theme_name
    )
    _save(fig_div, outdir / "diverging_bar.html", (outdir / "diverging_bar.png" if args.png else None), cfg)

    # 2) Diverging Stacked Bar
    df_stack = make_diverging_stacked_df(ncat=max(6, args.ncat // 2), sub_levels=3)
    fig_stack = dev.diverging_stacked_bar(
        df_stack, category="category", subcategory="subcategory", value="value",
        title="Diverging Stacked Bar — deviation from mean", theme_name=theme_name
    )
    _save(fig_stack, outdir / "diverging_stacked_bar.html", (outdir / "diverging_stacked_bar.png" if args.png else None), cfg)

    # 3) Spine (100% diverging)
    df_spine = make_spine_df(args.ncat)
    fig_spine = dev.spine(
        df_spine, category="category", pos_col="positive", neg_col="negative",
        title="Spine Chart (100% centered)", theme_name=theme_name
    )
    _save(fig_spine, outdir / "spine.html", (outdir / "spine.png" if args.png else None), cfg)

    # 4) Surplus / Deficit Line vs Target
    df_sd = make_surplus_deficit_df(60, varying_target=args.varying_target)
    if args.varying_target:
        fig_sd = dev.surplus_deficit_line(
            df_sd, time="time", value="value", target="target",
            title="Surplus / Deficit vs Varying Target", theme_name=theme_name
        )
    else:
        fig_sd = dev.surplus_deficit_line(
            df_sd, time="time", value="value", target=100.0,
            title="Surplus / Deficit vs Target=100", theme_name=theme_name
        )
    _save(fig_sd, outdir / "surplus_deficit_line.html", (outdir / "surplus_deficit_line.png" if args.png else None), cfg)

    print(f"Wrote charts to: {outdir.resolve()}")
    if args.png:
        print("Static PNGs created (Playwright). If any PNG failed, see warnings above.")


if __name__ == "__main__":
    main()
    