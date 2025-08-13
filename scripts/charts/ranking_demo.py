from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from src.chart.common import export_html, export_png
from src.chart import ranking as rk


# ---------------------------
# Helpers
# ---------------------------

def _save(fig, out_html: Path, out_png: Path | None, title: str):
    export_html(fig, out_html, title=title)
    if out_png:
        export_png(fig, out_png)


# ---------------------------
# Synthetic data builders
# ---------------------------

def seed(n: int = 42) -> None:
    np.random.seed(n)

def df_rank_items(n: int = 12) -> pd.DataFrame:
    cats = [f"Item {i+1}" for i in range(n)]
    vals = np.random.randint(10, 500, size=n)
    return pd.DataFrame({"category": cats, "value": vals})

def df_strip(n_groups: int = 6, points_per: int = 80) -> pd.DataFrame:
    rows = []
    for g in range(n_groups):
        mu = np.random.uniform(40, 70)
        sig = np.random.uniform(6, 12)
        vals = np.random.normal(mu, sig, size=points_per)
        rows.extend([(f"G{g+1}", v) for v in vals])
    return pd.DataFrame(rows, columns=["category", "value"])

def df_slope(n_entities: int = 10) -> pd.DataFrame:
    t0, t1 = pd.Timestamp("2024-01-01"), pd.Timestamp("2024-06-01")
    ents = [f"E{i+1}" for i in range(n_entities)]
    base = np.random.uniform(50, 150, size=n_entities)
    growth = np.random.uniform(-30, 40, size=n_entities)
    v0 = base
    v1 = base + growth
    rows = [(e, t0, v) for e, v in zip(ents, v0)] + [(e, t1, v) for e, v in zip(ents, v1)]
    return pd.DataFrame(rows, columns=["entity", "time", "value"])

def df_bump(n_entities: int = 14, n_times: int = 6) -> pd.DataFrame:
    times = [f"T{i+1}" for i in range(n_times)]
    ents = [f"E{i+1}" for i in range(n_entities)]
    rows = []
    for e in ents:
        baseline = np.random.uniform(60, 120)
        trend = np.random.uniform(-6, 6)
        for ti, t in enumerate(times):
            noise = np.random.normal(0, 10)
            rows.append((e, t, max(0, baseline + trend * ti + noise)))
    return pd.DataFrame(rows, columns=["entity", "time", "value"])


# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Ranking charts demo")
    ap.add_argument("-o", "--outdir", default="data/gold/_demo_charts/ranking", help="Output directory")
    ap.add_argument("--theme", default="dark_blue", help="Theme name")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--png", action="store_true", help="Also export PNGs")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    seed(args.seed)
    theme_name = args.theme

    # 1) Ordered Bar (horizontal)
    d1 = df_rank_items(12)
    fig1 = rk.ordered_bar(d1, category="category", value="value",
                          ascending=False, title="Ordered Bar (Top→Bottom)", theme_name=theme_name)
    _save(fig1, outdir / "ordered_bar.html", (outdir / "ordered_bar.png" if args.png else None), "Ordered Bar")

    # 2) Ordered Column (vertical)
    d2 = df_rank_items(12)
    fig2 = rk.ordered_column(d2, category="category", value="value",
                             ascending=False, title="Ordered Column (Left→Right)", theme_name=theme_name)
    _save(fig2, outdir / "ordered_column.html", (outdir / "ordered_column.png" if args.png else None), "Ordered Column")

    # 3) Lollipop (horizontal)
    d3 = df_rank_items(15)
    fig3 = rk.lollipop(d3, category="category", value="value",
                       ascending=False, topk=12, title="Lollipop (Top 12)", theme_name=theme_name)
    _save(fig3, outdir / "lollipop.html", (outdir / "lollipop.png" if args.png else None), "Lollipop")

    # 4) Dot / Strip Plot (jittered)
    d4 = df_strip(n_groups=6, points_per=100)
    fig4 = rk.dot_strip_plot(d4, category="category", value="value",
                             title="Dot / Strip Plot", theme_name=theme_name, jitter=0.3)
    _save(fig4, outdir / "dot_strip_plot.html", (outdir / "dot_strip_plot.png" if args.png else None), "Dot / Strip Plot")

    # 5) Slope Chart (two time points)
    d5 = df_slope(n_entities=10)
    fig5 = rk.slope(d5, entity="entity", time="time", value="value",
                    title="Slope Chart (Earliest vs Latest)", theme_name=theme_name)
    _save(fig5, outdir / "slope.html", (outdir / "slope.png" if args.png else None), "Slope Chart")

    # 6) Ordered Proportional Symbols
    d6 = df_rank_items(14)
    fig6 = rk.ordered_proportional_symbol(d6, category="category", value="value",
                                          ascending=False, topk=12,
                                          title="Ordered Proportional Symbols", theme_name=theme_name)
    _save(fig6, outdir / "ordered_proportional_symbol.html",
          (outdir / "ordered_proportional_symbol.png" if args.png else None),
          "Ordered Proportional Symbols")

    # 7) Bump Chart (ranks over time)
    d7 = df_bump(n_entities=14, n_times=6)
    fig7 = rk.bump(d7, entity="entity", time="time", value="value",
                   topk=10, title="Bump Chart (Top 10)", theme_name=theme_name)
    _save(fig7, outdir / "bump.html", (outdir / "bump.png" if args.png else None), "Bump Chart")

    print(f"Wrote charts to: {outdir.resolve()}")
    if args.png:
        print("Static PNGs created (via export_png).")


if __name__ == "__main__":
    main()
