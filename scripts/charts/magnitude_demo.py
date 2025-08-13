from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from src.chart import magnitude as mag
from src.chart.common import export_html, export_png
from src.config_model.model import load_config


# ---------------------------
# Data generators
# ---------------------------

def _seed(n: int) -> None:
    np.random.seed(n)

def df_simple(n_cat: int = 8) -> pd.DataFrame:
    cats = [f"C{i+1}" for i in range(n_cat)]
    vals = np.random.randint(10, 120, size=n_cat)
    return pd.DataFrame({"category": cats, "value": vals})

def df_paired(n_cat: int = 8) -> pd.DataFrame:
    cats = [f"C{i+1}" for i in range(n_cat)]
    a = np.random.randint(40, 140, size=n_cat)
    b = (a * (0.6 + 0.8 * np.random.rand(n_cat))).astype(int)  # correlated-ish
    return pd.DataFrame({"category": cats, "this": a, "last": b})

def df_radar(n_entities: int = 3, metrics: int = 5) -> pd.DataFrame:
    ents = [f"E{i+1}" for i in range(n_entities)]
    cols = [f"M{k+1}" for k in range(metrics)]
    rows = []
    for e in ents:
        rows.append([e] + list(np.round(np.random.uniform(20, 95, size=metrics), 1)))
    return pd.DataFrame(rows, columns=["entity"] + cols)

def df_parallel(rows: int = 100) -> pd.DataFrame:
    x1 = np.random.normal(0, 1.0, size=rows)
    x2 = 0.7 * x1 + np.random.normal(0, 0.7, size=rows)
    x3 = np.random.uniform(-2, 2, size=rows)
    x4 = np.sin(x1) + np.random.normal(0, 0.2, size=rows)
    return pd.DataFrame({"X1": x1, "X2": x2, "X3": x3, "X4": x4})

def df_mekko(n_x: int = 3, n_y: int = 3) -> pd.DataFrame:
    xs = [f"Region {c}" for c in list("ABC")[:n_x]]
    ys = [f"Channel {c}" for c in list("XYZ")[:n_y]]
    rows = []
    for x in xs:
        for y in ys:
            rows.append((x, y, int(np.random.randint(20, 120))))
    return pd.DataFrame(rows, columns=["cat_x", "cat_y", "value"])

def df_isotype(n_cat: int = 6) -> pd.DataFrame:
    cats = [f"Item {i+1}" for i in range(n_cat)]
    vals = np.random.randint(8, 45, size=n_cat)
    return pd.DataFrame({"category": cats, "value": vals})

def df_bullet(n_rows: int = 6) -> pd.DataFrame:
    names = [f"KPI {i+1}" for i in range(n_rows)]
    target = np.random.randint(60, 120, size=n_rows)
    value = (target * (0.7 + 0.6 * np.random.rand(n_rows))).astype(float)
    # optional per-row band hints (leave None to let function derive)
    b1 = (target * 0.6).astype(float)
    b2 = (target * 0.9).astype(float)
    return pd.DataFrame({"name": names, "value": value, "target": target, "b1": b1, "b2": b2})

def df_grouped_symbol(n_cat: int = 4, n_groups: int = 3) -> pd.DataFrame:
    cats = [f"P{i+1}" for i in range(n_cat)]
    grps = [f"G{j+1}" for j in range(n_groups)]
    rows = []
    for c in cats:
        for g in grps:
            rows.append((c, g, int(np.random.randint(10, 80))))
    return pd.DataFrame(rows, columns=["category", "group", "value"])


# ---------------------------
# Helpers
# ---------------------------

def _save(fig, out_html: Path, out_png: Path | None, cfg, title: str | None = None):
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
    ap = argparse.ArgumentParser(description="Magnitude charts demo")
    ap.add_argument("-o", "--outdir", type=Path, required=True)
    ap.add_argument("--theme", default=None, help="Theme name (defaults to config.env.theme)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--png", action="store_true", help="Also export PNG via Playwright")
    args = ap.parse_args()

    cfg = load_config()
    theme_name = args.theme or getattr(cfg.env, "theme", "dark_blue")
    _seed(args.seed)

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Column / Bar
    d1 = df_simple(8)
    _save(mag.column(d1, category="category", value="value", theme_name=theme_name),
          outdir / "column.html", (outdir / "column.png" if args.png else None), cfg, "Column")
    _save(mag.bar(d1, category="category", value="value", theme_name=theme_name),
          outdir / "bar.html", (outdir / "bar.png" if args.png else None), cfg, "Bar")

    # 2) Paired Column / Bar
    d2 = df_paired(7)
    _save(mag.paired_column(d2, category="category", value_a="this", value_b="last", theme_name=theme_name),
          outdir / "paired_column.html", (outdir / "paired_column.png" if args.png else None), cfg, "Paired Column")
    _save(mag.paired_bar(d2, category="category", value_a="this", value_b="last", theme_name=theme_name),
          outdir / "paired_bar.html", (outdir / "paired_bar.png" if args.png else None), cfg, "Paired Bar")

    # 3) Radar
    d3 = df_radar(3, 5)
    _save(mag.radar(d3, entity="entity", metrics=[c for c in d3.columns if c != "entity"], theme_name=theme_name),
          outdir / "radar.html", (outdir / "radar.png" if args.png else None), cfg, "Radar")

    # 4) Parallel Coordinates
    d4 = df_parallel(120)
    _save(mag.parallel_coordinates(d4, theme_name=theme_name),
          outdir / "parallel_coordinates.html", (outdir / "parallel_coordinates.png" if args.png else None), cfg, "Parallel Coordinates")

    # 5) Marimekko (mosaic-style via treemap)
    d5 = df_mekko(3, 3)
    _save(mag.marimekko(d5, cat_x="cat_x", cat_y="cat_y", value="value", theme_name=theme_name),
          outdir / "marimekko.html", (outdir / "marimekko.png" if args.png else None), cfg, "Marimekko (mosaic-style)")

    # 6) Isotype (pictogram)
    d6 = df_isotype(6)
    _save(mag.isotype(d6, category="category", value="value", unit=5, per_row=10, symbol="square", theme_name=theme_name),
          outdir / "isotype.html", (outdir / "isotype.png" if args.png else None), cfg, "Isotype (1 icon = 5 units)")

    # 7) Bullet chart
    d7 = df_bullet(6)
    _save(mag.bullet_chart(d7, category="name", value="value", target="target", band1="b1", band2="b2", theme_name=theme_name),
          outdir / "bullet_chart.html", (outdir / "bullet_chart.png" if args.png else None), cfg, "Bullet Chart")

    # 8) Lollipop (vertical)
    d8 = df_simple(10)
    _save(mag.lollipop(d8, category="category", value="value", orientation="v", sort="desc", theme_name=theme_name),
          outdir / "lollipop_v.html", (outdir / "lollipop_v.png" if args.png else None), cfg, "Lollipop (vertical)")
    # 9) Lollipop (horizontal, top-k)
    _save(mag.lollipop(d8, category="category", value="value", orientation="h", sort="desc", topk=7, theme_name=theme_name),
          outdir / "lollipop_h_topk.html", (outdir / "lollipop_h_topk.png" if args.png else None), cfg, "Lollipop (horizontal, top-k)")

    # 10) Grouped symbol chart
    d9 = df_grouped_symbol(4, 3)
    _save(mag.grouped_symbol(d9, category="category", group="group", value="value",
                             unit=5, per_row=10, symbol="square", theme_name=theme_name),
          outdir / "grouped_symbol.html", (outdir / "grouped_symbol.png" if args.png else None), cfg,
          "Grouped Symbols (1 icon = 5 units)")

    print(f"Wrote charts to: {outdir.resolve()}")
    if args.png:
        print("Static PNGs attempted via Playwright. If any failed, see warnings above.")


if __name__ == "__main__":
    main()
