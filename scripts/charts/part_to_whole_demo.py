from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from src.chart import part_to_whole as ptw
from src.chart.common import export_html, export_png
from src.config_model.model import load_config


# ---------------------------
# Data generators
# ---------------------------

def _seed(n: int) -> None:
    np.random.seed(n)

def df_stacked(n_cat: int = 6, parts: int = 3) -> pd.DataFrame:
    cats = [f"C{i+1}" for i in range(n_cat)]
    prts = [f"P{j+1}" for j in range(parts)]
    rows = []
    for c in cats:
        base = np.abs(np.random.normal(40, 12, size=parts))
        base = np.maximum(base, 2.0)
        for p, v in zip(prts, base):
            rows.append((c, p, float(v)))
    return pd.DataFrame(rows, columns=["category", "part", "value"])

def df_treemap() -> pd.DataFrame:
    rows = []
    rows.append(("Total", "", 0))
    for seg in ["A", "B", "C"]:
        seg_val = np.random.uniform(200, 400)
        rows.append((seg, "Total", seg_val))
        for k in range(np.random.randint(2, 5)):
            rows.append((f"{seg}-{k+1}", seg, float(np.random.uniform(30, 120))))
    return pd.DataFrame(rows, columns=["label", "parent", "value"])

def df_pie(n: int = 8) -> pd.DataFrame:
    labs = [f"S{i+1}" for i in range(n)]
    vals = np.random.uniform(10, 60, size=n)
    return pd.DataFrame({"category": labs, "value": vals})

def df_waterfall() -> pd.DataFrame:
    labels = ["Start", "Revenue", "COGS", "OpEx", "Other", "Total"]
    values = [200, 120, -140, -60, 30, 0]
    return pd.DataFrame({"label": labels, "value": values})

def df_grid_single() -> pd.DataFrame:
    return pd.DataFrame({"pct": [np.random.uniform(0.25, 0.9)]})

def df_grid_grouped(k: int = 4) -> pd.DataFrame:
    cats = [f"G{i+1}" for i in range(k)]
    pcts = np.random.uniform(0.2, 0.95, size=k) * 100
    return pd.DataFrame({"group": cats, "pct": pcts})

def df_voronoi(n: int = 42, with_groups: bool = True) -> pd.DataFrame:
    centers = np.array([[0, 0], [5, 2], [2, 6]])
    pts = []
    groups = []
    for i, (cx, cy) in enumerate(centers):
        m = n // len(centers)
        cloud = np.random.normal([cx, cy], [0.8, 0.8], size=(m, 2))
        pts.append(cloud)
        if with_groups:
            groups.extend([f"Cluster {i+1}"] * m)
    XY = np.vstack(pts)
    df = pd.DataFrame(XY, columns=["x", "y"])
    if with_groups:
        df["grp"] = groups
    return df

def df_mekko() -> pd.DataFrame:
    cols = ["North", "South", "East", "West"]
    subs = ["A", "B", "C"]
    rows = []
    for c in cols:
        base = np.abs(np.random.normal(120, 40, size=len(subs)))
        base = base / base.sum() * np.random.uniform(300, 600)
        for s, v in zip(subs, base):
            rows.append((c, s, float(v)))
    return pd.DataFrame(rows, columns=["col", "sub", "value"])

def df_hemicycle(total_seats: int = 180, parties: int = 5) -> pd.DataFrame:
    """Random party seat allocation that sums to total_seats."""
    names = [f"Party {i+1}" for i in range(parties)]
    # Dirichlet → proportions → round → fix sum
    p = np.random.dirichlet(np.ones(parties))
    seats = np.floor(p * total_seats).astype(int)
    diff = total_seats - seats.sum()
    # Distribute remainder
    for i in np.argsort(-p)[:abs(diff)]:
        seats[i] += 1 if diff > 0 else -1
        diff = total_seats - seats.sum()
        if diff == 0:
            break
    seats = np.maximum(seats, 0)
    if seats.sum() == 0:
        seats[0] = total_seats
    return pd.DataFrame({"party": names, "seats": seats})


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
    ap = argparse.ArgumentParser(description="Part-to-Whole charts demo")
    ap.add_argument("-o", "--outdir", type=Path, required=True, help="Output directory")
    ap.add_argument("--theme", default=None, help="Theme name (defaults to config.env.theme)")
    ap.add_argument("--seed", type=int, default=7, help="Random seed")
    ap.add_argument("--png", action="store_true", help="Also export PNGs via Playwright")
    args = ap.parse_args()

    cfg = load_config()
    theme_name = args.theme or getattr(cfg.env, "theme", "dark_blue")
    _seed(args.seed)

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Stacked Bar
    df1 = df_stacked(n_cat=6, parts=3)
    fig1 = ptw.stacked_bar(df1, category="category", part="part", value="value",
                           title="Stacked Bar", theme_name=theme_name)
    _save(fig1, outdir / "stacked_bar.html", (outdir / "stacked_bar.png" if args.png else None), cfg)

    # 2) Treemap
    df2 = df_treemap()
    fig2 = ptw.treemap(df2, labels="label", parents="parent", value="value",
                       title="Treemap", theme_name=theme_name)
    _save(fig2, outdir / "treemap.html", (outdir / "treemap.png" if args.png else None), cfg)

    # 3) Pie (guarded)
    df3 = df_pie(n=8)
    fig3 = ptw.pie_guarded(df3, category="category", value="value", max_slices=5,
                           title="Pie (guarded, with Other)", theme_name=theme_name)
    _save(fig3, outdir / "pie_guarded.html", (outdir / "pie_guarded.png" if args.png else None), cfg)

    # 4) Donut (guarded)
    fig4 = ptw.donut_guarded(df3, category="category", value="value", max_slices=5,
                             title="Donut (guarded, with Other)", theme_name=theme_name)
    _save(fig4, outdir / "donut_guarded.html", (outdir / "donut_guarded.png" if args.png else None), cfg)

    # 5) Waterfall
    df5 = df_waterfall()
    fig5 = ptw.waterfall(df5, label="label", value="value", measure=None,
                         title="Waterfall", theme_name=theme_name)
    _save(fig5, outdir / "waterfall.html", (outdir / "waterfall.png" if args.png else None), cfg)

    # 6) Venn (2-set)
    fig6 = ptw.venn2(
        label_a="Group A", label_b="Group B",
        count_a=60, count_b=50, count_ab=20,
        title="Venn (2-set, schematic)", theme_name=theme_name,
    )
    _save(fig6, outdir / "venn2.html", (outdir / "venn2.png" if args.png else None), cfg)

    # 7) Venn (3-set)
    fig7 = ptw.venn3(
        label_a="A", label_b="B", label_c="C",
        count_a=80, count_b=70, count_c=65,
        count_ab=30, count_ac=25, count_bc=20,
        count_abc=10,
        title="Venn (3-set, schematic)", theme_name=theme_name,
    )
    _save(fig7, outdir / "venn3.html", (outdir / "venn3.png" if args.png else None), cfg)

    # 8) Gridplot (single)
    df8 = df_grid_single()
    fig8 = ptw.gridplot(df8.rename(columns={"pct": "percent"}), percent="percent",
                        title="Gridplot (single percent)", theme_name=theme_name)
    _save(fig8, outdir / "gridplot_single.html", (outdir / "gridplot_single.png" if args.png else None), cfg)

    # 9) Gridplot (grouped)
    df9 = df_grid_grouped(k=4)
    fig9 = ptw.gridplot(df9.rename(columns={"pct": "percent", "group": "category"}),
                        percent="percent", category="category",
                        title="Gridplot (grouped)", theme_name=theme_name)
    _save(fig9, outdir / "gridplot_grouped.html", (outdir / "gridplot_grouped.png" if args.png else None), cfg)

    # 10) Voronoi (SciPy required)
    df10 = df_voronoi(n=42, with_groups=True)
    try:
        fig10 = ptw.voronoi(df10, x="x", y="y", color="grp",
                            title="Voronoi (tessellation, colored by cluster)", theme_name=theme_name)
        _save(fig10, outdir / "voronoi.html", (outdir / "voronoi.png" if args.png else None), cfg)
    except ImportError as e:
        print(f"[WARN] Skipping Voronoi (SciPy missing): {e}")

    # 11) Marimekko (mosaic-style via treemap)
    df11 = df_mekko()
    fig11 = ptw.marimekko(df11, cat_x="col", cat_y="sub", value="value",
                          title="Marimekko (mosaic-style)", theme_name=theme_name)
    _save(fig11, outdir / "marimekko.html", (outdir / "marimekko.png" if args.png else None), cfg)

    # 12) Arc / Hemicycle (seats)
    df12 = df_hemicycle(total_seats=180, parties=5)
    fig12 = ptw.arc_hemicycle(df12, group="party", seats="seats",
                              rows=None,                    # let it auto-arrange rows
                              start_angle_deg=180, end_angle_deg=0,
                              radius=1.0,
                              title="Arc / Hemicycle (random seat allocation)",
                              theme_name=theme_name)
    _save(fig12, outdir / "arc_hemicycle.html", (outdir / "arc_hemicycle.png" if args.png else None), cfg)

    print(f"Wrote part-to-whole charts to: {outdir.resolve()}")
    if args.png:
        print("Static PNGs attempted via Playwright – see warnings above if failures occurred.")


if __name__ == "__main__":
    main()
