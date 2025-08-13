from pathlib import Path
import argparse
import pandas as pd

from src.chart.common import export_html, export_png
from src.chart import flow

def _save(fig, out_html: Path, out_png: Path | None, title: str):
    # Set figure title (write_html title kw isn't portable across plotly versions)
    fig.update_layout(title=title)
    export_html(fig, out_html)
    if out_png:
        try:
            export_png(fig, out_png)  # uses defaults from common.export_png
        except Exception as e:
            print(f"[WARN] PNG export failed for {out_png.name}: {e}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--outdir", default="data/gold/_demo_charts/flow")
    ap.add_argument("--png", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # --- Sankey sample ---
    sankey_df = pd.DataFrame({
        "src": ["A","A","B","B","C","C","C"],
        "dst": ["X","Y","X","Z","Y","Z","X"],
        "val": [10,5,8,4,6,3,7],
    })
    fig = flow.sankey(
        sankey_df, source="src", target="dst", value="val",
        title="Sankey: A/B/C → X/Y/Z", color_links_by="source"
    )
    _save(fig, outdir/"sankey.html", (outdir/"sankey.png" if args.png else None), "Sankey: A/B/C → X/Y/Z")

    # --- Chord sample ---
    chord_df = pd.DataFrame({
        "src": ["A","A","B","B","C","C","D","D"],
        "dst": ["B","C","C","D","D","A","A","B"],
        "w":   [5,3,4,2,6,3,4,5],
    })
    fig = flow.chord(chord_df, source="src", target="dst", value="w",
                     title="Chord-style circular network", curvature=0.55)
    _save(fig, outdir/"chord.html", (outdir/"chord.png" if args.png else None), "Chord-style circular network")

    # --- Network sample ---
    net_df = pd.DataFrame({
        "src": ["s1","s1","s2","s2","s3","s4","s5","s6"],
        "dst": ["s2","s3","s3","s4","s4","s5","s6","s1"],
        "w":   [2,4,1,5,3,2,4,1],
    })
    fig = flow.network(net_df, source="src", target="dst", value="w",
                       layout="spring", title="Network (spring layout)")
    _save(fig, outdir/"network.html", (outdir/"network.png" if args.png else None), "Network (spring layout)")

    # --- Waterfall sample ---
    # P&L-style walk: start -> deltas -> total
    wf_df = pd.DataFrame({
        "step":   ["Starting", "Sales", "Returns", "Marketing", "R&D", "Other", "Total"],
        "amount": [200,        80,      -30,       -25,        -15,   10,      0],
        "meas":   ["total",    "relative","relative","relative","relative","relative","total"],
    })
    fig = flow.waterfall(
        wf_df, label="step", value="amount", measure="meas",
        title="Waterfall: P&L Walk", valueformat=",.0f", valuesuffix=""
    )
    _save(fig, outdir/"waterfall.html", (outdir/"waterfall.png" if args.png else None), "Waterfall: P&L Walk")

    print(f"Saved to: {outdir.resolve()}")

if __name__ == "__main__":
    main()