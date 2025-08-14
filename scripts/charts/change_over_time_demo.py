from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

# Your chart module
from src.chart import change_over_time as cot

# ---------- small I/O helper ----------

def _save(fig, outdir: Path, name: str):
    outdir.mkdir(parents=True, exist_ok=True)
    html = outdir / f"{name}.html"
    png  = outdir / f"{name}.png"
    fig.write_html(str(html), include_plotlyjs="cdn")
    # PNG if kaleido is available
    try:
        fig.write_image(str(png), scale=2, width=1000, height=600)
    except Exception:
        pass

# ---------- synthetic data factories (reproducible) ----------

def _rng(seed=42):
    return np.random.default_rng(seed)

def make_line_multi(n=100, groups=("A", "B", "C")):
    rng = _rng(0)
    t0 = pd.Timestamp("2022-01-01")
    dates = pd.date_range(t0, periods=n, freq="D")
    out = []
    for i, g in enumerate(groups):
        base = np.cumsum(rng.normal(0, 0.6, size=n)) + i * 4
        out.append(pd.DataFrame({"time": dates, "value": base, "group": g}))
    return pd.concat(out, ignore_index=True)

def make_column(n=24):
    rng = _rng(1)
    t0 = pd.Timestamp("2023-01-01")
    months = pd.date_range(t0, periods=n, freq="MS")
    y = np.maximum(0, 15 + np.sin(np.linspace(0, 5, n))*8 + rng.normal(0, 2, n))
    return pd.DataFrame({"time": months, "value": y})

def make_column_line(n=24):
    d = make_column(n=n).copy()
    # rate (line) derived from month-over-month change
    v = d["value"].to_numpy()
    rate = np.r_[np.nan, (v[1:] - v[:-1]) / np.maximum(1e-9, v[:-1])] * 100
    d["rate_pct"] = rate
    return d

def make_slope(points=2, entities=8):
    rng = _rng(2)
    t_all = pd.to_datetime(["2019-01-01", "2021-01-01", "2023-01-01"])
    chosen = t_all[[0, -1]] if points == 2 else t_all[[0, 1, 2]]
    out = []
    for i in range(entities):
        vals = np.cumsum(rng.normal(0, 1, size=len(chosen))) + rng.uniform(5, 15)
        out.append(pd.DataFrame({"entity": f"E{i+1}", "time": chosen, "value": vals}))
    return pd.concat(out, ignore_index=True)

def make_area(n=36, groups=("A", "B", "C", "D")):
    rng = _rng(3)
    t0 = pd.Timestamp("2022-01-01")
    dates = pd.date_range(t0, periods=n, freq="MS")
    out = []
    for i, g in enumerate(groups):
        base = np.maximum(0, 20 + 6*np.sin(np.linspace(0, 3, n) + i) + rng.normal(0, 2, n))
        out.append(pd.DataFrame({"time": dates, "group": g, "value": base}))
    return pd.concat(out, ignore_index=True)

def make_ohlc(n=90):
    rng = _rng(4)
    t0 = pd.Timestamp("2024-01-01")
    days = pd.bdate_range(t0, periods=n)  # business days
    price = 100 + np.cumsum(rng.normal(0, 1.2, size=n))
    opens = price + rng.normal(0, 0.6, size=n)
    closes = price + rng.normal(0, 0.6, size=n)
    highs = np.maximum(opens, closes) + rng.uniform(0.3, 1.8, size=n)
    lows  = np.minimum(opens, closes) - rng.uniform(0.3, 1.8, size=n)
    return pd.DataFrame({"time": days, "open": opens, "high": highs, "low": lows, "close": closes})

def make_fan(n_hist=30, n_fwd=24):
    rng = _rng(5)
    t_hist = pd.date_range("2023-01-01", periods=n_hist, freq="MS")
    hist = 50 + np.cumsum(rng.normal(0, 1.0, size=n_hist))
    t_fwd = pd.date_range(t_hist[-1] + pd.offsets.MonthBegin(), periods=n_fwd, freq="MS")
    # baseline forward drift
    drift = np.linspace(0, 6, n_fwd)
    central = hist[-1] + drift
    # symmetric quantiles
    q10 = central - 6 - np.abs(rng.normal(0, 0.8, n_fwd))
    q20 = central - 4 - np.abs(rng.normal(0, 0.7, n_fwd))
    q30 = central - 2 - np.abs(rng.normal(0, 0.6, n_fwd))
    q40 = central - 1 - np.abs(rng.normal(0, 0.4, n_fwd))
    q60 = central + 1 + np.abs(rng.normal(0, 0.4, n_fwd))
    q70 = central + 2 + np.abs(rng.normal(0, 0.6, n_fwd))
    q80 = central + 4 + np.abs(rng.normal(0, 0.7, n_fwd))
    q90 = central + 6 + np.abs(rng.normal(0, 0.8, n_fwd))
    df_hist = pd.DataFrame({"time": t_hist, "central": hist})
    df_fwd  = pd.DataFrame({
        "time": t_fwd, "central": central,
        "q10": q10, "q20": q20, "q30": q30, "q40": q40, "q60": q60, "q70": q70, "q80": q80, "q90": q90
    })
    return pd.concat([df_hist, df_fwd], ignore_index=True)

def make_connected_scatter(n=60):
    rng = _rng(6)
    t = pd.date_range("2023-01-01", periods=n, freq="W")
    x = np.linspace(0, 4*np.pi, n)
    xvar = 50 + 10*np.cos(x) + rng.normal(0, 1.5, n)
    yvar = 40 +  8*np.sin(x) + rng.normal(0, 1.2, n)
    return pd.DataFrame({"time": t, "xvar": xvar, "yvar": yvar})

def make_calendar(days=270):
    rng = _rng(7)
    d = pd.date_range("2024-01-01", periods=days, freq="D")
    val = np.clip(10 + 4*np.sin(np.linspace(0, 12, days)) + rng.normal(0, 2, days), 0, None)
    return pd.DataFrame({"date": d, "value": val})

def make_priestley(n=10, with_groups=True):
    rng = _rng(8)
    base = pd.Timestamp("2023-01-01")
    rows = []
    for i in range(n):
        start = base + pd.Timedelta(days=int(rng.integers(0, 260)))
        dur   = int(rng.integers(20, 120))
        end   = start + pd.Timedelta(days=dur)
        row = {"label": f"Task {i+1}", "start": start, "end": end}
        if with_groups:
            row["group"] = f"Team {1 + (i % 3)}"
        rows.append(row)
    return pd.DataFrame(rows)

def make_circle_timeline(categories=("Americas", "EMEA", "APAC"), n_per=18):
    rng = _rng(9)
    out = []
    for c in categories:
        t = pd.date_range("2023-01-01", periods=n_per, freq="W")
        size = np.clip(rng.normal(100, 30, n_per), 10, None)
        out.append(pd.DataFrame({"time": t, "category": c, "size": size}))
    return pd.concat(out, ignore_index=True)

def make_vertical_timeline(n=25):
    rng = _rng(10)
    t = pd.date_range("2024-02-01", periods=n, freq="D") + pd.to_timedelta(rng.integers(0, 48, n), unit="h")
    cats = np.array([f"Lane {i}" for i in (1, 2, 3)])
    cat = rng.choice(cats, size=n, replace=True)
    return pd.DataFrame({"time": t, "category": cat})

def make_seismogram(n=400):
    rng = _rng(11)
    t = pd.date_range("2024-06-01", periods=n, freq="T")
    base = rng.normal(0, 0.5, n)
    # add a few bursts
    for center in (80, 180, 300):
        idx = slice(center-10, center+10)
        base[idx] += np.hanning(20) * rng.uniform(5, 10)
    return pd.DataFrame({"time": t, "amplitude": base})

def make_streamgraph(T=36, groups=("Alpha", "Beta", "Gamma", "Delta")):
    rng = _rng(12)
    t = pd.date_range("2022-01-01", periods=T, freq="MS")
    rows = []
    for g in groups:
        base = np.clip(20 + 5*np.sin(np.linspace(0, 3, T) + rng.uniform(0, 2*np.pi)) + rng.normal(0, 2, T), 0, None)
        rows.append(pd.DataFrame({"time": t, "group": g, "value": base}))
    return pd.concat(rows, ignore_index=True)


# ---------- main demo ----------

def main():
    ap = argparse.ArgumentParser(description="Change-over-time charts demo")
    ap.add_argument("-o", "--out", default="data/gold/_demo_charts/change_over_time", help="output directory")
    ap.add_argument("--theme", default="dark_blue", help="theme name in config")
    args = ap.parse_args()
    outdir = Path(args.out)

    # 1) Line (multi, irregular markers auto)
    d1 = make_line_multi()
    fig1 = cot.line(d1, time="time", value="value", group="group", markers=None, theme_name=args.theme,
                    title="Line (multi, auto-markers if irregular)")
    _save(fig1, outdir, "01_line")

    # 2) Column (single)
    d2 = make_column()
    fig2 = cot.column(d2, time="time", value="value", theme_name=args.theme, title="Column (single series)")
    _save(fig2, outdir, "02_column")

    # 3) Column + Line (amount vs rate)
    d3 = make_column_line()
    fig3 = cot.column_line(d3, time="time", column_value="value", line_value="rate_pct",
                           line_on_secondary=True, theme_name=args.theme,
                           title="Column + Line (amount vs rate %)")
    _save(fig3, outdir, "03_column_line")

    # 4) Slope (2-point)
    d4 = make_slope(points=2)
    fig4 = cot.slope(d4, entity="entity", time="time", value="value", points=2,
                     theme_name=args.theme, title="Slope (2-point)")
    _save(fig4, outdir, "04_slope_2pt")

    # 5) Area (stacked)
    d5 = make_area()
    fig5 = cot.area(d5, time="time", value="value", group="group", stack=True,
                    theme_name=args.theme, title="Stacked Area")
    _save(fig5, outdir, "05_area_stacked")

    # 6) Candlestick
    d6 = make_ohlc()
    fig6 = cot.candlestick(d6, time="time", open_="open", high="high", low="low", close="close",
                           theme_name=args.theme, title="Candlestick (daily)")
    _save(fig6, outdir, "06_candlestick")

    # 7) Fan chart (projection)
    d7 = make_fan()
    bands = [("q40", "q60"), ("q30", "q70"), ("q20", "q80"), ("q10", "q90")]
    fig7 = cot.fan_chart(d7, time="time", central="central", bands=bands,
                         theme_name=args.theme, band_colorscale="Blues",
                         title="Fan Chart (projection bands)")
    _save(fig7, outdir, "07_fan_chart")

    # 8) Connected Scatterplot (time-ordered)
    d8 = make_connected_scatter()
    fig8 = cot.connected_scatter(d8, time="time", xvar="xvar", yvar="yvar",
                                 theme_name=args.theme, title="Connected Scatter (progression)")
    _save(fig8, outdir, "08_connected_scatter")

    # 9) Calendar heatmap (daily)
    d9 = make_calendar()
    fig9 = cot.calendar_heatmap(d9, date="date", value="value",
                                theme_name=args.theme, colorscale="Turbo",
                                title="Calendar Heatmap (daily)")
    _save(fig9, outdir, "09_calendar_heatmap")

    # 10) Priestley timeline (durations)
    d10 = make_priestley(n=12, with_groups=True)
    fig10 = cot.priestley_timeline(d10, label="label", start="start", end="end",
                                   group="group", theme_name=args.theme,
                                   title="Priestley Timeline (durations)")
    _save(fig10, outdir, "10_priestley_timeline")

    # 11) Circle timeline (bubbles per category)
    d11 = make_circle_timeline()
    fig11 = cot.circle_timeline(d11, time="time", category="category", size_value="size",
                                theme_name=args.theme, title="Circle Timeline")
    _save(fig11, outdir, "11_circle_timeline")

    # 12) Vertical Timeline (time on Y; single path)
    d12 = make_vertical_timeline()
    # Map categories to a small lateral drift so the path wiggles meaningfully
    lane_map = {"Lane 1": -0.6, "Lane 2": 0.0, "Lane 3": 0.6}
    d12["x"] = d12["category"].map(lane_map).astype(float)
    fig12 = cot.vertical_timeline(d12, time="time", x="x",
                                  theme_name=args.theme, title="Vertical Timeline")
    _save(fig12, outdir, "12_vertical_timeline")

    # 13) Seismogram (impulses)
    d13 = make_seismogram()
    fig13 = cot.seismogram(d13, time="time", amplitude="amplitude",
                           theme_name=args.theme, title="Seismogram")
    _save(fig13, outdir, "13_seismogram")

    # 14) Streamgraph
    d14 = make_streamgraph()
    fig14 = cot.streamgraph(d14, time="time", group="group", value="value",
                            theme_name=args.theme, title="Streamgraph (centered stacked areas)")
    _save(fig14, outdir, "14_streamgraph")

    print(f"Wrote demo charts to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
