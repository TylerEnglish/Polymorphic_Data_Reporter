import os
import importlib
import pytest
from pathlib import Path

from src.config_model.model import load_config

cfg = load_config()

# Skip if playwright isn't importable or chromium not installed
playwright_ready = True
try:
    import playwright
except Exception:
    playwright_ready = False

@pytest.mark.skipif(not playwright_ready, reason="Playwright not available")
def test_export_png_smoke(tmp_path):
    try:
        common = importlib.import_module("src.chart.common")
    except Exception as e:
        pytest.skip(f"src/chart/common.py not ready: {e}")

    from plotly import graph_objects as go
    fig = go.Figure(data=[go.Bar(x=["A","B","C"], y=[1,3,2])])

    png = tmp_path / "bar.png"
    out = common.export_png(
        fig,
        out_path=str(png),
        width=getattr(cfg.charts, "png_width", 1200),
        height=getattr(cfg.charts, "png_height", 700),
        scale=getattr(cfg.charts, "png_scale", 2.0),
        engine="playwright",
    )
    assert Path(out).exists()
    assert Path(out).stat().st_size > 0

@pytest.mark.skipif(not playwright_ready, reason="Playwright not available")
def test_export_png_dimensions_and_invalid_path(tmp_path):
    common = importlib.import_module("src.chart.common")

    from plotly import graph_objects as go
    fig = go.Figure(data=[go.Scatter(x=[1,2,3], y=[3,1,2])])

    # tiny image still renders
    out_tiny = tmp_path / "tiny.png"
    out = common.export_png(fig, str(out_tiny), width=320, height=200, scale=1.0, engine="playwright")
    assert Path(out).exists()

    # invalid folder -> should create parents or raise with helpful error
    bad_dir = tmp_path / "missing" / "nested"
    bad_png = bad_dir / "chart.png"
    out = common.export_png(fig, str(bad_png), width=640, height=360, scale=1.0, engine="playwright")
    assert Path(out).exists()

@pytest.mark.skipif(not playwright_ready, reason="Playwright not available")
def test_export_png_rejects_unknown_engine(tmp_path):
    common = importlib.import_module("src.chart.common")
    from plotly import graph_objects as go
    fig = go.Figure(data=[go.Bar(x=["x"], y=[1])])

    with pytest.raises((ValueError, NotImplementedError)):
        common.export_png(fig, str(tmp_path / "x.png"), width=400, height=300, scale=1.0, engine="unknown")