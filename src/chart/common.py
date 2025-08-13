from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import plotly.io as pio
import plotly.graph_objects as go

# ---- Helpers ----

def _check(df, cols):
    for c in cols:
        if c not in df.columns:
            raise KeyError(f"Missing column: {c}")


# ---- Theme model ----

_THEMES: Dict[str, Dict[str, object]] = {
    # default dark blue theme
    "dark_blue": {
        "paper_bg": "#0f2435",
        "plot_bg": "#0b1e2d",
        "font": "#d7e3ee",
        "grid": "#2a4052",
        "axis": "#c6d3de",
        "primary": "#3BA1FF",
        "accent": "#FF6B6B",
        "good": "#2ecc71",
        "bad": "#e74c3c",
        "neutral": "#95a5a6",
        "seq": [
            "#3BA1FF", "#8BD3FF", "#50E3C2", "#FFD166",
            "#FF6B6B", "#C792EA", "#9CCC65", "#F78C6C",
            "#A7C5EB", "#FDD7AA",
        ],
    },
    "light": {
        "paper_bg": "#ffffff",
        "plot_bg": "#ffffff",
        "font": "#1f2937",
        "grid": "#e5e7eb",
        "axis": "#111827",
        "primary": "#2563eb",
        "accent": "#ef4444",
        "good": "#16a34a",
        "bad": "#dc2626",
        "neutral": "#6b7280",
        "seq": [
            "#2563eb", "#60a5fa", "#10b981", "#f59e0b",
            "#ef4444", "#a78bfa", "#84cc16", "#fb7185",
            "#06b6d4", "#fca5a5",
        ],
    },
}

def theme_from_cfg(theme_name: Optional[str]) -> Dict[str, object]:
    return _THEMES.get(theme_name or "dark_blue", _THEMES["dark_blue"]).copy()

def colorway(theme: Dict[str, object], n: Optional[int] = None) -> List[str]:
    seq = list(theme["seq"])  # type: ignore
    if n is None or n <= len(seq):
        return seq[: n or len(seq)]
    # cycle if more requested
    out: List[str] = []
    i = 0
    while len(out) < n:
        out.append(seq[i % len(seq)])
        i += 1
    return out

def default_template(theme: Dict[str, object]) -> go.layout.Template:
    # Build a layout template (cached by caller if needed)
    t = go.layout.Template()
    t.layout = go.Layout(
        paper_bgcolor=theme["paper_bg"],
        plot_bgcolor=theme["plot_bg"],
        font=dict(color=theme["font"], family="Inter, Segoe UI, Roboto, sans-serif"),
        colorway=colorway(theme),
        xaxis=dict(
            gridcolor=theme["grid"],
            zerolinecolor=theme["grid"],
            linecolor=theme["axis"],
            ticks="outside",
        ),
        yaxis=dict(
            gridcolor=theme["grid"],
            zerolinecolor=theme["grid"],
            linecolor=theme["axis"],
            ticks="outside",
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(0,0,0,0)",
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="left", x=0,
        ),
        margin=dict(l=60, r=20, t=60, b=60),
    )
    return t

def apply_theme(fig: go.Figure, theme: Dict[str, object]) -> go.Figure:
    tmpl = default_template(theme)
    fig.update_layout(template=tmpl)
    return fig

# ---- Export helpers ----

def export_html(fig: go.Figure, out_path: str | Path, title: Optional[str] = None) -> str:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Set title on the figure, not on write_html
    if title:
        fig.update_layout(title=title)

    # Older & current Plotly: no 'title' kw here
    fig.write_html(
        str(out),
        include_plotlyjs="cdn",
        full_html=True,
        # you can optionally set default size here if you want:
        # default_width="100%", default_height="100%",
    )
    return str(out)

def export_png(
    fig: go.Figure,
    out_path: str | Path,
    *,
    width: int = 1200,
    height: int = 700,
    scale: float = 2.0,
    engine: str = "playwright",
) -> str:
    """
    Save a static PNG by screenshotting an HTML rendering with Playwright.
    (Kaleido-free path; works on Py3.12/Windows.)

    Pure: writes only to out_path.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if engine != "playwright":
        raise ValueError("Only 'playwright' engine is supported here.")

    try:
        from tempfile import NamedTemporaryFile
        from playwright.sync_api import sync_playwright
    except Exception as e:
        raise RuntimeError(
            "Playwright not available. Install 'playwright' and run "
            "'python -m playwright install chromium'."
        ) from e

    # Render to a temporary HTML file
    with NamedTemporaryFile("w", suffix=".html", delete=False, encoding="utf-8") as tmp:
        html_path = Path(tmp.name)
        fig.write_html(
            str(html_path),
            include_plotlyjs="cdn",
            full_html=True,
            default_width=width,
            default_height=height,
        )

    # Screenshot with Playwright
    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--disable-gpu", "--no-sandbox"], headless=True)
        try:
            page = browser.new_page()
            # scale via larger viewport for higher native resolution
            page.set_viewport_size({"width": int(width * scale), "height": int(height * scale)})
            page.goto(html_path.as_uri(), wait_until="networkidle")
            # ensure clean margins
            page.evaluate("document.body.style.margin='0';document.documentElement.style.background='transparent';")
            page.screenshot(path=str(out), type="png", full_page=False)
        finally:
            browser.close()

    try:
        html_path.unlink(missing_ok=True)  # cleanup temp file
    except Exception:
        pass

    return str(out)
