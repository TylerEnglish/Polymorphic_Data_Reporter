from __future__ import annotations
from pathlib import Path
from typing import Optional
import tempfile

def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

def export_html(fig, out_html: Optional[str] = None) -> Path:
    """Write a self-contained HTML file for a Plotly figure and return its path."""
    from plotly.io import to_html
    html = to_html(fig, full_html=True, include_plotlyjs="cdn")
    if out_html is None:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
        tmp.write(html.encode("utf-8"))
        tmp.flush()
        tmp.close()
        return Path(tmp.name)
    out = Path(out_html)
    _ensure_parent(out)
    out.write_text(html, encoding="utf-8")
    return out

def export_png(
    fig,
    out_path: str,
    *,
    width: int = 1200,
    height: int = 700,
    scale: float = 2.0,
    engine: str = "playwright",
    timeout_ms: int = 10_000,
) -> str:
    """
    Export a Plotly figure to PNG.
    - engine="playwright" (default): render the HTML in headless Chromium and screenshot.
    - engine="kaleido": use fig.write_image if kaleido is installed.
    Returns the absolute path to the PNG.
    """
    out = Path(out_path).resolve()
    _ensure_parent(out)

    if engine == "kaleido":
        try:
            fig.write_image(str(out), format="png", width=width, height=height, scale=scale)
            return str(out)
        except Exception as e:
            raise RuntimeError(
                "Kaleido export failed. Either install kaleido or use engine='playwright'."
            ) from e

    if engine != "playwright":
        raise ValueError(f"Unknown engine '{engine}'. Use 'playwright' or 'kaleido'.")

    # --- Playwright route ---
    try:
        from playwright.sync_api import sync_playwright
    except Exception as e:
        raise RuntimeError(
            "Playwright not available. Install it and run 'playwright install chromium'."
        ) from e

    html_path = export_html(fig)  # tmp file
    html_uri = html_path.resolve().as_uri()  # works cross-platform (Windows too)

    with sync_playwright() as p:
        # Some environments need --allow-file-access-from-files to load local assets
        browser = p.chromium.launch(headless=True, args=["--allow-file-access-from-files"])
        # device_scale_factor gives crisp output; viewport controls capture size
        context = browser.new_context(
            viewport={"width": width, "height": height},
            device_scale_factor=scale,
        )
        page = context.new_page()
        page.goto(html_uri, wait_until="networkidle", timeout=timeout_ms)
        # If the plot extends beyond viewport, you could page.evaluate JS to size container.
        page.screenshot(path=str(out))
        context.close()
        browser.close()

    try:
        html_path.unlink(missing_ok=True)
    except Exception:
        pass

    return str(out)