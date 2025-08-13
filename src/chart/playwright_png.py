from __future__ import annotations
import json, os, tempfile, pathlib
from typing import Any

_HTML = """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
  <style>
    html,body,#chart { margin:0; padding:0; width:100%; height:100%; background:white; }
  </style>
</head>
<body>
  <div id="chart"></div>
  <script>
    const fig = {FIG_JSON};
    Plotly.newPlot('chart', fig.data, fig.layout, fig.config || {displayModeBar:false});
  </script>
</body>
</html>
"""

def write_png_via_playwright(fig_json: dict, out_path: str, width=1200, height=700, scale=2.0) -> str:
    from playwright.sync_api import sync_playwright  # lazy import

    html = _HTML.replace("{FIG_JSON}", json.dumps(fig_json))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        html_path = os.path.join(tmp, "plot.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": int(width), "height": int(height), "deviceScaleFactor": float(scale)})
            page.goto(pathlib.Path(html_path).as_uri(), wait_until="networkidle")
            page.locator("#chart").screenshot(path=out_path)
            browser.close()
    return out_path