from __future__ import annotations
from pathlib import Path
from datetime import datetime
import json
import traceback
import pandas as pd

from src.config_model.model import RootCfg
from src.gold.materialize import build_gold_from_silver

def find_slugs(silver_root: Path) -> list[str]:
    if not silver_root.exists():
        return []
    return sorted([
        d.name for d in silver_root.iterdir()
        if d.is_dir() and (d / "dataset.parquet").exists()
    ])

def _peek(df: pd.DataFrame, cols: list[str], n: int = 10) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    keep = [c for c in cols if c in df.columns]
    if not keep:
        return pd.DataFrame()
    out = df[keep].copy()
    for c in out.columns:
        if out[c].dtype == object:
            out[c] = out[c].astype(str).str.slice(0, 120)
    return out.head(n)

def _safe_sort(df: pd.DataFrame, by_candidates: list[str], ascending=False) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    for col in by_candidates:
        if col in df.columns:
            return df.sort_values(col, ascending=ascending)
    return df

def main(config_path: str | None = None, recheck: bool = True, render_charts: bool = True) -> None:
    cfg = RootCfg.load(config_path)
    try:
        cfg.reports.enabled_generators.charts = bool(render_charts)
    except Exception:
        pass

    root = Path.cwd()
    silver_root = root / "data" / "silver"
    slugs = find_slugs(silver_root)
    if not slugs:
        print(f"No slugs found under {silver_root}")
        return

    rows = []
    for slug in slugs:
        print(f"\n=== {slug} ===")
        try:
            arts = build_gold_from_silver(cfg, slug, recheck=recheck)

            # read artifacts
            topics_path    = Path(arts["topics"])
            selected_path  = Path(arts["topics_selected"])
            plan_path      = Path(arts["layout_plan"])
            manifest_path  = Path(arts["manifest"])

            t  = pd.read_parquet(topics_path)          if topics_path.exists()   else pd.DataFrame()
            ts = pd.read_parquet(selected_path)        if selected_path.exists() else pd.DataFrame()
            plan  = json.loads(plan_path.read_text(encoding="utf-8")) if plan_path.exists() else {"sections": []}
            manif = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {"components": [], "charts": []}

            n_candidates = len(t)
            n_selected   = len(ts)
            n_sections   = len(plan.get("sections", []))
            n_components = len(manif.get("components", []))
            n_charts     = len(manif.get("charts", []))

            print(f"  candidates={n_candidates}  selected={n_selected}  sections={n_sections}  components={n_components}  charts={n_charts}")

            # Prefer scored previews from selected; fall back to candidates
            if n_selected:
                print("  Selected topics (top by score_total):")
                ts_sorted = _safe_sort(ts, ["score_total"], ascending=False)
                print(_peek(ts_sorted, ["topic_id","family","score_total","primary_fields","secondary_fields","proposed_charts"], 8).to_string(index=False))
            else:
                print("  No topics selected — check thresholds or candidate generation for this slug.")

            if n_candidates:
                print("  Candidate snapshot:")
                t_sorted = _safe_sort(t, ["score_total", "effect_size"], ascending=False)
                print(_peek(t_sorted, ["topic_id","family","effect_size","coverage_pct","significance","primary_fields","proposed_charts"], 8).to_string(index=False))

            # Quick peeks saved under runs/<slug>/
            run_dir = Path("runs") / slug
            run_dir.mkdir(parents=True, exist_ok=True)
            _safe_sort(t, ["score_total", "effect_size"]).head(50)\
                .to_csv(run_dir / "topics_preview.csv", index=False)
            _safe_sort(ts, ["score_total"]).head(50)\
                .to_csv(run_dir / "topics_selected_preview.csv", index=False)
            (run_dir / "layout_plan.peek.json").write_text(
                json.dumps({"sections": plan.get("sections", [])[:3]}, indent=2), encoding="utf-8"
            )
            (run_dir / "manifest.peek.json").write_text(
                json.dumps({"components": manif.get("components", [])[:5], "charts": manif.get("charts", [])[:5]}, indent=2), encoding="utf-8"
            )

            rows.append({
                "slug": slug,
                "status": "ok",
                **arts,
                "candidates": n_candidates,
                "selected": n_selected,
                "sections": n_sections,
                "components": n_components,
                "charts": n_charts,
                "error": "",
            })
        except Exception as e:
            rows.append({
                "slug": slug,
                "status": "fail",
                "topics": "",
                "topics_selected": "",
                "layout_plan": "",
                "manifest": "",
                "candidates": 0,
                "selected": 0,
                "sections": 0,
                "components": 0,
                "charts": 0,
                "error": f"{type(e).__name__}: {e}",
            })
            traceback.print_exc()

    df = pd.DataFrame(rows)
    out_dir = Path("runs")
    out_dir.mkdir(exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_csv = out_dir / f"gold_summary_{stamp}.csv"
    df.to_csv(out_csv, index=False)
    print("\nSummary:")
    print(df[["slug","status","candidates","selected","sections","components","charts","error"]])
    print(f"\nSaved summary -> {out_csv}")

if __name__ == "__main__":
    main(config_path=None, recheck=True, render_charts=False)
