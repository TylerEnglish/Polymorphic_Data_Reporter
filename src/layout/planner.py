from __future__ import annotations
from typing import Dict, List, Any, Iterable
import json
import pandas as pd
from src.config_model.model import RootCfg


def _coerce_list(val: Any) -> List[str]:
    """Accept list | json-encoded list | scalar | None -> list[str]."""
    if val is None:
        return []
    if isinstance(val, list):
        return [str(x) for x in val]
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return []
        # try json list first
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except Exception:
            # fall back: treat as a single field name
            return [s]
        return []
    return [str(val)]


def _parse_json(val: Any, default: Any) -> Any:
    if isinstance(val, str):
        try:
            return json.loads(val)
        except Exception:
            return default
    return default if val is None else val


def _first_chart(row) -> str:
    pcs = row.get("proposed_charts")
    pcs = _coerce_list(pcs)
    if pcs:
        return str(pcs[0])

    # safe fallback based on family
    fam = row["family"]
    if fam == "trend": return "line"
    if fam == "correlation": return "scatter"
    if fam == "ranking": return "ordered_bar"
    if fam == "part_to_whole": return "treemap"
    if fam == "distribution": return "histogram"
    if fam == "deviation": return "diverging_bar"
    if fam == "cohort": return "table"
    if fam == "causal": return "bar"
    if fam == "kpi": return "kpi"
    return "table"


def _badges(row) -> List[str]:
    b: List[str] = []
    fam = str(row.get("family", "")).lower()

    # significance badge if we have a small p-value
    sig = _parse_json(row.get("significance"), {})
    try:
        p = float(sig.get("p_value"))
        if pd.notna(p) and p <= 0.05:
            b.append("significant")
    except Exception:
        pass

    # low coverage
    try:
        cov = float(row.get("coverage_pct", 0.0) or 0.0)
        if cov < 0.5:
            b.append("low-coverage")
    except Exception:
        pass

    # family badges
    if fam == "kpi":
        b.append("key-metric")
    if fam == "causal":
        b.append("causal")

    return b


def make_layout_plan(selected: pd.DataFrame, cfg: RootCfg, *, dataset_slug: str) -> Dict:
    if selected.empty:
        return {
            "report_title": f"{cfg.reports.html.title_prefix} {dataset_slug}",
            "dataset_slug": dataset_slug,
            "generated_at": pd.Timestamp.utcnow().isoformat(),
            "sections": [],
        }

    # ✅ sort by score_total descending; stable for ties
    sel = (
        selected.sort_values("score_total", ascending=False, kind="mergesort")
                .reset_index(drop=True)
    )

    sections: List[Dict] = []
    for i, r in sel.iterrows():
        cid = f"comp-{i+1:03d}"

        pf = _coerce_list(r.get("primary_fields"))
        sf = _coerce_list(r.get("secondary_fields"))
        pcs = _coerce_list(r.get("proposed_charts"))
        selected_chart = _first_chart(r)
        badges = _badges(r)

        components = [{
            "component_id": cid,
            "kind": selected_chart,
            "proposed_charts": pcs,
            "dataset_spec": {
                "family": r["family"],
                "primary_fields": pf,
                "secondary_fields": sf,
                "time_field": r.get("time_field"),
                "topic_id": r.get("topic_id"),
                "dataset_slug": dataset_slug,
            },
            "encodings": {},
            "export": {"html": True, "png": True},
            "badges": badges,
        }]

        title_bits = [str(r["family"]).title()]
        if pf:
            title_bits.append(" – " + ", ".join(pf))

        sections.append({
            "section_id": f"sec-{i+1:03d}",
            "title": "".join(title_bits),
            "topic_ref": r.get("topic_id"),
            "narrative_keys": [],
            "components": components,
        })

    return {
        "report_title": f"{cfg.reports.html.title_prefix} {dataset_slug}",
        "dataset_slug": dataset_slug,
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "sections": sections,
    }

