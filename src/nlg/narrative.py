from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List
import json

from ..io.storage import Storage
from ..io.catalog import Catalog
from .narrative_constants import INVENTORY_KEY

def summarize_inventory(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_kind: Dict[str, int] = {}
    by_slug: Dict[str, int] = {}
    for e in entries:
        by_kind[e["kind"]] = by_kind.get(e["kind"], 0) + 1
        by_slug[e["slug"]] = by_slug.get(e["slug"], 0) + 1
    return {
        "total_files": len(entries),
        "by_kind": dict(sorted(by_kind.items())),
        "by_slug": dict(sorted(by_slug.items())),
    }

def build_narrative(dataset_slug: str, bootstrap: Dict[str, Any], inventory_summary: Dict[str, Any]) -> str:
    lines = []
    lines.append(f"# Dataset: {dataset_slug}")
    lines.append(f"- Rows sampled: {bootstrap.get('row_count', 0)}")
    lines.append(f"- Columns: {bootstrap.get('column_count', 0)}")
    lines.append(f"- Schema confidence: {bootstrap.get('schema_confidence', 0.0):.2f}")
    lines.append("")
    lines.append("## Files discovered")
    lines.append(f"- Total: {inventory_summary['total_files']}")
    for k, v in inventory_summary["by_kind"].items():
        lines.append(f"  - {k}: {v}")
    lines.append("")
    lines.append("## Columns (role → confidence) and suggested fixes")

    for c in bootstrap.get("columns", []):
        hint = c.get("hints", {}) or {}
        m = c.get("metrics", {}) or {}
        s = c.get("suggestions", {}) or {}
        hb = []
        if hint.get("unit_hint"): hb.append(f"unit={hint['unit_hint']}")
        if hint.get("domain_guess"): hb.append(f"domain={hint['domain_guess']}")
        if hint.get("canonical_map_size"): hb.append(f"canon={hint['canonical_map_size']}")
        mb = []
        if "non_null_ratio" in m:
            mb.append(f"non-null={m['non_null_ratio']:.2f}")
        if "unique_ratio" in m:
            mb.append(f"uniq={m['unique_ratio']:.2f}")
        sugg = []
        if s.get("drop"): sugg.append("DROP")
        if s.get("impute"): sugg.append(f"impute={s['impute']}")
        if s.get("normalize"): sugg.append("normalize")
        if s.get("treat_as_null"): sugg.append(f"nullify={','.join(s['treat_as_null'])}")
        hb_s = f" [{', '.join(hb)}]" if hb else ""
        mb_s = f" {{{', '.join(mb)}}}" if mb else ""
        sg_s = f" -> {'; '.join(sugg)}" if sugg else ""
        lines.append(
            f"- {c['name']} : {c['dtype']} → {c['role']} ({c['role_confidence']:.2f}){hb_s}{mb_s}{sg_s}"
        )
    return "\n".join(lines)


def narrative_payload(dataset_slug: str, bootstrap: Dict[str, Any], entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    inventory_summary = summarize_inventory(entries)
    return {
        "dataset": dataset_slug,
        "schema_confidence": bootstrap.get("schema_confidence", 0.0),
        "bootstrap": bootstrap,
        INVENTORY_KEY: inventory_summary,
        "narrative_text": build_narrative(dataset_slug, bootstrap, inventory_summary),
    }
