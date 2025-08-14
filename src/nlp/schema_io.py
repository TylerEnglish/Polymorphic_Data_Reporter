from __future__ import annotations
from pathlib import Path, PurePosixPath
from typing import Dict, Any
from tomlkit import document, dumps  # table() not needed

def _none_to_empty(x):
    return "" if x is None else x

def to_toml(proposed: Dict[str, Any]) -> str:
    doc = document()
    doc.add("dataset", proposed["dataset"])
    doc.add("version", proposed.get("version", 1))
    doc.add("schema_confidence", round(float(proposed.get("schema_confidence", 0.0)), 4))

    cols_tbl = []
    for c in proposed["columns"]:
        hints = c.get("hints", {})
        cols_tbl.append({
            "name": c["name"],
            "dtype": c["dtype"],
            "role": c["role"],
            "role_confidence": round(float(c["role_confidence"]), 4),
            "hints": {
                "unit_hint": _none_to_empty(hints.get("unit_hint")),
                "domain_guess": _none_to_empty(hints.get("domain_guess")),
                "canonical_map": hints.get("canonical_map") or {},
            },
        })
    doc.add("columns", cols_tbl)
    if proposed.get("notes"):
        doc.add("notes", proposed["notes"])
    return dumps(doc)

def schema_path_for_slug(project_root: Path, slug: str) -> PurePosixPath:
    """
    Return a POSIX-style path so string comparisons are stable across OSes,
    while still behaving like a path (has .parent, .name, etc.).
    """
    base = PurePosixPath(Path(project_root).as_posix())
    return base.joinpath("schemas", f"{slug}.schema.toml")
