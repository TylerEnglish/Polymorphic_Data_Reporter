from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd

from ..io.catalog import Catalog
from ..io.readers import read_any
from ..io.storage import Storage
from ..config_model.model import RootCfg, NLPCfg
from .roles import guess_role, detect_unit_hint, canonicalize_categories, _dtype_str
from .schema import ColumnHints, ColumnSchema, ProposedSchema
from .schema_io import to_toml, schema_path_for_slug

def _sample_dataframe(storage: Storage, uris: List[str], sample_rows: int) -> pd.DataFrame:
    # unchanged
    frames: List[pd.DataFrame] = []
    for u in uris:
        try:
            df = read_any(storage, u, backend="pandas")
            if not isinstance(df, pd.DataFrame):
                df = pd.DataFrame(df)
            if len(df) > sample_rows:
                df = df.head(sample_rows)
            frames.append(df)
        except Exception:
            continue
        if sum(len(x) for x in frames) >= sample_rows:
            break
    return pd.concat(frames, axis=0, ignore_index=True) if frames else pd.DataFrame()

def propose_schema_for_df(slug: str, df: pd.DataFrame) -> ProposedSchema:
    # unchanged except you already optimized imports above
    from .schema import RoleConfidence
    cols: List[ColumnSchema] = []
    for c in df.columns:
        s = df[c]
        role, conf = guess_role(str(c), s)
        hints = ColumnHints(
            unit_hint=detect_unit_hint(s),
            canonical_map=canonicalize_categories(s),
            domain_guess=None,
        )
        cols.append(
            ColumnSchema(
                name=str(c),
                dtype=_dtype_str(s),
                role_confidence=RoleConfidence(role=role, confidence=float(conf)),
                hints=hints,
            )
        )
    schema_conf = float(sum(col.role_confidence.confidence for col in cols)) / max(1, len(cols)) if cols else 0.0
    return ProposedSchema(dataset_slug=slug, columns=cols, schema_confidence=schema_conf)

def run_nlp_bootstrap(
    storage: Storage,
    dataset_slug: str,
    *,
    project_root: Optional[Path] = None,
    nlp_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Reads small samples from raw/<slug> files, proposes schema, and returns artifacts.
    """
    # Base config from TOML if available, else use class defaults
    try:
        base_cfg = RootCfg.load().nlp.model_dump()
    except Exception:
        base_cfg = NLPCfg().model_dump()

    # Merge with caller overrides
    cfg = {**base_cfg, **(nlp_cfg or {})}

    # discover files for slug
    cat = Catalog(storage)
    entries = cat.inventory(dataset_slug=dataset_slug)
    uris = [e.uri for e in entries]

    df = _sample_dataframe(storage, uris, cfg["sample_rows"])
    proposed = propose_schema_for_df(dataset_slug, df) if not df.empty else ProposedSchema(dataset_slug, [], 0.0)

    bootstrap = {
        "dataset": dataset_slug,
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)) if not df.empty else 0,
        "schema_confidence": proposed.schema_confidence,
        "columns": [
            {
                "name": c.name,
                "dtype": c.dtype,
                "role": c.role_confidence.role,
                "role_confidence": c.role_confidence.confidence,
                "hints": {
                    "unit_hint": c.hints.unit_hint,
                    "domain_guess": c.hints.domain_guess,
                    "canonical_map_size": len(c.hints.canonical_map or {}),
                },
            }
            for c in proposed.columns
        ],
    }

    project_root = project_root or Path.cwd()
    schema_toml = to_toml(proposed.to_dict())
    schema_path = schema_path_for_slug(project_root, dataset_slug)

    bronze_dir = project_root.joinpath("data", "bronze", dataset_slug)
    bronze_dir.mkdir(parents=True, exist_ok=True)
    bootstrap_json_path = bronze_dir.joinpath("nlp_bootstrap.json")

    return {
        "proposed_schema": proposed,
        "bootstrap": bootstrap,
        "io": {
            "schema_toml_text": schema_toml,
            "schema_toml_path": str(schema_path),
            "bootstrap_json_path": str(bootstrap_json_path),
        },
        "nlp_cfg": cfg,
        "entries": [e.__dict__ for e in entries],
    }
