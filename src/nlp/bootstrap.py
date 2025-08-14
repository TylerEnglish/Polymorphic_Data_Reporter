from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional
from collections import Counter
import pandas as pd

from ..io.catalog import Catalog
from ..io.readers import read_any
from ..io.storage import Storage
from ..config_model.model import RootCfg, NLPCfg
from .roles import guess_role, detect_unit_hint, canonicalize_categories, _dtype_str, _safe_nunique
from .schema import ColumnHints, ColumnSchema, ProposedSchema
from .schema_io import to_toml, schema_path_for_slug

_EMPTY_TOKENS_CANDIDATES = {"", "na", "n/a", "none", "null", "-", "—"}

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

def _series_metrics(s: pd.Series) -> Dict[str, Any]:
    n = int(len(s))
    cover = float(s.notna().mean()) if n else 0.0
    nunique = _safe_nunique(s) if n else 0
    uniq_ratio = float(nunique / n) if n else 0.0
    avg_len = None
    empty_token_hits: List[str] = []
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        vals = s.dropna().astype(str).str.strip().str.lower()
        avg_len = float(vals.map(len).mean()) if not vals.empty else 0.0
        top = Counter(vals).most_common(200)
        empty_token_hits = [t for t, _ in top if t in _EMPTY_TOKENS_CANDIDATES]
    return {
        "non_null_ratio": cover,
        "nunique": nunique,
        "unique_ratio": uniq_ratio,
        "avg_len": avg_len,
        "empty_token_candidates": empty_token_hits,
    }

def _suggest_actions(colname: str, dtype: str, role: str, m: Dict[str, Any]) -> Dict[str, Any]:
    actions: Dict[str, Any] = {}
    cover = m.get("non_null_ratio", 0.0)
    uniq = m.get("unique_ratio", 0.0)

    if cover < 0.10 and role not in {"id", "time"}:
        actions["drop"] = True
    else:
        actions["drop"] = False
        if role == "numeric":
            actions["impute"] = "median"
        elif role in {"categorical", "text"}:
            actions["impute"] = "Unknown" if role == "categorical" else "N/A"
        elif role == "time":
            actions["impute"] = "parse/ffill"
        elif role == "bool":
            actions["impute"] = "mode"

    # Normalize hints (guard for None avg_len)
    avg_len = m.get("avg_len")
    if dtype == "string" and (avg_len or 0) >= 8:
        actions["normalize"] = {"strip": True, "lower": False}
    elif dtype == "string":
        actions["normalize"] = {"strip": True, "lower": False}

    # Optional: flag suspicious “id” that isn’t very unique
    if role == "id" and uniq < 0.9:
        actions["flag_low_uniqueness_for_id"] = True

    return actions


def propose_schema_for_df(slug: str, df: pd.DataFrame) -> ProposedSchema:
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

    cols_json = []
    for c in proposed.columns:
        s = df[c.name] if c.name in df.columns else pd.Series([], dtype="float64")
        m = _series_metrics(s)
        actions = _suggest_actions(c.name, c.dtype, c.role_confidence.role, m)
        cols_json.append({
            "name": c.name,
            "dtype": c.dtype,
            "role": c.role_confidence.role,
            "role_confidence": c.role_confidence.confidence,
            "metrics": m,
            "suggestions": actions,
            "hints": {
                "unit_hint": c.hints.unit_hint,
                "domain_guess": c.hints.domain_guess,
                "canonical_map_size": len(c.hints.canonical_map or {}),
            },
        })

    bootstrap = {
        "dataset": dataset_slug,
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)) if not df.empty else 0,
        "schema_confidence": proposed.schema_confidence,
        "columns": cols_json,
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
