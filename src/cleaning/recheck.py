from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import pandas as pd

from src.config_model.model import RootCfg
from src.io.writers import write_any
from src.io.storage import build_storage_from_config
from src.nlp.schema import ProposedSchema, ColumnSchema, ColumnHints, RoleConfidence
from src.cleaning.engine import run_clean_pass
from src.cleaning.rescore import rescore_after_clean
from src.utils.log import get_logger


# ------------------------ helpers: schema I/O & cfg filter ------------------------

def _load_frozen_schema(slug: str, project_root: Path) -> ProposedSchema:
    """
    Read a frozen schema from schemas/* with flexible filename support:
      - <slug>.schema.toml
      - <slug>.schemas.toml
      - <slug>.parquet.schema.toml
      - also falls back to globbing for <slug>*.(schema|schemas).toml
    """
    try:
        import tomllib  # py>=3.11
    except Exception:  # pragma: no cover
        import tomli as tomllib  # type: ignore

    schemas_dir = project_root / "schemas"
    base = slug
    stem = Path(slug).stem  # e.g. "x1" for "x1.parquet"

    # Try a prioritized list of exact names
    candidates = [
        f"{base}.schema.toml",
        f"{base}.schemas.toml",
        f"{base}.parquet.schema.toml" if not base.endswith(".parquet") else f"{base}.schema.toml",
        f"{stem}.schema.toml",
        f"{stem}.schemas.toml",
        f"{stem}.parquet.schema.toml",
    ]
    schema_path: Optional[Path] = None
    for name in candidates:
        p = schemas_dir / name
        if p.exists():
            schema_path = p
            break
    if schema_path is None:
        # Glob fallbacks
        globs = [
            f"{base}*.schema.toml",
            f"{base}*.schemas.toml",
            f"{stem}*.schema.toml",
            f"{stem}*.schemas.toml",
        ]
        for pattern in globs:
            found = list(schemas_dir.glob(pattern))
            if found:
                schema_path = found[0]
                break
    if schema_path is None:
        # No frozen schema found — return an empty-but-valid ProposedSchema
        return ProposedSchema(dataset_slug=slug, columns=[], schema_confidence=1.0)

    raw = tomllib.loads(schema_path.read_text(encoding="utf-8"))

    cols: List[ColumnSchema] = []
    raw_cols = raw.get("columns", [])
    if not isinstance(raw_cols, list):
        raw_cols = []  # be forgiving

    for c in raw_cols:
        if not isinstance(c, dict):
            continue
        rc = c.get("role_confidence", {}) or {}
        hints = c.get("hints", {}) or {}
        # role_confidence can be malformed (float/string); guard
        role_val = ""
        conf_val = 0.0
        try:
            if isinstance(rc, dict):
                role_val = str(rc.get("role", "") or "")
                conf_val = float(rc.get("confidence", 0.0) or 0.0)
        except Exception:
            role_val, conf_val = "", 0.0

        try:
            hints_obj = ColumnHints(**hints) if isinstance(hints, dict) else ColumnHints()
        except Exception:
            hints_obj = ColumnHints()

        cols.append(
            ColumnSchema(
                name=str(c.get("name", "")),
                dtype=str(c.get("dtype", "object")),
                role_confidence=RoleConfidence(role=role_val, confidence=conf_val),
                hints=hints_obj,
            )
        )

    return ProposedSchema(
        dataset_slug=str(raw.get("dataset_slug", slug)),
        columns=cols,
        schema_confidence=float(raw.get("schema_confidence", 1.0) or 1.0),
    )

def _non_destructive_cfg(cfg: RootCfg) -> RootCfg:
    """
    Clone config and remove any rules that drop columns; ensure outliers don't drop rows.
    """
    tmp = cfg.model_copy(deep=True)  # pydantic v2
    # strip any rule whose "then" contains drop_column()
    tmp.cleaning.rules = [r for r in tmp.cleaning.rules if "drop_column" not in r.then.replace(" ", "").lower()]
    # outliers: if configured to 'drop', force to 'flag' for this pass
    if getattr(tmp.cleaning.outliers, "handle", "flag") == "drop":
        tmp.cleaning.outliers.handle = "flag"
    return tmp


# ------------------------------- public API --------------------------------

@dataclass
class RecheckResult:
    slug: str
    rechecked_path: str
    schema_report_path: str
    rows: int
    cols: int
    schema_conf_after: float
    avg_role_conf_after: float


def recheck_silver(
    cfg: RootCfg,
    slug: str,
    *,
    silver_path: Optional[str] = None,
    write_dataset: bool = True,
) -> Tuple[pd.DataFrame, RecheckResult]:
    """
    Non-destructive re-check for silver:
      - respects frozen schema roles
      - improves numeric/categorical/datetime coercions
      - re-applies outlier/unit policy
      - never drops columns

    Returns (clean_df, RecheckResult). Also writes dataset.rechecked.parquet and schema_report.parquet.
    """
    log = get_logger("silver_recheck", cfg.logging.level, cfg.logging.structured_json)
    storage = build_storage_from_config(cfg)
    root = Path.cwd()

    # Paths
    silver_dir = root / "data" / "silver" / slug
    silver_path = silver_path or str(silver_dir / "dataset.parquet")
    rechecked_path = silver_dir / "dataset.rechecked.parquet"
    schema_report_path = silver_dir / "schema_report.parquet"

    # Load data + frozen schema
    df = pd.read_parquet(silver_path)
    proposed = _load_frozen_schema(slug, root)

    # Filter config to be non-destructive
    safe_cfg = _non_destructive_cfg(cfg)

    # One clean pass using frozen roles (no role guessing, no dropping)
    result = run_clean_pass(df, proposed, safe_cfg, extra_rules=None)
    clean_df = result.clean_df

    # Rescore (metrics only) for reporting
    res = rescore_after_clean(clean_df, proposed, cfg.profiling.roles, cfg.nlp)
    schema_conf_after = float(getattr(res, "schema_conf_after", 1.0))
    avg_role_conf_after = float(getattr(res, "avg_role_conf_after", 1.0))

    # Persist artifacts
    if write_dataset:
        write_any(storage, clean_df, str(rechecked_path), fmt="parquet")
    # Always refresh schema_report to reflect the post-recheck snapshot
    rep = pd.DataFrame(
        {
            "name": list(clean_df.columns),
            "dtype": [str(t) for t in clean_df.dtypes],
            "missing_pct": [float(clean_df[c].isna().mean()) for c in clean_df.columns],
        }
    )
    write_any(storage, rep, str(schema_report_path), fmt="parquet")

    info = RecheckResult(
        slug=slug,
        rechecked_path=str(rechecked_path),
        schema_report_path=str(schema_report_path),
        rows=len(clean_df),
        cols=clean_df.shape[1],
        schema_conf_after=schema_conf_after,
        avg_role_conf_after=avg_role_conf_after,
    )
    log.info(
        "silver_recheck_done",
        extra={
            "slug": slug,
            "rows": info.rows,
            "cols": info.cols,
            "rechecked_path": info.rechecked_path,
            "schema_report_path": info.schema_report_path,
            "schema_conf_after": schema_conf_after,
            "avg_role_conf_after": avg_role_conf_after,
        },
    )
    return clean_df, info
