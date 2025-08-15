from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Tuple, Optional, Iterable, Union
import pandas as pd

from ..nlp.schema import ProposedSchema
from ..nlp.roles import guess_role
from ..config_model.model import NLPCfg, ProfilingRolesCfg

@dataclass(frozen=True)
class RescoreResult:
    schema_conf_before: float
    schema_conf_after: float
    avg_role_conf_before: float
    avg_role_conf_after: float
    # Example:
    # "amount": {"before":0.72,"after":0.91,"role_before":"numeric","role_after":"numeric"}
    per_column: dict[str, dict[str, Union[float, str]]]

# ---- helpers (pure) ---------------------------------------------------------

def _safe_mean(vals: Iterable[Optional[float]]) -> float:
    xs = [float(v) for v in vals if v is not None]
    return float(sum(xs) / len(xs)) if xs else 0.0

def _extract_dt_formats(profiling_cfg: Any) -> list[str]:
    """
    Accept either the whole profiling cfg or the profiling.roles sub-config.
    """
    # direct roles-like object
    try:
        fmts = getattr(profiling_cfg, "datetime_formats", None)
        if isinstance(fmts, (list, tuple)):
            return list(fmts)
    except Exception:
        pass
    # parent has .roles
    try:
        roles = getattr(profiling_cfg, "roles", None)
        fmts = getattr(roles, "datetime_formats", None)
        if isinstance(fmts, (list, tuple)):
            return list(fmts)
    except Exception:
        pass
    # dict-like
    try:
        if isinstance(profiling_cfg, dict):
            if "datetime_formats" in profiling_cfg:
                fmts = profiling_cfg.get("datetime_formats") or []
                return list(fmts)
            roles = profiling_cfg.get("roles", {}) or {}
            fmts = roles.get("datetime_formats") or []
            return list(fmts)
    except Exception:
        pass
    return []

def _extract_prev_schema_info(prev_schema: ProposedSchema) -> Tuple[dict, dict[str, Optional[float]], float]:
    """
    Returns:
      - prev_cols: {name -> role} from prev_schema
      - prev_conf: {name -> confidence or None}
      - schema_conf_before: float
    Handles both dataclass form and the .to_dict() form produced by ProposedSchema.to_dict().
    """
    prev_cols: dict[str, str] = {}
    prev_conf: dict[str, Optional[float]] = {}

    # Start with schema_confidence from object if present
    schema_conf_before = 0.0
    try:
        schema_conf_before = float(getattr(prev_schema, "schema_confidence", 0.0))
    except Exception:
        schema_conf_before = 0.0

    # Prefer a stable dict representation
    prev_dict: Dict[str, Any] = {}
    try:
        prev_dict = prev_schema.to_dict()
    except Exception:
        prev_dict = {}

    # If the dict has an explicit schema_confidence, trust it
    if isinstance(prev_dict, dict):
        try:
            sc = prev_dict.get("schema_confidence", None)
            if sc is not None:
                schema_conf_before = float(sc)
        except Exception:
            pass

    # 1) Try dict form from ProposedSchema.to_dict()
    if isinstance(prev_dict, dict) and isinstance(prev_dict.get("columns", None), list):
        for c in prev_dict["columns"]:
            try:
                name = str(c.get("name"))
                role = str(c.get("role"))
                # Accept either 'role_confidence' (our schema.toml shape) or 'confidence'
                conf_val = c.get("role_confidence", None)
                if conf_val is None and ("confidence" in c):
                    conf_val = c.get("confidence")
                conf = float(conf_val) if conf_val is not None else None
                prev_cols[name] = role
                prev_conf[name] = conf
            except Exception:
                continue
        return prev_cols, prev_conf, float(schema_conf_before)

    # 2) Fallback: introspect dataclass-like fields
    try:
        cols = getattr(prev_schema, "columns", None) or []
        for c in cols:
            name = str(getattr(c, "name", None))
            # ColumnSchema has role_confidence: RoleConfidence(role, confidence)
            rc = getattr(c, "role_confidence", None)
            role = None
            conf = None
            if rc is not None:
                try:
                    role = getattr(rc, "role", None)
                    conf = getattr(rc, "confidence", None)
                except Exception:
                    role = None
                    conf = None
            # sometimes libraries put role/confidence directly (defensive)
            role = role or getattr(c, "role", None)
            conf = conf if conf is not None else getattr(c, "confidence", None)

            if name:
                if role is not None:
                    prev_cols[name] = str(role)
                if conf is not None:
                    try:
                        prev_conf[name] = float(conf)
                    except Exception:
                        prev_conf[name] = None
    except Exception:
        pass

    return prev_cols, prev_conf, float(schema_conf_before)

def _call_guess_role(
    name: str,
    s: pd.Series,
    profiling_cfg: ProfilingRolesCfg,
    nlp_cfg: NLPCfg,
) -> Tuple[str, Optional[float]]:
    """
    Try several signatures/return-shapes for guess_role, returning (role, confidence|None).
    """
    dt_formats = _extract_dt_formats(profiling_cfg)

    candidates = [
        (guess_role, (name, s, nlp_cfg, dt_formats), {}),
        (guess_role, (name, s, nlp_cfg), {}),
        (guess_role, (s, nlp_cfg, dt_formats), {}),
        (guess_role, (s, nlp_cfg), {}),
        (guess_role, (s,), {}),
    ]

    for fn, args, kwargs in candidates:
        try:
            out = fn(*args, **kwargs)
            # (role, conf)
            if isinstance(out, tuple) and len(out) >= 2:
                role = str(out[0])
                conf = out[1]
                try:
                    conf = float(conf)
                except Exception:
                    conf = None
                return role, conf
            # {"role":..., "confidence":...}
            if isinstance(out, dict) and "role" in out:
                role = str(out.get("role"))
                conf = out.get("confidence", None)
                try:
                    conf = float(conf) if conf is not None else None
                except Exception:
                    conf = None
                return role, conf
            # Just a role string?
            if isinstance(out, str):
                return out, None
        except Exception:
            continue

    # Fallback when guess_role is unavailable or unexpected
    return "text", None

# ---- public API -------------------------------------------------------------

def rescore_after_clean(
    df: pd.DataFrame,
    prev_schema: ProposedSchema,
    profiling_cfg: ProfilingRolesCfg,
    nlp_cfg: NLPCfg,
) -> RescoreResult:
    """
    Re-run role inference on the *cleaned* df and compute new confidences.
    'Before' confidences come from the previous schema (if present). We also
    carry role_before/role_after so the caller can reason about deltas.
    """
    prev_cols, prev_conf, schema_conf_before = _extract_prev_schema_info(prev_schema)

    per_col: dict[str, dict[str, Union[float, str]]] = {}
    after_confs: list[Optional[float]] = []
    before_confs_for_present_cols: list[Optional[float]] = []

    for name in df.columns:
        s = df[name]
        role_after, conf_after = _call_guess_role(name, s, profiling_cfg, nlp_cfg)
        after_confs.append(conf_after)

        bconf = prev_conf.get(name, None)
        before_confs_for_present_cols.append(bconf)
        role_before = prev_cols.get(name, "")

        per_col[name] = {
            "before": float(bconf) if bconf is not None else 0.0,
            "after": float(conf_after) if conf_after is not None else 0.0,
            "role_before": str(role_before),
            "role_after": str(role_after),
        }

    avg_role_conf_after = _safe_mean(after_confs)
    avg_role_conf_before = _safe_mean(before_confs_for_present_cols)

    # Overall "schema" confidence: average of per-column after confidences
    schema_conf_after = float(avg_role_conf_after)

    return RescoreResult(
        schema_conf_before=float(schema_conf_before),
        schema_conf_after=float(schema_conf_after),
        avg_role_conf_before=float(avg_role_conf_before),
        avg_role_conf_after=float(avg_role_conf_after),
        per_column=per_col,
    )
