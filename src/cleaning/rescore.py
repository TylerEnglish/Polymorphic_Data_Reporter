from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Tuple, Optional, Iterable
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
    per_column: dict[str, dict[str, float]]  # {"amount":{"before":0.7,"after":0.9}, ...}

# ---- helpers (pure) ---------------------------------------------------------

def _safe_mean(vals: Iterable[Optional[float]]) -> float:
    xs = [float(v) for v in vals if v is not None]
    return float(sum(xs) / len(xs)) if xs else 0.0

def _extract_prev_schema_info(prev_schema: ProposedSchema) -> Tuple[dict, dict[str, Optional[float]], float]:
    """
    Returns:
      - prev_cols: {name -> role} from prev_schema
      - prev_conf: {name -> confidence or None}
      - schema_conf_before: float
    """
    # Try to use a stable, dictionary representation
    prev = {}
    try:
        prev = prev_schema.to_dict()
    except Exception:
        # fall back to attribute access heuristics
        pass

    schema_conf_before = 0.0
    try:
        schema_conf_before = float(getattr(prev_schema, "schema_confidence", 0.0))
    except Exception:
        pass
    if isinstance(prev, dict):
        try:
            sc = prev.get("schema_confidence", None)
            if sc is not None:
                schema_conf_before = float(sc)
        except Exception:
            pass

    prev_cols: dict[str, str] = {}
    prev_conf: dict[str, Optional[float]] = {}

    # Common shape: {"columns":[{"name": ..., "role": ..., "confidence": ...}, ...]}
    if isinstance(prev, dict) and isinstance(prev.get("columns", None), list):
        for c in prev["columns"]:
            try:
                name = str(c.get("name"))
                role = str(c.get("role"))
                prev_cols[name] = role
                conf = c.get("confidence", None)
                prev_conf[name] = float(conf) if conf is not None else None
            except Exception:
                continue
    else:
        # Fallback: iterate attributes/fields if available
        try:
            cols = getattr(prev_schema, "columns", None) or []
            for c in cols:
                name = str(getattr(c, "name", None))
                role = str(getattr(c, "role", None))
                if name:
                    prev_cols[name] = role
                    conf = getattr(c, "confidence", None)
                    prev_conf[name] = float(conf) if conf is not None else None
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
    # Potential kwargs to pass through
    dt_formats = []
    try:
        dt_formats = getattr(getattr(profiling_cfg, "roles", None), "datetime_formats", []) or []
    except Exception:
        pass

    # Try common signatures
    candidates = [
        # (callable, args, kwargs)
        (guess_role, (name, s, nlp_cfg, dt_formats), {}),
        (guess_role, (name, s, nlp_cfg), {}),
        (guess_role, (s, nlp_cfg, dt_formats), {}),
        (guess_role, (s, nlp_cfg), {}),
        (guess_role, (s,), {}),
    ]

    for fn, args, kwargs in candidates:
        try:
            out = fn(*args, **kwargs)
            # Return-shape: (role, conf)
            if isinstance(out, tuple) and len(out) >= 2:
                role = str(out[0])
                conf = out[1]
                try:
                    conf = float(conf)
                except Exception:
                    conf = None
                return role, conf
            # Return-shape: {"role":..., "confidence":...}
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
    When the previous schema contains per-column confidences, we compare those
    to the freshly inferred confidences. If not present, we treat "before"
    confidences as 0.0 for comparison purposes.

    The overall schema_conf_after is a simple average of per-column "after"
    confidences (where available). A more nuanced aggregation can be added later.
    """
    prev_cols, prev_conf, schema_conf_before = _extract_prev_schema_info(prev_schema)

    per_col: dict[str, dict[str, float]] = {}
    after_confs: list[Optional[float]] = []
    before_confs_for_present_cols: list[Optional[float]] = []

    for name in df.columns:
        s = df[name]
        role_after, conf_after = _call_guess_role(name, s, profiling_cfg, nlp_cfg)
        after_confs.append(conf_after)

        # Previous confidence for the same column if known
        bconf = prev_conf.get(name, None)
        before_confs_for_present_cols.append(bconf)

        per_col[name] = {
            "before": float(bconf) if bconf is not None else 0.0,
            "after": float(conf_after) if conf_after is not None else 0.0,
        }

    avg_role_conf_after = _safe_mean(after_confs)
    # Compare apples-to-apples: only columns present now
    avg_role_conf_before = _safe_mean(before_confs_for_present_cols)

    # A simple overall schema confidence: average of per-column after-confs
    schema_conf_after = float(avg_role_conf_after)

    return RescoreResult(
        schema_conf_before=float(schema_conf_before),
        schema_conf_after=float(schema_conf_after),
        avg_role_conf_before=float(avg_role_conf_before),
        avg_role_conf_after=float(avg_role_conf_after),
        per_column=per_col,
    )
