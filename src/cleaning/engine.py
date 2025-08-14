from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import pandas as pd

from ..config_model.model import RootCfg
from ..nlp.schema import ProposedSchema
from ..nlp.roles import guess_role  # for rescore
from .metrics import profile_columns
from .dsl import compile_condition, eval_condition
from .registry import compile_actions_registry, parse_then

# ---- Data classes ----

@dataclass(frozen=True)
class RuleSpec:
    id: str
    priority: int
    when: str
    then: str

@dataclass(frozen=True)
class RuleHit:
    column: str
    rule_id: str
    before_type: str
    after_type: str
    before_role: str
    after_role: str | None
    notes: str | None = None

@dataclass(frozen=True)
class CleaningResult:
    clean_df: pd.DataFrame
    report: dict[str, Any]
    rescore: dict[str, Any]  # placeholder structure; wire to dedicated RescoreResult later
    column_report: dict[str, dict[str, Any]]
    dropped: dict[str, str]

# ---- Public API ----

def run_clean_pass(
    df: pd.DataFrame,
    proposed_schema: ProposedSchema,
    cfg: RootCfg,
) -> CleaningResult:
    """
    One pure cleaning pass:
      - profile
      - apply rules (priority order)
      - re-profile
      - rescore (NLP)
      - build iteration report
    """
    # TODO: wire policy/env from build_policy_from_config (caller can pass or import here)
    from .policy import build_policy_from_config
    rules, env = build_policy_from_config(cfg)
    registry = compile_actions_registry()

    schema_roles = {c["name"]: c["role"] for c in proposed_schema.to_dict()["columns"]}
    metrics_before = profile_columns(df, schema_roles, cfg.profiling.roles.datetime_formats)

    df_after, hits, col_report, dropped = apply_rules(df, schema_roles, rules, env, registry)

    metrics_after = profile_columns(df_after, schema_roles, cfg.profiling.roles.datetime_formats)

    # TODO: implement rescore (dedicated module). For now, return placeholders.
    rescore = {
        "schema_conf_before": float(proposed_schema.schema_confidence),
        "schema_conf_after": float(proposed_schema.schema_confidence),  # TODO
        "avg_role_conf_before": 0.0,  # TODO
        "avg_role_conf_after": 0.0,   # TODO
        "per_column": {},             # TODO
    }

    # TODO: implement detailed iteration report (separate module)
    report = {
        "row_count_before": int(len(df)),
        "row_count_after": int(len(df_after)),
        "column_count_before": int(df.shape[1]),
        "column_count_after": int(df_after.shape[1]),
        "rule_hits": [hit.__dict__ for hit in hits],
        "dropped_columns": dropped,
        "metrics_delta": {
            "missing_pct": _collect_metric_delta(metrics_before, metrics_after, "missing_pct"),
            "type_changes": _collect_type_changes(metrics_before, metrics_after),
        },
    }

    return CleaningResult(
        clean_df=df_after,
        report=report,
        rescore=rescore,
        column_report=col_report,
        dropped=dropped,
    )

def apply_rules(
    df: pd.DataFrame,
    schema_roles: dict[str, str],
    rules: list[RuleSpec],
    env: dict[str, Any],
    registry: dict[str, Any],
) -> tuple[pd.DataFrame, list[RuleHit], dict[str, dict[str, Any]], dict[str, str]]:
    """
    Apply rules in priority order per column. Two-phase metrics recompute is optional;
    keep single pass initially for determinism. Pure: returns new df and logs.
    """
    # Sort rules: higher priority first; stable
    rules_sorted = sorted(rules, key=lambda r: (-r.priority, r.id))

    # Pre-compile conditions & actions
    compiled: list[tuple[RuleSpec, Any, Any]] = []
    for r in rules_sorted:
        cond = compile_condition(r.when)
        action, params = parse_then(r.then, registry)
        compiled.append((r, cond, (action, params)))

    df2 = df.copy(deep=True)
    hits: list[RuleHit] = []
    col_report: dict[str, dict[str, Any]] = {}
    dropped: dict[str, str] = {}

    # initial metrics for per-column before/after recordkeeping
    from .metrics import profile_columns
    metrics = profile_columns(df2, schema_roles, env.get("profiling", {}).get("roles", {}).get("datetime_formats", []))

    for col in list(df2.columns):
        s = df2[col]
        before_type = metrics[col]["type"]
        before_role = schema_roles.get(col, "text")
        actions_taken: list[str] = []

        for r, cond, (act, params) in compiled:
            ctx = {
                **metrics[col],
                "name": col,
                "env": env,
                "params": _resolve_params(params, env),
                "cleaning": env.get("cleaning", {}),
                "profiling": env.get("profiling", {}),
                "schema_role": before_role,
            }
            try:
                if eval_condition(cond, ctx):
                    res = act(s, ctx)
                    if isinstance(res, tuple):
                        s, note = res
                    else:
                        s, note = res, None
                    actions_taken.append(note or r.id)
            except Exception:
                # Be defensive: if a rule explodes, skip it but keep engine pure
                continue

        # assign possibly-updated series
        df2[col] = s

        # optionally drop if marked; drop is represented as empty series by registry drop_column
        if s is not None and s.size == 0 and col in df2.columns:
            dropped[col] = "drop_column"
            df2 = df2.drop(columns=[col])
            continue

        # update simple hit record
        after_type = _series_dtype(df2[col])
        hits.append(RuleHit(
            column=col,
            rule_id=";".join(actions_taken) if actions_taken else "noop",
            before_type=before_type,
            after_type=after_type,
            before_role=before_role,
            after_role=None,  # filled in by rescore step if needed
            notes=", ".join(actions_taken) if actions_taken else None,
        ))

        col_report[col] = {
            "actions": actions_taken,
        }

    return df2, hits, col_report, dropped

# ---- small helpers ----

def _resolve_params(params: dict[str, Any], env: dict[str, Any]) -> dict[str, Any]:
    """
    Replace symbolic placeholders (like 'datetime_formats') with values from env.
    Keeps literals as-is.
    """
    # TODO: implement name resolution (strings that match keys in env become those values)
    return params

def _series_dtype(s: pd.Series) -> str:
    from ..nlp.roles import _dtype_str
    return _dtype_str(s)

def _collect_metric_delta(m1: dict[str, dict[str, Any]], m2: dict[str, dict[str, Any]], key: str) -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    for c in m1.keys():
        if c in m2:
            out[c] = (float(m1[c].get(key, 0.0)), float(m2[c].get(key, 0.0)))
    return out

def _collect_type_changes(m1: dict[str, dict[str, Any]], m2: dict[str, dict[str, Any]]) -> dict[str, tuple[str, str]]:
    out: dict[str, tuple[str, str]] = {}
    for c in m1.keys():
        if c in m2:
            t1 = str(m1[c].get("type"))
            t2 = str(m2[c].get("type"))
            if t1 != t2:
                out[c] = (t1, t2)
    return out
