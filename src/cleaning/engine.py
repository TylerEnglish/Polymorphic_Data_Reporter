from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any
import pandas as pd

from ..config_model.model import RootCfg
from ..nlp.schema import ProposedSchema
from .metrics import profile_columns
from .dsl import compile_condition, eval_condition
from .registry import compile_actions_registry, parse_then, NameRef
from .rescore import rescore_after_clean
from .report import build_iteration_report

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
    rescore: dict[str, Any]
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
    from .policy import build_policy_from_config
    rules, env = build_policy_from_config(cfg)
    registry = compile_actions_registry()

    schema_roles = {c["name"]: c["role"] for c in proposed_schema.to_dict()["columns"]}

    metrics_before = profile_columns(df, schema_roles, cfg.profiling.roles.datetime_formats)
    df_after, hits, col_report, dropped = apply_rules(df, schema_roles, rules, env, registry)
    metrics_after = profile_columns(df_after, schema_roles, cfg.profiling.roles.datetime_formats)

    # Rescore using NLP before/after confidences (returns a dataclass)
    rescore_result = rescore_after_clean(
        df_after,
        proposed_schema,
        cfg.profiling.roles,
        cfg.nlp,
    )
    # Build a single canonical report
    report = build_iteration_report(
        df_before=df,
        df_after=df_after,
        metrics_before=metrics_before,
        metrics_after=metrics_after,
        rule_hits=hits,
        rescore=rescore_result,  # report handles dataclass/dict
    )

    return CleaningResult(
        clean_df=df_after,
        report=report,
        rescore=asdict(rescore_result),
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
    Apply rules in priority order per column. Pure: returns new df and logs.
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

    # Initial metrics for per-column recordkeeping
    metrics = profile_columns(
        df2,
        schema_roles,
        env.get("profiling", {}).get("roles", {}).get("datetime_formats", []),
    )

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
                # Defensive: skip exploding rule but keep engine pure
                continue

        # assign possibly-updated series
        df2[col] = s

        # optionally drop if marked; registry.drop_column returns empty series
        if s is not None and getattr(s, "size", 0) == 0 and col in df2.columns:
            dropped[col] = "drop_column"
            df2 = df2.drop(columns=[col])
            continue

        # update hit record
        after_type = _series_dtype(df2[col])
        hits.append(RuleHit(
            column=col,
            rule_id=";".join(actions_taken) if actions_taken else "noop",
            before_type=before_type,
            after_type=after_type,
            before_role=before_role,
            after_role=None,  # can be filled later if we wire roles into rescore
            notes=", ".join(actions_taken) if actions_taken else None,
        ))

        col_report[col] = {"actions": actions_taken}

    return df2, hits, col_report, dropped

# ---- helpers ----

def _resolve_params(params: dict[str, Any], env: dict[str, Any]) -> dict[str, Any]:
    """
    Deep-resolve NameRef placeholders against env. Keep literals as-is.
    Supports nested dict/list/tuple structures.
    """
    def _resolve_ref(path: str) -> Any:
        cur: Any = env
        for part in path.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                return None
        return cur

    def _walk(val: Any) -> Any:
        if isinstance(val, NameRef):
            return _resolve_ref(val.path)
        if isinstance(val, dict):
            return {k: _walk(v) for k, v in val.items()}
        if isinstance(val, (list, tuple)):
            seq = [_walk(v) for v in val]
            return seq if isinstance(val, list) else tuple(seq)
        return val

    return _walk(params) if params else {}

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
