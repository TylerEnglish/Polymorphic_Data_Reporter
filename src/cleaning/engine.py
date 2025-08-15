from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import Any
import pandas as pd

from ..config_model.model import RootCfg
from ..nlp.schema import ProposedSchema
from .metrics import profile_columns
from .dsl import compile_condition, eval_condition
from .registry import compile_actions_registry, parse_then, NameRef
from .rescore import rescore_after_clean
from .report import build_iteration_report
from ..nlp.suggestions import plan_followups

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
    suggestions: dict[str, list[str]] = field(default_factory=dict)
    meets_thresholds: bool = False
    thresholds: dict[str, float] = field(default_factory=dict)

# ---- helpers ----

def _dtype_tag(s: pd.Series) -> str:
    from ..nlp.roles import _dtype_str
    return _dtype_str(s)


def _quick_facts(series: pd.Series, *, name: str, schema_role: str, df_index) -> dict:
    n = len(series) or 1
    t = _dtype_tag(series)
    try:
        nunique = int(series.nunique(dropna=True))
    except TypeError:
        nunique = int(series.astype(str).nunique(dropna=True))
    avg_len = None
    if t == "string":
        vals = series.dropna().astype(str)
        avg_len = float(vals.map(len).mean()) if not vals.empty else 0.0
    missing_pct = float(series.isna().mean())
    return {
        "name": name,
        "type": t,
        "role": schema_role,                  # keep schema role stable
        "missing_pct": missing_pct,
        "non_null_ratio": 1.0 - missing_pct,
        "nunique": nunique,
        "unique_ratio": float(nunique) / float(n),
        "cardinality": nunique,
        "avg_len": avg_len,
        "has_time_index": pd.api.types.is_datetime64_any_dtype(df_index),
        # extras used in some rules; safe defaults if not needed
        "mean": None, "std": None, "iqr": None,
        "bool_token_ratio": 0.0,
        "datetime_parse_ratio": 0.0,
        "is_monotonic_increasing": False,
    }

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

    try:
        min_schema = float(getattr(cfg.nlp, "min_schema_confidence", 0.90))
    except Exception:
        min_schema = 0.90
    try:
        min_role = float(getattr(cfg.nlp, "min_role_confidence", 0.90))
    except Exception:
        min_role = 0.90

    meets = (
        float(rescore_result.schema_conf_after) >= min_schema
        and float(rescore_result.avg_role_conf_after) >= min_role
    )

    suggestions: dict[str, list[str]] = {}
    if not meets:
        # plan_followups uses AFTER frame + proposed schema + the detailed rescore map
        suggestions = plan_followups(df_after, proposed_schema, asdict(rescore_result), cfg)

    # annotate the report so downstream writers/NLG can render targets & actions
    report["targets"] = {
        "min_schema_confidence": min_schema,
        "min_role_confidence": min_role,
        "schema_conf_after": float(rescore_result.schema_conf_after),
        "avg_role_conf_after": float(rescore_result.avg_role_conf_after),
        "met": bool(meets),
    }
    if suggestions:
        report["suggestions"] = suggestions

    return CleaningResult(
        clean_df=df_after,
        report=report,
        rescore=asdict(rescore_result),
        column_report=col_report,
        dropped=dropped,
        suggestions=suggestions,
        meets_thresholds=meets,
        thresholds={"min_schema_confidence": min_schema, "min_role_confidence": min_role},
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
    compiled = []
    for r in rules_sorted:
        cond = compile_condition(r.when)
        action, params = parse_then(r.then, registry)
        compiled.append((r, cond, (action, params)))

    df2 = df.copy(deep=True)
    hits, col_report, dropped = [], {}, {}

    # Initial metrics for per-column recordkeeping
    initial = profile_columns(df2, schema_roles, env.get("profiling", {}).get("roles", {}).get("datetime_formats", []))

    for col in list(df2.columns):
        s = df2[col]
        before_type = initial[col]["type"]
        before_role = schema_roles.get(col, "text")
        actions_taken = []

        for r, cond, (act, params) in compiled:
            # ⬇️ recompute facts for the *current* series before each rule
            facts = _quick_facts(s, name=col, schema_role=before_role, df_index=df2.index)
            ctx = {
                **facts,
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
                continue

        # assign possibly-updated series
        df2[col] = s

        # optionally drop if marked; registry.drop_column returns empty series
        if s is not None and getattr(s, "size", 0) == 0 and col in df2.columns:
            dropped[col] = "drop_column"
            df2 = df2.drop(columns=[col])
            continue

        # update hit record
        hits.append(RuleHit(
            column=col,
            rule_id=";".join(actions_taken) if actions_taken else "noop",
            before_type=before_type,
            after_type=_series_dtype(df2[col]),
            before_role=before_role,
            after_role=None,
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
