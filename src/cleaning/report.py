from __future__ import annotations
from typing import Any, TYPE_CHECKING
import pandas as pd
from dataclasses import is_dataclass, asdict

# Avoid circular import at runtime: only import RuleHit for type checking
if TYPE_CHECKING:  # pragma: no cover
    from .engine import RuleHit  # noqa: F401


def _coerce_float(x: Any) -> float | None:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _rescore_to_dict(rescore: Any) -> dict[str, Any]:
    """
    Accept either a plain dict or a dataclass (RescoreResult) and return a serializable dict.
    Unknown shapes fall back to {}.
    """
    if isinstance(rescore, dict):
        # Coerce numeric fields to float for consistency
        out = dict(rescore)
        for k in (
            "schema_conf_before",
            "schema_conf_after",
            "avg_role_conf_before",
            "avg_role_conf_after",
        ):
            if k in out:
                out[k] = _coerce_float(out[k])
        return out
    if is_dataclass(rescore):
        d = asdict(rescore)
        for k in (
            "schema_conf_before",
            "schema_conf_after",
            "avg_role_conf_before",
            "avg_role_conf_after",
        ):
            if k in d:
                d[k] = _coerce_float(d[k])
        return d
    return {}


def _columns_added_dropped(df_before: pd.DataFrame, df_after: pd.DataFrame) -> tuple[list[str], list[str]]:
    before = list(map(str, df_before.columns))
    after = list(map(str, df_after.columns))
    added = [c for c in after if c not in before]
    dropped = [c for c in before if c not in after]
    return added, dropped


def _type_changes_map(
    metrics_before: dict[str, dict[str, Any]],
    metrics_after: dict[str, dict[str, Any]],
) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    common = set(metrics_before.keys()) & set(metrics_after.keys())
    for c in sorted(common):
        t1 = str(metrics_before[c].get("type"))
        t2 = str(metrics_after[c].get("type"))
        if t1 != t2:
            out[c] = {"before": t1, "after": t2}
    return out


def _type_changes_tuples(
    metrics_before: dict[str, dict[str, Any]],
    metrics_after: dict[str, dict[str, Any]],
) -> dict[str, tuple[str, str]]:
    out: dict[str, tuple[str, str]] = {}
    common = set(metrics_before.keys()) & set(metrics_after.keys())
    for c in sorted(common):
        t1 = str(metrics_before[c].get("type"))
        t2 = str(metrics_after[c].get("type"))
        if t1 != t2:
            out[c] = (t1, t2)
    return out


def _missing_pct_delta_map(
    metrics_before: dict[str, dict[str, Any]],
    metrics_after: dict[str, dict[str, Any]],
) -> dict[str, dict[str, float | None]]:
    out: dict[str, dict[str, float | None]] = {}
    common = set(metrics_before.keys()) & set(metrics_after.keys())
    for c in sorted(common):
        b = _coerce_float(metrics_before[c].get("missing_pct"))
        a = _coerce_float(metrics_after[c].get("missing_pct"))
        if b is None and a is None:
            continue
        out[c] = {"before": b, "after": a, "delta": (None if (a is None or b is None) else (a - b))}
    return out


def _missing_pct_delta_tuples(
    metrics_before: dict[str, dict[str, Any]],
    metrics_after: dict[str, dict[str, Any]],
) -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    common = set(metrics_before.keys()) & set(metrics_after.keys())
    for c in sorted(common):
        b = float(metrics_before[c].get("missing_pct", 0.0) or 0.0)
        a = float(metrics_after[c].get("missing_pct", 0.0) or 0.0)
        out[c] = (b, a)
    return out


def _actions_by_column(rule_hits: list["RuleHit"] | list[dict[str, Any]]) -> dict[str, list[str]]:
    """
    Accepts either RuleHit objects or plain dicts with keys: column, notes, rule_id.
    Returns {column: [action notes...]} with de-duplication.
    """
    out: dict[str, list[str]] = {}
    for h in rule_hits:
        column = getattr(h, "column", None)
        if column is None and isinstance(h, dict):
            column = h.get("column")
        if column is None:
            continue

        notes = getattr(h, "notes", None)
        if notes is None and isinstance(h, dict):
            notes = h.get("notes")

        rule_id = getattr(h, "rule_id", None)
        if rule_id is None and isinstance(h, dict):
            rule_id = h.get("rule_id")

        if notes:
            parts = [p.strip() for p in str(notes).split(",") if p.strip()]
        elif rule_id:
            parts = [p.strip() for p in str(rule_id).split(";") if p.strip()]
        else:
            parts = []

        out.setdefault(column, [])
        out[column].extend(parts)

    # De-dupe while preserving order
    for k, vals in out.items():
        seen = set()
        deduped = []
        for v in vals:
            if v not in seen:
                seen.add(v)
                deduped.append(v)
        out[k] = deduped
    return out


def build_iteration_report(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    metrics_before: dict[str, dict[str, Any]],
    metrics_after: dict[str, dict[str, Any]],
    rule_hits: list["RuleHit"] | list[dict[str, Any]],
    rescore: dict[str, Any] | Any,  # supports dict or dataclass
) -> dict[str, Any]:
    """
    Assemble a serializable dict for loop_iter_k.json and NLG narrative.
    """
    rows_before, cols_before = int(len(df_before)), int(df_before.shape[1])
    rows_after, cols_after = int(len(df_after)), int(df_after.shape[1])

    added, dropped = _columns_added_dropped(df_before, df_after)
    type_chg_map = _type_changes_map(metrics_before, metrics_after)
    type_chg_tup = _type_changes_tuples(metrics_before, metrics_after)
    miss_delta_map = _missing_pct_delta_map(metrics_before, metrics_after)
    miss_delta_tup = _missing_pct_delta_tuples(metrics_before, metrics_after)

    # Serialize hits once for reuse
    hits_serialized = [
        (h.__dict__ if hasattr(h, "__dict__") else dict(h))  # tolerate dicts from tests
        for h in rule_hits
    ]
    actions_map = _actions_by_column(rule_hits)

    # Small human-readable summary
    cols_changed_count = len(type_chg_map)
    rules_triggered_count = sum(len(v) for v in actions_map.values())
    summary = (
        f"Rows: {rows_before} → {rows_after}. "
        f"Columns: {cols_before} → {cols_after}. "
        f"Dropped: {len(dropped)}, Added: {len(added)}, Type changes: {cols_changed_count}. "
        f"Rule applications: {rules_triggered_count}."
    )

    report: dict[str, Any] = {
        "shape": {
            "before": {"rows": rows_before, "columns": cols_before},
            "after": {"rows": rows_after, "columns": cols_after},
        },
        "columns": {
            "added": added,
            "dropped": dropped,
            "type_changes": type_chg_map,
        },
        "metrics": {
            "missing_pct": miss_delta_map,
        },
        "rules": {
            "hits": hits_serialized,
            "per_column_actions": actions_map,
            "total_applications": rules_triggered_count,
        },
        "rescore": _rescore_to_dict(rescore),
        "narrative": summary,
    }

    # ---- Back-compat top-level fields expected by older engine tests ----
    report.update(
        {
            "row_count_before": rows_before,
            "row_count_after": rows_after,
            "column_count_before": cols_before,
            "column_count_after": cols_after,
            "rule_hits": hits_serialized,
            "metrics_delta": {
                "missing_pct": miss_delta_tup,
                "type_changes": type_chg_tup,
            },
        }
    )

    return report
