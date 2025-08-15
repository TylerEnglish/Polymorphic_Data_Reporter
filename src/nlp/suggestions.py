from __future__ import annotations
from typing import Dict, List, Any
import pandas as pd

from ..config_model.model import RootCfg
from ..nlp.schema import ProposedSchema
from ..cleaning.metrics import profile_columns

def _to_list(x) -> list:
    try:
        return list(x) if isinstance(x, (list, tuple)) else []
    except Exception:
        return []

def _get(cfg: Any, path: str, default=None):
    cur = cfg
    for part in path.split("."):
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(part, None)
        else:
            cur = getattr(cur, part, None)
    return default if cur is None else cur

def _num_ratio(s: pd.Series) -> float:
    try:
        x = pd.to_numeric(s.astype(str).str.replace(",", ""), errors="coerce")
        return float(x.notna().mean())
    except Exception:
        return 0.0

def _bool_token_ratio(s: pd.Series) -> float:
    try:
        vals = s.dropna().astype(str).str.strip().str.lower()
        if vals.empty: 
            return 0.0
        tokens = {"true","false","t","f","y","n","yes","no","1","0"}
        return float(vals.isin(tokens).mean())
    except Exception:
        return 0.0

def plan_followups(
    df_after: pd.DataFrame,
    proposed: ProposedSchema,
    rescore: dict,         # dict as provided by engine (asdict)
    cfg: RootCfg,
) -> Dict[str, List[str]]:
    """
    Returns: { column -> list of 'then(...)' strings to try next }
    Only suggests for columns whose role-confidence is below cfg.nlp.min_role_confidence.
    """
    # thresholds & knobs from config (robust access)
    min_role = float(_get(cfg, "nlp.min_role_confidence", 0.90))
    dt_formats = _to_list(_get(cfg, "profiling.roles.datetime_formats", []))

    always_keep = set(_get(cfg, "cleaning.columns.always_keep", []) or [])
    drop_missing_pct = float(_get(cfg, "cleaning.columns.drop_missing_pct", 0.90))
    cat_max = int(_get(cfg, "cleaning.columns.cat_cardinality_max", 200))
    uniq_id_ratio = float(_get(cfg, "nlp.unique_id_ratio", 0.95))
    date_parse_min = float(_get(cfg, "nlp.date_parse_min_ratio", 0.60))
    bool_min = float(_get(cfg, "nlp.bool_token_min_ratio", 0.57))
    text_min_avg_len = float(_get(cfg, "nlp.text_min_avg_len", 8.0))

    # schema role map from proposed
    p = proposed.to_dict()
    schema_roles = {c["name"]: c["role"] for c in p.get("columns", [])}

    # quick per-column metrics on the AFTER frame
    metrics = profile_columns(df_after, schema_roles, dt_formats)

    out: Dict[str, List[str]] = {}
    per_col = dict(rescore.get("per_column", {}))

    for col in df_after.columns:
        pc = per_col.get(col, {})
        conf_after = float(pc.get("after", 0.0) or 0.0)
        if conf_after >= min_role:
            continue  # this column is fine

        m = metrics.get(col, {})
        dtype = str(m.get("type", "string"))
        role_after = str(pc.get("role_after", schema_roles.get(col, "text")))
        miss = float(m.get("missing_pct", 0.0) or 0.0)
        nunique = int(m.get("nunique", 0) or 0)
        uniq_ratio = float(m.get("unique_ratio", 0.0) or 0.0)
        avg_len = m.get("avg_len", None)

        s = df_after[col]
        recs: List[str] = []

        # 0) extreme sparsity (and not always-keep)
        if miss >= drop_missing_pct and col not in always_keep:
            recs.append("drop_column()")

        # 1) time-like but not parsed well
        if role_after == "time":
            dt_ratio = float(m.get("datetime_parse_ratio", 0.0) or 0.0)
            if dtype == "string" and dt_ratio < date_parse_min:
                recs.append(f'parse_datetime({dt_formats!r})')
            if miss > 0:
                recs.append('impute_dt("ffill")')

        # 2) numeric hiding in strings
        if dtype == "string":
            if _num_ratio(s) >= 0.80 and role_after in {"numeric", "id", "text", "categorical"}:
                recs.append("coerce_numeric()")
                if role_after == "id" and uniq_ratio < uniq_id_ratio:
                    # ID doesn’t look unique enough; treat as categorical to stabilize downstream
                    recs.append("cast_category()")
            # 3) boolean-looking strings
            if _bool_token_ratio(s) >= bool_min and role_after in {"bool","categorical","text"}:
                recs.append("normalize_null_tokens(null_tokens=cleaning.normalize_null_tokens.null_tokens, case_insensitive=true)")
                recs.append("cast_category()")

        # 4) categorical enforcement
        if role_after == "categorical" and nunique <= cat_max and dtype == "string":
            recs.append("cast_category()")

        # 5) missing value handling by role
        if miss > 0:
            if role_after in {"numeric","id"}:
                recs.append('impute(cleaning.impute.numeric_default)')
            elif role_after == "time":
                recs.append('impute_dt("ffill")')
            elif role_after == "categorical":
                recs.append('impute_value(cleaning.impute.categorical_default)')
            elif role_after == "text" and (avg_len or 0) >= text_min_avg_len:
                recs.append('impute_value(cleaning.impute.text_default)')
                recs.append('text_normalize(strip=cleaning.normalize.strip_text, lower=cleaning.normalize.lowercase_text)')

        # de-dupe while preserving order
        if recs:
            seen = set()
            uniq_recs = []
            for r in recs:
                if r not in seen:
                    uniq_recs.append(r); seen.add(r)
            out[col] = uniq_recs

    return out
