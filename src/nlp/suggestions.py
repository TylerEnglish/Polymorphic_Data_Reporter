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

def _bool_token(v: Any) -> str:
    return "true" if bool(v) else "false"

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
    cat_max_cfg = _get(cfg, "cleaning.columns.cat_cardinality_max", 200)
    try:
        cat_max = int(cat_max_cfg if cat_max_cfg is not None else 200)
    except Exception:
        cat_max = 200

    uniq_id_ratio = float(_get(cfg, "nlp.unique_id_ratio", 0.95))
    date_parse_min = float(_get(cfg, "nlp.date_parse_min_ratio", 0.60))
    bool_min = float(_get(cfg, "nlp.bool_token_min_ratio", 0.57))
    text_min_avg_len = float(_get(cfg, "nlp.text_min_avg_len", 8.0))

    # impute defaults (use safe literals if config missing)
    num_default = _get(cfg, "cleaning.impute.numeric_default", "median") or "median"
    cat_default = _get(cfg, "cleaning.impute.categorical_default", "Other") or "Other"
    text_default = _get(cfg, "cleaning.impute.text_default", "")  # empty string is fine

    strip_text = bool(_get(cfg, "cleaning.normalize.strip_text", True))
    lower_text = bool(_get(cfg, "cleaning.normalize.lowercase_text", False))

    # outlier policy env knobs exist in policy.env; using NameRefs in suggestion is safe
    # (detect, zscore_threshold, iqr_multiplier, winsor_limits)

    # schema role map from proposed
    p = proposed.to_dict()
    schema_roles = {c["name"]: c["role"] for c in p.get("columns", [])}

    # quick per-column metrics on the AFTER frame
    metrics = profile_columns(df_after, schema_roles, dt_formats)

    out: Dict[str, List[str]] = {}
    per_col = dict(rescore.get("per_column", {}))

    def _norm_role(r: str) -> str:
        r = (r or "").lower()
        if r in {"bool", "boolean"}: return "boolean"
        if r in {"int","float","number","numeric","double","decimal"}: return "numeric"
        if r in {"datetime","date","time"}: return "time"
        if r in {"category","categorical"}: return "categorical"
        if r in {"text","string"}: return "text"
        return r

    for col in df_after.columns:
        pc = per_col.get(col, {})
        conf_after = float(pc.get("after", 0.0) or 0.0)
        if conf_after >= min_role:
            continue  # this column is fine

        m = metrics.get(col, {})
        dtype = str(m.get("type", "string"))
        role_after = str(pc.get("role_after", schema_roles.get(col, "text")))
        role_norm = _norm_role(role_after)
        miss = float(m.get("missing_pct", 0.0) or 0.0)
        nunique = int(m.get("nunique", 0) or 0)
        uniq_ratio = float(m.get("unique_ratio", 0.0) or 0.0)
        avg_len = m.get("avg_len", None)
        vmin = m.get("min", None)
        vmax = m.get("max", None)

        name_l = str(col).lower()
        s = df_after[col]
        recs: List[str] = []

        # 0) extreme sparsity (and not always-keep)
        if miss >= drop_missing_pct and col not in always_keep and role_norm not in {"boolean","id","time"}:
            recs.append("drop_column()")

        # 1) time-like columns
        if role_norm == "time":
            dt_ratio = float(m.get("datetime_parse_ratio", 0.0) or 0.0)
            # string dates not parsing well -> try configured formats
            if dtype == "string" and dt_ratio < date_parse_min:
                recs.append(f"parse_datetime({dt_formats!r})")
            # numeric epoch by name cue
            if dtype in {"int", "float"} or any(k in name_l for k in ("time","date","timestamp","ts")):
                recs.append("parse_epoch()")
            if miss > 0:
                recs.append('impute_dt("ffill")')

        # 2) numeric values hiding in strings
        if dtype == "string":
            if _num_ratio(s) >= 0.80 and role_norm in {"numeric", "id", "text", "categorical"}:
                recs.append("coerce_numeric()")
                # percent hint by name
                if ("percent" in name_l) or ("pct" in name_l):
                    recs.append("standardize_units('percent')")
                # IDs that aren't unique enough → stabilize as categorical
                if role_after == "id" and uniq_ratio < uniq_id_ratio:
                    recs.append(f"cast_category({cat_max})")
            # 3) boolean-looking strings
            if _bool_token_ratio(s) >= bool_min and role_norm in {"boolean","categorical","text"}:
                # If registry exposes coerce_bool(), use it; else cast_category keeps values stable.
                recs.append("coerce_bool()")
                # Fallback/text hygiene still helpful:
                recs.append("normalize_null_tokens(null_tokens=cleaning.normalize_null_tokens.null_tokens, case_insensitive=true)")
                recs.append(f"cast_category({cat_max})")

        # 3.5) percent normalization for already-numeric columns
        if role_norm == "numeric" and (vmin is not None and vmax is not None):
            try:
                vmin_f = float(vmin); vmax_f = float(vmax)
                if (0.0 <= vmin_f) and (vmax_f <= 100.0) and (vmax_f > 1.0):
                    recs.append("standardize_units('percent')")
            except Exception:
                pass
        elif role_norm == "numeric" and (("percent" in name_l) or ("pct" in name_l)):
            recs.append("standardize_units('percent')")

        # 4) categorical enforcement / cleanup
        if role_norm == "categorical":
            if nunique <= cat_max:
                recs.append(f"cast_category({cat_max})")
            # Basic text hygiene to reduce accidental cardinality
            recs.append(f"text_normalize(strip={_bool_token(strip_text)}, lower={_bool_token(lower_text)})")
            recs.append("normalize_null_tokens()")
            # Optional rare-category consolidation (if you added the action)
            if uniq_ratio > 0 and uniq_ratio <= 0.50 and nunique > cat_max:
                recs.append("rare_cats(0.01, 'Other')")

        # 5) numeric outliers → winsorize to stabilize stats
        if role_norm == "numeric":
            std = m.get("std", None); iqr = m.get("iqr", None)
            try:
                if std is not None and float(std) > 0:
                    recs.append("outliers(detect, zscore_threshold, iqr_multiplier, 'winsorize', winsor_limits)")
            except Exception:
                pass

        # 6) missing value handling by role
        if miss > 0:
            if role_norm in {"numeric","id"}:
                recs.append(f'impute("{num_default}")')
            elif role_norm == "time":
                recs.append('impute_dt("ffill")')
            elif role_norm == "categorical":
                recs.append(f'impute_value("{cat_default}")')
            elif role_norm == "text" and (avg_len or 0) >= text_min_avg_len:
                recs.append(f'impute_value("{text_default}")')
                recs.append(f"text_normalize(strip={_bool_token(strip_text)}, lower={_bool_token(lower_text)})")

        # de-dupe while preserving order
        if recs:
            seen = set()
            uniq_recs = []
            for r in recs:
                if r not in seen:
                    uniq_recs.append(r); seen.add(r)
            out[col] = uniq_recs

    return out
