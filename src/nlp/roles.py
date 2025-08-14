from __future__ import annotations
import re
from typing import Optional, Tuple
import pandas as pd
import numpy as np

# ---- Load role-scoring config (safe fallback) ----
class _RoleCfgFallback:
    # weights
    name_weight: float = 0.45
    value_weight: float = 0.55
    # thresholds
    bool_token_min_ratio: float = 0.75
    date_parse_min_ratio: float = 0.60
    unique_id_ratio: float = 0.95
    categorical_max_unique_ratio: float = 0.02
    text_min_avg_len: float = 8.0
    min_non_null_ratio: float = 0.10
    # bonuses/penalties
    bonus_id_name: float = 0.10
    penalize_bool_for_many_tokens: float = 0.05

def _load_role_cfg():
    try:
        from ..config_model.model import RootCfg
        return RootCfg.load().nlp.role_scoring
    except Exception:
        return _RoleCfgFallback()

_ROLECFG = _load_role_cfg()

# ---- Patterns ----
ROLE_PATTERNS = [
    ("time", re.compile(r"(date|time|timestamp|dt|ts)\b", re.I)),
    ("id",   re.compile(r"(id|uuid|guid|key)\b", re.I)),
    ("bool", re.compile(r"^(is_|has_|flag_)", re.I)),
    ("geo",  re.compile(r"(lat|lng|lon|long|latitude|longitude)\b", re.I)),
]

NUMERIC_UNITS = [
    (re.compile(r"^\s*\$"), "currency"),
    (re.compile(r"\b(percent|pct|%)\b", re.I), "percent"),
    (re.compile(r"\b(k|thousand)\b", re.I), "magnitude_k"),
    (re.compile(r"\b(million|mm)\b", re.I), "magnitude_m"),
]

# Common bool-ish tokens for object/string columns
_BOOL_TOKENS = {"true","false","t","f","y","n","yes","no","1","0"}

def _non_null_ratio(s: pd.Series) -> float:
    return float(s.notna().mean()) if len(s) else 0.0

def _safe_nunique(s: pd.Series) -> int:
    try:
        return int(s.nunique(dropna=True))
    except TypeError:
        return int(s.astype(str).nunique(dropna=True))

def _dtype_str(s: pd.Series) -> str:
    dt = s.dtype
    if pd.api.types.is_datetime64_any_dtype(dt):
        return "datetime"
    if pd.api.types.is_integer_dtype(dt):
        return "int"
    if pd.api.types.is_float_dtype(dt):
        return "float"
    if pd.api.types.is_bool_dtype(dt):
        return "bool"
    return "string"

def _name_role_guess(colname: str) -> Optional[str]:
    for role, pat in ROLE_PATTERNS:
        if pat.search(colname):
            return role
    return None

def _bool_token_ratio(s_obj: pd.Series) -> float:
    vals = s_obj.dropna().astype(str).str.strip().str.lower()
    if vals.empty:
        return 0.0
    return float(vals.isin(_BOOL_TOKENS).mean())

def _value_role_guess(s: pd.Series) -> Optional[str]:
    # Too sparse to trust → treat as text-ish so downstream doesn't overfit
    non_null_ratio = _non_null_ratio(s)
    if non_null_ratio < _ROLECFG.min_non_null_ratio:
        return "text"

    if pd.api.types.is_datetime64_any_dtype(s):
        return "time"

    if pd.api.types.is_bool_dtype(s):
        return "bool"

    if pd.api.types.is_numeric_dtype(s):
        nunique = _safe_nunique(s)
        if s.size > 0 and nunique / max(1, s.size) >= _ROLECFG.unique_id_ratio:
            return "id"
        return "numeric"

    # strings/objects
    nunique = _safe_nunique(s)
    size = max(1, s.size)
    nunique_ratio = nunique / size

    # Bool tokens?
    bt = _bool_token_ratio(s)
    if bt >= _ROLECFG.bool_token_min_ratio:
        return "bool"

    # Categorical vs text: use unique ratio + avg length
    vals = s.dropna().astype(str)
    avg_len = float(vals.map(len).mean()) if not vals.empty else 0.0

    if nunique_ratio <= _ROLECFG.categorical_max_unique_ratio and nunique <= 200:
        return "categorical"

    if avg_len >= _ROLECFG.text_min_avg_len:
        return "text"

    # fallback: categorical for "short-ish" low-card strings
    return "categorical" if nunique <= 200 else "text"

def detect_unit_hint(s: pd.Series) -> Optional[str]:
    if not pd.api.types.is_object_dtype(s) and not pd.api.types.is_string_dtype(s):
        return None
    sample = s.dropna().astype(str).head(50)
    for pat, name in NUMERIC_UNITS:
        if sample.map(lambda x: bool(pat.search(x))).mean() > 0.2:
            return name
    return None

def canonicalize_categories(s: pd.Series) -> Optional[dict]:
    if not (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)):
        return None
    vals = s.dropna().astype(str)
    if vals.nunique(dropna=True) > 200:
        return None

    # map TRIMMED original -> canonical(lowercased+single-spaced)
    top = vals.value_counts().head(200).index.tolist()

    def canon(x: str) -> str:
        return re.sub(r"\s+", " ", x.strip().lower())

    mapping: dict[str, str] = {}
    for v in top:
        k = v.strip()
        mapping[k] = canon(v)

    return mapping or None

def _blend_conf(name_present: bool, value_present: bool, same: bool) -> float:
    # keep your current values; we’ll scale later by coverage
    nw, vw = float(_ROLECFG.name_weight), float(_ROLECFG.value_weight)
    if same and name_present and value_present:
        return min(0.99, 0.90 + 0.10 * max(nw, vw))  # ≈0.95
    if value_present and name_present:
        base = 0.75
        bump = 0.05 if vw >= nw else 0.0
        return base + bump  # ≈0.80
    if value_present:
        return 0.70 + 0.10 * vw  # ≈0.755
    if name_present:
        return 0.60 + 0.05 * nw  # ≈0.6225
    return 0.50

def _apply_quality_penalties(role: str, s: pd.Series, conf: float) -> float:
    # 1) coverage scaling: sparse columns -> near-zero confidence
    cover = _non_null_ratio(s)
    # keep some floor for non-empty columns; make it steeper if you want
    conf *= (0.25 + 0.75 * cover)  # cover=0 => 0.25x; cover=1 => 1x
    # 2) bool penalty if many distinct tokens
    if role == "bool":
        nunique = _safe_nunique(s.astype(str))
        if nunique > 4:
            conf -= _ROLECFG.penalize_bool_for_many_tokens
    # clamp
    return float(max(0.0, min(0.99, conf)))

def guess_role(colname: str, s: pd.Series) -> Tuple[str, float]:
    name_guess = _name_role_guess(colname) or ""
    value_guess = _value_role_guess(s) or ""

    if value_guess == "id" and name_guess:
        role, conf = name_guess, max(0.80, _blend_conf(True, True, False))
        return role, _apply_quality_penalties(role, s, conf)

    if name_guess and value_guess:
        if name_guess == value_guess:
            role, conf = name_guess, _blend_conf(True, True, True)
            return role, _apply_quality_penalties(role, s, conf)
        role, conf = value_guess, _blend_conf(True, True, False)
        return role, _apply_quality_penalties(role, s, conf)

    if value_guess:
        role, conf = value_guess, _blend_conf(False, True, False)
        return role, _apply_quality_penalties(role, s, conf)

    if name_guess:
        role, conf = name_guess, _blend_conf(True, False, False)
        return role, _apply_quality_penalties(role, s, conf)

    role, conf = "text", 0.5
    return role, _apply_quality_penalties(role, s, conf)
