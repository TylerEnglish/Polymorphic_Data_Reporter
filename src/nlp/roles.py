from __future__ import annotations
import re
from typing import Optional, Tuple, Any, Iterable
import pandas as pd
import numpy as np
import warnings

# ---- Fallback config (used when nlp_cfg isn't passed) ----
class _RoleCfgFallback:
    # weights
    name_weight: float = 0.45
    value_weight: float = 0.55
    # thresholds
    bool_token_min_ratio: float = 0.57
    date_parse_min_ratio: float = 0.60
    unique_id_ratio: float = 0.95
    categorical_max_unique_ratio: float = 0.02
    text_min_avg_len: float = 8.0
    min_non_null_ratio: float = 0.10
    # bonuses/penalties
    bonus_id_name: float = 0.10
    penalize_bool_for_many_tokens: float = 0.08

# We keep a module-level fallback, but prefer the live nlp_cfg passed in.
_FALLBACK = _RoleCfgFallback()

def _cfg_from(nlp_cfg: Any | None) -> _RoleCfgFallback:
    """
    Robustly read fields from your TOML-driven config object.
    Works with pydantic, dataclass, SimpleNamespace, or plain dicts.
    """
    if nlp_cfg is None:
        return _FALLBACK

    def _get(obj, path: str, default):
        cur = obj
        for part in path.split("."):
            if cur is None:
                return default
            # dict-like
            if isinstance(cur, dict):
                cur = cur.get(part, None)
            else:
                cur = getattr(cur, part, None)
        return default if cur is None else cur

    rc = _RoleCfgFallback()
    # weights under [nlp.role_scoring]
    rc.name_weight  = float(_get(nlp_cfg, "role_scoring.name_weight",  _FALLBACK.name_weight))
    rc.value_weight = float(_get(nlp_cfg, "role_scoring.value_weight", _FALLBACK.value_weight))

    # thresholds directly under [nlp]
    rc.bool_token_min_ratio        = float(getattr(nlp_cfg, "bool_token_min_ratio",        _FALLBACK.bool_token_min_ratio))
    rc.date_parse_min_ratio        = float(getattr(nlp_cfg, "date_parse_min_ratio",        _FALLBACK.date_parse_min_ratio))
    rc.unique_id_ratio             = float(getattr(nlp_cfg, "unique_id_ratio",             _FALLBACK.unique_id_ratio))
    rc.categorical_max_unique_ratio= float(getattr(nlp_cfg, "categorical_max_unique_ratio",_FALLBACK.categorical_max_unique_ratio))
    rc.text_min_avg_len            = float(getattr(nlp_cfg, "text_min_avg_len",            _FALLBACK.text_min_avg_len))
    rc.min_non_null_ratio          = float(getattr(nlp_cfg, "min_non_null_ratio",          _FALLBACK.min_non_null_ratio))
    rc.bonus_id_name               = float(getattr(nlp_cfg, "bonus_id_name",               _FALLBACK.bonus_id_name))
    rc.penalize_bool_for_many_tokens = float(getattr(nlp_cfg, "penalize_bool_for_many_tokens", _FALLBACK.penalize_bool_for_many_tokens))
    return rc

# ---- Name patterns & units ----
ROLE_PATTERNS = [
    ("time", re.compile(r"(date|time|timestamp|dt|ts)\b", re.I)),
    ("id",   re.compile(r"(?:^|[_-])(id|uuid|guid|key)\b", re.I)),
    ("bool", re.compile(r"^(is_|has_|flag_)", re.I)),
    ("geo",  re.compile(r"(lat|lng|lon|long|latitude|longitude)\b", re.I)),
]

NUMERIC_UNITS = [
    (re.compile(r"^\s*\$"), "currency"),
    (re.compile(r"\b(percent|pct|%)\b", re.I), "percent"),
    (re.compile(r"\b(k|thousand)\b", re.I), "magnitude_k"),
    (re.compile(r"\b(million|mm)\b", re.I), "magnitude_m"),
]

_BOOL_TOKENS = {"true","false","t","f","y","n","yes","no","1","0"}

# ---- Utilities shared by other modules ----
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

# ---- Local helpers ----
def _non_null_ratio(s: pd.Series) -> float:
    return float(s.notna().mean()) if len(s) else 0.0

def _safe_nunique(s: pd.Series) -> int:
    try:
        return int(s.nunique(dropna=True))
    except TypeError:
        return int(s.astype(str).nunique(dropna=True))

def _bool_token_ratio(s_obj: pd.Series) -> float:
    vals = s_obj.dropna().astype(str).str.strip().str.lower()
    if vals.empty:
        return 0.0
    return float(vals.isin(_BOOL_TOKENS).mean())

def _datetime_parse_ratio(s: pd.Series, fmts: list[str] | None) -> float:
    # Already datetime-like?
    if pd.api.types.is_datetime64_any_dtype(s):
        return 1.0
    try:
        x = s.dropna().astype(str)
        if x.empty:
            return 0.0

        parsed = None
        for f in (fmts or []):
            try:
                cand = pd.to_datetime(x, format=f, errors="coerce")
            except Exception:
                continue
            parsed = cand if parsed is None else parsed.fillna(cand)

        if parsed is None:
            # suppress Pandas' per-element parse warning on fallback
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                parsed = pd.to_datetime(x, errors="coerce")

        return float(parsed.notna().mean())
    except Exception:
        return 0.0

def _avg_len(s: pd.Series) -> float:
    try:
        vals = s.dropna().astype(str)
        return float(vals.map(len).mean()) if not vals.empty else 0.0
    except Exception:
        return 0.0

def _name_role_guess(colname: str) -> Optional[str]:
    for role, pat in ROLE_PATTERNS:
        if pat.search(colname or ""):
            return role
    return None

def _value_role_guess(
    s: pd.Series,
    cfg: _RoleCfgFallback,
    dt_fmts: list[str] | None
) -> Optional[str]:
    non_null_ratio = _non_null_ratio(s)
    if non_null_ratio < cfg.min_non_null_ratio:
        return "text"

    # datetime / bool / numeric native dtypes
    if pd.api.types.is_datetime64_any_dtype(s):
        return "time"
    if pd.api.types.is_bool_dtype(s):
        return "bool"
    if pd.api.types.is_numeric_dtype(s):
        nunique = _safe_nunique(s)
        if s.size > 0 and nunique / max(1, s.size) >= cfg.unique_id_ratio:
            return "id"
        return "numeric"

    # ---- object/string branch ----
    vals = s.dropna().astype(str)

    # 1) date-like strings?
    if not vals.empty:
        dt_ratio = _datetime_parse_ratio(vals, dt_fmts or [])
        if dt_ratio >= cfg.date_parse_min_ratio:
            return "time"

    # 2) numeric-like strings?
    if not vals.empty:
        # tolerate commas; expand here if you want $, % handling
        num = pd.to_numeric(vals.str.replace(",", ""), errors="coerce")
        numeric_ratio = float(num.notna().mean())
        if numeric_ratio >= 0.80:  # tune if needed
            nunique = int(num.dropna().nunique())
            size = max(1, s.size)
            if nunique / size >= cfg.unique_id_ratio:
                return "id"
            return "numeric"

    # 3) boolean-ish tokens?
    bt = _bool_token_ratio(s)
    if bt >= cfg.bool_token_min_ratio:
        return "bool"

    # 4) categorical vs text
    nunique = _safe_nunique(s)
    size = max(1, s.size)
    nunique_ratio = nunique / size
    avg_len = _avg_len(vals)

    if nunique_ratio <= cfg.categorical_max_unique_ratio and nunique <= 200:
        return "categorical"
    if avg_len >= cfg.text_min_avg_len:
        return "text"
    return "categorical" if nunique <= 200 else "text"

def detect_unit_hint(s: pd.Series) -> Optional[str]:
    if not (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)):
        return None
    sample = s.dropna().astype(str).head(50)
    for pat, name in NUMERIC_UNITS:
        try:
            if sample.map(lambda x: bool(pat.search(x))).mean() > 0.2:
                return name
        except Exception:
            continue
    return None

def canonicalize_categories(s: pd.Series) -> Optional[dict]:
    if not (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)):
        return None
    vals = s.dropna().astype(str)
    if vals.nunique(dropna=True) > 200:
        return None

    def canon(x: str) -> str:
        return re.sub(r"\s+", " ", x.strip().lower())

    mapping: dict[str, str] = {}
    for v in vals.value_counts().head(200).index.tolist():
        mapping[v.strip()] = canon(v)
    return mapping or None

def _blend_conf(name_present: bool, value_present: bool, same: bool, cfg: _RoleCfgFallback) -> float:
    nw, vw = float(cfg.name_weight), float(cfg.value_weight)
    if same and name_present and value_present:
        return min(0.99, 0.90 + 0.10 * max(nw, vw))  # ~0.95
    if value_present and name_present:
        base = 0.75
        bump = 0.05 if vw >= nw else 0.0
        return base + bump                                  # ~0.80
    if value_present:
        return 0.70 + 0.10 * vw                              # ~0.755
    if name_present:
        return 0.60 + 0.05 * nw                              # ~0.6225
    return 0.50

def _apply_quality_penalties(role: str, s: pd.Series, conf: float, cfg: _RoleCfgFallback) -> float:
    # 1) coverage scaling
    cover = _non_null_ratio(s)
    conf *= (0.25 + 0.75 * cover)  # cover=0 -> 0.25x; cover=1 -> 1x
    # 2) bool penalty if too many distinct tokens
    if role == "bool":
        try:
            nunique = _safe_nunique(s.astype(str))
            if nunique > 4:
                conf -= cfg.penalize_bool_for_many_tokens
        except Exception:
            pass
    return float(max(0.0, min(0.99, conf)))

# ---- Public API: must accept the signatures rescore() tries ----
def guess_role(
    colname: str | pd.Series,
    s: Optional[pd.Series] = None,
    nlp_cfg: Any = None,
    datetime_formats: Optional[list[str]] = None,
) -> Tuple[str, float]:
    """
    Return (role, confidence in [0,1]).
    Accepts multiple signatures:
      - guess_role(name, series, nlp_cfg, datetime_formats)
      - guess_role(name, series, nlp_cfg)
      - guess_role(series, nlp_cfg, datetime_formats)
      - guess_role(series, nlp_cfg)
      - guess_role(series)
    """
    # Normalize arguments to (name, series)
    name: str = ""
    series: pd.Series

    if isinstance(colname, pd.Series) and s is None:
        series = colname
        name = ""
    else:
        name = str(colname) if colname is not None else ""
        series = s  # type: ignore

    if series is None:
        # Defensive default
        return "text", 0.0

    cfg = _cfg_from(nlp_cfg)
    # get datetime formats if they were passed as a roles subconfig or list
    dt_fmts = datetime_formats if isinstance(datetime_formats, list) else []

    name_guess  = _name_role_guess(name) or ""
    value_guess = _value_role_guess(series, cfg, dt_fmts) or ""

    # If value suggests 'id' but name suggests something else, prefer the name (soft rule)
    if value_guess == "id" and name_guess:
        role, conf = name_guess, max(0.80, _blend_conf(True, True, False, cfg))
        return role, _apply_quality_penalties(role, series, conf, cfg)

    if name_guess and value_guess:
        if name_guess == value_guess:
            role, conf = name_guess, _blend_conf(True, True, True, cfg)
            return role, _apply_quality_penalties(role, series, conf, cfg)
        role, conf = value_guess, _blend_conf(True, True, False, cfg)
        return role, _apply_quality_penalties(role, series, conf, cfg)

    if value_guess:
        role, conf = value_guess, _blend_conf(False, True, False, cfg)
        return role, _apply_quality_penalties(role, series, conf, cfg)

    if name_guess:
        role, conf = name_guess, _blend_conf(True, False, False, cfg)
        return role, _apply_quality_penalties(role, series, conf, cfg)

    role, conf = "text", 0.5
    return role, _apply_quality_penalties(role, series, conf, cfg)
