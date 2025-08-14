from __future__ import annotations
import re
from typing import Optional, Tuple
import pandas as pd
import numpy as np

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

def _safe_nunique(s: pd.Series) -> int:
    try:
        return int(s.nunique(dropna=True))
    except TypeError:
        # unhashable (e.g., dicts/lists); coerce to string for a best-effort estimate
        return int(s.astype(str).nunique(dropna=True))

def _dtype_str(s: pd.Series) -> str:
    # Normalize to a friendly dtype name
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

def _value_role_guess(s: pd.Series) -> Optional[str]:
    if pd.api.types.is_datetime64_any_dtype(s):
        return "time"
    if pd.api.types.is_bool_dtype(s):
        return "bool"
    if pd.api.types.is_numeric_dtype(s):
        nunique = _safe_nunique(s)
        if s.size > 0 and nunique / max(1, s.size) > 0.95:
            return "id"
        return "numeric"
    # strings/objects
    nunique = _safe_nunique(s)
    if nunique <= 2:
        return "bool"
    if nunique <= 200:
        return "categorical"
    return "text"

def detect_unit_hint(s: pd.Series) -> Optional[str]:
    if not pd.api.types.is_object_dtype(s) and not pd.api.types.is_string_dtype(s):
        return None
    # look at a small sample of non-null values
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

    # top originals; map TRIMMED original -> canonical(lowercased+single-spaced)
    top = vals.value_counts().head(200).index.tolist()

    def canon(x: str) -> str:
        return re.sub(r"\s+", " ", x.strip().lower())

    mapping: dict[str, str] = {}
    for v in top:
        k = v.strip()
        mapping[k] = canon(v)

    return mapping or None

def guess_role(colname: str, s: pd.Series) -> Tuple[str, float]:
    name_guess = _name_role_guess(colname) or ""
    value_guess = _value_role_guess(s) or ""

    # If value thinks it's an id but the name strongly suggests a semantic role, prefer the name.
    if value_guess == "id" and name_guess:
        return name_guess, 0.80

    if name_guess == value_guess and name_guess:
        return name_guess, 0.95
    if value_guess:
        return value_guess, 0.8 if name_guess else 0.7
    if name_guess:
        return name_guess, 0.65
    return "text", 0.5
