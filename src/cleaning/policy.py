from __future__ import annotations
from typing import Any, Tuple
from dataclasses import asdict, is_dataclass

from ..config_model.model import RootCfg
from .engine import RuleSpec

def _to_plain(obj: Any) -> Any:
    """
    Convert structured objects (Pydantic v1/v2, dataclasses, SimpleNamespace)
    into plain Python dicts/lists/primitives.
    """
    # Pydantic v2
    if hasattr(obj, "model_dump") and callable(getattr(obj, "model_dump")):
        try:
            return obj.model_dump()
        except Exception:
            pass
    # Pydantic v1
    if hasattr(obj, "dict") and callable(getattr(obj, "dict")):
        try:
            return obj.dict()
        except Exception:
            pass
    # Dataclass
    if is_dataclass(obj):
        try:
            return asdict(obj)
        except Exception:
            pass
    # SimpleNamespace / generic object with __dict__
    if hasattr(obj, "__dict__"):
        try:
            return {k: _to_plain(v) for k, v in vars(obj).items()}
        except Exception:
            pass
    # dict
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    # list/tuple
    if isinstance(obj, (list, tuple)):
        t = type(obj)
        return t(_to_plain(v) for v in obj)
    # primitive
    return obj

def build_policy_from_config(root_cfg: RootCfg, dataset_slug: str | None = None) -> tuple[list[RuleSpec], dict[str, Any]]:
    """
    Build RuleSpec list from config.cleaning.rules and create an 'env' dict with
    all names the DSL/actions can reference (e.g., datetime_formats).
    Pure: no IO. Robust to Pydantic models, dataclasses, or namespaces.
    """
    # Rules
    rules = [
        RuleSpec(
            id=str(r.id),
            priority=int(r.priority),
            when=str(r.when),
            then=str(r.then),
        )
        for r in getattr(getattr(root_cfg, "cleaning", None), "rules", [])  # safe for namespaces
    ]

    # Sections (safe getattr)
    cleaning = getattr(root_cfg, "cleaning", None)
    profiling = getattr(root_cfg, "profiling", None)
    nlp = getattr(root_cfg, "nlp", None)

    # Shortcuts with defaults
    datetime_formats = getattr(getattr(profiling, "roles", None), "datetime_formats", []) if profiling else []
    cat_cardinality_max = getattr(getattr(cleaning, "columns", None), "cat_cardinality_max", None)
    numeric_default = getattr(getattr(cleaning, "impute", None), "numeric_default", None)
    categorical_default = getattr(getattr(cleaning, "impute", None), "categorical_default", None)
    text_default = getattr(getattr(cleaning, "impute", None), "text_default", None)

    outliers = getattr(cleaning, "outliers", None)
    winsor_limits_raw = getattr(outliers, "winsor_limits", (0.01, 0.99)) if outliers else (0.01, 0.99)
    winsor_limits: Tuple[float, float] = (
        tuple(winsor_limits_raw) if isinstance(winsor_limits_raw, (list, tuple)) else (0.01, 0.99)
    )
    zscore_threshold = getattr(outliers, "zscore_threshold", 3.0) if outliers else 3.0
    iqr_multiplier = getattr(outliers, "iqr_multiplier", 1.5) if outliers else 1.5
    detect = getattr(outliers, "method", "zscore") if outliers else "zscore"

    env: dict[str, Any] = {
        "cleaning": _to_plain(cleaning) if cleaning is not None else {},
        "profiling": _to_plain(profiling) if profiling is not None else {},
        "nlp": _to_plain(nlp) if nlp is not None else {},
        "datetime_formats": datetime_formats,
        "cat_cardinality_max": cat_cardinality_max,
        "numeric_default": numeric_default,
        "categorical_default": categorical_default,
        "text_default": text_default,
        "winsor_limits": winsor_limits,
        "zscore_threshold": zscore_threshold,
        "iqr_multiplier": iqr_multiplier,
        "detect": detect,
    }

    if dataset_slug:
        env["dataset_slug"] = dataset_slug

    return rules, env
