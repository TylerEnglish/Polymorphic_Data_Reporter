from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from ..config_model.model import RootCfg
from .engine import RuleSpec

def build_policy_from_config(root_cfg: RootCfg, dataset_slug: str | None = None) -> tuple[list[RuleSpec], dict[str, Any]]:
    """
    Build RuleSpec list from config.cleaning.rules and create an 'env' dict with
    all names the DSL/actions can reference (e.g., datetime_formats).
    Pure: no IO.
    """
    rules = [
        RuleSpec(
            id=r.id,
            priority=int(r.priority),
            when=str(r.when),
            then=str(r.then),
        )
        for r in root_cfg.cleaning.rules
    ]
    # Bind env (read-only) for DSL and actions
    env: dict[str, Any] = {
        "cleaning": root_cfg.cleaning.model_dump(),
        "profiling": root_cfg.profiling.model_dump(),
        "nlp": root_cfg.nlp.model_dump(),
        # Shortcuts (common names used in 'then' specs)
        "datetime_formats": root_cfg.profiling.roles.datetime_formats,
        "cat_cardinality_max": root_cfg.cleaning.columns.cat_cardinality_max,
        "numeric_default": root_cfg.cleaning.impute.numeric_default,
        "categorical_default": root_cfg.cleaning.impute.categorical_default,
        "text_default": root_cfg.cleaning.impute.text_default,
        "winsor_limits": tuple(root_cfg.cleaning.outliers.winsor_limits),
        "zscore_threshold": root_cfg.cleaning.outliers.zscore_threshold,
        "iqr_multiplier": root_cfg.cleaning.outliers.iqr_multiplier,
        "detect": root_cfg.cleaning.outliers.method,
    }
    return rules, env
