from __future__ import annotations
from typing import Any, Dict
import pandas as pd

from .engine import RuleHit

def build_iteration_report(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    metrics_before: dict[str, dict[str, Any]],
    metrics_after: dict[str, dict[str, Any]],
    rule_hits: list[RuleHit],
    rescore: dict[str, Any],  # or RescoreResult when implemented
) -> dict[str, Any]:
    """
    Assemble a serializable dict for loop_iter_k.json and NLG narrative.
    """
    # TODO: implement
    raise NotImplementedError
