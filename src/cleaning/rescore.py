from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict
import pandas as pd

from ..nlp.schema import ProposedSchema
from ..nlp.roles import guess_role
from ..config_model.model import NLPCfg, ProfilingRolesCfg

@dataclass(frozen=True)
class RescoreResult:
    schema_conf_before: float
    schema_conf_after: float
    avg_role_conf_before: float
    avg_role_conf_after: float
    per_column: dict[str, dict[str, float]]  # {"amount":{"before":0.7,"after":0.9}, ...}

def rescore_after_clean(
    df: pd.DataFrame,
    prev_schema: ProposedSchema,
    profiling_cfg: ProfilingRolesCfg,
    nlp_cfg: NLPCfg,
) -> RescoreResult:
    """
    Re-run role inference on cleaned df and compute new confidences.
    Apply coverage/quality weighting (TODO).
    """
    # TODO: implement
    raise NotImplementedError
