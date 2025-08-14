from __future__ import annotations

# Public API re-exports (keep small & stable)
from .engine import (
    RuleSpec,
    RuleHit,
    CleaningResult,
    run_clean_pass,
    apply_rules,
)
from .policy import build_policy_from_config
from .dsl import compile_condition, eval_condition
