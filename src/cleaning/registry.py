from __future__ import annotations
from typing import Any, Callable, Dict, Tuple
import pandas as pd

# Each action gets (series, ctx) and returns a new series (and optional notes).
ActionFn = Callable[[pd.Series, dict[str, Any]], tuple[pd.Series, str | None] | pd.Series]

def compile_actions_registry() -> dict[str, ActionFn]:
    """
    Map action names -> callables.
    Add your plug-ins here.
    """
    from .rules_builtin.types import (
        coerce_numeric_from_string,
        parse_datetime_from_string,
        cast_category_if_small,
    )
    from .rules_builtin.missing import impute_numeric, impute_value
    from .rules_builtin.outliers import outliers_apply  # small wrapper
    from .rules_builtin.text_norm import text_normalize
    # units handled inside numeric coercion; optional dedicated step:
    from .rules_builtin.units import standardize_units_numeric

    # Wrap primitives into (s, ctx) -> (s2, notes)
    def _wrap(fn, note_fmt: str) -> ActionFn:
        def inner(s: pd.Series, ctx: dict[str, Any]):
            s2 = fn(s, **ctx.get("params", {}))
            return s2, note_fmt
        return inner

    registry: dict[str, ActionFn] = {
        "coerce_numeric": _wrap(coerce_numeric_from_string, "coerce_numeric"),
        "parse_datetime": _wrap(parse_datetime_from_string, "parse_datetime"),
        "cast_category": _wrap(cast_category_if_small, "cast_category"),
        "impute": _wrap(impute_numeric, "impute"),
        "impute_value": _wrap(impute_value, "impute_value"),
        "outliers": _wrap(outliers_apply, "outliers"),
        "text_normalize": _wrap(text_normalize, "text_normalize"),
        "standardize_units": _wrap(standardize_units_numeric, "standardize_units"),
        "drop_column": lambda s, ctx: (pd.Series(dtype="float64"), "drop_column"),  # engine removes it
    }
    return registry

def parse_then(spec: str, registry: dict[str, ActionFn]) -> tuple[ActionFn, dict[str, Any]]:
    """
    Parse a 'then' string like:
      'impute("median")'
      'parse_datetime(datetime_formats)'
      'text_normalize(strip=True, lower=False)'
    into (callable, bound_params).

    NOTE: we do a tiny parser instead of eval. Keep spec simple.
    """
    # TODO: parse function name + (args/kwargs); resolve names only from ctx['env'].
    raise NotImplementedError("parse_then not implemented yet.")
