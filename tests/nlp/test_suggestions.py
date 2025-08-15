from __future__ import annotations
from types import SimpleNamespace
import pandas as pd
import pytest

from src.nlp.suggestions import plan_followups


# --- helpers -----------------------------------------------------------------

def _schema(columns: list[dict]):
    """
    Minimal ProposedSchema-like with .to_dict().
    """
    class _PS:
        def __init__(self, cols):
            self._cols = cols

        def to_dict(self):
            return {
                "dataset": "unit_test",
                "schema_confidence": 0.0,
                "columns": [
                    {
                        "name": c["name"],
                        "role": c.get("role", "text"),
                        "role_confidence": c.get("role_confidence", 0.0),
                    }
                    for c in self._cols
                ],
            }
    return _PS(columns)


# --- tests -------------------------------------------------------------------

def test_plan_followups_suggests_actions_below_threshold():
    # amount_str: 80% numeric-like (meets 0.80 threshold)
    # dt_str: ~40% parseable → should push parse_datetime
    # cat: has nulls, role=categorical → impute/cast suggestions
    df = pd.DataFrame(
        {
            "amount_str": ["1", "2", "3", "4", None],
            "dt_str": ["2024-01-01", "bad", None, "2024/02/01", ""],
            "cat": ["a", None, "b", None, "a"],
        }
    )

    prev = _schema(
        [
            {"name": "amount_str", "role": "numeric"},
            {"name": "dt_str",     "role": "time"},
            {"name": "cat",        "role": "categorical"},
        ]
    )

    # keep column-level after conf < 0.90 to trigger suggestions
    rescore = {
        "schema_conf_before": 0.0,
        "schema_conf_after": 0.50,
        "avg_role_conf_before": 0.0,
        "avg_role_conf_after": 0.50,
        "per_column": {
            "amount_str": {"before": 0.2, "after": 0.5, "role_before": "numeric", "role_after": "numeric"},
            "dt_str":     {"before": 0.2, "after": 0.5, "role_before": "time",    "role_after": "time"},
            "cat":        {"before": 0.1, "after": 0.4, "role_before": "categorical", "role_after": "categorical"},
        },
    }

    # Empty cfg is fine; suggestions.py uses safe defaults via _get(...)
    cfg = SimpleNamespace()

    out = plan_followups(df, prev, rescore, cfg)
    assert isinstance(out, dict) and out, "expected non-empty suggestions"

    # amount_str: should try to coerce to numeric
    amt_recs = " | ".join(out.get("amount_str", []))
    assert "coerce_numeric" in amt_recs

    # dt_str: should try to parse datetimes (and may also suggest impute_dt)
    dt_recs = " | ".join(out.get("dt_str", []))
    assert ("parse_datetime(" in dt_recs) or ("impute_dt(" in dt_recs)

    # cat: should suggest either impute_value(...) or cast_category()
    cat_recs = " | ".join(out.get("cat", []))
    assert ("impute_value(" in cat_recs) or ("cast_category()" in cat_recs)


def test_plan_followups_noop_when_above_threshold():
    df = pd.DataFrame(
        {
            "amount_str": ["1", "2", "3", "4"],
            "dt_str": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
            "cat": ["a", "a", "b", "b"],
        }
    )
    prev = _schema(
        [
            {"name": "amount_str", "role": "numeric", "role_confidence": 0.95},
            {"name": "dt_str",     "role": "time",    "role_confidence": 0.95},
            {"name": "cat",        "role": "categorical", "role_confidence": 0.95},
        ]
    )

    rescore = {
        "schema_conf_before": 0.9,
        "schema_conf_after": 0.96,
        "avg_role_conf_before": 0.9,
        "avg_role_conf_after": 0.96,
        "per_column": {
            "amount_str": {"before": 0.9, "after": 0.96, "role_before": "numeric", "role_after": "numeric"},
            "dt_str":     {"before": 0.9, "after": 0.96, "role_before": "time",    "role_after": "time"},
            "cat":        {"before": 0.9, "after": 0.96, "role_before": "categorical", "role_after": "categorical"},
        },
    }

    cfg = SimpleNamespace()
    out = plan_followups(df, prev, rescore, cfg)
    assert not out or all(len(v) == 0 for v in out.values())
