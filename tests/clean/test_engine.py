from __future__ import annotations
import pandas as pd
import numpy as np
import types
import pytest

from src.cleaning.engine import (
    apply_rules,
    run_clean_pass,
    RuleSpec,
)
from src.cleaning.registry import compile_actions_registry


def _df():
    return pd.DataFrame(
        {
            # numeric-looking strings; will be coerced
            "amount": ["$1.00", "2", None],
            # text that needs normalization (strip/NFKC)
            "name": ["  Hi  ", None, " Café "],
            # datetimes in mixed formats
            "dt": ["2024-01-02", "01/03/2024", "bad"],
            # numeric with an outlier
            "score": [10, 12, 1000],
        }
    )


def _schema_roles():
    # engine uses roles only from here (not dtype) to trigger numeric coercion
    return {
        "amount": "numeric",
        "name": "text",
        "dt": "time",
        "score": "numeric",
    }


def test_apply_rules_basic_flow():
    df = _df()
    schema_roles = _schema_roles()
    registry = compile_actions_registry()

    # Rules:
    #  1) normalize strings
    #  2) coerce numeric for numeric role columns
    #  3) parse datetime with formats coming from env (NameRef resolution)
    #  4) outlier flag on numeric
    rules = [
        RuleSpec(id="norm_text", priority=10, when='type == "string"', then='text_normalize(strip=true)'),
        RuleSpec(id="num_coerce", priority=9, when='schema_role == "numeric"', then='coerce_numeric()'),
        RuleSpec(id="parse_dt", priority=9, when='role == "time"', then='parse_datetime(datetime_formats)'),
        RuleSpec(id="flag_outliers", priority=5, when='type in ["int","float"]', then='outliers(method="zscore", zscore_threshold=3.0, handle="flag")'),
    ]

    env = {
        "profiling": {"roles": {"datetime_formats": ["%Y-%m-%d", "%m/%d/%Y"]}},
        "datetime_formats": ["%Y-%m-%d", "%m/%d/%Y"],
    }

    out_df, hits, col_report, dropped = apply_rules(df, schema_roles, rules, env, registry)

    # amount → numeric
    assert out_df["amount"].dtype.kind in ("f", "i")
    assert list(out_df["amount"].astype(float))[:2] == [1.0, 2.0]
    assert np.isnan(float(out_df["amount"].iloc[2]))

    # name → trimmed / normalized (strip only; we didn’t lower)
    assert out_df["name"].tolist() == ["Hi", None, "Café"]

    # dt → datetime parsed via formats
    assert str(out_df["dt"].dtype).startswith("datetime64")
    assert pd.Timestamp(2024, 1, 2) == out_df["dt"].iloc[0]
    assert pd.Timestamp(2024, 1, 3) == out_df["dt"].iloc[1]
    assert pd.isna(out_df["dt"].iloc[2])

    # outlier rule run on numeric columns; ensure note present for 'score'
    assert "score" in col_report
    assert any("outliers" in note for note in (col_report["score"]["actions"] or []))

    # no drops in this scenario
    assert dropped == {}

    # sanity: we get a RuleHit per column
    cols_hit = {h.column for h in hits}
    assert set(cols_hit) == set(df.columns)


class _DummySchema:
    """
    Lightweight stand-in for ProposedSchema for run_clean_pass.
    """
    def __init__(self, columns, conf: float = 0.42) -> None:
        self._cols = columns
        self.schema_confidence = conf

    def to_dict(self):
        return {"columns": self._cols}


def test_run_clean_pass_with_policy_monkeypatch(monkeypatch):
    df = _df()

    # Build a dummy ProposedSchema instance
    proposed = _DummySchema(
        [
            {"name": "amount", "role": "numeric"},
            {"name": "name", "role": "text"},
            {"name": "dt", "role": "time"},
            {"name": "score", "role": "numeric"},
        ],
        conf=0.50,
    )

    # Minimal cfg stub with attributes engine expects
    cfg = types.SimpleNamespace(
        profiling=types.SimpleNamespace(
            roles=types.SimpleNamespace(datetime_formats=["%Y-%m-%d", "%m/%d/%Y"])
        ),
        nlp=types.SimpleNamespace(),
    )

    # Prepare the rules/env returned by policy.build_policy_from_config
    rules = [
        RuleSpec(id="norm_text", priority=10, when='type == "string"', then='text_normalize(strip=true)'),
        RuleSpec(id="num_coerce", priority=9, when='schema_role == "numeric"', then='coerce_numeric()'),
        RuleSpec(id="parse_dt", priority=9, when='role == "time"', then='parse_datetime(datetime_formats)'),
        RuleSpec(id="flag_outliers", priority=5, when='type in ["int","float"]', then='outliers(method="zscore", zscore_threshold=3.0, handle="flag")'),
    ]
    env = {
        "profiling": {"roles": {"datetime_formats": ["%Y-%m-%d", "%m/%d/%Y"]}},
        "datetime_formats": ["%Y-%m-%d", "%m/%d/%Y"],
        "cleaning": {},
        "nlp": {},
    }

    # Monkeypatch policy builder the engine imports inside run_clean_pass
    import src.cleaning.policy as policy_mod
    monkeypatch.setattr(policy_mod, "build_policy_from_config", lambda _cfg: (rules, env))

    # Run
    from src.cleaning.engine import run_clean_pass  # import after patch to be explicit
    result = run_clean_pass(df, proposed, cfg)

    # Cleaned DF checks (same as apply_rules)
    out = result.clean_df
    assert out["name"].tolist() == ["Hi", None, "Café"]
    assert out["amount"].dtype.kind in ("f", "i")
    assert str(out["dt"].dtype).startswith("datetime64")

    # Report structure
    rep = result.report
    for key in (
        "row_count_before",
        "row_count_after",
        "column_count_before",
        "column_count_after",
        "rule_hits",
        "metrics_delta",
        "rescore",
    ):
        assert key in rep

    # Rescore keys exist and are numeric
    resc = rep["rescore"]
    for k in ("schema_conf_before", "schema_conf_after", "avg_role_conf_before", "avg_role_conf_after"):
        assert isinstance(resc[k], float)

    # Column report contains action notes
    assert "score" in result.column_report
    assert any("outliers" in note for note in result.column_report["score"]["actions"])
