from __future__ import annotations
from types import SimpleNamespace as NS

from src.cleaning.policy import build_policy_from_config
from src.cleaning.engine import RuleSpec

def _cfg():
    # Minimal, runtime-friendly fake config using SimpleNamespace
    cleaning = NS(
        rules=[
            NS(id="r1", priority=5, when='role == "numeric"', then='impute("median")'),
            NS(id="r2", priority=1, when='type == "string"', then="text_normalize(strip=True)"),
        ],
        columns=NS(cat_cardinality_max=200, drop_missing_pct=0.9, min_unique_ratio=0.001),
        impute=NS(numeric_default="median", categorical_default="Unknown", text_default=""),
        outliers=NS(method="zscore", zscore_threshold=3.0, iqr_multiplier=1.5, winsor_limits=[0.01, 0.99]),
    )
    profiling = NS(roles=NS(datetime_formats=["%Y-%m-%d", "%m/%d/%Y"]))
    nlp = NS(model="dummy", other={"a": 1})
    root = NS(cleaning=cleaning, profiling=profiling, nlp=nlp)
    return root

def test_build_policy_rules_and_env_shortcuts():
    cfg = _cfg()
    rules, env = build_policy_from_config(cfg, dataset_slug="sales_q1")

    # Rules mapped correctly
    assert isinstance(rules, list) and len(rules) == 2
    assert all(isinstance(r, RuleSpec) for r in rules)
    assert rules[0].id == "r1" and rules[0].priority == 5 and rules[0].then.startswith("impute")
    assert rules[1].id == "r2" and rules[1].priority == 1 and "text_normalize" in rules[1].then

    # Full sections available as dicts
    assert isinstance(env["cleaning"], dict)
    assert isinstance(env["profiling"], dict)
    assert isinstance(env["nlp"], dict)

    # Shortcuts wired
    assert env["datetime_formats"] == ["%Y-%m-%d", "%m/%d/%Y"]
    assert env["cat_cardinality_max"] == 200
    assert env["numeric_default"] == "median"
    assert env["categorical_default"] == "Unknown"
    assert env["text_default"] == ""

    # Outlier shortcuts normalized and present
    assert env["winsor_limits"] == (0.01, 0.99)
    assert env["zscore_threshold"] == 3.0
    assert env["iqr_multiplier"] == 1.5
    assert env["detect"] == "zscore"

    # Dataset slug propagated
    assert env.get("dataset_slug") == "sales_q1"

def test_build_policy_with_missing_optional_fields_is_robust():
    # Remove optional bits to ensure defaults don't crash
    cleaning = NS(
        rules=[NS(id="r1", priority=0, when="true", then="drop_column")],
        columns=NS(cat_cardinality_max=100),
        impute=NS(numeric_default="mean", categorical_default="NA", text_default=None),
        outliers=NS(method="iqr", zscore_threshold=None, iqr_multiplier=2.0, winsor_limits=(0.05, 0.95)),
    )
    profiling = NS(roles=NS(datetime_formats=[]))
    nlp = NS()
    cfg = NS(cleaning=cleaning, profiling=profiling, nlp=nlp)

    rules, env = build_policy_from_config(cfg)

    assert len(rules) == 1 and isinstance(rules[0], RuleSpec)
    assert env["datetime_formats"] == []
    assert env["winsor_limits"] == (0.05, 0.95)
    assert env["iqr_multiplier"] == 2.0
    assert env["detect"] == "iqr"
    assert isinstance(env["cleaning"], dict) and isinstance(env["profiling"], dict) and isinstance(env["nlp"], dict)
