from __future__ import annotations
import pandas as pd
import numpy as np

from src.cleaning.report import build_iteration_report
from src.cleaning.engine import RuleHit
from src.cleaning.metrics import profile_columns
from src.cleaning.rescore import RescoreResult


def _frames_and_metrics():
    df_before = pd.DataFrame(
        {
            "a": [1, None, 3],
            "b": [" Hi ", "There", None],
            "dt": ["2024-01-01", None, "2024-01-03"],
        }
    )
    # after: b stripped, a imputed to fill NaN with 2, dt dropped, c added
    df_after = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0],
            "b": ["Hi", "There", None],
            "c": [10, 20, 30],
        }
    )

    schema_roles = {"a": "numeric", "b": "text", "dt": "time", "c": "numeric"}
    fmts = ["%Y-%m-%d", "%m/%d/%Y"]

    mb = profile_columns(df_before, schema_roles, fmts)
    ma = profile_columns(df_after, schema_roles, fmts)
    return df_before, df_after, mb, ma


def test_build_iteration_report_with_dict_rescore():
    dfb, dfa, mb, ma = _frames_and_metrics()
    hits = [
        RuleHit(column="a", rule_id="impute", before_type="float", after_type="float", before_role="numeric", after_role=None, notes="impute"),
        RuleHit(column="b", rule_id="text_normalize", before_type="string", after_type="string", before_role="text", after_role=None, notes="text_normalize"),
        RuleHit(column="dt", rule_id="drop_column", before_type="string", after_type="string", before_role="time", after_role=None, notes="drop_column"),
    ]
    rescore = {
        "schema_conf_before": 0.5,
        "schema_conf_after": 0.55,
        "avg_role_conf_before": 0.4,
        "avg_role_conf_after": 0.6,
        "per_column": {"a": {"before": 0.4, "after": 0.7}},
    }

    rep = build_iteration_report(dfb, dfa, mb, ma, hits, rescore)

    # shape
    assert rep["shape"]["before"]["rows"] == 3 and rep["shape"]["after"]["rows"] == 3
    assert rep["shape"]["before"]["columns"] == 3 and rep["shape"]["after"]["columns"] == 3

    # added/dropped
    assert "dt" in rep["columns"]["dropped"]
    assert "c" in rep["columns"]["added"]

    # missing % delta: a decreased (since we imputed), b largely same except trimming doesn't affect NaNs
    a_md = rep["metrics"]["missing_pct"]["a"]
    assert isinstance(a_md["before"], float) and isinstance(a_md["after"], float)
    assert a_md["delta"] < 0  # fewer missing after impute

    # per-column actions collected
    assert "a" in rep["rules"]["per_column_actions"]
    assert "impute" in rep["rules"]["per_column_actions"]["a"]
    assert "text_normalize" in rep["rules"]["per_column_actions"]["b"]

    # rescore keys present and coerced to floats
    for k in ("schema_conf_before", "schema_conf_after", "avg_role_conf_before", "avg_role_conf_after"):
        assert isinstance(rep["rescore"][k], float)

    # narrative exists
    assert "Rows:" in rep["narrative"]


def test_build_iteration_report_with_dataclass_rescore():
    dfb, dfa, mb, ma = _frames_and_metrics()
    hits = []
    rc = RescoreResult(
        schema_conf_before=0.3,
        schema_conf_after=0.35,
        avg_role_conf_before=0.2,
        avg_role_conf_after=0.25,
        per_column={"b": {"before": 0.3, "after": 0.31}},
    )
    rep = build_iteration_report(dfb, dfa, mb, ma, hits, rc)
    assert rep["rescore"]["schema_conf_after"] == 0.35
    assert "per_column" in rep["rescore"]
