from __future__ import annotations
import pandas as pd

from src.cleaning.rules_builtin.prune import (
    column_stats,
    decide_sparse_columns,
    decide_constant_like_columns,
    apply_prune,
    quarantine_unknown_columns,
)

def _df():
    return pd.DataFrame(
        {
            "keep_me": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "mostly_null": [None, None, None, None, None, None, None, None, None, 1],
            "constant": ["x"] * 10,
            "id": [None] * 10,   # should not drop due to always_keep
        }
    )

def test_column_stats_shapes_and_values():
    df = _df()
    st = column_stats(df)
    assert set(st.columns) == {"dtype","rows","non_null","missing_pct","nunique","unique_ratio"}
    assert st.loc["keep_me","rows"] == len(df)
    assert 0.0 <= st.loc["mostly_null","missing_pct"] <= 1.0
    assert st.loc["constant","nunique"] == 1

def test_decide_helpers_respect_thresholds_and_always_keep():
    df = _df()
    st = column_stats(df)
    # missing ≥ 0.9 → drop
    sparse = decide_sparse_columns(st, drop_missing_pct=0.90, always_keep=["id"])
    assert "mostly_null" in sparse and "id" not in sparse
    # unique_ratio ≤ 0.2 should drop our constant (1/10 = 0.1)
    const = decide_constant_like_columns(st, min_unique_ratio=0.2, always_keep=["id"])
    assert "constant" in const and "id" not in const

def test_apply_prune_drops_and_reports_once():
    df = _df()
    pruned, report, st = apply_prune(
        df,
        drop_missing_pct=0.90,
        min_unique_ratio=0.2,  # len=10, constant has 1/10=0.1
        always_keep=["id"],
    )
    # original untouched
    assert "mostly_null" in df.columns and "constant" in df.columns
    # pruned removes both
    assert "mostly_null" not in pruned.columns
    assert "constant" not in pruned.columns
    assert "id" in pruned.columns
    # report contains both with reasons
    reasons = {r["name"]: r["reason"] for r in report}
    assert reasons["mostly_null"] == "sparse"
    assert reasons["constant"] == "constant"

def test_quarantine_unknown_columns_renames_only_extras():
    df = pd.DataFrame({"a":[1,2], "b":[3,4], "c":[5,6]})
    out, extras = quarantine_unknown_columns(df, allowed=["a","b"])
    assert extras == ["c"]
    assert "c" not in out.columns
    assert "__quarantine__c" in out.columns
    # idempotent
    out2, extras2 = quarantine_unknown_columns(out, allowed=["a","b"])
    assert extras2 == []
    assert out2.equals(out)
