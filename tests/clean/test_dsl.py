from __future__ import annotations
import pytest

from src.cleaning.dsl import compile_condition, eval_condition


def _ctx():
    # minimal but realistic metrics/env context like engine.apply_rules constructs
    return {
        "name": "amount",
        "type": "float",
        "role": "numeric",
        "missing_pct": 0.15,
        "non_null_ratio": 0.85,
        "nunique": 123,
        "unique_ratio": 0.0123,
        "cardinality": 123,
        "avg_len": None,
        "has_time_index": False,
        "mean": 42.0,
        "std": 3.0,
        "iqr": 4.5,
        "cleaning": {
            "columns": {
                "drop_missing_pct": 0.90,
                "min_unique_ratio": 0.001,
                "always_keep": ["id", "date", "timestamp"],
                "cat_cardinality_max": 200,
            },
            "normalize": {"strip_text": True, "lowercase_text": False},
            "impute": {"numeric_default": "median", "categorical_default": "Unknown"},
            "outliers": {"method": "zscore", "zscore_threshold": 3.0, "iqr_multiplier": 1.5},
        },
        "profiling": {
            "roles": {"datetime_formats": ["%Y-%m-%d", "%m/%d/%Y"]},
            "outliers": {"method": "zscore", "zscore_threshold": 3.0},
        },
        # some ad-hoc vars to test membership
        "fmt_list": ["%Y-%m-%d", "%H:%M:%S"],
        "flag": True,
        "note": None,
        "value": "a\"b",  # used for string escape tests
    }


def test_basic_numbers_strings_bools_none_and():
    expr = 'missing_pct >= 0.10 and role == "numeric"'
    fn = compile_condition(expr, allowed_vars={"missing_pct", "role"})
    assert eval_condition(fn, _ctx()) is True

    expr2 = "missing_pct < 0.10 and role == 'numeric'"
    fn2 = compile_condition(expr2, allowed_vars={"missing_pct", "role"})
    assert eval_condition(fn2, _ctx()) is False


def test_or_not_parentheses_precedence():
    # not (A or B) and role != "id"
    expr = 'not (missing_pct > 0.5 or unique_ratio < 0.001) and role != "id"'
    fn = compile_condition(expr, allowed_vars={"missing_pct", "unique_ratio", "role"})
    # unique_ratio=0.0123, so "unique_ratio < 0.001" is False; missing_pct>0.5 is False
    # not(False or False) -> True; role != "id" -> True
    assert eval_condition(fn, _ctx()) is True


def test_membership_in_and_notin_over_nested_env():
    # name in always_keep?
    expr = "name in cleaning.columns.always_keep"
    fn = compile_condition(expr, allowed_vars={"name", "cleaning"})
    assert eval_condition(fn, _ctx()) is False  # "amount" not in ["id","date","timestamp"]

    expr2 = "name notin cleaning.columns.always_keep"
    fn2 = compile_condition(expr2, allowed_vars={"name", "cleaning"})
    assert eval_condition(fn2, _ctx()) is True


def test_membership_against_ctx_value_works():
    # RHS provided via ctx (not a literal)
    expr = "name in fmt_list"
    fn = compile_condition(expr, allowed_vars={"name", "fmt_list"})
    assert eval_condition(fn, _ctx()) is False  # "amount" not in the fmt_list


def test_membership_against_list_literal_works():
    # list literals are supported and safe
    expr = 'name in ["id", "amount", "total"]'
    fn = compile_condition(expr, allowed_vars={"name"})
    assert eval_condition(fn, _ctx()) is True


def test_string_quotes_and_escapes():
    # value is a"b (with a quoted double-quote inside literal)
    expr = r'value == "a\"b"'
    fn = compile_condition(expr, allowed_vars={"value"})
    assert eval_condition(fn, _ctx()) is True

    expr2 = r"value == 'a\"b'"
    fn2 = compile_condition(expr2, allowed_vars={"value"})
    assert eval_condition(fn2, _ctx()) is True


def test_true_false_none_tokens():
    expr = "flag == true and note == None"
    fn = compile_condition(expr, allowed_vars={"flag", "note"})
    assert eval_condition(fn, _ctx()) is True


def test_safe_functions_allowed():
    expr = 'startswith(name, "am") and notnull(mean) and std >= 3'
    fn = compile_condition(expr, allowed_vars={"name", "mean", "std"})
    assert eval_condition(fn, _ctx()) is True


def test_allowed_vars_gate_disallowed_root_raises():
    # profiling.* not allowed here
    expr = "profiling.roles.datetime_formats == None"
    with pytest.raises(ValueError):
        compile_condition(expr, allowed_vars={"cleaning"})  # profiling not in allowlist


def test_missing_identifier_resolves_to_none_and_comparisons_fail_cleanly():
    expr = "does_not_exist == 1 or does_not_exist in cleaning.columns.always_keep"
    fn = compile_condition(expr, allowed_vars={"does_not_exist", "cleaning"})
    # unknown -> None; None == 1 -> False; None in [...] -> False
    assert eval_condition(fn, _ctx()) is False


def test_bad_syntax_raises():
    with pytest.raises(ValueError):
        compile_condition("missing_pct > ", allowed_vars={"missing_pct"})


def test_nested_paths_work_and_parentheses_nesting():
    expr = '(cleaning.columns.drop_missing_pct >= 0.8) and ( (role != "id") and (missing_pct < 0.9) )'
    fn = compile_condition(expr, allowed_vars={"cleaning", "role", "missing_pct"})
    assert eval_condition(fn, _ctx()) is True
