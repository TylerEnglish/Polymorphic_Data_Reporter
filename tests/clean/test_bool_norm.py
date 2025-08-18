import pandas as pd
import numpy as np
import pytest

from src.cleaning.rules_builtin.bool_norm import coerce_bool_from_tokens

def assert_boolean_series_equal(a: pd.Series, b: pd.Series):
    # Helper that compares nullable boolean series including <NA>
    assert a.dtype == "boolean"
    assert b.dtype == "boolean"
    pd.testing.assert_series_equal(a.reset_index(drop=True), b.reset_index(drop=True), check_names=False)

def test_basic_tokens_case_whitespace_and_punct():
    s = pd.Series([" yes ", "No", "t", "F", " TRUE!! ", "(false)", None, ""])
    out = coerce_bool_from_tokens(s)
    exp = pd.Series([True, False, True, False, True, False, pd.NA, pd.NA], dtype="boolean")
    assert_boolean_series_equal(out, exp)

def test_on_off_enabled_disabled_active_inactive_defaults():
    s = pd.Series(["on", "OFF", "Enabled", "disabled", "ACTIVE", "inactive", "meh"])
    out = coerce_bool_from_tokens(s)
    exp = pd.Series([True, False, True, False, True, False, pd.NA], dtype="boolean")
    assert_boolean_series_equal(out, exp)

def test_regex_patterns_enabled():
    s = pd.Series(["go", "stop", "proceed", "halt", "GO!", "halt."])
    out = coerce_bool_from_tokens(
        s,
        true_regex=[r"go", r"proceed"],
        false_regex=[r"stop", r"halt"]
    )
    exp = pd.Series([True, False, True, False, True, False], dtype="boolean")
    assert_boolean_series_equal(out, exp)

def test_numeric_dtype_01_only_default():
    s = pd.Series([1, 0, 1.0, 0.0, np.nan, 2, -1])
    out = coerce_bool_from_tokens(s)  # numeric fast-path
    exp = pd.Series([True, False, True, False, pd.NA, pd.NA, pd.NA], dtype="boolean")
    assert_boolean_series_equal(out, exp)

def test_numeric_dtype_truthy_when_allowed():
    s = pd.Series([2, -3, 0, 0.0, np.nan])
    out = coerce_bool_from_tokens(s, numeric_01_only=False, allow_numeric_truthy=True)
    exp = pd.Series([True, True, False, False, pd.NA], dtype="boolean")
    assert_boolean_series_equal(out, exp)

def test_numeric_strings_handled_same_as_numeric():
    s = pd.Series(["1", "0", "2", "-1", "0.0", "3.14", "nope"])
    out = coerce_bool_from_tokens(s)
    exp = pd.Series([True, False, pd.NA, pd.NA, False, pd.NA, pd.NA], dtype="boolean")
    assert_boolean_series_equal(out, exp)

def test_case_sensitive_off_when_requested():
    s = pd.Series(["Yes", "No", "YES", "NO"])
    out_ci = coerce_bool_from_tokens(s, case_insensitive=True)
    out_cs = coerce_bool_from_tokens(s, case_insensitive=False)
    exp_ci = pd.Series([True, False, True, False], dtype="boolean")
    # With case_sensitive, only exact-case "yes"/"no" (lowercase) would match; these shouldn't.
    exp_cs = pd.Series([pd.NA, pd.NA, pd.NA, pd.NA], dtype="boolean")
    assert_boolean_series_equal(out_ci, exp_ci)
    assert_boolean_series_equal(out_cs, exp_cs)

def test_preserve_invalid_when_drop_invalid_false_returns_object():
    s = pd.Series(["yes", "Nope", "0", "what", None])
    out = coerce_bool_from_tokens(s, drop_invalid=False)
    # Mixed series: True, original "Nope" untouched, False, "what" untouched, None stays None
    assert out.dtype == object
    assert list(out) == [True, "Nope", False, "what", None]

def test_strip_nonword_edges_enables_matching():
    s = pd.Series(["(Yes!)", "[no]", '"true"', "false,", "##on##", "off."])
    out = coerce_bool_from_tokens(s, strip_nonword_edges=True)
    exp = pd.Series([True, False, True, False, True, False], dtype="boolean")
    assert_boolean_series_equal(out, exp)

def test_conflicting_mappings_become_na():
    # Create a situation where "ok" is both true and false via custom tokens
    s = pd.Series(["ok", "OK", "fine"])
    out = coerce_bool_from_tokens(
        s,
        true_tokens={"ok"},
        false_tokens={"ok"}
    )
    exp = pd.Series([pd.NA, pd.NA, pd.NA], dtype="boolean")
    assert_boolean_series_equal(out, exp)

def test_handles_categorical_and_object_dtypes():
    s = pd.Series(["yes", "no", "maybe"], dtype="category")
    out = coerce_bool_from_tokens(s)
    exp = pd.Series([True, False, pd.NA], dtype="boolean")
    assert_boolean_series_equal(out, exp)

def test_none_empty_and_whitespace_only_become_na():
    s = pd.Series([None, "", "   "])
    out = coerce_bool_from_tokens(s)
    exp = pd.Series([pd.NA, pd.NA, pd.NA], dtype="boolean")
    assert_boolean_series_equal(out, exp)
