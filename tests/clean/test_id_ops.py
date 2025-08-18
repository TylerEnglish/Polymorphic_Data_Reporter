import pandas as pd
from src.cleaning.rules_builtin.id_ops import zero_pad, keep_alnum


# --- tiny helpers to compare string series (like other rule test files) ---
def assert_series_equal_no_name(a: pd.Series, b: pd.Series):
    pd.testing.assert_series_equal(
        a.reset_index(drop=True),
        b.reset_index(drop=True),
        check_names=False,
    )

def assert_str_series_equal(a: pd.Series, b: pd.Series):
    assert a.dtype.name == "string"
    assert b.dtype.name == "string"
    assert_series_equal_no_name(a, b)


# ------------------------------ zero_pad -------------------------------------

def test_zero_pad_defaults_strings_and_ints():
    s = pd.Series(["7", "  42 ", "0003", None, 123])
    out = zero_pad(s)  # width=5
    exp = pd.Series(["00007", "00042", "00003", pd.NA, "00123"], dtype="string")
    assert_str_series_equal(out, exp)

def test_zero_pad_custom_fillchar_and_sign():
    s = pd.Series(["-7", "+9", "0"])
    out = zero_pad(s, width=4, fillchar="X")
    exp = pd.Series(["-XX7", "+XX9", "XXX0"], dtype="string")
    assert_str_series_equal(out, exp)

def test_zero_pad_width_smaller_keeps_original():
    s = pd.Series(["ABCDE", "12345", "999999"])
    out = zero_pad(s, width=5)
    exp = pd.Series(["ABCDE", "12345", "999999"], dtype="string")
    assert_str_series_equal(out, exp)

def test_zero_pad_numeric_only_mode():
    s = pd.Series([" INV-007 ", "A12B", "++-3", None])
    out = zero_pad(s, width=4, numeric_only=True)
    # " INV-007 " -> "0007", "A12B"->"0012", "++-3" -> sign+digits match is "-3" -> "-0003"
    exp = pd.Series(["0007", "0012", "-0003", pd.NA], dtype="string")
    assert_str_series_equal(out, exp)


# ------------------------------ keep_alnum ----------------------------------

def test_keep_alnum_removes_non_alnum_default():
    s = pd.Series(["A-1_2.3 /B", None, "###", "Åland 123"])
    out = keep_alnum(s)
    exp = pd.Series(["A123B", pd.NA, "", "Åland123"], dtype="string")
    assert_str_series_equal(out, exp)

def test_keep_alnum_allowed_and_collapse_runs():
    s = pd.Series(["A--B---C", "A - B -- C", "A__B____C"])
    out = keep_alnum(s, allowed={"-", "_"})
    exp = pd.Series(["A-B-C", "A-B-C", "A_B_C"], dtype="string")
    assert_str_series_equal(out, exp)

def test_keep_alnum_case_and_empty_to_na():
    s = pd.Series([" ab-12 ", "!!!", "", None])
    out = keep_alnum(s, allowed={"-"}, case="upper", empty_to_na=True, strip=True)
    exp = pd.Series(["AB-12", pd.NA, pd.NA, pd.NA], dtype="string")
    assert_str_series_equal(out, exp)

def test_keep_alnum_unicode_letters_preserved():
    s = pd.Series(["münchen-βeta_42"])
    out = keep_alnum(s, allowed={"-","_"})
    exp = pd.Series(["münchen-βeta_42"], dtype="string")
    assert_str_series_equal(out, exp)
