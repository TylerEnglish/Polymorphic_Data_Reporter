import pandas as pd
import re
import math

from src.cleaning.rules_builtin.regex_ops import regex_replace, extract_digits


# ---------------- helpers ----------------

def assert_series_equal_no_name(a: pd.Series, b: pd.Series):
    pd.testing.assert_series_equal(
        a.reset_index(drop=True),
        b.reset_index(drop=True),
        check_names=False
    )

def assert_str_series_equal(a: pd.Series, b: pd.Series):
    assert a.dtype.name == "string"
    assert b.dtype.name == "string"
    assert_series_equal_no_name(a, b)


# ---------------- tests ------------------

def test_regex_replace_basic_and_flags():
    s = pd.Series([" Foo ", "foo\nbar", None])
    # case-sensitive (no change for " Foo ")
    out1 = regex_replace(s, pattern=r"foo", repl="X")
    exp1 = pd.Series([" Foo ", "X\nbar", pd.NA], dtype="string")
    assert_str_series_equal(out1, exp1)

    # case-insensitive
    out2 = regex_replace(s, pattern=r"foo", repl="X", flags="I")
    exp2 = pd.Series([" X ", "X\nbar", pd.NA], dtype="string")
    assert_str_series_equal(out2, exp2)

    # multiline: replace 'bar' only if at start of line
    out3 = regex_replace(pd.Series(["a\nbar\nz"]), pattern=r"^bar$", repl="XXX", flags="M")
    exp3 = pd.Series(["a\nXXX\nz"], dtype="string")
    assert_str_series_equal(out3, exp3)


def test_regex_replace_backrefs_named_groups_and_count():
    s = pd.Series(["A-12-345", "B-9-8"])
    out = regex_replace(
        s,
        pattern=r"(?P<L>[A-Z])-(?P<X>\d+)-(?P<Y>\d+)",
        repl=r"\g<L>:\g<X>|\g<Y>",
    )
    exp = pd.Series(["A:12|345", "B:9|8"], dtype="string")
    assert_str_series_equal(out, exp)

    # count=1 should only replace first 'o' in each string
    s2 = pd.Series(["foo", "ooo"])
    out2 = regex_replace(s2, pattern=re.compile("o"), repl="X", count=1)
    exp2 = pd.Series(["fXo", "oXo"], dtype="string")
    assert_str_series_equal(out2, exp2)


def test_regex_replace_literal_and_compiled_list():
    s = pd.Series(["(abc) [abc] {abc}"])
    # literal mode makes the brackets taken as exact characters
    out = regex_replace(s, pattern="(abc)", repl="[ABC]", literal=True)
    exp = pd.Series(["[ABC] [abc] {abc}"], dtype="string")
    assert_str_series_equal(out, exp)

    # multiple patterns sequentially
    pats = [re.compile(r"\[abc\]"), r"\{abc\}"]
    reps = ["[XYZ]", "{XYZ}"]
    out2 = regex_replace(s, pattern=pats, repl=reps)
    exp2 = pd.Series(["(abc) [XYZ] {XYZ}"], dtype="string")
    assert_str_series_equal(out2, exp2)


def test_extract_digits_unicode_and_ascii_modes():
    s = pd.Series(["A-1B2C3", "١٢٣", None])  # "١٢٣" are Arabic-Indic digits
    out = extract_digits(s)  # unicode-aware by default
    exp = pd.Series(["123", "١٢٣", pd.NA], dtype="string")
    assert_str_series_equal(out, exp)

    out2 = extract_digits(s, ascii_only=True, empty_to_na=True)
    exp2 = pd.Series(["123", pd.NA, pd.NA], dtype="string")
    assert_str_series_equal(out2, exp2)


def test_extract_digits_with_sign_and_decimal():
    s = pd.Series(["  +12.34% ", "(56.78)", "-9,001", "no digits"])
    out = extract_digits(s, keep_sign=True, keep_decimal=True, ascii_only=True, empty_to_na=True)
    # We don't interpret parentheses here; just character filtering:
    # "+12.34" , "56.78" , "-9001" , <NA>
    exp = pd.Series(["+12.34", "56.78", "-9001", pd.NA], dtype="string")
    assert_str_series_equal(out, exp)


def test_non_string_input_returns_copy():
    s = pd.Series([1, 2, 3], dtype="int64")
    out1 = regex_replace(s, pattern=r"\d", repl="X")  # should return copy unchanged
    out2 = extract_digits(s)                          # should return copy unchanged
    pd.testing.assert_series_equal(out1, s)
    pd.testing.assert_series_equal(out2, s)
