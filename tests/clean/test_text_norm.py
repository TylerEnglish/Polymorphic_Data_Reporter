from __future__ import annotations
import pandas as pd

from src.cleaning.rules_builtin.text_norm import (
    text_normalize,
    normalize_null_tokens,
)

def test_text_normalize_basic_strip_lower_nfkc_and_punct():
    s = pd.Series([
        "  Hello\u00A0world  ",      # NBSP + spaces
        "“Smart—quotes” & Café",     # curly quotes + em dash + accented char
        None,
        pd.NA,
    ])
    out = text_normalize(s, strip=True, lower=False)

    assert out.iloc[0] == "Hello world"             # collapsed + stripped
    assert out.iloc[1] == '"Smart-quotes" & Café'   # curly → ascii, dash → -, accent preserved via NFKC
    assert out.iloc[2] is None
    assert pd.isna(out.iloc[3])

    # With lower=True
    out2 = text_normalize(pd.Series(["  Foo   BAR  "]), strip=True, lower=True)
    assert out2.iloc[0] == "foo bar"

def test_text_normalize_passthrough_non_string_series():
    s_num = pd.Series([1, 2, 3])
    out = text_normalize(s_num)
    pd.testing.assert_series_equal(out, s_num)
    assert out is not s_num  # purity: new object

def test_normalize_null_tokens_defaults():
    s = pd.Series(["", "—", "n/a", "NaN", " value ", "NULL", None, pd.NA])
    out = normalize_null_tokens(s)  # default tokens include "", dashes, n/a, nan, null

    assert pd.isna(out.iloc[0])  # ""
    assert pd.isna(out.iloc[1])  # em dash
    assert pd.isna(out.iloc[2])  # n/a
    assert pd.isna(out.iloc[3])  # "NaN" → lower "nan"
    assert out.iloc[4] == "value"  # trimmed
    assert pd.isna(out.iloc[5])  # "NULL"
    assert pd.isna(out.iloc[6])  # None
    assert pd.isna(out.iloc[7])  # pd.NA

def test_normalize_null_tokens_custom_and_no_text_norm_first():
    s = pd.Series(["  Missing ", "keep", "NONE"])
    out = normalize_null_tokens(s, null_tokens={"Missing"}, apply_text_normalize_first=True)
    assert pd.isna(out.iloc[0])  # normalized strip makes "Missing" match
    assert out.iloc[1] == "keep"
    # case-insensitive default catches NONE
    assert pd.isna(out.iloc[2])
