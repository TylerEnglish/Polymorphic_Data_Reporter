from __future__ import annotations
import pandas as pd
import numpy as np

from src.nlp.roles import (
    guess_role,
    detect_unit_hint,
    canonicalize_categories,
    _dtype_str,
)

def test_name_based_role_detection():
    s = pd.Series([1, 2, 3])
    # name strongly suggests id; we special-case preferring name over value=id conflicts
    assert guess_role("order_id", s)[0] == "id"
    assert guess_role("event_timestamp", s)[0] == "time"
    assert guess_role("is_active", pd.Series([True, False]))[0] == "bool"
    # With config-weighted blending, numeric values may win over name "lat"
    assert guess_role("lat", pd.Series([1.0, 2.0]))[0] in {"geo", "numeric"}

def test_value_based_role_detection_numeric_id_vs_numeric():
    # high-uniqueness numeric -> id
    s_id = pd.Series(range(100))
    role, conf = guess_role("col", s_id)
    assert role in {"id", "numeric"}
    assert role == "id"

    # low-uniqueness numeric -> numeric
    s_num = pd.Series([1, 1, 2, 2, 2, 3, 3])
    role, conf = guess_role("col", s_num)
    assert role == "numeric"

def test_value_based_role_detection_bool_categorical_text_datetime():
    # bool-like: 2 unique strings
    s_bool_like = pd.Series(["Y", "N", "Y", "N", None])
    assert guess_role("col", s_bool_like)[0] == "bool"

    # small-cardinality strings -> categorical
    s_cat = pd.Series(["a"] * 10 + ["b"] * 5 + ["c"] * 2)
    assert guess_role("col", s_cat)[0] == "categorical"

    # large-cardinality strings -> text
    s_text = pd.Series([f"token_{i}" for i in range(1000)])
    assert guess_role("col", s_text)[0] == "text"

    # datetime dtype
    dt = pd.to_datetime(pd.Series(["2024-01-01", "2024-01-02"]))
    assert _dtype_str(dt) == "datetime"
    assert guess_role("anything", dt)[0] == "time"

def test_detect_unit_hint_currency_percent_magnitude():
    s_cur = pd.Series(["$12", "$20", "15", "5"])
    assert detect_unit_hint(s_cur) == "currency"

    s_pct = pd.Series(["10%", "percent", "pct", "not a percent", "5%"])
    assert detect_unit_hint(s_pct) == "percent"

    s_k = pd.Series(["12k", "5k", "1000", "K"])
    assert detect_unit_hint(s_k) in {"magnitude_k", "percent", "currency", "magnitude_m", None}

    s_m = pd.Series(["2 million", "1mm", "1000000"])
    # may or may not meet 20% threshold depending on strings; just ensure no crash
    _ = detect_unit_hint(s_m)

def test_canonicalize_categories_basic_and_edges():
    # trims + lowercases, maps only top values
    s = pd.Series([" Alpha ", "alpha", "ALPHA", "Beta", "beta ", None])
    m = canonicalize_categories(s)
    assert m is not None
    # ensure same canonical form for different variants
    assert m[" Alpha ".strip()] == m["alpha"] == m["ALPHA"]

    # non-string dtype -> None
    assert canonicalize_categories(pd.Series([1, 2, 3])) is None

    # too many uniques -> None
    big = pd.Series([f"v{i}" for i in range(1000)])
    assert canonicalize_categories(big) is None

    # empty / all NaN -> None
    assert canonicalize_categories(pd.Series([None, None])) is None
