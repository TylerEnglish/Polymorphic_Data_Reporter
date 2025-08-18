import pandas as pd

from src.cleaning.rules_builtin.categorical import (
    consolidate_rare_categories,
    map_category_synonyms,
    extract_category_key,
    extract_category_anchor,
    fuzzy_map_categories,
)

def assert_str_series_equal(a: pd.Series, b: pd.Series):
    # normalize dtype to string for fair compare; keep NA mask
    a = a.astype("string"); b = b.astype("string")
    pd.testing.assert_series_equal(a.reset_index(drop=True), b.reset_index(drop=True), check_names=False)

def test_consolidate_rare_categories_basic():
    s = pd.Series(["A", "A", "B", "C", "C", "D", None, "A"])
    out = consolidate_rare_categories(s, min_freq=0.20, other_label="Other")
    # A=3/7 ~ 0.428 keep; C=2/7 ~ 0.285 keep; B=1/7 rare; D=1/7 rare; None stays None
    exp = pd.Series(["A", "A", "Other", "C", "C", "Other", None, "A"], dtype="string").astype("category")
    assert set(out.cat.categories) == {"A", "C", "Other"}
    assert list(out.astype("string")) == list(exp.astype("string"))

def test_map_category_synonyms_case_insensitive_and_strip():
    s = pd.Series([" us ", "U.S.", "USA", "United States", None])
    mapping = {"us": "USA", "u.s.": "USA", "united states": "USA"}
    out = map_category_synonyms(s, mapping=mapping, case_insensitive=True, strip=True)
    exp = pd.Series(["USA", "USA", "USA", "USA", None], dtype="string")
    assert_str_series_equal(out, exp)

def test_extract_category_key_regex_simple():
    s = pd.Series(["Arkansas McDonals", "72901 McDonalds", "ma.McDonals", "Drive to mcdonalds?"])
    # allow common misspell variants
    patt = [r"\bmc(?:donalds?|donals?)\b"]
    out = extract_category_key(s, patterns=patt, case_insensitive=True)
    exp = pd.Series(["McDonals", "McDonalds", "McDonals", "mcdonalds"], dtype="string")
    assert_str_series_equal(out, exp)

def test_extract_category_anchor_dominant_token():
    s = pd.Series(["Mcdonals", "Arkansas McDonalds", "72901 McDonals", "ma.McDonals"])
    out = extract_category_anchor(s, allow_digits=False)
    # All rows should choose "mcdonals" (lowercased anchor) because it's the globally-most-frequent token
    exp = pd.Series(["mcdonals", "mcdonalds", "mcdonals", "mcdonals"], dtype="string")
    # Note: second row contains "mcdonalds" token; the function preserves the token form from the row.
    assert_str_series_equal(out, exp)

def test_fuzzy_map_categories_corrects_misspelling():
    s = pd.Series(["mcdonals", "mcdonalds", "mcdonaldd", "mc_donalds"])
    out = fuzzy_map_categories(s, candidates=["McDonalds"], case_insensitive=True, max_distance=2)
    exp = pd.Series(["McDonalds", "McDonalds", "McDonalds", "McDonalds"], dtype="string")
    assert_str_series_equal(out, exp)

def test_end_to_end_pipeline_regex_then_fuzzy():
    s = pd.Series(["Mcdonals", "Arkansas McDonalds", "72901 McDonals", "ma.McDonals", None])
    patt = [r"\bmc(?:donalds?|donals?)\b"]
    keys = extract_category_key(s, patterns=patt, case_insensitive=True)
    canon = fuzzy_map_categories(keys, candidates=["McDonalds"], case_insensitive=True, max_distance=2)
    exp = pd.Series(["McDonalds", "McDonalds", "McDonalds", "McDonalds", None], dtype="string")
    assert_str_series_equal(canon, exp)

def test_end_to_end_anchor_then_fuzzy():
    s = pd.Series(["Mcdonals", "Arkansas McDonalds", "72901 McDonals", "ma.McDonals"])
    anchors = extract_category_anchor(s, allow_digits=False)
    canon = fuzzy_map_categories(anchors, candidates=["McDonalds"], case_insensitive=True, max_distance=2)
    exp = pd.Series(["McDonalds", "McDonalds", "McDonalds", "McDonalds"], dtype="string")
    assert_str_series_equal(canon, exp)

def test_extract_category_key_capture_group_and_default():
    s = pd.Series(["City: Springfield (IL)", "City: Boston (MA)", "No City"])
    # capture group inside parentheses
    patt = [r"City:\s*([A-Za-z ]+)\s*\([A-Z]{2}\)"]
    out = extract_category_key(s, patterns=patt, capture_group=1, default="Unknown")
    exp = pd.Series(["Springfield ", "Boston ", "Unknown"], dtype="string")
    assert_str_series_equal(out, exp)

def test_handles_categorical_dtype_inputs():
    s = pd.Series(["Mcdonals", "Arkansas McDonalds", "72901 McDonals"], dtype="category")
    anchors = extract_category_anchor(s)
    assert anchors.dtype.name == "string"
    canon = fuzzy_map_categories(anchors, candidates=["McDonalds"])
    assert all(c == "McDonalds" for c in canon.dropna().unique())
