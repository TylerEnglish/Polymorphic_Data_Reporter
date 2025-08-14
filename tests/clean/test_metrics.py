from __future__ import annotations
import numpy as np
import pandas as pd

from src.cleaning.metrics import profile_columns


def _df_and_schema():
    df = pd.DataFrame(
        {
            "num": [1, 2, 3, 4],
            "floaty": [1.0, 2.0, np.nan, 4.0],
            "txt": pd.Series(["yes", "no", "maybe", "True"], dtype="object"),
            # 2 parseable, 1 bad, 1 NA -> parse ratio 2/3 over non-nulls
            "dtstr": ["2024-01-02", "01/03/2024", "bad", pd.NA],
            "inc": [1, 2, 3, 4],
            "rand": [3, 1, 4, 2],
            "unhash": [[1], [1], [2], None],
            "dt": pd.to_datetime(
                ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]
            ),
        }
    )
    # Exactly one role=='time' so has_time_index == True
    schema_roles = {
        "num": "numeric",
        "floaty": "numeric",
        "txt": "text",
        "dtstr": "time",   # string dates to be parsed
        "inc": "numeric",
        "rand": "numeric",
        "unhash": "text",
        "dt": "text",      # datetime dtype but not marked as 'time'
    }
    fmts = ["%Y-%m-%d", "%m/%d/%Y"]
    return df, schema_roles, fmts


def test_profile_numeric_stats_and_monotonic():
    df, schema_roles, fmts = _df_and_schema()
    prof = profile_columns(df, schema_roles, fmts)

    # mean/std/iqr/min/max for numeric
    m = prof["num"]
    assert m["type"] in {"int", "float"}
    assert np.isclose(m["mean"], 2.5)
    assert np.isclose(m["std"], 1.2909944487358056)  # sample std for [1,2,3,4]
    assert np.isclose(m["iqr"], 1.5)                 # Q3(3.25) - Q1(1.75)
    assert np.isclose(m["min"], 1.0)
    assert np.isclose(m["max"], 4.0)

    # monotonic flag behavior
    assert prof["inc"]["is_monotonic_increasing"] is True
    assert prof["rand"]["is_monotonic_increasing"] is False


def test_profile_string_tokens_and_avg_len():
    df, schema_roles, fmts = _df_and_schema()
    prof = profile_columns(df, schema_roles, fmts)

    t = prof["txt"]
    assert t["type"] == "string"
    # yes, no, True are tokens; "maybe" is not -> 3/4 = 0.75
    assert np.isclose(t["bool_token_ratio"], 0.75)
    # avg length of ["yes"(3), "no"(2), "maybe"(5), "True"(4)] = 14/4 = 3.5
    assert np.isclose(t["avg_len"], 3.5)


def test_profile_datetime_parse_ratio_and_time_index():
    df, schema_roles, fmts = _df_and_schema()
    prof = profile_columns(df, schema_roles, fmts)

    # exactly one role=='time' => has_time_index True for all columns
    for c in df.columns:
        assert prof[c]["has_time_index"] is True

    # dtstr string column with 2 parseable entries out of 3 non-nulls -> 2/3
    r = prof["dtstr"]
    assert r["role"] == "time"
    assert np.isclose(r["datetime_parse_ratio"], 2 / 3, rtol=1e-6, atol=1e-8)

    # real datetime dtype: parse ratio should be 1.0
    assert r["type"] == "string"  # dtstr stays string
    assert prof["dt"]["type"] in {"datetime", "date"}
    assert np.isclose(prof["dt"]["datetime_parse_ratio"], 1.0)


def test_profile_handles_unhashables_missing_and_keys_exist():
    df, schema_roles, fmts = _df_and_schema()
    prof = profile_columns(df, schema_roles, fmts)

    # unhashable values handled via safe nunique
    u = prof["unhash"]
    assert u["nunique"] == 2
    assert u["cardinality"] == 2

    # missing percentage on floaty (1 NaN of 4)
    f = prof["floaty"]
    assert np.isclose(f["missing_pct"], 0.25)
    assert np.isclose(f["non_null_ratio"], 0.75)

    # essential keys exist for every column
    keys = {
        "name", "type", "role", "missing_pct", "non_null_ratio",
        "nunique", "unique_ratio", "cardinality", "avg_len",
        "has_time_index", "mean", "std", "iqr", "min", "max",
        "bool_token_ratio", "datetime_parse_ratio", "is_monotonic_increasing",
    }
    for c, info in prof.items():
        assert keys.issubset(info.keys())
