from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import numpy as np
import pytest

from src.config_model.model import RootCfg
from src.cleaning.recheck import recheck_silver


@pytest.fixture()
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """
    Create an isolated project root:
      - config/config.toml  (with essential cleaning rules)
      - schemas/cakes.schema.toml (frozen roles)
      - data/silver/cakes/dataset.parquet (messy cake data)
    """
    root = tmp_path
    monkeypatch.chdir(root)  # important: recheck() uses Path.cwd()
    (root / "config").mkdir(parents=True, exist_ok=True)
    (root / "schemas").mkdir(parents=True, exist_ok=True)
    (root / "data" / "silver" / "cakes").mkdir(parents=True, exist_ok=True)
    (root / "data" / "raw").mkdir(parents=True, exist_ok=True)
    (root / "data" / "gold").mkdir(parents=True, exist_ok=True)

    # Minimal TOML with the key rules the re-check relies on.
    cfg_toml = f"""
[env]
project_name = "Polymorphic Data Reporter"
timezone = "America/Chicago"
seed = 42
theme = "dark_blue"

[storage.local]
enabled = true
raw_root = "data/raw"
gold_root = "data/gold"

[storage.minio]
enabled = false

[profiling.roles]
cat_cardinality_max = 120
datetime_formats = ["%Y-%m-%d", "%m/%d/%Y", "%Y-%m-%d %H:%M:%S"]

[profiling.outliers]
method = "zscore"
zscore_threshold = 3.0
iqr_multiplier = 1.5

[cleaning.columns]
drop_missing_pct = 0.90
min_unique_ratio = 0.001
always_keep = ["id", "date", "timestamp"]
cat_cardinality_max = 200

[cleaning.normalize]
strip_text = true
lowercase_text = true
standardize_dates = true
enforce_categorical = true

[cleaning.impute]
numeric_default = "median"
categorical_default = "Unknown"
text_default = "N/A"
time_aware_interpolation = true

[cleaning.outliers]
method = "zscore"
zscore_threshold = 3.0
iqr_multiplier = 1.5
handle = "flag"
winsor_limits = [0.01, 0.99]

[cleaning.normalize_null_tokens]
null_tokens = ["", "NA", "N/A", "None", "NULL", "NaN", "-", "<NA>"]

[cleaning.datetime]
utc = true
dayfirst = false
yearfirst = false
formats = ["%Y-%m-%d", "%m/%d/%Y", "%Y-%m-%d %H:%M:%S"]

[cleaning.regex]
unicode_minus = '[-−-–—]'
thousands_between_digits = '(?<=\\d)[,_](?=\\d{3}\\b)'

# ---- rules (trimmed to just what we need for the test) ----
[[cleaning.rules]]
id = "normalize-nulls-first"
priority = 130
when = 'type == "string"'
then = 'normalize_null_tokens(null_tokens=cleaning.normalize_null_tokens.null_tokens, case_insensitive=true)'

[[cleaning.rules]]
id = "num:normalize-unicode-minus"
priority = 125
when = 'type == "string" and role in ["numeric","id"]'
then = 'regex_replace(pattern=cleaning.regex.unicode_minus, repl="-")'

[[cleaning.rules]]
id = "num:strip-thousands"
priority = 123
when = 'type == "string" and role in ["numeric","id"]'
then = 'regex_replace(pattern=cleaning.regex.thousands_between_digits, repl="")'

[[cleaning.rules]]
id = "coerce-numeric"
priority = 120
when = 'role == "numeric" and type == "string" and not icontains(name,"id")'
then = "coerce_numeric()"

[[cleaning.rules]]
id = "bool:tokens"
priority = 126
when = 'bool_token_ratio >= 0.57 or istartswith(name,"bool_")'
then = "coerce_bool()"

[[cleaning.rules]]
id = "cat:cast-small"
priority = 90
when = "role == 'categorical' and cardinality <= cleaning.columns.cat_cardinality_max"
then = "cast_category()"

[[cleaning.rules]]
id = "dt:parse-formats"
priority = 90
when = "role == 'time' and type == 'string'"
then = "parse_datetime(datetime_formats)"

[[cleaning.rules]]
id = "dt:parse-epoch"
priority = 86
when = "role == 'time' and type in ['int','float']"
then = "parse_epoch()"

[[cleaning.rules]]
id = "units:percent-by-range"
priority = 74
when = "role == 'numeric' and min >= 0 and max <= 100 and max > 1"
then = "standardize_units('percent')"

[[cleaning.rules]]
id = "units:percent-by-name"
priority = 76
when = "role == 'numeric' and (icontains(name, 'percent') or icontains(name, 'pct'))"
then = "standardize_units('percent')"

[topics.thresholds]
min_corr_for_scatter = 0.35
min_slope_for_trend = 0.02
max_categories_bar = 20
max_series_line = 8
max_charts_total = 12
"""
    (root / "config" / "config.toml").write_text(cfg_toml, encoding="utf-8")

    # Frozen schema for cakes (roles drive the rules in the re-check).
    schema_toml = """
dataset_slug = "cakes"
schema_confidence = 1.0


[[columns]]
name = "order_id"
dtype = "int64"
  [columns.role_confidence]
  role = "id"
  confidence = 1.0

[[columns]]
name = "baked_at"
dtype = "string"
  [columns.role_confidence]
  role = "time"
  confidence = 1.0

[[columns]]
name = "cake_type"
dtype = "string"
  [columns.role_confidence]
  role = "categorical"
  confidence = 1.0

[[columns]]
name = "slices_sold"
dtype = "string"
  [columns.role_confidence]
  role = "numeric"
  confidence = 1.0

[[columns]]
name = "price_usd"
dtype = "string"
  [columns.role_confidence]
  role = "numeric"
  confidence = 1.0

[[columns]]
name = "percent_icing"
dtype = "int64"
  [columns.role_confidence]
  role = "numeric"
  confidence = 1.0

[[columns]]
name = "bool_returned"
dtype = "string"
  [columns.role_confidence]
  role = "boolean"
  confidence = 1.0

[[columns]]
name = "zip_code"
dtype = "string"
  [columns.role_confidence]
  role = "id"
  confidence = 1.0
"""
    (root / "schemas" / "cakes.schema.toml").write_text(schema_toml, encoding="utf-8")

    # Messy cake data (strings, unicode minus, thousands separators, booleans as tokens, etc.)
    # Note: keep 'baked_at' as strings that match formats, to exercise dt parsing.
    data = pd.DataFrame(
        {
            "order_id": [1, 2, 3, 4, 5],
            "baked_at": ["2024-01-02", "2024-01-03", "2024-01-04", "2024-02-01", "2024-02-02"],
            "cake_type": ["Chocolate ", " vanilla", "Red Velvet", "CHOCOLATE", "vanilla  "],
            "slices_sold": ["1,234", "25", "100", "2,500", "0"],
            # Unicode minus on first row (U+2212), thousands on third.
            "price_usd": ["−12.99", "8", "1,234.50", "0", "19.99"],
            # Integers in [0,100] should standardize to 0..1
            "percent_icing": [25, 80, 100, 0, 50],
            # Name starts with bool_ to guarantee the bool rule triggers
            "bool_returned": ["Yes", "no", "TRUE", "0", "False"],
            "zip_code": ["7", "60601", "02139", " 94107", "77002"],
        }
    )

    # Write silver dataset
    silver_path = root / "data" / "silver" / "cakes" / "dataset.parquet"
    data.to_parquet(silver_path, index=False)

    return root


def test_recheck_cakes(project: Path):
    # Load config (RootCfg resolves paths relative to the config dir)
    cfg = RootCfg.load(project / "config" / "config.toml")

    # Run re-checker
    df_rechecked, info = recheck_silver(cfg, slug="cakes")

    # --- Files were written
    rechecked_path = project / "data" / "silver" / "cakes" / "dataset.rechecked.parquet"
    schema_report = project / "data" / "silver" / "cakes" / "schema_report.parquet"
    assert rechecked_path.exists(), "dataset.rechecked.parquet was not written"
    assert schema_report.exists(), "schema_report.parquet was not refreshed"

    # --- Shape + no drops
    df_orig = pd.read_parquet(project / "data" / "silver" / "cakes" / "dataset.parquet")
    assert set(df_orig.columns).issubset(
        set(df_rechecked.columns)
    ), "Re-check must not drop original columns"

    # --- Numeric coercions
    # slices_sold: "1,234" -> 1234, etc.
    assert pd.api.types.is_numeric_dtype(df_rechecked["slices_sold"]), "slices_sold should be numeric after re-check"
    # price_usd: "−12.99" (unicode minus) -> -12.99, "1,234.50" -> 1234.50
    assert pd.api.types.is_numeric_dtype(df_rechecked["price_usd"]), "price_usd should be numeric after re-check"
    first_price = float(df_rechecked.loc[0, "price_usd"])
    third_price = float(df_rechecked.loc[2, "price_usd"])
    assert np.isclose(first_price, -12.99, atol=1e-6)
    assert np.isclose(third_price, 1234.50, atol=1e-6)

    # --- Percent standardization to 0..1
    assert df_rechecked["percent_icing"].max() <= 1.0 + 1e-9
    assert df_rechecked["percent_icing"].min() >= 0.0 - 1e-9

    # --- Boolean coercion
    br = df_rechecked["bool_returned"]
    # Accept either boolean dtype or 0/1 ints; coerce to numeric and check set
    br_as_num = pd.to_numeric(br.astype(str).str.replace("True", "1").str.replace("False", "0"), errors="coerce")
    uniq = set(int(x) for x in br_as_num.dropna().unique())
    assert uniq.issubset({0, 1}), f"bool_returned should map to 0/1 or True/False, got {uniq}"

    # --- Datetime parsing
    assert pd.api.types.is_datetime64_any_dtype(
        df_rechecked["baked_at"]
    ), "baked_at should be parsed to datetime64"

    # --- Categorical tightening (case/whitespace normalized; may be cast to category)
    # Make sure categories are normalized lower/trimmed per rule set
    normalized = df_rechecked["cake_type"].astype(str).str.strip().str.lower().unique().tolist()
    assert "chocolate" in normalized and "vanilla" in normalized

    # --- Info object sanity
    assert info.slug == "cakes"
    assert Path(info.rechecked_path).exists()
    assert Path(info.schema_report_path).exists()
    assert info.rows == len(df_rechecked)
    assert info.cols == df_rechecked.shape[1]
    assert 0.0 <= info.schema_conf_after <= 1.0
    assert 0.0 <= info.avg_role_conf_after <= 1.0
