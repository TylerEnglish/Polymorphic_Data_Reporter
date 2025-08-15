---

config.toml:

```
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
enabled = true
endpoint = "http://minio:9000"
access_key = "admin"
secret_key = "admin"
secure = false
bucket = "datasets"
raw_prefix = "raw/"
gold_prefix = "gold/"

[auth]
# If Airflow UI or any optional service needs creds in-app
username = "admin"
password = "admin"

[sources]
# discover from both local and MinIO
locations = [
  "file://data/raw/sales_demo",
  "s3://datasets/raw/iot_sensors/"
]

[duckdb]
persist = true  # physical .duckdb written under data/gold/<slug>/tables/

[profiling.roles]
cat_cardinality_max = 120
datetime_formats = ["%Y-%m-%d", "%m/%d/%Y", "%Y-%m-%d %H:%M:%S"]

[profiling.outliers]
method = "zscore"         # zscore | iqr
zscore_threshold = 3.0
iqr_multiplier = 1.5

# ---------- NLP / NLG ----------

[nlp]
sample_rows = 20000
min_schema_confidence = 0.90
min_role_confidence = 0.90
max_iter = 1000
min_improvement = 0.0
enable_domain_templates = true
granularity = "slug"  # slug | subdir | file

[nlp.role_scoring]
name_weight = 0.45
value_weight = 0.55

# thresholds
bool_token_min_ratio = 0.57     # share of values that look boolean
date_parse_min_ratio = 0.60     # share of values that parse as dates
unique_id_ratio = 0.95          # uniqueness ratio to consider "id-ish"
categorical_max_unique_ratio = 0.02
text_min_avg_len = 8            # avg string length to lean "text" vs "categorical"
min_non_null_ratio = 0.10       # ignore columns with less data than this

# bonuses/penalties
bonus_id_name = 0.10            # extra if name strongly looks like id (e.g., *_id)
penalize_bool_for_many_tokens = 0.08  # slight penalty if too many distinct tokens for bool

[nlg.constants]
inventory_key = "_inventory"
narrative_filename = "narrative.txt"

# ---------- Dynamic Cleaning Policy (highly flexible) ----------

[cleaning.columns]
drop_missing_pct = 0.90
min_unique_ratio = 0.001
always_keep = ["id", "date", "timestamp"]
cat_cardinality_max = 200           # enforce category dtype under this

[cleaning.normalize]
strip_text = true
lowercase_text = false
standardize_dates = true
enforce_categorical = true

[cleaning.impute]
numeric_default = "median"          # mean|median|ffill|bfill|interpolate
categorical_default = "Unknown"
text_default = "N/A"
time_aware_interpolation = true

[cleaning.outliers]
detect = "zscore"                   # zscore|iqr
zscore_threshold = 3.0
iqr_multiplier = 1.5
handle = "flag"                     # flag|winsorize|drop
winsor_limits = [0.01, 0.99]

# Rulepacks: ordered lists of conditional rules (mini DSL)
# Builtins are loaded automatically; you can add/override here.
[cleaning.normalize_null_tokens]
null_tokens = ["", "NA", "N/A", "N\\A", "None", "NULL", "NaN", "-", "—", "<NA>", "<Null>", "<None>", "nil", "missing"]

[[cleaning.rules]]
id = "normalize-nulls-first"
priority = 120
when = 'type == "string"'
then = 'normalize_null_tokens(null_tokens=cleaning.normalize_null_tokens.null_tokens, case_insensitive=true)'

[[cleaning.rules]]
id = "coerce-numeric"
priority = 100
when = 'role == "numeric" and type == "string"'
then = 'coerce_numeric()'

[[cleaning.rules]]
id = "coerce-numeric-from-string-heuristic"
priority = 110   # bump above parse/impute to ensure it runs early
when = 'type == "string" and role != "time" and avg_len <= 24 and missing_pct < 0.99'
then = 'coerce_numeric()'

[[cleaning.rules]]
id = "parse-datetime"
priority = 100
when = 'role == "time" and type == "string"'
then = 'parse_datetime(datetime_formats)'

[[cleaning.rules]]
id = "impute-numeric-any"
priority = 95
when = 'missing_pct > 0 and (type in ["int","float"] or role in ["numeric","id"])'
then = 'impute(numeric_default)'

[[cleaning.rules]]
id = "impute-numeric-time"
priority = 92
when = 'role == "numeric" and missing_pct > 0 and has_time_index'
then = 'impute("interpolate")'

[[cleaning.rules]]
id = "impute-numeric-default"
priority = 80
when = 'role == "numeric" and missing_pct > 0'
then = 'impute(numeric_default)'

[[cleaning.rules]]
id = "impute-categorical"
priority = 80
when = 'role == "categorical" and type == "string" and missing_pct > 0'
then = 'impute_value(categorical_default)'

[[cleaning.rules]]
id = "impute-text"
priority = 55
when = 'role == "text" and type == "string" and missing_pct > 0 and avg_len >= 8'
then = 'impute_value(text_default)'

[[cleaning.rules]]
id = "flag-outliers"
priority = 70
when = 'role == "numeric"'
then = 'outliers(detect, zscore_threshold, iqr_multiplier, handle, winsor_limits)'

[[cleaning.rules]]
id = "normalize-text"
priority = 50
when = 'role == "text" and type == "string"'
then = 'text_normalize(strip=cleaning.normalize.strip_text, lower=cleaning.normalize.lowercase_text)'

[[cleaning.rules]]
id = "enforce-categorical"
priority = 50
when = 'role == "categorical" and type == "string" and cardinality <= cleaning.columns.cat_cardinality_max'
then = 'cast_category()'

[[cleaning.rules]]
id = "ensure-text-string-dtype"
priority = 49
when = 'role == "text"'
then = 'cast_string()'

[[cleaning.rules]]
id = "impute-time-default"
priority = 80
when = 'role == "time" and missing_pct > 0'
then = 'impute("ffill")'   # or "bfill"

[[cleaning.rules]]
id = "impute-time-forward"
priority = 90
when = 'role == "time" and missing_pct > 0 and has_time_index'
then = 'impute_dt("ffill")'

[[cleaning.rules]]
id = "impute-time-median"
priority = 80
when = 'role == "time" and missing_pct > 0'
then = 'impute_dt("median")'

[[cleaning.rules]]
id = "materialize-missing-text-literal"
priority = 44
when = 'type == "string" and missing_pct > 0'
then = 'materialize_missing_as(cleaning.impute.text_default)'

[[cleaning.rules]]
id = "final-null-guard-text"
priority = 46  # runs after your other imputations (50+), before drops (40)
when = 'type == "string" and missing_pct > 0'
then = 'impute_value(text_default)'

[[cleaning.rules]]
id = "normalize-nulls-last"
priority = 45
when = 'type == "string"'
then = 'normalize_null_tokens(null_tokens=cleaning.normalize_null_tokens.null_tokens, case_insensitive=true, apply_text_normalize_first=false)'

[[cleaning.rules]]
id = "drop-sparse"
priority = 40
when = 'missing_pct >= cleaning.columns.drop_missing_pct and name notin cleaning.columns.always_keep'
then = 'drop_column()'

[[cleaning.rules]]
id = "drop-constant"
priority = 40
when = 'unique_ratio <= cleaning.columns.min_unique_ratio and name notin cleaning.columns.always_keep'
then = 'drop_column()'

[[cleaning.rules]]
id = "parse-datetime-by-name-heuristic"
priority = 100
when = 'type == "string" and (icontains(name,"date") or icontains(name,"time") or icontains(name,"dt_"))'
then = 'parse_datetime(datetime_formats)'

# ---------- Topic Selection & Charts ----------

[topics.thresholds]
min_corr_for_scatter = 0.35
min_slope_for_trend = 0.02
max_categories_bar = 20
max_series_line = 8
max_charts_total = 12

[charts]
max_charts_per_topic = 6
facet_max_series = 8
topk_categories = 20
prefer_small_multiples = true
allow_pie_when_n_le = 5
enable_advanced = ["treemap","sankey","calendar_heatmap","parcoords"]
export_static_png = true

[weights]
suitability = 3.0
effect_size = 2.5
signal_quality = 2.0
readability = 1.5
complexity = 1.0

# ---------- Output & Orchestration ----------

[reports.enabled_generators]
csv = true
json = true
parquet = true
charts = true
html = true
pdf = true

[reports.html]
template = "base.html"
title_prefix = "Report:"
embed_interactive = true

[reports.pdf]
engine = "chromium"          # chromium
page_size = "Letter"
margins = "0.5in"

[airflow]
dag_id = "polymorphic_report_dag"
schedule = "0 2 * * *"
max_active_runs = 1
catchup = false
concurrency = 8
task_retries = 2
username = "admin"
password = "admin"

[docker]
compose_file = "docker/docker-compose.yml"
airflow_image = "local/airflow:latest"
chrome_image = "local/chrome:latest"

[logging]
level = "INFO"
structured_json = true

[publishing]
enabled = false
target = "s3://datasets/gold-exports/"
access_key = "admin"
secret_key = "admin"

```

---
script code

code script:
```
from __future__ import annotations
import argparse, json
from pathlib import Path
from copy import deepcopy
from collections import Counter
from typing import Optional, List, Tuple
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype

# project imports
from src.config_model.model import RootCfg
from src.cleaning.engine import run_clean_pass
from src.cleaning.rescore import rescore_after_clean
from src.nlp.roles import _dtype_str, guess_role
from src.nlp.schema import ProposedSchema, ColumnSchema, ColumnHints, RoleConfidence
from src.nlp.schema_io import to_toml, schema_path_for_slug

# Try to use your IO stack, but fall back to pandas readers
try:
    from src.io.storage import Storage  # type: ignore
    from src.io.catalog import Catalog  # type: ignore
    from src.io.readers import read_any  # type: ignore
except Exception:  # pragma: no cover
    Storage = None
    Catalog = None
    read_any = None

SUPPORTED_EXTS = {".csv", ".parquet", ".json", ".ndjson", ".xlsx", ".xls"}

# ------------------------ IO helpers ------------------------

def _bronze_dir(root: Path, slug: str) -> Path:
    return root / "data" / "bronze" / slug

def _silver_dir(root: Path, slug: str) -> Path:
    return root / "data" / "silver" / slug

def _bootstrap_path(bronze_dir: Path) -> Path:
    return bronze_dir / "nlp_bootstrap.json"

def _normalized_log_path(bronze_dir: Path) -> Path:
    return bronze_dir / "normalized.json"

def _load_bootstrap(bootstrap_json: Path) -> dict:
    if not bootstrap_json.exists():
        return {"dataset": "", "schema_confidence": 0.0, "columns": []}
    try:
        return json.loads(bootstrap_json.read_text(encoding="utf-8"))
    except Exception:
        return {"dataset": "", "schema_confidence": 0.0, "columns": []}

def _save_bootstrap(bootstrap_json: Path, data: dict) -> None:
    bootstrap_json.write_text(json.dumps(data, indent=2), encoding="utf-8")

def _save_schema_toml(project_root: Path, slug: str, proposed: ProposedSchema) -> Path:
    out_path = schema_path_for_slug(project_root, slug)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path_fsys = Path(out_path)
    out_path_fsys.parent.mkdir(parents=True, exist_ok=True)
    out_path_fsys.write_text(to_toml(proposed.to_dict()), encoding="utf-8")
    return out_path_fsys

def _append_normalized_log(path: Path, record: dict) -> None:
    logs = []
    if path.exists():
        try:
            logs = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(logs, list):
                logs = []
        except Exception:
            logs = []
    logs.append(record)
    path.write_text(json.dumps(logs, indent=2), encoding="utf-8")

# ------------------------ Raw ingestion (same behavior as run_clean_once) ------------------------

def _discover_uris(project_root: Path, slug: str) -> List[str]:
    uris: List[str] = []
    try:
        if Storage and Catalog:
            storage = Storage()
            entries = Catalog(storage).inventory(dataset_slug=slug)
            uris = [e.uri for e in entries]  # type: ignore[attr-defined]
    except Exception:
        uris = []
    if not uris:
        raw_dir = project_root / "data" / "raw" / slug
        if raw_dir.exists():
            for p in raw_dir.rglob("*"):
                if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
                    uris.append(str(p))
    return uris

def _read_one(uri: str) -> Optional[pd.DataFrame]:
    if read_any and Storage:
        try:
            df = read_any(Storage(), uri, backend="pandas")  # type: ignore
            return df if isinstance(df, pd.DataFrame) else pd.DataFrame(df)
        except Exception:
            pass
    p = Path(uri)
    ext = p.suffix.lower()
    try:
        if ext == ".parquet":
            return pd.read_parquet(uri)
        if ext == ".csv":
            return pd.read_csv(uri)
        if ext in {".json", ".ndjson"}:
            try:
                return pd.read_json(uri, lines=True)
            except ValueError:
                return pd.read_json(uri)
        if ext in {".xlsx", ".xls"}:
            return pd.read_excel(uri)
    except Exception:
        return None
    return None

def _best_datetime_parse(obj_series: pd.Series) -> pd.Series:
    s = obj_series
    def _plausible(ts: pd.Series) -> pd.Series:
        try:
            years = ts.dt.year
            return ts.where(years.between(1970, 2100))
        except Exception:
            return ts
    try:
        cand_a = pd.to_datetime(s, errors="coerce"); cand_a = _plausible(cand_a)
    except Exception:
        cand_a = pd.Series(pd.NaT, index=s.index)
    try:
        nums = pd.to_numeric(s, errors="coerce")
        cand_b = pd.to_datetime(nums, unit="ms", errors="coerce"); cand_b = _plausible(cand_b)
    except Exception:
        cand_b = pd.Series(pd.NaT, index=s.index)
    try:
        nums = pd.to_numeric(s, errors="coerce")
        cand_c = pd.to_datetime(nums, unit="s", errors="coerce"); cand_c = _plausible(cand_c)
    except Exception:
        cand_c = pd.Series(pd.NaT, index=s.index)
    candidates = [cand_a, cand_b, cand_c]
    nn_counts = [int(c.notna().sum()) for c in candidates]
    best_idx = int(np.argmax(nn_counts))
    non_null = int(s.notna().sum())
    if non_null == 0:
        return candidates[best_idx]
    if nn_counts[best_idx] / max(1, non_null) >= 0.60:
        return candidates[best_idx].dt.tz_localize(None)
    return pd.Series(pd.NaT, index=s.index)

def _prepare_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        s = out[col]
        if pd.api.types.is_datetime64_any_dtype(s) or pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
            continue
        if isinstance(s.dtype, CategoricalDtype):
            out[col] = s.astype("string"); continue
        if s.dtype == object:
            if any(k in str(col).lower() for k in ("date", "time", "timestamp", "dt", "ts")):
                parsed = _best_datetime_parse(s)
                if parsed.notna().any():
                    out[col] = parsed; continue
            nums = pd.to_numeric(s, errors="coerce")
            if nums.notna().mean() >= 0.80:
                out[col] = nums; continue
            def _to_text(x):
                if x is None or (isinstance(x, float) and np.isnan(x)): return None
                if isinstance(x, (dict, list, tuple)):
                    try: return json.dumps(x, ensure_ascii=False)
                    except Exception: return str(x)
                return str(x)
            try:
                out[col] = s.map(_to_text).astype("string")
            except Exception:
                out[col] = s.astype("string")
            continue
        try:
            out[col] = out[col].astype("string")
        except Exception:
            pass
    return out

def _ensure_bronze(df: pd.DataFrame, bronze_dir: Path):
    bronze_dir.mkdir(parents=True, exist_ok=True)
    out = bronze_dir / "normalize.parquet"
    df_arrow = _prepare_for_parquet(df)
    leftover_obj = [c for c in df_arrow.columns if df_arrow[c].dtype == object]
    if leftover_obj:
        print(f"[bronze] warning: object dtype columns remain → {leftover_obj}")
    df_arrow.to_parquet(out, index=False)
    print(f"[bronze] wrote {out} ({len(df_arrow)} rows, {df_arrow.shape[1]} cols)")

def _ingest_all_raw(project_root: Path, slug: str, *, max_files: Optional[int], sort_by: List[str]) -> pd.DataFrame:
    uris = _discover_uris(project_root, slug)
    if not uris:
        raise FileNotFoundError(f"No input files found under data/raw/{slug}/")
    if max_files is not None:
        uris = uris[:max_files]
    frames: List[pd.DataFrame] = []
    per_file_rows: List[Tuple[str, int]] = []
    for u in uris:
        df = _read_one(u)
        if df is None or df.empty:
            continue
        df = df.copy()
        df["_source_uri"] = u
        frames.append(df)
        per_file_rows.append((u, len(df)))
    if not frames:
        raise RuntimeError(f"Discovered {len(uris)} files but none were readable/non-empty.")
    df_all = pd.concat(frames, ignore_index=True, sort=False)
    sort_cols = [c for c in sort_by if c in df_all.columns]
    if sort_cols:
        try:
            df_all = df_all.sort_values(sort_cols, kind="mergesort", ignore_index=True)
        except Exception:
            pass
    by_ext = Counter(Path(u).suffix.lower() for u, _ in per_file_rows)
    total_rows = sum(n for _, n in per_file_rows)
    print(f"[ingest] files={len(per_file_rows)}  by_ext={dict(by_ext)}  rows_total={total_rows}  rows_concat={len(df_all)}")
    return df_all

def _have_bronze(bronze_dir: Path) -> bool:
    return (bronze_dir / "normalize.parquet").exists()

def _load_df(bronze_dir: Path) -> pd.DataFrame:
    parq = bronze_dir / "normalize.parquet"
    if not parq.exists():
        raise FileNotFoundError(f"normalize.parquet not found at {parq}")
    return pd.read_parquet(parq)

# ------------------------ schema/role utils ------------------------

def _infer_roles_for_df(df: pd.DataFrame, cfg: RootCfg) -> list[dict]:
    cols = []
    for name in df.columns:
        s = df[name]
        role, conf = guess_role(name, s, cfg.nlp, cfg.profiling.roles.datetime_formats)
        try:
            conf_f = float(conf) if conf is not None else 0.0
        except Exception:
            conf_f = 0.0
        cols.append({
            "name": str(name),
            "dtype": _dtype_str(s),
            "role": str(role),
            "role_confidence": conf_f,
        })
    return cols

def _proposed_from_cols(slug: str, cols: list[dict]) -> ProposedSchema:
    rc = []
    confs = []
    for c in cols:
        rc.append(ColumnSchema(
            name=c["name"],
            dtype=c["dtype"],
            role_confidence=RoleConfidence(role=c["role"], confidence=float(c.get("role_confidence", 0.0))),
            hints=ColumnHints(),
        ))
        confs.append(float(c.get("role_confidence", 0.0)))
    avg_conf = float(sum(confs) / len(confs)) if confs else 0.0
    return ProposedSchema(dataset_slug=slug, columns=rc, schema_confidence=avg_conf)

def _bootstrap_overlay_roles(old_bootstrap: dict, new_cols: list[dict], slug: str, schema_conf: float) -> dict:
    by_name = {c["name"]: c for c in new_cols}
    cols_out = []
    seen = set()
    for c in old_bootstrap.get("columns", []):
        name = c.get("name")
        if name in by_name:
            updated = deepcopy(c)
            updated["role"] = by_name[name]["role"]
            updated["role_confidence"] = by_name[name]["role_confidence"]
            updated["dtype"] = by_name[name]["dtype"]
            cols_out.append(updated); seen.add(name)
        else:
            cols_out.append(c)
    for name, c in by_name.items():
        if name not in seen:
            cols_out.append({
                "name": c["name"],
                "dtype": c["dtype"],
                "role": c["role"],
                "role_confidence": c["role_confidence"],
            })
    out = deepcopy(old_bootstrap)
    out["dataset"] = slug or out.get("dataset") or ""
    out["schema_confidence"] = float(schema_conf)
    out["columns"] = cols_out
    return out

# ------------------------ scoring / progress ------------------------

def _frame_missing_rate(df: pd.DataFrame) -> float:
    if df.size == 0:
        return 0.0
    try:
        return float(df.isna().sum().sum()) / float(df.shape[0] * df.shape[1])
    except Exception:
        return 0.0

def _score(rescore_dict: dict, df_after: pd.DataFrame) -> float:
    sc = float(rescore_dict.get("schema_conf_after", 0.0) or 0.0)
    avg = float(rescore_dict.get("avg_role_conf_after", 0.0) or 0.0)
    miss_penalty = _frame_missing_rate(df_after)
    return sc * 0.7 + avg * 0.3 - 0.25 * miss_penalty

# ------------------------ main loop ------------------------

def main():
    ap = argparse.ArgumentParser(description="Ingest (if needed) + iteratively clean one slug, printing improvements.")
    ap.add_argument("--config", default="config/config.toml")
    ap.add_argument("--slug", default="x1", help="Dataset slug under data/raw/<slug> and data/bronze/<slug>")
    ap.add_argument("--head", type=int, default=5)
    ap.add_argument("--max-iters", type=int, default=6)
    ap.add_argument("--min-improve", type=float, default=1e-4, help="Min score delta to keep going")
    ap.add_argument("--patience", type=int, default=2, help="Early stop after N non-improving rounds")
    ap.add_argument("--write-back", action="store_true", help="Persist updated TOML/JSON each improving round")

    # speed/UX
    ap.add_argument("--subset", type=int, default=0, help="If >0, sample this many rows for faster loops")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hash-key", default="", help="Optional column to do deterministic hash-based sampling")
    ap.add_argument("--save-best", action="store_true", help="Write best cleaned df to data/silver/<slug>/cleaned_best.parquet")

    # NEW: raw ingestion controls
    ap.add_argument("--from-raw", action="store_true", help="Ingest from data/raw/<slug>/** and write bronze before tuning")
    ap.add_argument("--rebuild-bronze", action="store_true", help="Force rebuild bronze even if it exists")
    ap.add_argument("--max-files", type=int, default=None, help="Limit number of raw files to read")
    ap.add_argument("--sort-by", nargs="*", default=[], help="Columns to sort by after merge (e.g., --sort-by event_time row_id)")

    args = ap.parse_args()

    project_root = Path.cwd()
    cfg = RootCfg.load(args.config)
    bronze = _bronze_dir(project_root, args.slug)
    bronze.mkdir(parents=True, exist_ok=True)

    # -------- ingest / ensure bronze --------
    need_ingest = args.from_raw or args.rebuild_bronze or not _have_bronze(bronze)
    if need_ingest:
        print(f"[ingest] building bronze for slug={args.slug}")
        df_raw_full = _ingest_all_raw(project_root, args.slug, max_files=args.max_files, sort_by=args.sort_by)
        _ensure_bronze(df_raw_full, bronze)
    else:
        print(f"[ingest] using existing bronze for slug={args.slug}")

    # -------- load working frame --------
    df_raw_full = _load_df(bronze)

    # Optional sampling for quick iterations
    if args.subset and args.subset > 0 and len(df_raw_full) > args.subset:
        if args.hash_key and args.hash_key in df_raw_full.columns:
            idx = pd.util.hash_pandas_object(df_raw_full[args.hash_key]).astype("uint64")
            frac = args.subset / float(len(df_raw_full))
            cutoff = int((2**64 - 1) * frac)
            df_raw = df_raw_full[idx <= cutoff].head(args.subset).copy()
        else:
            df_raw = df_raw_full.sample(n=args.subset, random_state=args.seed).copy()
    else:
        df_raw = df_raw_full.copy()

    bootstrap_json = _bootstrap_path(bronze)
    bootstrap = _load_bootstrap(bootstrap_json)

    # initial proposed from bootstrap (if present) else infer from raw
    if bootstrap.get("columns"):
        cols_init = [{
            "name": c["name"],
            "dtype": c.get("dtype") or (_dtype_str(df_raw[c["name"]]) if c["name"] in df_raw else "string"),
            "role": c.get("role", "text"),
            "role_confidence": float(c.get("role_confidence", 0.0)),
        } for c in bootstrap["columns"] if c.get("name") in df_raw.columns]
    else:
        cols_init = _infer_roles_for_df(df_raw, cfg)

    proposed = _proposed_from_cols(args.slug, cols_init)

    # --- baselines
    print(f"[load] slug={args.slug}  rows={len(df_raw_full)}  subset_rows={len(df_raw)}")
    def _frame_missing_rate(df: pd.DataFrame) -> float:
        if df.size == 0: return 0.0
        return float(df.isna().sum().sum()) / float(df.shape[0] * df.shape[1])

    miss_before = _frame_missing_rate(df_raw)
    r0 = rescore_after_clean(df_raw, proposed, cfg.profiling.roles, cfg.nlp)
    score0 = _score(r0.__dict__ if hasattr(r0, "__dict__") else dict(r0), df_raw)

    print(f"[BASE] schema_conf={r0.schema_conf_after:.4f}  avg_role_conf={r0.avg_role_conf_after:.4f}  miss={miss_before:.4f}  score={score0:.5f}")
    print("\n=== BEFORE: HEAD({}) ===".format(args.head))
    try:
        print(df_raw.head(args.head).to_string(index=False))
    except Exception:
        print(df_raw.head(args.head))
    print("\n=== BEFORE: DTYPES ===")
    for c, t in df_raw.dtypes.items():
        print(f"  {c}: {t}")

    best = {
        "iter": -1,
        "score": score0,
        "schema_conf": r0.schema_conf_after,
        "avg_role_conf": r0.avg_role_conf_after,
        "miss_rate": miss_before,
        "df": df_raw,
    }

    no_improve_rounds = 0

    for it in range(1, args.max_iters + 1):
        print("\n" + "=" * 80)
        print(f"ITER {it}")

        # clean starting from the sampled/raw frame with the *current* proposal
        result = run_clean_pass(df_raw, proposed, cfg)
        df_after = result.clean_df

        # rescore (prefer the one bundled in result)
        rescore_dict = result.rescore if isinstance(result.rescore, dict) else {}
        if not rescore_dict:
            r = rescore_after_clean(df_after, proposed, cfg.profiling.roles, cfg.nlp)
            rescore_dict = r.__dict__ if hasattr(r, "__dict__") else dict(r)

        miss_after = _frame_missing_rate(df_after)
        score = _score(rescore_dict, df_after)

        print(f"[SCORES] schema_conf={rescore_dict.get('schema_conf_after',0):.4f}  "
              f"avg_role_conf={rescore_dict.get('avg_role_conf_after',0):.4f}  "
              f"miss={miss_after:.4f}  score={score:.5f}  Δscore={score-best['score']:.5f}")

        print("\n=== AFTER: HEAD({}) ===".format(args.head))
        try:
            print(df_after.head(args.head).to_string(index=False))
        except Exception:
            print(df_after.head(args.head))
        print("\n=== AFTER: DTYPES ===")
        for c, t in df_after.dtypes.items():
            print(f"  {c}: {t}")

        actions = result.report.get("rules", {}).get("per_column_actions", {})
        fired = {c: acts for c, acts in actions.items() if acts}
        print("\n[Rules fired]")
        if not fired:
            print("  (none)")
        else:
            for c, acts in fired.items():
                print(f"  {c}: {', '.join(acts)}")

        _append_normalized_log(
            _normalized_log_path(bronze),
            {
                "iter": it,
                "schema_conf": rescore_dict.get("schema_conf_after"),
                "avg_role_conf": rescore_dict.get("avg_role_conf_after"),
                "score": score,
                "missing_rate": miss_after,
                "rules_total": result.report.get("rules", {}).get("total_applications"),
                "shape_after": list(df_after.shape),
            },
        )

        # check improvement
        if score > best["score"] + args.min_improve:
            no_improve_rounds = 0
            best.update({
                "iter": it,
                "score": score,
                "schema_conf": float(rescore_dict.get("schema_conf_after", 0.0)),
                "avg_role_conf": float(rescore_dict.get("avg_role_conf_after", 0.0)),
                "miss_rate": miss_after,
                "df": df_after,
            })

            # Build new roles from the NEW cleaned df and write back artifacts
            new_cols = _infer_roles_for_df(df_after, cfg)
            new_proposed = _proposed_from_cols(args.slug, new_cols)

            if args.write_back:
                updated_bootstrap = _bootstrap_overlay_roles(
                    _load_bootstrap(_bootstrap_path(bronze)),
                    new_cols,
                    slug=args.slug,
                    schema_conf=new_proposed.schema_confidence,
                )
                _save_bootstrap(_bootstrap_path(bronze), updated_bootstrap)
                out_toml = _save_schema_toml(project_root, args.slug, new_proposed)
                print(f"[WRITE] nlp_bootstrap.json updated; schema TOML -> {out_toml}")

            proposed = new_proposed

            if args.save_best:
                silver = _silver_dir(project_root, args.slug)
                silver.mkdir(parents=True, exist_ok=True)
                outp = silver / "cleaned_best.parquet"
                df_after.to_parquet(outp, index=False)
                print(f"[SAVE] best cleaned -> {outp}")
        else:
            no_improve_rounds += 1
            print(f"[STOPPING] no improvement ({no_improve_rounds}/{args.patience})")
            if no_improve_rounds >= args.patience:
                break

    print("\n" + "-" * 80)
    print(f"BEST @ iter {best['iter']}: score={best['score']:.5f}  "
          f"schema_conf={best['schema_conf']:.4f}  "
          f"avg_role_conf={best['avg_role_conf']:.4f}  "
          f"miss={best['miss_rate']:.4f}")
    print(f"Logs: {_normalized_log_path(bronze)}")
    if args.save_best:
        print(f"Best parquet: {(_silver_dir(project_root, args.slug) / 'cleaned_best.parquet')}")
    if args.write_back:
        print(f"Bootstrap: {_bootstrap_path(bronze)}")
        print(f"Schema TOML: {schema_path_for_slug(project_root, args.slug)}")

if __name__ == "__main__":
    main()
```



---
nlp and nlg code:

nlp/bootstrap:
```
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional
from collections import Counter
import pandas as pd

from ..io.catalog import Catalog
from ..io.readers import read_any
from ..io.storage import Storage
from ..config_model.model import RootCfg, NLPCfg
from .roles import guess_role, detect_unit_hint, canonicalize_categories, _dtype_str, _safe_nunique
from .schema import ColumnHints, ColumnSchema, ProposedSchema
from .schema_io import to_toml, schema_path_for_slug

_EMPTY_TOKENS_CANDIDATES = {"", "na", "n/a", "none", "null", "-", "—"}

def _sample_dataframe(storage: Storage, uris: List[str], sample_rows: int) -> pd.DataFrame:
    # unchanged
    frames: List[pd.DataFrame] = []
    for u in uris:
        try:
            df = read_any(storage, u, backend="pandas")
            if not isinstance(df, pd.DataFrame):
                df = pd.DataFrame(df)
            if len(df) > sample_rows:
                df = df.head(sample_rows)
            frames.append(df)
        except Exception:
            continue
        if sum(len(x) for x in frames) >= sample_rows:
            break
    return pd.concat(frames, axis=0, ignore_index=True) if frames else pd.DataFrame()

def _series_metrics(s: pd.Series) -> Dict[str, Any]:
    n = int(len(s))
    cover = float(s.notna().mean()) if n else 0.0
    nunique = _safe_nunique(s) if n else 0
    uniq_ratio = float(nunique / n) if n else 0.0
    avg_len = None
    empty_token_hits: List[str] = []
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        vals = s.dropna().astype(str).str.strip().str.lower()
        avg_len = float(vals.map(len).mean()) if not vals.empty else 0.0
        top = Counter(vals).most_common(200)
        empty_token_hits = [t for t, _ in top if t in _EMPTY_TOKENS_CANDIDATES]
    return {
        "non_null_ratio": cover,
        "nunique": nunique,
        "unique_ratio": uniq_ratio,
        "avg_len": avg_len,
        "empty_token_candidates": empty_token_hits,
    }

def _suggest_actions(colname: str, dtype: str, role: str, m: Dict[str, Any]) -> Dict[str, Any]:
    actions: Dict[str, Any] = {}
    cover = m.get("non_null_ratio", 0.0)
    uniq = m.get("unique_ratio", 0.0)

    if cover < 0.10 and role not in {"id", "time"}:
        actions["drop"] = True
    else:
        actions["drop"] = False
        if role == "numeric":
            actions["impute"] = "median"
        elif role in {"categorical", "text"}:
            actions["impute"] = "Unknown" if role == "categorical" else "N/A"
        elif role == "time":
            actions["impute"] = "parse/ffill"
        elif role == "bool":
            actions["impute"] = "mode"

    # Normalize hints (guard for None avg_len)
    avg_len = m.get("avg_len")
    if dtype == "string" and (avg_len or 0) >= 8:
        actions["normalize"] = {"strip": True, "lower": False}
    elif dtype == "string":
        actions["normalize"] = {"strip": True, "lower": False}

    # Optional: flag suspicious “id” that isn’t very unique
    if role == "id" and uniq < 0.9:
        actions["flag_low_uniqueness_for_id"] = True

    return actions


def propose_schema_for_df(slug: str, df: pd.DataFrame) -> ProposedSchema:
    from .schema import RoleConfidence
    cols: List[ColumnSchema] = []
    for c in df.columns:
        s = df[c]
        role, conf = guess_role(str(c), s)
        hints = ColumnHints(
            unit_hint=detect_unit_hint(s),
            canonical_map=canonicalize_categories(s),
            domain_guess=None,
        )
        cols.append(
            ColumnSchema(
                name=str(c),
                dtype=_dtype_str(s),
                role_confidence=RoleConfidence(role=role, confidence=float(conf)),
                hints=hints,
            )
        )
    schema_conf = float(sum(col.role_confidence.confidence for col in cols)) / max(1, len(cols)) if cols else 0.0
    return ProposedSchema(dataset_slug=slug, columns=cols, schema_confidence=schema_conf)

def run_nlp_bootstrap(
    storage: Storage,
    dataset_slug: str,
    *,
    project_root: Optional[Path] = None,
    nlp_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Reads small samples from raw/<slug> files, proposes schema, and returns artifacts.
    """
    # Base config from TOML if available, else use class defaults
    try:
        base_cfg = RootCfg.load().nlp.model_dump()
    except Exception:
        base_cfg = NLPCfg().model_dump()

    # Merge with caller overrides
    cfg = {**base_cfg, **(nlp_cfg or {})}

    # discover files for slug
    cat = Catalog(storage)
    entries = cat.inventory(dataset_slug=dataset_slug)
    uris = [e.uri for e in entries]

    df = _sample_dataframe(storage, uris, cfg["sample_rows"])
    proposed = propose_schema_for_df(dataset_slug, df) if not df.empty else ProposedSchema(dataset_slug, [], 0.0)

    cols_json = []
    for c in proposed.columns:
        s = df[c.name] if c.name in df.columns else pd.Series([], dtype="float64")
        m = _series_metrics(s)
        actions = _suggest_actions(c.name, c.dtype, c.role_confidence.role, m)
        cols_json.append({
            "name": c.name,
            "dtype": c.dtype,
            "role": c.role_confidence.role,
            "role_confidence": c.role_confidence.confidence,
            "metrics": m,
            "suggestions": actions,
            "hints": {
                "unit_hint": c.hints.unit_hint,
                "domain_guess": c.hints.domain_guess,
                "canonical_map_size": len(c.hints.canonical_map or {}),
            },
        })

    bootstrap = {
        "dataset": dataset_slug,
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)) if not df.empty else 0,
        "schema_confidence": proposed.schema_confidence,
        "columns": cols_json,
    }

    project_root = project_root or Path.cwd()
    schema_toml = to_toml(proposed.to_dict())
    schema_path = schema_path_for_slug(project_root, dataset_slug)

    bronze_dir = project_root.joinpath("data", "bronze", dataset_slug)
    bronze_dir.mkdir(parents=True, exist_ok=True)
    bootstrap_json_path = bronze_dir.joinpath("nlp_bootstrap.json")

    return {
        "proposed_schema": proposed,
        "bootstrap": bootstrap,
        "io": {
            "schema_toml_text": schema_toml,
            "schema_toml_path": str(schema_path),
            "bootstrap_json_path": str(bootstrap_json_path),
        },
        "nlp_cfg": cfg,
        "entries": [e.__dict__ for e in entries],
    }

```

nlp/roles:
```
from __future__ import annotations
import re
from typing import Optional, Tuple, Any, Iterable
import pandas as pd
import numpy as np
import warnings

# ---- Fallback config (used when nlp_cfg isn't passed) ----
class _RoleCfgFallback:
    # weights
    name_weight: float = 0.45
    value_weight: float = 0.55
    # thresholds
    bool_token_min_ratio: float = 0.57
    date_parse_min_ratio: float = 0.60
    unique_id_ratio: float = 0.95
    categorical_max_unique_ratio: float = 0.02
    text_min_avg_len: float = 8.0
    min_non_null_ratio: float = 0.10
    # bonuses/penalties
    bonus_id_name: float = 0.10
    penalize_bool_for_many_tokens: float = 0.08

# We keep a module-level fallback, but prefer the live nlp_cfg passed in.
_FALLBACK = _RoleCfgFallback()

def _cfg_from(nlp_cfg: Any | None) -> _RoleCfgFallback:
    """
    Robustly read fields from your TOML-driven config object.
    Works with pydantic, dataclass, SimpleNamespace, or plain dicts.
    """
    if nlp_cfg is None:
        return _FALLBACK

    def _get(obj, path: str, default):
        cur = obj
        for part in path.split("."):
            if cur is None:
                return default
            # dict-like
            if isinstance(cur, dict):
                cur = cur.get(part, None)
            else:
                cur = getattr(cur, part, None)
        return default if cur is None else cur

    rc = _RoleCfgFallback()
    # weights under [nlp.role_scoring]
    rc.name_weight  = float(_get(nlp_cfg, "role_scoring.name_weight",  _FALLBACK.name_weight))
    rc.value_weight = float(_get(nlp_cfg, "role_scoring.value_weight", _FALLBACK.value_weight))

    # thresholds directly under [nlp]
    rc.bool_token_min_ratio        = float(getattr(nlp_cfg, "bool_token_min_ratio",        _FALLBACK.bool_token_min_ratio))
    rc.date_parse_min_ratio        = float(getattr(nlp_cfg, "date_parse_min_ratio",        _FALLBACK.date_parse_min_ratio))
    rc.unique_id_ratio             = float(getattr(nlp_cfg, "unique_id_ratio",             _FALLBACK.unique_id_ratio))
    rc.categorical_max_unique_ratio= float(getattr(nlp_cfg, "categorical_max_unique_ratio",_FALLBACK.categorical_max_unique_ratio))
    rc.text_min_avg_len            = float(getattr(nlp_cfg, "text_min_avg_len",            _FALLBACK.text_min_avg_len))
    rc.min_non_null_ratio          = float(getattr(nlp_cfg, "min_non_null_ratio",          _FALLBACK.min_non_null_ratio))
    rc.bonus_id_name               = float(getattr(nlp_cfg, "bonus_id_name",               _FALLBACK.bonus_id_name))
    rc.penalize_bool_for_many_tokens = float(getattr(nlp_cfg, "penalize_bool_for_many_tokens", _FALLBACK.penalize_bool_for_many_tokens))
    return rc

# ---- Name patterns & units ----
ROLE_PATTERNS = [
    ("time", re.compile(r"(date|time|timestamp|dt|ts)\b", re.I)),
    ("id",   re.compile(r"(?:^|[_-])(id|uuid|guid|key)\b", re.I)),
    ("bool", re.compile(r"^(is_|has_|flag_)", re.I)),
    ("geo",  re.compile(r"(lat|lng|lon|long|latitude|longitude)\b", re.I)),
]

NUMERIC_UNITS = [
    (re.compile(r"^\s*\$"), "currency"),
    (re.compile(r"\b(percent|pct|%)\b", re.I), "percent"),
    (re.compile(r"\b(k|thousand)\b", re.I), "magnitude_k"),
    (re.compile(r"\b(million|mm)\b", re.I), "magnitude_m"),
]

_BOOL_TOKENS = {"true","false","t","f","y","n","yes","no","1","0"}

# ---- Utilities shared by other modules ----
def _dtype_str(s: pd.Series) -> str:
    dt = s.dtype
    if pd.api.types.is_datetime64_any_dtype(dt):
        return "datetime"
    if pd.api.types.is_integer_dtype(dt):
        return "int"
    if pd.api.types.is_float_dtype(dt):
        return "float"
    if pd.api.types.is_bool_dtype(dt):
        return "bool"
    return "string"

# ---- Local helpers ----
def _non_null_ratio(s: pd.Series) -> float:
    return float(s.notna().mean()) if len(s) else 0.0

def _safe_nunique(s: pd.Series) -> int:
    try:
        return int(s.nunique(dropna=True))
    except TypeError:
        return int(s.astype(str).nunique(dropna=True))

def _bool_token_ratio(s_obj: pd.Series) -> float:
    vals = s_obj.dropna().astype(str).str.strip().str.lower()
    if vals.empty:
        return 0.0
    return float(vals.isin(_BOOL_TOKENS).mean())

def _datetime_parse_ratio(s: pd.Series, fmts: list[str] | None) -> float:
    # Already datetime-like?
    if pd.api.types.is_datetime64_any_dtype(s):
        return 1.0
    try:
        x = s.dropna().astype(str)
        if x.empty:
            return 0.0

        parsed = None
        for f in (fmts or []):
            try:
                cand = pd.to_datetime(x, format=f, errors="coerce")
            except Exception:
                continue
            parsed = cand if parsed is None else parsed.fillna(cand)

        if parsed is None:
            # suppress Pandas' per-element parse warning on fallback
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                parsed = pd.to_datetime(x, errors="coerce")

        return float(parsed.notna().mean())
    except Exception:
        return 0.0

def _avg_len(s: pd.Series) -> float:
    try:
        vals = s.dropna().astype(str)
        return float(vals.map(len).mean()) if not vals.empty else 0.0
    except Exception:
        return 0.0

def _name_role_guess(colname: str) -> Optional[str]:
    for role, pat in ROLE_PATTERNS:
        if pat.search(colname or ""):
            return role
    return None

def _value_role_guess(
    s: pd.Series,
    cfg: _RoleCfgFallback,
    dt_fmts: list[str] | None
) -> Optional[str]:
    non_null_ratio = _non_null_ratio(s)
    if non_null_ratio < cfg.min_non_null_ratio:
        return "text"

    # datetime / bool / numeric native dtypes
    if pd.api.types.is_datetime64_any_dtype(s):
        return "time"
    if pd.api.types.is_bool_dtype(s):
        return "bool"
    if pd.api.types.is_numeric_dtype(s):
        nunique = _safe_nunique(s)
        if s.size > 0 and nunique / max(1, s.size) >= cfg.unique_id_ratio:
            return "id"
        return "numeric"

    # ---- object/string branch ----
    vals = s.dropna().astype(str)

    # 1) date-like strings?
    if not vals.empty:
        dt_ratio = _datetime_parse_ratio(vals, dt_fmts or [])
        if dt_ratio >= cfg.date_parse_min_ratio:
            return "time"

    # 2) numeric-like strings?
    if not vals.empty:
        # tolerate commas; expand here if you want $, % handling
        num = pd.to_numeric(vals.str.replace(",", ""), errors="coerce")
        numeric_ratio = float(num.notna().mean())
        if numeric_ratio >= 0.80:  # tune if needed
            nunique = int(num.dropna().nunique())
            size = max(1, s.size)
            if nunique / size >= cfg.unique_id_ratio:
                return "id"
            return "numeric"

    # 3) boolean-ish tokens?
    bt = _bool_token_ratio(s)
    if bt >= cfg.bool_token_min_ratio:
        return "bool"

    # 4) categorical vs text
    nunique = _safe_nunique(s)
    size = max(1, s.size)
    nunique_ratio = nunique / size
    avg_len = _avg_len(vals)

    if nunique_ratio <= cfg.categorical_max_unique_ratio and nunique <= 200:
        return "categorical"
    if avg_len >= cfg.text_min_avg_len:
        return "text"
    return "categorical" if nunique <= 200 else "text"

def detect_unit_hint(s: pd.Series) -> Optional[str]:
    if not (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)):
        return None
    sample = s.dropna().astype(str).head(50)
    for pat, name in NUMERIC_UNITS:
        try:
            if sample.map(lambda x: bool(pat.search(x))).mean() > 0.2:
                return name
        except Exception:
            continue
    return None

def canonicalize_categories(s: pd.Series) -> Optional[dict]:
    if not (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)):
        return None
    vals = s.dropna().astype(str)
    if vals.nunique(dropna=True) > 200:
        return None

    def canon(x: str) -> str:
        return re.sub(r"\s+", " ", x.strip().lower())

    mapping: dict[str, str] = {}
    for v in vals.value_counts().head(200).index.tolist():
        mapping[v.strip()] = canon(v)
    return mapping or None

def _blend_conf(name_present: bool, value_present: bool, same: bool, cfg: _RoleCfgFallback) -> float:
    nw, vw = float(cfg.name_weight), float(cfg.value_weight)
    if same and name_present and value_present:
        return min(0.99, 0.90 + 0.10 * max(nw, vw))  # ~0.95
    if value_present and name_present:
        base = 0.75
        bump = 0.05 if vw >= nw else 0.0
        return base + bump                                  # ~0.80
    if value_present:
        return 0.70 + 0.10 * vw                              # ~0.755
    if name_present:
        return 0.60 + 0.05 * nw                              # ~0.6225
    return 0.50

def _apply_quality_penalties(role: str, s: pd.Series, conf: float, cfg: _RoleCfgFallback) -> float:
    # 1) coverage scaling
    cover = _non_null_ratio(s)
    conf *= (0.25 + 0.75 * cover)  # cover=0 -> 0.25x; cover=1 -> 1x
    # 2) bool penalty if too many distinct tokens
    if role == "bool":
        try:
            nunique = _safe_nunique(s.astype(str))
            if nunique > 4:
                conf -= cfg.penalize_bool_for_many_tokens
        except Exception:
            pass
    return float(max(0.0, min(0.99, conf)))

# ---- Public API: must accept the signatures rescore() tries ----
def guess_role(
    colname: str | pd.Series,
    s: Optional[pd.Series] = None,
    nlp_cfg: Any = None,
    datetime_formats: Optional[list[str]] = None,
) -> Tuple[str, float]:
    """
    Return (role, confidence in [0,1]).
    Accepts multiple signatures:
      - guess_role(name, series, nlp_cfg, datetime_formats)
      - guess_role(name, series, nlp_cfg)
      - guess_role(series, nlp_cfg, datetime_formats)
      - guess_role(series, nlp_cfg)
      - guess_role(series)
    """
    # Normalize arguments to (name, series)
    name: str = ""
    series: pd.Series

    if isinstance(colname, pd.Series) and s is None:
        series = colname
        name = ""
    else:
        name = str(colname) if colname is not None else ""
        series = s  # type: ignore

    if series is None:
        # Defensive default
        return "text", 0.0

    cfg = _cfg_from(nlp_cfg)
    # get datetime formats if they were passed as a roles subconfig or list
    dt_fmts = datetime_formats if isinstance(datetime_formats, list) else []

    name_guess  = _name_role_guess(name) or ""
    value_guess = _value_role_guess(series, cfg, dt_fmts) or ""

    # If value suggests 'id' but name suggests something else, prefer the name (soft rule)
    if value_guess == "id" and name_guess:
        role, conf = name_guess, max(0.80, _blend_conf(True, True, False, cfg))
        return role, _apply_quality_penalties(role, series, conf, cfg)

    if name_guess and value_guess:
        if name_guess == value_guess:
            role, conf = name_guess, _blend_conf(True, True, True, cfg)
            return role, _apply_quality_penalties(role, series, conf, cfg)
        role, conf = value_guess, _blend_conf(True, True, False, cfg)
        return role, _apply_quality_penalties(role, series, conf, cfg)

    if value_guess:
        role, conf = value_guess, _blend_conf(False, True, False, cfg)
        return role, _apply_quality_penalties(role, series, conf, cfg)

    if name_guess:
        role, conf = name_guess, _blend_conf(True, False, False, cfg)
        return role, _apply_quality_penalties(role, series, conf, cfg)

    role, conf = "text", 0.5
    return role, _apply_quality_penalties(role, series, conf, cfg)

```

nlp/schema_io:
```
from __future__ import annotations
from pathlib import Path, PurePosixPath
from typing import Dict, Any
from tomlkit import document, dumps  # table() not needed

def _none_to_empty(x):
    return "" if x is None else x

def to_toml(proposed: Dict[str, Any]) -> str:
    doc = document()
    doc.add("dataset", proposed["dataset"])
    doc.add("version", proposed.get("version", 1))
    doc.add("schema_confidence", round(float(proposed.get("schema_confidence", 0.0)), 4))

    cols_tbl = []
    for c in proposed["columns"]:
        hints = c.get("hints", {})
        cols_tbl.append({
            "name": c["name"],
            "dtype": c["dtype"],
            "role": c["role"],
            "role_confidence": round(float(c["role_confidence"]), 4),
            "hints": {
                "unit_hint": _none_to_empty(hints.get("unit_hint")),
                "domain_guess": _none_to_empty(hints.get("domain_guess")),
                "canonical_map": hints.get("canonical_map") or {},
            },
        })
    doc.add("columns", cols_tbl)
    if proposed.get("notes"):
        doc.add("notes", proposed["notes"])
    return dumps(doc)

def schema_path_for_slug(project_root: Path, slug: str) -> PurePosixPath:
    """
    Return a POSIX-style path so string comparisons are stable across OSes,
    while still behaving like a path (has .parent, .name, etc.).
    """
    base = PurePosixPath(Path(project_root).as_posix())
    return base.joinpath("schemas", f"{slug}.schema.toml")

```

nlp/schema.py:
```
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any

@dataclass(frozen=True)
class RoleConfidence:
    role: str
    confidence: float

@dataclass(frozen=True)
class ColumnHints:
    unit_hint: Optional[str] = None
    canonical_map: Optional[Dict[str, str]] = None  # for categories/text normalization
    domain_guess: Optional[str] = None

@dataclass(frozen=True)
class ColumnSchema:
    name: str
    dtype: str
    role_confidence: RoleConfidence
    hints: ColumnHints

@dataclass(frozen=True)
class ProposedSchema:
    dataset_slug: str
    columns: List[ColumnSchema]
    schema_confidence: float
    version: int = 1
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset_slug,
            "version": self.version,
            "schema_confidence": self.schema_confidence,
            "columns": [
                {
                    "name": c.name,
                    "dtype": c.dtype,
                    "role": c.role_confidence.role,
                    "role_confidence": c.role_confidence.confidence,
                    "hints": {
                        "unit_hint": c.hints.unit_hint,
                        "domain_guess": c.hints.domain_guess,
                        "canonical_map": c.hints.canonical_map or {},
                    },
                }
                for c in self.columns
            ],
            "notes": self.notes or "",
        }

```

nlg/narrative_constants.py:
```
from __future__ import annotations

try:
    from ..config_model.model import RootCfg
    _cfg = RootCfg.load()
    INVENTORY_KEY = _cfg.nlg.inventory_key
    NARRATIVE_FILENAME = _cfg.nlg.narrative_filename
except Exception:
    # Fallbacks if config cannot be loaded (tests, minimal environments, etc.)
    INVENTORY_KEY = "_inventory"
    NARRATIVE_FILENAME = "narrative.txt"

```

nlg/narrative:
```
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List
import json

from ..io.storage import Storage
from ..io.catalog import Catalog
from .narrative_constants import INVENTORY_KEY

def summarize_inventory(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_kind: Dict[str, int] = {}
    by_slug: Dict[str, int] = {}
    for e in entries:
        by_kind[e["kind"]] = by_kind.get(e["kind"], 0) + 1
        by_slug[e["slug"]] = by_slug.get(e["slug"], 0) + 1
    return {
        "total_files": len(entries),
        "by_kind": dict(sorted(by_kind.items())),
        "by_slug": dict(sorted(by_slug.items())),
    }

def build_narrative(dataset_slug: str, bootstrap: Dict[str, Any], inventory_summary: Dict[str, Any]) -> str:
    lines = []
    lines.append(f"# Dataset: {dataset_slug}")
    lines.append(f"- Rows sampled: {bootstrap.get('row_count', 0)}")
    lines.append(f"- Columns: {bootstrap.get('column_count', 0)}")
    lines.append(f"- Schema confidence: {bootstrap.get('schema_confidence', 0.0):.2f}")
    lines.append("")
    lines.append("## Files discovered")
    lines.append(f"- Total: {inventory_summary['total_files']}")
    for k, v in inventory_summary["by_kind"].items():
        lines.append(f"  - {k}: {v}")
    lines.append("")
    lines.append("## Columns (role → confidence) and suggested fixes")

    for c in bootstrap.get("columns", []):
        hint = c.get("hints", {}) or {}
        m = c.get("metrics", {}) or {}
        s = c.get("suggestions", {}) or {}
        hb = []
        if hint.get("unit_hint"): hb.append(f"unit={hint['unit_hint']}")
        if hint.get("domain_guess"): hb.append(f"domain={hint['domain_guess']}")
        if hint.get("canonical_map_size"): hb.append(f"canon={hint['canonical_map_size']}")
        mb = []
        if "non_null_ratio" in m:
            mb.append(f"non-null={m['non_null_ratio']:.2f}")
        if "unique_ratio" in m:
            mb.append(f"uniq={m['unique_ratio']:.2f}")
        sugg = []
        if s.get("drop"): sugg.append("DROP")
        if s.get("impute"): sugg.append(f"impute={s['impute']}")
        if s.get("normalize"): sugg.append("normalize")
        if s.get("treat_as_null"): sugg.append(f"nullify={','.join(s['treat_as_null'])}")
        hb_s = f" [{', '.join(hb)}]" if hb else ""
        mb_s = f" {{{', '.join(mb)}}}" if mb else ""
        sg_s = f" -> {'; '.join(sugg)}" if sugg else ""
        lines.append(
            f"- {c['name']} : {c['dtype']} → {c['role']} ({c['role_confidence']:.2f}){hb_s}{mb_s}{sg_s}"
        )
    return "\n".join(lines)


def narrative_payload(dataset_slug: str, bootstrap: Dict[str, Any], entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    inventory_summary = summarize_inventory(entries)
    return {
        "dataset": dataset_slug,
        "schema_confidence": bootstrap.get("schema_confidence", 0.0),
        "bootstrap": bootstrap,
        INVENTORY_KEY: inventory_summary,
        "narrative_text": build_narrative(dataset_slug, bootstrap, inventory_summary),
    }

```


nlp/suggestions.py:
```
from __future__ import annotations
from typing import Dict, List, Any
import pandas as pd

from ..config_model.model import RootCfg
from ..nlp.schema import ProposedSchema
from ..cleaning.metrics import profile_columns

def _to_list(x) -> list:
    try:
        return list(x) if isinstance(x, (list, tuple)) else []
    except Exception:
        return []

def _get(cfg: Any, path: str, default=None):
    cur = cfg
    for part in path.split("."):
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(part, None)
        else:
            cur = getattr(cur, part, None)
    return default if cur is None else cur

def _num_ratio(s: pd.Series) -> float:
    try:
        x = pd.to_numeric(s.astype(str).str.replace(",", ""), errors="coerce")
        return float(x.notna().mean())
    except Exception:
        return 0.0

def _bool_token_ratio(s: pd.Series) -> float:
    try:
        vals = s.dropna().astype(str).str.strip().str.lower()
        if vals.empty: 
            return 0.0
        tokens = {"true","false","t","f","y","n","yes","no","1","0"}
        return float(vals.isin(tokens).mean())
    except Exception:
        return 0.0

def plan_followups(
    df_after: pd.DataFrame,
    proposed: ProposedSchema,
    rescore: dict,         # dict as provided by engine (asdict)
    cfg: RootCfg,
) -> Dict[str, List[str]]:
    """
    Returns: { column -> list of 'then(...)' strings to try next }
    Only suggests for columns whose role-confidence is below cfg.nlp.min_role_confidence.
    """
    # thresholds & knobs from config (robust access)
    min_role = float(_get(cfg, "nlp.min_role_confidence", 0.90))
    dt_formats = _to_list(_get(cfg, "profiling.roles.datetime_formats", []))

    always_keep = set(_get(cfg, "cleaning.columns.always_keep", []) or [])
    drop_missing_pct = float(_get(cfg, "cleaning.columns.drop_missing_pct", 0.90))
    cat_max = int(_get(cfg, "cleaning.columns.cat_cardinality_max", 200))
    uniq_id_ratio = float(_get(cfg, "nlp.unique_id_ratio", 0.95))
    date_parse_min = float(_get(cfg, "nlp.date_parse_min_ratio", 0.60))
    bool_min = float(_get(cfg, "nlp.bool_token_min_ratio", 0.57))
    text_min_avg_len = float(_get(cfg, "nlp.text_min_avg_len", 8.0))

    # schema role map from proposed
    p = proposed.to_dict()
    schema_roles = {c["name"]: c["role"] for c in p.get("columns", [])}

    # quick per-column metrics on the AFTER frame
    metrics = profile_columns(df_after, schema_roles, dt_formats)

    out: Dict[str, List[str]] = {}
    per_col = dict(rescore.get("per_column", {}))

    for col in df_after.columns:
        pc = per_col.get(col, {})
        conf_after = float(pc.get("after", 0.0) or 0.0)
        if conf_after >= min_role:
            continue  # this column is fine

        m = metrics.get(col, {})
        dtype = str(m.get("type", "string"))
        role_after = str(pc.get("role_after", schema_roles.get(col, "text")))
        miss = float(m.get("missing_pct", 0.0) or 0.0)
        nunique = int(m.get("nunique", 0) or 0)
        uniq_ratio = float(m.get("unique_ratio", 0.0) or 0.0)
        avg_len = m.get("avg_len", None)

        s = df_after[col]
        recs: List[str] = []

        # 0) extreme sparsity (and not always-keep)
        if miss >= drop_missing_pct and col not in always_keep:
            recs.append("drop_column()")

        # 1) time-like but not parsed well
        if role_after == "time":
            dt_ratio = float(m.get("datetime_parse_ratio", 0.0) or 0.0)
            if dtype == "string" and dt_ratio < date_parse_min:
                recs.append(f'parse_datetime({dt_formats!r})')
            if miss > 0:
                recs.append('impute_dt("ffill")')

        # 2) numeric hiding in strings
        if dtype == "string":
            if _num_ratio(s) >= 0.80 and role_after in {"numeric", "id", "text", "categorical"}:
                recs.append("coerce_numeric()")
                if role_after == "id" and uniq_ratio < uniq_id_ratio:
                    # ID doesn’t look unique enough; treat as categorical to stabilize downstream
                    recs.append("cast_category()")
            # 3) boolean-looking strings
            if _bool_token_ratio(s) >= bool_min and role_after in {"bool","categorical","text"}:
                recs.append("normalize_null_tokens(null_tokens=cleaning.normalize_null_tokens.null_tokens, case_insensitive=true)")
                recs.append("cast_category()")

        # 4) categorical enforcement
        if role_after == "categorical" and nunique <= cat_max and dtype == "string":
            recs.append("cast_category()")

        # 5) missing value handling by role
        if miss > 0:
            if role_after in {"numeric","id"}:
                recs.append('impute(cleaning.impute.numeric_default)')
            elif role_after == "time":
                recs.append('impute_dt("ffill")')
            elif role_after == "categorical":
                recs.append('impute_value(cleaning.impute.categorical_default)')
            elif role_after == "text" and (avg_len or 0) >= text_min_avg_len:
                recs.append('impute_value(cleaning.impute.text_default)')
                recs.append('text_normalize(strip=cleaning.normalize.strip_text, lower=cleaning.normalize.lowercase_text)')

        # de-dupe while preserving order
        if recs:
            seen = set()
            uniq_recs = []
            for r in recs:
                if r not in seen:
                    uniq_recs.append(r); seen.add(r)
            out[col] = uniq_recs

    return out


```

---

cleaning files

dsl.py:
```
from __future__ import annotations
from typing import Any, Callable, List, Tuple, Optional, Sequence, Dict
from dataclasses import dataclass
import re

AllowedCallable = Callable[[dict[str, Any]], bool]

# =========================
# Tokenizer
# =========================

class Token:
    __slots__ = ("typ", "val", "pos")
    def __init__(self, typ: str, val: Any, pos: int) -> None:
        self.typ = typ
        self.val = val
        self.pos = pos
    def __repr__(self) -> str:
        return f"Token({self.typ!r}, {self.val!r}, pos={self.pos})"

_WHITESPACE = set(" \t\r\n")
_IDENT_START = set("_abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
_IDENT_CONT = _IDENT_START.union(set("0123456789"))
_DIGITS = set("0123456789")

def _is_ident_start(ch: str) -> bool:
    return ch in _IDENT_START

def _is_ident_cont(ch: str) -> bool:
    return ch in _IDENT_CONT or ch == "."

def _read_while(s: str, i: int, pred) -> Tuple[str, int]:
    j = i
    n = len(s)
    while j < n and pred(s[j]):
        j += 1
    return s[i:j], j

def _read_string(s: str, i: int) -> Tuple[str, int]:
    quote = s[i]
    i += 1
    out: List[str] = []
    n = len(s)
    esc = False
    while i < n:
        ch = s[i]
        i += 1
        if esc:
            if ch in ['\\', '"', "'"]:
                out.append(ch)
            elif ch == "n":
                out.append("\n")
            elif ch == "t":
                out.append("\t")
            elif ch == "r":
                out.append("\r")
            else:
                out.append(ch)
            esc = False
        else:
            if ch == "\\":
                esc = True
            elif ch == quote:
                return "".join(out), i
            else:
                out.append(ch)
    raise ValueError("Unterminated string literal")

def _read_number(s: str, i: int) -> Tuple[float | int, int]:
    j = i
    n = len(s)
    if j < n and s[j] == "-":
        j += 1
    int_part, j = _read_while(s, j, lambda c: c in _DIGITS)
    if int_part == "":
        raise ValueError(f"Invalid number at {i}")
    # fractional?
    if j < n and s[j] == ".":
        j += 1
        frac, j = _read_while(s, j, lambda c: c in _DIGITS)
        if frac == "":
            raise ValueError(f"Invalid float at {i}")
        return float(s[i:j]), j
    return int(s[i:j]), j

def _tokenize(expr: str) -> List[Token]:
    s = expr
    i = 0
    n = len(s)
    toks: List[Token] = []
    while i < n:
        ch = s[i]
        if ch in _WHITESPACE:
            i += 1; continue

        # punctuation
        if ch == "(":
            toks.append(Token("LPAREN", "(", i)); i += 1; continue
        if ch == ")":
            toks.append(Token("RPAREN", ")", i)); i += 1; continue
        if ch == "[":
            toks.append(Token("LBRACK", "[", i)); i += 1; continue
        if ch == "]":
            toks.append(Token("RBRACK", "]", i)); i += 1; continue
        if ch == ",":
            toks.append(Token("COMMA", ",", i)); i += 1; continue

        # two-char ops
        if i + 1 < n:
            two = s[i:i+2]
            if two in ("==", "!=", "<=", ">="):
                toks.append(Token("OP", two, i)); i += 2; continue

        # single-char comp
        if ch in "<>":
            toks.append(Token("OP", ch, i)); i += 1; continue

        # strings
        if ch in ("'", '"'):
            val, j = _read_string(s, i)
            toks.append(Token("STR", val, i))
            i = j; continue

        # numbers
        if ch in _DIGITS or (ch == "-" and i+1 < n and s[i+1] in _DIGITS):
            val, j = _read_number(s, i)
            toks.append(Token("NUM", val, i))
            i = j; continue

        # identifiers/keywords
        if _is_ident_start(ch):
            raw, j = _read_while(s, i, _is_ident_cont)
            low = raw.lower()
            if low == "and": toks.append(Token("AND", "and", i))
            elif low == "or": toks.append(Token("OR", "or", i))
            elif low == "not": toks.append(Token("NOT", "not", i))
            elif low == "in": toks.append(Token("IN", "in", i))
            elif low == "notin": toks.append(Token("NOTIN", "notin", i))
            elif low in ("true","false"): toks.append(Token("BOOL", True if low=="true" else False, i))
            elif low in ("none","null"): toks.append(Token("NONE", None, i))
            else:
                toks.append(Token("ID", raw, i))
            i = j; continue

        raise ValueError(f"Unexpected character {ch!r} at position {i}")

    toks.append(Token("EOF", None, n))
    return toks

# =========================
# Parser (recursive descent)
# =========================

class Parser:
    def __init__(self, tokens: List[Token]) -> None:
        self.toks = tokens
        self.i = 0

    def _peek(self) -> Token:
        return self.toks[self.i]

    def _eat(self, typ: Optional[str] = None) -> Token:
        t = self._peek()
        if typ and t.typ != typ:
            raise ValueError(f"Expected {typ}, got {t.typ} at {t.pos}")
        self.i += 1
        return t

    def parse(self):
        node = self._parse_or()
        if self._peek().typ != "EOF":
            raise ValueError(f"Unexpected token {self._peek()}")
        return node

    # or_expr := and_expr ('OR' and_expr)*
    def _parse_or(self):
        left = self._parse_and()
        while self._peek().typ == "OR":
            self._eat("OR")
            right = self._parse_and()
            left = ("or", left, right)
        return left

    # and_expr := not_expr ('AND' not_expr)*
    def _parse_and(self):
        left = self._parse_not()
        while self._peek().typ == "AND":
            self._eat("AND"); right = self._parse_not()
            left = ("and", left, right)
        return left

    # not_expr := ('NOT' not_expr) | comparison
    def _parse_not(self):
        if self._peek().typ == "NOT":
            self._eat("NOT")
            node = self._parse_not()
            return ("not", node)
        return self._parse_comparison()

    # comparison := arith ( (OP|IN|NOT IN) arith )?
    def _parse_comparison(self):
        left = self._parse_arith()
        t = self._peek()
        if t.typ == "NOT":
            # 'NOT' 'IN'
            self._eat("NOT")
            if self._peek().typ != "IN":
                raise ValueError(f"Expected IN after NOT at {t.pos}")
            self._eat("IN")
            right = self._parse_arith()
            return ("cmp", "notin", left, right)
        
        if t.typ == "NOTIN":
            self._eat("NOTIN")
            right = self._parse_arith()
            return ("cmp", "notin", left, right)

        if t.typ in ("OP","IN"):
            op = self._eat().val
            right = self._parse_arith()
            return ("cmp", op, left, right)
        return left

    # For now, arith just forwards to primary (we don't need +,-,*,/ in conditions)
    def _parse_arith(self):
        return self._parse_primary()

    # primary := literal | identifier | funcall | list | '(' expr ')'
    def _parse_primary(self):
        t = self._peek()
        if t.typ == "LPAREN":
            self._eat("LPAREN")
            node = self._parse_or()
            self._eat("RPAREN")
            return node
        if t.typ in ("STR","NUM","BOOL","NONE"):
            self._eat(t.typ)
            return ("lit", t.val)
        if t.typ == "LBRACK":
            return self._parse_list()
        if t.typ == "ID":
            # identifier or funcall
            name = self._eat("ID").val
            if self._peek().typ == "LPAREN":
                # funcall
                self._eat("LPAREN")
                args: List[Any] = []
                if self._peek().typ != "RPAREN":
                    while True:
                        args.append(self._parse_or())
                        if self._peek().typ == "COMMA":
                            self._eat("COMMA"); continue
                        break
                self._eat("RPAREN")
                return ("call", name, args)
            return ("id", name)
        raise ValueError(f"Unexpected token {t} in primary")

    def _parse_list(self):
        # [expr, expr, ...]
        l_tok = self._eat("LBRACK")
        items: List[Any] = []
        if self._peek().typ != "RBRACK":
            while True:
                items.append(self._parse_or())
                if self._peek().typ == "COMMA":
                    self._eat("COMMA"); continue
                break
        self._eat("RBRACK")
        return ("list", items)

# =========================
# Safe evaluation
# =========================

def _resolve_identifier(path: str, ctx: dict[str, Any]) -> Any:
    parts = path.split(".")
    cur: Any = ctx.get(parts[0], None)
    for p in parts[1:]:
        if isinstance(cur, dict):
            cur = cur.get(p, None)
        else:
            return None
    return cur

def _to_seq(x: Any) -> Optional[Sequence]:
    if isinstance(x, (list, tuple, set)):
        return list(x)  # normalize set to list
    return None

def _safe_compare(op: str, a: Any, b: Any) -> bool:
    try:
        if op == "==": return a == b
        if op == "!=": return a != b
        if op == "<":  return a <  b
        if op == "<=": return a <= b
        if op == ">":  return a >  b
        if op == ">=": return a >= b
        if op == "in":
            # allow string containment as well
            if isinstance(b, str) and not isinstance(a, (list, tuple, set, dict)):
                return str(a) in b
            seq = _to_seq(b)
            return (a in seq) if seq is not None else False
        if op == "notin":
            if isinstance(b, str) and not isinstance(a, (list, tuple, set, dict)):
                return str(a) not in b
            seq = _to_seq(b)
            return (a not in seq) if seq is not None else True
    except Exception:
        return False
    return False

# Whitelisted safe functions
def _fn_len(x): 
    try: return 0 if x is None else len(x)  # type: ignore
    except Exception: return 0
def _fn_isnull(x): 
    try:
        import pandas as pd
        return bool(pd.isna(x))
    except Exception:
        return x is None
def _fn_notnull(x): 
    return not _fn_isnull(x)
def _fn_startswith(x, y): 
    return str(x).startswith(str(y)) if x is not None and y is not None else False
def _fn_endswith(x, y): 
    return str(x).endswith(str(y)) if x is not None and y is not None else False
def _fn_contains(x, y): 
    return str(y) in str(x) if x is not None and y is not None else False
def _fn_icontains(x, y): 
    return str(y).lower() in str(x).lower() if x is not None and y is not None else False
def _fn_matches(x, pattern):
    try: return bool(re.search(str(pattern), str(x))) if x is not None else False
    except re.error: return False
def _fn_imatches(x, pattern):
    try: return bool(re.search(str(pattern), str(x), flags=re.I)) if x is not None else False
    except re.error: return False

SAFE_FUNCS: Dict[str, Callable[..., Any]] = {
    "len": _fn_len,
    "isnull": _fn_isnull,
    "notnull": _fn_notnull,
    "startswith": _fn_startswith,
    "endswith": _fn_endswith,
    "contains": _fn_contains,
    "icontains": _fn_icontains,
    "matches": _fn_matches,
    "imatches": _fn_imatches,
}

def _collect_roots(ast) -> set[str]:
    typ = ast[0]
    if typ == "id":
        name: str = ast[1]
        return {name.split(".", 1)[0]}
    if typ in ("lit",):
        return set()
    if typ == "list":
        roots: set[str] = set()
        for item in ast[1]:
            roots |= _collect_roots(item)
        return roots
    if typ == "call":
        roots = set()
        for arg in ast[2]:
            roots |= _collect_roots(arg)
        return roots
    if typ == "not":
        return _collect_roots(ast[1])
    if typ in ("and", "or"):
        return _collect_roots(ast[1]) | _collect_roots(ast[2])
    if typ == "cmp":
        _, _op, l, r = ast
        return _collect_roots(l) | _collect_roots(r)
    return set()

def _compile(ast, allowed_funcs: Optional[set[str]]):
    typ = ast[0]

    if typ == "lit":
        val = ast[1]
        return lambda ctx: val

    if typ == "id":
        name = ast[1]
        return lambda ctx: _resolve_identifier(name, ctx)

    if typ == "list":
        items = [_compile(a, allowed_funcs) for a in ast[1]]
        return lambda ctx: [f(ctx) for f in items]

    if typ == "call":
        name = ast[1]
        if allowed_funcs is not None and name not in allowed_funcs:
            raise ValueError(f"Disallowed function: {name}")
        fn = SAFE_FUNCS.get(name)
        if fn is None:
            raise ValueError(f"Unknown function: {name}")
        arg_nodes = [_compile(a, allowed_funcs) for a in ast[2]]
        return lambda ctx: fn(*[g(ctx) for g in arg_nodes])

    if typ == "not":
        f = _compile(ast[1], allowed_funcs)
        return lambda ctx: (not bool(f(ctx)))

    if typ == "and":
        lf = _compile(ast[1], allowed_funcs); rf = _compile(ast[2], allowed_funcs)
        return lambda ctx: (bool(lf(ctx)) and bool(rf(ctx)))

    if typ == "or":
        lf = _compile(ast[1], allowed_funcs); rf = _compile(ast[2], allowed_funcs)
        return lambda ctx: (bool(lf(ctx)) or bool(rf(ctx)))

    if typ == "cmp":
        _, op, l, r = ast
        lf = _compile(l, allowed_funcs); rf = _compile(r, allowed_funcs)
        return lambda ctx: _safe_compare(op, lf(ctx), rf(ctx))

    raise ValueError(f"Unknown AST node: {ast!r}")

# =========================
# Public API
# =========================

def compile_condition(
    expr: str,
    allowed_vars: set[str] | None = None,
    *,
    allowed_funcs: set[str] | None = None,
) -> AllowedCallable:
    """
    Compile a safe callable(ctx)->bool from `expr`.

    - `allowed_vars`: restrict which *root* names may be referenced (e.g., {'name','role','missing_pct','cleaning'})
    - `allowed_funcs`: restrict callable names; defaults to a whitelist of SAFE_FUNCS

    Raises ValueError on syntax or safety violations.
    """
    tokens = _tokenize(expr)
    parser = Parser(tokens)
    ast = parser.parse()

    if allowed_vars is not None:
        roots = _collect_roots(ast)
        bad = [r for r in roots if r not in allowed_vars]
        if bad:
            raise ValueError(f"Use of disallowed identifiers: {bad}")

    if allowed_funcs is None:
        allowed_funcs = set(SAFE_FUNCS.keys())

    fn = _compile(ast, allowed_funcs)

    def _wrapped(ctx: dict[str, Any]) -> bool:
        try:
            return bool(fn(ctx))
        except Exception:
            # Any runtime type error becomes False, keeping engine robust.
            return False
    return _wrapped

def eval_condition(fn: AllowedCallable, ctx: dict[str, Any]) -> bool:
    return bool(fn(ctx))


```

engine.py:
```
from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import Any
import pandas as pd

from ..config_model.model import RootCfg
from ..nlp.schema import ProposedSchema
from .metrics import profile_columns
from .dsl import compile_condition, eval_condition
from .registry import compile_actions_registry, parse_then, NameRef
from .rescore import rescore_after_clean
from .report import build_iteration_report
from ..nlp.suggestions import plan_followups

# ---- Data classes ----

@dataclass(frozen=True)
class RuleSpec:
    id: str
    priority: int
    when: str
    then: str

@dataclass(frozen=True)
class RuleHit:
    column: str
    rule_id: str
    before_type: str
    after_type: str
    before_role: str
    after_role: str | None
    notes: str | None = None

@dataclass(frozen=True)
class CleaningResult:
    clean_df: pd.DataFrame
    report: dict[str, Any]
    rescore: dict[str, Any]
    column_report: dict[str, dict[str, Any]]
    dropped: dict[str, str]
    suggestions: dict[str, list[str]] = field(default_factory=dict)
    meets_thresholds: bool = False
    thresholds: dict[str, float] = field(default_factory=dict)

# ---- helpers ----

def _dtype_tag(s: pd.Series) -> str:
    from ..nlp.roles import _dtype_str
    return _dtype_str(s)


def _quick_facts(series: pd.Series, *, name: str, schema_role: str, df_index) -> dict:
    n = len(series) or 1
    t = _dtype_tag(series)
    try:
        nunique = int(series.nunique(dropna=True))
    except TypeError:
        nunique = int(series.astype(str).nunique(dropna=True))
    avg_len = None
    if t == "string":
        vals = series.dropna().astype(str)
        avg_len = float(vals.map(len).mean()) if not vals.empty else 0.0
    missing_pct = float(series.isna().mean())
    return {
        "name": name,
        "type": t,
        "role": schema_role,                  # keep schema role stable
        "missing_pct": missing_pct,
        "non_null_ratio": 1.0 - missing_pct,
        "nunique": nunique,
        "unique_ratio": float(nunique) / float(n),
        "cardinality": nunique,
        "avg_len": avg_len,
        "has_time_index": pd.api.types.is_datetime64_any_dtype(df_index),
        # extras used in some rules; safe defaults if not needed
        "mean": None, "std": None, "iqr": None,
        "bool_token_ratio": 0.0,
        "datetime_parse_ratio": 0.0,
        "is_monotonic_increasing": False,
    }

# ---- Public API ----

def run_clean_pass(
    df: pd.DataFrame,
    proposed_schema: ProposedSchema,
    cfg: RootCfg,
) -> CleaningResult:
    """
    One pure cleaning pass:
      - profile
      - apply rules (priority order)
      - re-profile
      - rescore (NLP)
      - build iteration report
    """
    from .policy import build_policy_from_config
    rules, env = build_policy_from_config(cfg)
    registry = compile_actions_registry()

    schema_roles = {c["name"]: c["role"] for c in proposed_schema.to_dict()["columns"]}

    metrics_before = profile_columns(df, schema_roles, cfg.profiling.roles.datetime_formats)
    df_after, hits, col_report, dropped = apply_rules(df, schema_roles, rules, env, registry)
    metrics_after = profile_columns(df_after, schema_roles, cfg.profiling.roles.datetime_formats)

    # Rescore using NLP before/after confidences (returns a dataclass)
    rescore_result = rescore_after_clean(
        df_after,
        proposed_schema,
        cfg.profiling.roles,
        cfg.nlp,
    )
    # Build a single canonical report
    report = build_iteration_report(
        df_before=df,
        df_after=df_after,
        metrics_before=metrics_before,
        metrics_after=metrics_after,
        rule_hits=hits,
        rescore=rescore_result,  # report handles dataclass/dict
    )

    try:
        min_schema = float(getattr(cfg.nlp, "min_schema_confidence", 0.90))
    except Exception:
        min_schema = 0.90
    try:
        min_role = float(getattr(cfg.nlp, "min_role_confidence", 0.90))
    except Exception:
        min_role = 0.90

    meets = (
        float(rescore_result.schema_conf_after) >= min_schema
        and float(rescore_result.avg_role_conf_after) >= min_role
    )

    suggestions: dict[str, list[str]] = {}
    if not meets:
        # plan_followups uses AFTER frame + proposed schema + the detailed rescore map
        suggestions = plan_followups(df_after, proposed_schema, asdict(rescore_result), cfg)

    # annotate the report so downstream writers/NLG can render targets & actions
    report["targets"] = {
        "min_schema_confidence": min_schema,
        "min_role_confidence": min_role,
        "schema_conf_after": float(rescore_result.schema_conf_after),
        "avg_role_conf_after": float(rescore_result.avg_role_conf_after),
        "met": bool(meets),
    }
    if suggestions:
        report["suggestions"] = suggestions

    return CleaningResult(
        clean_df=df_after,
        report=report,
        rescore=asdict(rescore_result),
        column_report=col_report,
        dropped=dropped,
        suggestions=suggestions,
        meets_thresholds=meets,
        thresholds={"min_schema_confidence": min_schema, "min_role_confidence": min_role},
     )


def apply_rules(
    df: pd.DataFrame,
    schema_roles: dict[str, str],
    rules: list[RuleSpec],
    env: dict[str, Any],
    registry: dict[str, Any],
) -> tuple[pd.DataFrame, list[RuleHit], dict[str, dict[str, Any]], dict[str, str]]:
    """
    Apply rules in priority order per column. Pure: returns new df and logs.
    """
    # Sort rules: higher priority first; stable
    rules_sorted = sorted(rules, key=lambda r: (-r.priority, r.id))

    # Pre-compile conditions & actions
    compiled = []
    for r in rules_sorted:
        cond = compile_condition(r.when)
        action, params = parse_then(r.then, registry)
        compiled.append((r, cond, (action, params)))

    df2 = df.copy(deep=True)
    hits, col_report, dropped = [], {}, {}

    # Initial metrics for per-column recordkeeping
    initial = profile_columns(df2, schema_roles, env.get("profiling", {}).get("roles", {}).get("datetime_formats", []))

    for col in list(df2.columns):
        s = df2[col]
        before_type = initial[col]["type"]
        before_role = schema_roles.get(col, "text")
        actions_taken = []

        for r, cond, (act, params) in compiled:
            # ⬇️ recompute facts for the *current* series before each rule
            facts = _quick_facts(s, name=col, schema_role=before_role, df_index=df2.index)
            ctx = {
                **facts,
                "env": env,
                "params": _resolve_params(params, env),
                "cleaning": env.get("cleaning", {}),
                "profiling": env.get("profiling", {}),
                "schema_role": before_role,
            }
            try:
                if eval_condition(cond, ctx):
                    res = act(s, ctx)
                    if isinstance(res, tuple):
                        s, note = res
                    else:
                        s, note = res, None
                    actions_taken.append(note or r.id)
            except Exception:
                continue

        # assign possibly-updated series
        df2[col] = s

        # optionally drop if marked; registry.drop_column returns empty series
        if s is not None and getattr(s, "size", 0) == 0 and col in df2.columns:
            dropped[col] = "drop_column"
            df2 = df2.drop(columns=[col])
            continue

        # update hit record
        hits.append(RuleHit(
            column=col,
            rule_id=";".join(actions_taken) if actions_taken else "noop",
            before_type=before_type,
            after_type=_series_dtype(df2[col]),
            before_role=before_role,
            after_role=None,
            notes=", ".join(actions_taken) if actions_taken else None,
        ))
        col_report[col] = {"actions": actions_taken}

    return df2, hits, col_report, dropped

# ---- helpers ----

def _resolve_params(params: dict[str, Any], env: dict[str, Any]) -> dict[str, Any]:
    """
    Deep-resolve NameRef placeholders against env. Keep literals as-is.
    Supports nested dict/list/tuple structures.
    """
    def _resolve_ref(path: str) -> Any:
        cur: Any = env
        for part in path.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                return None
        return cur

    def _walk(val: Any) -> Any:
        if isinstance(val, NameRef):
            return _resolve_ref(val.path)
        if isinstance(val, dict):
            return {k: _walk(v) for k, v in val.items()}
        if isinstance(val, (list, tuple)):
            seq = [_walk(v) for v in val]
            return seq if isinstance(val, list) else tuple(seq)
        return val

    return _walk(params) if params else {}

def _series_dtype(s: pd.Series) -> str:
    from ..nlp.roles import _dtype_str
    return _dtype_str(s)

def _collect_metric_delta(m1: dict[str, dict[str, Any]], m2: dict[str, dict[str, Any]], key: str) -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    for c in m1.keys():
        if c in m2:
            out[c] = (float(m1[c].get(key, 0.0)), float(m2[c].get(key, 0.0)))
    return out

def _collect_type_changes(m1: dict[str, dict[str, Any]], m2: dict[str, dict[str, Any]]) -> dict[str, tuple[str, str]]:
    out: dict[str, tuple[str, str]] = {}
    for c in m1.keys():
        if c in m2:
            t1 = str(m1[c].get("type"))
            t2 = str(m2[c].get("type"))
            if t1 != t2:
                out[c] = (t1, t2)
    return out


```

metrics.py:
```
from __future__ import annotations
from typing import Any, Dict, Tuple
import pandas as pd
import numpy as np

from ..nlp.roles import _dtype_str as _nlp_dtype_str  # keep dtype mapping consistent

# ---- Public API ----

def profile_columns(
    df: pd.DataFrame,
    schema_roles: dict[str, str],
    datetime_formats: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """
    Build a metrics context per column used by the DSL and reports.
    Pure: no side effects.

    Returns: { col -> metrics dict }
    Keys expected by DSL:
      - name, type, role
      - missing_pct, non_null_ratio
      - nunique, unique_ratio, cardinality
      - avg_len (strings only; else None)
      - has_time_index (bool)
      - mean, std, iqr (numeric only; else None)
      - bool_token_ratio (strings only)
      - unit_hint (optional; may be filled by upstream schema hints)

    Additional safe goodies (may be useful later):
      - min, max (numeric only; else None)
      - datetime_parse_ratio (strings w/ role 'time'; else 0.0)
      - is_monotonic_increasing (datetime or numeric)
    """
    out: dict[str, dict[str, Any]] = {}
    n = int(len(df))

    # simple heuristic; refine later if needed
    time_cols = [c for c, r in schema_roles.items() if r == "time"]
    has_time_index = len(time_cols) == 1

    for c in df.columns:
        s = df[c]
        t = _series_dtype_str(s)
        role = schema_roles.get(c, "text")

        # basic counts
        try:
            missing_pct = float(s.isna().mean()) if n else 0.0
        except Exception:
            # highly exotic dtypes – fall back to treating nothing as missing
            missing_pct = 0.0
        non_null_ratio = 1.0 - missing_pct

        nunique = _safe_nunique(s)
        unique_ratio = float(nunique) / float(n or 1)

        # type-specific metrics
        if t in {"int", "float"}:
            stats = _numeric_stats(s)
            avg_len = None
            btr = 0.0
            dt_ratio = 0.0
            is_mono = _is_monotonic_numeric(s)
        elif t == "string":
            stats = {"mean": None, "std": None, "iqr": None, "min": None, "max": None}
            avg_len = _string_avg_len(s)
            btr = _bool_token_ratio(s)
            # If schema hints this column is time-like, estimate how parseable it is
            dt_ratio = _datetime_parse_ratio(s, datetime_formats) if role == "time" else 0.0
            is_mono = False
        elif t in {"datetime", "date"}:
            stats = {"mean": None, "std": None, "iqr": None, "min": None, "max": None}
            avg_len = None
            btr = 0.0
            dt_ratio = 1.0  # already datetime-like
            is_mono = bool(getattr(s, "is_monotonic_increasing", False))
        else:
            stats = {"mean": None, "std": None, "iqr": None, "min": None, "max": None}
            avg_len = None
            btr = 0.0
            dt_ratio = 0.0
            is_mono = False

        out[c] = {
            "name": c,
            "type": t,
            "role": role,
            "missing_pct": float(missing_pct),
            "non_null_ratio": float(non_null_ratio),
            "nunique": int(nunique),
            "unique_ratio": float(unique_ratio),
            "cardinality": int(nunique),
            "avg_len": avg_len,
            "has_time_index": bool(has_time_index),
            "mean": stats["mean"],
            "std": stats["std"],
            "iqr": stats["iqr"],
            "min": stats.get("min"),
            "max": stats.get("max"),
            "bool_token_ratio": float(btr),
            "datetime_parse_ratio": float(dt_ratio),
            "is_monotonic_increasing": bool(is_mono),
            # "unit_hint": None,  # caller can enrich from schema.hints if desired
        }
    return out

# ---- Helpers (pure) ----

def _series_dtype_str(s: pd.Series) -> str:
    # use same mapping as NLP for consistency
    return _nlp_dtype_str(s)

def _safe_nunique(s: pd.Series) -> int:
    """Robust nunique that handles unhashables and treats nulls correctly."""
    try:
        # Fast path for hashables
        return int(s.nunique(dropna=True))
    except TypeError:
        # Drop real nulls first so None/pd.NA/nan aren't turned into "None"/"nan"
        non_null = s[~s.isna()]
        try:
            # Many unhashables (lists, dicts) can be compared via repr safely enough for counts
            return int(non_null.astype(str).nunique())
        except Exception:
            # Last resort: map to repr explicitly, then count
            return int(pd.Series(non_null.map(repr)).nunique())

def _string_avg_len(s: pd.Series) -> float | None:
    try:
        vals = s.dropna().astype(str)
        return float(vals.map(len).mean()) if not vals.empty else 0.0
    except Exception:
        return None

def _numeric_stats(s: pd.Series) -> Dict[str, float | None]:
    """
    Return mean, std (ddof=1), iqr, min, max for numeric-ish series.
    Coerces safely; returns None for empty results.
    """
    try:
        x = pd.to_numeric(s, errors="coerce").dropna()
        if x.empty:
            return {"mean": None, "std": None, "iqr": None, "min": None, "max": None}
        q1 = float(x.quantile(0.25))
        q3 = float(x.quantile(0.75))
        return {
            "mean": float(x.mean()),
            "std": float(x.std(ddof=1)),
            "iqr": float(q3 - q1),
            "min": float(x.min()),
            "max": float(x.max()),
        }
    except Exception:
        return {"mean": None, "std": None, "iqr": None, "min": None, "max": None}

def _bool_token_ratio(s: pd.Series) -> float:
    """
    Share of values that look like boolean tokens (true/false/yes/no/1/0…),
    after trimming & lowercasing; non-strings are coerced to str first.
    """
    try:
        vals = s.dropna().astype(str).str.strip().str.lower()
        if vals.empty:
            return 0.0
        tokens = {"true", "false", "t", "f", "y", "n", "yes", "no", "1", "0"}
        return float(vals.isin(tokens).mean())
    except Exception:
        return 0.0

def _datetime_parse_ratio(s: pd.Series, fmts: list[str] | None) -> float:
    """
    For string series, attempt parse using provided formats, falling back to pandas'
    general parser. Returns fraction of non-null values that parse to a timestamp.
    Pure; does not mutate.
    """
    try:
        x = s.dropna().astype(str)
        if x.empty:
            return 0.0
        parsed = None
        for f in (fmts or []):
            try:
                cand = pd.to_datetime(x, format=f, errors="coerce")
            except Exception:
                continue
            parsed = cand if parsed is None else parsed.fillna(cand)
        if parsed is None:
            parsed = pd.to_datetime(x, errors="coerce")
        non_null = int(x.shape[0])
        ok = int(parsed.notna().sum())
        return float(ok) / float(non_null or 1)
    except Exception:
        return 0.0

def _is_monotonic_numeric(s: pd.Series) -> bool:
    try:
        x = pd.to_numeric(s, errors="coerce")
        return bool(x.is_monotonic_increasing)
    except Exception:
        return False


```

policy.py:
```
from __future__ import annotations
from typing import Any, Tuple
from dataclasses import asdict, is_dataclass

from ..config_model.model import RootCfg
from .engine import RuleSpec

def _to_plain(obj: Any) -> Any:
    """
    Convert structured objects (Pydantic v1/v2, dataclasses, SimpleNamespace)
    into plain Python dicts/lists/primitives.
    """
    # Pydantic v2
    if hasattr(obj, "model_dump") and callable(getattr(obj, "model_dump")):
        try:
            return obj.model_dump()
        except Exception:
            pass
    # Pydantic v1
    if hasattr(obj, "dict") and callable(getattr(obj, "dict")):
        try:
            return obj.dict()
        except Exception:
            pass
    # Dataclass
    if is_dataclass(obj):
        try:
            return asdict(obj)
        except Exception:
            pass
    # SimpleNamespace / generic object with __dict__
    if hasattr(obj, "__dict__"):
        try:
            return {k: _to_plain(v) for k, v in vars(obj).items()}
        except Exception:
            pass
    # dict
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    # list/tuple
    if isinstance(obj, (list, tuple)):
        t = type(obj)
        return t(_to_plain(v) for v in obj)
    # primitive
    return obj

def build_policy_from_config(root_cfg: RootCfg, dataset_slug: str | None = None) -> tuple[list[RuleSpec], dict[str, Any]]:
    """
    Build RuleSpec list from config.cleaning.rules and create an 'env' dict with
    all names the DSL/actions can reference (e.g., datetime_formats).
    Pure: no IO. Robust to Pydantic models, dataclasses, or namespaces.
    """
    # Rules
    rules = [
        RuleSpec(
            id=str(r.id),
            priority=int(r.priority),
            when=str(r.when),
            then=str(r.then),
        )
        for r in getattr(getattr(root_cfg, "cleaning", None), "rules", [])  # safe for namespaces
    ]

    # Sections (safe getattr)
    cleaning = getattr(root_cfg, "cleaning", None)
    profiling = getattr(root_cfg, "profiling", None)
    nlp = getattr(root_cfg, "nlp", None)

    # Shortcuts with defaults
    datetime_formats = getattr(getattr(profiling, "roles", None), "datetime_formats", []) if profiling else []
    cat_cardinality_max = getattr(getattr(cleaning, "columns", None), "cat_cardinality_max", None)
    numeric_default = getattr(getattr(cleaning, "impute", None), "numeric_default", None)
    categorical_default = getattr(getattr(cleaning, "impute", None), "categorical_default", None)
    text_default = getattr(getattr(cleaning, "impute", None), "text_default", None)

    outliers = getattr(cleaning, "outliers", None)
    winsor_limits_raw = getattr(outliers, "winsor_limits", (0.01, 0.99)) if outliers else (0.01, 0.99)
    winsor_limits: Tuple[float, float] = (
        tuple(winsor_limits_raw) if isinstance(winsor_limits_raw, (list, tuple)) else (0.01, 0.99)
    )
    zscore_threshold = getattr(outliers, "zscore_threshold", 3.0) if outliers else 3.0
    iqr_multiplier = getattr(outliers, "iqr_multiplier", 1.5) if outliers else 1.5
    detect = getattr(outliers, "method", "zscore") if outliers else "zscore"

    env: dict[str, Any] = {
        "cleaning": _to_plain(cleaning) if cleaning is not None else {},
        "profiling": _to_plain(profiling) if profiling is not None else {},
        "nlp": _to_plain(nlp) if nlp is not None else {},
        "datetime_formats": datetime_formats,
        "cat_cardinality_max": cat_cardinality_max,
        "numeric_default": numeric_default,
        "categorical_default": categorical_default,
        "text_default": text_default,
        "winsor_limits": winsor_limits,
        "zscore_threshold": zscore_threshold,
        "iqr_multiplier": iqr_multiplier,
        "detect": detect,
    }

    if dataset_slug:
        env["dataset_slug"] = dataset_slug

    return rules, env


```

registry.py:
```
from __future__ import annotations
from typing import Any, Callable, Dict
import pandas as pd
from dataclasses import dataclass
import re

# -------- public types --------

@dataclass(frozen=True)
class NameRef:
    """Symbolic reference that should be resolved from ctx['env'] later."""
    path: str

# Each action gets (series, ctx) and returns a new series (and optional notes).
ActionFn = Callable[[pd.Series, dict[str, Any]], tuple[pd.Series, str | None] | pd.Series]


# -------- registry --------

def compile_actions_registry() -> dict[str, ActionFn]:
    """
    Map action names -> callables.
    Add your plug-ins here.
    """
    from .rules_builtin.types import (
        coerce_numeric_from_string,
        parse_datetime_from_string,
        cast_category_if_small,
        cast_string_dtype,
    )
    from .rules_builtin.missing import impute_numeric, impute_value, impute_datetime
    from .rules_builtin.outliers import apply_outlier_policy  # returns (series, mask)
    from .rules_builtin.text_norm import text_normalize, normalize_null_tokens
    from .rules_builtin.units import standardize_numeric_units  # returns (series, meta)

    # Wrap primitives into (s, ctx) -> (s2, notes)
    def _wrap(fn, note_fmt: str) -> ActionFn:
        def inner(s: pd.Series, ctx: dict[str, Any]):
            s2 = fn(s, **ctx.get("params", {}))
            return s2, note_fmt
        return inner

    # outliers: function returns (series, mask) → build a helpful note
    def _wrap_outliers(fn) -> ActionFn:
        def inner(s: pd.Series, ctx: dict[str, Any]):
            s2, mask = fn(s, **ctx.get("params", {}))
            n_flagged = 0
            try:
                n_flagged = int(getattr(mask, "sum", lambda: 0)())
            except Exception:
                pass
            return s2, f"outliers(n={n_flagged})"
        return inner

    # units: function returns (series, meta) → summarize meta in note
    def _wrap_units(fn) -> ActionFn:
        def inner(s: pd.Series, ctx: dict[str, Any]):
            s2, meta = fn(s, **ctx.get("params", {}))  # (series, dict)
            unit_in  = str(meta.get("unit_in", ""))
            unit_out = str(meta.get("unit_out", ""))
            rescaled = ",rescaled" if meta.get("rescaled") else ""
            return s2, f"standardize_units({unit_in}->{unit_out}{rescaled})"
        return inner

    registry: dict[str, ActionFn] = {
        "coerce_numeric":   _wrap(coerce_numeric_from_string, "coerce_numeric"),
        "parse_datetime":   _wrap(parse_datetime_from_string, "parse_datetime"),
        "cast_category":    _wrap(cast_category_if_small, "cast_category"),
        "impute":           _wrap(impute_numeric, "impute"),
        "impute_value":     _wrap(impute_value, "impute_value"),
        "materialize_missing_as": _wrap( 
            lambda s, **p: impute_value(s, **{**p, "force": True}),
            "materialize_missing_as"
        ),
        "impute_dt":        _wrap(impute_datetime, "impute_dt"),
        "cast_string": _wrap(cast_string_dtype, "cast_string"),
        "outliers":         _wrap_outliers(apply_outlier_policy),
        "text_normalize":   _wrap(text_normalize, "text_normalize"),
        "normalize_null_tokens": _wrap(normalize_null_tokens, "normalize_null_tokens"),
        "standardize_units": _wrap_units(standardize_numeric_units),
        "drop_column":      lambda s, ctx: (pd.Series(dtype="float64"), "drop_column"),
    }
    return registry


# -------- parse_then (safe, no eval) --------

def parse_then(spec: str, registry: dict[str, ActionFn]) -> tuple[ActionFn, dict[str, Any]]:
    """
    Parse a 'then' string like:
      'impute("median")'
      'parse_datetime(datetime_formats)'
      'parse_datetime(["%Y-%m-%d","%m/%d/%Y"])'
      'text_normalize(strip=True, lower=cleaning.normalize.lowercase_text)'
      'outliers(detect, zscore_threshold, iqr_multiplier, handle, winsor_limits)'

    Returns: (callable, params_dict)
    - Strings, numbers, booleans, None, list literals are parsed as literals
    - Dotted identifiers become NameRef(path) for later resolution via ctx['env']
    """

    # ---- Tokenizer ----
    class Tok:
        __slots__ = ("typ", "val", "pos")
        def __init__(self, typ: str, val: Any, pos: int) -> None:
            self.typ, self.val, self.pos = typ, val, pos
        def __repr__(self) -> str:
            return f"Tok({self.typ!r},{self.val!r}@{self.pos})"

    WHITESPACE = set(" \t\r\n")
    DIG = set("0123456789")
    ID0 = set("_abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
    IDN = ID0.union(DIG).union({"."})

    def _read_while(s: str, i: int, pred) -> tuple[str, int]:
        j = i
        n = len(s)
        while j < n and pred(s[j]):
            j += 1
        return s[i:j], j

    def _read_string(s: str, i: int) -> tuple[str, int]:
        quote = s[i]
        i += 1
        out = []
        n = len(s)
        esc = False
        while i < n:
            ch = s[i]; i += 1
            if esc:
                if ch in ['\\', '"', "'"]:
                    out.append(ch)
                elif ch == "n":
                    out.append("\n")
                elif ch == "t":
                    out.append("\t")
                elif ch == "r":
                    out.append("\r")
                else:
                    out.append(ch)
                esc = False
            else:
                if ch == "\\":
                    esc = True
                elif ch == quote:
                    return "".join(out), i
                else:
                    out.append(ch)
        raise ValueError("Unterminated string literal")

    def _read_number(s: str, i: int) -> tuple[int | float, int]:
        j = i
        n = len(s)
        if j < n and s[j] == "-":
            j += 1
        ints, j = _read_while(s, j, lambda c: c in DIG)
        if ints == "":
            raise ValueError(f"Invalid number at {i}")
        if j < n and s[j] == ".":
            j += 1
            frac, j = _read_while(s, j, lambda c: c in DIG)
            if frac == "":
                raise ValueError(f"Invalid float at {i}")
            return float(s[i:j]), j
        return int(s[i:j]), j

    def _tokenize(text: str) -> list[Tok]:
        s = text
        i, n = 0, len(s)
        toks: list[Tok] = []
        while i < n:
            ch = s[i]
            if ch in WHITESPACE:
                i += 1; continue
            if ch == "(":
                toks.append(Tok("LP", "(", i)); i += 1; continue
            if ch == ")":
                toks.append(Tok("RP", ")", i)); i += 1; continue
            if ch == ",":
                toks.append(Tok("COMMA", ",", i)); i += 1; continue
            if ch == "=":
                toks.append(Tok("EQ", "=", i)); i += 1; continue
            if ch == "[":
                toks.append(Tok("LB", "[", i)); i += 1; continue
            if ch == "]":
                toks.append(Tok("RB", "]", i)); i += 1; continue

            if ch in ("'", '"'):
                val, j = _read_string(s, i)
                toks.append(Tok("STR", val, i)); i = j; continue

            if ch in DIG or (ch == "-" and i + 1 < n and s[i + 1] in DIG):
                val, j = _read_number(s, i)
                toks.append(Tok("NUM", val, i)); i = j; continue

            if ch in ID0:
                raw, j = _read_while(s, i, lambda c: c in IDN)
                low = raw.lower()
                if low in ("true", "false"):
                    toks.append(Tok("BOOL", True if low == "true" else False, i))
                elif low in ("none", "null"):
                    toks.append(Tok("NONE", None, i))
                else:
                    toks.append(Tok("ID", raw, i))
                i = j; continue

            raise ValueError(f"Unexpected character {ch!r} at position {i}")
        toks.append(Tok("EOF", None, n))
        return toks

    # ---- Parser ----
    class P:
        def __init__(self, toks: list[Tok]) -> None:
            self.t = toks; self.i = 0
        def peek(self) -> Tok:
            return self.t[self.i]
        def eat(self, typ: str | None = None) -> Tok:
            tok = self.peek()
            if typ and tok.typ != typ:
                raise ValueError(f"Expected {typ}, got {tok.typ} at {tok.pos}")
            self.i += 1
            return tok

        def parse(self) -> tuple[str, list[Any], dict[str, Any]]:
            # func '(' args? ')'
            name_tok = self.eat("ID")
            self.eat("LP")
            pos_args: list[Any] = []
            kw_args: dict[str, Any] = {}
            if self.peek().typ != "RP":
                while True:
                    nxt = self.peek()
                    if nxt.typ == "ID":
                        # look-ahead for '=' to decide kw vs positional
                        save_i = self.i
                        key_tok = self.eat("ID")
                        if self.peek().typ == "EQ":
                            self.eat("EQ")
                            val = self.parse_value()
                            kw_args[key_tok.val] = val
                        else:
                            self.i = save_i
                            val = self.parse_value()
                            pos_args.append(val)
                    else:
                        val = self.parse_value()
                        pos_args.append(val)
                    if self.peek().typ == "COMMA":
                        self.eat("COMMA"); continue
                    break
            self.eat("RP")
            return name_tok.val, pos_args, kw_args

        def parse_value(self) -> Any:
            tok = self.peek()
            if tok.typ == "STR":
                return self.eat("STR").val
            if tok.typ == "NUM":
                return self.eat("NUM").val
            if tok.typ == "BOOL":
                return self.eat("BOOL").val
            if tok.typ == "NONE":
                self.eat("NONE"); return None
            if tok.typ == "LB":
                return self.parse_list()
            if tok.typ == "ID":
                # dotted identifier => NameRef
                return NameRef(self.eat("ID").val)
            raise ValueError(f"Unexpected token {tok} in value")

        def parse_list(self) -> list[Any]:
            self.eat("LB")
            items: list[Any] = []
            if self.peek().typ != "RB":
                while True:
                    items.append(self.parse_value())
                    if self.peek().typ == "COMMA":
                        self.eat("COMMA"); continue
                    break
            self.eat("RB")
            return items

    # positional param names for known actions
    ACTION_POS_PARAMS: dict[str, list[str]] = {
        "coerce_numeric":    ["unit_hint"],
        "parse_datetime":    ["formats"],
        "cast_category":     ["max_card"],
        "impute":            ["method"],
        "impute_value":      ["value"],
         "materialize_missing_as": ["value"],
        "cast_string":              [],
        "impute_dt": ["method", "value"],
        "outliers":          ["method", "zscore_threshold", "iqr_multiplier", "handle", "winsor_limits"],
        "text_normalize":    ["strip", "lower"],
        "normalize_null_tokens": ["null_tokens", "case_insensitive", "apply_text_normalize_first"], 
        "standardize_units": ["unit_hint"],
        "drop_column":       [],
    }

    toks = _tokenize(spec)
    parser = P(toks)
    action_name, pos_args, kw_args = parser.parse()

    if action_name not in registry:
        raise ValueError(f"Unknown action '{action_name}' in spec: {spec!r}")

    # map positional → names
    names = ACTION_POS_PARAMS.get(action_name, [])
    if len(pos_args) > len(names):
        raise ValueError(
            f"Too many positional args for {action_name}: expected ≤ {len(names)}, got {len(pos_args)}"
        )

    params: dict[str, Any] = {}
    for i, a in enumerate(pos_args):
        if i < len(names):
            params[names[i]] = a
    params.update(kw_args)  # kwargs override

    return registry[action_name], params


```

report.py:
```
from __future__ import annotations
from typing import Any, TYPE_CHECKING
import pandas as pd
from dataclasses import is_dataclass, asdict

# Avoid circular import at runtime: only import RuleHit for type checking
if TYPE_CHECKING:  # pragma: no cover
    from .engine import RuleHit  # noqa: F401


def _coerce_float(x: Any) -> float | None:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _rescore_to_dict(rescore: Any) -> dict[str, Any]:
    """
    Accept either a plain dict or a dataclass (RescoreResult) and return a serializable dict.
    Unknown shapes fall back to {}.
    """
    if isinstance(rescore, dict):
        # Coerce numeric fields to float for consistency
        out = dict(rescore)
        for k in (
            "schema_conf_before",
            "schema_conf_after",
            "avg_role_conf_before",
            "avg_role_conf_after",
        ):
            if k in out:
                out[k] = _coerce_float(out[k])
        return out
    if is_dataclass(rescore):
        d = asdict(rescore)
        for k in (
            "schema_conf_before",
            "schema_conf_after",
            "avg_role_conf_before",
            "avg_role_conf_after",
        ):
            if k in d:
                d[k] = _coerce_float(d[k])
        return d
    return {}


def _columns_added_dropped(df_before: pd.DataFrame, df_after: pd.DataFrame) -> tuple[list[str], list[str]]:
    before = list(map(str, df_before.columns))
    after = list(map(str, df_after.columns))
    added = [c for c in after if c not in before]
    dropped = [c for c in before if c not in after]
    return added, dropped


def _type_changes_map(
    metrics_before: dict[str, dict[str, Any]],
    metrics_after: dict[str, dict[str, Any]],
) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    common = set(metrics_before.keys()) & set(metrics_after.keys())
    for c in sorted(common):
        t1 = str(metrics_before[c].get("type"))
        t2 = str(metrics_after[c].get("type"))
        if t1 != t2:
            out[c] = {"before": t1, "after": t2}
    return out


def _type_changes_tuples(
    metrics_before: dict[str, dict[str, Any]],
    metrics_after: dict[str, dict[str, Any]],
) -> dict[str, tuple[str, str]]:
    out: dict[str, tuple[str, str]] = {}
    common = set(metrics_before.keys()) & set(metrics_after.keys())
    for c in sorted(common):
        t1 = str(metrics_before[c].get("type"))
        t2 = str(metrics_after[c].get("type"))
        if t1 != t2:
            out[c] = (t1, t2)
    return out


def _missing_pct_delta_map(
    metrics_before: dict[str, dict[str, Any]],
    metrics_after: dict[str, dict[str, Any]],
) -> dict[str, dict[str, float | None]]:
    out: dict[str, dict[str, float | None]] = {}
    common = set(metrics_before.keys()) & set(metrics_after.keys())
    for c in sorted(common):
        b = _coerce_float(metrics_before[c].get("missing_pct"))
        a = _coerce_float(metrics_after[c].get("missing_pct"))
        if b is None and a is None:
            continue
        out[c] = {"before": b, "after": a, "delta": (None if (a is None or b is None) else (a - b))}
    return out


def _missing_pct_delta_tuples(
    metrics_before: dict[str, dict[str, Any]],
    metrics_after: dict[str, dict[str, Any]],
) -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    common = set(metrics_before.keys()) & set(metrics_after.keys())
    for c in sorted(common):
        b = float(metrics_before[c].get("missing_pct", 0.0) or 0.0)
        a = float(metrics_after[c].get("missing_pct", 0.0) or 0.0)
        out[c] = (b, a)
    return out


def _actions_by_column(rule_hits: list["RuleHit"] | list[dict[str, Any]]) -> dict[str, list[str]]:
    """
    Accepts either RuleHit objects or plain dicts with keys: column, notes, rule_id.
    Returns {column: [action notes...]} with de-duplication.
    """
    out: dict[str, list[str]] = {}
    for h in rule_hits:
        column = getattr(h, "column", None)
        if column is None and isinstance(h, dict):
            column = h.get("column")
        if column is None:
            continue

        notes = getattr(h, "notes", None)
        if notes is None and isinstance(h, dict):
            notes = h.get("notes")

        rule_id = getattr(h, "rule_id", None)
        if rule_id is None and isinstance(h, dict):
            rule_id = h.get("rule_id")

        if notes:
            parts = [p.strip() for p in str(notes).split(",") if p.strip()]
        elif rule_id:
            parts = [p.strip() for p in str(rule_id).split(";") if p.strip()]
        else:
            parts = []

        out.setdefault(column, [])
        out[column].extend(parts)

    # De-dupe while preserving order
    for k, vals in out.items():
        seen = set()
        deduped = []
        for v in vals:
            if v not in seen:
                seen.add(v)
                deduped.append(v)
        out[k] = deduped
    return out


def build_iteration_report(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    metrics_before: dict[str, dict[str, Any]],
    metrics_after: dict[str, dict[str, Any]],
    rule_hits: list["RuleHit"] | list[dict[str, Any]],
    rescore: dict[str, Any] | Any,  # supports dict or dataclass
) -> dict[str, Any]:
    """
    Assemble a serializable dict for loop_iter_k.json and NLG narrative.
    """
    rows_before, cols_before = int(len(df_before)), int(df_before.shape[1])
    rows_after, cols_after = int(len(df_after)), int(df_after.shape[1])

    added, dropped = _columns_added_dropped(df_before, df_after)
    type_chg_map = _type_changes_map(metrics_before, metrics_after)
    type_chg_tup = _type_changes_tuples(metrics_before, metrics_after)
    miss_delta_map = _missing_pct_delta_map(metrics_before, metrics_after)
    miss_delta_tup = _missing_pct_delta_tuples(metrics_before, metrics_after)

    # Serialize hits once for reuse
    hits_serialized = [
        (h.__dict__ if hasattr(h, "__dict__") else dict(h))  # tolerate dicts from tests
        for h in rule_hits
    ]
    actions_map = _actions_by_column(rule_hits)

    # Small human-readable summary
    cols_changed_count = len(type_chg_map)
    rules_triggered_count = sum(len(v) for v in actions_map.values())
    summary = (
        f"Rows: {rows_before} → {rows_after}. "
        f"Columns: {cols_before} → {cols_after}. "
        f"Dropped: {len(dropped)}, Added: {len(added)}, Type changes: {cols_changed_count}. "
        f"Rule applications: {rules_triggered_count}."
    )

    report: dict[str, Any] = {
        "shape": {
            "before": {"rows": rows_before, "columns": cols_before},
            "after": {"rows": rows_after, "columns": cols_after},
        },
        "columns": {
            "added": added,
            "dropped": dropped,
            "type_changes": type_chg_map,
        },
        "metrics": {
            "missing_pct": miss_delta_map,
        },
        "rules": {
            "hits": hits_serialized,
            "per_column_actions": actions_map,
            "total_applications": rules_triggered_count,
        },
        "rescore": _rescore_to_dict(rescore),
        "narrative": summary,
    }

    # ---- Back-compat top-level fields expected by older engine tests ----
    report.update(
        {
            "row_count_before": rows_before,
            "row_count_after": rows_after,
            "column_count_before": cols_before,
            "column_count_after": cols_after,
            "rule_hits": hits_serialized,
            "metrics_delta": {
                "missing_pct": miss_delta_tup,
                "type_changes": type_chg_tup,
            },
        }
    )

    return report


```

rescore.py:
```
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any
import pandas as pd

from ..config_model.model import RootCfg
from ..nlp.schema import ProposedSchema
from .metrics import profile_columns
from .dsl import compile_condition, eval_condition
from .registry import compile_actions_registry, parse_then, NameRef
from .rescore import rescore_after_clean
from .report import build_iteration_report

# ---- Data classes ----

@dataclass(frozen=True)
class RuleSpec:
    id: str
    priority: int
    when: str
    then: str

@dataclass(frozen=True)
class RuleHit:
    column: str
    rule_id: str
    before_type: str
    after_type: str
    before_role: str
    after_role: str | None
    notes: str | None = None

@dataclass(frozen=True)
class CleaningResult:
    clean_df: pd.DataFrame
    report: dict[str, Any]
    rescore: dict[str, Any]
    column_report: dict[str, dict[str, Any]]
    dropped: dict[str, str]

# ---- helpers ----

def _dtype_tag(s: pd.Series) -> str:
    from ..nlp.roles import _dtype_str
    return _dtype_str(s)


def _quick_facts(series: pd.Series, *, name: str, schema_role: str, df_index) -> dict:
    n = len(series) or 1
    t = _dtype_tag(series)
    try:
        nunique = int(series.nunique(dropna=True))
    except TypeError:
        nunique = int(series.astype(str).nunique(dropna=True))
    avg_len = None
    if t == "string":
        vals = series.dropna().astype(str)
        avg_len = float(vals.map(len).mean()) if not vals.empty else 0.0
    missing_pct = float(series.isna().mean())
    return {
        "name": name,
        "type": t,
        "role": schema_role,                  # keep schema role stable
        "missing_pct": missing_pct,
        "non_null_ratio": 1.0 - missing_pct,
        "nunique": nunique,
        "unique_ratio": float(nunique) / float(n),
        "cardinality": nunique,
        "avg_len": avg_len,
        "has_time_index": pd.api.types.is_datetime64_any_dtype(df_index),
        # extras used in some rules; safe defaults if not needed
        "mean": None, "std": None, "iqr": None,
        "bool_token_ratio": 0.0,
        "datetime_parse_ratio": 0.0,
        "is_monotonic_increasing": False,
    }

# ---- Public API ----

def run_clean_pass(
    df: pd.DataFrame,
    proposed_schema: ProposedSchema,
    cfg: RootCfg,
) -> CleaningResult:
    """
    One pure cleaning pass:
      - profile
      - apply rules (priority order)
      - re-profile
      - rescore (NLP)
      - build iteration report
    """
    from .policy import build_policy_from_config
    rules, env = build_policy_from_config(cfg)
    registry = compile_actions_registry()

    schema_roles = {c["name"]: c["role"] for c in proposed_schema.to_dict()["columns"]}

    metrics_before = profile_columns(df, schema_roles, cfg.profiling.roles.datetime_formats)
    df_after, hits, col_report, dropped = apply_rules(df, schema_roles, rules, env, registry)
    metrics_after = profile_columns(df_after, schema_roles, cfg.profiling.roles.datetime_formats)

    # Rescore using NLP before/after confidences (returns a dataclass)
    rescore_result = rescore_after_clean(
        df_after,
        proposed_schema,
        cfg.profiling.roles,
        cfg.nlp,
    )
    # Build a single canonical report
    report = build_iteration_report(
        df_before=df,
        df_after=df_after,
        metrics_before=metrics_before,
        metrics_after=metrics_after,
        rule_hits=hits,
        rescore=rescore_result,  # report handles dataclass/dict
    )

    return CleaningResult(
        clean_df=df_after,
        report=report,
        rescore=asdict(rescore_result),
        column_report=col_report,
        dropped=dropped,
    )

def apply_rules(
    df: pd.DataFrame,
    schema_roles: dict[str, str],
    rules: list[RuleSpec],
    env: dict[str, Any],
    registry: dict[str, Any],
) -> tuple[pd.DataFrame, list[RuleHit], dict[str, dict[str, Any]], dict[str, str]]:
    """
    Apply rules in priority order per column. Pure: returns new df and logs.
    """
    # Sort rules: higher priority first; stable
    rules_sorted = sorted(rules, key=lambda r: (-r.priority, r.id))

    # Pre-compile conditions & actions
    compiled = []
    for r in rules_sorted:
        cond = compile_condition(r.when)
        action, params = parse_then(r.then, registry)
        compiled.append((r, cond, (action, params)))

    df2 = df.copy(deep=True)
    hits, col_report, dropped = [], {}, {}

    # Initial metrics for per-column recordkeeping
    initial = profile_columns(df2, schema_roles, env.get("profiling", {}).get("roles", {}).get("datetime_formats", []))

    for col in list(df2.columns):
        s = df2[col]
        before_type = initial[col]["type"]
        before_role = schema_roles.get(col, "text")
        actions_taken = []

        for r, cond, (act, params) in compiled:
            # ⬇️ recompute facts for the *current* series before each rule
            facts = _quick_facts(s, name=col, schema_role=before_role, df_index=df2.index)
            ctx = {
                **facts,
                "env": env,
                "params": _resolve_params(params, env),
                "cleaning": env.get("cleaning", {}),
                "profiling": env.get("profiling", {}),
                "schema_role": before_role,
            }
            try:
                if eval_condition(cond, ctx):
                    res = act(s, ctx)
                    if isinstance(res, tuple):
                        s, note = res
                    else:
                        s, note = res, None
                    actions_taken.append(note or r.id)
            except Exception:
                continue

        # assign possibly-updated series
        df2[col] = s

        # optionally drop if marked; registry.drop_column returns empty series
        if s is not None and getattr(s, "size", 0) == 0 and col in df2.columns:
            dropped[col] = "drop_column"
            df2 = df2.drop(columns=[col])
            continue

        # update hit record
        hits.append(RuleHit(
            column=col,
            rule_id=";".join(actions_taken) if actions_taken else "noop",
            before_type=before_type,
            after_type=_series_dtype(df2[col]),
            before_role=before_role,
            after_role=None,
            notes=", ".join(actions_taken) if actions_taken else None,
        ))
        col_report[col] = {"actions": actions_taken}

    return df2, hits, col_report, dropped

# ---- helpers ----

def _resolve_params(params: dict[str, Any], env: dict[str, Any]) -> dict[str, Any]:
    """
    Deep-resolve NameRef placeholders against env. Keep literals as-is.
    Supports nested dict/list/tuple structures.
    """
    def _resolve_ref(path: str) -> Any:
        cur: Any = env
        for part in path.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                return None
        return cur

    def _walk(val: Any) -> Any:
        if isinstance(val, NameRef):
            return _resolve_ref(val.path)
        if isinstance(val, dict):
            return {k: _walk(v) for k, v in val.items()}
        if isinstance(val, (list, tuple)):
            seq = [_walk(v) for v in val]
            return seq if isinstance(val, list) else tuple(seq)
        return val

    return _walk(params) if params else {}

def _series_dtype(s: pd.Series) -> str:
    from ..nlp.roles import _dtype_str
    return _dtype_str(s)

def _collect_metric_delta(m1: dict[str, dict[str, Any]], m2: dict[str, dict[str, Any]], key: str) -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    for c in m1.keys():
        if c in m2:
            out[c] = (float(m1[c].get(key, 0.0)), float(m2[c].get(key, 0.0)))
    return out

def _collect_type_changes(m1: dict[str, dict[str, Any]], m2: dict[str, dict[str, Any]]) -> dict[str, tuple[str, str]]:
    out: dict[str, tuple[str, str]] = {}
    for c in m1.keys():
        if c in m2:
            t1 = str(m1[c].get("type"))
            t2 = str(m2[c].get("type"))
            if t1 != t2:
                out[c] = (t1, t2)
    return out


```