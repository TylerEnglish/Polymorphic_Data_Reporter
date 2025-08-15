```
polymorphic-data-reporter/
├─ config/
│  └─ config.toml
├─ data/
│  ├─ raw/
│  ├─ bronze/
│  ├─ silver/
│  └─ gold/
│     └─ <dataset_slug>/
│        ├─ tables/
│        ├─ artifacts/
│        ├─ reports/
│        └─ manifest.json
├─ docker/
│  ├─ docker-compose.yml
│  ├─ airflow.Dockerfile
│  ├─ chrome.Dockerfile
│  └─ env/
│     ├─ airflow.env
│     └─ minio.env
├─ airflow/
│  ├─ dags/
│  │  └─ polymorphic_report_dag.py
│  └─ include/
├─ schemas/
│  └─ <dataset_slug>.schema.toml
├─ src/
│  ├─ config_model/
│  │  ├─ __init__.py
│  │  └─ model.py
│  ├─ io/
│  │  ├─ storage.py          # local + MinIO S3 abstraction
│  │  ├─ readers.py          # csv/json/parquet/duckdb
│  │  ├─ writers.py          # csv/json/parquet/html/png/pdf
│  │  └─ catalog.py
│  ├─ ingestion/
│  │  └─ ingest.py
│  ├─ profiling/
│  │  ├─ roles.py
│  │  ├─ stats.py
│  │  └─ profile.py
│  ├─ cleaning/              # <<< new: all cleaning rulepacks + engine
│  │  ├─ engine.py           # compiles + executes rule graph (pure)
│  │  ├─ registry.py         # registers rules from packs
│  │  ├─ policy.py           # builds policy from config (currying)
│  │  ├─ dsl.py              # simple condition DSL (missing_pct > 0.9 etc.)
│  │  ├─ rules_builtin/
│  │  │  ├─ types.py         # coercions, datetime parse, category cast
│  │  │  ├─ missing.py       # numeric/cat/text imputation
│  │  │  ├─ outliers.py      # zscore/iqr detect + flag/winsorize/drop
│  │  │  ├─ text_norm.py     # trim, case, unicode normalize
│  │  │  ├─ units.py         # unit standardization (config-driven)
│  │  │  └─ prune.py         # drop sparse/constant columns, quarantine extras
│  │  └─ rules_custom/       # your future plug-ins (per project/dataset)
│  ├─ topics/
│  │  ├─ candidates.py
│  │  ├─ scoring.py
│  │  └─ select.py
│  ├─ layout/
│  │  ├─ planner.py
│  │  └─ chart_picker.py
│  ├─ chart/                 # <<< new: one home for all chart families
│  │  ├─ __init__.py
│  │  ├─ common.py           # theme, axes, ids, export (HTML/PNG)
│  │  ├─ change_over_time.py # Time charts
│  │  ├─ deviation.py        # diverging bar, spine, surplus/deficit
│  │  ├─ correlation.py      # scatter/bubble, xy-heatmap, col+line
│  │  ├─ ranking.py          # ordered bar, lollipop, slope, bump, strip
│  │  ├─ distribution.py     # hist/violin/box/
│  │  ├─ magnitude.py        # paired bars, radar, parcoords
│  │  ├─ part_to_whole.py    # stacked, treemap, pie (guarded)
│  │  ├─ spatial.py          # choropleth, proportional symbol
│  │  └─ flow.py             # sankey / (network extras later)
│  ├─ nlg/
│  │  ├─ narrative_constants.py
│  │  └─ narrative.py
│  ├─ nlg/
│  │  ├─ bootstrap.py
│  │  ├─ roles.py
│  │  ├─ schema_io.py
│  │  └─ schema.py
│  ├─ reports/
│  │  ├─ base.py
│  │  ├─ generators/
│  │  │  ├─ csv_report.py
│  │  │  ├─ json_report.py
│  │  │  ├─ parquet_report.py
│  │  │  ├─ chart_report.py
│  │  │  ├─ html_report.py
│  │  │  └─ pdf_report.py
│  │  └─ orchestrator.py
│  ├─ templating/
│  │  ├─ env.py
│  │  └─ templates/
│  │     ├─ base.html
│  │     ├─ section.html
│  │     ├─ components/
│  │     │  ├─ table.html
│  │     │  ├─ chart_embed.html
│  │     │  └─ badges.html
│  │     └─ styles/theme.css
│  ├─ gold/
│  │  └─ materialize.py
│  ├─ manifest/
│  │  └─ build.py
│  └─ utils/
│     ├─ fp.py
│     ├─ ids.py
│     ├─ time.py
│     └─ log.py
├─ scripts/
└─ tests/
```


config/config.toml:
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
max_iter = 10
min_improvement = 0.03
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

[[cleaning.rules]]
id = "coerce-numeric"
priority = 100
when = 'role == "numeric" and type == "string"'
then = 'coerce_numeric()'

[[cleaning.rules]]
id = "parse-datetime"
priority = 100
when = 'role == "time" and type == "string"'
then = 'parse_datetime(datetime_formats)'

[[cleaning.rules]]
id = "impute-numeric-time"
priority = 90
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
when = 'role == "categorical" and missing_pct > 0'
then = 'impute_value(categorical_default)'

[[cleaning.rules]]
id = "impute-text"
priority = 80
when = 'role == "text" and missing_pct > 0'
then = 'impute_value(text_default)'

[[cleaning.rules]]
id = "flag-outliers"
priority = 70
when = 'role == "numeric"'
then = 'outliers(detect, zscore_threshold, iqr_multiplier, handle, winsor_limits)'

[[cleaning.rules]]
id = "normalize-text"
priority = 60
when = 'role == "text"'
then = 'text_normalize(strip=cleaning.normalize.strip_text, lower=cleaning.normalize.lowercase_text)'

[[cleaning.rules]]
id = "enforce-categorical"
priority = 50
when = 'role == "categorical" and cardinality <= cleaning.columns.cat_cardinality_max'
then = 'cast_category()'

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


src/config_model/model.py:
```
from __future__ import annotations
from typing import List, Literal, Optional
from dataclasses import dataclass
from pathlib import Path
import os
from pydantic import (
    BaseModel,
    Field,
    ValidationError,
    model_validator,
    ConfigDict,
    PrivateAttr, 
)


# ---------- Leaf models ----------

class EnvCfg(BaseModel):
    project_name: str
    timezone: str
    seed: int = 42
    theme: str = "dark_blue"


class StorageLocalCfg(BaseModel):
    enabled: bool = True
    raw_root: str
    gold_root: str


class StorageMinioCfg(BaseModel):
    enabled: bool = False
    endpoint: str = "http://minio:9000"
    access_key: str = "admin"
    secret_key: str = "admin"
    secure: bool = False
    bucket: str = "datasets"
    raw_prefix: str = "raw/"
    gold_prefix: str = "gold/"


class AuthCfg(BaseModel):
    username: str = "admin"
    password: str = "admin"


class SourcesCfg(BaseModel):
    locations: List[str] = []


class DuckDBCfg(BaseModel):
    persist: bool = True


class ProfilingRolesCfg(BaseModel):
    cat_cardinality_max: int = 120
    datetime_formats: List[str] = ["%Y-%m-%d", "%m/%d/%Y", "%Y-%m-%d %H:%M:%S"]


class ProfilingOutliersCfg(BaseModel):
    method: Literal["zscore", "iqr"] = "zscore"
    zscore_threshold: float = 3.0
    iqr_multiplier: float = 1.5


class CleaningColumnsCfg(BaseModel):
    drop_missing_pct: float = 0.90
    min_unique_ratio: float = 0.001
    always_keep: List[str] = ["id", "date", "timestamp"]
    cat_cardinality_max: int = 200


class CleaningNormalizeCfg(BaseModel):
    strip_text: bool = True
    lowercase_text: bool = False
    standardize_dates: bool = True
    enforce_categorical: bool = True


class CleaningImputeCfg(BaseModel):
    numeric_default: Literal["mean", "median", "ffill", "bfill", "interpolate"] = "median"
    categorical_default: str = "Unknown"
    text_default: str = "N/A"
    time_aware_interpolation: bool = True


class CleaningOutliersCfg(BaseModel):
    method: Literal["zscore", "iqr"] = "zscore"
    zscore_threshold: float = 3.0
    iqr_multiplier: float = 1.5
    handle: Literal["flag", "winsorize", "drop"] = "flag"
    winsor_limits: List[float] = [0.01, 0.99]


class CleaningRule(BaseModel):
    id: str
    priority: int = 50
    when: str
    then: str


class TopicsThresholdsCfg(BaseModel):
    min_corr_for_scatter: float = 0.35
    min_slope_for_trend: float = 0.02
    max_categories_bar: int = 20
    max_series_line: int = 8
    max_charts_total: int = 12


class ChartsCfg(BaseModel):
    max_charts_per_topic: int = 6
    facet_max_series: int = 8
    topk_categories: int = 20
    prefer_small_multiples: bool = True
    allow_pie_when_n_le: int = 5
    enable_advanced: List[str] = []
    export_static_png: bool = True
    # PNG (Playwright) settings
    png_width: int = 1200
    png_height: int = 700
    png_scale: float = 2.0


class WeightsCfg(BaseModel):
    suitability: float = 3.0
    effect_size: float = 2.5
    signal_quality: float = 2.0
    readability: float = 1.5
    complexity: float = 1.0


class ReportsEnabledCfg(BaseModel):
    # allow alias-based population so TOML key "json" still works
    model_config = ConfigDict(populate_by_name=True)

    csv: bool = True
    json_enabled: bool = Field(True, alias="json")  # <-- new name, same TOML key
    parquet: bool = True
    charts: bool = True
    html: bool = True
    pdf: bool = True


class ReportsHtmlCfg(BaseModel):
    template: str = "base.html"
    title_prefix: str = "Report:"
    embed_interactive: bool = True


class ReportsPdfCfg(BaseModel):
    engine: Literal["chromium"] = "chromium"
    page_size: str = "Letter"
    margins: str = "0.5in"


class AirflowCfg(BaseModel):
    dag_id: str = "polymorphic_report_dag"
    schedule: str = "0 2 * * *"
    max_active_runs: int = 1
    catchup: bool = False
    concurrency: int = 8
    task_retries: int = 2
    username: str = "admin"
    password: str = "admin"


class DockerCfg(BaseModel):
    compose_file: str = "docker/docker-compose.yml"
    airflow_image: str = "local/airflow:latest"
    chrome_image: str = "local/chrome:latest"


class LoggingCfg(BaseModel):
    level: str = "INFO"
    structured_json: bool = True


class PublishingCfg(BaseModel):
    enabled: bool = False
    target: str = "s3://datasets/gold-exports/"
    access_key: str = "admin"
    secret_key: str = "admin"

class RoleScoringCfg(BaseModel):
    # weights
    name_weight: float = 0.45
    value_weight: float = 0.55

    # thresholds
    bool_token_min_ratio: float = 0.75
    date_parse_min_ratio: float = 0.60
    unique_id_ratio: float = 0.95
    categorical_max_unique_ratio: float = 0.02
    text_min_avg_len: float = 8.0
    min_non_null_ratio: float = 0.10

    # bonuses/penalties
    bonus_id_name: float = 0.10
    penalize_bool_for_many_tokens: float = 0.05

    @model_validator(mode="after")
    def _weights_ok(self):
        s = float(self.name_weight) + float(self.value_weight)
        # keep it permissive: just prevent both zeros
        if s <= 0:
            # default back to your TOML defaults
            object.__setattr__(self, "name_weight", 0.45)
            object.__setattr__(self, "value_weight", 0.55)
        return self

class NLPCfg(BaseModel):
    sample_rows: int = 5000
    min_schema_confidence: float = 0.85
    min_role_confidence: float = 0.80
    max_iter: int = 3
    min_improvement: float = 0.03
    enable_domain_templates: bool = True
    granularity: Literal["slug", "subdir", "file"] = "slug"
    role_scoring: RoleScoringCfg = RoleScoringCfg()


class NLGCfg(BaseModel):
    inventory_key: str = "_inventory"
    narrative_filename: str = "narrative.txt"

# ---------- Grouped/nested models ----------

# Grouped/nested models
class StorageCfg(BaseModel):
    local: StorageLocalCfg
    minio: StorageMinioCfg

class ProfilingCfg(BaseModel):
    roles: ProfilingRolesCfg
    outliers: ProfilingOutliersCfg

class CleaningCfg(BaseModel):
    columns: CleaningColumnsCfg
    normalize: CleaningNormalizeCfg
    impute: CleaningImputeCfg
    outliers: CleaningOutliersCfg
    rules: list[CleaningRule] = []

class TopicsCfg(BaseModel):
    thresholds: TopicsThresholdsCfg

class ReportsCfg(BaseModel):
    enabled_generators: ReportsEnabledCfg
    html: ReportsHtmlCfg
    pdf: ReportsPdfCfg

class RootCfg(BaseModel):
    model_config = ConfigDict(extra="ignore")

    env: EnvCfg
    storage: StorageCfg
    auth: AuthCfg
    sources: SourcesCfg
    duckdb: DuckDBCfg
    profiling: ProfilingCfg
    cleaning: CleaningCfg
    topics: TopicsCfg
    charts: ChartsCfg
    weights: WeightsCfg
    reports: ReportsCfg
    airflow: AirflowCfg
    docker: DockerCfg
    logging: LoggingCfg
    publishing: PublishingCfg

    nlp: NLPCfg = NLPCfg()
    nlg: NLGCfg = NLGCfg()

    # Private attribute (not a field); used only to resolve relative paths
    _config_dir: Optional[Path] = PrivateAttr(default=None)  # <-- now defined

    @model_validator(mode="after")
    def _normalize_paths(self):
        if self._config_dir:
            # If the config file lives in a conventional "config" folder,
            # normalize relative paths against the PROJECT root (parent of "config").
            # Otherwise, fall back to resolving relative to the config file's directory.
            base_dir = self._config_dir.parent if self._config_dir.name.lower() == "config" else self._config_dir

            def _abs(p: str) -> str:
                pp = Path(p)
                return str(pp if pp.is_absolute() else (base_dir / pp).resolve())

            self.storage.local.raw_root = _abs(self.storage.local.raw_root)
            self.storage.local.gold_root = _abs(self.storage.local.gold_root)
        return self

    @classmethod
    def from_toml(cls, path: str | os.PathLike[str]) -> "RootCfg":
        try:
            import tomllib  # py>=3.11
        except Exception:
            import tomli as tomllib

        p = Path(path)

        def _parse_raw_dict() -> dict:
            # 1) Try normal binary parse
            try:
                with p.open("rb") as f:
                    return tomllib.load(f)
            except Exception:
                pass

            # 2) Retry: decode with utf-8-sig (strips BOM), strip accidental wrappers, then loads()
            text = p.read_text(encoding="utf-8-sig", errors="replace")
            # Remove accidental code fences or stray zero-width chars at the edges
            # Common paste artifact: begins with ``` or has \ufeff BOM
            cleaned = text.strip()
            if cleaned.startswith("```"):
                # Trim leading and trailing fences if present
                cleaned = cleaned.lstrip("`").strip()
                # If there is a trailing fence, remove it
                if cleaned.endswith("```"):
                    cleaned = cleaned.rstrip("`").strip()

            # Some editors leave a non-breaking space at start; remove non-ASCII spaces
            cleaned = cleaned.lstrip("\ufeff\u200b\u200c\u200d\u2060")

            try:
                return tomllib.loads(cleaned)
            except Exception as e:
                # Surface a helpful message including first 80 chars for diagnostics
                snippet = cleaned[:80].replace("\n", "\\n")
                raise RuntimeError(
                    f"Failed to parse TOML at {p} after BOM/cleanup. "
                    f"First chars: {snippet!r}"
                ) from e

        raw = _parse_raw_dict()

        # --- ensure nested dicts exist (unchanged) ---
        raw.setdefault("env", {})
        raw.setdefault("storage", {})
        raw["storage"].setdefault("local", {})
        raw["storage"].setdefault("minio", {})
        raw.setdefault("auth", {})
        raw.setdefault("sources", {})
        raw.setdefault("nlp", {})
        raw.setdefault("nlg", {})
        raw["nlp"].setdefault("role_scoring", {})
        raw.setdefault("duckdb", {})
        raw.setdefault("profiling", {})
        raw["profiling"].setdefault("roles", {})
        raw["profiling"].setdefault("outliers", {})
        raw.setdefault("cleaning", {})
        raw["cleaning"].setdefault("columns", {})
        raw["cleaning"].setdefault("normalize", {})
        raw["cleaning"].setdefault("impute", {})
        raw["cleaning"].setdefault("outliers", {})
        raw["cleaning"].setdefault("rules", [])
        raw.setdefault("topics", {})
        raw["topics"].setdefault("thresholds", {})
        raw.setdefault("charts", {})
        raw.setdefault("weights", {})
        raw.setdefault("reports", {})
        raw["reports"].setdefault("enabled_generators", {})
        raw["reports"].setdefault("html", {})
        raw["reports"].setdefault("pdf", {})
        raw.setdefault("airflow", {})
        raw.setdefault("docker", {})
        raw.setdefault("logging", {})
        raw.setdefault("publishing", {})

        # shim: TOML uses cleaning.outliers.detect -> model expects .method
        if "detect" in raw["cleaning"]["outliers"] and "method" not in raw["cleaning"]["outliers"]:
            raw["cleaning"]["outliers"]["method"] = raw["cleaning"]["outliers"]["detect"]

        cfg = cls(
            env=EnvCfg(**raw["env"]),
            storage=StorageCfg(
                local=StorageLocalCfg(**raw["storage"]["local"]),
                minio=StorageMinioCfg(**raw["storage"]["minio"]),
            ),
            auth=AuthCfg(**raw["auth"]),
            sources=SourcesCfg(**raw["sources"]),
            duckdb=DuckDBCfg(**raw["duckdb"]),
            profiling=ProfilingCfg(
                roles=ProfilingRolesCfg(**raw["profiling"]["roles"]),
                outliers=ProfilingOutliersCfg(**raw["profiling"]["outliers"]),
            ),
            cleaning=CleaningCfg(
                columns=CleaningColumnsCfg(**raw["cleaning"]["columns"]),
                normalize=CleaningNormalizeCfg(**raw["cleaning"]["normalize"]),
                impute=CleaningImputeCfg(**raw["cleaning"]["impute"]),
                outliers=CleaningOutliersCfg(**raw["cleaning"]["outliers"]),
                rules=[CleaningRule(**r) for r in raw["cleaning"]["rules"]],
            ),
            topics=TopicsCfg(thresholds=TopicsThresholdsCfg(**raw["topics"]["thresholds"])),
            charts=ChartsCfg(**raw["charts"]),
            weights=WeightsCfg(**raw["weights"]),
            reports=ReportsCfg(
                enabled_generators=ReportsEnabledCfg(**raw["reports"]["enabled_generators"]),
                html=ReportsHtmlCfg(**raw["reports"]["html"]),
                pdf=ReportsPdfCfg(**raw["reports"]["pdf"]),
            ),
            airflow=AirflowCfg(**raw["airflow"]),
            docker=DockerCfg(**raw["docker"]),
            logging=LoggingCfg(**raw["logging"]),
            publishing=PublishingCfg(**raw["publishing"]),
            nlp=NLPCfg(**raw.get("nlp", {})),
            nlg=NLGCfg(**raw.get("nlg", {})),
        )
        cfg._config_dir = p.parent.resolve()
        return cfg._normalize_paths()

    @classmethod
    def load(cls, path: str | None = None) -> "RootCfg":
        final = Path(path or os.environ.get("POLY_CFG", "config/config.toml")).resolve()
        return cls.from_toml(final)


def load_config(path: str | None = None) -> RootCfg:
    return RootCfg.load(path)

# Convenience import for callers
def load_config(path: str | None = None) -> RootCfg:
    return RootCfg.load(path)


```


src/utils/fp.py:
```
from __future__ import annotations
from functools import lru_cache, wraps
from typing import Any, Callable, Dict, Iterable, Iterator, List, Tuple, TypeVar, overload

from toolz import curry as _curry, compose as _compose, pipe as _pipe
from more_itertools import chunked as _chunked, unique_everseen as _unique_everseen

A = TypeVar("A")
B = TypeVar("B")
C = TypeVar("C")

# Re-export toolz versions under your API names
curry = _curry

def identity(x: A) -> A:
    return x

def const(x: A) -> Callable[..., A]:
    return lambda *_, **__: x

def pipe(x: A, *fns: Callable[[Any], Any]) -> Any:
    # Keep signature but delegate to toolz.pipe
    return _pipe(x, *fns) if fns else x

def compose(*fns: Callable[[Any], Any]) -> Callable[[Any], Any]:
    # toolz.compose composes right-to-left as expected
    return _compose(*fns)

def try_or(default: B) -> Callable[[Callable[[A], B]], Callable[[A], B]]:
    def _wrap(fn: Callable[[A], B]) -> Callable[[A], B]:
        @wraps(fn)
        def _inner(x: A) -> B:
            try:
                return fn(x)
            except Exception:
                return default
        return _inner
    return _wrap

@overload
def maybe(x: None, *_: Callable[[Any], Any]) -> None: ...
@overload
def maybe(x: A, *fns: Callable[[Any], Any]) -> Any: ...
def maybe(x: Any, *fns: Callable[[Any], Any]) -> Any:
    if x is None:
        return None
    return pipe(x, *fns)

def map_dict(d: Dict[A, B], fn: Callable[[Tuple[A, B]], Tuple[A, C]]) -> Dict[A, C]:
    return dict(fn(item) for item in d.items())

def filter_dict(d: Dict[A, B], pred: Callable[[Tuple[A, B]], bool]) -> Dict[A, B]:
    return dict(item for item in d.items() if pred(item))

def chunked(it: Iterable[A], size: int) -> Iterator[List[A]]:
    # Explicit guard (consistent, predictable)
    if size < 1:
        raise ValueError("chunk size must be >= 1")
    for chunk in _chunked(it, size):
        yield list(chunk)

def unique_stable(seq: Iterable[A]) -> List[A]:
    # Delegate to more-itertools; preserves first-seen order
    return list(_unique_everseen(seq))

def memoize(maxsize: int = 128):
    def deco(fn: Callable[..., B]) -> Callable[..., B]:
        return lru_cache(maxsize=maxsize)(fn)  # type: ignore
    return deco

```

src/utils/ids.py:
```
from __future__ import annotations
import hashlib, json, re
from typing import Any, Iterable, Mapping, Sequence
from slugify import slugify as _slugify

def slugify(s: str) -> str:
    return _slugify(s, lowercase=True, separator="-")

def _json_dumps_stable(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))

def stable_hash(obj: Any, algo: str = "sha256") -> str:
    h = hashlib.new(algo)
    if hasattr(obj, "to_dict"):
        obj = obj.to_dict()
    h.update(_json_dumps_stable(obj).encode("utf-8"))
    return h.hexdigest()

def short_id(obj: Any, n: int = 10) -> str:
    return stable_hash(obj)[:n]

def make_dataset_slug(uri_or_name: str) -> str:
    base = re.sub(r"[\\/]+", "-", uri_or_name.strip())
    return slugify(base)

def make_chart_id(topic_key: Mapping[str, Any], chart_kind: str) -> str:
    # topic_key should include dataset_slug, roles, fields, filters, time range…
    payload = {"topic": topic_key, "chart": chart_kind}
    return f"{chart_kind}-{short_id(payload)}"
```

src/utils/log.py:
```
from __future__ import annotations
import json, logging, sys
from typing import Any, Dict

class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        # attach extra if present
        for k, v in getattr(record, "__dict__", {}).items():
            if k not in ("name", "msg", "args", "levelname", "levelno",
                         "pathname", "filename", "module", "exc_info",
                         "exc_text", "stack_info", "lineno", "funcName",
                         "created", "msecs", "relativeCreated", "thread",
                         "threadName", "processName", "process"):
                payload[k] = v
        return json.dumps(payload, separators=(",", ":"))

def get_logger(name: str = "polymorphic", level: str = "INFO", structured_json: bool = True) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    handler = logging.StreamHandler(sys.stdout)
    if structured_json:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger
```

src/utils/time.py:
```
from __future__ import annotations
from typing import Iterable, List, Optional, Tuple
from datetime import datetime
import pandas as pd
import pytz

def parse_any_datetime(
    s: str,
    formats: Iterable[str] = ("%Y-%m-%d", "%m/%d/%Y", "%Y-%m-%d %H:%M:%S")
) -> Optional[datetime]:
    for f in formats:
        try:
            return datetime.strptime(s, f)
        except Exception:
            continue
    # pandas to_datetime as fallback
    try:
        dt = pd.to_datetime(s, errors="raise")
        if pd.isna(dt):
            return None
        return dt.to_pydatetime()
    except Exception:
        return None

def to_timezone(dt: datetime, tz: str) -> datetime:
    tzinfo = pytz.timezone(tz)
    if dt.tzinfo is None:
        return tzinfo.localize(dt)
    return dt.astimezone(tzinfo)

def infer_freq(ts: pd.Series) -> Optional[str]:
    # expects datetime64 series
    try:
        return pd.infer_freq(ts.sort_values())
    except Exception:
        return None

def is_regular_frequency(ts: pd.Series) -> bool:
    f = infer_freq(ts)
    return f is not None

def floor_period(dt: datetime, freq: str = "D") -> datetime:
    return pd.Timestamp(dt).floor(freq).to_pydatetime()

def ceil_period(dt: datetime, freq: str = "D") -> datetime:
    return pd.Timestamp(dt).ceil(freq).to_pydatetime()
```

----

my pyproject.toml:
```
[tool.poetry]
name = "polymorphic-data-reporter"
version = "0.1.0"
description = "Dynamic, polymorphic data reporter with ETL stages, automated profiling, chart picking, and templated reports."
readme = "README.md"
packages = [{ include = "src" }]

[tool.poetry.dependencies]
python = ">=3.11,<3.13"
# Core
pandas = "^2.3.1"
polars = "^1.32.2"
duckdb = "^1.3.2"
jinja2 = "^3.1.6"
pydantic = "^2.11.7"
plotly = "^5.24.1"
networkx = "^3.2.1"
scipy = "^1.13.1"
orjson = "^3.11.2"
python-slugify = "^8.0.4"
pyyaml = "^6.0.2"
tomlkit = "^0.13.2"
typing-extensions = "^4.12.2"
# Utilities (new)
toolz = "^0.12.1"
more-itertools = "^10.5.0"
# Storage / IO
minio = "^7.2.16"
boto3 = "^1.39.11"
s3fs = "^2024.12.0"
pyarrow = "^16.1.0"
# Static PNG via headless Chromium screenshot
playwright = "^1.54.0"
# Optional PDF engine
weasyprint = { version = "^62.3", optional = true }
faker = "^25.0.0"

[tool.poetry.group.dev.dependencies]
pytest = "^8.4.1"
pytest-cov = "^5.0.0"
black = "^24.10.0"
isort = "^5.13.2"
mypy = "^1.17.1"
ruff = "^0.6.9"

[tool.poetry.extras]
pdf = ["weasyprint"]

[tool.poetry.scripts]
discover-sources = "scripts.discover_sources:main"
ingest-one = "scripts.ingest_one:main"
profile-one = "scripts.profile_one:main"
clean-one = "scripts.clean_one:main"
topics-one = "scripts.topics_one:main"
report-one = "scripts.report_one:main"
e2e-one = "scripts.end_to_end:main"
nlp-run = "scripts.nlp.nlp_bootstrap:main"

[build-system]
requires = ["poetry-core>=1.9.0"]
build-backend = "poetry.core.masonry.api"


```


src/io/catalog.py:
```
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import re

from .storage import Storage

DATA_FILE_RE = re.compile(r".*\.(csv|json|ndjson|parquet|pq|feather|txt)$", re.IGNORECASE)

@dataclass(frozen=True)
class CatalogEntry:
    slug: str
    uri: str   # absolute path or s3:// uri
    kind: str  # file extension kind
    size: Optional[int] = None

class Catalog:
    """
    Extremely small dataset catalog that discovers files under raw roots and
    groups them per <dataset_slug>. You can store or export this as bronze/_inventory later.
    """
    def __init__(self, storage: Storage) -> None:
        self.storage = storage

    def discover_raw_local(self, dataset_slug: Optional[str] = None) -> List[CatalogEntry]:
        """
        Walk local raw root (recursively). If dataset_slug is provided, scope to that folder.
        """
        base = self.storage.raw_path(dataset_slug or "")
        files = self.storage.ls(base)
        entries: List[CatalogEntry] = []
        for f in files:
            if DATA_FILE_RE.match(f):
                kind = f.split(".")[-1].lower()
                # If user scoped to a dataset, keep that slug; otherwise infer from raw root.
                slug = dataset_slug or _slug_from_raw_root(self.storage, f)
                entries.append(CatalogEntry(slug=slug, uri=f, kind=kind))
        return entries

    def discover_raw_s3(self, dataset_slug: Optional[str] = None) -> List[CatalogEntry]:
        base = self.storage.raw_path(dataset_slug or "", s3=True)
        files = self.storage.ls(base)
        entries: List[CatalogEntry] = []
        for f in files:
            if DATA_FILE_RE.match(f):
                kind = f.split(".")[-1].lower()
                # Enforce scope and keep provided slug if set
                if dataset_slug:
                    if not f.lower().startswith(base.lower()):
                        continue
                    slug = dataset_slug
                else:
                    slug = _slug_from_s3_key(self.storage, f)
                entries.append(CatalogEntry(slug=slug, uri=f, kind=kind))
        return entries

    def inventory(self, use_s3: bool = False, dataset_slug: Optional[str] = None) -> List[CatalogEntry]:
        """
        Unified discovery entrypoint. Set dataset_slug to limit scope.
        """
        if use_s3:
            return self.discover_raw_s3(dataset_slug)
        return self.discover_raw_local(dataset_slug)

# ---- helpers to compute slugs from paths ----

def _slug_from_path(root: str, full: str) -> str:
    # turn /abs/.../data/raw/<slug>/file.ext -> <slug>
    try:
        i = full.lower().index(str(root).lower().rstrip(os_sep()) + os_sep())
        tail = full[i + len(str(root).rstrip(os_sep()) + os_sep()):]
        return tail.split(os_sep(), 1)[0]
    except Exception:
        # fallback: take directory name
        return Path(full).parent.name

def _slug_from_raw_root(storage: Storage, full: str) -> str:
    # guess the immediate child under raw root
    raw_root = storage.raw_path("")
    try:
        i = full.lower().index(str(raw_root).lower().rstrip(os_sep()) + os_sep())
        tail = full[i + len(str(raw_root).rstrip(os_sep()) + os_sep()):]
        return tail.split(os_sep(), 1)[0]
    except Exception:
        return Path(full).parent.name

def _slug_from_s3_key(storage: Storage, uri: str) -> str:
    # s3://bucket/raw/<slug>/file.ext (with optional prefix)
    # we use configured raw_prefix to find slug
    raw_prefix = (storage.cfg.s3_raw_prefix or "").strip("/")
    try:
        # normalize to key without s3://bucket/
        parts = uri.replace("s3://", "").split("/", 1)
        key = parts[1] if len(parts) > 1 else ""
        key_tail = key[len(raw_prefix) + 1 :] if raw_prefix and key.startswith(raw_prefix + "/") else key
        return key_tail.split("/", 1)[0]
    except Exception:
        return Path(uri).parent.name

def os_sep() -> str:
    from os import sep as _sep
    return _sep

```

src/io/readers.py:
```
from __future__ import annotations
from io import BytesIO, StringIO
from typing import Any, Dict, Optional, Tuple
import json
import pandas as pd
import pyarrow.parquet as pq  # fast path for parquet
# polars is optional at call-site; we keep pandas default to reduce surprises

from .storage import Storage

_EXTS = (".csv", ".json", ".ndjson", ".parquet", ".pq", ".feather", ".arrows", ".txt")

def _infer_ext(path: str) -> str:
    p = path.lower()
    for e in _EXTS:
        if p.endswith(e):
            return e
    # default to csv if unknown
    return ".csv"

def read_any(
    storage: Storage,
    path: str,
    *,
    fmt: Optional[str] = None,
    backend: str = "pandas",  # or "polars"
    csv_options: Optional[Dict[str, Any]] = None,
    json_lines: bool = True,
) -> Any:
    """
    Read a dataset into a DataFrame-like object using the selected backend.
    - storage: Storage facade
    - path: local path or s3://bucket/key
    - fmt: optional override ("csv","json","parquet","feather")
    - backend: "pandas" (default) or "polars"
    """
    csv_options = csv_options or {}
    fmt = (fmt or _infer_ext(path)).lstrip(".").lower()

    if backend not in {"pandas", "polars"}:
        raise ValueError("backend must be 'pandas' or 'polars'")

    data = storage.read_bytes(path)

    if backend == "polars":
        import polars as pl  # local import
        if fmt in {"parquet", "pq"}:
            return pl.read_parquet(BytesIO(data))
        if fmt == "feather":
            return pl.read_ipc(BytesIO(data))
        if fmt in {"json", "ndjson"}:
            # Heuristic: if first non-space is '[' or '{', treat as JSON array/object, not NDJSON.
            head = data.lstrip()[:1]
            looks_like_array_or_object = head in (b"[", b"{")
            if json_lines and not looks_like_array_or_object:
                # Try native NDJSON; on failure fall back to pandas->polars.
                try:
                    return pl.read_ndjson(BytesIO(data))
                except Exception:
                    pass
            # JSON array/object fallback via pandas -> polars for robustness across versions
            import pandas as _pd  # alias to avoid shadowing
            pdf = _pd.read_json(BytesIO(data))
            return pl.from_pandas(pdf)
        # csv / txt default
        return pl.read_csv(BytesIO(data), **csv_options)

    # pandas
    if fmt in {"parquet", "pq"}:
        # Faster via pyarrow directly from bytes
        return pq.read_table(BytesIO(data)).to_pandas()
    if fmt == "feather":
        return pd.read_feather(BytesIO(data))
    if fmt in {"json", "ndjson"}:
        # If caller specifies json_lines, honor it unconditionally.
        # This avoids pandas "Trailing data" errors on NDJSON like:
        # {"a":1}\n{"a":2}\n...
        if json_lines:
            return pd.read_json(BytesIO(data), lines=True)
        return pd.read_json(BytesIO(data))
    # csv / txt default
    return pd.read_csv(BytesIO(data), **csv_options)

```

src/io/storage.py:
```
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Protocol, Tuple, Union
import io
import os

# Optional deps (declared in your pyproject)
# We import lazily so local FS still works if S3/MinIO libs aren't present.
try:
    import s3fs  # type: ignore
except Exception:
    s3fs = None  # type: ignore

# -------- Public interface (easy to mock in tests) --------

class BlobStore(Protocol):
    def exists(self, path: str) -> bool: ...
    def ls(self, prefix: str) -> List[str]: ...
    def read_bytes(self, path: str) -> bytes: ...
    def write_bytes(self, path: str, data: bytes) -> None: ...
    def open(self, path: str, mode: str = "rb"): ...  # returns file-like (context manager)

@dataclass(frozen=True)
class StoreConfig:
    # Normalized roots (absolute for local; s3 URI for object store)
    local_raw_root: str
    local_gold_root: str
    s3_enabled: bool
    s3_endpoint: Optional[str]
    s3_bucket: Optional[str]
    s3_access_key: Optional[str]
    s3_secret_key: Optional[str]
    s3_secure: bool
    s3_raw_prefix: Optional[str]
    s3_gold_prefix: Optional[str]

# -------- Helpers --------

def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

def _is_s3_uri(p: str) -> bool:
    return p.startswith("s3://")

def _strip_scheme_file(p: str) -> str:
    return p[7:] if p.startswith("file://") else p

# -------- Local FS backend --------

class LocalStore(BlobStore):
    def exists(self, path: str) -> bool:
        return Path(_strip_scheme_file(path)).exists()

    def ls(self, prefix: str) -> List[str]:
        base = Path(_strip_scheme_file(prefix))
        if base.is_file():
            return [str(base)]
        if not base.exists():
            return []
        out: List[str] = []
        for p in base.rglob("*"):
            if p.is_file():
                out.append(str(p))
        return out

    def read_bytes(self, path: str) -> bytes:
        p = Path(_strip_scheme_file(path))
        return p.read_bytes()

    def write_bytes(self, path: str, data: bytes) -> None:
        p = Path(_strip_scheme_file(path))
        _ensure_parent_dir(p)
        p.write_bytes(data)

    def open(self, path: str, mode: str = "rb"):
        p = Path(_strip_scheme_file(path))
        _ensure_parent_dir(p)
        return p.open(mode)

# -------- S3/MinIO backend via s3fs --------

class S3Store(BlobStore):
    def __init__(
        self,
        endpoint_url: Optional[str],
        key: Optional[str],
        secret: Optional[str],
        secure: bool = False,
    ) -> None:
        if s3fs is None:
            raise RuntimeError("s3fs is not installed. Add it to dependencies.")
        client_kwargs = {}
        if endpoint_url:
            client_kwargs["endpoint_url"] = endpoint_url
        self._fs = s3fs.S3FileSystem(
            key=key or None,
            secret=secret or None,
            client_kwargs=client_kwargs,  # works for MinIO, too
            use_ssl=bool(secure),
        )

    def exists(self, path: str) -> bool:
        return self._fs.exists(path)

    def ls(self, prefix: str) -> List[str]:
        try:
            items = self._fs.ls(prefix, detail=False)
        except FileNotFoundError:
            return []
        # s3fs.ls returns keys with schema s3://bucket/key
        return [f"s3://{p}" if not p.startswith("s3://") else p for p in items]

    def read_bytes(self, path: str) -> bytes:
        with self._fs.open(path, "rb") as f:
            return f.read()

    def write_bytes(self, path: str, data: bytes) -> None:
        with self._fs.open(path, "wb") as f:
            f.write(data)

    def open(self, path: str, mode: str = "rb"):
        # Caller must close; usually used in a context manager
        return self._fs.open(path, mode)

# -------- Composite router --------

class Storage:
    """
    Unified storage facade. Knows how to route file://, s3://, and plain local paths.
    Also exposes helpers to build dataset-aware paths for raw/gold roots.
    """
    def __init__(self, cfg: StoreConfig) -> None:
        self.cfg = cfg
        self._local = LocalStore()
        self._s3: Optional[S3Store] = None
        if cfg.s3_enabled:
            self._s3 = S3Store(
                endpoint_url=cfg.s3_endpoint,
                key=cfg.s3_access_key,
                secret=cfg.s3_secret_key,
                secure=cfg.s3_secure,
            )

    # --- routing ---

    def _backend(self, path: str) -> BlobStore:
        if _is_s3_uri(path):
            if not self._s3:
                raise RuntimeError("S3 access requested but not enabled/configured.")
            return self._s3
        return self._local

    # --- direct passthrough ---

    def exists(self, path: str) -> bool:
        return self._backend(path).exists(path)

    def ls(self, prefix: str) -> List[str]:
        return self._backend(prefix).ls(prefix)

    def read_bytes(self, path: str) -> bytes:
        return self._backend(path).read_bytes(path)

    def write_bytes(self, path: str, data: bytes) -> None:
        self._backend(path).write_bytes(path, data)

    def open(self, path: str, mode: str = "rb"):
        return self._backend(path).open(path, mode)

    # --- path helpers bound to config roots ---

    def raw_path(self, *parts: str, s3: bool = False) -> str:
        if s3:
            if not self.cfg.s3_enabled or not self.cfg.s3_bucket:
                raise RuntimeError("S3 raw root requested but S3 is disabled.")
            prefix = self.cfg.s3_raw_prefix or ""
            key = "/".join([prefix.strip("/"), *[p.strip("/") for p in parts if p]]).strip("/")
            return f"s3://{self.cfg.s3_bucket}/{key}"
        base = Path(self.cfg.local_raw_root)
        return str(base.joinpath(*parts))

    def gold_path(self, *parts: str, s3: bool = False) -> str:
        if s3:
            if not self.cfg.s3_enabled or not self.cfg.s3_bucket:
                raise RuntimeError("S3 gold root requested but S3 is disabled.")
            prefix = self.cfg.s3_gold_prefix or ""
            key = "/".join([prefix.strip("/"), *[p.strip("/") for p in parts if p]]).strip("/")
            return f"s3://{self.cfg.s3_bucket}/{key}"
        base = Path(self.cfg.local_gold_root)
        return str(base.joinpath(*parts))

# -------- Factory from your RootCfg --------

def build_storage_from_config(root_cfg) -> Storage:
    # root_cfg is src.config_model.model.RootCfg
    loc = root_cfg.storage.local
    s3 = root_cfg.storage.minio
    cfg = StoreConfig(
        local_raw_root=loc.raw_root,
        local_gold_root=loc.gold_root,
        s3_enabled=bool(s3.enabled),
        s3_endpoint=s3.endpoint if s3.enabled else None,
        s3_bucket=s3.bucket if s3.enabled else None,
        s3_access_key=s3.access_key if s3.enabled else None,
        s3_secret_key=s3.secret_key if s3.enabled else None,
        s3_secure=bool(s3.secure) if s3.enabled else False,
        s3_raw_prefix=s3.raw_prefix if s3.enabled else None,
        s3_gold_prefix=s3.gold_prefix if s3.enabled else None,
    )
    return Storage(cfg)

```

src/io/writers.py:
```
from __future__ import annotations
from io import BytesIO
from typing import Any, Dict, Optional
import pandas as pd

from .storage import Storage

def write_any(
    storage: Storage,
    df: Any,
    path: str,
    *,
    fmt: Optional[str] = None,
    backend: str = "pandas",  # or "polars"
    csv_options: Optional[Dict] = None,
    json_lines: bool = False,
) -> None:
    """
    Write a DataFrame-like to the given path using the chosen backend.
    """
    csv_options = csv_options or {}
    fmt = (fmt or path.split(".")[-1]).lower()

    if backend == "polars":
        import polars as pl
        if not isinstance(df, pl.DataFrame):
            # Accept pandas and convert
            if isinstance(df, pd.DataFrame):
                df = pl.from_pandas(df)
            else:
                raise TypeError("Expected polars.DataFrame or pandas.DataFrame for backend='polars'.")

        if fmt in {"parquet", "pq"}:
            buf = BytesIO()
            df.write_parquet(buf)
            storage.write_bytes(path, buf.getvalue())
            return
        if fmt == "feather":
            buf = BytesIO()
            df.write_ipc(buf)
            storage.write_bytes(path, buf.getvalue())
            return
        if fmt in {"json", "ndjson"}:
            buf = BytesIO()
            if json_lines:
                df.write_ndjson(buf)
            else:
                df.write_json(buf)
            storage.write_bytes(path, buf.getvalue())
            return
        # csv default
        buf = BytesIO()
        df.write_csv(buf, **csv_options)
        storage.write_bytes(path, buf.getvalue())
        return

    # pandas default
    if not isinstance(df, pd.DataFrame):
        # try best-effort convert if it's polars
        try:
            import polars as pl
            if isinstance(df, pl.DataFrame):
                df = df.to_pandas()
        except Exception:
            pass
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected pandas.DataFrame (or convertible).")

    if fmt in {"parquet", "pq"}:
        buf = BytesIO()
        df.to_parquet(buf, index=False)
        storage.write_bytes(path, buf.getvalue())
        return
    if fmt == "feather":
        buf = BytesIO()
        df.to_feather(buf)
        storage.write_bytes(path, buf.getvalue())
        return
    if fmt in {"json", "ndjson"}:
        buf = BytesIO()
        df.to_json(buf, orient="records", lines=json_lines)
        storage.write_bytes(path, buf.getvalue())
        return
    # csv default
    buf = BytesIO()
    df.to_csv(buf, index=False, **csv_options)
    storage.write_bytes(path, buf.getvalue())

```