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
