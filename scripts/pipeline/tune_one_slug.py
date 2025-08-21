from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype

# project imports
from src.config_model.model import RootCfg
from src.cleaning.engine import RuleSpec, run_clean_pass
from src.cleaning.rescore import rescore_after_clean
from src.nlp.roles import _dtype_str
from src.nlp.schema import ColumnHints, ColumnSchema, ProposedSchema, RoleConfidence
from src.nlp.schema_io import schema_path_for_slug, to_toml

# optional IO backend
try:  # pragma: no cover
    from src.io.storage import Storage  # type: ignore
    from src.io.catalog import Catalog  # type: ignore
    from src.io.readers import read_any  # type: ignore
except Exception:  # pragma: no cover
    Storage = None
    Catalog = None
    read_any = None

SUPPORTED_EXTS = {".csv", ".parquet", ".json", ".ndjson", ".xlsx", ".xls"}

# ------------------------------- logging -------------------------------
import logging
LOG = logging.getLogger("tune_one_slug")
def _setup_logging(level: str = "INFO") -> None:
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

# ------------------------------- paths ---------------------------------
@dataclass(frozen=True)
class Paths:
    root: Path
    slug: str

    @property
    def bronze_dir(self) -> Path:
        return self.root / "data" / "bronze" / self.slug

    @property
    def silver_dir(self) -> Path:
        return self.root / "data" / "silver" / self.slug

    @property
    def bronze_parquet(self) -> Path:
        return self.bronze_dir / "normalize.parquet"

    @property
    def bootstrap_json(self) -> Path:
        return self.bronze_dir / "nlp_bootstrap.json"

    @property
    def normalized_log(self) -> Path:
        return self.bronze_dir / "normalized.json"

    @property
    def schema_toml(self) -> Path:
        return Path(schema_path_for_slug(self.root, self.slug))

# ------------------------------ JSON utils -----------------------------
def _json_load(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return json.loads(json.dumps(default))  # deep copy

def _json_dump(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")

def _append_json_list(path: Path, record: dict) -> None:
    try:
        logs = _json_load(path, [])
        if not isinstance(logs, list):
            logs = []
        logs.append(record)
        _json_dump(path, logs)
    except Exception:
        # best-effort log; don't crash pipeline on large logs
        pass

# ------------------------------- IO / bronze ---------------------------
def _discover_uris(root: Path, slug: str) -> List[str]:
    uris: List[str] = []
    try:
        if Storage and Catalog:
            storage = Storage()
            entries = Catalog(storage).inventory(dataset_slug=slug)
            uris = [e.uri for e in entries]  # type: ignore[attr-defined]
    except Exception:
        uris = []
    if not uris:
        raw_dir = root / "data" / "raw" / slug
        if raw_dir.exists():
            for p in raw_dir.rglob("*"):
                if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
                    uris.append(str(p))
    return uris

def _read_frame(uri: str) -> Optional[pd.DataFrame]:
    if read_any and Storage:
        try:
            df = read_any(Storage(), uri, backend="pandas")  # type: ignore
            return df if isinstance(df, pd.DataFrame) else pd.DataFrame(df)
        except Exception:
            pass
    ext = Path(uri).suffix.lower()
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
    def _as_series(val) -> pd.Series:
        return val if isinstance(val, pd.Series) else pd.Series(val, index=s.index)
    try:
        cand_a = _as_series(pd.to_datetime(s, errors="coerce")); cand_a = _plausible(cand_a)
    except Exception:
        cand_a = pd.Series(pd.NaT, index=s.index)
    try:
        nums = pd.to_numeric(s, errors="coerce")
        cand_b = _as_series(pd.to_datetime(nums, unit="ms", errors="coerce")); cand_b = _plausible(cand_b)
    except Exception:
        cand_b = pd.Series(pd.NaT, index=s.index)
    try:
        nums = pd.to_numeric(s, errors="coerce")
        cand_c = _as_series(pd.to_datetime(nums, unit="s", errors="coerce")); cand_c = _plausible(cand_c)
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
        if s.dtype == object or pd.api.types.is_string_dtype(s):
            if any(k in str(col).lower() for k in ("date", "time", "timestamp", "dt", "ts")):
                parsed = _best_datetime_parse(s)
                if parsed.notna().any():
                    out[col] = parsed; continue
            # stringify complex
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

def ensure_bronze(paths: Paths, df: pd.DataFrame) -> None:
    paths.bronze_dir.mkdir(parents=True, exist_ok=True)
    df_arrow = _prepare_for_parquet(df)
    leftover_obj = [c for c in df_arrow.columns if df_arrow[c].dtype == object]
    if leftover_obj:
        LOG.warning("Object dtypes remain after Arrow prep: %s", leftover_obj)
    df_arrow.to_parquet(paths.bronze_parquet, index=False)
    LOG.info("[bronze] wrote %s (%d rows, %d cols)", paths.bronze_parquet, len(df_arrow), df_arrow.shape[1])

def ingest_raw(paths: Paths, *, max_files: Optional[int], sort_by: Iterable[str]) -> pd.DataFrame:
    uris = _discover_uris(paths.root, paths.slug)
    if not uris:
        raise FileNotFoundError(f"No input files found under data/raw/{paths.slug}/")
    if max_files is not None:
        uris = uris[:max_files]
    frames: List[pd.DataFrame] = []
    per_file_rows: List[Tuple[str, int]] = []
    for u in uris:
        df = _read_frame(u)
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
    LOG.info("[ingest] files=%d by_ext=%s rows_total=%d rows_concat=%d",
             len(per_file_rows), dict(by_ext), sum(n for _, n in per_file_rows), len(df_all))
    return df_all

def load_bronze(paths: Paths) -> pd.DataFrame:
    if not paths.bronze_parquet.exists():
        raise FileNotFoundError(f"normalize.parquet not found at {paths.bronze_parquet}")
    return pd.read_parquet(paths.bronze_parquet)

# ----------------------- profiling & decisions ------------------------
import re
_BOOL_TOKENS = {"true","false","t","f","y","n","yes","no","1","0","on","off"}
_NUMERIC_RE = re.compile(
    r"""
    ^\s*
    [+\-]?
    (?:
      (?:\d{1,3}(?:[,_]\d{3})+|\d+)
    )
    (?:\.\d+)?            # optional decimal
    (?:\s*%){0,1}         # optional percent
    \s*$
    """, re.VERBOSE
)

def _is_stringlike(s: pd.Series) -> bool:
    return s.dtype == object or pd.api.types.is_string_dtype(s)

def _preview_norm_strings(s: pd.Series) -> pd.Series:
    if not _is_stringlike(s):
        return s.astype("string", copy=False)
    x = s.astype("string", copy=False)
    x = x.str.normalize("NFKC")
    x = x.str.replace(r"[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]", "", regex=True)
    x = x.str.replace(r"\s+", " ", regex=True).str.strip()
    # common unicode punctuation/whitespace fixes
    x = (x.str.replace("\u00A0", " ", regex=False)
           .str.replace("\u2013", "-", regex=False)
           .str.replace("\u2014", "-", regex=False)
           .str.replace("\u2212", "-", regex=False))
    return x

def _bool_like_ratio(s: pd.Series) -> float:
    if not _is_stringlike(s): return 0.0
    x = _preview_norm_strings(s).str.lower().dropna()
    if x.empty: return 0.0
    return float(x.isin(_BOOL_TOKENS).mean())

def _numeric_like_ratio(s: pd.Series) -> float:
    if pd.api.types.is_numeric_dtype(s): return 1.0
    if not _is_stringlike(s): return 0.0
    x = _preview_norm_strings(s).dropna()
    if x.empty: return 0.0
    return float(x.str.match(_NUMERIC_RE).mean())

def _alpha_ratio(s: pd.Series) -> float:
    if not _is_stringlike(s): return 0.0
    x = _preview_norm_strings(s).dropna()
    if x.empty: return 0.0
    return float(x.str.contains(r"[A-Za-z]").mean())

def _unique_ratio(s: pd.Series) -> float:
    n = len(s)
    if n == 0: return 0.0
    return float(s.nunique(dropna=True)) / float(n)

@dataclass
class ColumnDecision:
    name: str
    dtype_in: str
    target_role: str              # 'boolean'|'numeric'|'time'|'categorical'|'text'
    reason: str                   # human friendly rationale
    metrics: Dict[str, float]     # num_like, alpha, bool_like, unique_ratio, avg_len, nunique, missing_pct

def decide_types(df: pd.DataFrame,
                 *,
                 cat_card_max: int = 200,
                 cat_unique_ratio_max: float = 0.30,
                 cat_avg_len_max: int = 50,
                 bool_threshold: float = 0.90,
                 numeric_strict: float = 0.98,
                 alpha_tiny: float = 0.02) -> List[ColumnDecision]:
    decisions: List[ColumnDecision] = []
    for name in df.columns:
        s = df[name]
        dtype_in = _dtype_str(s)

        # base metrics
        missing_pct = float(pd.isna(s).mean())
        nunique = int(s.nunique(dropna=True))
        unique_ratio = _unique_ratio(s)

        # datetime/boolean/numeric native
        if pd.api.types.is_datetime64_any_dtype(s):
            decisions.append(ColumnDecision(name, dtype_in, "time", "pandas datetime64", {
                "missing_pct": missing_pct, "nunique": nunique, "unique_ratio": unique_ratio
            }))
            continue
        if pd.api.types.is_bool_dtype(s):
            decisions.append(ColumnDecision(name, dtype_in, "boolean", "pandas bool dtype", {
                "missing_pct": missing_pct, "nunique": nunique, "unique_ratio": unique_ratio
            }))
            continue
        if pd.api.types.is_numeric_dtype(s):
            decisions.append(ColumnDecision(name, dtype_in, "numeric", "pandas numeric dtype", {
                "missing_pct": missing_pct, "nunique": nunique, "unique_ratio": unique_ratio
            }))
            continue

        # string-like branch
        if _is_stringlike(s):
            x = _preview_norm_strings(s)
            avg_len = float(x.dropna().str.len().mean() or 0.0)
            bool_like = _bool_like_ratio(x)
            num_like = _numeric_like_ratio(x)
            alpha = _alpha_ratio(x)

            # strict boolean from tokens
            if bool_like >= bool_threshold:
                decisions.append(ColumnDecision(name, dtype_in, "boolean",
                    f"bool-like {bool_like:.2f} ≥ {bool_threshold}",
                    {"missing_pct": missing_pct, "nunique": nunique, "unique_ratio": unique_ratio,
                     "avg_len": avg_len, "bool_like": bool_like, "num_like": num_like, "alpha": alpha}))
                continue

            # strict numeric-from-string (content-based ONLY)
            if num_like >= numeric_strict and alpha <= alpha_tiny:
                decisions.append(ColumnDecision(name, dtype_in, "numeric",
                    f"numeric-like {num_like:.2f} ≥ {numeric_strict} and alpha {alpha:.2f} ≤ {alpha_tiny}",
                    {"missing_pct": missing_pct, "nunique": nunique, "unique_ratio": unique_ratio,
                     "avg_len": avg_len, "bool_like": bool_like, "num_like": num_like, "alpha": alpha}))
                continue

            # categorical from rows (bounded card, low unique ratio, short tokens, not numeric/boolean)
            if (2 <= nunique <= cat_card_max
                and unique_ratio <= cat_unique_ratio_max
                and avg_len <= cat_avg_len_max
                and num_like <= 0.10
                and bool_like <= 0.70):
                decisions.append(ColumnDecision(name, dtype_in, "categorical",
                    "row-based categorical: bounded cardinality, low unique ratio, short tokens, not numeric/bool",
                    {"missing_pct": missing_pct, "nunique": nunique, "unique_ratio": unique_ratio,
                     "avg_len": avg_len, "bool_like": bool_like, "num_like": num_like, "alpha": alpha}))
                continue

            # otherwise text
            decisions.append(ColumnDecision(name, dtype_in, "text", "default string/text", {
                "missing_pct": missing_pct, "nunique": nunique, "unique_ratio": unique_ratio,
                "avg_len": avg_len, "bool_like": bool_like, "num_like": num_like, "alpha": alpha
            }))
            continue

        # fallback: unknown -> text
        decisions.append(ColumnDecision(name, dtype_in, "text", "fallback", {
            "missing_pct": missing_pct, "nunique": nunique, "unique_ratio": unique_ratio
        }))

    return decisions

# --------------------------- schema / roles ----------------------------
def proposed_from_decisions(slug: str, decisions: List[ColumnDecision]) -> ProposedSchema:
    cols: List[ColumnSchema] = []
    confs: List[float] = []
    for d in decisions:
        # confidence from rule path (deterministic but meaningful)
        if d.target_role == "numeric":
            conf = 0.98
        elif d.target_role == "boolean":
            conf = 0.95
        elif d.target_role == "time":
            conf = 0.90
        elif d.target_role == "categorical":
            conf = 0.88
        else:
            conf = 0.70
        cols.append(ColumnSchema(
            name=d.name,
            dtype=d.dtype_in,
            role_confidence=RoleConfidence(role=d.target_role, confidence=conf),
            hints=ColumnHints(),
        ))
        confs.append(conf)
    return ProposedSchema(dataset_slug=slug, columns=cols, schema_confidence=(sum(confs)/len(confs) if confs else 0.0))

# --------------------------- rule pack builder -------------------------
def build_rules_for_decisions(df: pd.DataFrame, decisions: List[ColumnDecision]) -> List[RuleSpec]:
    rules: List[RuleSpec] = []
    # priorities: high first; we keep spacing so insertions are easy
    P0, P1, P2, P3 = 990, 980, 970, 960

    for d in decisions:
        col = d.name
        s = df[col]
        miss = float(pd.isna(s).mean())
        # helper to add a rule
        def add(then: str, prio: int) -> None:
            rules.append(RuleSpec(id=f"det-{col}-{then.split('(')[0]}", priority=prio, when=f'name == "{col}"', then=then))

        if d.target_role == "boolean":
            # normalize → coerce → impute
            add("normalize_null_tokens()", P0)
            add("text_normalize(strip=true, lower=true)", P1)
            add("coerce_bool()", P2)
            if miss > 0.0:
                add("impute_value(false)", P3)

        elif d.target_role == "numeric":
            if _is_stringlike(s):
                add("normalize_null_tokens()", P0)
                add("text_normalize(strip=true, lower=false)", P1)
                add("coerce_numeric()", P2)
            # optional numeric hygiene
            if miss > 0.0:
                add("impute()", P3)
            add("outliers()", P3-1)

        elif d.target_role == "categorical":
            add("normalize_null_tokens()", P0)
            add("text_normalize(strip=true, lower=false)", P1)
            add("cast_category()", P2)
            add("rare_cats(0.01, 'Other')", P3)

        elif d.target_role == "time":
            # keep as is; if it's stringy, try light parse
            if _is_stringlike(s):
                add("dt_parse()", P1)
            # rounding is optional; skip destructive operations by default

        else:  # text
            add("normalize_null_tokens()", P0)
            add("text_normalize(strip=true, lower=false)", P1)
            # impute text is often undesirable; leave NA as NA

    return rules

# ------------------------------- scoring -------------------------------
def frame_missing_rate(df: pd.DataFrame) -> float:
    if df.size == 0: return 0.0
    return float(df.isna().sum().sum()) / float(df.shape[0] * df.shape[1])

def score_frame(rescore_dict: dict, df_after: pd.DataFrame) -> float:
    sc = float(rescore_dict.get("schema_conf_after", 0.0) or 0.0)
    avg = float(rescore_dict.get("avg_role_conf_after", 0.0) or 0.0)
    miss_penalty = frame_missing_rate(df_after)
    return sc * 0.7 + avg * 0.3 - 0.25 * miss_penalty

# ------------------------------- runner --------------------------------
def _deterministic_sample(df: pd.DataFrame, n: int, *, seed: int, hash_key: str = "") -> pd.DataFrame:
    if n <= 0 or len(df) <= n: return df.copy()
    if hash_key and hash_key in df.columns:
        idx = pd.util.hash_pandas_object(df[hash_key]).astype("uint64")
        frac = n / float(len(df))
        cutoff = int((2**64 - 1) * frac)
        return df[idx <= cutoff].head(n).copy()
    return df.sample(n=n, random_state=seed).copy()

def main() -> None:
    ap = argparse.ArgumentParser(description="Deterministic cleaner: infer schema from rows; apply one pass.")
    ap.add_argument("--config", default="config/config.toml")
    ap.add_argument("--slug", default="x1")
    ap.add_argument("--head", type=int, default=5)
    ap.add_argument("--subset", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hash-key", default="")
    ap.add_argument("--from-raw", action="store_true")
    ap.add_argument("--rebuild-bronze", action="store_true")
    ap.add_argument("--max-files", type=int, default=None)
    ap.add_argument("--sort-by", nargs="*", default=[])
    ap.add_argument("--save-best", action="store_true")
    ap.add_argument("--write-back", action="store_true")
    ap.add_argument("--log-level", default="INFO")
    # knobs for categorical/boolean/numeric decisions (tune if needed)
    ap.add_argument("--cat-card-max", type=int, default=200)
    ap.add_argument("--cat-unique-max", type=float, default=0.30)
    ap.add_argument("--cat-avg-len-max", type=int, default=50)
    ap.add_argument("--bool-threshold", type=float, default=0.90)
    ap.add_argument("--numeric-strict", type=float, default=0.98)
    ap.add_argument("--alpha-tiny", type=float, default=0.02)

    args = ap.parse_args()
    _setup_logging(args.log_level)

    np.random.seed(args.seed)

    paths = Paths(root=Path.cwd(), slug=args.slug)
    cfg = RootCfg.load(args.config)
    paths.bronze_dir.mkdir(parents=True, exist_ok=True)

    # ingest / bronze
    need_ingest = args.from_raw or args.rebuild_bronze or not paths.bronze_parquet.exists()
    if need_ingest:
        LOG.info("[ingest] building bronze for slug=%s", args.slug)
        df_raw_full = ingest_raw(paths, max_files=args.max_files, sort_by=args.sort_by)
        ensure_bronze(paths, df_raw_full)

    LOG.info("[ingest] using bronze for slug=%s", args.slug)
    df_raw_full = load_bronze(paths)
    df_raw = _deterministic_sample(df_raw_full, args.subset, seed=args.seed, hash_key=args.hash_key)

    # snapshot
    with pd.option_context("display.max_colwidth", 200, "display.width", 200):
        LOG.info("\n=== BEFORE: HEAD(%d) ===\n%s", args.head, df_raw.head(args.head).to_string(index=False))
    LOG.info("\n=== BEFORE: DTYPES ===\n%s", "\n".join(f"  {c}: {t}" for c, t in df_raw.dtypes.items()))

    # decisions (ROW-BASED ONLY — no name hints)
    decisions = decide_types(
        df_raw,
        cat_card_max=args.cat_card_max,
        cat_unique_ratio_max=args.cat_unique_max,
        cat_avg_len_max=args.cat_avg_len_max,
        bool_threshold=args.bool_threshold,
        numeric_strict=args.numeric_strict,
        alpha_tiny=args.alpha_tiny,
    )

    # log the rationale compactly
    LOG.info("\n[decisions]")
    for d in decisions:
        m = d.metrics
        extras = ", ".join(f"{k}={m[k]:.2f}" for k in ["num_like","alpha","bool_like","unique_ratio"] if k in m)
        LOG.info("  %-20s → %-11s (%s)", d.name, d.target_role, (extras or d.reason))

    # schema proposal & rule pack
    proposed = proposed_from_decisions(args.slug, decisions)
    rules = build_rules_for_decisions(df_raw, decisions)

    # one cleaning pass
    result = run_clean_pass(df_raw, proposed, cfg, extra_rules=rules)
    df_after = result.clean_df

    # score/report
    r = rescore_after_clean(df_after, proposed, cfg.profiling.roles, cfg.nlp)
    score = score_frame(r.__dict__ if hasattr(r, "__dict__") else dict(r), df_after)

    LOG.info(
        "\n[SCORES] schema_conf=%.4f  avg_role_conf=%.4f  miss=%.4f  score=%.5f",
        getattr(r, "schema_conf_after", 0.0),
        getattr(r, "avg_role_conf_after", 0.0),
        frame_missing_rate(df_after),
        score,
    )
    with pd.option_context("display.max_colwidth", 200, "display.width", 200):
        LOG.info("\n=== AFTER: HEAD(%d) ===\n%s", args.head, df_after.head(args.head).to_string(index=False))
    LOG.info("\n=== AFTER: DTYPES ===\n%s", "\n".join(f"  {c}: {t}" for c, t in df_after.dtypes.items()))

    actions = result.report.get("rules", {}).get("per_column_actions", {}) or {}
    if not actions:
        LOG.info("\n[Rules fired]\n  (none)")
    else:
        LOG.info("\n[Rules fired]")
        for c, acts in actions.items():
            if acts:
                LOG.info("  %s: %s", c, ", ".join(acts))

    _append_json_list(paths.normalized_log, {
        "schema_conf": getattr(r, "schema_conf_after", None),
        "avg_role_conf": getattr(r, "avg_role_conf_after", None),
        "score": score,
        "missing_rate": frame_missing_rate(df_after),
        "width": df_after.shape[1],
        "rules_total": result.report.get("rules", {}).get("total_applications"),
    })

    # persist
    if args.save_best:
        paths.silver_dir.mkdir(parents=True, exist_ok=True)
        outp = paths.silver_dir / "cleaned_best.parquet"
        df_after.to_parquet(outp, index=False)
        LOG.info("[SAVE] cleaned -> %s", outp)

    if args.write_back:
        # bootstrap with the row-based roles we just inferred
        merged_cols = [
            {"name": d.name, "dtype": d.dtype_in, "role": d.target_role,
             "role_confidence": next(c for c in proposed.columns if c.name == d.name).role_confidence.confidence}
            for d in decisions
        ]
        updated_bootstrap = {
            "dataset": args.slug,
            "schema_confidence": proposed.schema_confidence,
            "columns": merged_cols,
        }
        _json_dump(paths.bootstrap_json, updated_bootstrap)
        paths.schema_toml.parent.mkdir(parents=True, exist_ok=True)
        paths.schema_toml.write_text(to_toml(proposed.to_dict()), encoding="utf-8")
        LOG.info("[WRITE] nlp_bootstrap.json and schema TOML updated -> %s", paths.schema_toml)

if __name__ == "__main__":
    main()
