from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Literal

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype

from src.config_model.model import RootCfg
from src.io.storage import build_storage_from_config
from src.io.catalog import Catalog
from src.io.readers import read_any
from src.io.writers import write_any

from src.nlp.roles import guess_role, _dtype_str
from src.nlp.schema import ColumnHints, ColumnSchema, ProposedSchema, RoleConfidence
from src.nlp.schema_io import schema_path_for_slug, to_toml

from src.cleaning.engine import RuleSpec, run_clean_pass
from src.cleaning.rescore import rescore_after_clean
from src.utils.log import get_logger

SUPPORTED_EXTS = {".csv", ".json", ".ndjson", ".parquet", ".pq", ".feather"}


# ---------------------------- small helpers ----------------------------

def discover_all_slugs(cfg: RootCfg) -> List[str]:
    storage = build_storage_from_config(cfg)
    cat = Catalog(storage)
    entries = cat.inventory(use_s3=False, dataset_slug=None)
    try:
        entries += cat.inventory(use_s3=True, dataset_slug=None)
    except Exception:
        pass
    return sorted({e.slug for e in entries})


def _best_datetime_parse(series: pd.Series) -> pd.Series:
    """Try natural, then epoch ms/sec. Only return if ≥60% parse success."""
    s = series.astype("string", copy=False)

    def _plausible(x: pd.Series) -> pd.Series:
        try:
            yr = x.dt.year
            return x.where(yr.between(1970, 2100))
        except Exception:
            return x

    opts = []
    try:
        a = pd.to_datetime(s, errors="coerce")
        a = _plausible(a)
        opts.append(a)
    except Exception:
        opts.append(pd.Series(pd.NaT, index=s.index))
    try:
        b = pd.to_numeric(s, errors="coerce")
        b = pd.to_datetime(b, unit="ms", errors="coerce")
        b = _plausible(b)
        opts.append(b)
    except Exception:
        opts.append(pd.Series(pd.NaT, index=s.index))
    try:
        c = pd.to_numeric(s, errors="coerce")
        c = pd.to_datetime(c, unit="s", errors="coerce")
        c = _plausible(c)
        opts.append(c)
    except Exception:
        opts.append(pd.Series(pd.NaT, index=s.index))

    nn = [int(x.notna().sum()) for x in opts]
    non_null = int(s.notna().sum())
    if non_null == 0:
        return opts[0]
    best = int(np.argmax(nn))
    if nn[best] / non_null >= 0.60:
        return opts[best].dt.tz_localize(None)
    return pd.Series(pd.NaT, index=s.index)


def _prepare_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    """
    Conservative normalization for bronze:
      - parse obvious datetime (by name hint or value pattern)
      - coerce numeric only when ≥80% numeric-like
      - stringify objects otherwise
    """
    out = df.copy()
    for col in out.columns:
        s = out[col]
        if pd.api.types.is_datetime64_any_dtype(s) or pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
            continue
        if isinstance(s.dtype, CategoricalDtype):
            out[col] = s.astype("string")
            continue
        if s.dtype == object or pd.api.types.is_string_dtype(s):
            n = str(col).lower()
            if any(k in n for k in ("date", "time", "timestamp", "dt", "ts")):
                parsed = _best_datetime_parse(s)
                if parsed.notna().any():
                    out[col] = parsed
                    continue
            # numeric-like?
            st = (
                s.astype("string", copy=False)
                .str.normalize("NFKC")
                .str.replace(r"\s+", " ", regex=True)
                .str.strip()
            )
            num_like = pd.to_numeric(
                st.str.replace(r"(?<=\d)[,_](?=\d{3}\b)", "", regex=True).str.replace("−", "-", regex=False),
                errors="coerce",
            )
            if float(num_like.notna().mean()) >= 0.80:
                out[col] = num_like
                continue

            # stringify JSON-like objects nicely
            def _to_text(x):
                if x is None or (isinstance(x, float) and np.isnan(x)):
                    return None
                if isinstance(x, (dict, list, tuple)):
                    try:
                        return json.dumps(x, ensure_ascii=False)
                    except Exception:
                        return str(x)
                return str(x)

            out[col] = s.map(_to_text).astype("string")
        else:
            try:
                out[col] = out[col].astype("string")
            except Exception:
                pass
    return out


def _to_proposed_schema(slug: str, df: pd.DataFrame, cfg: RootCfg) -> ProposedSchema:
    cols: List[ColumnSchema] = []
    confs: List[float] = []
    for name in df.columns:
        s = df[name]
        role, conf = guess_role(name, s, cfg.nlp, cfg.profiling.roles.datetime_formats)
        cols.append(
            ColumnSchema(
                name=str(name),
                dtype=_dtype_str(s),
                role_confidence=RoleConfidence(role=str(role), confidence=float(conf or 0.0)),
                hints=ColumnHints(),
            )
        )
        confs.append(float(conf or 0.0))
    avg_conf = float(sum(confs) / len(confs)) if confs else 0.0
    return ProposedSchema(dataset_slug=slug, columns=cols, schema_confidence=avg_conf)


def _safe_rules_from_suggestions(suggestions: Dict[str, List[str]]) -> List[RuleSpec]:
    """Turn plan_followups-like suggestions into RuleSpec (whitelist a few)."""
    allow = {
        "normalize_null_tokens",
        "text_normalize",
        "regex_replace",
        "coerce_bool",
        "coerce_numeric",
        "impute",
        "impute_value",
        "cast_category",
        "rare_cats",
        "parse_datetime",
        "parse_datetime_robust",
        "parse_epoch",
        "dt_round",
        "outliers",
        "standardize_units",
        "zero_as_missing",
    }
    rules: List[RuleSpec] = []
    for col, acts in (suggestions or {}).items():
        for a in acts or []:
            akey = a.split("(", 1)[0].strip()
            if akey in allow:
                rid = f"suggested-{col}-{akey}"
                rules.append(RuleSpec(id=rid, priority=900, when=f'name == "{col}"', then=a if "(" in a else f"{a}()"))
    return rules


# --------- NEW: load frozen schema (no new dependency; self-contained) ---------

def _load_frozen_schema(slug: str, project_root: Path) -> ProposedSchema:
    """
    Read schemas/<slug>.schema.toml and construct ProposedSchema.
    We avoid assuming a from_toml helper; keep it local and explicit.
    """
    try:
        import tomllib  # py>=3.11
    except Exception:
        import tomli as tomllib  # type: ignore

    p = Path(schema_path_for_slug(project_root, slug))
    raw = tomllib.loads(p.read_text(encoding="utf-8"))

    # Expect keys: dataset_slug, columns=[{name,dtype,role_confidence{role,confidence},hints{...}}], schema_confidence
    cols: List[ColumnSchema] = []
    for c in raw.get("columns", []):
        rc = c.get("role_confidence", {}) or {}
        hints = c.get("hints", {}) or {}
        cols.append(
            ColumnSchema(
                name=str(c.get("name", "")),
                dtype=str(c.get("dtype", "object")),
                role_confidence=RoleConfidence(role=str(rc.get("role", "")), confidence=float(rc.get("confidence", 0.0))),
                hints=ColumnHints(**hints) if isinstance(hints, dict) else ColumnHints(),
            )
        )
    return ProposedSchema(
        dataset_slug=str(raw.get("dataset_slug", slug)),
        columns=cols,
        schema_confidence=float(raw.get("schema_confidence", 1.0)),
    )


def _write_schema_report(df: pd.DataFrame, out_path: Path) -> None:
    rep = pd.DataFrame(
        {
            "name": list(df.columns),
            "dtype": [str(t) for t in df.dtypes],
            "missing_pct": [float(df[c].isna().mean()) for c in df.columns],
        }
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_any(build_storage_from_config(RootCfg.load()), rep, str(out_path), fmt="parquet")  # storage from default cfg


# ------------------------------- ingest --------------------------------

@dataclass
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
    def schema_toml(self) -> Path:
        return Path(schema_path_for_slug(self.root, self.slug))

    @property
    def iter_log(self) -> Path:
        return self.bronze_dir / "loop_iter.jsonl"


def _discover_uris(storage, slug: str) -> List[str]:
    cat = Catalog(storage)
    uris = [e.uri for e in cat.inventory(use_s3=False, dataset_slug=slug)]
    try:
        uris += [e.uri for e in cat.inventory(use_s3=True, dataset_slug=slug)]
    except Exception:
        pass
    good: List[str] = []
    for u in uris:
        for ext in SUPPORTED_EXTS:
            if u.lower().endswith(ext):
                good.append(u)
                break
    return sorted(set(good))


def _read_union(storage, uris: List[str]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for u in uris:
        try:
            df = read_any(storage, u, backend="pandas")
        except Exception:
            continue
        if df is None or len(df) == 0:
            continue
        df = pd.DataFrame(df).copy()
        df["_source_uri"] = u
        frames.append(df)
    if not frames:
        raise RuntimeError("No readable data files found for slug.")
    return pd.concat(frames, ignore_index=True, sort=False)


def _append_jsonl(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")


# ------------------------------ main flow ------------------------------

def ingest_to_silver(
    slug: str,
    cfg: RootCfg,
    *,
    subset: int = 0,
    sort_by: Iterable[str] = (),
    mode: Literal["auto", "bootstrap", "apply"] = "auto",
) -> Tuple[pd.DataFrame, ProposedSchema]:
    """
    Ingest raw → bronze → silver.

    Modes:
      - bootstrap: run NLP/clean loop and freeze schema (original behavior)
      - apply:     use frozen schema (no role guessing, no loop)
      - auto:      apply if schema exists, else bootstrap
    """
    log = get_logger("ingest", cfg.logging.level, cfg.logging.structured_json)
    storage = build_storage_from_config(cfg)
    paths = Paths(root=Path.cwd(), slug=slug)

    # Decide mode
    schema_exists = paths.schema_toml.exists()
    if mode == "auto":
        mode = "apply" if schema_exists else "bootstrap"
    log.info("ingest_mode", extra={"slug": slug, "mode": mode, "schema_exists": schema_exists})

    # 1) Discover + read all raw
    uris = _discover_uris(storage, slug)
    if not uris:
        raise FileNotFoundError(f"No supported files under raw/{slug} (local or S3).")
    log.info("raw_discovered", extra={"files": len(uris)})

    df_raw = _read_union(storage, uris)
    if sort_by:
        cols = [c for c in sort_by if c in df_raw.columns]
        if cols:
            try:
                df_raw = df_raw.sort_values(cols, kind="mergesort", ignore_index=True)
            except Exception:
                pass

    # 2) Bronze normalize (cheap/safe)
    bronze = _prepare_for_parquet(df_raw)
    paths.bronze_dir.mkdir(parents=True, exist_ok=True)
    write_any(storage, bronze, str(paths.bronze_parquet), fmt="parquet")
    log.info(
        "bronze_written",
        extra={"rows": len(bronze), "cols": bronze.shape[1], "path": str(paths.bronze_parquet)},
    )

    # Optional sample for bootstrap loop only
    sample = bronze
    if mode == "bootstrap" and subset and len(sample) > subset:
        sample = sample.sample(n=subset, random_state=cfg.env.seed)

    # 3) Proposed schema
    if mode == "apply":
        if not schema_exists:
            raise FileNotFoundError(f"No frozen schema at {paths.schema_toml} for apply mode.")
        proposed = _load_frozen_schema(slug, paths.root)
    else:
        # bootstrap: guess roles from sample
        proposed = _to_proposed_schema(slug, sample, cfg)

    # 4) Cleaning
    if mode == "apply":
        # Deterministic one-pass apply respecting frozen schema + your policy
        result = run_clean_pass(bronze, proposed, cfg, extra_rules=None)
        best_df = result.clean_df
        best_schema = proposed  # do not modify frozen schema
    else:
        # bootstrap loop until thresholds (existing behavior)
        min_schema = float(cfg.nlp.min_schema_confidence)
        min_role = float(cfg.nlp.min_role_confidence)
        max_iter = int(cfg.nlp.max_iter)
        min_improv = float(cfg.nlp.min_improvement)

        prev_score = -1e9
        best_df = sample
        best_schema = proposed
        paths.iter_log.unlink(missing_ok=True)

        for it in range(1, max_iter + 1):
            result = run_clean_pass(sample, proposed, cfg, extra_rules=None)
            df_after = result.clean_df

            r = (
                result.rescore
                if isinstance(result.rescore, dict)
                else rescore_after_clean(df_after, proposed, cfg.profiling.roles, cfg.nlp).__dict__
            )
            schema_conf = float(r.get("schema_conf_after", 0.0))
            avg_role_conf = float(r.get("avg_role_conf_after", 0.0))
            score = schema_conf * 0.7 + avg_role_conf * 0.3

            _append_jsonl(
                paths.iter_log,
                {
                    "iter": it,
                    "schema_conf": schema_conf,
                    "avg_role_conf": avg_role_conf,
                    "score": score,
                    "width": df_after.shape[1],
                    "missing_rate": float(df_after.isna().sum().sum())
                    / float(df_after.shape[0] * max(df_after.shape[1], 1)),
                    "rules_applied": result.report.get("rules", {}).get("total_applications", None),
                },
            )

            if score >= (prev_score + min_improv):
                best_df = df_after
                best_schema = _to_proposed_schema(slug, df_after, cfg)
                prev_score = score

            if schema_conf >= min_schema and avg_role_conf >= min_role:
                log.info(
                    "loop_stop_thresholds",
                    extra={"iter": it, "schema_conf": schema_conf, "avg_role_conf": avg_role_conf},
                )
                best_df = df_after
                best_schema = _to_proposed_schema(slug, df_after, cfg)
                break

            sugg = result.report.get("suggestions", {}) or {}
            extra = _safe_rules_from_suggestions(sugg)
            if not extra:
                log.info("loop_stop_no_suggestions", extra={"iter": it})
                break
            result2 = run_clean_pass(df_after, proposed, cfg, extra_rules=extra)
            sample = result2.clean_df
            r2 = (
                result2.rescore
                if isinstance(result2.rescore, dict)
                else rescore_after_clean(sample, proposed, cfg.profiling.roles, cfg.nlp).__dict__
            )
            schema_conf2 = float(r2.get("schema_conf_after", 0.0))
            avg_role_conf2 = float(r2.get("avg_role_conf_after", 0.0))
            score2 = schema_conf2 * 0.7 + avg_role_conf2 * 0.3
            _append_jsonl(
                paths.iter_log,
                {
                    "iter": it,
                    "after_suggestions": True,
                    "schema_conf": schema_conf2,
                    "avg_role_conf": avg_role_conf2,
                    "score": score2,
                },
            )
            if score2 > prev_score + min_improv:
                best_df = sample
                best_schema = _to_proposed_schema(slug, sample, cfg)
                prev_score = score2
            else:
                if it >= max_iter:
                    break

    # 5) Write silver + schema report
    paths.silver_dir.mkdir(parents=True, exist_ok=True)
    silver_path = paths.silver_dir / "dataset.parquet"
    write_any(storage, best_df, str(silver_path), fmt="parquet")

    rep_path = paths.silver_dir / "schema_report.parquet"
    rep = pd.DataFrame(
        {
            "name": list(best_df.columns),
            "dtype": [str(t) for t in best_df.dtypes],
            "missing_pct": [float(best_df[c].isna().mean()) for c in best_df.columns],
        }
    )
    write_any(storage, rep, str(rep_path), fmt="parquet")

    # 6) Freeze schema only in bootstrap
    if mode == "bootstrap":
        paths.schema_toml.parent.mkdir(parents=True, exist_ok=True)
        paths.schema_toml.write_text(to_toml(best_schema.to_dict()), encoding="utf-8")
        log.info("schema_frozen", extra={"path": str(paths.schema_toml)})

    log.info("silver_written", extra={"rows": len(best_df), "cols": best_df.shape[1], "path": str(silver_path)})

    return best_df, best_schema


# ------------------------------- CLI -----------------------------------

def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Ingest raw → bronze → silver. "
            "Modes: bootstrap (discover+loop+freeze), apply (use frozen schema), auto (default). "
            "If no --dataset/--slug is provided, all discovered slugs are processed."
        )
    )
    ap.add_argument("--config", default="config/config.toml")
    ap.add_argument("--dataset", "--slug", dest="slug", required=False)
    ap.add_argument("--subset", type=int, default=0, help="Optional sample rows for NLP/clean loop (bootstrap only)")
    ap.add_argument("--sort-by", nargs="*", default=["event_time", "row_id"], help="Stable sort columns when present")
    ap.add_argument("--mode", choices=["auto", "bootstrap", "apply"], default="auto")
    return ap


def main() -> None:
    args = _build_argparser().parse_args()
    cfg = RootCfg.load(args.config)
    sort_by = args.sort_by or ["event_time", "row_id"]

    if args.slug:
        ingest_to_silver(args.slug, cfg, subset=int(args.subset or 0), sort_by=sort_by, mode=args.mode)
        return

    slugs = discover_all_slugs(cfg)
    if not slugs:
        raise SystemExit("No datasets found under configured raw roots (local/S3).")
    log = get_logger("ingest", cfg.logging.level, cfg.logging.structured_json)
    log.info("running_all_slugs", extra={"count": len(slugs), "slugs": slugs})
    for s in slugs:
        log.info("start_slug", extra={"slug": s})
        try:
            ingest_to_silver(s, cfg, subset=int(args.subset or 0), sort_by=sort_by, mode=args.mode)
        except Exception as e:
            log.error("slug_failed", extra={"slug": s, "error": str(e)})
        else:
            log.info("slug_done", extra={"slug": s})


if __name__ == "__main__":
    main()
