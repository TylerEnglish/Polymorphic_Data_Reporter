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