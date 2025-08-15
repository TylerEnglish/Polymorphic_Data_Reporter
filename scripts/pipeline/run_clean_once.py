from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from collections import Counter
from typing import List, Tuple, Optional
import re
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype

# your internal modules
from src.config_model.model import RootCfg
from src.cleaning.engine import run_clean_pass
from src.cleaning.rescore import rescore_after_clean

# Try to use your IO stack, but fall back to direct pandas if not available
try:
    from src.io.storage import Storage  # type: ignore
    from src.io.catalog import Catalog  # type: ignore
    from src.io.readers import read_any  # type: ignore
except Exception:  # pragma: no cover
    Storage = None
    Catalog = None
    read_any = None

_TIME_NAME_RE = re.compile(r"(date|time|timestamp|dt|ts)\b", re.I)

# ------------------------ existing helpers (unchanged) ------------------------

def _load_df(bronze_dir: Path) -> pd.DataFrame:
    parq = bronze_dir / "normalize.parquet"
    if not parq.exists():
        raise FileNotFoundError(
            f"normalize.parquet not found at {parq}\n"
            "Run with --from-raw to build it from data/raw/<slug>/**."
        )
    return pd.read_parquet(parq)

def _load_proposed_schema(bootstrap_json: Path, df: pd.DataFrame):
    class _SchemaShim:
        def __init__(self, cols, conf: float | None = None):
            self._cols = cols
            self.schema_confidence = float(conf or 0.0)
        def to_dict(self):
            return {
                "columns": self._cols,
                "schema_confidence": self.schema_confidence,
            }

    cols_out = []
    schema_conf = 0.0

    if bootstrap_json.exists():
        obj = json.loads(bootstrap_json.read_text(encoding="utf-8"))
        src_cols = obj.get("columns") or obj.get("schema", {}).get("columns") or []
        for c in src_cols:
            name = c.get("name") or c.get("column") or c.get("col")
            if not name: continue
            role = c.get("role", "text")
            conf = c.get("confidence", None)
            cols_out.append({"name": str(name), "role": str(role), "confidence": conf})
        schema_conf = float(obj.get("schema_confidence", 0.0) or 0.0)

    if not cols_out:
        for name, dt in df.dtypes.items():
            if pd.api.types.is_datetime64_any_dtype(dt):
                role = "time"
            elif pd.api.types.is_numeric_dtype(dt):
                role = "numeric"
            elif str(name).lower().endswith("_id") or str(name).lower() in {"id","row_id","entity_id"}:
                role = "id"
            else:
                role = "text"
            cols_out.append({"name": str(name), "role": role})

    return _SchemaShim(cols_out, schema_conf)

def _print_head_and_dtypes(tag: str, df: pd.DataFrame, head: int):
    print(f"\n=== {tag}: HEAD({head}) ===")
    try:
        print(df.head(head).to_string(index=False))
    except Exception:
        print(df.head(head))
    print(f"\n=== {tag}: DTYPES ===")
    d = {c: str(t) for c, t in df.dtypes.items()}
    for k in d:
        print(f"  {k}: {d[k]}")

def _nlp_score_for_df(df: pd.DataFrame, proposed, cfg: RootCfg) -> tuple[float, float]:
    r = rescore_after_clean(
        df=df,
        prev_schema=proposed,
        profiling_cfg=cfg.profiling,
        nlp_cfg=cfg.nlp
    )
    return float(r.schema_conf_after), float(r.avg_role_conf_after)


# ------------------------ NEW: raw ingestion ------------------------

SUPPORTED_EXTS = {".csv", ".parquet", ".json", ".ndjson", ".xlsx", ".xls"}

def _discover_uris(project_root: Path, slug: str) -> List[str]:
    """Try Catalog first, fall back to filesystem crawl."""
    uris: List[str] = []
    try:
        if Storage and Catalog:
            storage = Storage()  # your Storage likely uses cwd / config internally
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
    """Read a single file → DataFrame, or None on failure."""
    # Prefer your project's reader
    if read_any and Storage:
        try:
            df = read_any(Storage(), uri, backend="pandas")  # type: ignore
            if isinstance(df, pd.DataFrame):
                return df
            return pd.DataFrame(df)
        except Exception:
            pass

    # Fallback readers
    p = Path(uri)
    ext = p.suffix.lower()
    try:
        if ext == ".parquet":
            return pd.read_parquet(uri)
        if ext == ".csv":
            return pd.read_csv(uri)
        if ext in {".json", ".ndjson"}:
            # Try ndjson if many lines
            try:
                return pd.read_json(uri, lines=True)
            except ValueError:
                return pd.read_json(uri)
        if ext in {".xlsx", ".xls"}:
            return pd.read_excel(uri)
    except Exception as e:
        print(f"[warn] failed to read {uri}: {e}", file=sys.stderr)
        return None
    return None

def _ingest_all_raw(project_root: Path, slug: str, *, max_files: Optional[int], sort_by: List[str]) -> pd.DataFrame:
    uris = _discover_uris(project_root, slug)
    if not uris:
        raise FileNotFoundError(f"No input files found under data/raw/{slug}/ (or Catalog inventory was empty).")
    if max_files is not None:
        uris = uris[:max_files]

    per_file_rows: List[Tuple[str, int]] = []
    frames: List[pd.DataFrame] = []

    for i, u in enumerate(uris, 1):
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

    # Optional sort if user asked and columns exist
    sort_cols = [c for c in sort_by if c in df_all.columns]
    if sort_cols:
        try:
            df_all = df_all.sort_values(sort_cols, kind="mergesort", ignore_index=True)
        except Exception:
            pass

    # Inventory summary
    by_ext = Counter(Path(u).suffix.lower() for u, _ in per_file_rows)
    total_rows = sum(n for _, n in per_file_rows)
    print(f"[ingest] files={len(per_file_rows)}  by_ext={dict(by_ext)}  rows_total={total_rows}  rows_concat={len(df_all)}")
    return df_all


def _best_datetime_parse(obj_series: pd.Series) -> pd.Series:
    """
    Try several parses for mixed string/number datetime columns and pick the best:
      - generic to_datetime for strings/iso/etc.
      - epoch milliseconds
      - epoch seconds
    We only accept candidates with years in a plausible range (1970..2100).
    Returns a datetime64[ns] Series (may be all NaT).
    """
    s = obj_series

    def _plausible(ts: pd.Series) -> pd.Series:
        try:
            years = ts.dt.year
            return ts.where(years.between(1970, 2100))
        except Exception:
            return ts

    # candidate A: generic parse (handles strings and existing datetimes)
    try:
        cand_a = pd.to_datetime(s, errors="coerce")
        cand_a = _plausible(cand_a)
    except Exception:
        cand_a = pd.Series(pd.NaT, index=s.index)

    # candidate B: epoch milliseconds (for numeric-looking values)
    try:
        nums = pd.to_numeric(s, errors="coerce")
        cand_b = pd.to_datetime(nums, unit="ms", errors="coerce")
        cand_b = _plausible(cand_b)
    except Exception:
        cand_b = pd.Series(pd.NaT, index=s.index)

    # candidate C: epoch seconds
    try:
        nums = pd.to_numeric(s, errors="coerce")
        cand_c = pd.to_datetime(nums, unit="s", errors="coerce")
        cand_c = _plausible(cand_c)
    except Exception:
        cand_c = pd.Series(pd.NaT, index=s.index)

    candidates = [cand_a, cand_b, cand_c]
    nn_counts = [int(c.notna().sum()) for c in candidates]
    best_idx = int(np.argmax(nn_counts))
    best = candidates[best_idx]

    # be conservative: only adopt if ≥60% of non-null values parsed
    non_null = int(s.notna().sum())
    if non_null == 0:
        return best
    if nn_counts[best_idx] / max(1, non_null) >= 0.60:
        return best.dt.tz_localize(None)
    return pd.Series(pd.NaT, index=s.index)

def _prepare_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make the frame Arrow-friendly:
      - Coerce obvious time columns to datetime
      - Coerce numeric-looking object columns to numeric
      - Stringify lists/dicts/other Python objects
      - Cast remaining text to pandas 'string' dtype
    """
    out = df.copy()

    for col in out.columns:
        s = out[col]

        # Skip if already Arrow-friendly
        if pd.api.types.is_datetime64_any_dtype(s):
            continue
        if pd.api.types.is_numeric_dtype(s):
            continue
        if pd.api.types.is_bool_dtype(s):
            continue

        # Categorical → string
        if isinstance(s.dtype, CategoricalDtype):
            out[col] = s.astype("string")
            continue

        # Objects / mixed
        if s.dtype == object:
            # 1) Time-ish by name? Try best datetime parse.
            if _TIME_NAME_RE.search(str(col)):
                parsed = _best_datetime_parse(s)
                if parsed.notna().any():
                    out[col] = parsed
                    continue  # good

            # 2) Numeric-like? (e.g., numbers + strings of numbers)
            nums = pd.to_numeric(s, errors="coerce")
            if nums.notna().mean() >= 0.80:
                out[col] = nums
                continue

            # 3) Lists/dicts/other python objects → JSON/text
            def _to_text(x):
                if x is None or (isinstance(x, float) and np.isnan(x)):
                    return None
                if isinstance(x, (dict, list, tuple)):
                    try:
                        return json.dumps(x, ensure_ascii=False)
                    except Exception:
                        return str(x)
                return str(x)

            try:
                out[col] = s.map(_to_text).astype("string")
            except Exception:
                out[col] = s.astype("string")
            continue

        # Fallback: coerce any remaining text-likes to 'string'
        try:
            out[col] = out[col].astype("string")
        except Exception:
            pass

    return out

def _ensure_bronze(df: pd.DataFrame, bronze_dir: Path):
    bronze_dir.mkdir(parents=True, exist_ok=True)
    out = bronze_dir / "normalize.parquet"

    df_arrow = _prepare_for_parquet(df)

    # Optional: show any remaining 'object' columns before write (should be none)
    leftover_obj = [c for c in df_arrow.columns if df_arrow[c].dtype == object]
    if leftover_obj:
        print(f"[bronze] warning: object dtype columns remain → {leftover_obj}")

    df_arrow.to_parquet(out, index=False)  # pyarrow engine by default
    print(f"[bronze] wrote {out} ({len(df_arrow)} rows, {df_arrow.shape[1]} cols)")

# ------------------------ main ------------------------

def main():
    ap = argparse.ArgumentParser(description="Run ONE cleaning pass for a single slug; can rebuild bronze from RAW.")
    ap.add_argument("--config", default="config/config.toml")
    ap.add_argument("--slug", default="f1", help="Dataset slug under data/raw/<slug> and data/bronze/<slug>")
    ap.add_argument("--head", type=int, default=5)

    # NEW options
    ap.add_argument("--from-raw", action="store_true",
                    help="Recursively read ALL files under data/raw/<slug>/**, merge them, write bronze/normalize.parquet, then clean.")
    ap.add_argument("--max-files", type=int, default=None,
                    help="Optional cap on number of files to read from RAW (debugging).")
    ap.add_argument("--sort-by", nargs="*", default=[],
                    help="Optional column(s) to sort by after merge (e.g., --sort-by event_time row_id).")

    args = ap.parse_args()

    project_root = Path.cwd()
    cfg = RootCfg.load(args.config)

    bronze_dir = project_root / "data" / "bronze" / args.slug
    bootstrap_json = bronze_dir / "nlp_bootstrap.json"

    print(f"[load] slug={args.slug}")

    # ---- Build/load the working frame ----
    if args.from_raw:
        df_before = _ingest_all_raw(project_root, args.slug, max_files=args.max_files, sort_by=args.sort_by)
        _ensure_bronze(df_before, bronze_dir)  # keep pipeline convention intact
    else:
        # Use existing bronze parquet
        df_before = _load_df(bronze_dir)

    proposed = _load_proposed_schema(bootstrap_json, df_before)

    def _missing_pct(df, cols):
        out = {}
        for c in cols:
            if c in df.columns:
                out[c] = float(df[c].isna().mean())
        return out

    # pick some likely time/num columns to watch, but filter by presence
    watch_candidates = ["num_03", "dt_01", "dt_02", "dt_03", "event_time"]
    watch = [c for c in watch_candidates if c in df_before.columns]

    # ---- BEFORE snapshots ----
    print("[MISS% BEFORE]", _missing_pct(df_before, watch))
    _print_head_and_dtypes("BEFORE", df_before, args.head)

    # BEFORE NLP score
    schema_before, avg_role_before = _nlp_score_for_df(df_before, proposed, cfg)
    print(f"\n[NLP] BEFORE  schema_conf_after={schema_before:.4f}  avg_role_conf_after={avg_role_before:.4f}")

    # ---- CLEAN once ----
    result = run_clean_pass(df_before, proposed, cfg)
    df_after = result.clean_df

    # ---- AFTER snapshots ----
    print("[MISS% AFTER ]", _missing_pct(df_after, [c for c in watch if c in df_after.columns]))

    # deltas for watched columns
    bmiss = _missing_pct(df_before, watch)
    amiss = _missing_pct(df_after,  watch)
    deltas = {c: (bmiss.get(c), amiss.get(c), (None if bmiss.get(c) is None or amiss.get(c) is None else amiss[c]-bmiss[c]))
              for c in set(bmiss) | set(amiss)}
    print("[MISS% Δ     ]", deltas)

    _print_head_and_dtypes("AFTER", df_after, args.head)

    # AFTER NLP score
    schema_after, avg_role_after = _nlp_score_for_df(df_after, proposed, cfg)
    print(f"\n[NLP] AFTER   schema_conf_after={schema_after:.4f}  avg_role_conf_after={avg_role_after:.4f}")

    # Summary + which rules actually fired
    print("\n=== SUMMARY ===")
    print(f"Rows: {len(df_before)} -> {len(df_after)} | Cols: {df_before.shape[1]} -> {df_after.shape[1]}")
    actions = result.report.get("rules", {}).get("per_column_actions", {})
    fired = {c: acts for c, acts in actions.items() if acts}
    if not fired:
        print("No column-level actions fired.")
    else:
        print("Actions by column (non-empty only):")
        for c, acts in fired.items():
            print(f"  {c}: {', '.join(acts)}")


if __name__ == "__main__":
    main()
