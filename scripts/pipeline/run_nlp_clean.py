from __future__ import annotations

import argparse
import json
import sys
import shutil
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from pandas.core.dtypes.dtypes import DatetimeTZDtype
import pandas as pd

from src.config_model.model import RootCfg
from src.io.storage import build_storage_from_config
from src.nlp.bootstrap import run_nlp_bootstrap
from src.nlg.narrative import narrative_payload
from src.cleaning.engine import run_clean_pass
from src.cleaning.metrics import profile_columns

# ---------- constants ----------
DATA_EXTS = {".csv", ".json", ".ndjson", ".parquet", ".pq", ".feather", ".arrows", ".txt"}
NORMALIZE_EXTS = {".parquet", ".pq", ".feather", ".csv", ".ndjson", ".json"}

# ---------- path helpers ----------
def _is_hidden(p: Path) -> bool:
    return p.name.startswith(".") or p.name.startswith("_")

def _is_data_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in DATA_EXTS

def _raw_root(cfg: RootCfg) -> Path:
    rr = Path(cfg.storage.local.raw_root)
    rr.mkdir(parents=True, exist_ok=True)
    return rr

def _bronze_root_from_raw(raw_root: Path) -> Path:
    br = raw_root.parent / "bronze"
    br.mkdir(parents=True, exist_ok=True)
    return br

def _silver_root_from_arg(out_root: Optional[str]) -> Path:
    sr = Path(out_root or "data/silver")
    sr.mkdir(parents=True, exist_ok=True)
    return sr

def _ensure_slug_dir_for_single_file(raw_root: Path, file_path: Path) -> str:
    slug = file_path.stem
    slug_dir = raw_root / slug
    slug_dir.mkdir(parents=True, exist_ok=True)
    staged = slug_dir / file_path.name
    if not staged.exists():
        shutil.copy2(file_path, staged)
    return slug

def _slugify_file(raw_root: Path, f: Path) -> str:
    rel = f.relative_to(raw_root)
    parts = list(rel.parts)
    parts[-1] = Path(parts[-1]).stem
    return "__".join(parts)

def _targets_by_slug(raw_root: Path) -> List[Tuple[str, Path]]:
    out: List[Tuple[str, Path]] = []
    for child in sorted(raw_root.iterdir()):
        if _is_hidden(child): continue
        if child.is_dir():
            out.append((child.name, child))
        elif _is_data_file(child):
            slug = _ensure_slug_dir_for_single_file(raw_root, child)
            out.append((slug, raw_root / slug))
    return out

def _targets_by_subdir(raw_root: Path) -> List[Tuple[str, Path]]:
    out: List[Tuple[str, Path]] = []
    for slug_dir in sorted(p for p in raw_root.iterdir() if p.is_dir() and not _is_hidden(p)):
        subs = [p for p in slug_dir.iterdir() if p.is_dir() and not _is_hidden(p)]
        if not subs:
            out.append((slug_dir.name, slug_dir)); continue
        for sd in subs:
            out.append((f"{slug_dir.name}__{sd.name}", sd))
    return out

def _targets_by_file(raw_root: Path) -> List[Tuple[str, Path]]:
    out: List[Tuple[str, Path]] = []
    for f in sorted(raw_root.rglob("*")):
        if _is_hidden(f) or not _is_data_file(f):
            continue
        slug = _slugify_file(raw_root, f)
        slug_dir = raw_root / slug
        slug_dir.mkdir(parents=True, exist_ok=True)
        staged = slug_dir / f.name
        if staged.resolve() != f.resolve() and not staged.exists():
            shutil.copy2(f, staged)
        out.append((slug, slug_dir))
    return out

def _resolve_targets(raw_root: Path, target: Optional[str], *, granularity: str) -> List[Tuple[str, Path]]:
    if target is not None:
        p = Path(target)
        if not p.is_absolute():
            candidate = (raw_root / target).resolve()
            if candidate.exists():
                p = candidate
            else:
                if (raw_root / target).exists():
                    p = (raw_root / target).resolve()
                elif Path(target).exists():
                    p = Path(target).resolve()
                else:
                    raise FileNotFoundError(f"Could not resolve selection '{target}' under {raw_root}")
        if p == raw_root:
            return {"slug": _targets_by_slug, "subdir": _targets_by_subdir, "file": _targets_by_file}[granularity](raw_root)
        if p.is_dir():
            return [(p.name, p)]
        if p.is_file():
            if p.parent == raw_root and _is_data_file(p):
                slug = _ensure_slug_dir_for_single_file(raw_root, p)
                return [(slug, raw_root / slug)]
            slug = p.stem
            slug_dir = raw_root / slug
            slug_dir.mkdir(parents=True, exist_ok=True)
            staged = slug_dir / p.name
            if not staged.exists():
                shutil.copy2(p, staged)
            return [(slug, slug_dir)]
        raise FileNotFoundError(f"Selection '{target}' does not exist or is not a file/folder.")
    return {"slug": _targets_by_slug, "subdir": _targets_by_subdir, "file": _targets_by_file}[granularity](raw_root)


def _strip_tz_inplace(df: pd.DataFrame) -> None:
    """
    Make any tz-aware datetimes naive UTC (for Parquet/Arrow).
    Avoids pandas' deprecated is_datetime64tz_dtype path.
    """
    for col in df.columns:
        dt = df[col].dtype
        # native tz-aware dtype
        if isinstance(dt, DatetimeTZDtype):
            try:
                df[col] = df[col].dt.tz_convert("UTC").dt.tz_localize(None)
            except Exception:
                # if it's already UTC or weird, just drop tz
                df[col] = df[col].dt.tz_localize(None)
        else:
            # object columns might still hold tz-aware Timestamps
            if df[col].dtype == "object":
                try:
                    s = pd.to_datetime(df[col], errors="coerce", utc=True)
                    if isinstance(s.dtype, DatetimeTZDtype):
                        df[col] = s.dt.tz_localize(None)
                except Exception:
                    pass

# ---------- NLP stage ----------
def _run_nlp(project_root: Path, cfg: RootCfg, slug: str) -> None:
    st = build_storage_from_config(cfg)
    out = run_nlp_bootstrap(st, slug, project_root=project_root)

    schema_path = Path(out["io"]["schema_toml_path"])
    schema_path.parent.mkdir(parents=True, exist_ok=True)
    schema_path.write_text(out["io"]["schema_toml_text"], encoding="utf-8")

    bootstrap_path = Path(out["io"]["bootstrap_json_path"])
    bootstrap_path.parent.mkdir(parents=True, exist_ok=True)
    bootstrap_path.write_text(json.dumps(out["bootstrap"], indent=2), encoding="utf-8")

    nar = narrative_payload(slug, out["bootstrap"], out["entries"])
    (bootstrap_path.parent / "narrative.json").write_text(json.dumps(nar, indent=2), encoding="utf-8")

    print(f"[nlp] ✓ {slug}: schema -> {schema_path}, bootstrap -> {bootstrap_path}, narrative -> {bootstrap_path.parent / 'narrative.json'}")

# ---------- normalize (robust + debug) ----------
def _read_df_auto(path: Path, *, debug: bool) -> pd.DataFrame:
    suf = path.suffix.lower()
    try:
        if suf in (".parquet", ".pq"): return pd.read_parquet(path)
        if suf == ".csv":               return pd.read_csv(path)
        if suf in (".feather", ".ft"):  return pd.read_feather(path)
        if suf == ".json":
            try:    return pd.read_json(path, lines=False)
            except ValueError:
                if debug: print(f"[norm:DEBUG] {path.name}: standard JSON failed; retrying as NDJSON")
                return pd.read_json(path, lines=True)
        if suf == ".ndjson":            return pd.read_json(path, lines=True)
        if suf == ".txt":               return pd.read_csv(path)
    except Exception as e:
        raise RuntimeError(f"read_error({path}): {e}")
    raise RuntimeError(f"Unsupported file type: {path.suffix}")

def _maybe_flatten_jsonlike(df: pd.DataFrame, *, debug: bool) -> pd.DataFrame:
    try:
        sample = df.head(50)
        dict_cols = [c for c in df.columns if any(isinstance(v, dict) for v in sample[c].dropna())]
        if dict_cols:
            if debug: print(f"[norm:DEBUG] flattening dict-like columns: {dict_cols}")
            extras = []
            for c in dict_cols:
                expanded = pd.json_normalize(df[c]).add_prefix(f"{c}.")
                extras.append(expanded)
            df = pd.concat([df.drop(columns=dict_cols).reset_index(drop=True)] + [x.reset_index(drop=True) for x in extras], axis=1)
    except Exception as e:
        if debug: print(f"[norm:DEBUG] flatten failed (continuing): {e}")
    return df

def _coerce_problem_object_cols(df: pd.DataFrame, datetime_formats: List[str], *, debug: bool) -> Tuple[pd.DataFrame, List[str]]:
    """
    Make Parquet-safe:
      - parse time-looking object cols to datetime
      - tz-aware -> naive UTC
      - object cols with mixed numeric/text -> cast to pandas StringDtype
      - object cols fully numeric after coercion -> Float64 (nullable)
      - otherwise -> StringDtype
    """
    from pandas.api.types import is_datetime64tz_dtype

    actions: List[str] = []
    for col in list(df.columns):
        s = df[col]
        try:
            # 1) Datetime-like in object
            if s.dtype == "object":
                sample = s.dropna().head(200).astype(str)
                looks_time = any(":" in v or "-" in v or "/" in v for v in sample)
                if looks_time:
                    parsed = None
                    for f in (datetime_formats or []):
                        try:
                            cand = pd.to_datetime(s, format=f, errors="coerce")
                        except Exception:
                            continue
                        parsed = cand if parsed is None else parsed.fillna(cand)
                    if parsed is None:
                        parsed = pd.to_datetime(s, errors="coerce")
                    df[col] = parsed
                    actions.append(f"{col}:to_datetime")
                    # continue; this column is now datetime
                else:
                    # 2) numeric-like / mixed
                    coer = pd.to_numeric(s, errors="coerce")
                    num_ratio = float(coer.notna().mean()) if len(s) else 0.0
                    # mixed numeric & text -> make strings to avoid Arrow guessing double
                    if 0.0 < num_ratio < 1.0:
                        df[col] = s.astype("string")
                        actions.append(f"{col}:object_mixed->string")
                        continue
                    # all numeric after coercion
                    if num_ratio == 1.0:
                        df[col] = coer.astype("Float64")
                        actions.append(f"{col}:object_numeric->float")
                        continue
                    # otherwise, just formalize as string
                    df[col] = s.astype("string")
                    actions.append(f"{col}:object->string")
                    continue

            # 3) tz-aware datetimes -> naive
            if is_datetime64tz_dtype(df[col].dtype):
                try:
                    df[col] = df[col].dt.tz_convert("UTC").dt.tz_localize(None)
                except Exception:
                    df[col] = df[col].dt.tz_localize(None)
                actions.append(f"{col}:tz_to_naive")
        except Exception as e:
            if debug: print(f"[norm:DEBUG] column '{col}' coercion error: {e}")
    return df, actions

def _find_normalize_anywhere(base: Path) -> Optional[Path]:
    for ext in NORMALIZE_EXTS:
        p = base / f"normalize{ext}"
        if p.exists(): return p
    for p in base.rglob("*"):
        if p.is_file() and p.suffix.lower() in NORMALIZE_EXTS and p.stem.lower() == "normalize":
            return p
    return None

def _find_any_data_file(base: Path) -> Optional[Path]:
    exclude = {"nlp_bootstrap.json", "narrative.json", "schema.toml"}
    for p in sorted(base.rglob("*")):
        if p.is_file() and p.suffix.lower() in DATA_EXTS and not _is_hidden(p) and p.name not in exclude:
            return p
    return None

def _guess_input_path(slug: str, raw_slug_dir: Path, bronze_root: Path) -> Tuple[Optional[Path], Optional[Path], str]:
    bronze_slug_dir = bronze_root / slug
    if bronze_slug_dir.exists():
        p = _find_normalize_anywhere(bronze_slug_dir)
        if p: return p, bronze_slug_dir, "normalize"
        p = _find_any_data_file(bronze_slug_dir)
        if p: return p, bronze_slug_dir, "fallback-data"
    if raw_slug_dir.exists():
        p = _find_normalize_anywhere(raw_slug_dir)
        if p: return p, raw_slug_dir, "normalize"
        p = _find_any_data_file(raw_slug_dir)
        if p: return p, raw_slug_dir, "fallback-data"
    return None, None, "not-found"

def _run_normalize(cfg: RootCfg, slug: str, raw_slug_dir: Path, *, debug: bool) -> Optional[Path]:
    raw_root = _raw_root(cfg)
    bronze_root = _bronze_root_from_raw(raw_root)
    norm_path, src_dir, reason = _guess_input_path(slug, raw_slug_dir, bronze_root)
    if not norm_path:
        print(f"[norm] ✗ {slug}: no normalize.* and no other data files found in {raw_slug_dir} or {bronze_root / slug}")
        return None
    if reason == "fallback-data":
        print(f"[norm] • {slug}: normalize.* not found; using data file: {norm_path}")

    try:
        df = _read_df_auto(norm_path, debug=debug)
    except Exception as e:
        print(f"[norm] ✗ {slug}: {e}")
        if debug: traceback.print_exc()
        return None

    df = _maybe_flatten_jsonlike(df, debug=debug)
    dt_formats = []
    try:
        dt_formats = list(getattr(getattr(cfg, "profiling", object()), "roles", object()).datetime_formats or [])
    except Exception:
        pass
    df, actions = _coerce_problem_object_cols(df, dt_formats, debug=debug)

    bronze_slug = _bronze_root_from_raw(raw_root) / slug
    bronze_slug.mkdir(parents=True, exist_ok=True)
    out_parq = bronze_slug / "normalize.parquet"

    _strip_tz_inplace(df)
    df.to_parquet(out_parq, index=False)
    try:
        df.to_parquet(out_parq, index=False)
    except Exception as e:
        if debug:
            print(f"[norm:DEBUG] parquet write failed; dtypes: {df.dtypes.to_dict()}")
            traceback.print_exc()
        print(f"[norm] ✗ {slug}: parquet write failed: {e}")
        return None

    if actions:
        print(f"[norm] ✓ {slug}: {out_parq} (fixes: {', '.join(actions)})")
    else:
        print(f"[norm] ✓ {slug}: {out_parq}")
    return out_parq

# ---------- schema handling / re-inference ----------
class _ProposedSchemaShim:
    def __init__(self, columns: List[Dict[str, Any]], conf: float = 0.0) -> None:
        self.schema_confidence = float(conf or 0.0)
        out = []
        for c in columns:
            if isinstance(c, dict) and "name" in c and "role" in c:
                out.append({"name": str(c["name"]), "role": str(c["role"])})
        self._cols = out
    def to_dict(self) -> Dict[str, Any]:
        return {"columns": self._cols}

def _load_bootstrap_schema(blob: Dict[str, Any]) -> Dict[str, Any]:
    cols = blob.get("columns") or blob.get("schema", {}).get("columns") or []
    conf = float(blob.get("schema_confidence", 0.0))
    out_cols: List[Dict[str, Any]] = []
    for c in cols:
        if isinstance(c, dict) and "name" in c and "role" in c:
            out_cols.append({"name": str(c["name"]), "role": str(c["role"])})
        elif isinstance(c, (list, tuple)) and len(c) >= 2:
            out_cols.append({"name": str(c[0]), "role": str(c[1])})
    return {"columns": out_cols, "schema_confidence": conf}

def _load_bootstrap_json(path: Path) -> Dict[str, Any]:
    return _load_bootstrap_schema(json.loads(path.read_text(encoding="utf-8")))

def _load_narrative_json(path: Path) -> Dict[str, Any]:
    j = json.loads(path.read_text(encoding="utf-8"))
    payload = j.get("bootstrap", j)
    return _load_bootstrap_schema(payload)

def _schema_from_files(slug: str, bronze_root: Path) -> Optional[_ProposedSchemaShim]:
    bslug = bronze_root / slug
    if (bslug / "nlp_bootstrap.json").exists():
        s = _load_bootstrap_json(bslug / "nlp_bootstrap.json")
        return _ProposedSchemaShim(s["columns"], conf=float(s.get("schema_confidence", 0.0) or 0.0))
    if (bslug / "narrative.json").exists():
        s = _load_narrative_json(bslug / "narrative.json")
        return _ProposedSchemaShim(s["columns"], conf=float(s.get("schema_confidence", 0.0) or 0.0))
    return None

# ---------- cleaning (single pass) ----------
def _run_clean_once(cfg: RootCfg, slug: str, bronze_root: Path, outdir: Path, proposed: _ProposedSchemaShim, *, save_head: int, debug: bool) -> Tuple[bool, Optional[dict[str, Any]], Optional[pd.DataFrame]]:
    norm = _find_normalize_anywhere(bronze_root / slug)
    if not norm:
        print(f"[clean] ✗ {slug}: normalize.* not found in {bronze_root / slug}")
        return False, None, None
    try:
        df = pd.read_parquet(norm)
    except Exception as e:
        print(f"[clean] ✗ {slug}: failed to read {norm}: {e}")
        if debug: traceback.print_exc()
        return False, None, None

    try:
        result = run_clean_pass(df, proposed, cfg)
    except Exception as e:
        print(f"[clean] ✗ {slug}: cleaning failed: {e}")
        if debug: traceback.print_exc()
        return False, None, None

    outdir.mkdir(parents=True, exist_ok=True)
    try:
        result.clean_df.to_parquet(outdir / "clean.parquet", index=False)
        (outdir / "clean_report.json").write_text(json.dumps(result.report, indent=2), encoding="utf-8")
        if save_head and save_head > 0:
            (outdir / f"head{save_head}.csv").write_text(result.clean_df.head(save_head).to_csv(index=False), encoding="utf-8")
    except Exception as e:
        print(f"[clean] ✗ {slug}: failed to write outputs: {e}")
        if debug: traceback.print_exc()
        return False, None, None

    print(f"[clean] ✓ {slug}: wrote {outdir/'clean.parquet'}, {outdir/'clean_report.json'}{', head%d.csv' % save_head if save_head else ''}")
    return True, result.report, result.clean_df

# ---------- surrogate confidence (when guess_role has no confidences) ----------
def _surrogate_conf(clean_df: pd.DataFrame, proposed: _ProposedSchemaShim, cfg: RootCfg, *, debug: bool) -> Tuple[float, float]:
    schema_roles = {c["name"]: c["role"] for c in proposed.to_dict()["columns"]}
    metrics = profile_columns(clean_df, schema_roles, getattr(getattr(cfg, "profiling", object()), "roles", object()).datetime_formats or [])
    per_scores: List[float] = []

    for name, m in metrics.items():
        role = m.get("role", "text")
        mtype = m.get("type", "string")
        score = 0.0
        try:
            if role == "time":
                score = float(m.get("datetime_parse_ratio", 0.0))
            elif role == "numeric":
                if mtype in ("int", "float"):
                    score = 1.0
                else:
                    coer = pd.to_numeric(clean_df[name], errors="coerce")
                    score = float(coer.notna().mean())
            elif role == "categorical":
                ur = float(m.get("unique_ratio", 0.0))
                # smaller unique ratio => stronger categorical signal
                score = max(0.0, min(1.0, 1.0 - 5.0 * ur))
            elif role == "text":
                al = m.get("avg_len")
                score = max(0.3, min(1.0, (float(al) / 16.0) if al else 0.3))
            else:
                score = 0.3
        except Exception:
            score = 0.0
        per_scores.append(score)

    avg = float(sum(per_scores) / len(per_scores)) if per_scores else 0.0
    if debug:
        print(f"[loop:DEBUG] surrogate_avg_conf={avg:.4f} (derived from types/metrics)")
    return avg, avg  # treat schema & role the same for a simple fallback

def _improvement(prev: Optional[Tuple[float, float]], cur: Tuple[float, float]) -> float:
    if prev is None:
        return float("inf")
    (ps, pr) = prev
    (cs, cr) = cur
    return max(cs - ps, cr - pr)

def _confidence_pair(report: dict[str, Any]) -> Tuple[float, float]:
    r = report.get("rescore", {}) if isinstance(report, dict) else {}
    sc_after = float(r.get("schema_conf_after") or 0.0)
    rc_after = float(r.get("avg_role_conf_after") or 0.0)
    return sc_after, rc_after

# ---------- main ----------
def main() -> None:
    ap = argparse.ArgumentParser(description="Iterative NLP→Normalize→Clean loop until confidence thresholds are met.")
    ap.add_argument("--select", default=None, help="Slug, folder path, or single file (optional).")
    ap.add_argument("--granularity", choices=["slug", "subdir", "file"], default=None, help="How to split datasets when --select omitted.")
    ap.add_argument("--config", default="config/config.toml", help="Path to config TOML.")
    ap.add_argument("--out-root", default="data/silver", help="Where to write cleaned outputs (default: data/silver).")
    ap.add_argument("--save-head", type=int, default=0, help="Also save headN.csv for inspection (0 to skip).")
    ap.add_argument("--no-nlp", action="store_true", help="Skip NLP stage.")
    ap.add_argument("--debug", action="store_true", help="Verbose debug logging and tracebacks.")
    ap.add_argument("--surrogate-conf", action="store_true", help="Use surrogate confidences if rescore returns zeros.")
    ap.add_argument("--force-max-iters-when-no-conf", action="store_true", help="Ignore min_improvement when confidences are all zero; run to max_iter.")
    args = ap.parse_args()

    try:
        cfg = RootCfg.load(args.config)
    except Exception as e:
        print(f"[pipeline] Failed to load config: {e}", file=sys.stderr); sys.exit(2)

    raw_root = _raw_root(cfg)
    bronze_root = _bronze_root_from_raw(raw_root)
    out_root = _silver_root_from_arg(args.out_root)

    nlp_cfg = getattr(cfg, "nlp", object())
    min_schema_conf = float(getattr(nlp_cfg, "min_schema_confidence", 0.9) or 0.9)
    min_role_conf   = float(getattr(nlp_cfg, "min_role_confidence", 0.9) or 0.9)
    max_iter        = int(getattr(nlp_cfg, "max_iter", 10) or 10)
    min_impr        = float(getattr(nlp_cfg, "min_improvement", 0.00) or 0.00)

    granularity = args.granularity or getattr(getattr(cfg, "nlp", object()), "granularity", "slug")
    try:
        targets = _resolve_targets(raw_root, args.select, granularity=granularity)
    except Exception as e:
        print(f"[pipeline] {e}", file=sys.stderr); sys.exit(2)

    if not targets:
        print(f"[pipeline] No targets found under {raw_root}"); return

    items = sorted({slug: p for slug, p in targets}.items())

    completed = 0
    for slug, slug_dir in items:
        # NLP
        if not args.no_nlp:
            try:
                _run_nlp(Path.cwd(), cfg, slug)
            except Exception as e:
                print(f"[nlp] ✗ {slug}: {e}")
                if args.debug: traceback.print_exc()

        # Normalize
        if not _run_normalize(cfg, slug, slug_dir, debug=args.debug):
            continue

        # Iterative clean/rescore
        out_slug_root = out_root / slug
        prev_pair: Optional[Tuple[float, float]] = None

        proposed = _schema_from_files(slug, bronze_root)
        if not proposed:
            try:
                df0 = pd.read_parquet((bronze_root / slug / "normalize.parquet"))
                # naive schema: everything text; let rules work
                cols = [{"name": c, "role": "text"} for c in df0.columns]
                proposed = _ProposedSchemaShim(cols, conf=0.0)
                print(f"[loop] • {slug}: no bootstrap schema found; starting with text-only roles.")
            except Exception as e:
                print(f"[loop] ✗ {slug}: unable to build starter schema: {e}")
                if args.debug: traceback.print_exc()
                continue

        success = False
        zeros_in_a_row = 0
        for k in range(1, max_iter + 1):
            iter_dir = out_slug_root / f"iter_{k}"
            ok, report, clean_df = _run_clean_once(cfg, slug, bronze_root, iter_dir, proposed, save_head=args.save_head, debug=args.debug)
            if not ok or report is None:
                break

            sc_after, rc_after = _confidence_pair(report)

            used_surrogate = False
            if sc_after == 0.0 and rc_after == 0.0 and args.surrogate_conf:
                sc_after, rc_after = _surrogate_conf(clean_df, proposed, cfg, debug=args.debug)
                used_surrogate = True
                print(f"[loop:DEBUG] {slug} iter {k}: using surrogate confidences schema={sc_after:.4f}, role={rc_after:.4f}")
            print(f"[loop] {slug} iter {k}: schema_conf_after={sc_after:.4f}, avg_role_conf_after={rc_after:.4f}")

            imp = _improvement(prev_pair, (sc_after, rc_after))
            if prev_pair is not None:
                print(f"[loop] {slug} iter {k}: improvement={0.0 if imp==float('inf') else imp:.4f}")

            # success
            if sc_after >= min_schema_conf and rc_after >= min_role_conf:
                success = True
                final_dir = out_slug_root
                for name in ("clean.parquet", "clean_report.json"):
                    shutil.copy2(iter_dir / name, final_dir / name)
                if args.save_head and (iter_dir / f"head{args.save_head}.csv").exists():
                    shutil.copy2(iter_dir / f"head{args.save_head}.csv", final_dir / f"head{args.save_head}.csv")
                print(f"[loop] ✓ {slug}: thresholds met (≥{min_schema_conf} schema, ≥{min_role_conf} role). Final written to {final_dir}")
                break

            # early-stop if not improving enough — unless we have no confidences and the user wants to run to max_iter
            if prev_pair is not None and imp < min_impr:
                if (used_surrogate or (sc_after == 0.0 and rc_after == 0.0)) and args.force_max_iters_when_no_conf:
                    zeros_in_a_row += 1
                    print(f"[loop] • {slug}: confidences not improving; continuing to max_iter due to --force-max-iters-when-no-conf (streak={zeros_in_a_row})")
                else:
                    print(f"[loop] • {slug}: improvement {imp:.4f} < min_improvement {min_impr:.4f} → stopping.")
                    final_dir = out_slug_root
                    for name in ("clean.parquet", "clean_report.json"):
                        shutil.copy2(iter_dir / name, final_dir / name)
                    if args.save_head and (iter_dir / f"head{args.save_head}.csv").exists():
                        shutil.copy2(iter_dir / f"head{args.save_head}.csv", final_dir / f"head{args.save_head}.csv")
                    break

            # re-infer schema for next iteration from the cleaned df (keeps roles current)
            try:
                # very light-weight: keep existing roles unless clearly numeric/time
                new_cols = []
                for c in clean_df.columns:
                    s = clean_df[c]
                    if pd.api.types.is_integer_dtype(s) or pd.api.types.is_float_dtype(s):
                        role = "numeric"
                    elif pd.api.types.is_datetime64_any_dtype(s):
                        role = "time"
                    else:
                        role = next((r["role"] for r in proposed.to_dict()["columns"] if r["name"] == c), "text")
                    new_cols.append({"name": c, "role": role})
                proposed = _ProposedSchemaShim(new_cols, conf=(sc_after or 0.0))
            except Exception as e:
                print(f"[loop] ✗ {slug}: failed to re-infer schema for next iter: {e}")
                if args.debug: traceback.print_exc()
                break

            prev_pair = (sc_after, rc_after)

        if success:
            completed += 1
        else:
            print(f"[loop:WARN] {slug}: thresholds not met; best outputs are in {out_slug_root}")

    print(f"[pipeline] Completed {completed}/{len(items)} datasets.")

if __name__ == "__main__":
    main()
