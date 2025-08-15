#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math
from pathlib import Path
from copy import deepcopy
import pandas as pd

# project imports
from src.config_model.model import RootCfg
from src.cleaning.engine import run_clean_pass
from src.cleaning.rescore import rescore_after_clean
from src.nlp.roles import _dtype_str, guess_role
from src.nlp.schema import ProposedSchema, ColumnSchema, ColumnHints, RoleConfidence
from src.nlp.schema_io import to_toml, schema_path_for_slug


# ------------------------ IO helpers ------------------------

def _bronze_dir(root: Path, slug: str) -> Path:
    return root / "data" / "bronze" / slug

def _load_df(bronze_dir: Path) -> pd.DataFrame:
    parq = bronze_dir / "normalize.parquet"
    if not parq.exists():
        raise FileNotFoundError(
            f"normalize.parquet not found at {parq}\n"
            "Run your normalize step first (or point --slug to one that has it)."
        )
    return pd.read_parquet(parq)

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
    out_path_fsys = Path(out_path)  # convert PurePosixPath → FS path
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
            hints=ColumnHints(),  # can thread unit/domain later if you want
        ))
        confs.append(float(c.get("role_confidence", 0.0)))
    avg_conf = float(sum(confs) / len(confs)) if confs else 0.0
    return ProposedSchema(dataset_slug=slug, columns=rc, schema_confidence=avg_conf)

def _bootstrap_overlay_roles(old_bootstrap: dict, new_cols: list[dict], slug: str, schema_conf: float) -> dict:
    by_name = {c["name"]: c for c in new_cols}
    cols_out = []
    seen = set()
    # update if present
    for c in old_bootstrap.get("columns", []):
        name = c.get("name")
        if name in by_name:
            updated = deepcopy(c)
            updated["role"] = by_name[name]["role"]
            updated["role_confidence"] = by_name[name]["role_confidence"]
            updated["dtype"] = by_name[name]["dtype"]
            cols_out.append(updated)
            seen.add(name)
        else:
            cols_out.append(c)
    # add any new columns not in bootstrap
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
    # share of missing cells across the frame
    try:
        return float(df.isna().sum().sum()) / float(df.shape[0] * df.shape[1])
    except Exception:
        return 0.0

def _score(rescore_dict: dict, df_after: pd.DataFrame) -> float:
    """
    Single scalar to compare iterations. Higher is better.
    You can tune weights freely; this mixes schema confidence and
    (negatively) the overall missing rate.
    """
    sc = float(rescore_dict.get("schema_conf_after", 0.0) or 0.0)
    avg = float(rescore_dict.get("avg_role_conf_after", 0.0) or 0.0)
    miss_penalty = _frame_missing_rate(df_after)
    return sc * 0.7 + avg * 0.3 - 0.25 * miss_penalty


# ------------------------ main loop ------------------------

def main():
    ap = argparse.ArgumentParser(description="Iteratively clean *one* slug and live-print improvements.")
    ap.add_argument("--config", default="config/config.toml")
    ap.add_argument("--slug", default="f1", help="Dataset slug under data/bronze/<slug>/normalize.parquet")
    ap.add_argument("--head", type=int, default=5)
    ap.add_argument("--max-iters", type=int, default=6)
    ap.add_argument("--min-improve", type=float, default=1e-4, help="Min score delta to keep going")
    ap.add_argument("--patience", type=int, default=2, help="Early stop after N non-improving rounds")
    ap.add_argument("--write-back", action="store_true", help="Persist updated TOML/JSON each improving round")
    args = ap.parse_args()

    project_root = Path.cwd()
    cfg = RootCfg.load(args.config)
    bronze = _bronze_dir(project_root, args.slug)
    bronze.mkdir(parents=True, exist_ok=True)

    df_raw = _load_df(bronze)
    bootstrap_json = _bootstrap_path(bronze)
    bootstrap = _load_bootstrap(bootstrap_json)

    # initial proposed from whatever we have now
    # (use bootstrap if present, else infer from raw)
    if bootstrap.get("columns"):
        cols_init = [{"name": c["name"], "dtype": c.get("dtype") or _dtype_str(df_raw[c["name"]]) if c["name"] in df_raw else "string",
                      "role": c.get("role", "text"),
                      "role_confidence": float(c.get("role_confidence", 0.0))}
                     for c in bootstrap["columns"] if c.get("name") in df_raw.columns]
    else:
        cols_init = _infer_roles_for_df(df_raw, cfg)

    proposed = _proposed_from_cols(args.slug, cols_init)

    # --- baselines
    print(f"[load] slug={args.slug}")
    miss_before = _frame_missing_rate(df_raw)

    # Re-score the *uncleaned* frame with current proposal for a comparable baseline
    rescore0 = rescore_after_clean(df_raw, proposed, cfg.profiling, cfg.nlp)
    score0 = _score(rescore0.__dict__ if hasattr(rescore0, "__dict__") else dict(rescore0), df_raw)

    print(f"[BASE] schema_conf={rescore0.schema_conf_after:.4f}  avg_role_conf={rescore0.avg_role_conf_after:.4f}  miss={miss_before:.4f}  score={score0:.5f}")
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
        "schema_conf": rescore0.schema_conf_after,
        "avg_role_conf": rescore0.avg_role_conf_after,
        "miss_rate": miss_before,
    }

    no_improve_rounds = 0

    for it in range(1, args.max_iters + 1):
        print("\n" + "=" * 80)
        print(f"ITER {it}")

        # always clean starting from raw with the *current* proposed schema/policy
        result = run_clean_pass(df_raw, proposed, cfg)
        df_after = result.clean_df

        # rescore (we already have it in result.rescore, but also safe to recompute)
        rescore_dict = result.rescore if isinstance(result.rescore, dict) else {}
        if not rescore_dict:
            r = rescore_after_clean(df_after, proposed, cfg.profiling, cfg.nlp)
            rescore_dict = r.__dict__ if hasattr(r, "__dict__") else dict(r)

        miss_after = _frame_missing_rate(df_after)
        score = _score(rescore_dict, df_after)

        # print quick deltas
        print(f"[SCORES] schema_conf={rescore_dict.get('schema_conf_after',0):.4f}  "
              f"avg_role_conf={rescore_dict.get('avg_role_conf_after',0):.4f}  "
              f"miss={miss_after:.4f}  score={score:.5f}  Δscore={score-best['score']:.5f}")

        # dtype & missing summaries
        print("\n=== AFTER: HEAD({}) ===".format(args.head))
        try:
            print(df_after.head(args.head).to_string(index=False))
        except Exception:
            print(df_after.head(args.head))
        print("\n=== AFTER: DTYPES ===")
        for c, t in df_after.dtypes.items():
            print(f"  {c}: {t}")

        # which rules fired
        actions = result.report.get("rules", {}).get("per_column_actions", {})
        fired = {c: acts for c, acts in actions.items() if acts}
        print("\n[Rules fired]")
        if not fired:
            print("  (none)")
        else:
            for c, acts in fired.items():
                print(f"  {c}: {', '.join(acts)}")

        # persist an iteration record
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
            })

            # --- Build new roles from the NEW cleaned df and write back artifacts
            new_cols = _infer_roles_for_df(df_after, cfg)
            new_proposed = _proposed_from_cols(args.slug, new_cols)

            if args.write_back:
                # 1) update bootstrap (overlay roles/confidence)
                updated_bootstrap = _bootstrap_overlay_roles(
                    _load_bootstrap(bootstrap_json),
                    new_cols,
                    slug=args.slug,
                    schema_conf=new_proposed.schema_confidence,
                )
                _save_bootstrap(bootstrap_json, updated_bootstrap)

                # 2) update schema TOML
                out_toml = _save_schema_toml(project_root, args.slug, new_proposed)
                print(f"[WRITE] nlp_bootstrap.json updated; schema TOML -> {out_toml}")

            # 3) advance the in-memory proposal so next iter uses the new roles
            proposed = new_proposed

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
    if args.write_back:
        print(f"Bootstrap: {_bootstrap_path(bronze)}")
        print(f"Schema TOML: {schema_path_for_slug(project_root, args.slug)}")


if __name__ == "__main__":
    main()
