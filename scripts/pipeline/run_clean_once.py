from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd

# your internal modules
from src.config_model.model import RootCfg
from src.cleaning.engine import run_clean_pass
from src.cleaning.rescore import rescore_after_clean

def _load_df(bronze_dir: Path) -> pd.DataFrame:
    parq = bronze_dir / "normalize.parquet"
    if not parq.exists():
        raise FileNotFoundError(
            f"normalize.parquet not found at {parq}\n"
            "Run your normalize step first (or point --slug to one that has it)."
        )
    return pd.read_parquet(parq)

def _load_proposed_schema(bootstrap_json: Path, df: pd.DataFrame):
    """
    Build a lightweight 'schema-like' shim that satisfies the cleaning/rescore calls.
    It needs .to_dict() with {'columns':[{'name','role','confidence'?}], 'schema_confidence'?}
    """
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
        # try common shapes
        src_cols = obj.get("columns") or obj.get("schema", {}).get("columns") or []
        for c in src_cols:
            name = c.get("name") or c.get("column") or c.get("col")
            if not name:
                continue
            role = c.get("role", "text")
            conf = c.get("confidence", None)
            cols_out.append({"name": str(name), "role": str(role), "confidence": conf})
        schema_conf = float(obj.get("schema_confidence", 0.0) or 0.0)

    # fallback if bootstrap was missing or empty
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
    """
    Returns (schema_conf_after, avg_role_conf_after) computed by running the same
    rescore inference on 'df'. This is our comparable 'NLP score' for that frame.
    """
    r = rescore_after_clean(
        df=df,
        prev_schema=proposed,
        profiling_cfg=cfg.profiling,   # carries roles sub-config
        nlp_cfg=cfg.nlp
    )
    return float(r.schema_conf_after), float(r.avg_role_conf_after)

def main():
    ap = argparse.ArgumentParser(description="Run ONE cleaning pass for a single slug and show before/after.")
    ap.add_argument("--config", default="config/config.toml")
    ap.add_argument("--slug", default="f1", help="Dataset slug under data/bronze/<slug>/normalize.parquet")
    ap.add_argument("--head", type=int, default=5)
    args = ap.parse_args()

    project_root = Path.cwd()
    cfg = RootCfg.load(args.config)

    bronze_dir = project_root / "data" / "bronze" / args.slug
    bootstrap_json = bronze_dir / "nlp_bootstrap.json"

    print(f"[load] slug={args.slug}")
    df_before = _load_df(bronze_dir)
    proposed = _load_proposed_schema(bootstrap_json, df_before)

    def _missing_pct(df, cols):
        out = {}
        for c in cols:
            if c in df.columns:
                out[c] = float(df[c].isna().mean())
        return out

    watch = ["num_03", "dt_01", "dt_02", "dt_03"]

    # ---- BEFORE snapshots ----
    print("[MISS% BEFORE]", _missing_pct(df_before, watch))
    _print_head_and_dtypes("BEFORE", df_before, args.head)

    # BEFORE NLP score (computed on uncleaned df)
    schema_before, avg_role_before = _nlp_score_for_df(df_before, proposed, cfg)
    print(f"\n[NLP] BEFORE  schema_conf_after={schema_before:.4f}  avg_role_conf_after={avg_role_before:.4f}")

    # ---- CLEAN once ----
    result = run_clean_pass(df_before, proposed, cfg)
    df_after = result.clean_df

    # ---- AFTER snapshots ----
    print("[MISS% AFTER ]", _missing_pct(df_after, [c for c in watch if c in df_after.columns]))

    # Optional: show deltas for watched columns
    bmiss = _missing_pct(df_before, watch)
    amiss = _missing_pct(df_after,  watch)
    deltas = {c: (bmiss.get(c), amiss.get(c), (None if bmiss.get(c) is None or amiss.get(c) is None else amiss[c]-bmiss[c]))
              for c in set(bmiss) | set(amiss)}
    print("[MISS% Δ     ]", deltas)

    _print_head_and_dtypes("AFTER", df_after, args.head)

    # AFTER NLP score (computed on cleaned df)
    schema_after, avg_role_after = _nlp_score_for_df(df_after, proposed, cfg)
    print(f"\n[NLP] AFTER   schema_conf_after={schema_after:.4f}  avg_role_conf_after={avg_role_after:.4f}")

    # Simple summary + which rules actually fired
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
