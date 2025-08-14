from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd

# ---------- Column stats (pure) ----------

def column_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute basic per-column stats used by pruning rules.
    Returns a DataFrame indexed by column name with:
      - dtype: pandas dtype name (string)
      - rows: total row count
      - non_null: count of non-null
      - missing_pct: fraction of nulls in [0,1]
      - nunique: number of distinct non-null values
      - unique_ratio: nunique / rows  (0 if rows==0)
    Pure: does not mutate input.
    """
    rows = len(df)
    stats: List[Dict[str, Any]] = []
    for name, s in df.items():
        non_null = int(s.notna().sum())
        missing_pct = float(1.0 - (non_null / rows)) if rows > 0 else 1.0
        # robust nunique even for unhashables
        try:
            nunique = int(s.nunique(dropna=True))
        except TypeError:
            nunique = int(s.astype(str).nunique(dropna=True))
        unique_ratio = float(nunique / rows) if rows > 0 else 0.0
        stats.append(
            {
                "name": name,
                "dtype": str(s.dtype),
                "rows": rows,
                "non_null": non_null,
                "missing_pct": missing_pct,
                "nunique": nunique,
                "unique_ratio": unique_ratio,
            }
        )
    out = pd.DataFrame(stats)
    if out.empty:
        # stable columns index even for empty frames
        return pd.DataFrame(
            columns=["dtype", "rows", "non_null", "missing_pct", "nunique", "unique_ratio"]
        )
    return out.set_index("name").sort_index()

# ---------- Decisions (pure) ----------

def decide_sparse_columns(
    stats: pd.DataFrame,
    *,
    drop_missing_pct: float,
    always_keep: Optional[List[str]] = None,
) -> List[str]:
    """
    Returns a list of column names whose missing_pct >= threshold and NOT in always_keep.
    """
    always_keep = set(always_keep or [])
    if stats.empty:
        return []
    mask = (stats["missing_pct"] >= float(drop_missing_pct))
    names = [n for n in stats.index[mask] if n not in always_keep]
    return names

def decide_constant_like_columns(
    stats: pd.DataFrame,
    *,
    min_unique_ratio: float,
    always_keep: Optional[List[str]] = None,
) -> List[str]:
    """
    Returns names where unique_ratio <= min_unique_ratio (effectively constant),
    excluding always_keep.
    """
    always_keep = set(always_keep or [])
    if stats.empty:
        return []
    mask = (stats["unique_ratio"] <= float(min_unique_ratio))
    names = [n for n in stats.index[mask] if n not in always_keep]
    return names

# ---------- Application (pure) ----------

def apply_prune(
    df: pd.DataFrame,
    *,
    drop_missing_pct: float = 0.90,
    min_unique_ratio: float = 0.001,
    always_keep: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]], pd.DataFrame]:
    """
    Decide and drop sparse / constant-like columns. Returns:
      - pruned DataFrame (copy, no mutation of input),
      - report: list of {name, action='drop', reason, missing_pct, unique_ratio, nunique, dtype},
      - stats DataFrame (from column_stats).

    Notes:
    - A column can satisfy both reasons; we keep the *first* matched reason with
      precedence: 'sparse' then 'constant'. If you want both, you can easily
      modify report logic.
    """
    always_keep = list(always_keep or [])
    st = column_stats(df)

    drop_sparse = decide_sparse_columns(st, drop_missing_pct=drop_missing_pct, always_keep=always_keep)
    drop_const  = decide_constant_like_columns(st, min_unique_ratio=min_unique_ratio, always_keep=always_keep)

    to_drop_set = set(drop_sparse) | set(drop_const)
    # Compose report with a single reason (sparse preferred if both)
    report: List[Dict[str, Any]] = []
    for name in sorted(to_drop_set):
        reason = "sparse" if name in drop_sparse else "constant"
        row = st.loc[name]
        report.append(
            {
                "name": name,
                "action": "drop",
                "reason": reason,
                "missing_pct": float(row["missing_pct"]),
                "unique_ratio": float(row["unique_ratio"]),
                "nunique": int(row["nunique"]),
                "dtype": str(row["dtype"]),
            }
        )

    # Pure drop (copy)
    pruned = df.drop(columns=list(to_drop_set), errors="ignore").copy(deep=True)
    return pruned, report, st

# ---------- Quarantine (pure) ----------

def quarantine_unknown_columns(
    df: pd.DataFrame,
    *,
    allowed: List[str],
    prefix: str = "__quarantine__",
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Rename columns not in `allowed` by prefixing them, effectively quarantining
    them from schema-driven pipelines. Returns (df_copy, quarantined_names).

    Pure & idempotent (running twice with same args makes no additional changes).
    """
    allowed_set = set(allowed or [])
    extras = [c for c in df.columns if c not in allowed_set and not c.startswith(prefix)]
    if not extras:
        return df.copy(deep=True), extras
    rename_map = {c: f"{prefix}{c}" for c in extras}
    out = df.rename(columns=rename_map).copy(deep=True)
    return out, extras
