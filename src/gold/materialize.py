from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Any
import json
import pandas as pd
import numpy as np
import re
from src.config_model.model import RootCfg
from src.io.storage import build_storage_from_config
from src.io.writers import write_any
from src.nlp.schema import ProposedSchema, ColumnSchema, ColumnHints, RoleConfidence
from src.cleaning.recheck import recheck_silver, _load_frozen_schema
from src.topics.candidates import build_candidates
from src.topics.scoring import score_topics
from src.topics.select import select_topics
from src.layout.planner import make_layout_plan
from src.utils.ids import make_chart_id


# ------------------------------- small helpers -------------------------------

def _effect(row: pd.Series) -> Dict[str, Any]:
    ed = row.get("effect_detail")
    if isinstance(ed, str):
        try:
            ed = json.loads(ed)
        except Exception:
            ed = {}
    return ed or {}

def _coerce_list(val: Any) -> List[str]:
    """Accept list | json-encoded list | scalar | None -> list[str]."""
    if val is None:
        return []
    if isinstance(val, list):
        return [str(x) for x in val]
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return []
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except Exception:
            return [s]
        return []
    return [str(val)]


def _charts_to_try(row: pd.Series) -> List[str]:
    pcs = row.get("proposed_charts")
    return _coerce_list(pcs)


def _max_charts_per_topic(cfg: RootCfg) -> int:
    # prefer config if present; otherwise default to 2
    try:
        return int(getattr(cfg.charts, "max_per_topic", 2))
    except Exception:
        return 2


def _derive_schema_from_df(df: pd.DataFrame, slug: str) -> ProposedSchema:
    """Minimal schema from dtypes with better id detection."""
    cols: List[ColumnSchema] = []
    id_pat = re.compile(r"(?:^id$|_id$|^id_|id\b|uuid|guid|key|hash|checksum|^row_?id$|^entity_?id$)", re.I)
    for name in df.columns:
        s = df[name]
        dtype_str = str(s.dtype)
        n = str(name)

        role, conf = "categorical", 0.55
        if pd.api.types.is_datetime64_any_dtype(s):
            role, conf = "time", 0.8
        elif pd.api.types.is_bool_dtype(s):
            role, conf = "boolean", 0.6
        elif id_pat.search(n):
            role, conf = "id", 0.9
        elif pd.api.types.is_numeric_dtype(s):
            # very high uniqueness numeric ≈ id
            try:
                uniq_ratio = float(pd.Series(s).nunique(dropna=True)) / max(1, len(s))
            except Exception:
                uniq_ratio = 1.0
            role, conf = (("id", 0.8) if uniq_ratio > 0.98 else ("numeric", 0.7))
        else:
            try:
                avg_len = float(s.astype("string").str.len().mean())
            except Exception:
                avg_len = 0.0
            role, conf = ("text", 0.6) if avg_len >= 16 else ("categorical", 0.55)

        cols.append(ColumnSchema(
            name=n,
            dtype=dtype_str,
            role_confidence=RoleConfidence(role=role, confidence=conf),
            hints=ColumnHints(),
        ))
    return ProposedSchema(dataset_slug=slug, columns=cols, schema_confidence=0.65)


def _first_time_col(df: pd.DataFrame, schema: ProposedSchema | None) -> Optional[str]:
    try:
        for c in (schema.columns if schema else []):
            rc = getattr(c, "role_confidence", None)
            if rc and getattr(rc, "role", "") == "time" and c.name in df.columns:
                return c.name
    except Exception:
        pass
    for c in df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns:
        return str(c)
    for guess in ["date", "timestamp", "time", "datetime"]:
        if guess in df.columns:
            return guess
    return None


def _coverage_pct(df: pd.DataFrame, cols: List[str]) -> float:
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return 0.0
    return float((~df[cols].isna()).all(axis=1).mean())


_ID_NAME_PAT = re.compile(
    r"(?:^id$|_id$|^id_|id\b|uuid|guid|key|hash|checksum|^row_?id$|^entity_?id$|_index$)",
    re.I,
)
_NOISE_TEXT_PAT = re.compile(
    r"(?:https?://|[A-Za-z]:\\|^/|\\|\.json$|\.csv$|\.parquet$|\.xml$|@|uuid|guid|^_source_uri$|uri|path|file|filename)",
    re.I,
)


def _is_id_like(df: pd.DataFrame, col: str) -> bool:
    n = str(col)
    if _ID_NAME_PAT.search(n):
        return True
    s = df[col]
    try:
        if pd.api.types.is_numeric_dtype(s):
            ur = float(pd.Series(s).nunique(dropna=True)) / max(1, len(s))
            if ur > 0.98:
                return True
    except Exception:
        pass
    return False

def _is_noise_categorical(df: pd.DataFrame, col: str) -> bool:
    n = str(col)
    if _NOISE_TEXT_PAT.search(n):
        return True
    s = df[col].astype("string")
    try:
        ur = float(s.nunique(dropna=True)) / max(1, len(s))
        if ur > 0.8:  # almost all unique -> labels are IDs
            return True
    except Exception:
        pass
    return False

def _num_coverage(df: pd.DataFrame, col: str) -> float:
    try:
        return float(df[col].notna().mean())
    except Exception:
        return 0.0

def _num_signal(df: pd.DataFrame, col: str) -> float:
    """Coverage × std as a simple usefulness score."""
    try:
        s = pd.to_numeric(df[col], errors="coerce")
        cov = float(s.notna().mean())
        sig = float(s.std(skipna=True) or 0.0)
        return cov * sig
    except Exception:
        return 0.0


def _fallback_candidates_from_df(
    df: pd.DataFrame,
    schema: ProposedSchema,
    slug: str,
    max_total: int = 12,
    *,
    cov_min: float = 0.3,
    cat_max_card: int = 50,
) -> pd.DataFrame:
    """Create a sensible, small set of topics from the data itself."""
    # thresholds informed by config when present
    try:
        cov_min = max(cov_min, float(getattr(schema, "schema_confidence", 0.0) or 0.0) * 0.2)
    except Exception:
        pass

    time_col = _first_time_col(df, schema)

    # --- candidate numeric columns
    num_cols = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c]) and not _is_id_like(df, c) and _num_coverage(df, c) >= cov_min
    ]
    # rank by a simple "usefulness" score
    num_cols = sorted(num_cols, key=lambda c: _num_signal(df, c), reverse=True)

    # --- candidate categorical columns (moderate cardinality, not noise)
    cat_candidates = []
    for c in df.columns:
        if c in num_cols:
            continue
        s = df[c]
        if pd.api.types.is_categorical_dtype(s) or pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
            if _is_noise_categorical(df, c) or _is_id_like(df, c):
                continue
            try:
                card = int(s.astype("string").nunique(dropna=True))
            except Exception:
                continue
            if 3 <= card <= cat_max_card:
                cat_candidates.append((c, card))
    # prefer smaller card first (clearer charts)
    cat_candidates.sort(key=lambda x: x[1])
    cat_cols = [c for c, _ in cat_candidates]

    rows: List[Dict[str, Any]] = []
    tid = 1
    def _tid(prefix: str) -> str:
        nonlocal tid
        t = f"{prefix}_{tid:03d}"
        tid += 1
        return t

    # KPI: top 3 numerics
    for m in num_cols[:3]:
        rows.append(dict(
            topic_id=_tid("kpi"),
            dataset_slug=slug,
            family="kpi",
            primary_fields=json.dumps([m]),
            secondary_fields=json.dumps([]),
            time_field=None,
            coverage_pct=_coverage_pct(df, [m]),
            effect_size=0.0,
            significance=json.dumps({}),
            proposed_charts=json.dumps(["kpi"]),
            n_obs=int(len(df)),
            score_total=1.0,
        ))

    # Trend: top 3 metrics with time present and decent joint coverage
    if time_col:
        for m in num_cols[:6]:
            joint_cov = _coverage_pct(df, [time_col, m])
            if joint_cov >= cov_min:
                rows.append(dict(
                    topic_id=_tid("trend"),
                    dataset_slug=slug,
                    family="trend",
                    primary_fields=json.dumps([m]),
                    secondary_fields=json.dumps([]),
                    time_field=time_col,
                    coverage_pct=joint_cov,
                    effect_size=0.0,
                    significance=json.dumps({}),
                    proposed_charts=json.dumps(["line", "column"]),
                    n_obs=int(len(df)),
                    score_total=0.95,
                ))
                if sum(1 for r in rows if r["family"] == "trend") >= 3:
                    break

    # Ranking: up to 2 {cat,num}
    for cat in cat_cols[:3]:
        for m in num_cols[:5]:
            joint_cov = _coverage_pct(df, [cat, m])
            if joint_cov < cov_min:
                continue
            rows.append(dict(
                topic_id=_tid("ranking"),
                dataset_slug=slug,
                family="ranking",
                primary_fields=json.dumps([cat, m]),
                secondary_fields=json.dumps([]),
                time_field=None,
                coverage_pct=joint_cov,
                effect_size=0.0,
                significance=json.dumps({}),
                proposed_charts=json.dumps(["ordered_bar", "ordered_column"]),
                n_obs=int(len(df)),
                score_total=0.9,
            ))
            break  # one num per cat

    # Distribution: first good numeric
    for m in num_cols:
        if _coverage_pct(df, [m]) >= cov_min:
            rows.append(dict(
                topic_id=_tid("distribution"),
                dataset_slug=slug,
                family="distribution",
                primary_fields=json.dumps([m]),
                secondary_fields=json.dumps([]),
                time_field=None,
                coverage_pct=_coverage_pct(df, [m]),
                effect_size=0.0,
                significance=json.dumps({}),
                proposed_charts=json.dumps(["histogram", "boxplot"]),
                n_obs=int(len(df)),
                score_total=0.88,
            ))
            break

    # Correlation: pick a genuinely correlated pair if possible
    if len(num_cols) >= 2:
        # quick sample for speed
        n = len(df)
        sample_n = min(n, 50000)
        rng = getattr(getattr(schema, "role_confidence", None), "seed", None)
        samp = df[num_cols].sample(n=sample_n, random_state=getattr(getattr(schema, "env", None), "seed", 42)) if n > sample_n else df[num_cols]
        corr = samp.corr(numeric_only=True).abs().replace(1.0, 0.0)
        best = (None, None, 0.0)
        for a in num_cols[:10]:
            for b in num_cols[:10]:
                if a == b or a not in corr.index or b not in corr.columns:
                    continue
                val = float(corr.loc[a, b])
                if val > best[2]:
                    best = (a, b, val)
        a, b, r = best
        if a and b:
            rows.append(dict(
                topic_id=_tid("correlation"),
                dataset_slug=slug,
                family="correlation",
                primary_fields=json.dumps([a, b]),
                secondary_fields=json.dumps([]),
                time_field=None,
                coverage_pct=_coverage_pct(df, [a, b]),
                effect_size=float(r or 0.0),
                significance=json.dumps({}),
                proposed_charts=json.dumps(["scatter", "xy_heatmap"]),
                n_obs=int(len(df)),
                score_total=0.87 + min(0.1, float(r or 0.0)),  # small bump for stronger corr
            ))

    fb = pd.DataFrame(rows)
    if fb.empty:
        return fb
    return fb.sort_values("score_total", ascending=False).head(max_total).reset_index(drop=True)

# ---------------------- per-component materialization ----------------------

def _materialize_component(df: pd.DataFrame, row: pd.Series) -> pd.DataFrame:
    fam = row["family"]
    pf = _coerce_list(row.get("primary_fields"))
    sf = _coerce_list(row.get("secondary_fields"))
    time = row.get("time_field") or None
    ed = _effect(row)

    # ---- derived categories from time (used for count rankings)
    if ed.get("derived_category") in ("day_of_week","hour_of_day") and time and time in df.columns:
        t = pd.to_datetime(df[time], errors="coerce")
        if ed["derived_category"] == "day_of_week":
            df = df.copy()
            df["day_of_week"] = t.dt.day_name()
            if pf and pf[0] in ("day_of_week","hour_of_day"):
                pf[0] = "day_of_week"
        else:
            df = df.copy()
            df["hour_of_day"] = t.dt.hour.astype("Int64").astype("string")
            if pf and pf[0] in ("day_of_week","hour_of_day"):
                pf[0] = "hour_of_day"

    # ---- numeric transforms (log1p / rate)
    # We always operate on a copy only if needed.
    if ed.get("transform") == "log1p" and pf:
        base = pf[0]
        if base in df.columns:
            df = df.copy()
            df[base] = np.log1p(pd.to_numeric(df[base], errors="coerce"))

    if ed.get("transform") == "rate" and {"num","den"}.issubset(ed.keys()):
        num, den = ed["num"], ed["den"]
        if num in df.columns and den in df.columns:
            df = df.copy()
            num_s = pd.to_numeric(df[num], errors="coerce")
            den_s = pd.to_numeric(df[den], errors="coerce").replace(0, np.nan)
            new_name = ed.get("name", f"rate_{num}_by_{den}")
            df[new_name] = (num_s / den_s).clip(lower=0)
            pf = [new_name]

    # --------------- existing families ---------------
    # KPI
    if fam == "kpi" and pf:
        m = pf[0]
        vals = pd.to_numeric(df[m], errors="coerce") if m in df.columns else pd.Series(dtype=float)
        return pd.DataFrame({"metric_name": [m], "value": [float(vals.mean(skipna=True) or 0.0)]})

    # Trend (time + metric)
    if fam == "trend" and time and pf:
        m = pf[0]
        if time in df.columns and m in df.columns:
            t = pd.to_datetime(df[time], errors="coerce")
            g = df[[time, m]].copy()
            g[time] = pd.to_datetime(t.dt.to_period("D").dt.to_timestamp())
            d = g.groupby(time, dropna=True, sort=True)[m].mean().reset_index()
            d.columns = [time, "value"]
            return d


    # ---- Correlation
    if fam == "correlation" and len(pf) >= 2:
        a, b = pf[0], pf[1]
        if a in df.columns and b in df.columns:
            # Try num-num shape first
            aa = pd.to_numeric(df[a], errors="coerce")
            bb = pd.to_numeric(df[b], errors="coerce")
            out = pd.DataFrame({a: aa, b: bb}).dropna()
            if len(out) >= 3:
                return out

            # Fallback: num-cat aggregate if either is non-numeric
            a_is_num = pd.api.types.is_numeric_dtype(df[a]) or pd.api.types.is_float_dtype(aa.dtype)
            b_is_num = pd.api.types.is_numeric_dtype(df[b]) or pd.api.types.is_float_dtype(bb.dtype)
            if a_is_num and not b_is_num:
                m, c = a, b
            elif b_is_num and not a_is_num:
                m, c = b, a
            else:
                m, c = (a, b)
            grp = (
                pd.concat([df[[c]].astype(str), pd.to_numeric(df[m], errors="coerce")], axis=1)
                  .dropna()
            )
            if not grp.empty:
                d = grp.groupby(c)[m].agg(["mean", "count"]).reset_index()
                return d

    # ---- Ranking: allow count metric (no numeric input needed)
    if fam == "ranking" and pf:
        if len(pf) == 1 or pf[1] == "__row_count__" or ed.get("metric") == "count":
            cat = pf[0]
            if cat in df.columns:
                d = (
                    df[[cat]].dropna().astype(str)
                      .assign(__ones__=1.0)
                      .groupby(cat)["__ones__"].sum()
                      .reset_index()
                      .rename(columns={"__ones__": "count"})
                      .sort_values("count", ascending=False)
                )
                return d
        elif len(pf) >= 2:
            cat, m = pf[0], pf[1]
            if cat in df.columns and m in df.columns:
                d = (
                    pd.concat([df[[cat]].astype(str), pd.to_numeric(df[m], errors="coerce")], axis=1)
                      .dropna()
                      .groupby(cat)[m].sum().reset_index()
                      .sort_values(m, ascending=False)
                )
                return d

    # ---- Part-to-whole: allow count metric
    if fam == "part_to_whole" and pf:
        if len(pf) == 1 or (len(pf) >= 2 and pf[1] == "__row_count__") or ed.get("metric") == "count":
            cat = pf[0]
            if cat in df.columns:
                g = (
                    df[[cat]].dropna().astype(str)
                      .assign(__ones__=1.0)
                      .groupby(cat)["__ones__"].sum()
                      .reset_index()
                      .rename(columns={"__ones__": "count"})
                )
                total = float(g["count"].sum() or 1.0)
                g["pct_total"] = (g["count"] / total).astype(float)
                return g.sort_values("pct_total", ascending=False)
        elif len(pf) >= 2:
            cat, m = pf[0], pf[1]
            if cat in df.columns and m in df.columns:
                g = (
                    pd.concat([df[[cat]].astype(str), pd.to_numeric(df[m], errors="coerce")], axis=1)
                      .dropna()
                      .groupby(cat)[m].sum().reset_index()
                )
                total = float(g[m].sum() or 1.0)
                g["pct_total"] = (g[m] / total).astype(float)
                return g.sort_values("pct_total", ascending=False)

    # ---- Distribution: just a single numeric series
    if fam == "distribution" and pf:
        m = pf[0]
        if m in df.columns:
            return pd.DataFrame({m: pd.to_numeric(df[m], errors="coerce")}).dropna()

    # ---- Deviation: z-scores per-index
    if fam == "deviation" and pf:
        m = pf[0]
        if m in df.columns:
            s = pd.to_numeric(df[m], errors="coerce")
            mu, sd = float(s.mean(skipna=True) or 0.0), float(s.std(skipna=True) or 0.0)
            if sd <= 0:
                z = pd.Series(np.zeros_like(s), index=s.index, dtype=float)
            else:
                z = (s - mu) / sd
            return pd.DataFrame({"index": s.index, "value": s.values, "z": z.values}).dropna()

    # ---- Cohort: month cohorts & age using Period arithmetic (robust)
    if fam == "cohort" and time and len(pf) >= 1:
        id_col = pf[0]
        if time in df.columns and id_col in df.columns:
            t = pd.to_datetime(df[time], errors="coerce")
            d0 = df[[id_col]].copy()
            d0["_t"] = pd.to_datetime(t.dt.to_period("M").dt.to_timestamp())
            d0["_first_seen"] = d0.groupby(id_col)["_t"].transform("min")
            p_t = d0["_t"].dt.to_period("M")
            p0 = d0["_first_seen"].dt.to_period("M")
            d0["_age"] = (p_t - p0).astype("Int64")
            tab = (
                d0.groupby(["_first_seen", "_age"], dropna=True)[id_col]
                   .nunique()
                   .reset_index(name="retained")
            )
            sizes = (
                tab[tab["_age"] == 0][["_first_seen", "retained"]]
                   .rename(columns={"retained": "cohort_size"})
            )
            out = tab.merge(sizes, on="_first_seen", how="left")
            out["retention_pct"] = (out["retained"] / out["cohort_size"]).astype(float)
            return out.sort_values(["_first_seen", "_age"]).reset_index(drop=True)

    # ---- Causal: echo effect details as a 1-row frame
    if fam == "causal":
        ed = row.get("effect_detail")
        if isinstance(ed, str):
            try:
                ed = json.loads(ed)
            except Exception:
                ed = {}
        ed = ed or {}
        return pd.DataFrame([{**{"design": row.get("causal_design")}, **ed}])

    # Fallback: pass-through (safe copy)
    return df.copy()


# -------------------------- chart rendering (generic) --------------------------

def _render_one_named_chart(chart: str, df_comp: pd.DataFrame, row: pd.Series, cfg: RootCfg):
    """
    Route a chart name -> concrete call without touching your chart modules.
    Returns (fig, title) or (None, None) if not applicable for the data shape.
    """
    theme = getattr(cfg.env, "theme", "dark_blue")
    fam = row["family"]
    pf = _coerce_list(row.get("primary_fields"))
    time = row.get("time_field") or None

    # Lazy imports so tests don’t pull heavy deps when charts are disabled
    from src.chart import change_over_time as cot
    from src.chart import correlation as corr
    from src.chart import ranking as rnk
    from src.chart import distribution as dist
    from src.chart import part_to_whole as p2w
    from src.chart import deviation as dev
    from src.chart import magnitude as mag

    # ---- trend shapes: df_comp has [time, value]
    if fam == "trend" and time and {"value", time}.issubset(df_comp.columns):
        if chart == "line":
            return cot.line(df_comp, time=time, value="value", theme_name=theme, title=f"Trend: {pf[0]}"), f"Trend: {pf[0]}"
        if chart == "column":
            return cot.column(df_comp, time=time, value="value", theme_name=theme, title=f"{pf[0]} by period"), f"{pf[0]} by period"
        if chart == "area":
            return cot.area(df_comp, time=time, value="value", group=None, stack=False, theme_name=theme, title=f"{pf[0]} area"), f"{pf[0]} area"
        if chart == "calendar_heatmap":
            return cot.calendar_heatmap(df_comp.rename(columns={time: "date"}), date="date", value="value", theme_name=theme, title=f"{pf[0]} calendar"), f"{pf[0]} calendar"

    # ---- correlation shapes
    if fam == "correlation":
        if len(pf) >= 2 and all(c in df_comp.columns for c in pf[:2]):
            x, y = pf[0], pf[1]
            if chart in ("scatter", "bubble"):
                return corr.scatter(df_comp, x=x, y=y, theme_name=theme, title=f"{x} vs {y}"), f"{x} vs {y}"
            if chart == "xy_heatmap":
                return corr.xy_heatmap(df_comp, x=x, y=y, theme_name=theme, title=f"Density: {x} vs {y}"), f"Density: {x} vs {y}"
        # num-cat aggregated table: columns [category, mean, count]
        cols = list(df_comp.columns)
        if len(cols) >= 2 and "mean" in cols:
            cat = cols[0]
            if chart in ("ordered_bar", "ordered_column"):
                return rnk.ordered_bar(df_comp.rename(columns={"mean": "value"}), category=cat, value="value", theme_name=theme, title=f"{cat}: mean"), f"{cat}: mean"
            if chart in ("violin", "boxplot"):
                tmp = df_comp.rename(columns={"mean": "value"})
                return rnk.dot_strip_plot(tmp, category=cat, value="value", theme_name=theme, title=f"{cat}: mean"), f"{cat}: mean"

    # ---- ranking: df_comp [category, metric]
    if fam == "ranking" and len(pf) >= 2:
        cat, m = pf[0], pf[1]
        if {cat, m}.issubset(df_comp.columns):
            if chart == "ordered_bar":
                return rnk.ordered_bar(df_comp, category=cat, value=m, theme_name=theme, title=f"Ranking: {cat} by {m}"), f"Ranking: {cat} by {m}"
            if chart == "ordered_column":
                return rnk.ordered_column(df_comp, category=cat, value=m, theme_name=theme, title=f"Ranking: {cat} by {m}"), f"Ranking: {cat} by {m}"
            if chart == "lollipop":
                return rnk.lollipop(df_comp, category=cat, value=m, theme_name=theme, title=f"{cat} lollipop"), f"{cat} lollipop"

    # ---- part-to-whole: df_comp [cat, m, pct_total]
    if fam == "part_to_whole" and len(pf) >= 2:
        cat, m = pf[0], pf[1]
        if {cat, m}.issubset(df_comp.columns):
            if chart == "treemap":
                return p2w.treemap(df_comp, labels=cat, parents=None, value=m, theme_name=theme, title="Treemap"), "Treemap"
            if chart == "pie_guarded":
                return p2w.pie_guarded(df_comp, category=cat, value=m, theme_name=theme, title="Pie"), "Pie"
            if chart == "donut_guarded":
                return p2w.donut_guarded(df_comp, category=cat, value=m, theme_name=theme, title="Donut"), "Donut"

    # ---- distribution: df_comp has a single numeric column m
    if fam == "distribution" and pf:
        m = pf[0]
        if m in df_comp.columns:
            if chart == "histogram":
                return dist.histogram(df_comp, value=m, theme_name=theme, title=f"Histogram: {m}"), f"Histogram: {m}"
            if chart == "boxplot":
                return dist.boxplot(df_comp, value=m, by=None, theme_name=theme, title=f"Boxplot: {m}"), f"Boxplot: {m}"
            if chart == "violin":
                return dist.violin(df_comp, value=m, by=None, theme_name=theme, title=f"Violin: {m}"), f"Violin: {m}"
            if chart == "beeswarm":
                return dist.beeswarm(df_comp, value=m, by=None, theme_name=theme, title=f"Beeswarm: {m}"), f"Beeswarm: {m}"
            if chart == "dot_strip_plot":
                return dist.dot_strip_plot(df_comp, value=m, by=None, theme_name=theme, title=f"Strip: {m}"), f"Strip: {m}"
            if chart == "barcode_plot":
                return dist.barcode_plot(df_comp, value=m, theme_name=theme, title=f"Barcode: {m}"), f"Barcode: {m}"
            if chart == "frequency_polygon":
                return dist.frequency_polygon(df_comp, value=m, theme_name=theme, title=f"Freq polygon: {m}"), f"Freq polygon: {m}"
            if chart == "cumulative_curve":
                return dist.cumulative_curve(df_comp, value=m, by=None, theme_name=theme, title=f"Cumulative: {m}"), f"Cumulative: {m}"

    # ---- deviation: df_comp has columns ["index","z"]
    if fam == "deviation":
        if {"index", "z"}.issubset(df_comp.columns) and chart == "diverging_bar":
            return dev.diverging_bar(df_comp, category="index", value="z", reference=0.0, theme_name=theme, title="Deviation (z)"), "Deviation (z)"

    # ---- causal: single-effect bar
    if fam == "causal":
        d = df_comp.copy()
        val = None
        label = None
        for k in ["ate", "did", "jump"]:
            if k in d.columns and len(d) > 0:
                try:
                    val = float(d.iloc[0][k])
                    label = k.upper()
                    break
                except Exception:
                    continue
        if val is not None and chart in ("bar", "column"):
            plot_df = pd.DataFrame({"label": [label or "Effect"], "effect": [val]})
            return mag.bar(plot_df, category="label", value="effect", title="Causal effect", theme_name=theme), "Causal effect"

    return None, None


def _render_charts_for_row(df_comp: pd.DataFrame, row: pd.Series, cfg: RootCfg, out_dir: Path) -> List[Dict[str, Optional[str]]]:
    if not getattr(cfg.reports.enabled_generators, "charts", False):
        return []
    charts = _charts_to_try(row)
    if not charts:
        return []
    made: List[Dict[str, Optional[str]]] = []
    max_per = _max_charts_per_topic(cfg)
    for chart in charts:
        try:
            fig, title = _render_one_named_chart(chart, df_comp, row, cfg)
        except Exception:
            fig, title = None, None
        if fig is None:
            continue
        chart_id = make_chart_id(
            {"dataset_slug": row.get("dataset_slug", ""), "topic_id": row["topic_id"]},
            f"{row['family']}-{chart}",
        )
        html_path = out_dir / f"{chart_id}.html"
        png_path = out_dir / f"{chart_id}.png"

        # export via common helpers
        from src.chart.common import export_html, export_png
        export_html(fig, str(html_path), title=title or chart_id)
        png_written = None
        if getattr(cfg.charts, "export_static_png", False):
            try:
                export_png(fig, str(png_path))
                png_written = str(png_path)
            except Exception:
                png_written = None

        made.append({
            "topic_id": row["topic_id"],
            "chart": chart,
            "html": str(html_path),
            "png": png_written,
        })
        if len(made) >= max_per:
            break
    return made


# ------------------------------- orchestrator -------------------------------

def build_gold_from_silver(cfg: RootCfg, slug: str, *, recheck: bool = True) -> Dict[str, str]:
    storage = build_storage_from_config(cfg)
    root = Path.cwd()

    silver_dir = root / "data" / "silver" / slug

    # Write the rechecked dataset so it actually appears on disk
    if recheck:
        df_silver, _ = recheck_silver(cfg, slug, write_dataset=True)
    else:
        df_silver = pd.read_parquet(silver_dir / "dataset.parquet")

    # Flexible schema loading (your _load_frozen_schema already supports multiple names)
    schema: ProposedSchema = _load_frozen_schema(slug, root)
    if not getattr(schema, "columns", None):
        schema = _derive_schema_from_df(df_silver, slug)
        print(f"  [gold] Using derived schema – {len(schema.columns)} columns")

    # Topics → score → select (with fallback when empty)
    topics = build_candidates(df_silver, schema, cfg)
    used_fallback = False
    if topics is None or len(topics) == 0:
        topics = _fallback_candidates_from_df(df_silver, schema, slug, max_total=getattr(cfg.topics.thresholds, "max_charts_total", 12))
        used_fallback = True

    if used_fallback:
        topics_scored = topics.copy()
        if "score_total" not in topics_scored.columns:
            topics_scored["score_total"] = 1.0
        topics_sel = topics_scored.sort_values("score_total", ascending=False).reset_index(drop=True)
    else:
        topics_scored = score_topics(topics, cfg)
        topics_sel = select_topics(topics_scored, cfg)

    print(f"  [gold] candidates={len(topics)}  selected={len(topics_sel)}")

    # Layout plan
    plan = make_layout_plan(topics_sel, cfg, dataset_slug=slug)

    # Paths & dirs
    gold_base = Path(storage.gold_path(slug))
    art_dir = gold_base / "artifacts"
    td_dir = gold_base / "transformed_data"
    charts_dir = art_dir / "charts"
    rep_dir = gold_base / "reports"
    for p in (art_dir, td_dir, charts_dir, rep_dir):
        p.mkdir(parents=True, exist_ok=True)

    # Persist topics & plan
    write_any(storage, topics, str(art_dir / "topics.parquet"), fmt="parquet")
    write_any(storage, topics_sel, str(art_dir / "topics_selected.parquet"), fmt="parquet")
    (art_dir / "layout_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")

    # Components + charts + manifest
    manifest = {"dataset_slug": slug, "components": [], "charts": []}
    for i, row in topics_sel.reset_index(drop=True).iterrows():
        comp_df = _materialize_component(df_silver, row)
        comp_path = td_dir / f"component_{i+1:03d}_{row['family']}.parquet"
        write_any(storage, comp_df, str(comp_path), fmt="parquet")

        manifest["components"].append({
            "topic_id": row["topic_id"],
            "family": row["family"],
            "path": str(comp_path),
            "proposed_charts": _charts_to_try(row),
        })

        chart_paths = _render_charts_for_row(comp_df, row, cfg, charts_dir)
        manifest["charts"].extend(chart_paths)

    (gold_base / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Quick console summary for this slug
    try:
        fam_counts = topics_sel["family"].value_counts().to_dict()
        sec = len(plan.get("sections", []))
        comps = sum(len(s.get("components", [])) for s in plan.get("sections", []))
        print(f"  sections={sec}  components={comps}  by_family={fam_counts}")
    except Exception:
        pass

    return {
        "topics": str(art_dir / "topics.parquet"),
        "topics_selected": str(art_dir / "topics_selected.parquet"),
        "layout_plan": str(art_dir / "layout_plan.json"),
        "manifest": str(gold_base / "manifest.json"),
    }
