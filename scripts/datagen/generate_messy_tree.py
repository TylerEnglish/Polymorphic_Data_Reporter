from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Optional: faker for more lifelike text (falls back if not installed)
try:
    from faker import Faker  # type: ignore
    _FAKER = Faker()
except Exception:
    _FAKER = None


# =========================================================
# RNG + small helpers
# =========================================================

def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _mask(n: int, p: float, rng: np.random.Generator) -> np.ndarray:
    return rng.random(n) < p


def _to_ns(dt: np.datetime64) -> np.datetime64:
    """Ensure numpy datetime64[ns]."""
    return dt.astype("datetime64[ns]")


def _window(batch_idx: int, rng: np.random.Generator) -> Tuple[np.datetime64, np.datetime64]:
    start = np.datetime64("2023-01-01") + np.timedelta64(int(batch_idx * 5), "D")
    span_days = int(rng.integers(3, 11))
    return _to_ns(start), _to_ns(start + np.timedelta64(span_days, "D"))

def _normalize_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make per-column dtypes Arrow-safe & consistent across chunks:
    - object -> string[pyarrow]  (handles mixed types / dirty cols)
    - boolean with NA -> pandas 'boolean'
    - int with NA -> pandas 'Int64'
    - datetimes -> datetime64[ns]
    """
    out = df.copy()

    for col in out.columns:
        s = out[col]
        if pd.api.types.is_object_dtype(s):
            # mixed types → store as strings in Parquet
            out[col] = s.astype("string[pyarrow]")
        elif pd.api.types.is_bool_dtype(s) and str(s.dtype) != "bool":
            # allow <NA> in booleans
            out[col] = s.astype("boolean")
        elif pd.api.types.is_integer_dtype(s) and s.isna().any():
            # nullable integers
            out[col] = s.astype("Int64")
        elif pd.api.types.is_datetime64_any_dtype(s):
            out[col] = pd.to_datetime(s, utc=False, errors="coerce")
        # floats are fine as-is
    return out

# =========================================================
# text + variants / typos
# =========================================================

_KEY_NEIGHBORS = {
    "a": "aqwsz", "b": "vghn", "c": "xdfv", "d": "ersfcx",
    "e": "wsdr", "f": "rtgvcd", "g": "tyhbvf", "h": "yujnbg",
    "i": "ujko", "j": "uikmnh", "k": "ijolm", "l": "kop",
    "m": "njk", "n": "bhjm", "o": "iklp", "p": "ol",
    "q": "wa", "r": "edft", "s": "awedxz", "t": "rfgy",
    "u": "yhji", "v": "cfgb", "w": "qase", "x": "zsdc",
    "y": "tugh", "z": "asx"
}

def _typo_token(tok: str, rng: np.random.Generator) -> str:
    if not tok:
        return tok
    ops = ["swap", "drop", "near", "insert", "space", "case"]
    op = rng.choice(ops, p=[0.18, 0.22, 0.28, 0.12, 0.10, 0.10])
    s = list(tok)
    if op == "swap" and len(s) >= 2:
        i = int(rng.integers(0, len(s)-1))
        s[i], s[i+1] = s[i+1], s[i]
        return "".join(s)
    if op == "drop":
        i = int(rng.integers(0, len(s)))
        return "".join(s[:i] + s[i+1:])
    if op == "near":
        i = int(rng.integers(0, len(s)))
        ch = s[i].lower()
        cand = _KEY_NEIGHBORS.get(ch, ch)
        r = rng.choice(list(cand)) if cand else ch
        s[i] = r
        return "".join(s)
    if op == "insert":
        i = int(rng.integers(0, len(s)+1))
        ch = rng.choice(list("abcdefghijklmnopqrstuvwxyz"))
        return "".join(s[:i] + [ch] + s[i:])
    if op == "space":
        return (" " * int(rng.integers(0, 3))) + tok + (" " * int(rng.integers(0, 3)))
    if op == "case":
        return tok.upper() if rng.random() < 0.5 else tok.capitalize()
    return tok


def _skewed_categories(n: int, rng: np.random.Generator) -> np.ndarray:
    """Uneven categories with many spellings/variants."""
    if _FAKER:
        base = [_FAKER.word() for _ in range(6)]
        base[:4] = ["alpha", "beta", "gamma", "delta"]
    else:
        base = ["alpha", "beta", "gamma", "delta", "kappa", "omega"]

    probs = np.array([0.50, 0.20, 0.12, 0.10, 0.05, 0.03], dtype=float)
    probs /= probs.sum()
    picks = rng.choice(base, size=n, p=probs)
    out = []
    for w in picks:
        if rng.random() < 0.5:
            out.append(_typo_token(w, rng))
        else:
            out.append(w)
    # add some near-null-ish values (not real nulls)
    for i in rng.integers(0, n, size=max(1, n // 50)):
        out[i] = rng.choice(["N/A", "UNK", "—", "", "  "], p=[.3, .25, .2, .15, .1])
    return np.array(out, dtype=object)


def _free_text(n: int, rng: np.random.Generator, dirty: bool) -> np.ndarray:
    out = []
    for _ in range(n):
        if _FAKER:
            txt = _FAKER.sentence(nb_words=int(rng.integers(3, 16)))
        else:
            L = int(rng.integers(0, 48))
            txt = "".join(rng.choice(list(" abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789—–-·•.,;:!?🙂éáô"),
                                     size=L))
        if dirty:
            if rng.random() < 0.5:
                txt = _typo_token(txt, rng)
        # Random leading/trailing spaces
        if rng.random() < 0.25:
            txt = (" " * int(rng.integers(0, 3))) + txt + (" " * int(rng.integers(0, 3)))
        out.append(txt)
    return np.array(out, dtype=object)


# =========================================================
# column value generators (vectorized)
# =========================================================

def gen_numeric(n: int, rng: np.random.Generator, dirty: bool) -> np.ndarray:
    x = rng.lognormal(mean=3.0, sigma=0.9, size=n)  # positive skew
    # outliers
    out_idx = rng.integers(0, n, size=max(1, n // 300))
    x[out_idx] *= rng.uniform(10, 300, size=len(out_idx))
    if not dirty:
        return x  # clean floats
    # dirty: cast some to messy strings/units
    out = x.astype(object)
    for i in rng.integers(0, n, size=max(1, n // 40)):
        r = rng.random()
        if r < 0.35: out[i] = f"{x[i]:.1f}k"
        elif r < .65: out[i] = "N/A"
        else: out[i] = "—"
    return out


def gen_bool(n: int, rng: np.random.Generator, dirty: bool) -> np.ndarray:
    if not dirty:
        return rng.choice([True, False], size=n, p=[0.6, 0.4])
    vals = np.array(
        ["Y", "N", "True", "False", "yes", "no", "1", "0", True, False, " t ", " f ", "", "—"],
        dtype=object
    )
    p = np.array([.15, .12, .09, .08, .08, .08, .08, .08, .05, .05, .05, .05, .01, .01])
    p = p / p.sum()
    return rng.choice(vals, size=n, p=p)


def gen_cat(n: int, rng: np.random.Generator, dirty: bool) -> np.ndarray:
    arr = _skewed_categories(n, rng)
    if not dirty:
        # strip + lower for a "clean" categorical
        return np.char.strip(np.char.lower(arr.astype(str)))
    return arr


def gen_datetime(n: int, rng: np.random.Generator, window: Tuple[np.datetime64, np.datetime64]) -> np.ndarray:
    start, end = window
    secs = int((end - start) / np.timedelta64(1, "s"))
    t = start + rng.integers(0, max(secs, 1), size=n).astype("timedelta64[s]")
    return t.astype("datetime64[ns]")


def gen_text(n: int, rng: np.random.Generator, dirty: bool) -> np.ndarray:
    return _free_text(n, rng, dirty)


# =========================================================
# schema construction per FILE (variable # of columns)
# =========================================================

def build_schema_for_file(
    rng: np.random.Generator,
    *,
    cols_min: int,
    cols_max: int,
) -> Dict[str, Dict[str, Any]]:
    """
    Decide how many columns of each type a single file will have, and per-column null probs.
    Ensures at least one complete (null_p=0) column for each present type.
    Also ensures at least one *very sparse* numeric (~90% nulls) if any numeric exist.
    """
    total_cols = int(rng.integers(cols_min, cols_max + 1))

    # draw random composition via Dirichlet (more numeric by default)
    types = ["num", "bool", "cat", "text", "dt"]
    w = rng.dirichlet(alpha=np.array([3.5, 1.2, 1.6, 1.6, 1.0]))
    counts = (np.round(w * total_cols)).astype(int)

    # adjust rounding drift
    diff = total_cols - int(counts.sum())
    if diff != 0:
        # fix by adding/removing from the largest bucket
        idx = int(np.argmax(counts))
        counts[idx] += diff

    # make sure no type with zero columns accidentally claims "complete"; that's fine.
    spec: Dict[str, Dict[str, Any]] = {}
    name_counters = {t: 0 for t in types}

    def _next_name(t: str) -> str:
        name_counters[t] += 1
        return f"{t}_{name_counters[t]:02d}"

    # pool of null probabilities (include 0 and 0.9)
    NULL_POOL = np.array([0.0, 0.02, 0.05, 0.12, 0.25, 0.5, 0.90])
    NULL_P = NULL_POOL / NULL_POOL.sum()  # not used directly; we'll choose manually

    # Build per-type columns
    for t, cnt in zip(types, counts):
        if cnt <= 0:
            continue

        # choose how many "complete" in this type (>=1 if cnt>0)
        n_complete = int(max(1, np.floor(0.15 * cnt)))
        # rest are mixed sparsity
        n_dirty = cnt - n_complete

        # ensure at least one very sparse numeric
        force_sparse_numeric = (t == "num" and cnt > 0)

        for i in range(cnt):
            name = _next_name(t)
            if i < n_complete:
                null_p = 0.0
                dirty = False  # complete columns are clean to guarantee no null/dirty issues
            else:
                # choose from pool; bias toward mid/high nulls
                null_p = float(rng.choice([0.05, 0.12, 0.25, 0.5, 0.9], p=[0.15, 0.25, 0.25, 0.20, 0.15]))
                dirty = True

            # On first numeric in dirty set, force 0.9 nulls (simulate "num_13"-like)
            if force_sparse_numeric and i == n_complete and n_dirty > 0:
                null_p = 0.90
                dirty = True
                force_sparse_numeric = False

            spec[name] = {"type": t, "null_p": null_p, "dirty": dirty}

    return spec


# =========================================================
# chunked row generation per FILE, given its schema
# =========================================================

def gen_chunk_for_file(
    n: int,
    rng: np.random.Generator,
    *,
    file_schema: Dict[str, Dict[str, Any]],
    batch_id: str,
    window: Tuple[np.datetime64, np.datetime64],
    entity_seed_start: int,
) -> pd.DataFrame:
    # Shared keys across the batch/folder
    start, end = window
    secs = int((end - start) / np.timedelta64(1, "s"))
    event_time = start + rng.integers(0, max(secs, 1), size=n).astype("timedelta64[s]")
    event_time = event_time.astype("datetime64[ns]")
    entity_id = (rng.zipf(a=1.5, size=n) + entity_seed_start).astype(np.int64)
    row_id = rng.integers(10**9, 10**12, size=n).astype(np.int64)

    out = {
        "batch_id": np.repeat(batch_id, n),
        "row_id": row_id,
        "entity_id": entity_id,
        "event_time": event_time,  # this one may later be complemented by extra dt_* columns from schema
    }

    # Generate each extra column
    for name, meta in file_schema.items():
        t = meta["type"]; null_p = float(meta["null_p"]); dirty = bool(meta["dirty"])

        if t == "num":
            col = gen_numeric(n, rng, dirty)
        elif t == "bool":
            col = gen_bool(n, rng, dirty)
        elif t == "cat":
            col = gen_cat(n, rng, dirty)
        elif t == "text":
            col = gen_text(n, rng, dirty)
        elif t == "dt":
            col = gen_datetime(n, rng, window)  # numpy datetime64[ns]
            if dirty:
                # flip ~5% of rows to string-formatted timestamps
                m = _mask(n, 0.05, rng)
                s = pd.Series(col, dtype="datetime64[ns]")  # ensure datetime64 series
                if m.any():
                    s.loc[m] = pd.to_datetime(s.loc[m], errors="coerce").dt.strftime("%Y/%m/%d %H:%M:%S")
                col = s.astype(object).to_numpy()  # allow mixing strings & datetimes
        else:
            raise ValueError(f"unknown type {t}")

        # apply nulls (but keep complete columns intact)
        if null_p > 0:
            m = _mask(n, null_p, rng)
            if t == "dt":
                # preserve dtype where possible
                col = pd.Series(col, dtype="object")
                col[m] = pd.NaT
            else:
                col = pd.Series(col, dtype="object")
                col[m] = None
            out[name] = col.values
        else:
            out[name] = col

    return pd.DataFrame(out)


# =========================================================
# writers (chunked)
# =========================================================

def write_parquet(path: Path, chunk_iter: Iterable[pd.DataFrame], compression: str = "zstd"):
    writer: Optional[pq.ParquetWriter] = None
    schema: Optional[pa.Schema] = None

    for chunk in chunk_iter:
        chunk = _normalize_for_parquet(chunk)
        table = pa.Table.from_pandas(chunk, preserve_index=False, schema=schema)

        if writer is None:
            schema = table.schema  # freeze schema from the first normalized chunk
            writer = pq.ParquetWriter(
                str(path), schema, compression=compression, use_dictionary=True
            )

        writer.write_table(table)

    if writer is not None:
        writer.close()

def write_csv(path: Path, chunk_iter: Iterable[pd.DataFrame]):
    first = True
    for chunk in chunk_iter:
        chunk.to_csv(path, mode="w" if first else "a", index=False, header=first)
        first = False


def write_json_lines(path: Path, chunk_iter: Iterable[pd.DataFrame]):
    with path.open("w", encoding="utf-8") as f:
        for chunk in chunk_iter:
            for rec in chunk.to_dict(orient="records"):
                f.write(json.dumps(rec, default=str, ensure_ascii=False))
                f.write("\n")


def write_excel(path: Path, first_chunk: pd.DataFrame):
    """Write just one chunk to a sheet to avoid huge RAM; caller supplies the first chunk explicitly."""
    try:
        with pd.ExcelWriter(path, engine="openpyxl") as xw:
            first_chunk.to_excel(xw, sheet_name="Sheet1", index=False, freeze_panes=(1, 1))
    except Exception:
        # Fallback to CSV if no engine
        path.with_suffix(".csv").write_text(first_chunk.to_csv(index=False), encoding="utf-8")


# =========================================================
# file generation orchestration
# =========================================================

def _file_chunks(
    *,
    rows_total: int,
    rows_per_chunk: int,
    rng: np.random.Generator,
    file_schema: Dict[str, Dict[str, Any]],
    batch_id: str,
    window: Tuple[np.datetime64, np.datetime64],
    entity_seed_start: int,
) -> Iterable[pd.DataFrame]:
    remaining = rows_total
    while remaining > 0:
        n = int(min(rows_per_chunk, remaining))
        remaining -= n
        yield gen_chunk_for_file(
            n, rng,
            file_schema=file_schema, batch_id=batch_id,
            window=window, entity_seed_start=entity_seed_start
        )


def _next_name(counter: Dict[str, int]) -> str:
    counter["i"] += 1
    return f"x{counter['i']}"


def build_tree(
    out_root: Path,
    *,
    seed: int = 123,
    root_files: int = 2,
    folders: int = 4,
    subfolders_prob: float = 0.5,
    files_per_folder_min: int = 2,
    files_per_folder_max: int = 4,
    rows_per_file_min: int = 25_000,
    rows_per_file_max: int = 120_000,
    rows_per_chunk: int = 25_000,
    cols_min: int = 8,
    cols_max: int = 28,
) -> None:
    """
    Create a heterogeneous tree of Parquet/CSV/XLSX/JSON.
    Each file has a **variable number of extra columns** with mixed types and null rates.
    Each folder is a batch sharing batch_id and a time window.
    Some columns per type are guaranteed complete (no nulls).
    """
    out_root.mkdir(parents=True, exist_ok=True)
    rng = _rng(seed)
    name_ctr = {"i": 0}
    fmts_cycle = ["parquet", "json", "csv", "xlsx"]

    # ---------- root files (batch_root) ----------
    batch_id = "batch_root"
    window = _window(0, rng)
    for k in range(root_files):
        fmt = fmts_cycle[k % len(fmts_cycle)]
        fname = _next_name(name_ctr)
        path = out_root / f"{fname}.{('parquet' if fmt=='parquet' else 'json' if fmt=='json' else 'csv' if fmt=='csv' else 'xlsx')}"
        rows = int(rng.integers(rows_per_file_min, rows_per_file_max))

        # schema per file
        file_schema = build_schema_for_file(rng, cols_min=cols_min, cols_max=cols_max)

        if fmt == "xlsx":
            # Excel: single chunk only
            first = gen_chunk_for_file(
                min(rows, rows_per_chunk), rng,
                file_schema=file_schema, batch_id=batch_id, window=window,
                entity_seed_start=10_000 + k * 10_000
            )
            write_excel(path, first)
        else:
            chunks = _file_chunks(
                rows_total=rows, rows_per_chunk=rows_per_chunk, rng=rng,
                file_schema=file_schema, batch_id=batch_id, window=window,
                entity_seed_start=10_000 + k * 10_000
            )
            if fmt == "parquet":
                write_parquet(path, chunks)
            elif fmt == "csv":
                write_csv(path, chunks)
            else:
                write_json_lines(path, chunks)

    # ---------- subfolders (batch_01 ..) ----------
    for fi in range(1, folders + 1):
        fdir = out_root / f"f{fi}"
        fdir.mkdir(exist_ok=True)
        batch_id = f"batch_{fi:02d}"
        window = _window(fi, rng)

        nfiles = int(rng.integers(files_per_folder_min, files_per_folder_max + 1))
        for j in range(nfiles):
            fmt = fmts_cycle[(fi + j) % len(fmts_cycle)]
            fname = _next_name(name_ctr)
            path = fdir / f"{fname}.{('parquet' if fmt=='parquet' else 'json' if fmt=='json' else 'csv' if fmt=='csv' else 'xlsx')}"
            rows = int(rng.integers(rows_per_file_min, rows_per_file_max))

            # schema per file (unique to file)
            file_schema = build_schema_for_file(rng, cols_min=cols_min, cols_max=cols_max)

            if fmt == "xlsx":
                first = gen_chunk_for_file(
                    min(rows, rows_per_chunk), rng,
                    file_schema=file_schema, batch_id=batch_id, window=window,
                    entity_seed_start=100_000 * fi + j * 10_000
                )
                write_excel(path, first)
            else:
                chunks = _file_chunks(
                    rows_total=rows, rows_per_chunk=rows_per_chunk, rng=rng,
                    file_schema=file_schema, batch_id=batch_id, window=window,
                    entity_seed_start=100_000 * fi + j * 10_000
                )
                if fmt == "parquet":
                    write_parquet(path, chunks)
                elif fmt == "csv":
                    write_csv(path, chunks)
                else:
                    write_json_lines(path, chunks)

            # maybe create a nested subfolder
            if rng.random() < subfolders_prob:
                sdir = fdir / f"f{fi}_{j+1}"
                sdir.mkdir(exist_ok=True)
                nsub = int(rng.integers(1, 3 + 1))
                for sj in range(nsub):
                    fmt2 = fmts_cycle[(fi + j + sj) % len(fmts_cycle)]
                    sname = _next_name(name_ctr)
                    spath = sdir / f"{sname}.{('parquet' if fmt2=='parquet' else 'json' if fmt2=='json' else 'csv' if fmt2=='csv' else 'xlsx')}"
                    rows2 = int(rng.integers(rows_per_file_min // 2, rows_per_file_max // 2))

                    file_schema2 = build_schema_for_file(rng, cols_min=cols_min, cols_max=cols_max)

                    if fmt2 == "xlsx":
                        first2 = gen_chunk_for_file(
                            min(rows2, rows_per_chunk), rng,
                            file_schema=file_schema2, batch_id=batch_id, window=window,
                            entity_seed_start=1_000_000 * fi + j * 100_000 + sj * 10_000
                        )
                        write_excel(spath, first2)
                    else:
                        chunks2 = _file_chunks(
                            rows_total=rows2, rows_per_chunk=rows_per_chunk, rng=rng,
                            file_schema=file_schema2, batch_id=batch_id, window=window,
                            entity_seed_start=1_000_000 * fi + j * 100_000 + sj * 10_000
                        )
                        if fmt2 == "parquet":
                            write_parquet(spath, chunks2)
                        elif fmt2 == "csv":
                            write_csv(spath, chunks2)
                        else:
                            write_json_lines(spath, chunks2)


# =========================================================
# CLI
# =========================================================

def main():
    ap = argparse.ArgumentParser(description="Generate messy multi-format dataset tree with variable columns and realistic null patterns.")
    ap.add_argument("-o", "--out", default="data/raw", help="output root directory")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--root-files", type=int, default=2)
    ap.add_argument("--folders", type=int, default=4)
    ap.add_argument("--subfolders-prob", type=float, default=0.5)
    ap.add_argument("--files-per-folder-min", type=int, default=2)
    ap.add_argument("--files-per-folder-max", type=int, default=4)
    ap.add_argument("--rows-per-file-min", type=int, default=25000)
    ap.add_argument("--rows-per-file-max", type=int, default=120000)
    ap.add_argument("--rows-per-chunk", type=int, default=25000)
    ap.add_argument("--cols-min", type=int, default=8, help="min extra columns per file (variable)")
    ap.add_argument("--cols-max", type=int, default=28, help="max extra columns per file (variable)")
    args = ap.parse_args()

    build_tree(
        out_root=Path(args.out),
        seed=args.seed,
        root_files=args.root_files,
        folders=args.folders,
        subfolders_prob=args.subfolders_prob,
        files_per_folder_min=args.files_per_folder_min,
        files_per_folder_max=args.files_per_folder_max,
        rows_per_file_min=args.rows_per_file_min,
        rows_per_file_max=args.rows_per_file_max,
        rows_per_chunk=args.rows_per_chunk,
        cols_min=args.cols_min,
        cols_max=args.cols_max,
    )
    print(f"✔ Wrote mixed-format messy dataset under {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()