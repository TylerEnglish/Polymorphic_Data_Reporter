from __future__ import annotations
import argparse
import json
import shutil
from pathlib import Path
from typing import List, Tuple, Optional

from src.config_model.model import RootCfg
from src.io.storage import build_storage_from_config
from src.nlp.bootstrap import run_nlp_bootstrap
from src.nlg.narrative import narrative_payload

# Recognized data file extensions
DATA_EXTS = {".csv", ".json", ".ndjson", ".parquet", ".pq", ".feather", ".arrows", ".txt"}


def _is_hidden(p: Path) -> bool:
    return p.name.startswith(".") or p.name.startswith("_")


def _is_data_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in DATA_EXTS


def _raw_root(cfg: RootCfg) -> Path:
    # RootCfg already normalizes absolute paths to raw_root
    rr = Path(cfg.storage.local.raw_root)
    rr.mkdir(parents=True, exist_ok=True)
    return rr


def _ensure_slug_dir_for_single_file(raw_root: Path, file_path: Path) -> str:
    """
    If `file_path` is a single file directly under raw/, we create raw/<stem>/
    and copy the file there so the bootstrap (which expects a slug) can discover it.
    Returns the slug (the stem).
    """
    slug = file_path.stem
    slug_dir = raw_root / slug
    slug_dir.mkdir(parents=True, exist_ok=True)
    staged = slug_dir / file_path.name
    if not staged.exists():
        shutil.copy2(file_path, staged)
    return slug


def _slugify_file(raw_root: Path, f: Path) -> str:
    """Stable dataset slug for a single file based on its relative path under raw/."""
    rel = f.relative_to(raw_root)
    parts = list(rel.parts)
    parts[-1] = Path(parts[-1]).stem  # drop one suffix
    return "__".join(parts)


# ---------- Granularity enumerators (used when --select is NOT provided) ----------

def _targets_by_slug(raw_root: Path) -> List[Tuple[str, Path]]:
    out: List[Tuple[str, Path]] = []
    for child in sorted(raw_root.iterdir()):
        if _is_hidden(child):
            continue
        if child.is_dir():
            out.append((child.name, child))
        elif _is_data_file(child):
            slug = _ensure_slug_dir_for_single_file(raw_root, child)
            out.append((slug, raw_root / slug))
    return out


def _targets_by_subdir(raw_root: Path) -> List[Tuple[str, Path]]:
    """
    Each immediate subdirectory inside each top-level slug becomes a dataset.
    If a slug has no subdirs, treat the slug itself as a dataset.
    """
    out: List[Tuple[str, Path]] = []
    for slug_dir in sorted(p for p in raw_root.iterdir() if p.is_dir() and not _is_hidden(p)):
        subdirs = [p for p in slug_dir.iterdir() if p.is_dir() and not _is_hidden(p)]
        if not subdirs:
            out.append((slug_dir.name, slug_dir))
            continue
        for sd in subdirs:
            ds_slug = f"{slug_dir.name}__{sd.name}"
            out.append((ds_slug, sd))
    return out


def _targets_by_file(raw_root: Path) -> List[Tuple[str, Path]]:
    """
    Every individual file becomes its own dataset. We stage each file into
    raw/<slugified_relative_path>/<originalname> for consistent processing.
    """
    out: List[Tuple[str, Path]] = []
    for f in sorted(raw_root.rglob("*")):
        if _is_hidden(f) or not _is_data_file(f):
            continue
        slug = _slugify_file(raw_root, f)
        slug_dir = raw_root / slug
        slug_dir.mkdir(parents=True, exist_ok=True)
        staged = slug_dir / f.name
        # Avoid copying if it's already that exact file
        if staged.resolve() != f.resolve() and not staged.exists():
            shutil.copy2(f, staged)
        out.append((slug, slug_dir))
    return out


def _resolve_targets(raw_root: Path, target: Optional[str], *, granularity: str = "slug") -> List[Tuple[str, Path]]:
    """
    Returns (dataset_slug, resolved_path) for each thing we should process.

    If `target` is provided -> resolve that one thing (slug/folder/file/absolute path).
    If `target` is None -> enumerate under raw_root according to granularity:
        - "slug": each top-level dir (and top-level data file)
        - "subdir": each subdir under each top-level slug (fallback to slug itself if no subdirs)
        - "file": each individual data file becomes a dataset
    """
    if target is not None:
        # Specific selection resolution (compatible with previous behavior)
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

        # If they pointed to raw_root, expand according to granularity
        if p == raw_root:
            if granularity == "slug":
                return _targets_by_slug(raw_root)
            if granularity == "subdir":
                return _targets_by_subdir(raw_root)
            if granularity == "file":
                return _targets_by_file(raw_root)
            raise ValueError(f"Unknown granularity: {granularity}")

        if p.is_dir():
            # Treat the directory itself as a dataset
            return [(p.name, p)]

        if p.is_file():
            if p.parent == raw_root and _is_data_file(p):
                slug = _ensure_slug_dir_for_single_file(raw_root, p)
                return [(slug, raw_root / slug)]
            # Outside raw/: stage to raw/<stem>/
            slug = p.stem
            slug_dir = raw_root / slug
            slug_dir.mkdir(parents=True, exist_ok=True)
            staged = slug_dir / p.name
            if not staged.exists():
                shutil.copy2(p, staged)
            return [(slug, slug_dir)]

        raise FileNotFoundError(f"Selection '{target}' does not exist or is not a file/folder.")

    # No explicit target: enumerate all by granularity
    if granularity == "slug":
        return _targets_by_slug(raw_root)
    if granularity == "subdir":
        return _targets_by_subdir(raw_root)
    if granularity == "file":
        return _targets_by_file(raw_root)
    raise ValueError(f"Unknown granularity: {granularity}")


def _run_one(project_root: Path, cfg: RootCfg, dataset_slug: str) -> None:
    st = build_storage_from_config(cfg)
    out = run_nlp_bootstrap(st, dataset_slug, project_root=project_root)

    # Write schema
    schema_path = Path(out["io"]["schema_toml_path"])
    schema_path.parent.mkdir(parents=True, exist_ok=True)
    schema_path.write_text(out["io"]["schema_toml_text"], encoding="utf-8")

    # Write bootstrap JSON
    bootstrap_path = Path(out["io"]["bootstrap_json_path"])
    bootstrap_path.parent.mkdir(parents=True, exist_ok=True)
    bootstrap_path.write_text(json.dumps(out["bootstrap"], indent=2), encoding="utf-8")

    # Narrative bundle next to bootstrap
    nar = narrative_payload(dataset_slug, out["bootstrap"], out["entries"])
    (bootstrap_path.parent / "narrative.json").write_text(json.dumps(nar, indent=2), encoding="utf-8")

    print(f"✓ {dataset_slug}:")
    print(f"  schema -> {schema_path}")
    print(f"  bootstrap -> {bootstrap_path}")
    print(f"  narrative -> {bootstrap_path.parent / 'narrative.json'}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run NLP bootstrap over data/raw (all slugs) or a specific file/folder."
    )
    ap.add_argument(
        "--select",
        default=None,
        help=(
            "Optional: a slug (folder under data/raw), a folder path, or a single file. "
            "If omitted, enumerates datasets under data/raw."
        ),
    )
    ap.add_argument(
        "--granularity",
        choices=["slug", "subdir", "file"],
        default=None,
        help="How to split datasets under data/raw when --select is omitted (default comes from config).",
    )
    ap.add_argument(
        "--config",
        default="config/config.toml",
        help="Path to config TOML (defaults to config/config.toml).",
    )
    args = ap.parse_args()

    project_root = Path.cwd()
    cfg = RootCfg.load(args.config)
    raw_root = _raw_root(cfg)

    # Prefer CLI flag; else fall back to config nlp.granularity; else default "slug"
    granularity = args.granularity or getattr(getattr(cfg, "nlp", object()), "granularity", "slug")

    targets = _resolve_targets(raw_root, args.select, granularity=granularity)
    if not targets:
        print(f"No targets found under {raw_root}")
        return

    for slug, _resolved in targets:
        _run_one(project_root, cfg, slug)


if __name__ == "__main__":
    main()
