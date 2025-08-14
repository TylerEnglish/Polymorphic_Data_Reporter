from __future__ import annotations
import argparse
import json
import shutil
from pathlib import Path
from typing import List, Tuple

from src.config_model.model import RootCfg
from src.io.storage import build_storage_from_config
from src.nlp.bootstrap import run_nlp_bootstrap
from src.nlg.narrative import narrative_payload


def _is_hidden(p: Path) -> bool:
    return p.name.startswith(".") or p.name.startswith("_")


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


def _resolve_targets(raw_root: Path, target: str | None) -> List[Tuple[str, Path]]:
    """
    Returns (dataset_slug, resolved_path) for each thing we should process.
    If `target` is None -> return all children under raw/ (dirs + top-level files).
    If `target` is provided -> it can be:
      - a slug name (directory name under raw/)
      - a direct path to a folder under raw/
      - a direct path to a single file under raw/
      - a bare filename under raw/ (e.g., 'foo.parquet')
      - an absolute/relative file path outside raw/ (will be staged to raw/<stem>/)
    """
    out: List[Tuple[str, Path]] = []

    if target is None:
        for child in sorted(raw_root.iterdir()):
            if _is_hidden(child):
                continue
            if child.is_dir():
                out.append((child.name, child))
            elif child.is_file():
                slug = _ensure_slug_dir_for_single_file(raw_root, child)
                out.append((slug, raw_root / slug))
        return out

    p = Path(target)
    if not p.is_absolute():
        # Try relative to raw_root
        candidate = (raw_root / target).resolve()
        if candidate.exists():
            p = candidate
        else:
            # If they passed a slug and the folder exists, accept it
            if (raw_root / target).exists():
                p = (raw_root / target).resolve()
            # If they passed a path outside raw_root, accept it if it exists
            elif Path(target).exists():
                p = Path(target).resolve()
            else:
                raise FileNotFoundError(f"Could not resolve selection '{target}' under {raw_root}")

    # If they pointed to raw_root, expand to everything
    if p == raw_root:
        return _resolve_targets(raw_root, None)

    if p.is_dir():
        return [(p.name, p)]
    if p.is_file():
        if p.parent == raw_root:
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
            "If omitted, processes all top-level items in data/raw."
        ),
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

    targets = _resolve_targets(raw_root, args.select)
    if not targets:
        print(f"No targets found under {raw_root}")
        return

    for slug, _resolved in targets:
        _run_one(project_root, cfg, slug)


if __name__ == "__main__":
    main()
