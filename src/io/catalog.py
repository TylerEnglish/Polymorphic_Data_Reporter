from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import re

from .storage import Storage

DATA_FILE_RE = re.compile(r".*\.(csv|json|ndjson|parquet|pq|feather|txt)$", re.IGNORECASE)

@dataclass(frozen=True)
class CatalogEntry:
    slug: str
    uri: str   # absolute path or s3:// uri
    kind: str  # file extension kind
    size: Optional[int] = None

class Catalog:
    """
    Extremely small dataset catalog that discovers files under raw roots and
    groups them per <dataset_slug>. You can store or export this as bronze/_inventory later.
    """
    def __init__(self, storage: Storage) -> None:
        self.storage = storage

    def discover_raw_local(self, dataset_slug: Optional[str] = None) -> List[CatalogEntry]:
        """
        Walk local raw root (recursively). If dataset_slug is provided, scope to that folder.
        """
        base = self.storage.raw_path(dataset_slug or "")
        files = self.storage.ls(base)
        entries: List[CatalogEntry] = []
        for f in files:
            if DATA_FILE_RE.match(f):
                kind = f.split(".")[-1].lower()
                # If user scoped to a dataset, keep that slug; otherwise infer from raw root.
                slug = dataset_slug or _slug_from_raw_root(self.storage, f)
                entries.append(CatalogEntry(slug=slug, uri=f, kind=kind))
        return entries

    def discover_raw_s3(self, dataset_slug: Optional[str] = None) -> List[CatalogEntry]:
        base = self.storage.raw_path(dataset_slug or "", s3=True)
        files = self.storage.ls(base)
        entries: List[CatalogEntry] = []
        for f in files:
            if DATA_FILE_RE.match(f):
                kind = f.split(".")[-1].lower()
                # Enforce scope and keep provided slug if set
                if dataset_slug:
                    if not f.lower().startswith(base.lower()):
                        continue
                    slug = dataset_slug
                else:
                    slug = _slug_from_s3_key(self.storage, f)
                entries.append(CatalogEntry(slug=slug, uri=f, kind=kind))
        return entries

    def inventory(self, use_s3: bool = False, dataset_slug: Optional[str] = None) -> List[CatalogEntry]:
        """
        Unified discovery entrypoint. Set dataset_slug to limit scope.
        """
        if use_s3:
            return self.discover_raw_s3(dataset_slug)
        return self.discover_raw_local(dataset_slug)

# ---- helpers to compute slugs from paths ----

def _slug_from_path(root: str, full: str) -> str:
    # turn /abs/.../data/raw/<slug>/file.ext -> <slug>
    try:
        i = full.lower().index(str(root).lower().rstrip(os_sep()) + os_sep())
        tail = full[i + len(str(root).rstrip(os_sep()) + os_sep()):]
        return tail.split(os_sep(), 1)[0]
    except Exception:
        # fallback: take directory name
        return Path(full).parent.name

def _slug_from_raw_root(storage: Storage, full: str) -> str:
    # guess the immediate child under raw root
    raw_root = storage.raw_path("")
    try:
        i = full.lower().index(str(raw_root).lower().rstrip(os_sep()) + os_sep())
        tail = full[i + len(str(raw_root).rstrip(os_sep()) + os_sep()):]
        return tail.split(os_sep(), 1)[0]
    except Exception:
        return Path(full).parent.name

def _slug_from_s3_key(storage: Storage, uri: str) -> str:
    # s3://bucket/raw/<slug>/file.ext (with optional prefix)
    # we use configured raw_prefix to find slug
    raw_prefix = (storage.cfg.s3_raw_prefix or "").strip("/")
    try:
        # normalize to key without s3://bucket/
        parts = uri.replace("s3://", "").split("/", 1)
        key = parts[1] if len(parts) > 1 else ""
        key_tail = key[len(raw_prefix) + 1 :] if raw_prefix and key.startswith(raw_prefix + "/") else key
        return key_tail.split("/", 1)[0]
    except Exception:
        return Path(uri).parent.name

def os_sep() -> str:
    from os import sep as _sep
    return _sep
