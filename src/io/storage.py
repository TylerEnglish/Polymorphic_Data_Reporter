from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Protocol, Tuple, Union
import io
import os

# Optional deps (declared in your pyproject)
# We import lazily so local FS still works if S3/MinIO libs aren't present.
try:
    import s3fs  # type: ignore
except Exception:
    s3fs = None  # type: ignore

# -------- Public interface (easy to mock in tests) --------

class BlobStore(Protocol):
    def exists(self, path: str) -> bool: ...
    def ls(self, prefix: str) -> List[str]: ...
    def read_bytes(self, path: str) -> bytes: ...
    def write_bytes(self, path: str, data: bytes) -> None: ...
    def open(self, path: str, mode: str = "rb"): ...  # returns file-like (context manager)

@dataclass(frozen=True)
class StoreConfig:
    # Normalized roots (absolute for local; s3 URI for object store)
    local_raw_root: str
    local_gold_root: str
    s3_enabled: bool
    s3_endpoint: Optional[str]
    s3_bucket: Optional[str]
    s3_access_key: Optional[str]
    s3_secret_key: Optional[str]
    s3_secure: bool
    s3_raw_prefix: Optional[str]
    s3_gold_prefix: Optional[str]

# -------- Helpers --------

def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

def _is_s3_uri(p: str) -> bool:
    return p.startswith("s3://")

def _strip_scheme_file(p: str) -> str:
    return p[7:] if p.startswith("file://") else p

# -------- Local FS backend --------

class LocalStore(BlobStore):
    def exists(self, path: str) -> bool:
        return Path(_strip_scheme_file(path)).exists()

    def ls(self, prefix: str) -> List[str]:
        base = Path(_strip_scheme_file(prefix))
        if base.is_file():
            return [str(base)]
        if not base.exists():
            return []
        out: List[str] = []
        for p in base.rglob("*"):
            if p.is_file():
                out.append(str(p))
        return out

    def read_bytes(self, path: str) -> bytes:
        p = Path(_strip_scheme_file(path))
        return p.read_bytes()

    def write_bytes(self, path: str, data: bytes) -> None:
        p = Path(_strip_scheme_file(path))
        _ensure_parent_dir(p)
        p.write_bytes(data)

    def open(self, path: str, mode: str = "rb"):
        p = Path(_strip_scheme_file(path))
        _ensure_parent_dir(p)
        return p.open(mode)

# -------- S3/MinIO backend via s3fs --------

class S3Store(BlobStore):
    def __init__(
        self,
        endpoint_url: Optional[str],
        key: Optional[str],
        secret: Optional[str],
        secure: bool = False,
    ) -> None:
        if s3fs is None:
            raise RuntimeError("s3fs is not installed. Add it to dependencies.")
        client_kwargs = {}
        if endpoint_url:
            client_kwargs["endpoint_url"] = endpoint_url
        self._fs = s3fs.S3FileSystem(
            key=key or None,
            secret=secret or None,
            client_kwargs=client_kwargs,  # works for MinIO, too
            use_ssl=bool(secure),
        )

    def exists(self, path: str) -> bool:
        return self._fs.exists(path)

    def ls(self, prefix: str) -> List[str]:
        try:
            items = self._fs.ls(prefix, detail=False)
        except FileNotFoundError:
            return []
        # s3fs.ls returns keys with schema s3://bucket/key
        return [f"s3://{p}" if not p.startswith("s3://") else p for p in items]

    def read_bytes(self, path: str) -> bytes:
        with self._fs.open(path, "rb") as f:
            return f.read()

    def write_bytes(self, path: str, data: bytes) -> None:
        with self._fs.open(path, "wb") as f:
            f.write(data)

    def open(self, path: str, mode: str = "rb"):
        # Caller must close; usually used in a context manager
        return self._fs.open(path, mode)

# -------- Composite router --------

class Storage:
    """
    Unified storage facade. Knows how to route file://, s3://, and plain local paths.
    Also exposes helpers to build dataset-aware paths for raw/gold roots.
    """
    def __init__(self, cfg: StoreConfig) -> None:
        self.cfg = cfg
        self._local = LocalStore()
        self._s3: Optional[S3Store] = None
        if cfg.s3_enabled:
            self._s3 = S3Store(
                endpoint_url=cfg.s3_endpoint,
                key=cfg.s3_access_key,
                secret=cfg.s3_secret_key,
                secure=cfg.s3_secure,
            )

    # --- routing ---

    def _backend(self, path: str) -> BlobStore:
        if _is_s3_uri(path):
            if not self._s3:
                raise RuntimeError("S3 access requested but not enabled/configured.")
            return self._s3
        return self._local

    # --- direct passthrough ---

    def exists(self, path: str) -> bool:
        return self._backend(path).exists(path)

    def ls(self, prefix: str) -> List[str]:
        return self._backend(prefix).ls(prefix)

    def read_bytes(self, path: str) -> bytes:
        return self._backend(path).read_bytes(path)

    def write_bytes(self, path: str, data: bytes) -> None:
        self._backend(path).write_bytes(path, data)

    def open(self, path: str, mode: str = "rb"):
        return self._backend(path).open(path, mode)

    # --- path helpers bound to config roots ---

    def raw_path(self, *parts: str, s3: bool = False) -> str:
        if s3:
            if not self.cfg.s3_enabled or not self.cfg.s3_bucket:
                raise RuntimeError("S3 raw root requested but S3 is disabled.")
            prefix = self.cfg.s3_raw_prefix or ""
            key = "/".join([prefix.strip("/"), *[p.strip("/") for p in parts if p]]).strip("/")
            return f"s3://{self.cfg.s3_bucket}/{key}"
        base = Path(self.cfg.local_raw_root)
        return str(base.joinpath(*parts))

    def gold_path(self, *parts: str, s3: bool = False) -> str:
        if s3:
            if not self.cfg.s3_enabled or not self.cfg.s3_bucket:
                raise RuntimeError("S3 gold root requested but S3 is disabled.")
            prefix = self.cfg.s3_gold_prefix or ""
            key = "/".join([prefix.strip("/"), *[p.strip("/") for p in parts if p]]).strip("/")
            return f"s3://{self.cfg.s3_bucket}/{key}"
        base = Path(self.cfg.local_gold_root)
        return str(base.joinpath(*parts))

# -------- Factory from your RootCfg --------

def build_storage_from_config(root_cfg) -> Storage:
    # root_cfg is src.config_model.model.RootCfg
    loc = root_cfg.storage.local
    s3 = root_cfg.storage.minio
    cfg = StoreConfig(
        local_raw_root=loc.raw_root,
        local_gold_root=loc.gold_root,
        s3_enabled=bool(s3.enabled),
        s3_endpoint=s3.endpoint if s3.enabled else None,
        s3_bucket=s3.bucket if s3.enabled else None,
        s3_access_key=s3.access_key if s3.enabled else None,
        s3_secret_key=s3.secret_key if s3.enabled else None,
        s3_secure=bool(s3.secure) if s3.enabled else False,
        s3_raw_prefix=s3.raw_prefix if s3.enabled else None,
        s3_gold_prefix=s3.gold_prefix if s3.enabled else None,
    )
    return Storage(cfg)
