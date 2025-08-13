from __future__ import annotations
import hashlib, json, re
from typing import Any, Iterable, Mapping, Sequence
from slugify import slugify as _slugify

def slugify(s: str) -> str:
    return _slugify(s, lowercase=True, separator="-")

def _json_dumps_stable(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))

def stable_hash(obj: Any, algo: str = "sha256") -> str:
    h = hashlib.new(algo)
    if hasattr(obj, "to_dict"):
        obj = obj.to_dict()
    h.update(_json_dumps_stable(obj).encode("utf-8"))
    return h.hexdigest()

def short_id(obj: Any, n: int = 10) -> str:
    return stable_hash(obj)[:n]

def make_dataset_slug(uri_or_name: str) -> str:
    base = re.sub(r"[\\/]+", "-", uri_or_name.strip())
    return slugify(base)

def make_chart_id(topic_key: Mapping[str, Any], chart_kind: str) -> str:
    # topic_key should include dataset_slug, roles, fields, filters, time range…
    payload = {"topic": topic_key, "chart": chart_kind}
    return f"{chart_kind}-{short_id(payload)}"