from __future__ import annotations

"""
Topic modeling primitives for the reporting pipeline.

Design goals
------------
- Immutable & compact: dataclass(frozen=True, slots=True) to reduce memory use.
- Deterministic IDs: stable hashing over (family, fields) with normalization.
- Safe & fast validation: O(k) over small sequences; no dataframe deps here.
- Easy (de)serialization: helpers to go to/from dicts (JSON-ready).

This module intentionally avoids importing heavy libraries (numpy/pandas/etc).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import hashlib
import json


# ------------------------------- helpers ------------------------------------


def _as_tuple_str(seq: Optional[Iterable[Any]]) -> Tuple[str, ...]:
    """
    Convert any iterable to a tuple[str, ...], skipping Nones and trimming spaces.
    Dedupe while preserving order. O(k) time/space where k=len(seq).
    """
    if not seq:
        return tuple()
    out: List[str] = []
    seen = set()
    for x in seq:
        if x is None:
            continue
        s = str(x).strip()
        if not s:
            continue
        # preserve first occurrence
        if s not in seen:
            seen.add(s)
            out.append(s)
    return tuple(out)


def _clamp01(x: float) -> float:
    try:
        if x != x:  # NaN
            return 0.0
    except Exception:
        return 0.0
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)


def _is_jsonable(obj: Any) -> bool:
    try:
        json.dumps(obj)
        return True
    except Exception:
        return False


def _compact_json(obj: Mapping[str, Any]) -> str:
    """
    Compact JSON (sorted keys, no spaces) for stable downstream storage.
    """
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def topic_id(family: str, fields: Sequence[str]) -> str:
    """
    Stable ID constructed from the lowercase family and the provided fields
    as-is (order matters). We do not sort fields so that the caller can encode
    semantic ordering (e.g., time then measure).

    Output: '{family}-{hash10}'
    """
    fam = str(family or "").strip().lower()
    flds = _as_tuple_str(fields)
    base = f"{fam}:" + "|".join(flds)
    digest = hashlib.sha1(base.encode("utf-8")).hexdigest()[:10]
    return f"{fam}-{digest}"


# ------------------------------- dataclass -----------------------------------


@dataclass(frozen=True, slots=True)
class TopicRow:
    """
    A single candidate/topic the system might render.

    Fields
    ------
    topic_id : str
        Stable identifier (see topic_id()).
    family : str
        Broad visualization/analysis family (e.g., 'kpi', 'trend', 'correlation').
    primary_fields : Tuple[str, ...]
        Columns (or derived features) that are primary in the visualization.
    secondary_fields : Tuple[str, ...]
        Auxiliary columns/features (e.g., treatment, grouping, time where not primary).
    time_field : Optional[str]
        Name of the time column when relevant to the topic.
    n_obs : int
        Number of observations that support the topic (post-filter/aggregation).
    coverage_pct : float
        Fraction (0..1) of rows that contributed data to the topic.
    effect_size : float
        Magnitude measure used for ranking within-family (scale depends on family).
    effect_detail : Dict[str, Any]
        Arbitrary structured metadata (e.g., {'r_pearson': 0.43}).
    significance : Dict[str, Any]
        Significance metadata (e.g., {'p_value': 0.03, 'test': 'chi2'}).
    causal_design : Optional[str]
        If applicable: 'ab', 'did', 'psm', 'rdd', 'granger', etc.
    assumptions_met : Optional[Sequence[str]]
        Heuristic checklist for causal topics (stored as a tuple).
    readability : float
        Heuristic (0..1) for how readable the chart will be.
    complexity_penalty : float
        Penalty (>=0) for complex topics (timelines with many points, etc).
    proposed_charts : Sequence[str]
        Ordered list of chart types suitable for this topic.

    Performance
    -----------
    Construction is O(k) in the combined length of sequences; memory is O(k).
    """

    topic_id: str
    family: str
    primary_fields: Tuple[str, ...] = field(default_factory=tuple)
    secondary_fields: Tuple[str, ...] = field(default_factory=tuple)
    time_field: Optional[str] = None
    n_obs: int = 0
    coverage_pct: float = 0.0
    effect_size: float = 0.0
    effect_detail: Dict[str, Any] = field(default_factory=dict)
    significance: Dict[str, Any] = field(default_factory=dict)
    causal_design: Optional[str] = None
    assumptions_met: Optional[Tuple[str, ...]] = None
    readability: float = 1.0
    complexity_penalty: float = 0.0
    proposed_charts: Tuple[str, ...] = field(default_factory=tuple)

    # ------------------------------ validation ------------------------------

    def __post_init__(self) -> None:
        # Normalize family
        fam = (self.family or "").strip().lower()

        # Normalize sequences
        pf = _as_tuple_str(self.primary_fields)
        sf = _as_tuple_str(self.secondary_fields)
        charts = _as_tuple_str(self.proposed_charts)
        assumptions: Optional[Tuple[str, ...]] = (
            _as_tuple_str(self.assumptions_met) if self.assumptions_met else None
        )

        # Normalize scalar fields
        time_f = (self.time_field or "").strip() if self.time_field else None
        cov = _clamp01(float(self.coverage_pct))
        read = _clamp01(float(self.readability))
        comp = float(self.complexity_penalty)
        if comp < 0.0:
            comp = 0.0

        # Validate dicts are JSON-able (avoid surprises downstream)
        eff = dict(self.effect_detail or {})
        sig = dict(self.significance or {})
        if not _is_jsonable(eff):
            raise ValueError("effect_detail must be JSON-serializable")
        if not _is_jsonable(sig):
            raise ValueError("significance must be JSON-serializable")

        # Validate counts
        nobs = int(self.n_obs)
        if nobs < 0:
            nobs = 0

        # Validate topic_id shape (cheap check)
        tid = (self.topic_id or "").strip()
        if not tid:
            raise ValueError("topic_id cannot be empty")

        # Freeze normalized state
        object.__setattr__(self, "family", fam)
        object.__setattr__(self, "primary_fields", pf)
        object.__setattr__(self, "secondary_fields", sf)
        object.__setattr__(self, "proposed_charts", charts)
        object.__setattr__(self, "assumptions_met", assumptions)
        object.__setattr__(self, "time_field", time_f)
        object.__setattr__(self, "coverage_pct", cov)
        object.__setattr__(self, "readability", read)
        object.__setattr__(self, "complexity_penalty", comp)
        object.__setattr__(self, "effect_detail", eff)
        object.__setattr__(self, "significance", sig)
        object.__setattr__(self, "n_obs", nobs)

    # --------------------------- convenience API ----------------------------

    @property
    def key(self) -> Tuple[str, str, Tuple[str, ...], Tuple[str, ...]]:
        """
        Deduplication key: (family, topic_id, primary_fields, secondary_fields).
        Useful for set/dict membership. O(1).
        """
        return (self.family, self.topic_id, self.primary_fields, self.secondary_fields)

    def to_record(self, *, json_strings: bool = True) -> Dict[str, Any]:
        """
        Convert to a flat dict suitable for DataFrame rows or Parquet writing.

        Parameters
        ----------
        json_strings : bool
            If True, dumps effect_detail/significance as compact JSON strings.
            If False, leaves them as dicts.

        Returns
        -------
        Dict[str, Any]
        """
        eff = _compact_json(self.effect_detail) if json_strings else dict(self.effect_detail)
        sig = _compact_json(self.significance) if json_strings else dict(self.significance)
        return {
            "topic_id": self.topic_id,
            "family": self.family,
            "primary_fields": list(self.primary_fields),
            "secondary_fields": list(self.secondary_fields),
            "time_field": self.time_field,
            "n_obs": self.n_obs,
            "coverage_pct": self.coverage_pct,
            "effect_size": self.effect_size,
            "effect_detail": eff,
            "significance": sig,
            "causal_design": self.causal_design,
            "assumptions_met": list(self.assumptions_met) if self.assumptions_met else [],
            "readability": self.readability,
            "complexity_penalty": self.complexity_penalty,
            "proposed_charts": list(self.proposed_charts),
        }

    @staticmethod
    def from_record(rec: Mapping[str, Any]) -> TopicRow:
        """
        Inverse of to_record(json_strings=*) – accepts either dicts or JSON strings
        for effect_detail / significance, and lists/tuples for sequences.

        Parameters
        ----------
        rec : Mapping[str, Any]

        Returns
        -------
        TopicRow
        """
        def _parse_maybe_json(x: Any) -> Dict[str, Any]:
            if isinstance(x, str):
                try:
                    return json.loads(x)
                except Exception:
                    # if it's not valid JSON, store as string payload
                    return {"_raw": x}
            if isinstance(x, Mapping):
                return dict(x)
            return {}

        return TopicRow(
            topic_id=str(rec.get("topic_id", "")).strip(),
            family=str(rec.get("family", "")).strip().lower(),
            primary_fields=_as_tuple_str(rec.get("primary_fields") or ()),
            secondary_fields=_as_tuple_str(rec.get("secondary_fields") or ()),
            time_field=(str(rec.get("time_field")).strip() if rec.get("time_field") else None),
            n_obs=int(rec.get("n_obs", 0)),
            coverage_pct=float(rec.get("coverage_pct", 0.0)),
            effect_size=float(rec.get("effect_size", 0.0)),
            effect_detail=_parse_maybe_json(rec.get("effect_detail")),
            significance=_parse_maybe_json(rec.get("significance")),
            causal_design=(str(rec.get("causal_design")).strip() if rec.get("causal_design") else None),
            assumptions_met=_as_tuple_str(rec.get("assumptions_met") or ()),
            readability=float(rec.get("readability", 1.0)),
            complexity_penalty=float(rec.get("complexity_penalty", 0.0)),
            proposed_charts=_as_tuple_str(rec.get("proposed_charts") or ()),
        )


__all__ = [
    "TopicRow",
    "topic_id",
]
