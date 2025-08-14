from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any

@dataclass(frozen=True)
class RoleConfidence:
    role: str
    confidence: float

@dataclass(frozen=True)
class ColumnHints:
    unit_hint: Optional[str] = None
    canonical_map: Optional[Dict[str, str]] = None  # for categories/text normalization
    domain_guess: Optional[str] = None

@dataclass(frozen=True)
class ColumnSchema:
    name: str
    dtype: str
    role_confidence: RoleConfidence
    hints: ColumnHints

@dataclass(frozen=True)
class ProposedSchema:
    dataset_slug: str
    columns: List[ColumnSchema]
    schema_confidence: float
    version: int = 1
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset_slug,
            "version": self.version,
            "schema_confidence": self.schema_confidence,
            "columns": [
                {
                    "name": c.name,
                    "dtype": c.dtype,
                    "role": c.role_confidence.role,
                    "role_confidence": c.role_confidence.confidence,
                    "hints": {
                        "unit_hint": c.hints.unit_hint,
                        "domain_guess": c.hints.domain_guess,
                        "canonical_map": c.hints.canonical_map or {},
                    },
                }
                for c in self.columns
            ],
            "notes": self.notes or "",
        }
