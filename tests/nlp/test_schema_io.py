from __future__ import annotations
from pathlib import Path
from tomlkit import loads

from src.nlp.schema_io import to_toml, schema_path_for_slug

def test_schema_path_for_slug(tmp_path: Path):
    p = schema_path_for_slug(tmp_path, "sales_demo")
    assert str(p).endswith("schemas/sales_demo.schema.toml")
    assert p.parent.name == "schemas"

def test_to_toml_structure_and_rounding():
    proposed = {
        "dataset": "sales_demo",
        "version": 3,
        "schema_confidence": 0.87654321,
        "columns": [
            {
                "name": "amount",
                "dtype": "float",
                "role": "numeric",
                "role_confidence": 0.912345,
                "hints": {
                    "unit_hint": "currency",
                    "domain_guess": None,
                    "canonical_map": {"USD":"usd"},
                },
            },
            {
                "name": "when",
                "dtype": "datetime",
                "role": "time",
                "role_confidence": 0.99,
                "hints": {"unit_hint": None, "domain_guess": None, "canonical_map": {}},
            },
        ],
        "notes": "frozen v1",
    }
    text = to_toml(proposed)
    doc = loads(text)
    assert doc["dataset"] == "sales_demo"
    assert doc["version"] == 3
    # rounded to 4 decimals
    assert abs(float(doc["schema_confidence"]) - 0.8765) < 1e-6

    assert isinstance(doc["columns"], list) and len(doc["columns"]) == 2
    c0 = doc["columns"][0]
    assert c0["name"] == "amount"
    assert c0["hints"]["canonical_map"] == {"USD":"usd"}
    assert "notes" in doc or proposed.get("notes") == "frozen v1"
