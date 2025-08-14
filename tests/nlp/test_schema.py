from __future__ import annotations
from src.nlp.schema import (
    RoleConfidence, ColumnHints, ColumnSchema, ProposedSchema
)

def test_proposed_schema_to_dict_roundtrip_minimal():
    rc = RoleConfidence(role="numeric", confidence=0.9)
    hints = ColumnHints(unit_hint="currency", canonical_map={"USD":"usd"}, domain_guess="finance")
    col = ColumnSchema(name="amount", dtype="float", role_confidence=rc, hints=hints)
    ps = ProposedSchema(dataset_slug="sales_demo", columns=[col], schema_confidence=0.88, version=2, notes="ok")

    d = ps.to_dict()
    assert d["dataset"] == "sales_demo"
    assert d["version"] == 2
    assert d["schema_confidence"] == 0.88
    assert d["notes"] == "ok"

    assert d["columns"][0]["name"] == "amount"
    assert d["columns"][0]["dtype"] == "float"
    assert d["columns"][0]["role"] == "numeric"
    assert d["columns"][0]["role_confidence"] == 0.9
    assert d["columns"][0]["hints"]["unit_hint"] == "currency"
    assert d["columns"][0]["hints"]["domain_guess"] == "finance"
    assert d["columns"][0]["hints"]["canonical_map"] == {"USD":"usd"}

def test_proposed_schema_to_dict_defaults_and_empty_map():
    rc = RoleConfidence(role="text", confidence=0.5)
    hints = ColumnHints()  # no values set
    col = ColumnSchema(name="comment", dtype="string", role_confidence=rc, hints=hints)
    ps = ProposedSchema(dataset_slug="ds", columns=[col], schema_confidence=0.0)

    d = ps.to_dict()
    assert d["version"] == 1
    assert d["notes"] == ""
    assert d["columns"][0]["hints"]["canonical_map"] == {}
