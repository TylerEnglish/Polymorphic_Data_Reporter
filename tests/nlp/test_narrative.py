from __future__ import annotations
from src.nlg.narrative import summarize_inventory, build_narrative, narrative_payload
from src.nlg.narrative_constants import INVENTORY_KEY

def _entries():
    return [
        {"slug":"sales_demo","uri":"/x/a.csv","kind":"csv"},
        {"slug":"sales_demo","uri":"/x/b.json","kind":"json"},
        {"slug":"iot","uri":"/x/c.parquet","kind":"parquet"},
    ]

def test_summarize_inventory_counts_and_sorting():
    inv = summarize_inventory(_entries())
    assert inv["total_files"] == 3
    # sorted by key in function
    assert list(inv["by_kind"].keys()) == sorted(inv["by_kind"].keys())
    assert inv["by_kind"]["csv"] == 1
    assert inv["by_slug"]["sales_demo"] == 2

def test_build_narrative_text_includes_core_sections():
    bootstrap = {
        "row_count": 10,
        "column_count": 2,
        "schema_confidence": 0.876,
        "columns": [
            {"name":"amount","dtype":"float","role":"numeric","role_confidence":0.9,"hints":{"unit_hint":"currency","domain_guess":None,"canonical_map_size":1}},
            {"name":"when","dtype":"datetime","role":"time","role_confidence":0.99,"hints":{"unit_hint":None,"domain_guess":"finance","canonical_map_size":0}},
        ],
    }
    inv_summary = summarize_inventory(_entries())
    txt = build_narrative("sales_demo", bootstrap, inv_summary)
    assert "# Dataset: sales_demo" in txt
    assert "Files discovered" in txt
    assert "Columns" in txt
    assert "amount" in txt and "when" in txt
    assert "unit=currency" in txt
    assert "domain=finance" in txt

def test_narrative_payload_bundle():
    bootstrap = {"schema_confidence": 0.5, "row_count": 0, "column_count": 0, "columns": []}
    entries = _entries()
    out = narrative_payload("ds", bootstrap, entries)
    assert out["dataset"] == "ds"
    assert out["schema_confidence"] == 0.5
    assert INVENTORY_KEY in out
    assert isinstance(out["narrative_text"], str) and len(out["narrative_text"]) > 0
