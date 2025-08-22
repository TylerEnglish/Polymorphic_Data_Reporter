from __future__ import annotations

import json
import pytest

from src.topics.model import TopicRow, topic_id


def test_topic_id_deterministic_and_order_sensitive():
    # family normalized to lowercase
    a = topic_id("Trend", ["time", "metric"])
    b = topic_id("trend", ["time", "metric"])
    assert a == b
    # order matters
    c = topic_id("trend", ["metric", "time"])
    assert c != a
    # duplicates in fields are removed (first occurrence kept)
    d = topic_id("trend", ["time", "metric", "time"])
    assert d == a
    # shape: prefix + 10 hex chars
    assert a.startswith("trend-")
    assert len(a.split("-")[1]) == 10
    int(a.split("-")[1], 16)  # should be valid hex


def test_topicrow_normalization_and_validation():
    tid = topic_id("trend", ["time", "metric"])
    tr = TopicRow(
        topic_id=tid,
        family=" Trend ",
        primary_fields=(" time ", "metric", "time"),   # dup + whitespace
        secondary_fields=(None, " aux ", "aux"),       # None + dup + whitespace
        time_field="  time  ",
        n_obs=-5,                                      # will clamp to 0
        coverage_pct=1.23,                             # clamp to 1.0
        effect_size=0.42,
        effect_detail={"b": 2, "a": 1},
        significance={"test": "t", "p_value": 0.05},
        causal_design=None,
        assumptions_met=[" SUTVA ", "SUTVA"],          # dedupe + trim
        readability=-0.1,                              # clamp to 0
        complexity_penalty=-3.0,                       # floor at 0
        proposed_charts=[" line ", "line", "column"],  # dedupe + trim
    )

    # family lowercased/trimmed
    assert tr.family == "trend"
    # sequences normalized to tuples, deduped, trimmed, order preserved
    assert tr.primary_fields == ("time", "metric")
    assert tr.secondary_fields == ("aux",)
    assert tr.proposed_charts == ("line", "column")
    assert tr.assumptions_met == ("SUTVA",)
    # scalar normalizations
    assert tr.time_field == "time"
    assert tr.coverage_pct == 1.0
    assert tr.readability == 0.0
    assert tr.complexity_penalty == 0.0
    assert tr.n_obs == 0
    # dicts preserved
    assert tr.effect_detail == {"a": 1, "b": 2} or tr.effect_detail == {"b": 2, "a": 1}
    assert tr.significance["test"] == "t"
    assert tr.significance["p_value"] == 0.05
    # slots -> no __dict__
    assert not hasattr(tr, "__dict__")


def test_topicrow_invalid_topic_id():
    with pytest.raises(ValueError):
        TopicRow(
            topic_id="",
            family="kpi",
            primary_fields=("m",),
            secondary_fields=(),
            time_field=None,
            n_obs=1,
            coverage_pct=0.5,
            effect_size=0.0,
            effect_detail={},
            significance={},
            causal_design=None,
            assumptions_met=(),
            readability=1.0,
            complexity_penalty=0.0,
            proposed_charts=(),
        )


def test_topicrow_non_jsonable_effect_detail_raises():
    with pytest.raises(ValueError):
        TopicRow(
            topic_id=topic_id("kpi", ["m"]),
            family="kpi",
            primary_fields=("m",),
            secondary_fields=(),
            time_field=None,
            n_obs=10,
            coverage_pct=0.5,
            effect_size=0.0,
            effect_detail={"bad": {1, 2, 3}},  # sets are not JSON-serializable
            significance={},
            causal_design=None,
            assumptions_met=(),
            readability=1.0,
            complexity_penalty=0.0,
            proposed_charts=("kpi",),
        )


def test_to_record_json_strings_true():
    tr = TopicRow(
        topic_id=topic_id("correlation", ["x", "y"]),
        family="correlation",
        primary_fields=("x", "y"),
        secondary_fields=(),
        time_field=None,
        n_obs=100,
        coverage_pct=0.75,
        effect_size=0.3,
        effect_detail={"b": 2, "a": 1},
        significance={"p_value": 0.01, "test": "pearson"},
        causal_design=None,
        assumptions_met=(),
        readability=0.9,
        complexity_penalty=0.1,
        proposed_charts=("scatter", "xy_heatmap"),
    )
    rec = tr.to_record(json_strings=True)
    # JSON-compact (sorted keys, no spaces)
    assert isinstance(rec["effect_detail"], str)
    assert rec["effect_detail"] == '{"a":1,"b":2}'
    assert isinstance(rec["significance"], str)
    # Order not guaranteed here because keys differ; compare via json
    assert json.loads(rec["significance"]) == {"p_value": 0.01, "test": "pearson"}
    # sequences serialized as lists
    assert rec["primary_fields"] == ["x", "y"]
    assert rec["proposed_charts"] == ["scatter", "xy_heatmap"]


def test_to_record_json_strings_false():
    tr = TopicRow(
        topic_id=topic_id("kpi", ["m"]),
        family="kpi",
        primary_fields=("m",),
        secondary_fields=(),
        time_field=None,
        n_obs=5,
        coverage_pct=0.5,
        effect_size=0.0,
        effect_detail={"a": 1},
        significance={"s": 2},
        causal_design=None,
        assumptions_met=(),
        readability=1.0,
        complexity_penalty=0.0,
        proposed_charts=("kpi",),
    )
    rec = tr.to_record(json_strings=False)
    assert isinstance(rec["effect_detail"], dict)
    assert rec["effect_detail"] == {"a": 1}
    assert isinstance(rec["significance"], dict)
    assert rec["significance"] == {"s": 2}


def test_from_record_parses_json_and_normalizes_sequences():
    rec = {
        "topic_id": topic_id("trend", ["time", "m"]),
        "family": "Trend",
        "primary_fields": [" time ", "m", "time"],
        "secondary_fields": ["aux", "aux"],
        "time_field": " time ",
        "n_obs": 10,
        "coverage_pct": 0.8,
        "effect_size": 0.12,
        "effect_detail": '{"b":2,"a":1}',
        "significance": '{"test":"t","p_value":0.04}',
        "causal_design": None,
        "assumptions_met": [" A ", "A"],
        "readability": 0.7,
        "complexity_penalty": 0.0,
        "proposed_charts": [" line ", "line", "column"],
    }
    tr = TopicRow.from_record(rec)
    # family normalized
    assert tr.family == "trend"
    # sequences trimmed/tuple/dedup
    assert tr.primary_fields == ("time", "m")
    assert tr.secondary_fields == ("aux",)
    assert tr.proposed_charts == ("line", "column")
    assert tr.assumptions_met == ("A",)
    # time field trimmed
    assert tr.time_field == "time"
    # dicts parsed from JSON
    assert tr.effect_detail == {"a": 1, "b": 2} or tr.effect_detail == {"b": 2, "a": 1}
    assert tr.significance == {"test": "t", "p_value": 0.04}


def test_from_record_with_raw_string_effect_detail():
    rec = {
        "topic_id": topic_id("kpi", ["m"]),
        "family": "kpi",
        "primary_fields": ["m"],
        "secondary_fields": [],
        "time_field": None,
        "n_obs": 1,
        "coverage_pct": 0.1,
        "effect_size": 0.0,
        "effect_detail": "not json",
        "significance": {},
        "causal_design": None,
        "assumptions_met": [],
        "readability": 1.0,
        "complexity_penalty": 0.0,
        "proposed_charts": ["kpi"],
    }
    tr = TopicRow.from_record(rec)
    assert tr.effect_detail == {"_raw": "not json"}


def test_key_property_matches_same_fields():
    tid = topic_id("distribution", ["m"])
    a = TopicRow(
        topic_id=tid,
        family="distribution",
        primary_fields=("m",),
        secondary_fields=(),
        time_field=None,
        n_obs=100,
        coverage_pct=1.0,
        effect_size=0.5,
        effect_detail={},
        significance={},
        causal_design=None,
        assumptions_met=(),
        readability=1.0,
        complexity_penalty=0.0,
        proposed_charts=("histogram",),
    )
    b = TopicRow(
        topic_id=tid,
        family="distribution",
        primary_fields=("m",),
        secondary_fields=(),
        time_field=None,
        n_obs=50,
        coverage_pct=0.5,
        effect_size=0.1,
        effect_detail={},
        significance={},
        causal_design=None,
        assumptions_met=(),
        readability=0.8,
        complexity_penalty=0.0,
        proposed_charts=("histogram",),
    )
    assert a.key == b.key
