from __future__ import annotations
import types
import pytest
import pandas as pd

# Import the module so we can monkeypatch its local symbol `guess_role`
from src.cleaning import rescore as R


# --- helpers -----------------------------------------------------------------

def _prev_schema(columns: list[dict], schema_conf: float = 0.0):
    """
    Build a minimal ProposedSchema-like object with both attributes and to_dict().
    `columns` should be a list of dicts, each with at least {'name':..., 'role':...}
    and optionally {'confidence': ...}.
    """
    class _PS:
        def __init__(self, cols, sc):
            # attribute-style access
            self.columns = [types.SimpleNamespace(**c) for c in cols]
            self.schema_confidence = sc
            # keep original dicts for to_dict path
            self._cols = cols
            self._sc = sc

        def to_dict(self):
            return {"columns": self._cols, "schema_confidence": self._sc}

    return _PS(columns, schema_conf)


class _DummyRolesCfg:
    def __init__(self, fmts=None):
        class _Roles: pass
        self.roles = _Roles()
        self.roles.datetime_formats = list(fmts or ["%Y-%m-%d"])


class _DummyNLPCfg:
    pass


# --- tests -------------------------------------------------------------------

def test_rescore_basic_averages(monkeypatch):
    """
    Prev schema has per-column confidences; guess_role returns confidences for both.
    We expect:
      - avg_role_conf_before = mean(previous per-column confs)
      - avg_role_conf_after  = mean(new per-column confs)
      - schema_conf_after == avg_role_conf_after
    """
    df = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})

    prev = _prev_schema(
        [
            {"name": "a", "role": "numeric", "confidence": 0.6},
            {"name": "b", "role": "text",    "confidence": 0.4},
        ],
        schema_conf=0.5,
    )

    def stub_guess_role(name, s, nlp_cfg, dt_formats):
        if name == "a":
            return ("numeric", 0.9)
        return ("text", 0.3)

    monkeypatch.setattr(R, "guess_role", stub_guess_role)

    profiling = _DummyRolesCfg(["%Y-%m-%d", "%m/%d/%Y"])
    nlp_cfg = _DummyNLPCfg()

    out = R.rescore_after_clean(df, prev, profiling, nlp_cfg)

    assert out.schema_conf_before == pytest.approx(0.5)
    assert out.avg_role_conf_before == pytest.approx((0.6 + 0.4) / 2.0)
    assert out.avg_role_conf_after == pytest.approx((0.9 + 0.3) / 2.0)
    assert out.schema_conf_after == pytest.approx(out.avg_role_conf_after)

    assert out.per_column["a"]["before"] == pytest.approx(0.6)
    assert out.per_column["a"]["after"] == pytest.approx(0.9)
    assert out.per_column["b"]["before"] == pytest.approx(0.4)
    assert out.per_column["b"]["after"] == pytest.approx(0.3)


def test_missing_prev_conf_and_mixed_guess_role_shapes(monkeypatch):
    """
    When previous schema lacks per-column confidences, 'before' should be 0.0.
    Also exercise guess_role returning a dict and a plain string for different columns.
    Only non-None 'after' confidences should contribute to averages.
    """
    df = pd.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})

    prev = _prev_schema(
        [
            {"name": "x", "role": "numeric"},   # no confidence key
            {"name": "y", "role": "text"},      # no confidence key
        ],
        schema_conf=0.0,
    )

    def stub_guess_role(name, s, nlp_cfg, dt_formats):
        if name == "x":
            return {"role": "numeric", "confidence": 0.8}  # dict shape
        # return only a role string (no confidence)
        return "text"

    monkeypatch.setattr(R, "guess_role", stub_guess_role)

    profiling = _DummyRolesCfg()
    nlp_cfg = _DummyNLPCfg()

    out = R.rescore_after_clean(df, prev, profiling, nlp_cfg)

    # 'before' confidences were missing → treated as 0.0
    assert out.avg_role_conf_before == pytest.approx(0.0)

    # Only x has an 'after' confidence; average should be 0.8
    assert out.avg_role_conf_after == pytest.approx(0.8)
    assert out.schema_conf_after == pytest.approx(0.8)

    assert out.per_column["x"]["before"] == pytest.approx(0.0)
    assert out.per_column["x"]["after"] == pytest.approx(0.8)
    assert out.per_column["y"]["before"] == pytest.approx(0.0)
    assert out.per_column["y"]["after"] == pytest.approx(0.0)  # no conf → 0.0 saved in map
