from __future__ import annotations
import pytest

from src.cleaning.registry import compile_actions_registry, parse_then, NameRef

def test_impute_positional_maps_to_method_and_returns_action():
    reg = compile_actions_registry()
    action, params = parse_then('impute("median")', reg)
    assert callable(action)
    assert params == {"method": "median"}

def test_parse_datetime_symbolic_list_ref():
    reg = compile_actions_registry()
    action, params = parse_then("parse_datetime(datetime_formats)", reg)
    assert callable(action)
    assert isinstance(params.get("formats"), NameRef)
    assert params["formats"].path == "datetime_formats"

def test_parse_datetime_list_literal():
    reg = compile_actions_registry()
    action, params = parse_then('parse_datetime(["%Y-%m-%d","%m/%d/%Y"])', reg)
    assert callable(action)
    assert params == {"formats": ["%Y-%m-%d", "%m/%d/%Y"]}

def test_text_normalize_kw_with_nested_symbol_refs():
    reg = compile_actions_registry()
    spec = "text_normalize(strip=cleaning.normalize.strip_text, lower=cleaning.normalize.lowercase_text)"
    action, params = parse_then(spec, reg)
    assert isinstance(params["strip"], NameRef) and params["strip"].path == "cleaning.normalize.strip_text"
    assert isinstance(params["lower"], NameRef) and params["lower"].path == "cleaning.normalize.lowercase_text"

def test_outliers_all_symbolic_params_order_and_names():
    reg = compile_actions_registry()
    action, params = parse_then("outliers(detect, zscore_threshold, iqr_multiplier, handle, winsor_limits)", reg)
    assert isinstance(params["method"], NameRef) and params["method"].path == "detect"
    assert isinstance(params["zscore_threshold"], NameRef) and params["zscore_threshold"].path == "zscore_threshold"
    assert isinstance(params["iqr_multiplier"], NameRef) and params["iqr_multiplier"].path == "iqr_multiplier"
    assert isinstance(params["handle"], NameRef) and params["handle"].path == "handle"
    assert isinstance(params["winsor_limits"], NameRef) and params["winsor_limits"].path == "winsor_limits"

def test_kwargs_override_positional_when_conflict():
    reg = compile_actions_registry()
    action, params = parse_then('coerce_numeric("percent", unit_hint="currency")', reg)
    assert params["unit_hint"] == "currency"

def test_bools_numbers_and_none_literals():
    reg = compile_actions_registry()
    action, params = parse_then("text_normalize(strip=True, lower=False)", reg)
    assert params == {"strip": True, "lower": False}

def test_unknown_action_raises():
    reg = compile_actions_registry()
    with pytest.raises(ValueError):
        parse_then("unknown_action(1,2,3)", reg)

def test_bad_syntax_raises():
    reg = compile_actions_registry()
    with pytest.raises(ValueError):
        parse_then("impute(", reg)
