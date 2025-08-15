from __future__ import annotations
from typing import Any, Callable, Dict
import pandas as pd
from dataclasses import dataclass
import re

# -------- public types --------

@dataclass(frozen=True)
class NameRef:
    """Symbolic reference that should be resolved from ctx['env'] later."""
    path: str

# Each action gets (series, ctx) and returns a new series (and optional notes).
ActionFn = Callable[[pd.Series, dict[str, Any]], tuple[pd.Series, str | None] | pd.Series]


# -------- registry --------

def compile_actions_registry() -> dict[str, ActionFn]:
    """
    Map action names -> callables.
    Add your plug-ins here.
    """
    from .rules_builtin.types import (
        coerce_numeric_from_string,
        parse_datetime_from_string,
        cast_category_if_small,
        cast_string_dtype,
    )
    from .rules_builtin.missing import impute_numeric, impute_value, impute_datetime
    from .rules_builtin.outliers import apply_outlier_policy  # returns (series, mask)
    from .rules_builtin.text_norm import text_normalize, normalize_null_tokens
    from .rules_builtin.units import standardize_numeric_units  # returns (series, meta)

    # Wrap primitives into (s, ctx) -> (s2, notes)
    def _wrap(fn, note_fmt: str) -> ActionFn:
        def inner(s: pd.Series, ctx: dict[str, Any]):
            s2 = fn(s, **ctx.get("params", {}))
            return s2, note_fmt
        return inner

    # outliers: function returns (series, mask) → build a helpful note
    def _wrap_outliers(fn) -> ActionFn:
        def inner(s: pd.Series, ctx: dict[str, Any]):
            s2, mask = fn(s, **ctx.get("params", {}))
            n_flagged = 0
            try:
                n_flagged = int(getattr(mask, "sum", lambda: 0)())
            except Exception:
                pass
            return s2, f"outliers(n={n_flagged})"
        return inner

    # units: function returns (series, meta) → summarize meta in note
    def _wrap_units(fn) -> ActionFn:
        def inner(s: pd.Series, ctx: dict[str, Any]):
            s2, meta = fn(s, **ctx.get("params", {}))  # (series, dict)
            unit_in  = str(meta.get("unit_in", ""))
            unit_out = str(meta.get("unit_out", ""))
            rescaled = ",rescaled" if meta.get("rescaled") else ""
            return s2, f"standardize_units({unit_in}->{unit_out}{rescaled})"
        return inner

    registry: dict[str, ActionFn] = {
        "coerce_numeric":   _wrap(coerce_numeric_from_string, "coerce_numeric"),
        "parse_datetime":   _wrap(parse_datetime_from_string, "parse_datetime"),
        "cast_category":    _wrap(cast_category_if_small, "cast_category"),
        "impute":           _wrap(impute_numeric, "impute"),
        "impute_value":     _wrap(impute_value, "impute_value"),
        "materialize_missing_as": _wrap( 
            lambda s, **p: impute_value(s, **{**p, "force": True}),
            "materialize_missing_as"
        ),
        "impute_dt":        _wrap(impute_datetime, "impute_dt"),
        "cast_string": _wrap(cast_string_dtype, "cast_string"),
        "outliers":         _wrap_outliers(apply_outlier_policy),
        "text_normalize":   _wrap(text_normalize, "text_normalize"),
        "normalize_null_tokens": _wrap(normalize_null_tokens, "normalize_null_tokens"),
        "standardize_units": _wrap_units(standardize_numeric_units),
        "drop_column":      lambda s, ctx: (pd.Series(dtype="float64"), "drop_column"),
    }
    return registry


# -------- parse_then (safe, no eval) --------

def parse_then(spec: str, registry: dict[str, ActionFn]) -> tuple[ActionFn, dict[str, Any]]:
    """
    Parse a 'then' string like:
      'impute("median")'
      'parse_datetime(datetime_formats)'
      'parse_datetime(["%Y-%m-%d","%m/%d/%Y"])'
      'text_normalize(strip=True, lower=cleaning.normalize.lowercase_text)'
      'outliers(detect, zscore_threshold, iqr_multiplier, handle, winsor_limits)'

    Returns: (callable, params_dict)
    - Strings, numbers, booleans, None, list literals are parsed as literals
    - Dotted identifiers become NameRef(path) for later resolution via ctx['env']
    """

    # ---- Tokenizer ----
    class Tok:
        __slots__ = ("typ", "val", "pos")
        def __init__(self, typ: str, val: Any, pos: int) -> None:
            self.typ, self.val, self.pos = typ, val, pos
        def __repr__(self) -> str:
            return f"Tok({self.typ!r},{self.val!r}@{self.pos})"

    WHITESPACE = set(" \t\r\n")
    DIG = set("0123456789")
    ID0 = set("_abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
    IDN = ID0.union(DIG).union({"."})

    def _read_while(s: str, i: int, pred) -> tuple[str, int]:
        j = i
        n = len(s)
        while j < n and pred(s[j]):
            j += 1
        return s[i:j], j

    def _read_string(s: str, i: int) -> tuple[str, int]:
        quote = s[i]
        i += 1
        out = []
        n = len(s)
        esc = False
        while i < n:
            ch = s[i]; i += 1
            if esc:
                if ch in ['\\', '"', "'"]:
                    out.append(ch)
                elif ch == "n":
                    out.append("\n")
                elif ch == "t":
                    out.append("\t")
                elif ch == "r":
                    out.append("\r")
                else:
                    out.append(ch)
                esc = False
            else:
                if ch == "\\":
                    esc = True
                elif ch == quote:
                    return "".join(out), i
                else:
                    out.append(ch)
        raise ValueError("Unterminated string literal")

    def _read_number(s: str, i: int) -> tuple[int | float, int]:
        j = i
        n = len(s)
        if j < n and s[j] == "-":
            j += 1
        ints, j = _read_while(s, j, lambda c: c in DIG)
        if ints == "":
            raise ValueError(f"Invalid number at {i}")
        if j < n and s[j] == ".":
            j += 1
            frac, j = _read_while(s, j, lambda c: c in DIG)
            if frac == "":
                raise ValueError(f"Invalid float at {i}")
            return float(s[i:j]), j
        return int(s[i:j]), j

    def _tokenize(text: str) -> list[Tok]:
        s = text
        i, n = 0, len(s)
        toks: list[Tok] = []
        while i < n:
            ch = s[i]
            if ch in WHITESPACE:
                i += 1; continue
            if ch == "(":
                toks.append(Tok("LP", "(", i)); i += 1; continue
            if ch == ")":
                toks.append(Tok("RP", ")", i)); i += 1; continue
            if ch == ",":
                toks.append(Tok("COMMA", ",", i)); i += 1; continue
            if ch == "=":
                toks.append(Tok("EQ", "=", i)); i += 1; continue
            if ch == "[":
                toks.append(Tok("LB", "[", i)); i += 1; continue
            if ch == "]":
                toks.append(Tok("RB", "]", i)); i += 1; continue

            if ch in ("'", '"'):
                val, j = _read_string(s, i)
                toks.append(Tok("STR", val, i)); i = j; continue

            if ch in DIG or (ch == "-" and i + 1 < n and s[i + 1] in DIG):
                val, j = _read_number(s, i)
                toks.append(Tok("NUM", val, i)); i = j; continue

            if ch in ID0:
                raw, j = _read_while(s, i, lambda c: c in IDN)
                low = raw.lower()
                if low in ("true", "false"):
                    toks.append(Tok("BOOL", True if low == "true" else False, i))
                elif low in ("none", "null"):
                    toks.append(Tok("NONE", None, i))
                else:
                    toks.append(Tok("ID", raw, i))
                i = j; continue

            raise ValueError(f"Unexpected character {ch!r} at position {i}")
        toks.append(Tok("EOF", None, n))
        return toks

    # ---- Parser ----
    class P:
        def __init__(self, toks: list[Tok]) -> None:
            self.t = toks; self.i = 0
        def peek(self) -> Tok:
            return self.t[self.i]
        def eat(self, typ: str | None = None) -> Tok:
            tok = self.peek()
            if typ and tok.typ != typ:
                raise ValueError(f"Expected {typ}, got {tok.typ} at {tok.pos}")
            self.i += 1
            return tok

        def parse(self) -> tuple[str, list[Any], dict[str, Any]]:
            # func '(' args? ')'
            name_tok = self.eat("ID")
            self.eat("LP")
            pos_args: list[Any] = []
            kw_args: dict[str, Any] = {}
            if self.peek().typ != "RP":
                while True:
                    nxt = self.peek()
                    if nxt.typ == "ID":
                        # look-ahead for '=' to decide kw vs positional
                        save_i = self.i
                        key_tok = self.eat("ID")
                        if self.peek().typ == "EQ":
                            self.eat("EQ")
                            val = self.parse_value()
                            kw_args[key_tok.val] = val
                        else:
                            self.i = save_i
                            val = self.parse_value()
                            pos_args.append(val)
                    else:
                        val = self.parse_value()
                        pos_args.append(val)
                    if self.peek().typ == "COMMA":
                        self.eat("COMMA"); continue
                    break
            self.eat("RP")
            return name_tok.val, pos_args, kw_args

        def parse_value(self) -> Any:
            tok = self.peek()
            if tok.typ == "STR":
                return self.eat("STR").val
            if tok.typ == "NUM":
                return self.eat("NUM").val
            if tok.typ == "BOOL":
                return self.eat("BOOL").val
            if tok.typ == "NONE":
                self.eat("NONE"); return None
            if tok.typ == "LB":
                return self.parse_list()
            if tok.typ == "ID":
                # dotted identifier => NameRef
                return NameRef(self.eat("ID").val)
            raise ValueError(f"Unexpected token {tok} in value")

        def parse_list(self) -> list[Any]:
            self.eat("LB")
            items: list[Any] = []
            if self.peek().typ != "RB":
                while True:
                    items.append(self.parse_value())
                    if self.peek().typ == "COMMA":
                        self.eat("COMMA"); continue
                    break
            self.eat("RB")
            return items

    # positional param names for known actions
    ACTION_POS_PARAMS: dict[str, list[str]] = {
        "coerce_numeric":    ["unit_hint"],
        "parse_datetime":    ["formats"],
        "cast_category":     ["max_card"],
        "impute":            ["method"],
        "impute_value":      ["value"],
         "materialize_missing_as": ["value"],
        "cast_string":              [],
        "impute_dt": ["method", "value"],
        "outliers":          ["method", "zscore_threshold", "iqr_multiplier", "handle", "winsor_limits"],
        "text_normalize":    ["strip", "lower"],
        "normalize_null_tokens": ["null_tokens", "case_insensitive", "apply_text_normalize_first"], 
        "standardize_units": ["unit_hint"],
        "drop_column":       [],
    }

    toks = _tokenize(spec)
    parser = P(toks)
    action_name, pos_args, kw_args = parser.parse()

    if action_name not in registry:
        raise ValueError(f"Unknown action '{action_name}' in spec: {spec!r}")

    # map positional → names
    names = ACTION_POS_PARAMS.get(action_name, [])
    if len(pos_args) > len(names):
        raise ValueError(
            f"Too many positional args for {action_name}: expected ≤ {len(names)}, got {len(pos_args)}"
        )

    params: dict[str, Any] = {}
    for i, a in enumerate(pos_args):
        if i < len(names):
            params[names[i]] = a
    params.update(kw_args)  # kwargs override

    return registry[action_name], params
