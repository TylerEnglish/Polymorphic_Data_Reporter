from __future__ import annotations
from typing import Any, Callable, List, Tuple, Optional, Sequence, Dict
from dataclasses import dataclass
import re

AllowedCallable = Callable[[dict[str, Any]], bool]

# =========================
# Tokenizer
# =========================

class Token:
    __slots__ = ("typ", "val", "pos")
    def __init__(self, typ: str, val: Any, pos: int) -> None:
        self.typ = typ
        self.val = val
        self.pos = pos
    def __repr__(self) -> str:
        return f"Token({self.typ!r}, {self.val!r}, pos={self.pos})"

_WHITESPACE = set(" \t\r\n")
_IDENT_START = set("_abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
_IDENT_CONT = _IDENT_START.union(set("0123456789"))
_DIGITS = set("0123456789")

def _is_ident_start(ch: str) -> bool:
    return ch in _IDENT_START

def _is_ident_cont(ch: str) -> bool:
    return ch in _IDENT_CONT or ch == "."

def _read_while(s: str, i: int, pred) -> Tuple[str, int]:
    j = i
    n = len(s)
    while j < n and pred(s[j]):
        j += 1
    return s[i:j], j

def _read_string(s: str, i: int) -> Tuple[str, int]:
    quote = s[i]
    i += 1
    out: List[str] = []
    n = len(s)
    esc = False
    while i < n:
        ch = s[i]
        i += 1
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

def _read_number(s: str, i: int) -> Tuple[float | int, int]:
    j = i
    n = len(s)
    if j < n and s[j] == "-":
        j += 1
    int_part, j = _read_while(s, j, lambda c: c in _DIGITS)
    if int_part == "":
        raise ValueError(f"Invalid number at {i}")
    # fractional?
    if j < n and s[j] == ".":
        j += 1
        frac, j = _read_while(s, j, lambda c: c in _DIGITS)
        if frac == "":
            raise ValueError(f"Invalid float at {i}")
        return float(s[i:j]), j
    return int(s[i:j]), j

def _tokenize(expr: str) -> List[Token]:
    s = expr
    i = 0
    n = len(s)
    toks: List[Token] = []
    while i < n:
        ch = s[i]
        if ch in _WHITESPACE:
            i += 1; continue

        # punctuation
        if ch == "(":
            toks.append(Token("LPAREN", "(", i)); i += 1; continue
        if ch == ")":
            toks.append(Token("RPAREN", ")", i)); i += 1; continue
        if ch == "[":
            toks.append(Token("LBRACK", "[", i)); i += 1; continue
        if ch == "]":
            toks.append(Token("RBRACK", "]", i)); i += 1; continue
        if ch == ",":
            toks.append(Token("COMMA", ",", i)); i += 1; continue

        # two-char ops
        if i + 1 < n:
            two = s[i:i+2]
            if two in ("==", "!=", "<=", ">="):
                toks.append(Token("OP", two, i)); i += 2; continue

        # single-char comp
        if ch in "<>":
            toks.append(Token("OP", ch, i)); i += 1; continue

        # strings
        if ch in ("'", '"'):
            val, j = _read_string(s, i)
            toks.append(Token("STR", val, i))
            i = j; continue

        # numbers
        if ch in _DIGITS or (ch == "-" and i+1 < n and s[i+1] in _DIGITS):
            val, j = _read_number(s, i)
            toks.append(Token("NUM", val, i))
            i = j; continue

        # identifiers/keywords
        if _is_ident_start(ch):
            raw, j = _read_while(s, i, _is_ident_cont)
            low = raw.lower()
            if low == "and": toks.append(Token("AND", "and", i))
            elif low == "or": toks.append(Token("OR", "or", i))
            elif low == "not": toks.append(Token("NOT", "not", i))
            elif low == "in": toks.append(Token("IN", "in", i))
            elif low == "notin": toks.append(Token("NOTIN", "notin", i))
            elif low in ("true","false"): toks.append(Token("BOOL", True if low=="true" else False, i))
            elif low in ("none","null"): toks.append(Token("NONE", None, i))
            else:
                toks.append(Token("ID", raw, i))
            i = j; continue

        raise ValueError(f"Unexpected character {ch!r} at position {i}")

    toks.append(Token("EOF", None, n))
    return toks

# =========================
# Parser (recursive descent)
# =========================

class Parser:
    def __init__(self, tokens: List[Token]) -> None:
        self.toks = tokens
        self.i = 0

    def _peek(self) -> Token:
        return self.toks[self.i]

    def _eat(self, typ: Optional[str] = None) -> Token:
        t = self._peek()
        if typ and t.typ != typ:
            raise ValueError(f"Expected {typ}, got {t.typ} at {t.pos}")
        self.i += 1
        return t

    def parse(self):
        node = self._parse_or()
        if self._peek().typ != "EOF":
            raise ValueError(f"Unexpected token {self._peek()}")
        return node

    # or_expr := and_expr ('OR' and_expr)*
    def _parse_or(self):
        left = self._parse_and()
        while self._peek().typ == "OR":
            self._eat("OR")
            right = self._parse_and()
            left = ("or", left, right)
        return left

    # and_expr := not_expr ('AND' not_expr)*
    def _parse_and(self):
        left = self._parse_not()
        while self._peek().typ == "AND":
            self._eat("AND"); right = self._parse_not()
            left = ("and", left, right)
        return left

    # not_expr := ('NOT' not_expr) | comparison
    def _parse_not(self):
        if self._peek().typ == "NOT":
            self._eat("NOT")
            node = self._parse_not()
            return ("not", node)
        return self._parse_comparison()

    # comparison := arith ( (OP|IN|NOT IN) arith )?
    def _parse_comparison(self):
        left = self._parse_arith()
        t = self._peek()
        if t.typ == "NOT":
            # 'NOT' 'IN'
            self._eat("NOT")
            if self._peek().typ != "IN":
                raise ValueError(f"Expected IN after NOT at {t.pos}")
            self._eat("IN")
            right = self._parse_arith()
            return ("cmp", "notin", left, right)
        
        if t.typ == "NOTIN":
            self._eat("NOTIN")
            right = self._parse_arith()
            return ("cmp", "notin", left, right)

        if t.typ in ("OP","IN"):
            op = self._eat().val
            right = self._parse_arith()
            return ("cmp", op, left, right)
        return left

    # For now, arith just forwards to primary (we don't need +,-,*,/ in conditions)
    def _parse_arith(self):
        return self._parse_primary()

    # primary := literal | identifier | funcall | list | '(' expr ')'
    def _parse_primary(self):
        t = self._peek()
        if t.typ == "LPAREN":
            self._eat("LPAREN")
            node = self._parse_or()
            self._eat("RPAREN")
            return node
        if t.typ in ("STR","NUM","BOOL","NONE"):
            self._eat(t.typ)
            return ("lit", t.val)
        if t.typ == "LBRACK":
            return self._parse_list()
        if t.typ == "ID":
            # identifier or funcall
            name = self._eat("ID").val
            if self._peek().typ == "LPAREN":
                # funcall
                self._eat("LPAREN")
                args: List[Any] = []
                if self._peek().typ != "RPAREN":
                    while True:
                        args.append(self._parse_or())
                        if self._peek().typ == "COMMA":
                            self._eat("COMMA"); continue
                        break
                self._eat("RPAREN")
                return ("call", name, args)
            return ("id", name)
        raise ValueError(f"Unexpected token {t} in primary")

    def _parse_list(self):
        # [expr, expr, ...]
        l_tok = self._eat("LBRACK")
        items: List[Any] = []
        if self._peek().typ != "RBRACK":
            while True:
                items.append(self._parse_or())
                if self._peek().typ == "COMMA":
                    self._eat("COMMA"); continue
                break
        self._eat("RBRACK")
        return ("list", items)

# =========================
# Safe evaluation
# =========================

def _resolve_identifier(path: str, ctx: dict[str, Any]) -> Any:
    parts = path.split(".")
    cur: Any = ctx.get(parts[0], None)
    for p in parts[1:]:
        if isinstance(cur, dict):
            cur = cur.get(p, None)
        else:
            return None
    return cur

def _to_seq(x: Any) -> Optional[Sequence]:
    if isinstance(x, (list, tuple, set)):
        return list(x)  # normalize set to list
    return None

def _safe_compare(op: str, a: Any, b: Any) -> bool:
    try:
        if op == "==": return a == b
        if op == "!=": return a != b
        if op == "<":  return a <  b
        if op == "<=": return a <= b
        if op == ">":  return a >  b
        if op == ">=": return a >= b
        if op == "in":
            # allow string containment as well
            if isinstance(b, str) and not isinstance(a, (list, tuple, set, dict)):
                return str(a) in b
            seq = _to_seq(b)
            return (a in seq) if seq is not None else False
        if op == "notin":
            if isinstance(b, str) and not isinstance(a, (list, tuple, set, dict)):
                return str(a) not in b
            seq = _to_seq(b)
            return (a not in seq) if seq is not None else True
    except Exception:
        return False
    return False

# Whitelisted safe functions
def _fn_len(x): 
    try: return 0 if x is None else len(x)  # type: ignore
    except Exception: return 0
def _fn_isnull(x): 
    try:
        import pandas as pd
        return bool(pd.isna(x))
    except Exception:
        return x is None
def _fn_notnull(x): 
    return not _fn_isnull(x)
def _fn_startswith(x, y): 
    return str(x).startswith(str(y)) if x is not None and y is not None else False
def _fn_endswith(x, y): 
    return str(x).endswith(str(y)) if x is not None and y is not None else False
def _fn_contains(x, y): 
    return str(y) in str(x) if x is not None and y is not None else False
def _fn_icontains(x, y): 
    return str(y).lower() in str(x).lower() if x is not None and y is not None else False
def _fn_matches(x, pattern):
    try: return bool(re.search(str(pattern), str(x))) if x is not None else False
    except re.error: return False
def _fn_imatches(x, pattern):
    try: return bool(re.search(str(pattern), str(x), flags=re.I)) if x is not None else False
    except re.error: return False

SAFE_FUNCS: Dict[str, Callable[..., Any]] = {
    "len": _fn_len,
    "isnull": _fn_isnull,
    "notnull": _fn_notnull,
    "startswith": _fn_startswith,
    "endswith": _fn_endswith,
    "contains": _fn_contains,
    "icontains": _fn_icontains,
    "matches": _fn_matches,
    "imatches": _fn_imatches,
}

def _collect_roots(ast) -> set[str]:
    typ = ast[0]
    if typ == "id":
        name: str = ast[1]
        return {name.split(".", 1)[0]}
    if typ in ("lit",):
        return set()
    if typ == "list":
        roots: set[str] = set()
        for item in ast[1]:
            roots |= _collect_roots(item)
        return roots
    if typ == "call":
        roots = set()
        for arg in ast[2]:
            roots |= _collect_roots(arg)
        return roots
    if typ == "not":
        return _collect_roots(ast[1])
    if typ in ("and", "or"):
        return _collect_roots(ast[1]) | _collect_roots(ast[2])
    if typ == "cmp":
        _, _op, l, r = ast
        return _collect_roots(l) | _collect_roots(r)
    return set()

def _compile(ast, allowed_funcs: Optional[set[str]]):
    typ = ast[0]

    if typ == "lit":
        val = ast[1]
        return lambda ctx: val

    if typ == "id":
        name = ast[1]
        return lambda ctx: _resolve_identifier(name, ctx)

    if typ == "list":
        items = [_compile(a, allowed_funcs) for a in ast[1]]
        return lambda ctx: [f(ctx) for f in items]

    if typ == "call":
        name = ast[1]
        if allowed_funcs is not None and name not in allowed_funcs:
            raise ValueError(f"Disallowed function: {name}")
        fn = SAFE_FUNCS.get(name)
        if fn is None:
            raise ValueError(f"Unknown function: {name}")
        arg_nodes = [_compile(a, allowed_funcs) for a in ast[2]]
        return lambda ctx: fn(*[g(ctx) for g in arg_nodes])

    if typ == "not":
        f = _compile(ast[1], allowed_funcs)
        return lambda ctx: (not bool(f(ctx)))

    if typ == "and":
        lf = _compile(ast[1], allowed_funcs); rf = _compile(ast[2], allowed_funcs)
        return lambda ctx: (bool(lf(ctx)) and bool(rf(ctx)))

    if typ == "or":
        lf = _compile(ast[1], allowed_funcs); rf = _compile(ast[2], allowed_funcs)
        return lambda ctx: (bool(lf(ctx)) or bool(rf(ctx)))

    if typ == "cmp":
        _, op, l, r = ast
        lf = _compile(l, allowed_funcs); rf = _compile(r, allowed_funcs)
        return lambda ctx: _safe_compare(op, lf(ctx), rf(ctx))

    raise ValueError(f"Unknown AST node: {ast!r}")

# =========================
# Public API
# =========================

def compile_condition(
    expr: str,
    allowed_vars: set[str] | None = None,
    *,
    allowed_funcs: set[str] | None = None,
) -> AllowedCallable:
    """
    Compile a safe callable(ctx)->bool from `expr`.

    - `allowed_vars`: restrict which *root* names may be referenced (e.g., {'name','role','missing_pct','cleaning'})
    - `allowed_funcs`: restrict callable names; defaults to a whitelist of SAFE_FUNCS

    Raises ValueError on syntax or safety violations.
    """
    tokens = _tokenize(expr)
    parser = Parser(tokens)
    ast = parser.parse()

    if allowed_vars is not None:
        roots = _collect_roots(ast)
        bad = [r for r in roots if r not in allowed_vars]
        if bad:
            raise ValueError(f"Use of disallowed identifiers: {bad}")

    if allowed_funcs is None:
        allowed_funcs = set(SAFE_FUNCS.keys())

    fn = _compile(ast, allowed_funcs)

    def _wrapped(ctx: dict[str, Any]) -> bool:
        try:
            return bool(fn(ctx))
        except Exception:
            # Any runtime type error becomes False, keeping engine robust.
            return False
    return _wrapped

def eval_condition(fn: AllowedCallable, ctx: dict[str, Any]) -> bool:
    return bool(fn(ctx))
