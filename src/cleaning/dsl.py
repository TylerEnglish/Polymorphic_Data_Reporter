from __future__ import annotations
from typing import Any, Callable, List, Tuple, Optional, Sequence, Dict, Iterable
import re
import unicodedata

# Public callable type: a compiled condition receives a context dict and returns bool
AllowedCallable = Callable[[dict[str, Any]], bool]

# =============================================================================
# Tokenizer
# =============================================================================

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
    # fractional
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

        # three-char ops first
        if i + 2 < n:
            tri = s[i:i+3]
            if tri in ("!~*",):  # regex not case-insensitive
                toks.append(Token("OP", tri, i)); i += 3; continue

        # two-char ops
        if i + 1 < n:
            two = s[i:i+2]
            if two in ("==", "!=", "<=", ">=", "=~", "!~", "~*"):
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
            elif low == "is": toks.append(Token("IS", "is", i))
            elif low == "like": toks.append(Token("LIKE", "like", i))
            elif low == "ilike": toks.append(Token("ILIKE", "ilike", i))
            elif low == "between": toks.append(Token("BETWEEN", "between", i))
            elif low in ("true","false"): toks.append(Token("BOOL", True if low=="true" else False, i))
            elif low in ("none","null"): toks.append(Token("NONE", None, i))
            else:
                toks.append(Token("ID", raw, i))
            i = j; continue

        raise ValueError(f"Unexpected character {ch!r} at position {i}")

    toks.append(Token("EOF", None, n))
    return toks

# =============================================================================
# Parser (recursive descent)
# =============================================================================

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

    # comparison := arith ( (OP|IN|NOT IN|IS [NOT]|LIKE|ILIKE|BETWEEN [AND]) arith ... )?
    def _parse_comparison(self):
        left = self._parse_arith()
        t = self._peek()

        # IS / IS NOT
        if t.typ == "IS":
            self._eat("IS")
            negate = False
            if self._peek().typ == "NOT":
                self._eat("NOT"); negate = True
            right = self._parse_arith()
            return ("cmp", "!=" if negate else "==", left, right)

        # NOT IN / NOT LIKE / NOT ILIKE / NOT BETWEEN
        if t.typ == "NOT":
            self._eat("NOT")
            nxt = self._peek()
            if nxt.typ == "IN":
                self._eat("IN")
                right = self._parse_arith()
                return ("cmp", "notin", left, right)
            if nxt.typ == "LIKE":
                self._eat("LIKE")
                right = self._parse_arith()
                return ("cmp", "notlike", left, right)
            if nxt.typ == "ILIKE":
                self._eat("ILIKE")
                right = self._parse_arith()
                return ("cmp", "notilike", left, right)
            if nxt.typ == "BETWEEN":
                self._eat("BETWEEN")
                low = self._parse_arith()
                self._eat("AND")
                high = self._parse_arith()
                return ("between", False, left, low, high)  # False -> NOT
            raise ValueError(f"Expected IN/LIKE/ILIKE/BETWEEN after NOT at {t.pos}")

        # NOTIN single-token
        if t.typ == "NOTIN":
            self._eat("NOTIN")
            right = self._parse_arith()
            return ("cmp", "notin", left, right)

        # BETWEEN / LIKE / ILIKE / IN / OP
        if t.typ == "BETWEEN":
            self._eat("BETWEEN")
            low = self._parse_arith()
            self._eat("AND")
            high = self._parse_arith()
            return ("between", True, left, low, high)  # True -> positive between

        if t.typ in ("LIKE","ILIKE","IN","OP"):
            op_tok = self._eat()
            right = self._parse_arith()
            return ("cmp", op_tok.val, left, right)

        return left

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
            name = self._eat("ID").val
            if self._peek().typ == "LPAREN":
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
        self._eat("LBRACK")
        items: List[Any] = []
        if self._peek().typ != "RBRACK":
            while True:
                items.append(self._parse_or())
                if self._peek().typ == "COMMA":
                    self._eat("COMMA"); continue
                break
        self._eat("RBRACK")
        return ("list", items)

# =============================================================================
# Safe evaluation helpers
# =============================================================================

def _resolve_identifier(path: str, ctx: dict[str, Any]) -> Any:
    """
    Resolve dotted identifiers from ctx.
    Supports dicts and lightweight attribute access on objects/dataclasses.
    """
    parts = path.split(".")
    cur: Any = ctx.get(parts[0], None)
    for p in parts[1:]:
        if isinstance(cur, dict):
            cur = cur.get(p, None)
        else:
            try:
                cur = getattr(cur, p)
            except Exception:
                return None
    return cur

def _try_import_pd_np():
    pd = np = None
    try:
        import pandas as _pd  # type: ignore
        pd = _pd
    except Exception:
        pass
    try:
        import numpy as _np  # type: ignore
        np = _np
    except Exception:
        pass
    return pd, np

def _iterable_of_values(x: Any) -> Optional[Iterable]:
    pd, np = _try_import_pd_np()
    if x is None:
        return None
    if isinstance(x, (list, tuple, set)):
        return list(x)
    if np is not None and isinstance(x, np.ndarray):  # type: ignore
        return x.tolist()
    if pd is not None and isinstance(x, pd.Series):  # type: ignore
        return x.tolist()
    return None

def _to_seq(x: Any) -> Optional[List]:
    it = _iterable_of_values(x)
    if it is None:
        return None
    return list(it)

def _as_str(x: Any) -> str:
    if x is None:
        return ""
    try:
        return str(x)
    except Exception:
        return ""

def _normalize_ascii(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")

def _wildcard_to_regex(pat: str) -> str:
    # SQL-style %,_ to regex; escaped pattern first
    esc = re.escape(pat)
    esc = esc.replace(r"\%", ".*").replace(r"\_", ".")
    return f"^{esc}$"

def _levenshtein(a: str, b: str) -> int:
    if a == b: return 0
    if not a: return len(b)
    if not b: return len(a)
    # DP with two rows
    prev = list(range(len(b) + 1))
    cur = [0] * (len(b) + 1)
    for i, ca in enumerate(a, 1):
        cur[0] = i
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur[j] = min(prev[j] + 1,       # deletion
                         cur[j-1] + 1,       # insertion
                         prev[j-1] + cost)   # substitution
        prev, cur = cur, prev
    return prev[-1]

def _similarity(a: str, b: str) -> float:
    a = _normalize_ascii(a or "").lower().strip()
    b = _normalize_ascii(b or "").lower().strip()
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    d = _levenshtein(a, b)
    m = max(len(a), len(b))
    return 1.0 - (d / float(m))

# =============================================================================
# Comparators (safe)
# =============================================================================

def _safe_compare(op: str, a: Any, b: Any) -> bool:
    try:
        if op == "==": return a == b
        if op == "!=": return a != b
        if op == "<":  return a <  b
        if op == "<=": return a <= b
        if op == ">":  return a >  b
        if op == ">=": return a >= b

        # Membership
        if op == "in":
            if isinstance(b, str) and not isinstance(a, (list, tuple, set, dict)):
                return _as_str(a) in b
            seq = _to_seq(b)
            return (a in seq) if seq is not None else False
        if op == "notin":
            if isinstance(b, str) and not isinstance(a, (list, tuple, set, dict)):
                return _as_str(a) not in b
            seq = _to_seq(b)
            return (a not in seq) if seq is not None else True

        # SQL-like
        if op == "like":
            rgx = _wildcard_to_regex(_as_str(b))
            return bool(re.match(rgx, _as_str(a)))
        if op == "ilike":
            rgx = _wildcard_to_regex(_as_str(b))
            return bool(re.match(rgx, _as_str(a), flags=re.I))
        if op == "notlike":
            rgx = _wildcard_to_regex(_as_str(b))
            return not bool(re.match(rgx, _as_str(a)))
        if op == "notilike":
            rgx = _wildcard_to_regex(_as_str(b))
            return not bool(re.match(rgx, _as_str(a), flags=re.I))

        # Regex operators
        if op == "=~":   # search
            return bool(re.search(_as_str(b), _as_str(a)))
        if op == "!~":
            return not bool(re.search(_as_str(b), _as_str(a)))
        if op == "~*":
            return bool(re.search(_as_str(b), _as_str(a), flags=re.I))
        if op == "!~*":
            return not bool(re.search(_as_str(b), _as_str(a), flags=re.I))
    except Exception:
        return False
    return False

# =============================================================================
# String / regex / normalization helpers
# =============================================================================

def _fn_len(x):
    try: return 0 if x is None else len(x)  # type: ignore
    except Exception: return 0

def _fn_isnull(x):
    try:
        import pandas as pd
        return bool(pd.isna(x))
    except Exception:
        return x is None

def _fn_notnull(x): return not _fn_isnull(x)

def _fn_lower(x): return _as_str(x).lower()
def _fn_upper(x): return _as_str(x).upper()
def _fn_strip(x): return _as_str(x).strip()
def _fn_lstrip(x): return _as_str(x).lstrip()
def _fn_rstrip(x): return _as_str(x).rstrip()
def _fn_norm(x):  return _normalize_ascii(_as_str(x))

def _fn_startswith(x, y): return _as_str(x).startswith(_as_str(y)) if x is not None and y is not None else False
def _fn_endswith(x, y):   return _as_str(x).endswith(_as_str(y))   if x is not None and y is not None else False
def _fn_istartswith(x, y):
    xs, ys = _as_str(x), _as_str(y); return xs.lower().startswith(ys.lower())
def _fn_iendswith(x, y):
    xs, ys = _as_str(x), _as_str(y); return xs.lower().endswith(ys.lower())
def _fn_contains(x, y):  return _as_str(y) in _as_str(x) if x is not None and y is not None else False
def _fn_icontains(x, y): return _as_str(y).lower() in _as_str(x).lower() if x is not None and y is not None else False
def _fn_iequals(x, y):   return _as_str(x).lower() == _as_str(y).lower()

def _fn_matches(x, pattern):
    try: return bool(re.search(_as_str(pattern), _as_str(x))) if x is not None else False
    except re.error: return False
def _fn_imatches(x, pattern):
    try: return bool(re.search(_as_str(pattern), _as_str(x), flags=re.I)) if x is not None else False
    except re.error: return False
def _fn_fullmatch(x, pattern):
    try: return bool(re.fullmatch(_as_str(pattern), _as_str(x))) if x is not None else False
    except re.error: return False
def _fn_ifullmatch(x, pattern):
    try: return bool(re.fullmatch(_as_str(pattern), _as_str(x), flags=re.I)) if x is not None else False
    except re.error: return False

def _fn_like(x, pat):   return _safe_compare("like", _as_str(x), _as_str(pat))
def _fn_ilike(x, pat):  return _safe_compare("ilike", _as_str(x), _as_str(pat))

# Sequence-wide string checks
def _fn_any_icontains(seq, needle):
    s = _to_seq(seq)
    if s is None: return False
    n = _as_str(needle).lower()
    return any(_as_str(v).lower().find(n) >= 0 for v in s)

def _fn_all_icontains(seq, needle):
    s = _to_seq(seq)
    if not s: return False
    n = _as_str(needle).lower()
    return all(_as_str(v).lower().find(n) >= 0 for v in s)

def _fn_any_matches(seq, pattern, flags: int = 0):
    s = _to_seq(seq)
    if s is None: return False
    try:
        rgx = re.compile(_as_str(pattern), flags=flags)
        return any(rgx.search(_as_str(v)) for v in s)
    except re.error:
        return False

def _fn_all_matches(seq, pattern, flags: int = 0):
    s = _to_seq(seq)
    if not s: return False
    try:
        rgx = re.compile(_as_str(pattern), flags=flags)
        return all(rgx.search(_as_str(v)) for v in s)
    except re.error:
        return False

# =============================================================================
# Boolean recognition / dtype-ish helpers
# =============================================================================

_BOOL_TRUE = {"true","t","1","y","yes","on","ok"}
_BOOL_FALSE = {"false","f","0","n","no","off"}

# typo/misspelling tolerant, e.g. "treu","flase","ye","onn"
_BOOL_TRUE_RX = re.compile(r"^(?:t(?:rue)?|y(?:es)?|1|on|ok|ye)$", re.I)
_BOOL_FALSE_RX = re.compile(r"^(?:f(?:alse)?|n(?:o)?|0|off)$", re.I)

def _is_bool_token(x: Any) -> bool:
    s = _as_str(x).strip()
    if not s: return False
    if s.lower() in _BOOL_TRUE or s.lower() in _BOOL_FALSE:
        return True
    return bool(_BOOL_TRUE_RX.match(s) or _BOOL_FALSE_RX.match(s))

def _to_bool(x: Any) -> Optional[bool]:
    if isinstance(x, bool): return x
    s = _as_str(x).strip()
    if not s: return None
    ls = s.lower()
    if ls in _BOOL_TRUE: return True
    if ls in _BOOL_FALSE: return False
    if _BOOL_TRUE_RX.match(s): return True
    if _BOOL_FALSE_RX.match(s): return False
    return None

def _fn_isbool(x): return _is_bool_token(x) or isinstance(x, bool)
def _fn_istrue(x): 
    tb = _to_bool(x); return (tb is True)
def _fn_isfalse(x):
    tb = _to_bool(x); return (tb is False)
def _fn_to_bool(x):
    tb = _to_bool(x)
    return tb if tb is not None else False
def _fn_boolish(x): return _is_bool_token(x)

def _fn_boolish_ratio(seq) -> float:
    items = _to_seq(seq)
    if not items: return 0.0
    n = len(items)
    hit = 0
    for v in items:
        if _is_bool_token(v) or isinstance(v, bool):
            hit += 1
    return hit / max(1, n)

def _fn_percent_true(seq) -> float:
    items = _to_seq(seq) or []
    n = len(items)
    if n == 0: return 0.0
    t = sum(1 for v in items if _to_bool(v) is True)
    return t / n

def _fn_percent_false(seq) -> float:
    items = _to_seq(seq) or []
    n = len(items)
    if n == 0: return 0.0
    f = sum(1 for v in items if _to_bool(v) is False)
    return f / n

# Name heuristics (regex-based) for role inference
_BOOL_NAME_RX = re.compile(r"^(?:is|has|can|should|flag|bool|enabled|active|valid|success|deleted|complete|paid|approved|verified)(?:_|$)", re.I)
_ID_NAME_RX   = re.compile(r"(?:^|_)(?:id|uuid|guid|key)(?:_|$)", re.I)
_DT_NAME_RX   = re.compile(r"(?:date|time|timestamp|dt|ts)", re.I)

def _fn_looks_like_bool_name(x): return bool(_BOOL_NAME_RX.search(_as_str(x)))
def _fn_looks_like_id_name(x):   return bool(_ID_NAME_RX.search(_as_str(x)))
def _fn_looks_like_dt_name(x):   return bool(_DT_NAME_RX.search(_as_str(x)))

# Dtype helpers (pandas-friendly, but safe without pandas)
def _fn_is_boolean_dtype(x) -> bool:
    s = _as_str(x).lower()
    return any(k in s for k in ("bool", "boolean[", "boolean]"))

def _fn_is_numeric_dtype(x) -> bool:
    s = _as_str(x).lower()
    return any(k in s for k in ("int", "float", "double", "number", "numeric", "decimal"))

def _fn_is_datetime_dtype(x) -> bool:
    s = _as_str(x).lower()
    return any(k in s for k in ("datetime64", "timestamp", "datetime", "datetimetz"))

# =============================================================================
# Numeric parsing / stats helpers
# =============================================================================

_NUM_RX = re.compile(r"^\s*([+-]?((\d+(\.\d*)?)|(\.\d+))([eE][+-]?\d+)?)\s*$")

def _to_number(x: Any) -> Optional[float]:
    if x is None: return None
    if isinstance(x, (int, float)): return float(x)
    s = _as_str(x)
    m = _NUM_RX.match(s)
    if not m: return None
    try:
        return float(m.group(1))
    except Exception:
        return None

def _fn_is_numeric(x) -> bool:
    return _to_number(x) is not None

def _fn_to_number(x, default: Optional[float] = None) -> Optional[float]:
    v = _to_number(x)
    return v if v is not None else default

def _fn_null_ratio(seq) -> float:
    items = _to_seq(seq) or []
    n = len(items)
    if n == 0: return 0.0
    nulls = 0
    for v in items:
        try:
            import pandas as pd  # type: ignore
            if bool(pd.isna(v)):
                nulls += 1
        except Exception:
            if v is None:
                nulls += 1
    return nulls / n

def _fn_unique_count(seq) -> int:
    items = _to_seq(seq) or []
    try:
        return len(set(items))
    except Exception:
        # Fallback for unhashables
        seen = []
        for v in items:
            try:
                h = hash(v)
                if v not in seen:
                    seen.append(v)
            except Exception:
                if str(v) not in seen:
                    seen.append(str(v))
        return len(seen)

# =============================================================================
# Fuzzy / similarity helpers
# =============================================================================

def _fn_similarity(a, b) -> float:
    return _similarity(_as_str(a), _as_str(b))

def _fn_similar(a, b, threshold: float = 0.8) -> bool:
    return _fn_similarity(a, b) >= float(threshold)

def _fn_any_similar(seq, target, threshold: float = 0.8) -> bool:
    s = _to_seq(seq)
    if s is None: return False
    tgt = _as_str(target)
    th = float(threshold)
    return any(_similarity(_as_str(v), tgt) >= th for v in s)

# =============================================================================
# Misc helpers / utilities
# =============================================================================

def _fn_exists(x) -> bool: return x is not None
def _fn_coalesce(a, b): return a if a is not None else b
def _fn_oneof(x, lst) -> bool:
    seq = _to_seq(lst)
    if seq is None: return False
    return x in seq

def _fn_iin(x, lst) -> bool:
    seq = _to_seq(lst)
    if seq is None: return False
    xs = _as_str(x).lower()
    return any(_as_str(v).lower() == xs for v in seq)

# =============================================================================
# Whitelisted safe functions
# =============================================================================

SAFE_FUNCS: Dict[str, Callable[..., Any]] = {
    # Null / existence
    "len": _fn_len,
    "isnull": _fn_isnull,
    "notnull": _fn_notnull,
    "exists": _fn_exists,
    "coalesce": _fn_coalesce,
    "oneof": _fn_oneof,
    "iin": _fn_iin,

    # String transforms / normalization
    "lower": _fn_lower,
    "upper": _fn_upper,
    "strip": _fn_strip,
    "lstrip": _fn_lstrip,
    "rstrip": _fn_rstrip,
    "norm": _fn_norm,

    # String predicates
    "startswith": _fn_startswith,
    "endswith": _fn_endswith,
    "istartswith": _fn_istartswith,
    "iendswith": _fn_iendswith,
    "contains": _fn_contains,
    "icontains": _fn_icontains,
    "iequals": _fn_iequals,
    "like": _fn_like,
    "ilike": _fn_ilike,

    # Regex helpers
    "matches": _fn_matches,
    "imatches": _fn_imatches,
    "fullmatch": _fn_fullmatch,
    "ifullmatch": _fn_ifullmatch,
    "any_icontains": _fn_any_icontains,
    "all_icontains": _fn_all_icontains,
    "any_matches": _fn_any_matches,
    "all_matches": _fn_all_matches,

    # Boolean helpers
    "isbool": _fn_isbool,
    "istrue": _fn_istrue,
    "isfalse": _fn_isfalse,
    "to_bool": _fn_to_bool,
    "boolish": _fn_boolish,
    "boolish_ratio": _fn_boolish_ratio,
    "percent_true": _fn_percent_true,
    "percent_false": _fn_percent_false,
    "looks_like_bool_name": _fn_looks_like_bool_name,
    "looks_like_id_name": _fn_looks_like_id_name,
    "looks_like_dt_name": _fn_looks_like_dt_name,

    # Dtype-ish
    "is_boolean_dtype": _fn_is_boolean_dtype,
    "is_numeric_dtype": _fn_is_numeric_dtype,
    "is_datetime_dtype": _fn_is_datetime_dtype,

    # Numeric parsing / stats
    "is_numeric": _fn_is_numeric,
    "to_number": _fn_to_number,
    "null_ratio": _fn_null_ratio,
    "unique_count": _fn_unique_count,

    # Fuzzy / similarity
    "similarity": _fn_similarity,
    "similar": _fn_similar,
    "any_similar": _fn_any_similar,
}

# =============================================================================
# AST utilities & compiler
# =============================================================================

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
    if typ == "between":
        # ("between", positive_bool, left, low, high)
        return _collect_roots(ast[2]) | _collect_roots(ast[3]) | _collect_roots(ast[4])
    return set()

def _compile(ast, allowed_funcs_set: Optional[set[str]]):
    typ = ast[0]

    if typ == "lit":
        val = ast[1]
        return lambda ctx: val

    if typ == "id":
        name = ast[1]
        return lambda ctx: _resolve_identifier(name, ctx)

    if typ == "list":
        items = [_compile(a, allowed_funcs_set) for a in ast[1]]
        return lambda ctx: [f(ctx) for f in items]

    if typ == "call":
        name = ast[1]
        if allowed_funcs_set is not None and name not in allowed_funcs_set:
            raise ValueError(f"Disallowed function: {name}")
        fn = SAFE_FUNCS.get(name)
        if fn is None:
            raise ValueError(f"Unknown function: {name}")
        arg_nodes = [_compile(a, allowed_funcs_set) for a in ast[2]]
        return lambda ctx: fn(*[g(ctx) for g in arg_nodes])

    if typ == "not":
        f = _compile(ast[1], allowed_funcs_set)
        return lambda ctx: (not bool(f(ctx)))

    if typ == "and":
        lf = _compile(ast[1], allowed_funcs_set); rf = _compile(ast[2], allowed_funcs_set)
        return lambda ctx: (bool(lf(ctx)) and bool(rf(ctx)))

    if typ == "or":
        lf = _compile(ast[1], allowed_funcs_set); rf = _compile(ast[2], allowed_funcs_set)
        return lambda ctx: (bool(lf(ctx)) or bool(rf(ctx)))

    if typ == "cmp":
        _, op, l, r = ast
        lf = _compile(l, allowed_funcs_set); rf = _compile(r, allowed_funcs_set)
        return lambda ctx: _safe_compare(op, lf(ctx), rf(ctx))

    if typ == "between":
        # ("between", positive_bool, left, low, high)
        _, positive, l, low, high = ast
        lf = _compile(l, allowed_funcs_set)
        lowf = _compile(low, allowed_funcs_set)
        highf = _compile(high, allowed_funcs_set)
        def _between(ctx):
            try:
                v = lf(ctx)
                lo = lowf(ctx)
                hi = highf(ctx)
                ok = (v is not None and lo is not None and hi is not None and (v >= lo) and (v <= hi))
                return ok if positive else (not ok)
            except Exception:
                return False if positive else True
        return _between

    raise ValueError(f"Unknown AST node: {ast!r}")

# =============================================================================
# Public API
# =============================================================================

def compile_condition(
    expr: str,
    allowed_vars: set[str] | None = None,
    *,
    allowed_funcs: set[str] | None = None,
) -> AllowedCallable:
    """
    Compile a safe callable(ctx)->bool from `expr`.

    Grammar highlights:
      - Logical: AND, OR, NOT, parentheses
      - Comparisons: == != < <= > >=
      - Membership: IN, NOT IN (lists or sequences)
      - SQL-like: LIKE, ILIKE, NOT LIKE, NOT ILIKE (%, _ wildcards)
      - Regex: =~, !~, ~*, !~*   (search; * is case-insensitive)
      - Nullish/identity: IS [NOT] <literal>   (e.g., IS NULL, IS TRUE)
      - Ranges: BETWEEN / NOT BETWEEN     (inclusive)
      - Literals: strings '...' / "...", numbers, TRUE/FALSE, NULL/NONE
      - Lists: [expr, expr, ...]
      - Calls: any function in SAFE_FUNCS (unioned with `allowed_funcs`)

    Safety:
      - Only whitelisted functions can be called.
      - No eval/exec/dynamic attribute traversal (dict/attribute only).
      - Identifier roots can be restricted via `allowed_vars`.
    """
    tokens = _tokenize(expr)
    parser = Parser(tokens)
    ast = parser.parse()

    if allowed_vars is not None:
        roots = _collect_roots(ast)
        bad = [r for r in roots if r not in allowed_vars]
        if bad:
            raise ValueError(f"Use of disallowed identifiers: {bad}")

    # Always allow SAFE_FUNCS; union with any caller-provided set
    allowed_set = set(SAFE_FUNCS.keys()) | (set(allowed_funcs) if allowed_funcs else set())

    fn = _compile(ast, allowed_set)

    def _wrapped(ctx: dict[str, Any]) -> bool:
        try:
            return bool(fn(ctx))
        except Exception:
            # Any runtime error becomes False, keeping the engine robust.
            return False
    return _wrapped

def eval_condition(fn: AllowedCallable, ctx: dict[str, Any]) -> bool:
    return bool(fn(ctx))
