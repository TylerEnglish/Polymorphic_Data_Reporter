from __future__ import annotations
from typing import Iterable, Sequence, Union, Callable, Optional
import re
import pandas as pd

__all__ = ["regex_replace", "extract_digits"]

PatternLike = Union[str, re.Pattern]
ReplLike = Union[str, Callable[[re.Match], str]]


# ----------------------------- helpers ---------------------------------

def _to_string_series(s: pd.Series) -> pd.Series:
    """Convert to pandas StringDtype, preserving NA."""
    try:
        return s.astype("string")
    except Exception:
        return s.astype(object).astype("string")


def _parse_flags(flags: Optional[Union[str, int, Iterable[str]]]) -> int:
    """
    Accepts:
      - None
      - int bitmask (re.*)
      - "IMSXA" (any order, case-insensitive)
      - iterable of one-letter flag strings
    """
    if flags is None:
        return 0
    if isinstance(flags, int):
        return int(flags)

    # Normalize to a list of upper-case single letters
    if isinstance(flags, str):
        items = list(flags.upper())
    else:
        items = [str(f).upper() for f in flags]

    m = {"I": re.IGNORECASE, "M": re.MULTILINE, "S": re.DOTALL,
         "X": re.VERBOSE, "A": re.ASCII}
    out = 0
    for ch in items:
        out |= m.get(ch, 0)
    return out


def _compile_one(p: PatternLike, flags: int, *, literal: bool) -> re.Pattern:
    if isinstance(p, re.Pattern):
        # Respect precompiled flags/pattern; user can still pass extra `flags`
        # which we OR in by recompiling the raw pattern.
        base_flags = p.flags | flags
        return re.compile(p.pattern if not literal else re.escape(p.pattern), base_flags)
    # string
    patt = re.escape(p) if literal else p
    return re.compile(patt, flags)


def _compile_many(
    patterns: Union[PatternLike, Sequence[PatternLike]],
    flags: int,
    *,
    literal: bool,
) -> list[re.Pattern]:
    if isinstance(patterns, (str, re.Pattern)):
        return [_compile_one(patterns, flags, literal=literal)]
    return [_compile_one(p, flags, literal=literal) for p in patterns]


# ----------------------------- public API ---------------------------------

def regex_replace(
    s: pd.Series,
    *,
    pattern: Union[PatternLike, Sequence[PatternLike]],
    repl: Union[ReplLike, Sequence[ReplLike]],
    flags: Union[None, str, int, Iterable[str]] = None,
    count: int = 0,
    literal: bool = False,
) -> pd.Series:
    """
    Regex (or literal) replacement on string-like series.

    Parameters
    ----------
    s : pd.Series
        Input series. Non-string/object series are returned unchanged (copy).
    pattern : str | re.Pattern | sequence thereof
        Pattern(s) to apply. If a sequence is given, replacements are applied in order.
    repl : str | callable | sequence thereof
        Replacement(s). If a single replacement is given with multiple patterns, it is reused.
        If a sequence is passed, it must have the same length as `pattern`.
        Replacement strings support group backreferences (\\1, \\g<name>).
    flags : None | str | int | iterable of flags, default None
        Supports letters "I", "M", "S", "X", "A"; or an integer bitmask (re.*).
    count : int, default 0
        Maximum number of pattern occurrences to replace per element (0 → replace all).
    literal : bool, default False
        If True, treat `pattern` as a literal string (re.escape) rather than a regex.

    Returns
    -------
    pd.Series (StringDtype)

    Notes
    -----
    - NA values remain NA.
    - If pattern compilation fails, the original series is returned (copied).
    """
    if not (pd.api.types.is_object_dtype(s.dtype) or pd.api.types.is_string_dtype(s.dtype)):
        return s.copy(deep=True)

    x = _to_string_series(s)
    try:
        fl = _parse_flags(flags)
        rxs = _compile_many(pattern, fl, literal=literal)
    except Exception:
        return x.copy(deep=True)

    # Normalize replacement(s)
    if isinstance(repl, (list, tuple)):
        if len(repl) != len(rxs):
            raise ValueError("When passing multiple patterns, `repl` must match length or be a single str/callable.")
        repls: list[ReplLike] = list(repl)
    else:
        repls = [repl] * len(rxs)

    def _apply_all(val: object) -> object:
        if pd.isna(val):
            return pd.NA
        txt = str(val)
        for rx, rp in zip(rxs, repls):
            try:
                if count == 1:
                    # Prefer the first match whose start > 0; else the first match
                    matches = list(rx.finditer(txt))
                    if not matches:
                        continue
                    # pick first non-initial if available
                    chosen = None
                    for m in matches:
                        if m.start() > 0:
                            chosen = m
                            break
                    if chosen is None:
                        chosen = matches[0]

                    # Build replacement text
                    if callable(rp):
                        repl_text = rp(chosen)
                    else:
                        # Let re handle backrefs/groups on the matched text
                        repl_text = rx.sub(rp, chosen.group(0), count=1)

                    txt = txt[:chosen.start()] + repl_text + txt[chosen.end():]
                else:
                    # Standard behavior for count==0 (replace all) or count>1
                    if callable(rp):
                        txt = rx.sub(rp, txt, count=count)
                    else:
                        txt = rx.sub(rp, txt, count=count)
            except Exception:
                # If replacement fails for this pattern, skip it
                continue
        return txt

    out = x.map(_apply_all)
    return out.astype("string")


def extract_digits(
    s: pd.Series,
    *,
    ascii_only: bool = False,
    keep_sign: bool = False,
    keep_decimal: bool = False,
    empty_to_na: bool = False,
) -> pd.Series:
    """
    Keep only digits from strings. Unicode-aware by default.

    Parameters
    ----------
    ascii_only : bool, default False
        If True, keep only ASCII digits [0-9]; otherwise, keep any Unicode digit (str.isdigit()).
    keep_sign : bool, default False
        If True, preserve a single leading '+' or '-' if present in the cleaned result.
    keep_decimal : bool, default False
        If True, preserve a single '.' decimal point (first occurrence) in the cleaned result.
    empty_to_na : bool, default False
        If True, convert empty strings to <NA>.

    Returns
    -------
    pd.Series (StringDtype)

    Notes
    -----
    - Non-string/object series are returned unchanged (copy).
    - NA remains NA.
    """
    if not (pd.api.types.is_object_dtype(s.dtype) or pd.api.types.is_string_dtype(s.dtype)):
        return s.copy(deep=True)

    x = _to_string_series(s)

    def _filter_one(val: object) -> object:
        if pd.isna(val):
            return pd.NA
        txt = str(val)

        out = []
        sign_kept = False
        dot_kept = False

        for ch in txt:
            is_digit = ch.isdigit() if not ascii_only else ("0" <= ch <= "9")

            if is_digit:
                out.append(ch)
                continue

            if keep_decimal and not dot_kept and ch == ".":
                out.append(".")
                dot_kept = True
                continue

            if keep_sign and not sign_kept and ch in "+-":
                # Only allow sign at the beginning of the cleaned output
                if not out:
                    out.append(ch)
                    sign_kept = True
                # else ignore
                continue

            # else ignore char

        result = "".join(out)
        if empty_to_na and result == "":
            return pd.NA
        return result

    out = x.map(_filter_one)
    return out.astype("string")
