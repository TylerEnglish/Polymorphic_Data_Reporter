from __future__ import annotations
from typing import Any, Callable

# Minimal, safe, interpretable DSL.
# TODO: Implement a tiny parser:
#   - tokens: identifiers, strings, numbers, booleans, and/or/not, == != < <= > >=, in/notin, parentheses
#   - variables resolved from the provided context dict
# For now, we wire the public API and raise NotImplemented in compile.

AllowedCallable = Callable[[dict[str, Any]], bool]

def compile_condition(expr: str, allowed_vars: set[str] | None = None) -> AllowedCallable:
    """
    Return a callable(ctx) -> bool that evaluates `expr` safely.
    `allowed_vars` can restrict which top-level names may appear.

    NOTE: Do NOT eval. Hand-parse (TODO).
    """
    # TODO: parse 'expr' into an AST and build a safe evaluator
    raise NotImplementedError("DSL compile_condition not implemented yet.")

def eval_condition(fn: AllowedCallable, ctx: dict[str, Any]) -> bool:
    """Invoke compiled condition. This method stays tiny & pure."""
    return bool(fn(ctx))
