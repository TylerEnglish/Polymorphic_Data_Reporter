from __future__ import annotations
from functools import lru_cache, wraps
from typing import Any, Callable, Dict, Iterable, Iterator, List, Tuple, TypeVar, overload

from toolz import curry as _curry, compose as _compose, pipe as _pipe
from more_itertools import chunked as _chunked, unique_everseen as _unique_everseen

A = TypeVar("A")
B = TypeVar("B")
C = TypeVar("C")

# Re-export toolz versions under your API names
curry = _curry

def identity(x: A) -> A:
    return x

def const(x: A) -> Callable[..., A]:
    return lambda *_, **__: x

def pipe(x: A, *fns: Callable[[Any], Any]) -> Any:
    # Keep signature but delegate to toolz.pipe
    return _pipe(x, *fns) if fns else x

def compose(*fns: Callable[[Any], Any]) -> Callable[[Any], Any]:
    # toolz.compose composes right-to-left as expected
    return _compose(*fns)

def try_or(default: B) -> Callable[[Callable[[A], B]], Callable[[A], B]]:
    def _wrap(fn: Callable[[A], B]) -> Callable[[A], B]:
        @wraps(fn)
        def _inner(x: A) -> B:
            try:
                return fn(x)
            except Exception:
                return default
        return _inner
    return _wrap

@overload
def maybe(x: None, *_: Callable[[Any], Any]) -> None: ...
@overload
def maybe(x: A, *fns: Callable[[Any], Any]) -> Any: ...
def maybe(x: Any, *fns: Callable[[Any], Any]) -> Any:
    if x is None:
        return None
    return pipe(x, *fns)

def map_dict(d: Dict[A, B], fn: Callable[[Tuple[A, B]], Tuple[A, C]]) -> Dict[A, C]:
    return dict(fn(item) for item in d.items())

def filter_dict(d: Dict[A, B], pred: Callable[[Tuple[A, B]], bool]) -> Dict[A, B]:
    return dict(item for item in d.items() if pred(item))

def chunked(it: Iterable[A], size: int) -> Iterator[List[A]]:
    # Explicit guard (consistent, predictable)
    if size < 1:
        raise ValueError("chunk size must be >= 1")
    for chunk in _chunked(it, size):
        yield list(chunk)

def unique_stable(seq: Iterable[A]) -> List[A]:
    # Delegate to more-itertools; preserves first-seen order
    return list(_unique_everseen(seq))

def memoize(maxsize: int = 128):
    def deco(fn: Callable[..., B]) -> Callable[..., B]:
        return lru_cache(maxsize=maxsize)(fn)  # type: ignore
    return deco
