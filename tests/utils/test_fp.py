import math
import pytest
from src.utils.fp import (
    curry, identity, const, pipe, compose, try_or, maybe,
    map_dict, filter_dict, chunked, unique_stable, memoize
)

def test_identity_const():
    assert identity(10) == 10
    f = const("x")
    assert f(1,2,3) == "x"
    assert f() == "x"

def test_pipe_basic_and_empty():
    assert pipe(2, lambda x: x + 1, lambda x: x * 3) == 9
    # empty pipeline should return input unchanged
    assert pipe("abc") == "abc"

def test_compose_right_to_left():
    f = compose(lambda x: x + 1, lambda x: x * 2)  # (x*2)+1
    assert f(3) == 7

def test_curry_positional_kwargs_defaults():
    @curry
    def f(a, b, c=10, *, d=1):
        return (a + b + c) * d

    # Once required positional args (a, b) are satisfied, it executes using defaults
    assert f(1)(2) == (1 + 2 + 10) * 1

    # Override optional c at the call that completes required args
    assert f(1)(2, c=3) == (1 + 2 + 3) * 1

    # Provide all at once
    assert f(1, 2, 3, d=4) == (1 + 2 + 3) * 4

    # Partial with kwargs that don't satisfy required positional args yet
    g = f(1, d=5)   # still missing b, so returns a curried fn
    assert g(2, 3) == (1 + 2 + 3) * 5

    # Also allow finishing with only b (uses default c)
    assert g(2) == (1 + 2 + 10) * 5

    # You can also set c by keyword when finishing
    assert f(1, d=7)(2, c=0) == (1 + 2 + 0) * 7

def test_curry_overapplication_allowed():
    @curry
    def add(a, b, c): return a + b + c
    # toolz.curry allows calling with extra args once arity is satisfied
    assert add(1)(2, 3) == 6
    assert add(1, 2)(3) == 6
    assert add(1, 2, 3) == 6

def test_try_or_swallow_exception_and_default():
    @try_or(default=-1)
    def risky(x): return int(x)
    assert risky("123") == 123
    assert risky("bad") == -1

def test_maybe_none_short_circuit():
    assert maybe(None, lambda x: x + 1) is None
    assert maybe(3, lambda x: x + 1, lambda x: x * 2) == 8

def test_map_filter_dict():
    d = {"a": 1, "b": 2, "c": 3}
    d2 = map_dict(d, lambda kv: (kv[0].upper(), kv[1] * 10))
    assert d2 == {"A": 10, "B": 20, "C": 30}
    d3 = filter_dict(d2, lambda kv: kv[1] >= 20)
    assert d3 == {"B": 20, "C": 30}

def test_chunked_various_sizes():
    assert list(chunked([1,2,3,4,5], 2)) == [[1,2],[3,4],[5]]
    assert list(chunked([], 3)) == []
    assert list(chunked(range(5), 10)) == [[0,1,2,3,4]]
    with pytest.raises(ValueError):
        list(chunked([1], 0))

def test_unique_stable_preserves_first_occurrence():
    seq = [1,1,2,3,2,4,1,5]
    assert unique_stable(seq) == [1,2,3,4,5]

def test_memoize_decorator_basic():
    calls = {"n": 0}
    @memoize(maxsize=16)
    def fib(n: int) -> int:
        calls["n"] += 1
        if n < 2: return n
        return fib(n-1) + fib(n-2)

    assert fib(10) == 55
    # memoized should drastically reduce calls vs naive exponential
    assert calls["n"] < 2 * 10  # loose upper bound
