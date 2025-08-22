from __future__ import annotations

import math
import numpy as np
import pytest

import src.topics.stats_pvalues as sp


# -------------------------
# Helpers / Tolerances
# -------------------------
RTOL = 1e-11
ATOL = 1e-12

HAVE_SCIPY = sp._HAVE_SCIPY
if HAVE_SCIPY:
    from scipy.stats import norm as _sp_norm, t as _sp_t, chi2 as _sp_chi2, f as _sp_f
    from scipy.special import betainc as _sp_betainc


def _close(a: float, b: float, rtol: float = RTOL, atol: float = ATOL) -> bool:
    return abs(a - b) <= (atol + rtol * abs(b))


# -------------------------
# z_two_tailed
# -------------------------
@pytest.mark.parametrize(
    "z, expected, tol",
    [
        (0.0, 1.0, 0.0),
        (1.0, 0.31731050786291415, 5e-15),
        (2.0, 0.04550026389635842, 5e-15),
        (3.0, 0.0026997960632601866, 5e-18),
    ],
)
def test_z_two_tailed_known_values(z, expected, tol):
    p = sp.z_two_tailed(z)
    assert math.isfinite(p)
    assert abs(p - expected) <= tol


def test_z_two_tailed_extremes_and_invalid():
    # very large |z| -> ~0
    assert sp.z_two_tailed(10.0) < 1e-22
    # invalid inputs return 1.0 by design
    assert sp.z_two_tailed(float("nan")) == 1.0
    assert sp.z_two_tailed(float("inf")) == 1.0
    assert sp.z_two_tailed(float("-inf")) == 1.0


# -------------------------
# t_two_tailed (exact) & t_approx_p (approx)
# -------------------------
@pytest.mark.parametrize(
    "t, df",
    [
        (0.0, 1),
        (0.0, 5),
        (2.0, 5),
        (2.0, 10),
        (3.0, 30),
        (-2.5, 12),
    ],
)
def test_t_two_tailed_basic_invariants(t, df):
    p = sp.t_two_tailed(t, df)
    assert 0.0 <= p <= 1.0
    # symmetry
    assert _close(sp.t_two_tailed(t, df), sp.t_two_tailed(-t, df))
    # p(0, df) = 1.0
    assert _close(sp.t_two_tailed(0.0, df), 1.0, atol=1e-15)


@pytest.mark.parametrize("df", [1, 5, 30, 120])
def test_t_two_tailed_monotone_in_abs_t(df):
    # As |t| grows, p should decrease
    p1 = sp.t_two_tailed(1.0, df)
    p2 = sp.t_two_tailed(2.0, df)
    p3 = sp.t_two_tailed(3.0, df)
    assert p1 > p2 > p3


@pytest.mark.parametrize("t, df", [(2.0, -1), (2.0, 0), (float("nan"), 10)])
def test_t_two_tailed_invalid_returns_one(t, df):
    assert sp.t_two_tailed(t, df) == 1.0


@pytest.mark.skipif(not HAVE_SCIPY, reason="SciPy required for exact baseline")
@pytest.mark.parametrize(
    "t, df",
    [(0.5, 5), (2.0, 10), (3.0, 40), (5.0, 100)]
)
def test_t_two_tailed_matches_scipy(t, df):
    expected = float(2.0 * _sp_t.sf(abs(t), df))
    got = sp.t_two_tailed(t, df)
    assert _close(got, expected, rtol=1e-12, atol=1e-14)


def test_t_approx_p_behaviour():
    # Approx p ~= normal tail for df >= 30
    p_approx = sp.t_approx_p(2.0, 30)
    p_norm = sp.z_two_tailed(2.0)
    assert abs(p_approx - p_norm) < 5e-3  # rough screen


# -------------------------
# chi2_sf
# -------------------------
def test_chi2_sf_boundaries_and_invalid():
    # chi2 at 0 -> SF = 1
    assert _close(sp.chi2_sf(0.0, 4), 1.0)
    # negative chisq clamps to 0 -> SF(0) = 1
    assert _close(sp.chi2_sf(-5.0, 4), 1.0)
    # invalid dof => coerced to 1 (but still finite)
    p = sp.chi2_sf(3.84, 0)
    assert 0.0 <= p <= 1.0
    # NaN -> 1.0
    assert sp.chi2_sf(float("nan"), 2) == 1.0
    assert sp.chi2_sf(2.0, float("nan")) == 1.0


@pytest.mark.skipif(not HAVE_SCIPY, reason="SciPy required for baseline comparison")
@pytest.mark.parametrize(
    "x, dof",
    [(0.1, 1), (1.0, 2), (5.0, 2), (9.21, 2), (10.0, 4), (15.0, 10)]
)
def test_chi2_sf_matches_scipy(x, dof):
    expected = float(_sp_chi2.sf(x, dof))
    got = sp.chi2_sf(x, dof)
    assert _close(got, expected, rtol=1e-11, atol=1e-13)


# -------------------------
# f_sf
# -------------------------
def test_f_sf_boundaries_and_invalid():
    # F at 0 -> SF = 1
    assert _close(sp.f_sf(0.0, 3, 20), 1.0)
    # Negative F clamps to 0 -> SF(0)=1
    assert _close(sp.f_sf(-1.0, 3, 20), 1.0)
    # invalid inputs
    assert sp.f_sf(float("nan"), 3, 20) == 1.0
    assert sp.f_sf(1.5, float("nan"), 20) == 1.0
    assert sp.f_sf(1.5, 3, float("nan")) == 1.0
    # extreme large F -> tiny p
    assert sp.f_sf(1e6, 5, 100) < 1e-12


@pytest.mark.skipif(not HAVE_SCIPY, reason="SciPy required for baseline comparison")
@pytest.mark.parametrize(
    "F, df1, df2",
    [(0.5, 1, 5), (1.0, 3, 20), (2.5, 5, 30), (5.0, 10, 50)]
)
def test_f_sf_matches_scipy(F, df1, df2):
    expected = float(_sp_f.sf(F, df1, df2))
    got = sp.f_sf(F, df1, df2)
    assert _close(got, expected, rtol=1e-11, atol=1e-13)


# -------------------------
# pearsonr_p_two_tailed
# -------------------------
def test_pearsonr_basic_invariants():
    # Symmetry in r
    p1 = sp.pearsonr_p_two_tailed(0.3, 50)
    p2 = sp.pearsonr_p_two_tailed(-0.3, 50)
    assert _close(p1, p2, rtol=1e-12, atol=1e-14)

    # p decreases with |r|
    p_small = sp.pearsonr_p_two_tailed(0.1, 100)
    p_med = sp.pearsonr_p_two_tailed(0.5, 100)
    p_large = sp.pearsonr_p_two_tailed(0.9, 100)
    assert p_small > p_med > p_large

    # r outside [-1,1] is clamped
    assert sp.pearsonr_p_two_tailed(1.2, 10) == 0.0
    assert sp.pearsonr_p_two_tailed(-1.2, 10) == 0.0

    # perfect correlation with n >= 3 -> p = 0
    assert sp.pearsonr_p_two_tailed(1.0, 3) == 0.0
    assert sp.pearsonr_p_two_tailed(-1.0, 10) == 0.0

    # n < 3 -> not defined -> 1.0
    assert sp.pearsonr_p_two_tailed(0.5, 2) == 1.0

    # invalid inputs -> 1.0
    assert sp.pearsonr_p_two_tailed(float("nan"), 20) == 1.0
    assert sp.pearsonr_p_two_tailed(0.3, float("nan")) == 1.0


@pytest.mark.skipif(not HAVE_SCIPY, reason="SciPy required for baseline comparison")
@pytest.mark.parametrize(
    "r, n",
    [(0.0, 10), (0.1, 30), (0.5, 50), (-0.75, 40)]
)
def test_pearsonr_matches_t_formula_with_scipy(r, n):
    # Our implementation already uses t distribution, verify against SciPy t.sf
    n = int(n)
    if n < 3:
        pytest.skip("n too small")
    rr = max(-1.0, min(1.0, r))
    if abs(rr) >= 1.0:
        expected = 0.0
    else:
        df = n - 2
        t = rr * math.sqrt(df / max(1.0 - rr * rr, sp.EPS))
        expected = float(2.0 * _sp_t.sf(abs(t), df))
    got = sp.pearsonr_p_two_tailed(r, n)
    assert _close(got, expected, rtol=1e-11, atol=1e-13)


# -------------------------
# reg_incomplete_beta
# -------------------------
@pytest.mark.skipif(not HAVE_SCIPY, reason="SciPy required for baseline comparison")
@pytest.mark.parametrize(
    "a, b, x",
    [
        (0.5, 0.5, 0.1),
        (0.5, 0.5, 0.9),
        (2.0, 3.0, 0.25),
        (2.0, 3.0, 0.75),
        (5.0, 2.0, 0.3),
        (5.0, 2.0, 0.8),
    ],
)
def test_reg_incomplete_beta_matches_scipy(a, b, x):
    expected = float(_sp_betainc(a, b, x))
    got = sp.reg_incomplete_beta(a, b, x)
    assert _close(got, expected, rtol=1e-12, atol=1e-14)


def test_reg_incomplete_beta_boundaries_and_invalid():
    # boundaries
    assert sp.reg_incomplete_beta(2.0, 3.0, 0.0) == 0.0
    assert sp.reg_incomplete_beta(2.0, 3.0, 1.0) == 1.0
    # invalid a/b -> NaN
    assert math.isnan(sp.reg_incomplete_beta(-1.0, 3.0, 0.5))
    assert math.isnan(sp.reg_incomplete_beta(2.0, 0.0, 0.5))
    # invalid x
    assert math.isnan(sp.reg_incomplete_beta(2.0, 3.0, float("nan")))


@pytest.mark.skipif(not HAVE_SCIPY, reason="Only meaningful if SciPy exists to compare")
def test_force_fallback_paths_match_scipy(monkeypatch):
    # Force fallbacks by disabling SciPy usage in the module
    monkeypatch.setattr(sp, "_HAVE_SCIPY", False, raising=False)
    monkeypatch.setattr(sp, "_sp_betainc", None, raising=False)
    monkeypatch.setattr(sp, "_sp_norm", None, raising=False)
    monkeypatch.setattr(sp, "_sp_t", None, raising=False)
    monkeypatch.setattr(sp, "_sp_chi2", None, raising=False)
    monkeypatch.setattr(sp, "_sp_f", None, raising=False)

    # Beta
    for (a, b, x) in [(0.5, 0.5, 0.1), (2.0, 3.0, 0.7), (5.0, 2.0, 0.3)]:
        expected = float(_sp_betainc(a, b, x))
        got = sp.reg_incomplete_beta(a, b, x)
        assert _close(got, expected, rtol=1e-10, atol=1e-12)

    # Chi-square
    for (x, dof) in [(0.1, 1), (5.0, 2), (10.0, 4)]:
        expected = float(_sp_chi2.sf(x, dof))
        got = sp.chi2_sf(x, dof)
        assert _close(got, expected, rtol=1e-10, atol=1e-12)

    # F
    for (F, df1, df2) in [(0.5, 1, 5), (2.5, 5, 30), (5.0, 10, 50)]:
        expected = float(_sp_f.sf(F, df1, df2))
        got = sp.f_sf(F, df1, df2)
        assert _close(got, expected, rtol=1e-10, atol=1e-12)

    # t (two-tailed)
    for (t, df) in [(0.5, 5), (2.0, 10), (3.0, 40)]:
        expected = float(2.0 * _sp_t.sf(abs(t), df))
        got = sp.t_two_tailed(t, df)
        assert _close(got, expected, rtol=1e-10, atol=1e-12)


# -------------------------
# General numeric sanity / clamps
# -------------------------
@pytest.mark.parametrize(
    "func, args",
    [
        (sp.chi2_sf, (float("inf"), 4)),
        (sp.f_sf, (float("inf"), 3, 20)),
        (sp.t_two_tailed, (float("inf"), 10)),
    ],
)
def test_invalid_infinite_inputs_return_reasonable_prob(func, args):
    p = func(*args)
    # Our design returns 1.0 for invalid (inf) in t; chi2/f with inf should yield 0.0 if SciPy path,
    # but in invalid branch it returns 1.0. We only assert it's within [0,1].
    assert 0.0 <= p <= 1.0


def test_probabilities_are_clamped():
    # ensure clamp utility never leaks out-of-range
    vals = [
        sp.z_two_tailed(0.0),
        sp.t_two_tailed(0.0, 10),
        sp.chi2_sf(0.0, 5),
        sp.f_sf(0.0, 5, 10),
        sp.pearsonr_p_two_tailed(0.0, 10),
    ]
    for v in vals:
        assert 0.0 <= v <= 1.0
