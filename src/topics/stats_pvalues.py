from __future__ import annotations

"""
Lightweight p-value utilities with SciPy acceleration and safe fallbacks.

Design goals
------------
- Prefer SciPy (fast, stable); gracefully fall back to pure-Python/numpy.
- Guard against divisions by zero, log(0), NaNs/Infs; clamp to [0, 1].
- Deterministic O(1) time/memory per scalar call (small CF/series loops).
- Useful primitives for z, t, chi-square, F, and Pearson r.

Public API
----------
- z_two_tailed(z)                  -> float
- t_approx_p(t, df)                -> float        (fast approximation)
- t_two_tailed(t, df)              -> float        (exact via beta; SciPy when available)
- chi2_sf(chisq, dof)              -> float        (upper tail)
- chi2_p_value(chisq, dof)         -> float        (alias)
- f_sf(F, df1, df2)                -> float        (upper tail)
- f_p_value(F, df1, df2)           -> float        (alias)
- pearsonr_p_two_tailed(r, n)      -> float
- reg_incomplete_beta(a, b, x)     -> float        (regularized I_x(a,b))
"""

from typing import Optional
import math
import numpy as np

# Try SciPy accelerators if available
try:
    from scipy.stats import norm as _sp_norm, t as _sp_t, chi2 as _sp_chi2, f as _sp_f
    from scipy.special import betainc as _sp_betainc  # regularized incomplete beta I_x(a,b)
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover - only hits in environments without SciPy
    _sp_norm = _sp_t = _sp_chi2 = _sp_f = None
    _sp_betainc = None
    _HAVE_SCIPY = False


# ---- numeric guards & constants ----
EPS = 1e-15       # to avoid log(0) and zero divisors
TINY = 1e-300     # tiny positive for safe reciprocals
MAXITER = 10_000  # hard cap for CF/series (converges much sooner)
DF_NORMAL_SWITCH = 1_000  # t-dist uses Normal approx beyond this df (fallback path)


def _is_finite(x: float) -> bool:
    try:
        return np.isfinite(x).item()
    except Exception:
        try:
            return math.isfinite(float(x))
        except Exception:
            return False


def _clamp01(p: float) -> float:
    if not _is_finite(p):
        return 1.0
    if p < 0.0:
        return 0.0
    if p > 1.0:
        return 1.0
    return p


def _safe_div(num: float, den: float, fallback: float = 0.0) -> float:
    if not (_is_finite(num) and _is_finite(den)) or den == 0.0:
        return fallback
    return num / den


# =========================
# Normal / Z
# =========================
def z_two_tailed(z: float) -> float:
    """
    Two-tailed p-value for a z-score (Normal(0,1)).

    Returns 1.0 on invalid input.
    Complexity: O(1)
    """
    if not _is_finite(z):
        return 1.0
    az = abs(float(z))
    if _HAVE_SCIPY:  # fast & stable
        return _clamp01(2.0 * _sp_norm.sf(az))
    # fallback via erf
    from math import erf, sqrt
    p = 2.0 * (1.0 - 0.5 * (1.0 + erf(az / sqrt(2.0))))
    return _clamp01(p)


# Legacy fast approximation (kept for backward compatibility)
def t_approx_p(t: float, df: float) -> float:
    """
    Approximate two-tailed p-value for Student-t with df degrees of freedom.

    - For df >= 30, uses Normal(z=t) two-tailed.
    - For smaller df, a crude heavy-tail approximation (screening only).
    - Returns 1.0 on invalid input.

    Prefer t_two_tailed for exact values.
    Complexity: O(1)
    """
    if not (_is_finite(t) and _is_finite(df)) or df <= 0:
        return 1.0
    if df >= 30:
        return z_two_tailed(t)
    a = 1.0 + (t * t) / max(df, 1.0)
    one_tail = 0.5 * a ** (-(df + 1.0) / 2.0)
    return _clamp01(2.0 * one_tail)


# =========================
# Gamma / Beta fallbacks
# =========================
def _reg_lower_gamma(a: float, x: float) -> float:
    """Regularized lower incomplete gamma P(a,x) using series."""
    if x <= 0.0:
        return 0.0
    if a <= 0.0 or not (_is_finite(a) and _is_finite(x)):
        return 0.0
    gln = math.lgamma(a)
    ap = a
    summ = 1.0 / a
    delt = summ
    for _ in range(MAXITER):
        ap += 1.0
        delt *= x / ap
        summ += delt
        if abs(delt) < abs(summ) * 1e-14:
            break
    return math.exp(-x + a * math.log(max(x, EPS)) - gln) * summ


def _reg_upper_gamma(a: float, x: float) -> float:
    """Regularized upper incomplete gamma Q(a,x) via continued fraction."""
    if x <= 0.0:
        return 1.0
    if a <= 0.0 or not (_is_finite(a) and _is_finite(x)):
        return 1.0
    gln = math.lgamma(a)
    b0 = x + 1.0 - a
    c = 1.0 / TINY
    d = 1.0 / max(b0, TINY)
    h = d
    for i in range(1, MAXITER + 1):
        an = -i * (i - a)
        b = b0 + 2.0 * i
        d = 1.0 / max(b + an * d, TINY)
        c = max(b + _safe_div(an, c, 0.0), TINY)
        delt = c * d
        h *= delt
        if abs(delt - 1.0) < 1e-14:
            break
    pref = math.exp(-x + a * math.log(max(x, EPS)) - gln)
    return pref * h


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction core for regularized incomplete beta (Lentz)."""
    if not all(_is_finite(v) for v in (a, b, x)) or a <= 0.0 or b <= 0.0:
        return float("nan")
    if x <= 0.0 or x >= 1.0:
        return 0.0

    am, bm = 1.0, 1.0
    az = 1.0
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    bz = 1.0 - _safe_div(qab * x, qap, 0.0)

    aold = 0.0
    for m in range(1, MAXITER + 1):
        em = float(m)
        tem = 2.0 * em

        d = _safe_div(em * (b - em) * x, (qam + tem) * (a + tem), 0.0)
        ap = az + d * am
        bp = bz + d * bm

        d = -_safe_div((a + em) * (qab + em) * x, (a + tem) * (qap + tem), 0.0)
        app = ap + d * az
        bpp = bp + d * bz

        if bpp == 0.0:
            bpp = TINY
        am, bm = ap / bpp, bp / bpp
        az = app / bpp
        bz = 1.0

        if abs(az - aold) < 1e-14 * max(1.0, abs(az)):
            return az
        aold = az
    return az


def reg_incomplete_beta(a: float, b: float, x: float) -> float:
    """
    Regularized incomplete beta I_x(a,b).
    Uses SciPy when available; otherwise uses a stable CF fallback.
    """
    if _HAVE_SCIPY:
        try:
            # SciPy's betainc is already the regularized I_x(a,b)
            if not all(_is_finite(v) for v in (a, b, x)) or a <= 0.0 or b <= 0.0:
                return float("nan")
            xx = 0.0 if x <= 0.0 else (1.0 if x >= 1.0 else float(x))
            return float(_sp_betainc(float(a), float(b), xx))
        except Exception:
            pass  # fall back

    # Fallback
    if not all(_is_finite(v) for v in (a, b, x)) or a <= 0.0 or b <= 0.0:
        return float("nan")
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0

    lnB = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)

    def _front(_a: float, _b: float, _x: float) -> float:
        return math.exp(
            _a * math.log(max(_x, EPS)) +
            _b * math.log(max(1.0 - _x, EPS)) -
            lnB
        )

    if x < (a + 1.0) / (a + b + 2.0):
        cf = _betacf(a, b, x)
        if not _is_finite(cf):
            return float("nan")
        return _front(a, b, x) * cf / a
    else:
        cf = _betacf(b, a, 1.0 - x)
        if not _is_finite(cf):
            return float("nan")
        return 1.0 - _front(b, a, 1.0 - x) * cf / b


# =========================
# Chi-square (upper tail)
# =========================
def chi2_sf(chisq: float, dof: int) -> float:
    """
    Survival function P[Chi^2_dof >= chisq].
    Uses SciPy if available; fallback via regularized gamma Q(k/2, x/2).
    Complexity: O(1)
    """
    if not (_is_finite(chisq) and _is_finite(dof)):
        return 1.0
    dof_int = max(int(dof), 1)
    x = max(float(chisq), 0.0)

    if _HAVE_SCIPY:
        try:
            return _clamp01(float(_sp_chi2.sf(x, dof_int)))
        except Exception:
            pass

    # Fallback
    a = 0.5 * dof_int
    xx = 0.5 * x
    if xx < a + 1.0:
        p_lower = _reg_lower_gamma(a, xx)
        return _clamp01(1.0 - p_lower)
    else:
        q_upper = _reg_upper_gamma(a, xx)
        return _clamp01(q_upper)


def chi2_p_value(chisq: float, dof: int) -> float:
    """Alias for chi2_sf."""
    return chi2_sf(chisq, dof)


# =========================
# F (upper tail) via I_x(a,b)
# =========================
def f_sf(F: float, df1: int, df2: int) -> float:
    """
    Survival function P[F_{df1,df2} >= F].
    Uses SciPy if available; otherwise 1 - I_x(a,b), x = df1*F/(df1*F+df2).
    Complexity: O(1)
    """
    if not all(_is_finite(v) for v in (F, df1, df2)):
        return 1.0
    F = max(float(F), 0.0)
    df1 = max(int(df1), 1)
    df2 = max(int(df2), 1)

    if _HAVE_SCIPY:
        try:
            return _clamp01(float(_sp_f.sf(F, df1, df2)))
        except Exception:
            pass

    denom = max(df1 * F + df2, EPS)
    x = (df1 * F) / denom
    a = 0.5 * df1
    b = 0.5 * df2
    cdf = reg_incomplete_beta(a, b, x)
    return _clamp01(1.0 - cdf if _is_finite(cdf) else 1.0)


def f_p_value(F: float, df1: int, df2: int) -> float:
    """Alias for f_sf."""
    return f_sf(F, df1, df2)


# =========================
# Student-t (two-tailed)
# =========================
def _t_two_tailed_fallback(t: float, df_int: int) -> float:
    """
    Exact two-tailed p-value via regularized incomplete beta (fallback path).
    Complexity: O(1)
    """
    at = abs(float(t))
    if df_int >= DF_NORMAL_SWITCH:
        # approximate with Normal in extreme df
        return z_two_tailed(at)

    a = 0.5 * df_int
    b = 0.5
    denom = max(df_int + at * at, EPS)
    x = df_int / denom
    ib = reg_incomplete_beta(a, b, x)
    if not _is_finite(ib):
        return 1.0
    return _clamp01(2.0 * 0.5 * ib)  # two-tailed = 2 * one-tail


def t_two_tailed(t: float, df: int | float) -> float:
    """
    Exact two-tailed p-value for Student-t(df).
    Uses SciPy when available; otherwise stable beta-based fallback.
    Returns 1.0 on invalid input.
    Complexity: O(1)
    """
    if not (_is_finite(t) and _is_finite(df)) or df <= 0:
        return 1.0
    df_int = max(int(df), 1)

    if _HAVE_SCIPY:
        try:
            return _clamp01(float(2.0 * _sp_t.sf(abs(float(t)), df_int)))
        except Exception:
            pass

    return _t_two_tailed_fallback(t, df_int)


# =========================
# Pearson r
# =========================
def pearsonr_p_two_tailed(r: float, n: int) -> float:
    """
    Two-tailed p for Pearson correlation coefficient.

    Uses t = r * sqrt((n-2)/(1 - r^2)), df = n - 2.
    Conservatively returns 1.0 for invalid inputs or n < 3.
    Complexity: O(1)
    """
    if not (_is_finite(r) and _is_finite(n)):
        return 1.0
    n = int(n)
    if n < 3:
        return 1.0
    rr = float(max(-1.0, min(1.0, r)))
    if abs(rr) >= 1.0:
        # Perfect correlation -> p = 0 when n>=3
        return 0.0
    df = n - 2
    denom = max(1.0 - rr * rr, EPS)
    t = rr * math.sqrt(max(df, 1) / denom)
    return t_two_tailed(t, df)


__all__ = [
    "EPS",
    "TINY",
    "MAXITER",
    "z_two_tailed",
    "t_approx_p",
    "t_two_tailed",
    "chi2_sf",
    "chi2_p_value",
    "f_sf",
    "f_p_value",
    "pearsonr_p_two_tailed",
    "reg_incomplete_beta",
]
