"""Normal distribution quantiles and tail probabilities, numpy only.

Power calculations need the inverse normal CDF and nothing in numpy provides
it. Rather than pull in scipy for two functions, this implements Acklam's
rational approximation with one Halley refinement step, which is accurate to
about 1e-15 across the usable range and is checked against published
quantiles in the test suite.

Keeping the dependency list at numpy is not stylistic. A reviewer who has to
create an environment before seeing a number usually does not see the number.
"""
from __future__ import annotations

import math
from typing import Sequence, Union

import numpy as np

__all__ = ["norm_ppf", "norm_cdf", "norm_sf"]

# Acklam's coefficients for the rational approximation to the normal quantile.
_A = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
      1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
_B = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
      6.680131188771972e+01, -1.328068155288572e+01)
_C = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
      -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
_D = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
      3.754408661907416e+00)

_P_LOW = 0.02425
_P_HIGH = 1.0 - _P_LOW


def norm_cdf(x: Union[float, Sequence[float], np.ndarray]):
    """Standard normal CDF, via the error function."""
    arr = np.asarray(x, dtype=float)
    out = 0.5 * (1.0 + _erf(arr / math.sqrt(2.0)))
    return float(out) if np.isscalar(x) or arr.ndim == 0 else out


def norm_sf(x: Union[float, Sequence[float], np.ndarray]):
    """Upper tail, 1 - cdf(x). Written directly to stay accurate far out."""
    arr = np.asarray(x, dtype=float)
    out = 0.5 * _erfc(arr / math.sqrt(2.0))
    return float(out) if np.isscalar(x) or arr.ndim == 0 else out


def norm_ppf(p: Union[float, Sequence[float], np.ndarray]):
    """Inverse standard normal CDF.

    Raises on p outside (0, 1) rather than returning an infinity, because a
    silent infinity in a power calculation becomes a silent "you need 0
    samples" downstream.
    """
    arr = np.asarray(p, dtype=float)
    if np.any(arr <= 0.0) or np.any(arr >= 1.0):
        raise ValueError("norm_ppf requires 0 < p < 1")

    out = np.empty_like(arr)
    low = arr < _P_LOW
    high = arr > _P_HIGH
    mid = ~(low | high)

    if np.any(low):
        q = np.sqrt(-2.0 * np.log(arr[low]))
        out[low] = _tail(q)
    if np.any(high):
        q = np.sqrt(-2.0 * np.log(1.0 - arr[high]))
        out[high] = -_tail(q)
    if np.any(mid):
        q = arr[mid] - 0.5
        r = q * q
        num = (((((_A[0] * r + _A[1]) * r + _A[2]) * r + _A[3]) * r + _A[4]) * r + _A[5]) * q
        den = ((((_B[0] * r + _B[1]) * r + _B[2]) * r + _B[3]) * r + _B[4]) * r + 1.0
        out[mid] = num / den

    # One Halley step against the CDF removes the approximation's residual.
    err = norm_cdf(out) - arr
    pdf = np.exp(-0.5 * out * out) / math.sqrt(2.0 * math.pi)
    out = out - err / (pdf + 1e-300) / (1.0 + out * err / (2.0 * (pdf + 1e-300)))

    return float(out) if np.isscalar(p) or arr.ndim == 0 else out


def _tail(q: np.ndarray) -> np.ndarray:
    num = ((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]
    den = (((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0
    return num / den


def _erf(x: np.ndarray) -> np.ndarray:
    return np.vectorize(math.erf, otypes=[float])(x)


def _erfc(x: np.ndarray) -> np.ndarray:
    return np.vectorize(math.erfc, otypes=[float])(x)
