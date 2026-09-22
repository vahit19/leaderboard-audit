"""Power: the smallest difference a table could have detected, and how many
items it would take to detect the difference you care about.

This is the question a leaderboard never answers. A table reports that A beat B
by 1.2 points. Whether the evaluation could have detected 1.2 points at all is
a property of the item count and the variance, and it is knowable before any
model is run.
"""
from __future__ import annotations

import math
from typing import Dict, Sequence

import numpy as np

from .normal import norm_ppf

__all__ = ["paired_sd", "minimum_detectable_effect", "required_items",
           "power_at_effect"]


def paired_sd(a: Sequence[float], b: Sequence[float]) -> float:
    """Standard deviation of the per-item differences between two systems.

    This, not the spread of the scores themselves, is what sets resolution.
    Two systems that are both erratic but erratic together are easy to tell
    apart; two steady systems that disagree unpredictably are not.
    """
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if d.size < 2:
        return float("nan")
    return float(np.std(d, ddof=1))


def minimum_detectable_effect(
    n_items: int,
    sd_diff: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> float:
    """Smallest true difference detectable at the given alpha and power.

    Normal approximation to the paired test. It is the standard formula and it
    is slightly optimistic for small n, which is stated rather than hidden: the
    real threshold is a little higher than this, never lower, so a comparison
    this call already flags as underpowered certainly is.
    """
    if n_items < 2 or not np.isfinite(sd_diff) or sd_diff <= 0:
        return float("nan")
    z_alpha = norm_ppf(1.0 - alpha / 2.0)
    z_power = norm_ppf(power)
    return float((z_alpha + z_power) * sd_diff / math.sqrt(n_items))


def required_items(
    effect: float,
    sd_diff: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> float:
    """Items needed to detect `effect` at the given alpha and power."""
    if effect <= 0 or not np.isfinite(sd_diff) or sd_diff <= 0:
        return float("nan")
    z_alpha = norm_ppf(1.0 - alpha / 2.0)
    z_power = norm_ppf(power)
    return float(math.ceil(((z_alpha + z_power) * sd_diff / effect) ** 2))


def power_at_effect(
    n_items: int,
    effect: float,
    sd_diff: float,
    alpha: float = 0.05,
) -> float:
    """Probability of detecting a true difference of `effect` with `n_items`."""
    from .normal import norm_cdf
    if n_items < 2 or sd_diff <= 0 or not np.isfinite(sd_diff):
        return float("nan")
    z_alpha = norm_ppf(1.0 - alpha / 2.0)
    lam = abs(effect) * math.sqrt(n_items) / sd_diff
    return float(norm_cdf(lam - z_alpha) + norm_cdf(-lam - z_alpha))


def resolution_report(
    a: Sequence[float],
    b: Sequence[float],
    alpha: float = 0.05,
    power: float = 0.80,
) -> Dict[str, float]:
    """MDE and required-n for one pair, from the observed paired spread."""
    a_arr = np.asarray(a, dtype=float)
    sd = paired_sd(a_arr, b)
    n = int(a_arr.size)
    observed = float(np.mean(a_arr - np.asarray(b, dtype=float)))
    return {
        "n_items": n,
        "observed_delta": observed,
        "sd_diff": sd,
        "mde": minimum_detectable_effect(n, sd, alpha, power),
        "power_at_observed": power_at_effect(n, observed, sd, alpha),
        "items_for_observed": required_items(abs(observed), sd, alpha, power),
    }
