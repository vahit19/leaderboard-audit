"""Exact nonparametric tests and multiplicity control, numpy only.

Leaderboards compare systems on the same items, so every comparison here is
paired. The tests are exact combinatorial tail probabilities wherever the
design is small enough to enumerate, and a signed permutation test above that,
because normal approximations misreport in exactly the small-n regime where
benchmark tables live.

Every test reports the smallest two-sided p-value its design can reach. That
number is the honest answer to "could this comparison ever have come out
significant", and it belongs next to the p-value, not in a footnote.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "attainable_p_floor",
    "wilcoxon_signed_rank",
    "sign_test",
    "cliffs_delta",
    "holm",
    "bootstrap_ci",
    "paired_summary",
]

#: Above this many pairs, exact enumeration is replaced by a signed
#: permutation test. 2**22 sign patterns is about 4M rows, which is the point
#: where enumeration stops being the cheaper option.
EXACT_LIMIT = 22


def attainable_p_floor(n_pairs: int) -> float:
    """Smallest two-sided p a paired design with `n_pairs` items can reach.

    Attained when every item moves the same way. Ties and zero differences only
    raise it, so this is an optimistic bound and is the right thing to check
    before drawing a conclusion from a short table.
    """
    if n_pairs < 1:
        return 1.0
    return min(1.0, 2.0 / (2.0 ** n_pairs))


def wilcoxon_signed_rank(
    x: Sequence[float],
    y: Sequence[float],
    n_permutations: int = 100_000,
    seed: int = 0,
) -> Dict[str, float]:
    """Two-sided Wilcoxon signed-rank test for paired samples.

    Exact by full enumeration when the number of non-zero differences is at
    most EXACT_LIMIT, and a seeded signed permutation test above it. The
    returned dict says which was used, because "p = .03" from 100k
    permutations is a different claim from an exact .03.
    """
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired samples must have equal length")

    d = a - b
    n_all = int(d.size)
    d = d[d != 0.0]
    n = int(d.size)

    if n == 0:
        return {
            "statistic": float("nan"), "p_value": 1.0, "n_pairs": 0,
            "n_dropped_zero": n_all, "p_floor": 1.0, "method": "degenerate",
        }

    ranks = _average_ranks(np.abs(d))
    total = float(ranks.sum())
    w_plus = float(ranks[d > 0].sum())
    statistic = min(w_plus, total - w_plus)

    if n <= EXACT_LIMIT:
        bits = _bit_matrix(n)
        null_w = bits @ ranks
        method = "exact"
    else:
        rng = np.random.default_rng(seed)
        bits = rng.integers(0, 2, size=(n_permutations, n)).astype(float)
        null_w = bits @ ranks
        method = "permutation(%d)" % n_permutations

    null_min = np.minimum(null_w, total - null_w)
    p = float((null_min <= statistic + 1e-12).mean())

    return {
        "statistic": statistic,
        "p_value": min(1.0, p),
        "n_pairs": n,
        "n_dropped_zero": n_all - n,
        "p_floor": attainable_p_floor(n),
        "method": method,
    }


def sign_test(x: Sequence[float], y: Sequence[float]) -> Dict[str, float]:
    """Exact two-sided sign test. Uses only the direction of each difference.

    Weaker than signed-rank, and deliberately offered: when scores are ordinal
    or capped, differences are not on a meaningful interval scale and ranking
    their magnitudes assumes more than the data supports.
    """
    d = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    d = d[d != 0.0]
    n = int(d.size)
    if n == 0:
        return {"p_value": 1.0, "n_pairs": 0, "n_positive": 0,
                "p_floor": 1.0, "method": "degenerate"}

    k = int((d > 0).sum())
    from math import comb
    tail = min(k, n - k)
    p = 2.0 * sum(comb(n, i) for i in range(tail + 1)) / (2.0 ** n)
    return {
        "p_value": min(1.0, p), "n_pairs": n, "n_positive": k,
        "p_floor": attainable_p_floor(n), "method": "exact",
    }


def cliffs_delta(x: Sequence[float], y: Sequence[float]) -> float:
    """Cliff's delta in [-1, 1]. Magnitude 1 means complete rank separation."""
    a = np.asarray(x, dtype=float)[:, None]
    b = np.asarray(y, dtype=float)[None, :]
    return float(np.sign(a - b).mean())


def holm(
    p_values: Sequence[float],
    labels: Optional[Sequence[str]] = None,
) -> List[Tuple[str, float, float]]:
    """Holm-Bonferroni step-down correction, returned in the caller's order.

    A leaderboard of k systems has k(k-1)/2 pairwise comparisons, so for 31
    systems that is 465 tests. Without correction, roughly 23 of them come out
    significant at .05 on pure noise. This is the difference between a ranking
    and a list.
    """
    p = np.asarray(p_values, dtype=float)
    m = int(p.size)
    if labels is None:
        labels = ["c%d" % i for i in range(m)]
    if len(labels) != m:
        raise ValueError("labels and p_values must match in length")

    order = np.argsort(p, kind="mergesort")
    adj = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * float(p[idx]))
        adj[idx] = min(1.0, running)
    return [(labels[i], float(p[i]), float(adj[i])) for i in range(m)]


def bootstrap_ci(
    values: Sequence[float],
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Tuple[float, float]:
    """Seeded percentile bootstrap CI of the mean."""
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    means = v[idx].mean(axis=1)
    return (float(np.quantile(means, alpha / 2.0)),
            float(np.quantile(means, 1.0 - alpha / 2.0)))


def paired_summary(
    name: str,
    treatment: Sequence[float],
    control: Sequence[float],
    seed: int = 0,
) -> Dict[str, object]:
    """Everything one paired comparison should report, in one row."""
    t = np.asarray(treatment, dtype=float)
    c = np.asarray(control, dtype=float)
    test = wilcoxon_signed_rank(t, c, seed=seed)
    diffs = t - c
    lo, hi = bootstrap_ci(diffs, seed=seed)
    return {
        "comparison": name,
        "delta": float(diffs.mean()),
        "ci_low": lo,
        "ci_high": hi,
        "cliffs_delta": cliffs_delta(t, c),
        "p_value": float(test["p_value"]),
        "p_floor": float(test["p_floor"]),
        "n_pairs": int(test["n_pairs"]),
        "method": test["method"],
    }


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------

def _average_ranks(v: np.ndarray) -> np.ndarray:
    order = np.argsort(v, kind="mergesort")
    ranks = np.empty(v.size, dtype=float)
    ranks[order] = np.arange(1, v.size + 1, dtype=float)
    sorted_v = v[order]
    i = 0
    while i < sorted_v.size:
        j = i
        while j + 1 < sorted_v.size and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    return ranks


def _bit_matrix(n: int) -> np.ndarray:
    idx = np.arange(2 ** n, dtype=np.int64)[:, None]
    bit = np.arange(n, dtype=np.int64)[None, :]
    return ((idx >> bit) & 1).astype(float)
