"""How much of a leaderboard's ordering is supported by its own data.

Three answers, from cheapest to strictest:

  rank_intervals   resample the items and watch where each system lands. A
                   system whose rank swings from 3rd to 19th does not have a
                   rank, it has a range.

  pairwise         every pair of systems compared on the items they share,
                   Holm-corrected across the whole pairwise family. For 31
                   systems that family is 465 tests, and uncorrected it would
                   hand back about 23 significant results on pure noise.

  tiers            collapse the ordering into groups that the data can tell
                   apart. This is the headline number: "31 systems, 6 tiers".

The resampling unit is the item, not the score. Items are what a benchmark has
a finite number of, and generalising to "other items like these" is the claim a
leaderboard is implicitly making.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .stats.exact import holm, paired_summary
from .stats.power import minimum_detectable_effect, paired_sd

__all__ = ["rank_intervals", "pairwise_comparisons", "tiers",
           "homogeneous_subsets", "nontransitive_pairs"]


def rank_intervals(
    scores: np.ndarray,
    systems: Sequence[str],
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> List[Dict[str, object]]:
    """Bootstrap each system's rank by resampling items.

    `scores` is (n_items, n_systems). Items are resampled together across all
    systems, which keeps the pairing: every bootstrap replicate is still a
    complete table, so an item that is hard for everyone stays hard for
    everyone. Resampling scores independently per system would break that and
    understate the uncertainty.

    Rank 1 is best.
    """
    scores = np.asarray(scores, dtype=float)
    n_items, n_systems = scores.shape
    if n_systems != len(systems):
        raise ValueError("systems must match the number of columns")
    if n_items < 2:
        raise ValueError("need at least 2 items to resample")

    rng = np.random.default_rng(seed)
    observed_mean = np.nanmean(scores, axis=0)
    observed_rank = _ranks_desc(observed_mean)

    boot_ranks = np.empty((n_boot, n_systems), dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n_items, size=n_items)
        boot_ranks[b] = _ranks_desc(np.nanmean(scores[idx], axis=0))

    lo = np.quantile(boot_ranks, alpha / 2.0, axis=0)
    hi = np.quantile(boot_ranks, 1.0 - alpha / 2.0, axis=0)

    rows: List[Dict[str, object]] = []
    for j, name in enumerate(systems):
        rows.append({
            "system": name,
            "score": float(observed_mean[j]),
            "rank": int(observed_rank[j]),
            "rank_low": int(np.floor(lo[j])),
            "rank_high": int(np.ceil(hi[j])),
            "rank_span": int(np.ceil(hi[j]) - np.floor(lo[j]) + 1),
            "best_seen": int(boot_ranks[:, j].min()),
            "worst_seen": int(boot_ranks[:, j].max()),
        })
    rows.sort(key=lambda r: r["rank"])
    return rows


def pairwise_comparisons(
    scores: np.ndarray,
    systems: Sequence[str],
    alpha: float = 0.05,
    seed: int = 0,
) -> List[Dict[str, object]]:
    """Every pair, paired over items, Holm-corrected across the whole family."""
    scores = np.asarray(scores, dtype=float)
    n_systems = scores.shape[1]

    rows: List[Dict[str, object]] = []
    for i in range(n_systems):
        for j in range(i + 1, n_systems):
            a, b = scores[:, i], scores[:, j]
            keep = ~(np.isnan(a) | np.isnan(b))
            if keep.sum() < 2:
                continue
            row = paired_summary(
                "%s vs %s" % (systems[i], systems[j]),
                a[keep], b[keep], seed=seed,
            )
            row["system_a"] = systems[i]
            row["system_b"] = systems[j]
            row["sd_diff"] = paired_sd(a[keep], b[keep])
            row["mde"] = minimum_detectable_effect(
                int(keep.sum()), row["sd_diff"], alpha=alpha
            )
            rows.append(row)

    if rows:
        adjusted = holm([float(r["p_value"]) for r in rows],
                        [str(r["comparison"]) for r in rows])
        for row, (_, _, adj) in zip(rows, adjusted):
            row["holm_p"] = adj
            row["distinguishable"] = bool(adj < alpha)
    return rows


def tiers(
    scores: np.ndarray,
    systems: Sequence[str],
    alpha: float = 0.05,
    seed: int = 0,
) -> Dict[str, object]:
    """Collapse the ordering into tiers the data can actually separate.

    Systems are taken best-first. A tier stays open while the next system is
    not distinguishable from the system that opened the tier; when it is, a new
    tier opens. Comparing against the tier leader rather than the previous
    system stops a long chain of individually-tiny gaps from being merged into
    one tier that spans a real difference end to end.

    The result is a partition, so every system sits in exactly one tier and the
    count is a single number. `homogeneous_subsets` gives the stricter,
    overlapping view for readers who want it.
    """
    scores = np.asarray(scores, dtype=float)
    comparisons = pairwise_comparisons(scores, systems, alpha=alpha, seed=seed)
    distinguishable = {
        frozenset((r["system_a"], r["system_b"])): r["distinguishable"]
        for r in comparisons
    }

    order = list(np.argsort(-np.nanmean(scores, axis=0), kind="mergesort"))
    names = [systems[i] for i in order]
    means = np.nanmean(scores, axis=0)

    assignment: List[Dict[str, object]] = []
    tier_index = 0
    leader: Optional[str] = None
    for name in names:
        if leader is None:
            tier_index = 1
            leader = name
        elif distinguishable.get(frozenset((leader, name)), False):
            tier_index += 1
            leader = name
        assignment.append({
            "system": name,
            "score": float(means[systems.index(name)]),
            "tier": tier_index,
        })

    return {
        "n_systems": len(systems),
        "n_tiers": tier_index,
        "alpha": alpha,
        "assignment": assignment,
        "nontransitive": nontransitive_pairs(comparisons, names),
    }


def homogeneous_subsets(
    comparisons: Sequence[Dict[str, object]],
    systems_in_order: Sequence[str],
) -> List[List[str]]:
    """Maximal runs of the ordering in which no pair is distinguishable.

    These overlap, which is correct and is the reason they are reported
    alongside `tiers` rather than instead of it: "not distinguishable" is not
    transitive, so a strict grouping of an ordered table cannot be a partition
    without losing something.
    """
    distinguishable = {
        frozenset((r["system_a"], r["system_b"])): r["distinguishable"]
        for r in comparisons
    }

    def all_same(members: Sequence[str]) -> bool:
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                if distinguishable.get(frozenset((members[i], members[j])), False):
                    return False
        return True

    runs: List[List[str]] = []
    n = len(systems_in_order)
    for start in range(n):
        end = start
        while end + 1 < n and all_same(systems_in_order[start:end + 2]):
            end += 1
        run = list(systems_in_order[start:end + 1])
        if not any(set(run).issubset(set(existing)) for existing in runs):
            runs.append(run)
    return runs


def nontransitive_pairs(
    comparisons: Sequence[Dict[str, object]],
    systems_in_order: Sequence[str],
) -> List[Dict[str, str]]:
    """Find orderings the significance pattern cannot support.

    A triple where the 1st and 3rd are indistinguishable while the 1st and 2nd
    are distinguishable is not a contradiction in the data, but it does mean
    the table is not cleanly ordered, and a reader who sees only ranks would
    never know. Reporting these is cheap and they are exactly the places where
    a leaderboard misleads.
    """
    distinguishable = {
        frozenset((r["system_a"], r["system_b"])): r["distinguishable"]
        for r in comparisons
    }
    out: List[Dict[str, str]] = []
    n = len(systems_in_order)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a, b, c = (systems_in_order[i], systems_in_order[j],
                           systems_in_order[k])
                ab = distinguishable.get(frozenset((a, b)), False)
                ac = distinguishable.get(frozenset((a, c)), False)
                if ab and not ac:
                    out.append({"better": a, "middle": b, "worse": c})
    return out


def _ranks_desc(values: np.ndarray) -> np.ndarray:
    """Competition ranks, 1 = highest value. Ties share the better rank."""
    order = np.argsort(-values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    ranks[order] = np.arange(1, values.size + 1, dtype=float)
    sorted_v = values[order]
    i = 0
    while i < sorted_v.size:
        j = i
        while j + 1 < sorted_v.size and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i]]
        i = j + 1
    return ranks
