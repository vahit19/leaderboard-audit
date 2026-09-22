"""Where the spread in a leaderboard actually comes from.

A score varies for three separable reasons:

  item       some items are hard for everyone. This inflates the spread of raw
             scores but cancels in a paired comparison, so it does not hurt
             resolution -- and mistaking it for noise is how people conclude a
             benchmark is hopeless when it is fine.

  system     real differences between systems. This is the only part anyone
             wants to measure.

  residual   the same system, on the same item, scored differently on a repeat
             run. Sampling temperature, grader disagreement, harness flakiness.
             This is the part that sets the floor on what can be resolved, and
             it is invisible without repeat runs.

Estimated by the method of moments from a two-way layout with replication. No
scipy, no fitted mixed model: the components are simple functions of the
one-way and two-way mean squares, and a variance component that comes out
negative is reported as zero with a note rather than hidden.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

__all__ = ["variance_components", "noise_floor"]


def variance_components(
    repeats: Dict[Tuple[str, str], List[float]],
    systems: List[str],
    items: List[str],
) -> Dict[str, object]:
    """Split total variance into item, system, and residual parts.

    `repeats` maps (system, item) to the list of scores from each run. Cells
    with a single run contribute to the item and system terms but not to the
    residual, so a table with no repeats anywhere returns residual = None
    rather than zero: unmeasured is not the same as absent.
    """
    n_rep = [len(v) for v in repeats.values()]
    if not n_rep:
        raise ValueError("no cells")

    if max(n_rep) < 2:
        return {
            "item_var": None, "system_var": None, "residual_var": None,
            "measurable": False,
            "note": (
                "every cell has a single run, so measurement noise cannot be "
                "separated from real differences. Re-run at least a subset of "
                "items twice; that subset alone fixes the residual term."
            ),
        }

    # cell means and within-cell sums of squares
    cell_mean = np.full((len(items), len(systems)), np.nan)
    ss_within = 0.0
    df_within = 0
    i_index = {it: i for i, it in enumerate(items)}
    s_index = {s: j for j, s in enumerate(systems)}

    for (s, it), values in repeats.items():
        if s not in s_index or it not in i_index:
            continue
        arr = np.asarray(values, dtype=float)
        cell_mean[i_index[it], s_index[s]] = arr.mean()
        if arr.size > 1:
            ss_within += float(((arr - arr.mean()) ** 2).sum())
            df_within += arr.size - 1

    residual_var = ss_within / df_within if df_within > 0 else None

    valid = ~np.isnan(cell_mean)
    grand = float(np.nanmean(cell_mean))
    item_means = np.nanmean(cell_mean, axis=1)
    system_means = np.nanmean(cell_mean, axis=0)

    item_var = float(np.nanvar(item_means, ddof=1)) if valid.any() else 0.0
    system_var = float(np.nanvar(system_means, ddof=1)) if valid.any() else 0.0

    # The observed spread of group means carries a share of the residual; take
    # it back out so the components are not double counted.
    mean_reps = float(np.mean([len(v) for v in repeats.values() if len(v) >= 1]))
    if residual_var is not None and mean_reps > 0:
        item_var = max(0.0, item_var - residual_var / (len(systems) * mean_reps))
        system_var = max(0.0, system_var - residual_var / (len(items) * mean_reps))

    total = (item_var or 0.0) + (system_var or 0.0) + (residual_var or 0.0)
    share = (lambda v: float(v / total) if total > 0 and v is not None else None)

    return {
        "item_var": item_var,
        "system_var": system_var,
        "residual_var": residual_var,
        "item_share": share(item_var),
        "system_share": share(system_var),
        "residual_share": share(residual_var),
        "grand_mean": grand,
        "mean_repeats": mean_reps,
        "measurable": True,
        "note": _interpret(share(system_var), share(residual_var)),
    }


def noise_floor(
    residual_var: Optional[float],
    n_items: int,
    n_repeats: float = 1.0,
) -> Optional[float]:
    """Smallest difference repeat-run noise alone would let you resolve.

    Two systems compared on `n_items` items cannot be separated below roughly
    this gap, however many systems are added to the table. Adding systems does
    not buy resolution; adding items or repeat runs does.

    `n_repeats` matters: a cell score is the mean of its runs, so averaging k
    runs cuts the residual each cell carries by k. Leaving it out reports a
    floor that is sqrt(k) too high, which would wrongly condemn tables that
    already paid for repeats.
    """
    if residual_var is None or residual_var <= 0 or n_items < 1:
        return None
    per_cell = residual_var / max(1.0, float(n_repeats))
    return float(2.8 * np.sqrt(2.0 * per_cell / n_items))


def _interpret(system_share: Optional[float],
               residual_share: Optional[float]) -> str:
    if system_share is None or residual_share is None:
        return "not enough structure to interpret"
    if residual_share > system_share:
        return (
            "repeat-run noise exceeds the spread between systems (%.0f%% vs "
            "%.0f%%). Differences in this table are mostly measurement, and "
            "ranking it more finely will not help." %
            (100 * residual_share, 100 * system_share)
        )
    if residual_share > 0.5 * system_share:
        return (
            "repeat-run noise is %.0f%% of variance against %.0f%% for real "
            "system differences. Close ranks are not trustworthy; the wide "
            "gaps are." % (100 * residual_share, 100 * system_share)
        )
    return (
        "system differences (%.0f%%) dominate repeat-run noise (%.0f%%). The "
        "ordering is carrying real signal." %
        (100 * system_share, 100 * residual_share)
    )
