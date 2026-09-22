"""Turning the analysis into something a reader acts on.

The headline is one sentence: how many systems, how many tiers. Everything
below it exists to let someone check that sentence, or to tell them what to
change so the next table says more.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence

import numpy as np

__all__ = ["headline", "render_table", "write_json", "plot_tiers",
           "recommendations"]


def headline(tier_result: Dict[str, object]) -> str:
    n_systems = int(tier_result["n_systems"])
    n_tiers = int(tier_result["n_tiers"])
    if n_tiers <= 1:
        return ("%d systems, 1 tier: this table does not separate anything."
                % n_systems)
    if n_tiers == n_systems:
        return ("%d systems, %d tiers: every rank is supported."
                % (n_systems, n_tiers))
    return ("%d systems, %d distinguishable tiers: %d of the %d rank gaps are "
            "not supported by the data." %
            (n_systems, n_tiers, n_systems - n_tiers, n_systems - 1))


def recommendations(
    tier_result: Dict[str, object],
    comparisons: Sequence[Dict[str, object]],
    n_items: int,
    variance: Optional[Dict[str, object]] = None,
) -> List[str]:
    """Concrete next actions, or nothing. No filler advice."""
    out: List[str] = []

    n_systems = int(tier_result["n_systems"])
    n_tiers = int(tier_result["n_tiers"])
    unresolved = [c for c in comparisons if not c.get("distinguishable")]

    if unresolved:
        mdes = [c["mde"] for c in unresolved if np.isfinite(c.get("mde", np.nan))]
        if mdes:
            typical = float(np.median(mdes))
            out.append(
                "At %d items this table cannot see differences below about "
                "%.3f. %d of %d pairs fall under that line."
                % (n_items, typical, len(unresolved), len(comparisons))
            )

        # Only pairs with a real gap are worth more items. A pair whose
        # observed difference is essentially zero is not underpowered, it is
        # two systems that perform the same, and quoting an item count for it
        # produces a number in the millions that no one can act on.
        reachable = [
            c for c in unresolved
            if np.isfinite(c.get("mde", np.nan))
            and abs(c["delta"]) > 0
            and (c["mde"] / abs(c["delta"])) ** 2 <= 100.0
        ]
        if reachable:
            worst = max(reachable, key=lambda c: c["mde"] / abs(c["delta"]))
            factor = (worst["mde"] / abs(worst["delta"])) ** 2
            out.append(
                "%d pair(s) are within reach: about %.0fx more items (%d -> "
                "%d) would separate the hardest of them. Adding systems will "
                "not help." % (len(reachable), factor, n_items,
                               int(n_items * factor))
            )
        indistinguishable = len(unresolved) - len(reachable)
        if indistinguishable:
            out.append(
                "%d pair(s) differ by less than their own measurement noise. "
                "No practical item count separates these; report them as tied."
                % indistinguishable
            )

    nontransitive = tier_result.get("nontransitive") or []
    if nontransitive:
        ex = nontransitive[0]
        out.append(
            "%d ordering(s) the significance pattern cannot support, e.g. %s "
            "beats %s but is not separable from %s. Present these as a tier, "
            "not as ranks." % (len(nontransitive), ex["better"], ex["middle"],
                               ex["worse"])
        )

    if variance is not None:
        if not variance.get("measurable"):
            out.append(
                "No repeat runs, so measurement noise is unmeasured. Re-run a "
                "subset of items twice; that alone bounds how much of this "
                "table is noise."
            )
        else:
            residual = variance.get("residual_share")
            if residual is not None and residual > 0.3:
                out.append(
                    "Repeat-run noise is %.0f%% of total variance. Averaging "
                    "over more runs per item buys resolution more cheaply than "
                    "adding items." % (100 * residual)
                )

    underpowered = [c for c in comparisons
                    if c.get("p_floor", 0) > 0.05]
    if underpowered:
        out.append(
            "%d comparison(s) sit on too few shared items to ever reach p<.05, "
            "whatever the result." % len(underpowered)
        )
    return out


def render_table(rows: Sequence[Dict[str, object]],
                 columns: Sequence[str]) -> str:
    """Fixed-width table. No dependency, identical output everywhere."""
    if not rows:
        return "(no rows)"
    widths = []
    for col in columns:
        cells = [_fmt(r.get(col)) for r in rows]
        widths.append(max(len(col), max(len(c) for c in cells)))
    head = "  ".join(c.ljust(w) for c, w in zip(columns, widths))
    rule = "  ".join("-" * w for w in widths)
    body = ["  ".join(_fmt(r.get(c)).ljust(w) for c, w in zip(columns, widths))
            for r in rows]
    return "\n".join([head, rule] + body)


def _fmt(v: object) -> str:
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        if not np.isfinite(v):
            return "-"
        if v != 0 and abs(v) < 0.001:
            return "%.1e" % v
        return "%.3f" % v
    return str(v)


def write_json(path: str, payload: object) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=default)


def plot_tiers(
    rank_rows: Sequence[Dict[str, object]],
    tier_result: Dict[str, object],
    path: str,
    title: str = "Leaderboard ranks with bootstrap intervals",
) -> Optional[str]:
    """One figure: each system's score, its rank interval, and its tier.

    Returns the path, or None if matplotlib is not installed -- the audit is
    complete without it.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    tier_of = {a["system"]: a["tier"] for a in tier_result["assignment"]}
    rows = sorted(rank_rows, key=lambda r: r["rank"])
    names = [r["system"] for r in rows]
    y = np.arange(len(rows))[::-1]

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, max(3.0, 0.32 * len(rows) + 1.4)))

    palette = ["#3b6ea5", "#c25e00", "#3f7d53", "#8a4f9e",
               "#9e3f4f", "#4f7f8a", "#7d6b3f"]
    for row, yy in zip(rows, y):
        tier = tier_of.get(row["system"], 1)
        colour = palette[(tier - 1) % len(palette)]
        ax.plot([row["rank_low"], row["rank_high"]], [yy, yy],
                color=colour, linewidth=3.0, solid_capstyle="round", alpha=0.75)
        ax.plot([row["rank"]], [yy], "o", color=colour, markersize=5.5)

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("rank (1 = best); bar shows the 95% bootstrap interval")
    ax.set_title("%s\n%s" % (title, headline(tier_result)), fontsize=10)
    ax.grid(axis="x", alpha=0.25, linewidth=0.6)
    ax.invert_xaxis()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
