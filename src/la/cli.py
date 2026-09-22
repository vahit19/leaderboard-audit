"""Command line entry point.

    python -m la.cli results.csv
    python -m la.cli results.csv --alpha 0.01 --out audit/
"""
from __future__ import annotations

import argparse
import os
import platform
import sys
import time
from typing import Optional

import numpy as np

from .data import load_csv
from .ranks import (homogeneous_subsets, pairwise_comparisons, rank_intervals,
                    tiers)
from .report import (headline, plot_tiers, recommendations, render_table,
                     write_json)
from .variance import noise_floor, variance_components


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="leaderboard-audit",
        description="How much of a leaderboard's ordering is supported by its "
                    "own data.",
    )
    p.add_argument("csv", help="long table: system,item,score[,run]")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--power", type=float, default=0.80,
                   help="target power for the resolution calculations")
    p.add_argument("--boot", type=int, default=10_000,
                   help="bootstrap replicates for rank intervals")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results")
    p.add_argument("--top", type=int, default=0,
                   help="limit the printed pairwise table to the N closest "
                        "pairs (0 prints none; the JSON always has all)")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    started = time.time()

    table = load_csv(args.csv)
    described = table.describe()

    complete = table.complete()
    if complete.n_items < 2:
        print("Only %d item(s) are shared by every system. A paired audit "
              "needs at least 2." % complete.n_items, file=sys.stderr)
        return 2
    if complete.n_systems < 2:
        print("Need at least 2 systems.", file=sys.stderr)
        return 2

    scores = complete.scores
    systems = complete.systems

    rank_rows = rank_intervals(scores, systems, n_boot=args.boot,
                               alpha=args.alpha, seed=args.seed)
    comparisons = pairwise_comparisons(scores, systems, alpha=args.alpha,
                                       seed=args.seed)
    tier_result = tiers(scores, systems, alpha=args.alpha, seed=args.seed)
    ordered = [a["system"] for a in tier_result["assignment"]]
    subsets = homogeneous_subsets(comparisons, ordered)

    try:
        var = variance_components(complete.repeats, systems, complete.items)
    except ValueError:
        var = None
    floor = noise_floor(
        (var or {}).get("residual_var"),
        complete.n_items,
        (var or {}).get("mean_repeats", 1.0) or 1.0,
    ) if var else None

    advice = recommendations(tier_result, comparisons, complete.n_items, var)

    manifest = {
        "input": described,
        "alpha": args.alpha,
        "power_target": args.power,
        "bootstrap_replicates": args.boot,
        "seed": args.seed,
        "items_used": complete.n_items,
        "items_dropped_incomplete": table.n_items - complete.n_items,
        "pairwise_tests": len(comparisons),
        "noise_floor": floor,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "wall_clock_seconds": round(time.time() - started, 2),
    }

    out = args.out
    write_json(os.path.join(out, "ranks.json"), rank_rows)
    write_json(os.path.join(out, "pairwise.json"), comparisons)
    write_json(os.path.join(out, "tiers.json"), tier_result)
    write_json(os.path.join(out, "subsets.json"), subsets)
    write_json(os.path.join(out, "variance.json"), var)
    write_json(os.path.join(out, "manifest.json"), manifest)
    figure = plot_tiers(rank_rows, tier_result,
                        os.path.join(out, "ranks.png"))

    if not args.quiet:
        _print(table, complete, rank_rows, comparisons, tier_result, subsets,
               var, floor, advice, manifest, figure, out, args)
    return 0


def _print(table, complete, rank_rows, comparisons, tier_result, subsets,
           var, floor, advice, manifest, figure, out, args) -> None:
    bar = "=" * 78
    print("\n" + bar)
    print("LEADERBOARD AUDIT  |  %s" % (table.source or args.csv))
    print(bar)
    print("\n  " + headline(tier_result))

    print("\nInput")
    print("  %d systems x %d items" % (complete.n_systems, table.n_items), end="")
    dropped = table.n_items - complete.n_items
    if dropped:
        print("  (%d item(s) dropped: not scored by every system)" % dropped)
    else:
        print()
    if complete.has_repeats:
        print("  repeat runs present: %d-%d per cell"
              % (manifest["input"]["min_repeats"],
                 manifest["input"]["max_repeats"]))
    else:
        print("  no repeat runs: measurement noise cannot be separated")

    print("\nRanks (bar = 95% bootstrap interval over items)")
    tier_of = {a["system"]: a["tier"] for a in tier_result["assignment"]}
    rows = []
    for r in rank_rows:
        rows.append({
            "system": r["system"],
            "score": r["score"],
            "rank": r["rank"],
            "rank_95%": "%d-%d" % (r["rank_low"], r["rank_high"]),
            "span": r["rank_span"],
            "tier": tier_of.get(r["system"], "-"),
        })
    print(render_table(rows, ["system", "score", "rank", "rank_95%", "span",
                              "tier"]))

    print("\nTiers (systems in one tier are not distinguishable at alpha=%.2f)"
          % args.alpha)
    current, line = None, []
    for a in tier_result["assignment"]:
        if a["tier"] != current:
            if line:
                print("  tier %d: %s" % (current, ", ".join(line)))
            current, line = a["tier"], []
        line.append(a["system"])
    if line:
        print("  tier %d: %s" % (current, ", ".join(line)))

    if args.top:
        closest = sorted(comparisons, key=lambda c: abs(c["delta"]))[:args.top]
        print("\nClosest %d pairs" % len(closest))
        print(render_table(closest, ["comparison", "delta", "ci_low",
                                     "ci_high", "mde", "holm_p",
                                     "distinguishable"]))

    if var and var.get("measurable"):
        print("\nWhere the variance comes from")
        print("  items      %5.1f%%  (hard for everyone; cancels in pairing)"
              % (100 * (var["item_share"] or 0)))
        print("  systems    %5.1f%%  (the part you want to measure)"
              % (100 * (var["system_share"] or 0)))
        print("  repeat run %5.1f%%  (same system, same item, different score)"
              % (100 * (var["residual_share"] or 0)))
        print("  " + var["note"])
        if floor:
            print("  noise floor: differences below %.3f are not resolvable "
                  "at this item count." % floor)
    elif var:
        print("\nWhere the variance comes from")
        print("  " + var["note"])

    if advice:
        print("\nWhat would make this table say more")
        for line_ in advice:
            print("  - " + line_)

    print("\nArtifacts in %s/" % out)
    for name in ("ranks.json", "pairwise.json", "tiers.json", "subsets.json",
                 "variance.json", "manifest.json"):
        print("  " + os.path.join(out, name))
    if figure:
        print("  " + figure)
    print()


if __name__ == "__main__":
    raise SystemExit(main())
