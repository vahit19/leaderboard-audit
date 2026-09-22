"""Generate an example leaderboard with a known answer.

The point of a synthetic example is not realism, it is that the truth is known.
Here twelve systems are drawn from four genuinely distinct ability levels, with
three systems per level that differ by nothing at all. A correct audit should
recover roughly four tiers and should not claim to separate the systems inside
a level, however far apart their observed scores land.

Usage:
    python examples/make_example.py --out examples/example.csv
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np

# Four true ability levels, three systems each. Systems inside a level are
# identical by construction: any ordering among them is noise.
TRUE_LEVELS = {
    "strong": 0.78,
    "good": 0.71,
    "fair": 0.62,
    "weak": 0.44,
}
PER_LEVEL = 3


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--items", type=int, default=60,
                   help="tasks in the benchmark")
    p.add_argument("--runs", type=int, default=3,
                   help="repeat runs per system per item")
    p.add_argument("--item-spread", type=float, default=0.16,
                   help="how much item difficulty varies (hard for everyone)")
    p.add_argument("--run-noise", type=float, default=0.11,
                   help="grader and sampling noise between repeat runs")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="examples/example.csv")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    rng = np.random.default_rng(args.seed)

    systems = []
    for level, ability in TRUE_LEVELS.items():
        for k in range(1, PER_LEVEL + 1):
            systems.append(("%s-%d" % (level, k), ability))

    # Item difficulty shifts every system together. It inflates raw spread but
    # cancels under pairing, which is exactly what the audit should show.
    item_effect = rng.normal(0.0, args.item_spread, size=args.items)

    rows = []
    for item_idx in range(args.items):
        for name, ability in systems:
            base = ability + item_effect[item_idx]
            for run in range(1, args.runs + 1):
                score = base + rng.normal(0.0, args.run_noise)
                rows.append({
                    "system": name,
                    "item": "task_%03d" % item_idx,
                    "run": run,
                    "score": round(float(np.clip(score, 0.0, 1.0)), 4),
                })

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["system", "item", "run", "score"])
        writer.writeheader()
        writer.writerows(rows)

    print("wrote %s" % args.out)
    print("  %d systems, %d items, %d runs each = %d rows"
          % (len(systems), args.items, args.runs, len(rows)))
    print("  ground truth: %d levels (%s), %d systems per level"
          % (len(TRUE_LEVELS), ", ".join(TRUE_LEVELS), PER_LEVEL))
    print("  a correct audit recovers about %d tiers, not %d"
          % (len(TRUE_LEVELS), len(systems)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
