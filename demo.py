"""Run the whole pipeline end to end, offline, in one command.

    python demo.py

Measures twelve simulated systems on a real item file with a model judge,
grading every answer three times, then audits the result. No API key, no
network, no spend -- and a known right answer to check the audit against.

The simulated systems come from four true ability levels, three ids each. The
audit is not told this. What it reports is what the measurement supports, which
at this item count is fewer than four tiers: a binary judge over 40 items
cannot separate abilities twelve points apart, and saying so is the point.

Pass --live to run the same pipeline against real models through OpenRouter.
That spends money and requires OPENROUTER_API_KEY, so it asks first.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
ITEMS = os.path.join(ROOT, "examples", "arithmetic.jsonl")

STUB_SYSTEMS = ",".join(
    "stub/%s-%d" % (level, k)
    for level in ("strong", "good", "fair", "weak")
    for k in (1, 2, 3)
)


def run(args, env=None):
    print("\n$ " + " ".join(args[1:]))
    result = subprocess.run(args, cwd=ROOT, env=env)
    if result.returncode not in (0,):
        raise SystemExit(result.returncode)


def la(env, *args):
    run([sys.executable, "-m", "la.cli", *args], env=env)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--live", action="store_true",
                   help="use real models through OpenRouter (costs money)")
    p.add_argument("--systems", default="openai/gpt-4o-mini,"
                                        "anthropic/claude-3.5-haiku,"
                                        "google/gemini-flash-1.5",
                   help="--live only: models to compare")
    p.add_argument("--judge", default="openai/gpt-4o-mini",
                   help="--live only: grading model")
    p.add_argument("--budget", type=float, default=2.0,
                   help="--live only: hard spend ceiling in USD")
    p.add_argument("--limit", type=int, default=40)
    args = p.parse_args()

    env = dict(os.environ)
    env["PYTHONPATH"] = SRC + os.pathsep + env.get("PYTHONPATH", "")

    if not os.path.exists(ITEMS):
        print("missing %s" % ITEMS, file=sys.stderr)
        return 2

    if not args.live:
        out = os.path.join(ROOT, "run_demo")
        la(env, "run",
           "--items", ITEMS, "--systems", STUB_SYSTEMS,
           "--scorer", "judge:stub-judge", "--grade-repeats", "3",
           "--stub", "--limit", str(args.limit),
           "--out", out, "--cache", os.path.join(ROOT, ".cache_demo"))
        la(env, "audit", os.path.join(out, "scores.csv"),
           "--out", os.path.join(ROOT, "results"), "--top", "6")
        return 0

    # Live path: estimate, confirm, then run under a ceiling.
    out = os.path.join(ROOT, "run_live")
    la(env, "estimate",
       "--items", ITEMS, "--systems", args.systems,
       "--scorer", "judge:" + args.judge, "--grade-repeats", "3",
       "--limit", str(args.limit), "--out", out)

    answer = input("\nProceed with a live run capped at $%.2f? [y/N] "
                   % args.budget).strip().lower()
    if answer not in ("y", "yes"):
        print("Nothing was spent.")
        return 0

    la(env, "run",
       "--items", ITEMS, "--systems", args.systems,
       "--scorer", "judge:" + args.judge, "--grade-repeats", "3",
       "--limit", str(args.limit), "--budget", str(args.budget),
       "--out", out, "--cache", os.path.join(ROOT, ".cache"))
    la(env, "audit", os.path.join(out, "scores.csv"),
       "--out", os.path.join(ROOT, "results_live"), "--top", "6")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
