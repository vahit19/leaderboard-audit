"""Command line: measure, then audit.

    la estimate --items data.jsonl --systems a,b --repeats 3   # cost, no spend
    la run      --items data.jsonl --systems a,b --budget 5     # measure
    la audit    run/scores.csv                                  # analyse

`estimate` exists so nobody discovers the price of a run by paying it. `run`
refuses to start without an explicit --budget for the same reason.
"""
from __future__ import annotations

import argparse
import os
import platform
import sys
import time
from typing import List, Optional

import numpy as np

from . import __version__
from .budget import Budget, estimate_tokens
from .cache import DiskCache
from .data import load_csv
from .env import load_dotenv, loaded_keys
from .judge import JudgePanel, build_scorer
from .providers import (OpenRouterProvider, ProviderError, ReplayProvider,
                        StubProvider)
from .ranks import (homogeneous_subsets, pairwise_comparisons, rank_intervals,
                    tiers)
from .report import (headline, plot_tiers, recommendations, render_table,
                     write_json)
from .run import (DEFAULT_SYSTEM_PROMPT, RunConfig, console_progress,
                  run_evaluation, write_long_csv)
from .tasks import load_csv_items, load_hf, load_jsonl
from .variance import noise_floor, variance_components


# ---------------------------------------------------------------------------
# parsers
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="la",
        description="Measure an evaluation, then audit what its ordering "
                    "supports.",
    )
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="command", required=True)

    _add_audit(sub.add_parser("audit", help="analyse an existing table"))
    _add_run(sub.add_parser("run", help="run models and grade them"))
    _add_run(sub.add_parser("estimate", help="project a run's cost, spend nothing"))
    return p


def _add_audit(p: argparse.ArgumentParser) -> None:
    p.add_argument("csv", help="long table: system,item,score[,run]")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--boot", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results")
    p.add_argument("--top", type=int, default=0,
                   help="also print the N closest pairs")
    p.add_argument("--quiet", action="store_true")


def _add_run(p: argparse.ArgumentParser) -> None:
    p.add_argument("--items", required=True,
                   help="path to a .jsonl/.csv file, or hf:<dataset> to pull "
                        "from the Hugging Face hub")
    p.add_argument("--systems", required=True,
                   help="comma-separated model ids to compare")
    p.add_argument("--limit", type=int, default=None,
                   help="use only the first N items")
    p.add_argument("--prompt-field", default="question")
    p.add_argument("--reference-field", default="answer")
    p.add_argument("--id-field", default=None)
    p.add_argument("--split", default="test", help="hf: split")

    p.add_argument("--scorer", default="exact",
                   help="exact | contains | numeric | judge:<model>")
    p.add_argument("--panel", default=None,
                   help="comma-separated judge models; grades are the panel "
                        "mean and every judge's grade is kept")
    p.add_argument("--judge-temperature", type=float, default=0.0)

    p.add_argument("--answer-repeats", type=int, default=1,
                   help="answers per item (needs --temperature > 0)")
    p.add_argument("--grade-repeats", type=int, default=1,
                   help="gradings per answer; this is what measures the judge")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--system-prompt", default=None,
                   help="instruction sent with every item. The default tells "
                        "the model to answer directly, which suppresses "
                        "step-by-step reasoning and depresses scores on tasks "
                        "that need it -- set it deliberately.")

    p.add_argument("--budget", type=float, default=None,
                   help="hard spend ceiling in USD; required for a live run")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--out", default="run")
    p.add_argument("--cache", default=".cache")
    p.add_argument("--env-file", default=None,
                   help="path to a .env holding the API key, when it does not "
                        "live at the repository root. Read into the process "
                        "environment only; never copied into any artefact.")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--offline", action="store_true",
                   help="replay from cache only; no key, no network, no spend")
    p.add_argument("--stub", action="store_true",
                   help="simulated models and judge; no key, no network, no "
                        "spend, and a known right answer to check against")
    p.add_argument("--stub-judge-noise", type=float, default=0.10,
                   help="stub only: how much repeat gradings disagree")


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "audit":
        return cmd_audit(args)
    if args.command == "run":
        return cmd_run(args, dry=False)
    if args.command == "estimate":
        return cmd_run(args, dry=True)
    return 2


# ---------------------------------------------------------------------------
# run / estimate
# ---------------------------------------------------------------------------

def _load_items(args):
    if args.items.startswith("hf:"):
        return load_hf(args.items[3:], split=args.split,
                       prompt_field=args.prompt_field,
                       reference_field=args.reference_field,
                       id_field=args.id_field, limit=args.limit)
    if args.items.endswith(".jsonl"):
        return load_jsonl(args.items, prompt_field=args.prompt_field,
                          reference_field=args.reference_field,
                          id_field=args.id_field, limit=args.limit)
    return load_csv_items(args.items, prompt_field=args.prompt_field,
                          reference_field=args.reference_field,
                          id_field=args.id_field, limit=args.limit)


def _build_provider(args, budget: Budget):
    cache = DiskCache(args.cache, enabled=True)
    if args.stub:
        # Four true ability levels with three ids each, so the audit has a
        # known answer to be checked against: four tiers, not twelve.
        accuracies = {}
        for base, value in (("strong", 0.86), ("good", 0.74),
                            ("fair", 0.61), ("weak", 0.40)):
            for k in (1, 2, 3):
                accuracies["stub/%s-%d" % (base, k)] = value
        return StubProvider(accuracies=accuracies, accuracy=0.65,
                            judge_noise=args.stub_judge_noise), cache
    if args.offline:
        return ReplayProvider(cache), cache
    return OpenRouterProvider(cache=cache, budget=budget), cache


def cmd_run(args, dry: bool) -> int:
    load_dotenv(args.env_file)
    items = _load_items(args)
    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    if len(systems) < 2:
        print("Need at least 2 systems to compare.", file=sys.stderr)
        return 2

    budget = Budget(args.budget)

    if not dry and not args.stub and not args.offline and args.budget is None:
        print("Refusing to start a live run without --budget. This loop calls "
              "a paid API once per cell; run `la estimate ...` first.",
              file=sys.stderr)
        return 2

    try:
        provider, cache = _build_provider(args, budget)
    except ProviderError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if args.panel:
        judges = [build_scorer("judge:" + m.strip(), provider,
                               args.judge_temperature)
                  for m in args.panel.split(",") if m.strip()]
        scorer = JudgePanel(judges)
    else:
        scorer = build_scorer(args.scorer, provider, args.judge_temperature)

    try:
        config = RunConfig(
            systems=systems, items=items, scorer=scorer,
            answer_repeats=args.answer_repeats,
            grade_repeats=args.grade_repeats,
            temperature=args.temperature, max_tokens=args.max_tokens,
            system_prompt=(args.system_prompt
                           if args.system_prompt is not None
                           else DEFAULT_SYSTEM_PROMPT),
            concurrency=args.concurrency, out_dir=args.out,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if dry:
        return _estimate(config, provider, items, systems, args)

    print("Running %d cells: %d systems x %d items x %d answer x %d grade"
          % (config.total_cells, len(systems), len(items),
             args.answer_repeats, args.grade_repeats))
    print("  budget ceiling: %s"
          % ("$%.2f" % args.budget if args.budget else "none (stub/offline)"))

    result = run_evaluation(config, provider, budget,
                            resume=not args.no_resume,
                            progress=console_progress())

    scores_csv = os.path.join(args.out, "scores.csv")
    written, skipped = write_long_csv(result.rows, scores_csv)
    write_json(os.path.join(args.out, "run_summary.json"),
               {"config": config.describe(), "result": result.summary(),
                "cache": cache.stats()})

    print("\nRun complete")
    summary = result.summary()
    print("  %d rows written to %s" % (written, scores_csv))
    if skipped:
        print("  %d row(s) skipped" % skipped)
    if summary["errors"]:
        print("  %d cell(s) errored; see %s"
              % (summary["errors"], os.path.join(args.out, "errors.jsonl")))
    if summary["unparsed_grades"]:
        print("  %d grade(s) did not follow the required output format"
              % summary["unparsed_grades"])
    b = summary["budget"]
    print("  spent $%.4f over %d billed calls (%d served from cache)"
          % (b["spent_usd"], b["billed_calls"], b["cached_calls"]))
    if result.stopped_early:
        print("\n  STOPPED EARLY: " + result.stop_reason)
        print("  Completed work is cached. Raise --budget and re-run to "
              "continue where it left off.")
        return 1

    print("\nNext: la audit %s" % scores_csv)
    return 0


def _estimate(config, provider, items, systems, args) -> int:
    """Project the cost without making a single billable call."""
    sample = items[: min(20, len(items))]
    avg_prompt = int(np.mean([estimate_tokens(i.prompt) for i in sample]))
    answer_tokens = args.max_tokens // 2      # answers rarely fill the cap

    print("Cost estimate (no calls made)")
    print("  %d cells: %d systems x %d items x %d answer x %d grade"
          % (config.total_cells, len(systems), len(items),
             args.answer_repeats, args.grade_repeats))
    print("  ~%d prompt tokens per item" % avg_prompt)

    total = 0.0
    unpriced: List[str] = []
    for model in systems:
        price = provider.price_for(model)
        if price is None:
            unpriced.append(model)
            continue
        calls = len(items) * args.answer_repeats
        cost = price.cost(avg_prompt * calls, answer_tokens * calls)
        total += cost
        print("  %-42s ~$%.3f" % (model, cost))

    if args.panel or args.scorer.startswith("judge:"):
        judge_models = ([m.strip() for m in args.panel.split(",")]
                        if args.panel else [args.scorer[len("judge:"):]])
        grade_calls = (len(items) * len(systems) * args.answer_repeats
                       * args.grade_repeats)
        for model in judge_models:
            price = provider.price_for(model)
            if price is None:
                unpriced.append(model)
                continue
            cost = price.cost((avg_prompt + answer_tokens) * grade_calls,
                              16 * grade_calls)
            total += cost
            print("  %-42s ~$%.3f  (grading)" % (model, cost))

    print("\n  projected total: ~$%.2f" % total)
    if unpriced:
        print("  no price available for: %s" % ", ".join(sorted(set(unpriced))))
        print("  the projection above excludes them.")
    print("\n  This is an estimate from a crude token count. Run with "
          "--budget %.0f to cap the real spend." % max(1.0, total * 1.5))
    return 0


# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------

def cmd_audit(args) -> int:
    started = time.time()
    table = load_csv(args.csv)
    described = table.describe()

    complete = table.complete()
    if complete.n_items < 2:
        print("Only %d item(s) are shared by every system; a paired audit "
              "needs at least 2." % complete.n_items, file=sys.stderr)
        return 2
    if complete.n_systems < 2:
        print("Need at least 2 systems.", file=sys.stderr)
        return 2

    scores, systems = complete.scores, complete.systems
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
    floor = noise_floor((var or {}).get("residual_var"), complete.n_items,
                        (var or {}).get("mean_repeats", 1.0) or 1.0) if var else None
    advice = recommendations(tier_result, comparisons, complete.n_items, var)

    manifest = {
        "input": described, "alpha": args.alpha,
        "bootstrap_replicates": args.boot, "seed": args.seed,
        "items_used": complete.n_items,
        "items_dropped_incomplete": table.n_items - complete.n_items,
        "pairwise_tests": len(comparisons), "noise_floor": floor,
        "tool_version": __version__,
        "python": sys.version.split()[0], "numpy": np.__version__,
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
    figure = plot_tiers(rank_rows, tier_result, os.path.join(out, "ranks.png"))

    if not args.quiet:
        _print_audit(table, complete, rank_rows, comparisons, tier_result,
                     var, floor, advice, manifest, figure, out, args)
    return 0


def _print_audit(table, complete, rank_rows, comparisons, tier_result, var,
                 floor, advice, manifest, figure, out, args) -> None:
    bar = "=" * 78
    print("\n" + bar)
    print("LEADERBOARD AUDIT  |  %s" % (table.source or args.csv))
    print(bar)
    print("\n  " + headline(tier_result))

    print("\nInput")
    dropped = table.n_items - complete.n_items
    print("  %d systems x %d items%s"
          % (complete.n_systems, table.n_items,
             "  (%d dropped: not scored by every system)" % dropped
             if dropped else ""))
    if complete.has_repeats:
        print("  repeat runs: %d-%d per cell"
              % (manifest["input"]["min_repeats"],
                 manifest["input"]["max_repeats"]))
    else:
        print("  no repeat runs: measurement noise cannot be separated")

    print("\nRanks (interval = 95% bootstrap over items)")
    tier_of = {a["system"]: a["tier"] for a in tier_result["assignment"]}
    print(render_table(
        [{"system": r["system"], "score": r["score"], "rank": r["rank"],
          "rank_95%": "%d-%d" % (r["rank_low"], r["rank_high"]),
          "span": r["rank_span"], "tier": tier_of.get(r["system"], "-")}
         for r in rank_rows],
        ["system", "score", "rank", "rank_95%", "span", "tier"]))

    print("\nTiers (same tier = not distinguishable at alpha=%.2f)" % args.alpha)
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
        print(render_table(closest, ["comparison", "delta", "ci_low", "ci_high",
                                     "mde", "holm_p", "distinguishable"]))

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
        print("\nWhere the variance comes from\n  " + var["note"])

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
