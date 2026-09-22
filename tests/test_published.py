"""Re-derive every number the README claims, from the tables in this repo.

A README states results. Nothing normally stops those numbers drifting from
the data as code changes around them, and a repository whose subject is
"check whether the result holds up" cannot be the one with unchecked results
in its own front page.

So each claim below is recomputed from the shipped `scores.csv` files. If a
change to the analysis moves a published number, this fails and names it.
Nothing here calls the network or needs an API key.
"""
from __future__ import annotations

import collections
import csv
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

from la.compare import compare_graders  # noqa: E402
from la.data import load_csv  # noqa: E402
from la.ranks import tiers  # noqa: E402

TRUTH = os.path.join(ROOT, "run_gsm8k_numeric", "scores.csv")
JUDGES = {
    "gpt-4o-mini": os.path.join(ROOT, "run_gsm8k_judge", "scores.csv"),
    "gemini-2.5-flash-lite": os.path.join(
        ROOT, "run_j_gemini-2.5-flash-lite", "scores.csv"),
    "claude-3-haiku": os.path.join(
        ROOT, "run_j_claude-3-haiku", "scores.csv"),
}

LLAMA70 = "meta-llama/llama-3.3-70b-instruct"
LLAMA8 = "meta-llama/llama-3.1-8b-instruct"
QWEN = "qwen/qwen-2.5-7b-instruct"

#: Rank of llama-3.3-70b under each grader, as the README and the first figure
#: both state it.
PUBLISHED_RANKS = {
    "ground truth": 1,
    "gpt-4o-mini": 5,
    "gemini-2.5-flash-lite": 2,
    "claude-3-haiku": 4,
}


def rows(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def mean_by_system(path, binarise=False):
    """Per-system mean, majority-thresholding repeat grades of one answer."""
    cells = collections.defaultdict(list)
    for row in rows(path):
        cells[(row["system"], row["item"])].append(float(row["score"]))
    per_system = collections.defaultdict(list)
    for (system, _), values in cells.items():
        value = float(np.mean(values))
        per_system[system].append(1.0 if (binarise and value >= 0.5)
                                  else (0.0 if binarise else value))
    return {s: float(np.mean(v)) for s, v in per_system.items()}


def rank_of(path, system, binarise):
    means = mean_by_system(path, binarise=binarise)
    return sorted(means, key=lambda s: -means[s]).index(system) + 1


# -- the headline -----------------------------------------------------------

def test_llama_70b_rank_under_each_grader_matches_the_readme():
    """1st on ground truth, 5th / 2nd / 4th under the three judges."""
    got = {"ground truth": rank_of(TRUTH, LLAMA70, binarise=False)}
    for name, path in JUDGES.items():
        got[name] = rank_of(path, LLAMA70, binarise=True)
    assert got == PUBLISHED_RANKS, got


def test_ground_truth_table_collapses_to_two_tiers():
    """Five of six models are indistinguishable at 200 items."""
    table = load_csv(TRUTH).complete()
    result = tiers(table.scores, table.systems, seed=0)
    assert result["n_systems"] == 6
    assert result["n_tiers"] == 2, result["assignment"]
    weakest = [a for a in result["assignment"] if a["system"] == LLAMA8]
    assert weakest and weakest[0]["tier"] == 2


def test_every_grader_agrees_the_weakest_model_is_last():
    """The bias reorders the top, not the bottom -- worth stating, because it
    is what makes the reordering easy to miss."""
    assert rank_of(TRUTH, LLAMA8, binarise=False) == 6
    for path in JUDGES.values():
        assert rank_of(path, LLAMA8, binarise=True) == 6


# -- the judge is not flaky -------------------------------------------------

def test_repeat_grading_agrees_on_at_least_98_percent_of_answers():
    """The README says 98.3%. Anything near it supports the argument; a drop
    would mean the 'this is bias, not noise' claim needs rewriting."""
    cells = collections.defaultdict(list)
    for row in rows(JUDGES["gpt-4o-mini"]):
        cells[(row["system"], row["item"])].append(float(row["score"]))
    triples = [v for v in cells.values() if len(v) == 3]
    assert len(triples) > 1000, len(triples)
    agree = sum(1 for v in triples if len(set(v)) == 1) / len(triples)
    assert agree >= 0.98, agree


def test_the_judge_table_carries_three_grades_per_answer():
    runs = {row["run"] for row in rows(JUDGES["gpt-4o-mini"])}
    assert runs == {"a1_g1", "a1_g2", "a1_g3"}, runs


# -- the bias ---------------------------------------------------------------

def test_gpt4o_mini_is_harsh_to_llama_70b_and_the_bias_survives_correction():
    """-6.0 points at Holm p = .005 in the README."""
    result = compare_graders(load_csv(TRUTH), load_csv(JUDGES["gpt-4o-mini"]))
    row = {r["system"]: r for r in result["per_system"]}[LLAMA70]
    assert row["bias"] < -0.04, row["bias"]
    assert row["too_harsh"] > 10 * max(1, row["too_generous"])
    assert row["holm_p"] < 0.01, row["holm_p"]
    assert row["biased"]


def test_gpt4o_mini_is_generous_to_qwen():
    result = compare_graders(load_csv(TRUTH), load_csv(JUDGES["gpt-4o-mini"]))
    row = {r["system"]: r for r in result["per_system"]}[QWEN]
    assert row["bias"] > 0.02, row["bias"]


def test_bias_spread_ordering_across_the_three_judges():
    """gemini is the most accurate, gpt-4o-mini the least. The README leans on
    this ordering, so it is pinned rather than left to drift."""
    spread = {}
    for name, path in JUDGES.items():
        spread[name] = compare_graders(load_csv(TRUTH),
                                       load_csv(path))["bias_spread"]
    assert spread["gemini-2.5-flash-lite"] < spread["claude-3-haiku"]
    assert spread["claude-3-haiku"] < spread["gpt-4o-mini"]
    assert spread["gemini-2.5-flash-lite"] < 0.05
    assert spread["gpt-4o-mini"] > 0.09


def test_claude_haiku_is_generous_to_every_system():
    """A level shift, most generous to the weakest model -- which compresses
    the gap the benchmark exists to measure."""
    result = compare_graders(load_csv(TRUTH), load_csv(JUDGES["claude-3-haiku"]))
    rows_by = {r["system"]: r for r in result["per_system"]}
    assert all(r["bias"] >= 0 for r in result["per_system"]), \
        {k: v["bias"] for k, v in rows_by.items()}
    assert rows_by[LLAMA8]["bias"] == max(r["bias"] for r in result["per_system"])


def test_every_judge_reorders_the_table():
    for name, path in JUDGES.items():
        result = compare_graders(load_csv(TRUTH), load_csv(path))
        assert result["ranking"]["n_inversions"] >= 1, name
        assert result["ranking"]["top_changed"], name


# -- the tables themselves --------------------------------------------------

def test_all_four_tables_cover_the_same_six_systems():
    expected = set(load_csv(TRUTH).systems)
    assert len(expected) == 6
    for name, path in JUDGES.items():
        assert set(load_csv(path).systems) == expected, name


def test_every_table_has_roughly_two_hundred_items():
    for path in [TRUTH] + list(JUDGES.values()):
        table = load_csv(path)
        assert 195 <= table.n_items <= 200, (path, table.n_items)


def test_scores_are_all_zero_or_one():
    for path in [TRUTH] + list(JUDGES.values()):
        values = {float(row["score"]) for row in rows(path)}
        assert values <= {0.0, 1.0}, (path, sorted(values)[:5])


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print("PASS  %s" % name)
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print("FAIL  %s: %s" % (name, exc))
    print("\n%d failed" % failures if failures else "\nall passed")
    raise SystemExit(1 if failures else 0)
