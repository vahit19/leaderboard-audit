"""Tests for grader comparison.

The property that matters here is the one a repeat-grading study cannot see: a
grader can be perfectly repeatable and still wrong in a direction that depends
on which system produced the answer. These build that situation on purpose and
check it is reported.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from la.compare import (align, compare_graders, grader_bias,  # noqa: E402
                        ranking_changes)
from la.data import from_records  # noqa: E402


def table(scores_by_system, n_items=60, source=""):
    """Build a table where each system has a fixed per-item score pattern."""
    records = []
    for system, scores in scores_by_system.items():
        for i, value in enumerate(scores[:n_items]):
            records.append({"system": system, "item": "i%03d" % i,
                            "score": float(value)})
    return from_records(records, source=source)


def binary(rng, n, rate):
    return (rng.random(n) < rate).astype(float)


def truth_and_biased(n=120, seed=0, generous=(), harsh=(), flip=0.20):
    """A reference grader and a candidate that is generous to some systems and
    harsh to others, by construction."""
    rng = np.random.default_rng(seed)
    base = {"a": 0.90, "b": 0.88, "c": 0.86, "d": 0.70}
    ref, cand = {}, {}
    for system, rate in base.items():
        truth = binary(rng, n, rate)
        ref[system] = truth
        graded = truth.copy()
        mask = rng.random(n) < flip
        if system in generous:
            graded[mask & (truth == 0.0)] = 1.0
        elif system in harsh:
            graded[mask & (truth == 1.0)] = 0.0
        cand[system] = graded
    return table(ref, n, "reference"), table(cand, n, "candidate")


# -- alignment --------------------------------------------------------------

def test_align_keeps_only_shared_systems_and_items():
    ref = table({"a": [1.0] * 10, "b": [0.0] * 10}, 10)
    cand = table({"b": [1.0] * 6, "c": [1.0] * 6}, 6)
    systems, items, r, c = align(ref, cand)
    assert systems == ["b"]
    assert len(items) == 6
    assert r.shape == c.shape == (6, 1)


def test_align_refuses_tables_with_nothing_in_common():
    ref = table({"a": [1.0] * 5}, 5)
    cand = table({"z": [1.0] * 5}, 5)
    try:
        align(ref, cand)
    except ValueError as exc:
        assert "nothing to compare" in str(exc)
        return
    raise AssertionError("disjoint tables should raise")


# -- bias -------------------------------------------------------------------

def test_identical_graders_show_no_bias_and_no_inversions():
    ref = table({"a": [1.0, 0.0] * 30, "b": [1.0, 1.0] * 30}, 60)
    result = compare_graders(ref, ref)
    assert result["overall_agreement"] == 1.0
    assert abs(result["bias_spread"]) < 1e-12
    assert result["ranking"]["n_inversions"] == 0
    assert all(not r["biased"] for r in result["per_system"])


def test_a_generous_grader_is_reported_as_generous():
    ref, cand = truth_and_biased(generous=("d",), seed=1)
    rows = {r["system"]: r for r in compare_graders(ref, cand)["per_system"]}
    assert rows["d"]["bias"] > 0.0
    assert rows["d"]["too_generous"] > rows["d"]["too_harsh"]


def test_a_harsh_grader_is_reported_as_harsh():
    ref, cand = truth_and_biased(harsh=("a",), seed=2)
    rows = {r["system"]: r for r in compare_graders(ref, cand)["per_system"]}
    assert rows["a"]["bias"] < 0.0
    assert rows["a"]["too_harsh"] > rows["a"]["too_generous"]


def test_bias_spread_captures_the_gap_between_best_and_worst_treated():
    ref, cand = truth_and_biased(generous=("d",), harsh=("a",), seed=3)
    result = compare_graders(ref, cand)
    # One system pushed up and another pushed down: the spread must exceed
    # either one alone.
    biases = {r["system"]: r["bias"] for r in result["per_system"]}
    assert result["bias_spread"] >= abs(biases["a"])
    assert result["bias_spread"] >= abs(biases["d"])


def test_per_system_bias_is_holm_corrected():
    ref, cand = truth_and_biased(harsh=("a",), seed=4)
    for row in compare_graders(ref, cand)["per_system"]:
        assert row["holm_p"] >= row["p_value"] - 1e-12


# -- ranking ----------------------------------------------------------------

def test_a_biased_grader_that_reorders_the_table_is_caught():
    """The finding this module exists for: systems 2 points apart, a grader
    that is 6 points harsh to the leader, and the order flips."""
    n = 200
    rng = np.random.default_rng(5)
    truth_a = binary(rng, n, 0.92)
    truth_b = binary(rng, n, 0.90)
    ref = table({"leader": truth_a, "runner_up": truth_b}, n)

    harsh = truth_a.copy()
    correct = np.where(truth_a == 1.0)[0]
    harsh[correct[: int(0.08 * n)]] = 0.0      # 8 points of harshness
    cand = table({"leader": harsh, "runner_up": truth_b}, n)

    result = compare_graders(ref, cand)
    assert result["ranking"]["n_inversions"] == 1
    assert result["ranking"]["top_changed"]
    assert "which system is first" in result["verdict"]


def test_ranking_changes_reports_movement_and_direction():
    ref = table({"a": [1.0] * 20, "b": [0.0] * 20}, 20)
    cand = table({"a": [0.0] * 20, "b": [1.0] * 20}, 20)
    systems, _, r, c = align(ref, cand)
    changes = ranking_changes(systems, r, c)
    assert changes["n_inversions"] == 1
    assert changes["top_changed"]
    assert {m["system"] for m in changes["moved"]} == {"a", "b"}


def test_no_inversion_when_the_bias_is_uniform():
    """A grader that is equally harsh to everyone shifts the scores but not
    the ordering, and must not be reported as reordering anything."""
    n = 120
    rng = np.random.default_rng(6)
    ref_scores = {"a": binary(rng, n, 0.90), "b": binary(rng, n, 0.80),
                  "c": binary(rng, n, 0.70)}
    cand_scores = {}
    for system, truth in ref_scores.items():
        shifted = truth.copy()
        correct = np.where(truth == 1.0)[0]
        shifted[correct[: int(0.10 * len(correct))]] = 0.0
        cand_scores[system] = shifted

    result = compare_graders(table(ref_scores, n), table(cand_scores, n))
    assert result["ranking"]["n_inversions"] == 0
    assert not result["ranking"]["top_changed"]


# -- the verdict ------------------------------------------------------------

def test_verdict_states_overall_agreement():
    ref, cand = truth_and_biased(harsh=("a",), seed=7)
    verdict = compare_graders(ref, cand)["verdict"]
    assert "overall agreement" in verdict


def test_verdict_warns_that_repeat_grading_would_not_find_this():
    ref, cand = truth_and_biased(generous=("d",), harsh=("a",), seed=8,
                                 flip=0.5)
    result = compare_graders(ref, cand)
    if result["ranking"]["n_inversions"]:
        assert "repeat grading" in result["verdict"]


def test_high_overall_agreement_can_still_reorder_the_table():
    """The headline number hides the problem, which is why it is not the
    headline.

    Built exactly rather than sampled: a random draw is not guaranteed to put
    the intended system ahead, and a test whose premise depends on the seed is
    testing the seed.
    """
    n = 300
    leader = np.array([1.0] * 279 + [0.0] * 21)          # 93.0%
    runner_up = np.array([1.0] * 273 + [0.0] * 27)       # 91.0%
    ref = table({"leader": leader, "runner_up": runner_up}, n)

    harsh = leader.copy()
    harsh[:9] = 0.0                                      # 3 points of harshness
    assert harsh.mean() < runner_up.mean()               # the premise
    cand = table({"leader": harsh, "runner_up": runner_up}, n)

    result = compare_graders(ref, cand)
    assert result["overall_agreement"] > 0.98            # looks fine
    assert result["ranking"]["n_inversions"] == 1        # is not
    assert result["ranking"]["top_changed"]


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
