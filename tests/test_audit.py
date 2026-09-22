"""Tests for the audit itself.

The central one is `test_recovers_known_tiers`: build a table whose true
structure is known, and check the tool reports that structure rather than the
twelve-way ordering the raw means suggest. A ranking tool that cannot pass this
is worse than no tool, because it launders noise into a number.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from la.data import ResultTable, from_records  # noqa: E402
from la.ranks import (homogeneous_subsets, nontransitive_pairs,  # noqa: E402
                      pairwise_comparisons, rank_intervals, tiers)
from la.report import headline, recommendations  # noqa: E402
from la.variance import noise_floor, variance_components  # noqa: E402


def make_table(levels, per_level=3, n_items=60, n_runs=3,
               item_spread=0.16, run_noise=0.11, seed=0):
    """Synthetic table with a known number of true ability levels."""
    rng = np.random.default_rng(seed)
    systems = [("%s-%d" % (name, k), value)
               for name, value in levels.items()
               for k in range(1, per_level + 1)]
    item_effect = rng.normal(0.0, item_spread, size=n_items)
    records = []
    for i in range(n_items):
        for name, ability in systems:
            for run in range(1, n_runs + 1):
                score = ability + item_effect[i] + rng.normal(0.0, run_noise)
                records.append({"system": name, "item": "t%03d" % i,
                                "run": run,
                                "score": float(np.clip(score, 0.0, 1.0))})
    return from_records(records, source="synthetic")


LEVELS = {"strong": 0.78, "good": 0.71, "fair": 0.62, "weak": 0.44}


# -- the central claim ------------------------------------------------------

def test_recovers_known_tiers():
    table = make_table(LEVELS).complete()
    result = tiers(table.scores, table.systems, seed=0)
    assert result["n_systems"] == 12
    assert result["n_tiers"] == len(LEVELS), result["n_tiers"]


def test_systems_from_the_same_level_land_in_the_same_tier():
    table = make_table(LEVELS).complete()
    result = tiers(table.scores, table.systems, seed=0)
    tier_of = {a["system"]: a["tier"] for a in result["assignment"]}
    for level in LEVELS:
        assigned = {tier_of["%s-%d" % (level, k)] for k in (1, 2, 3)}
        assert len(assigned) == 1, (level, assigned)


def test_identical_systems_are_never_declared_distinguishable():
    """Twelve systems with the same true ability must collapse to one tier."""
    flat = {"same": 0.65}
    table = make_table(flat, per_level=12).complete()
    result = tiers(table.scores, table.systems, seed=0)
    assert result["n_tiers"] == 1, result["assignment"]


def test_genuinely_separated_systems_are_all_distinguished():
    wide = {"a": 0.9, "b": 0.7, "c": 0.5, "d": 0.3, "e": 0.1}
    table = make_table(wide, per_level=1, run_noise=0.03).complete()
    result = tiers(table.scores, table.systems, seed=0)
    assert result["n_tiers"] == 5


# -- rank intervals ---------------------------------------------------------

def test_rank_intervals_bracket_the_observed_rank():
    table = make_table(LEVELS).complete()
    rows = rank_intervals(table.scores, table.systems, n_boot=2000, seed=0)
    for r in rows:
        assert r["rank_low"] <= r["rank"] <= r["rank_high"], r


def test_tied_systems_have_overlapping_rank_intervals():
    table = make_table(LEVELS).complete()
    rows = {r["system"]: r for r in
            rank_intervals(table.scores, table.systems, n_boot=2000, seed=0)}
    a, b = rows["good-1"], rows["good-2"]
    assert a["rank_low"] <= b["rank_high"] and b["rank_low"] <= a["rank_high"]


def test_rank_intervals_narrow_as_items_are_added():
    def span(n_items):
        t = make_table(LEVELS, n_items=n_items, seed=3).complete()
        rows = rank_intervals(t.scores, t.systems, n_boot=2000, seed=0)
        return float(np.mean([r["rank_span"] for r in rows]))
    assert span(200) <= span(25)


def test_rank_intervals_reject_a_single_item():
    table = make_table(LEVELS, n_items=1).complete()
    try:
        rank_intervals(table.scores, table.systems)
    except ValueError:
        return
    raise AssertionError("one item should not produce rank intervals")


# -- pairwise ---------------------------------------------------------------

def test_pairwise_covers_every_pair_once():
    table = make_table(LEVELS).complete()
    rows = pairwise_comparisons(table.scores, table.systems, seed=0)
    n = len(table.systems)
    assert len(rows) == n * (n - 1) // 2
    seen = {frozenset((r["system_a"], r["system_b"])) for r in rows}
    assert len(seen) == len(rows)


def test_pairwise_reports_holm_adjusted_values():
    table = make_table(LEVELS).complete()
    rows = pairwise_comparisons(table.scores, table.systems, seed=0)
    for r in rows:
        assert r["holm_p"] >= r["p_value"] - 1e-12
        assert isinstance(r["distinguishable"], bool)


def test_far_apart_systems_are_distinguishable_and_close_ones_are_not():
    table = make_table(LEVELS).complete()
    rows = {frozenset((r["system_a"], r["system_b"])): r
            for r in pairwise_comparisons(table.scores, table.systems, seed=0)}
    assert rows[frozenset(("strong-1", "weak-1"))]["distinguishable"]
    assert not rows[frozenset(("good-1", "good-2"))]["distinguishable"]


# -- subsets and transitivity ----------------------------------------------

def test_homogeneous_subsets_cover_every_system():
    table = make_table(LEVELS).complete()
    result = tiers(table.scores, table.systems, seed=0)
    ordered = [a["system"] for a in result["assignment"]]
    rows = pairwise_comparisons(table.scores, table.systems, seed=0)
    subsets = homogeneous_subsets(rows, ordered)
    covered = {s for group in subsets for s in group}
    assert covered == set(table.systems)


def test_nontransitive_pairs_is_empty_on_cleanly_separated_data():
    wide = {"a": 0.9, "b": 0.6, "c": 0.3}
    table = make_table(wide, per_level=1, run_noise=0.03).complete()
    rows = pairwise_comparisons(table.scores, table.systems, seed=0)
    ordered = [a["system"] for a in
               tiers(table.scores, table.systems, seed=0)["assignment"]]
    assert nontransitive_pairs(rows, ordered) == []


# -- variance ---------------------------------------------------------------

def test_variance_components_recover_the_injected_noise():
    table = make_table(LEVELS, run_noise=0.11, n_runs=4, seed=5)
    var = variance_components(table.repeats, table.systems, table.items)
    assert var["measurable"]
    # residual variance should be close to run_noise**2
    assert abs(var["residual_var"] - 0.11 ** 2) < 0.002, var["residual_var"]


def test_variance_is_unmeasurable_without_repeats():
    table = make_table(LEVELS, n_runs=1)
    var = variance_components(table.repeats, table.systems, table.items)
    assert var["measurable"] is False
    assert "single run" in var["note"]


def test_noise_floor_falls_when_runs_are_averaged():
    one = noise_floor(0.01, n_items=60, n_repeats=1)
    four = noise_floor(0.01, n_items=60, n_repeats=4)
    assert abs(one / four - 2.0) < 1e-9


def test_noise_floor_is_none_without_a_residual():
    assert noise_floor(None, 60) is None


# -- data handling ----------------------------------------------------------

def test_missing_cells_are_dropped_and_counted():
    table = make_table(LEVELS, n_items=20, n_runs=1)
    table.scores[0, 0] = np.nan
    complete = table.complete()
    assert complete.n_items == table.n_items - 1
    assert table.n_missing == 1


def test_from_records_rejects_a_missing_column():
    try:
        from_records([{"system": "a", "item": "x"}])
    except ValueError as exc:
        assert "score" in str(exc)
        return
    raise AssertionError("missing column should raise")


def test_from_records_rejects_a_non_numeric_score():
    try:
        from_records([{"system": "a", "item": "x", "score": "n/a"}])
    except ValueError as exc:
        assert "numeric" in str(exc)
        return
    raise AssertionError("non-numeric score should raise")


def test_result_table_rejects_a_mismatched_matrix():
    try:
        ResultTable(["a", "b"], ["x"], np.zeros((3, 3)))
    except ValueError:
        return
    raise AssertionError("shape mismatch should raise")


def test_repeat_runs_are_averaged_into_the_cell():
    table = from_records([
        {"system": "a", "item": "x", "score": 0.0},
        {"system": "a", "item": "x", "score": 1.0},
    ])
    assert table.scores[0, 0] == 0.5
    assert table.has_repeats


# -- reporting --------------------------------------------------------------

def test_headline_states_the_unsupported_gap_count():
    table = make_table(LEVELS).complete()
    result = tiers(table.scores, table.systems, seed=0)
    text = headline(result)
    assert "12 systems" in text and "4 distinguishable tiers" in text


def test_recommendations_do_not_ask_for_absurd_item_counts():
    """Identical systems must be reported as tied, not as underpowered."""
    table = make_table({"same": 0.65}, per_level=6).complete()
    result = tiers(table.scores, table.systems, seed=0)
    rows = pairwise_comparisons(table.scores, table.systems, seed=0)
    advice = recommendations(result, rows, table.n_items, None)
    joined = " ".join(advice)
    assert "report them as tied" in joined
    for word in joined.split():
        digits = word.replace(",", "").replace(".", "")
        if digits.isdigit():
            assert int(digits) < 10 ** 6, word


def test_recommendations_flag_absent_repeat_runs():
    table = make_table(LEVELS, n_runs=1)
    complete = table.complete()
    var = variance_components(table.repeats, table.systems, table.items)
    result = tiers(complete.scores, complete.systems, seed=0)
    rows = pairwise_comparisons(complete.scores, complete.systems, seed=0)
    advice = recommendations(result, rows, complete.n_items, var)
    assert any("repeat" in a for a in advice)


# -- reproducibility --------------------------------------------------------

def test_same_seed_gives_identical_output():
    table = make_table(LEVELS).complete()
    a = rank_intervals(table.scores, table.systems, n_boot=1000, seed=7)
    b = rank_intervals(table.scores, table.systems, n_boot=1000, seed=7)
    assert a == b


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
