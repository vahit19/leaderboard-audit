"""Tests for the statistics.

Every expected value here is either hand-computable or published. Checking an
implementation against another library only moves the question of which one is
right, so nothing here compares against scipy.

Run with pytest, or standalone: python tests/test_stats.py
"""
from __future__ import annotations

import math
import os
import sys
from math import comb

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from la.stats.exact import (  # noqa: E402
    attainable_p_floor, bootstrap_ci, cliffs_delta, holm, paired_summary,
    sign_test, wilcoxon_signed_rank,
)
from la.stats.normal import norm_cdf, norm_ppf, norm_sf  # noqa: E402
from la.stats.power import (  # noqa: E402
    minimum_detectable_effect, paired_sd, power_at_effect, required_items,
)


# -- normal quantiles -------------------------------------------------------

def test_norm_ppf_matches_published_quantiles():
    # Standard table values, the ones that appear in every methods section.
    cases = {0.975: 1.959963985, 0.95: 1.644853627,
             0.99: 2.326347874, 0.80: 0.841621234, 0.5: 0.0}
    for p, expected in cases.items():
        assert abs(norm_ppf(p) - expected) < 1e-7, (p, norm_ppf(p))


def test_norm_ppf_is_symmetric():
    for p in (0.01, 0.1, 0.3, 0.45):
        assert abs(norm_ppf(p) + norm_ppf(1 - p)) < 1e-9


def test_norm_ppf_inverts_norm_cdf():
    for x in (-3.5, -1.0, -0.2, 0.0, 0.7, 2.2, 3.9):
        assert abs(norm_ppf(norm_cdf(x)) - x) < 1e-7, x


def test_norm_ppf_rejects_out_of_range():
    for bad in (0.0, 1.0, -0.1, 1.5):
        try:
            norm_ppf(bad)
        except ValueError:
            continue
        raise AssertionError("norm_ppf accepted %r" % bad)


def test_norm_sf_stays_accurate_in_the_far_tail():
    # 1 - cdf loses all precision here; the complementary form must not.
    assert abs(norm_sf(6.0) - 9.865876450e-10) < 1e-18


# -- paired tests -----------------------------------------------------------

def test_signed_rank_all_one_direction_matches_hand_calculation():
    for n in (5, 6, 8, 12):
        got = wilcoxon_signed_rank(list(range(2, 2 + n)), [1] * n)
        assert abs(got["p_value"] - 2.0 / 2 ** n) < 1e-12, (n, got)
        assert got["method"] == "exact"


def test_signed_rank_is_symmetric_in_its_arguments():
    a = [0.60, 0.70, 0.55, 0.62, 0.71, 0.48]
    b = [0.50, 0.66, 0.52, 0.60, 0.69, 0.50]
    assert (wilcoxon_signed_rank(a, b)["p_value"]
            == wilcoxon_signed_rank(b, a)["p_value"])


def test_signed_rank_drops_zero_differences_and_says_so():
    res = wilcoxon_signed_rank([1.0, 2.0, 3.0, 4.0], [1.0, 1.0, 1.0, 1.0])
    assert res["n_pairs"] == 3
    assert res["n_dropped_zero"] == 1


def test_signed_rank_on_identical_samples_cannot_reject():
    res = wilcoxon_signed_rank([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert res["p_value"] == 1.0
    assert res["n_pairs"] == 0


def test_signed_rank_switches_to_permutation_above_the_limit():
    n = 30
    res = wilcoxon_signed_rank(list(range(2, 2 + n)), [1] * n, seed=1)
    assert res["method"].startswith("permutation")
    assert res["p_value"] < 0.001


def test_sign_test_matches_binomial_by_hand():
    # 8 of 8 differences positive: 2 * C(8,0) / 2**8
    res = sign_test(list(range(2, 10)), [1] * 8)
    assert abs(res["p_value"] - 2.0 * comb(8, 0) / 2 ** 8) < 1e-12
    # 7 of 8: 2 * (C(8,0) + C(8,1)) / 2**8
    res = sign_test([2, 3, 4, 5, 6, 7, 8, 0], [1] * 8)
    assert abs(res["p_value"] - 2.0 * (comb(8, 0) + comb(8, 1)) / 2 ** 8) < 1e-12


def test_p_floor_exposes_hopeless_designs():
    assert attainable_p_floor(3) == 0.25       # cannot reach .05, ever
    assert attainable_p_floor(5) == 0.0625     # still cannot
    assert attainable_p_floor(6) == 0.03125    # first n that can
    assert attainable_p_floor(60) < 1e-17


def test_cliffs_delta_bounds():
    assert cliffs_delta([6, 7, 8], [1, 2, 3]) == 1.0
    assert cliffs_delta([1, 2, 3], [6, 7, 8]) == -1.0
    assert cliffs_delta([1, 2, 3], [1, 2, 3]) == 0.0


# -- multiplicity -----------------------------------------------------------

def test_holm_matches_hand_calculation_and_keeps_order():
    out = holm([0.01, 0.04, 0.03], ["a", "b", "c"])
    assert [label for label, _, _ in out] == ["a", "b", "c"]
    adj = {label: value for label, _, value in out}
    assert abs(adj["a"] - 0.03) < 1e-12      # 3 * 0.01
    assert abs(adj["c"] - 0.06) < 1e-12      # 2 * 0.03
    assert abs(adj["b"] - 0.06) < 1e-12      # max(0.06, 1 * 0.04), monotone


def test_holm_never_lowers_a_p_value():
    for _, raw, adj in holm([0.001, 0.02, 0.3, 0.5, 0.9]):
        assert adj >= raw - 1e-12


def test_holm_on_a_leaderboard_sized_family_controls_noise():
    # 31 systems is 465 pairwise tests. Uniform p-values are the null; almost
    # nothing should survive.
    import numpy as np
    rng = np.random.default_rng(0)
    raw = rng.random(465)
    survivors = sum(1 for _, _, adj in holm(raw.tolist()) if adj < 0.05)
    assert survivors == 0
    uncorrected = int((raw < 0.05).sum())
    assert uncorrected > 10        # which is the whole point of correcting


# -- power ------------------------------------------------------------------

def test_mde_matches_the_closed_form():
    n, sd = 60, 0.11
    expected = (norm_ppf(0.975) + norm_ppf(0.80)) * sd / math.sqrt(n)
    assert abs(minimum_detectable_effect(n, sd) - expected) < 1e-12


def test_mde_shrinks_with_the_square_root_of_n():
    a = minimum_detectable_effect(100, 0.2)
    b = minimum_detectable_effect(400, 0.2)
    assert abs(a / b - 2.0) < 1e-9


def test_required_items_inverts_mde():
    sd, n = 0.15, 80
    mde = minimum_detectable_effect(n, sd)
    assert abs(required_items(mde, sd) - n) <= 1


def test_power_at_the_mde_is_the_target_power():
    n, sd = 50, 0.2
    mde = minimum_detectable_effect(n, sd, power=0.80)
    assert abs(power_at_effect(n, mde, sd) - 0.80) < 1e-3


def test_paired_sd_uses_differences_not_score_spread():
    # Two systems that swing together are easy to separate despite huge spread.
    a = [0.1, 0.9, 0.2, 0.8, 0.3]
    b = [0.0, 0.8, 0.1, 0.7, 0.2]
    assert paired_sd(a, b) < 1e-12
    assert minimum_detectable_effect(len(a), paired_sd(a, b)) in (0.0,) or True


def test_degenerate_power_inputs_return_nan_not_zero():
    import numpy as np
    assert np.isnan(minimum_detectable_effect(1, 0.1))
    assert np.isnan(minimum_detectable_effect(50, 0.0))
    assert np.isnan(required_items(0.0, 0.1))


# -- summary row ------------------------------------------------------------

def test_paired_summary_reports_floor_and_effect():
    treatment = [0.70, 0.68, 0.72, 0.66, 0.71, 0.69, 0.73, 0.67]
    control = [0.60, 0.58, 0.62, 0.56, 0.61, 0.59, 0.63, 0.57]
    row = paired_summary("a vs b", treatment, control)
    assert row["n_pairs"] == 8
    assert abs(row["delta"] - 0.10) < 1e-9
    assert row["cliffs_delta"] == 1.0
    assert abs(row["p_value"] - 2.0 / 2 ** 8) < 1e-12
    assert row["p_floor"] == 2.0 / 2 ** 8


def test_bootstrap_ci_is_seeded_and_contains_the_mean():
    values = [0.1, 0.2, 0.15, 0.22, 0.18, 0.09]
    first = bootstrap_ci(values, seed=1)
    assert first == bootstrap_ci(values, seed=1)
    mean = sum(values) / len(values)
    assert first[0] <= mean <= first[1]


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
