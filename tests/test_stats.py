"""Tests for driftwatch.stats: PSI, KS, binning, and edge cases."""

import math

from driftwatch import stats


def test_psi_identical_distributions_near_zero():
    ref = [float(i % 50) for i in range(1000)]
    cur = [float((i * 7) % 50) for i in range(1000)]
    psi, *_ = stats.psi_numeric(ref, cur)
    assert psi < 0.05


def test_psi_shifted_distribution_large():
    ref = [float(i) for i in range(1000)]
    cur = [float(i) + 500.0 for i in range(1000)]
    psi, *_ = stats.psi_numeric(ref, cur)
    assert psi > 0.25


def test_ks_against_hand_computed_value():
    # a = {1,2,3,4}, b = {3,4,5,6}: CDF gap is maximal (0.5) at x = 2 and x = 4.
    assert stats.ks_2samp([1, 2, 3, 4], [3, 4, 5, 6]) == 0.5


def test_ks_identical_is_zero():
    assert stats.ks_2samp([1, 2, 3], [1, 2, 3]) == 0.0


def test_ks_completely_separated_is_one():
    assert stats.ks_2samp([1, 2], [10, 20]) == 1.0


def test_psi_categorical_unseen_category():
    # 'diamond' never appears in reference: must not crash, must flag drift.
    ref = ["basic"] * 800 + ["plus"] * 200
    cur = ["basic"] * 700 + ["plus"] * 200 + ["diamond"] * 100
    psi, cats, *_ = stats.psi_categorical(ref, cur)
    assert "diamond" in cats
    assert math.isfinite(psi)
    assert psi > 0


def test_constant_feature_no_crash_and_zero_psi():
    psi, edges, ref_counts, cur_counts = stats.psi_numeric([5.0] * 100, [5.0] * 100)
    assert psi == 0.0
    # all values land in a single bin; degenerate bins stay empty but harmless
    assert sum(1 for c in ref_counts if c > 0) == 1
    assert stats.ks_2samp([5.0] * 100, [5.0] * 100) == 0.0


def test_empty_bin_smoothing_stays_finite():
    # current has values far outside every reference bin -> zero-count bins
    ref_counts = [100, 100, 100]
    cur_counts = [0, 0, 300]
    psi = stats.psi_from_counts(ref_counts, cur_counts)
    assert math.isfinite(psi)
    assert psi > 0.25


def test_psi_verdict_thresholds():
    assert stats.psi_verdict(0.0) == "no significant change"
    assert stats.psi_verdict(0.099) == "no significant change"
    assert stats.psi_verdict(0.1) == "moderate shift — watch"
    assert stats.psi_verdict(0.25) == "moderate shift — watch"
    assert stats.psi_verdict(0.251) == "significant drift"


def test_quantile_binning_uses_reference_edges():
    ref = list(range(100))
    edges = stats.quantile_bin_edges(ref, n_bins=10)
    assert len(edges) == 11
    assert edges[0] == float("-inf") and edges[-1] == float("inf")
    counts = stats.binned_counts(ref, edges)
    assert sum(counts) == 100
    assert all(c == 10 for c in counts)
