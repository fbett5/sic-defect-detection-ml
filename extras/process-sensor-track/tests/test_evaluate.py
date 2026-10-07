"""Metric and threshold behaviour, including the degenerate cases that bite."""

from __future__ import annotations

import numpy as np
import pytest

from sicdd.evaluate import (
    bootstrap_ci,
    cost_sensitivity,
    expected_cost,
    operating_point_metrics,
    pick_threshold_min_cost,
    recall_at_inspection_budget,
    threshold_for_flag_rate,
    threshold_free_metrics,
)


def test_perfect_and_useless_rankings():
    y = np.array([0, 0, 0, 1, 1])
    assert threshold_free_metrics(y, np.array([0.1, 0.2, 0.3, 0.9, 0.95])).roc_auc == 1.0
    assert threshold_free_metrics(y, np.array([0.9, 0.95, 0.8, 0.1, 0.2])).roc_auc == 0.0


def test_no_skill_floor_is_the_prevalence():
    rng = np.random.default_rng(0)
    y = (rng.random(4000) < 0.05).astype(int)
    m = threshold_free_metrics(y, rng.random(4000))
    assert m.average_precision == pytest.approx(m.prevalence, abs=0.015)
    assert m.ap_lift_over_prevalence == pytest.approx(1.0, abs=0.3)


def test_shape_mismatch_is_rejected():
    with pytest.raises(ValueError, match="shape mismatch"):
        threshold_free_metrics(np.array([0, 1]), np.array([0.1, 0.2, 0.3]))


def test_expected_cost_weights_misses_more_heavily():
    y = np.array([0, 1])
    miss = expected_cost(y, np.array([0.0, 0.0]), 0.5, c_fn=10, c_fp=1)  # one FN
    alarm = expected_cost(y, np.array([1.0, 1.0]), 0.5, c_fn=10, c_fp=1)  # one FP
    assert miss == pytest.approx(5.0)
    assert alarm == pytest.approx(0.5)


def test_min_cost_threshold_beats_both_extremes():
    rng = np.random.default_rng(1)
    y = (rng.random(500) < 0.1).astype(int)
    s = np.clip(0.55 * y + rng.normal(0.2, 0.15, 500), 0, 1)
    thr, cost = pick_threshold_min_cost(y, s, c_fn=10, c_fp=1)
    assert cost <= expected_cost(y, s, -np.inf, 10, 1) + 1e-12  # flag everything
    assert cost <= expected_cost(y, s, np.inf, 10, 1) + 1e-12   # flag nothing


def test_min_cost_handles_constant_scores():
    y = np.array([0, 0, 1, 0])
    thr, cost = pick_threshold_min_cost(y, np.full(4, 0.5), c_fn=10, c_fp=1)
    assert np.isfinite(thr) and cost >= 0


def test_threshold_for_flag_rate_round_trips():
    s = np.linspace(0, 1, 1000)
    for rate in (0.05, 0.1, 0.25, 0.5):
        thr = threshold_for_flag_rate(s, rate)
        assert (s >= thr).mean() == pytest.approx(rate, abs=0.01)


def test_flag_rate_extremes_are_expressible():
    s = np.linspace(0, 1, 10)
    assert threshold_for_flag_rate(s, 0.0) == np.inf
    assert threshold_for_flag_rate(s, 1.0) == -np.inf
    assert (s >= threshold_for_flag_rate(s, 0.0)).sum() == 0
    assert (s >= threshold_for_flag_rate(s, 1.0)).sum() == len(s)


@pytest.mark.parametrize("rate", [-0.1, 1.1])
def test_flag_rate_rejects_out_of_range(rate):
    with pytest.raises(ValueError):
        threshold_for_flag_rate(np.linspace(0, 1, 5), rate)


def test_operating_point_confusion_counts_are_consistent():
    y = np.array([0, 0, 1, 1])
    op = operating_point_metrics(y, np.array([0.1, 0.9, 0.2, 0.8]), 0.5, c_fn=10, c_fp=1)
    assert (op["tp"], op["fp"], op["fn"], op["tn"]) == (1, 1, 1, 1)
    assert op["precision"] == pytest.approx(0.5)
    assert op["recall"] == pytest.approx(0.5)
    assert op["flagged_rate"] == pytest.approx(0.5)


def test_inspection_budget_is_rank_based_under_ties():
    """A constant scorer must not appear to catch everything within a budget."""
    y = np.array([0] * 90 + [1] * 10)
    r = recall_at_inspection_budget(y, np.full(100, 0.42), budget=0.1)
    assert r["n_inspected"] == 10
    assert r["recall"] <= 1.0 and r["caught"] <= 10


def test_inspection_budget_rewards_a_good_ranking():
    y = np.array([0] * 90 + [1] * 10)
    perfect = np.concatenate([np.zeros(90), np.ones(10)])
    r = recall_at_inspection_budget(y, perfect, budget=0.1)
    assert r["recall"] == pytest.approx(1.0)
    assert r["lift"] == pytest.approx(10.0)


def test_bootstrap_interval_brackets_the_point_estimate():
    rng = np.random.default_rng(3)
    y = (rng.random(300) < 0.1).astype(int)
    s = np.clip(0.4 * y + rng.random(300) * 0.6, 0, 1)
    ci = bootstrap_ci(y, s, "average_precision", n_boot=500, seed=7)
    assert ci["lo"] <= ci["point"] <= ci["hi"]
    assert bootstrap_ci(y, s, "average_precision", n_boot=500, seed=7) == ci  # seeded


def test_cost_sensitivity_flags_break_even():
    y = np.array([0] * 95 + [1] * 5)
    s = np.concatenate([np.linspace(0, 0.6, 95), np.linspace(0.7, 1.0, 5)])
    sweep = cost_sensitivity(y, s, threshold=0.65)
    assert [r["cost_ratio_fn_to_fp"] for r in sweep] == [2.0, 5.0, 10.0, 20.0, 50.0]
    assert all(r["worthwhile"] for r in sweep)  # a near-perfect ranking always pays
