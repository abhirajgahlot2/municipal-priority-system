"""
Tests for backend.genetic_optimizer – Member 3 GA work-order optimizer
======================================================================
"""

import sys
import os
import unittest

# Ensure project root is on the path so "backend" is importable as a package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from backend.genetic_optimizer import (
    optimize_work_order,
    _order_crossover,
    _swap_mutation,
    _compute_fitness,
    _urgency,
    _norm_safety,
    _baseline_order,
)
import random


# ============================================================
# HELPERS
# ============================================================

def _make_complaint(uid, severity=5, traffic=5, public=5, weather=5,
                    days=5, freq=5, safety=0.5, status="Open",
                    complaint_type="Test", descriptor="test",
                    agency="TEST", freq_count=2):
    """Build a minimal complaint dictionary."""
    return {
        "unique_key": uid,
        "severity": severity,
        "traffic_impact": traffic,
        "public_impact": public,
        "weather_risk": weather,
        "days_pending": days,
        "complaint_frequency": freq,
        "frequency_count": freq_count,
        "safety_risk": safety,
        "status": status,
        "complaint_type": complaint_type,
        "descriptor": descriptor,
        "agency": agency,
    }


# ============================================================
# TEST CLASS
# ============================================================

class TestGeneticOptimizer(unittest.TestCase):
    """Focused tests for optimize_work_order and its GA components."""

    # --------------------------------------------------------
    # EMPTY AND SINGLE
    # --------------------------------------------------------

    def test_empty_input(self):
        """Empty complaints list returns an empty order."""
        result = optimize_work_order([], [])
        self.assertEqual(result["optimized_order"], [])
        self.assertIsNone(result["total_travel_distance"])
        self.assertIsNone(result["total_estimated_repair_time"])
        self.assertEqual(result["fitness"], 0.0)
        self.assertIsInstance(result["notes"], list)

    def test_single_complaint(self):
        """One complaint returns that ID and a finite fitness."""
        c = _make_complaint(100)
        result = optimize_work_order([c], [0.8])
        self.assertEqual(result["optimized_order"], [100])
        self.assertIsInstance(result["fitness"], float)
        self.assertTrue(result["fitness"] >= 0.0)

    # --------------------------------------------------------
    # INPUT VALIDATION
    # --------------------------------------------------------

    def test_duplicate_ids(self):
        """Duplicate unique_key values raise ValueError."""
        c1 = _make_complaint(1)
        c2 = _make_complaint(1)
        with self.assertRaises(ValueError):
            optimize_work_order([c1, c2], [0.5, 0.6])

    def test_invalid_priority_scores_length(self):
        """Mismatched lengths raise ValueError."""
        c1 = _make_complaint(1)
        with self.assertRaises(ValueError):
            optimize_work_order([c1], [0.5, 0.6])

    def test_invalid_priority_score_out_of_range_high(self):
        """Score > 1 raises ValueError."""
        c1 = _make_complaint(1)
        with self.assertRaises(ValueError):
            optimize_work_order([c1], [1.5])

    def test_invalid_priority_score_out_of_range_low(self):
        """Score < 0 raises ValueError."""
        c1 = _make_complaint(1)
        with self.assertRaises(ValueError):
            optimize_work_order([c1], [-0.1])

    def test_non_finite_priority_score_nan(self):
        """NaN priority score raises ValueError."""
        c1 = _make_complaint(1)
        with self.assertRaises(ValueError):
            optimize_work_order([c1], [float("nan")])

    def test_non_finite_priority_score_inf(self):
        """Inf priority score raises ValueError."""
        c1 = _make_complaint(1)
        with self.assertRaises(ValueError):
            optimize_work_order([c1], [float("inf")])

    def test_invalid_ga_settings_negative_pop(self):
        """Negative population_size raises ValueError."""
        c1 = _make_complaint(1)
        with self.assertRaises(ValueError):
            optimize_work_order([c1], [0.5], {"population_size": -1})

    def test_invalid_ga_settings_mutation_rate(self):
        """mutation_rate > 1 raises ValueError."""
        c1 = _make_complaint(1)
        with self.assertRaises(ValueError):
            optimize_work_order([c1], [0.5], {"mutation_rate": 2.0})

    # --------------------------------------------------------
    # ACTIVE vs CLOSED FILTERING
    # --------------------------------------------------------

    def test_active_only_filtering(self):
        """Closed complaints are excluded from the result by default."""
        c1 = _make_complaint(1, status="Open")
        c2 = _make_complaint(2, status="Closed")
        c3 = _make_complaint(3, status="Open")
        result = optimize_work_order([c1, c2, c3], [0.9, 0.8, 0.7])
        self.assertIn(1, result["optimized_order"])
        self.assertNotIn(2, result["optimized_order"])
        self.assertIn(3, result["optimized_order"])
        self.assertIn(2, result["excluded_complaints"])

    def test_include_closed_override(self):
        """include_closed=True keeps closed complaints."""
        c1 = _make_complaint(1, status="Open")
        c2 = _make_complaint(2, status="Closed")
        result = optimize_work_order(
            [c1, c2], [0.9, 0.8], {"include_closed": True, "random_seed": 42}
        )
        self.assertIn(1, result["optimized_order"])
        self.assertIn(2, result["optimized_order"])
        self.assertEqual(result["excluded_complaints"], [])

    # --------------------------------------------------------
    # PERMUTATION INTEGRITY
    # --------------------------------------------------------

    def test_all_complaints_appear_once(self):
        """Every active unique_key appears exactly once in optimised order."""
        complaints = [_make_complaint(i) for i in range(10)]
        scores = [i / 10.0 for i in range(10)]
        result = optimize_work_order(complaints, scores, {"random_seed": 42})
        self.assertEqual(sorted(result["optimized_order"]), list(range(10)))

    # --------------------------------------------------------
    # CROSSOVER AND MUTATION PRESERVE PERMUTATIONS
    # --------------------------------------------------------

    def test_crossover_preserves_permutation(self):
        """OX crossover produces a valid permutation."""
        rng = random.Random(123)
        parent1 = list(range(20))
        parent2 = list(range(20))
        rng.shuffle(parent2)
        child = _order_crossover(parent1, parent2, rng)
        self.assertEqual(sorted(child), list(range(20)))

    def test_mutation_preserves_permutation(self):
        """Swap mutation preserves a valid permutation."""
        rng = random.Random(456)
        individual = list(range(20))
        mutated = _swap_mutation(individual, 0.5, rng)  # high rate for coverage
        self.assertEqual(sorted(mutated), list(range(20)))

    # --------------------------------------------------------
    # PRIORITY AND SAFETY AFFECT FITNESS
    # --------------------------------------------------------

    def test_priority_affects_fitness(self):
        """Higher-priority complaint placed first yields lower fitness."""
        c_high = _make_complaint(1, severity=9, traffic=9, public=9,
                                 safety=0.9, days=20, freq=8, weather=8)
        c_low  = _make_complaint(2, severity=1, traffic=1, public=1,
                                 safety=0.1, days=1, freq=0, weather=1)
        p_high, p_low = 0.95, 0.1

        # Urgencies
        u_high = _urgency(c_high, p_high)
        u_low  = _urgency(c_low, p_low)
        s_high = _norm_safety(c_high["safety_risk"])
        s_low  = _norm_safety(c_low["safety_risk"])

        # Order: high first
        fit_good = _compute_fitness([0, 1], [u_high, u_low], [s_high, s_low])
        # Order: low first
        fit_bad  = _compute_fitness([1, 0], [u_high, u_low], [s_high, s_low])

        self.assertLess(fit_good, fit_bad)

    def test_safety_affects_fitness(self):
        """Placing a safety-critical complaint late increases cost via the penalty."""
        c_safe  = _make_complaint(1, safety=0.9)
        c_other = _make_complaint(2, safety=0.1)
        u = [_urgency(c_safe, 0.5), _urgency(c_other, 0.5)]
        s = [_norm_safety(c_safe["safety_risk"]), _norm_safety(c_other["safety_risk"])]

        fit_good = _compute_fitness([0, 1], u, s)  # safe first
        fit_bad  = _compute_fitness([1, 0], u, s)  # safe last
        self.assertLess(fit_good, fit_bad)

    # --------------------------------------------------------
    # FEATURE INCORPORATION
    # --------------------------------------------------------

    def test_days_pending_incorporated(self):
        """Higher days_pending increases urgency."""
        c1 = _make_complaint(1, days=0)
        c2 = _make_complaint(2, days=25)
        u1 = _urgency(c1, 0.5)
        u2 = _urgency(c2, 0.5)
        self.assertGreater(u2, u1)

    def test_traffic_public_weather_frequency_incorporated(self):
        """Non-zero traffic, public, weather, frequency increase urgency."""
        base = _make_complaint(1, traffic=0, public=0, weather=0, freq=0)
        high = _make_complaint(2, traffic=8, public=8, weather=8, freq=8)
        u_base = _urgency(base, 0.5)
        u_high = _urgency(high, 0.5)
        self.assertGreater(u_high, u_base)

    # --------------------------------------------------------
    # NO FABRICATED METRICS
    # --------------------------------------------------------

    def test_no_fabricated_travel_time(self):
        """travel_distance and repair_time must be None without real data."""
        complaints = [_make_complaint(i) for i in range(3)]
        scores = [0.5] * 3
        result = optimize_work_order(complaints, scores, {"random_seed": 1})
        self.assertIsNone(result["total_travel_distance"])
        self.assertIsNone(result["total_estimated_repair_time"])

    # --------------------------------------------------------
    # TRAVEL / WORKER CONSTRAINTS RESPECTED
    # --------------------------------------------------------

    def test_unsupported_constraints_reported(self):
        """Passing distance_matrix or worker_count is noted but doesn't crash."""
        c = _make_complaint(1)
        result = optimize_work_order([c], [0.5], {
            "distance_matrix": [[0, 1], [1, 0]],
            "worker_count": 3,
        })
        note_text = " ".join(result["notes"])
        self.assertIn("distance_matrix", note_text)
        self.assertIn("worker_count", note_text)

    # --------------------------------------------------------
    # REPRODUCIBILITY
    # --------------------------------------------------------

    def test_reproducibility(self):
        """Identical random seeds produce identical results."""
        complaints = [_make_complaint(i, severity=i, safety=i/10) for i in range(8)]
        scores = [i / 8 for i in range(8)]

        r1 = optimize_work_order(complaints, scores, {"random_seed": 999})
        r2 = optimize_work_order(complaints, scores, {"random_seed": 999})
        self.assertEqual(r1["optimized_order"], r2["optimized_order"])
        self.assertEqual(r1["fitness"], r2["fitness"])

    # --------------------------------------------------------
    # BASELINE USES SAME OBJECTIVE
    # --------------------------------------------------------

    def test_baseline_uses_same_objective(self):
        """Baseline fitness is computed using the same cost function."""
        complaints = [
            _make_complaint(1, severity=9, safety=0.9),
            _make_complaint(2, severity=1, safety=0.1),
        ]
        scores = [0.9, 0.1]
        result = optimize_work_order(complaints, scores, {
            "random_seed": 42, "population_size": 20, "n_generations": 30,
        })
        # Both values should be finite and non-negative
        self.assertIsInstance(result["baseline_fitness"], float)
        self.assertGreaterEqual(result["baseline_fitness"], 0.0)
        self.assertIsInstance(result["fitness"], float)
        self.assertGreaterEqual(result["fitness"], 0.0)
        # GA should be at least as good as baseline (lower or equal)
        self.assertLessEqual(result["fitness"], result["baseline_fitness"] + 1e-9)


if __name__ == "__main__":
    unittest.main()
