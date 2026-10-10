import unittest

from backend.genetic_optimizer import optimize_work_order


class TestWorkOrder(unittest.TestCase):
    def setUp(self):
        self.settings = {
            "population_size": 20,
            "n_generations": 10,
            "random_seed": 42,
        }

    @staticmethod
    def make_complaint(unique_key, status="Open", severity=5, safety_risk=0.5):
        return {
            "unique_key": unique_key,
            "status": status,
            "severity": severity,
            "traffic_impact": severity,
            "public_impact": severity,
            "weather_risk": severity,
            "days_pending": severity,
            "complaint_frequency": severity,
            "safety_risk": safety_risk,
        }

    def test_work_order_contains_active_complaints_and_excludes_closed(self):
        complaints = [
            self.make_complaint("high", severity=9, safety_risk=0.9),
            self.make_complaint("closed", status="Closed", severity=10),
            self.make_complaint("low", severity=2, safety_risk=0.2),
        ]

        result = optimize_work_order(
            complaints,
            [0.9, 1.0, 0.2],
            self.settings,
        )

        self.assertEqual(set(result["optimized_order"]), {"high", "low"})
        self.assertEqual(len(result["optimized_order"]), 2)
        self.assertEqual(result["excluded_complaints"], ["closed"])

    def test_work_order_is_empty_when_all_complaints_are_closed(self):
        complaints = [
            self.make_complaint("closed-1", status="Closed"),
            self.make_complaint("closed-2", status="closed"),
        ]

        result = optimize_work_order(complaints, [0.5, 0.8], self.settings)

        self.assertEqual(result["optimized_order"], [])
        self.assertEqual(result["excluded_complaints"], ["closed-1", "closed-2"])
        self.assertEqual(result["fitness"], 0.0)

    def test_work_order_ordering_is_reproducible_for_same_seed(self):
        complaints = [
            self.make_complaint(f"complaint-{index}", severity=index + 1)
            for index in range(5)
        ]
        scores = [0.1, 0.3, 0.5, 0.7, 0.9]

        first = optimize_work_order(complaints, scores, self.settings)
        second = optimize_work_order(complaints, scores, self.settings)

        self.assertEqual(first["optimized_order"], second["optimized_order"])
        self.assertEqual(first["fitness"], second["fitness"])

    def test_disabled_distance_and_repair_metrics_are_not_fabricated(self):
        complaint = self.make_complaint("single")

        result = optimize_work_order([complaint], [0.5], self.settings)

        self.assertIsNone(result["total_travel_distance"])
        self.assertIsNone(result["total_estimated_repair_time"])
        self.assertEqual(result["optimized_order"], ["single"])


if __name__ == "__main__":
    unittest.main()
