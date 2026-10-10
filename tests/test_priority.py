import unittest

from backend.fuzzy_engine import calculate_safety_risk
from backend.pipeline import _estimate_priority_score, _priority_category


class TestPriorityInference(unittest.TestCase):
    def test_priority_category_boundaries(self):
        cases = [
            (0.0, "LOW"),
            (0.3999, "LOW"),
            (0.4, "MEDIUM"),
            (0.5999, "MEDIUM"),
            (0.6, "HIGH"),
            (0.7999, "HIGH"),
            (0.8, "CRITICAL"),
            (1.0, "CRITICAL"),
        ]

        for score, expected in cases:
            with self.subTest(score=score):
                self.assertEqual(_priority_category(score), expected)

    def test_fallback_priority_score_is_bounded_and_uses_all_features(self):
        minimum = {
            "safety_risk": 0,
            "severity": 0,
            "traffic_impact": 0,
            "public_impact": 0,
            "weather_risk": 0,
            "days_pending": 0,
            "complaint_frequency": 0,
        }
        maximum = {
            "safety_risk": 1,
            "severity": 10,
            "traffic_impact": 10,
            "public_impact": 10,
            "weather_risk": 10,
            "days_pending": 30,
            "complaint_frequency": 10,
        }

        self.assertEqual(_estimate_priority_score(minimum), 0.0)
        self.assertEqual(_estimate_priority_score(maximum), 1.0)
        self.assertGreater(
            _estimate_priority_score(maximum),
            _estimate_priority_score(minimum),
        )

    def test_fuzzy_safety_risk_is_bounded_and_increases_for_higher_risk(self):
        low_risk = {
            "severity": 3,
            "traffic_impact": 1,
            "public_impact": 2,
            "weather_risk": 1,
            "days_pending": 0,
            "complaint_frequency": 0,
        }
        high_risk = {
            "severity": 9,
            "traffic_impact": 10,
            "public_impact": 10,
            "weather_risk": 9,
            "days_pending": 25,
            "complaint_frequency": 10,
        }

        low_score = calculate_safety_risk(low_risk)
        high_score = calculate_safety_risk(high_risk)

        self.assertGreaterEqual(low_score, 0.0)
        self.assertLessEqual(high_score, 1.0)
        self.assertGreater(high_score, low_score)

    def test_fuzzy_safety_risk_rejects_missing_fields(self):
        with self.assertRaisesRegex(ValueError, "Missing complaint fields"):
            calculate_safety_risk({"severity": 5})


if __name__ == "__main__":
    unittest.main()
