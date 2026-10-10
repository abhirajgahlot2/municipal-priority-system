import unittest

import pandas as pd

from backend.feature_engineering import (
    combine_text,
    derive_complaint_frequency,
    derive_days_pending,
    derive_public_impact,
    derive_severity,
    derive_traffic_impact,
    derive_weather_risk,
    engineer_features,
)


class TestFeatureEngineering(unittest.TestCase):
    def test_combine_text_lowercases_complaint_fields(self):
        row = {
            "complaint_type": "Street Light",
            "descriptor": "Light Out",
        }

        self.assertEqual(combine_text(row), "street light light out")

    def test_severity_distinguishes_hazard_from_minor_complaint(self):
        hazard = {
            "complaint_type": "Pothole",
            "descriptor": "Dangerous road obstruction",
        }
        minor_noise = {
            "complaint_type": "Noise",
            "descriptor": "Minor issue",
        }

        self.assertEqual(derive_severity(hazard), 10.0)
        self.assertEqual(derive_severity(minor_noise), 2.0)

    def test_traffic_public_and_weather_scores_cover_key_categories(self):
        traffic_signal = {
            "complaint_type": "Traffic Signal",
            "descriptor": "Signal malfunction",
        }
        water_main_break = {
            "complaint_type": "Water Main Break",
            "descriptor": "Water outage",
        }
        flooding = {
            "complaint_type": "Street Flooding",
            "descriptor": "Flooding",
        }

        self.assertEqual(derive_traffic_impact(traffic_signal), 10.0)
        self.assertEqual(derive_public_impact(water_main_break), 9.0)
        self.assertEqual(derive_weather_risk(flooding), 9.0)

    def test_days_pending_handles_closed_invalid_and_negative_durations(self):
        closed = {
            "created_date": "2026-01-01",
            "closed_date": "2026-01-04",
        }
        invalid_date = {"created_date": "not-a-date", "closed_date": None}
        negative_duration = {
            "created_date": "2026-01-02",
            "closed_date": "2026-01-01",
        }

        self.assertEqual(derive_days_pending(closed), 3.0)
        self.assertEqual(derive_days_pending(invalid_date), 0.0)
        self.assertEqual(derive_days_pending(negative_duration), 0.0)

    def test_frequency_counts_same_type_and_zip_only_within_30_days(self):
        records = [
            {
                "complaint_type": "Pothole",
                "incident_zip": "10001",
                "created_date": f"2026-01-0{day}",
            }
            for day in range(1, 7)
        ]
        records.extend([
            {
                "complaint_type": "Noise",
                "incident_zip": "10001",
                "created_date": "2026-01-07",
            },
            {
                "complaint_type": "Pothole",
                "incident_zip": "10002",
                "created_date": "2026-01-08",
            },
            {
                "complaint_type": "Pothole",
                "incident_zip": "10001",
                "created_date": "2026-01-09",
            },
            {
                "complaint_type": "Pothole",
                "incident_zip": "10001",
                "created_date": "2026-03-01",
            },
        ])

        result = derive_complaint_frequency(pd.DataFrame(records))
        rows_by_date = result.set_index("created_date")

        self.assertEqual(
            result.loc[:5, "frequency_count"].tolist(),
            [0, 1, 2, 3, 4, 5],
        )
        self.assertEqual(
            result.loc[:5, "complaint_frequency"].tolist(),
            [0.0, 3.0, 6.0, 8.0, 10.0, 10.0],
        )
        self.assertEqual(rows_by_date.loc[pd.Timestamp("2026-01-07"), "frequency_count"], 0)
        self.assertEqual(rows_by_date.loc[pd.Timestamp("2026-01-08"), "frequency_count"], 0)
        self.assertEqual(rows_by_date.loc[pd.Timestamp("2026-01-09"), "frequency_count"], 6)
        self.assertEqual(rows_by_date.loc[pd.Timestamp("2026-03-01"), "frequency_count"], 0)

    def test_engineer_features_fills_text_and_adds_finite_features_without_mutating_input(self):
        raw = pd.DataFrame([
            {
                "complaint_type": "Pothole",
                "descriptor": "Road damage",
                "incident_zip": "10001",
                "created_date": "2026-01-01",
                "closed_date": "2026-01-03",
            },
            {
                "complaint_type": None,
                "descriptor": None,
                "incident_zip": None,
                "created_date": "invalid-date",
                "closed_date": None,
            },
        ])
        original_columns = set(raw.columns)

        result = engineer_features(raw)

        expected_features = {
            "severity",
            "traffic_impact",
            "public_impact",
            "weather_risk",
            "days_pending",
            "frequency_count",
            "complaint_frequency",
        }
        self.assertTrue(expected_features.issubset(result.columns))
        self.assertEqual(set(raw.columns), original_columns)
        self.assertEqual(result.loc[1, "complaint_type"], "unknown")
        self.assertEqual(result.loc[1, "descriptor"], "Unknown")
        self.assertEqual(result.loc[1, "days_pending"], 0.0)
        self.assertTrue(result[list(expected_features)].notna().all().all())
        self.assertTrue(result["severity"].between(0, 10).all())
        self.assertTrue(result["traffic_impact"].between(0, 10).all())
        self.assertTrue(result["public_impact"].between(0, 10).all())
        self.assertTrue(result["weather_risk"].between(0, 10).all())
        self.assertTrue(result["complaint_frequency"].between(0, 10).all())


if __name__ == "__main__":
    unittest.main()
