import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import backend.pipeline as pipeline
from data import database


class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        self.db_path_patch = patch.object(
            database, "DB_PATH", Path(self.temp_dir.name) / "pipeline.db"
        )
        self.db_path_patch.start()

    def tearDown(self):
        self.db_path_patch.stop()
        self.temp_dir.cleanup()

    def test_pipeline_uses_form_features_and_persists_work_order(self):
        complaint_input = {
            "unique_key": "web-complaint-1",
            "complaint_type": "Pothole",
            "descriptor": "Large road hazard",
            "incident_zip": "10001",
            "agency": "DOT",
            "status": "Open",
            "severity": 8.5,
            "traffic_impact": 7.0,
            "public_impact": 6.5,
            "weather_risk": 3.5,
            "days_pending": 2,
            "complaint_frequency": 4.0,
            "estimated_repair_time": 5.5,
        }

        with patch.object(pipeline, "_load_model_if_available", return_value=None):
            result = pipeline.run_pipeline(complaint_input)

        self.assertEqual(result["complaint_id"], "web-complaint-1")
        self.assertEqual(result["optimized_order"], ["web-complaint-1"])
        self.assertEqual(result["features"]["severity"], 8.5)
        self.assertEqual(result["features"]["traffic_impact"], 7.0)
        self.assertEqual(result["features"]["public_impact"], 6.5)
        self.assertEqual(result["features"]["weather_risk"], 3.5)
        self.assertEqual(result["features"]["estimated_repair_time"], 5.5)

        saved_complaint = database.get_complaint_by_key("web-complaint-1")
        self.assertIsNotNone(saved_complaint)
        self.assertEqual(saved_complaint["severity"], 8.5)
        saved_work_order = database.get_work_order(result["work_order_id"])
        self.assertIsNotNone(saved_work_order)
        self.assertEqual(
            saved_work_order["items"][0]["complaint_unique_key"], "web-complaint-1"
        )

    def test_pipeline_rejects_out_of_range_form_features(self):
        with self.assertRaisesRegex(ValueError, "severity must be between 0.0 and 10.0"):
            pipeline._build_feature_row(
                {"complaint_type": "Pothole", "severity": 11}
            )

    def test_pipeline_estimates_different_repair_times_by_complaint(self):
        pothole = pipeline._build_feature_row({
            "complaint_type": "Pothole",
            "descriptor": "Road surface damage",
        })
        sanitation = pipeline._build_feature_row({
            "complaint_type": "Sanitation Condition",
            "descriptor": "Overflowing garbage",
        })

        self.assertNotEqual(
            pothole["estimated_repair_time"],
            sanitation["estimated_repair_time"],
        )

    def test_pipeline_generates_readable_unique_complaint_ids(self):
        first_id = pipeline._build_feature_row({})["unique_key"]
        second_id = pipeline._build_feature_row({})["unique_key"]

        self.assertRegex(first_id, r"^CMP-\d{8}-[A-F0-9]{8}$")
        self.assertNotEqual(first_id, second_id)


if __name__ == "__main__":
    unittest.main()
