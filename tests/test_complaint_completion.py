import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from data import database


class TestComplaintCompletion(unittest.TestCase):
    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        self.db_path_patch = patch.object(
            database, "DB_PATH", Path(self.temp_dir.name) / "completion.db"
        )
        self.db_path_patch.start()
        database.init_db()

    def tearDown(self):
        self.db_path_patch.stop()
        self.temp_dir.cleanup()

    def test_completion_updates_complaint_work_order_item_and_order(self):
        complaint = {
            "unique_key": "complaint-1",
            "created_date": "2026-10-11T09:00:00",
            "agency": "DOT",
            "complaint_type": "Pothole",
            "descriptor": "Road surface damaged",
            "incident_zip": "10001",
            "status": "open",
            "severity": 8.0,
            "traffic_impact": 8.0,
            "public_impact": 7.0,
            "weather_risk": 6.0,
            "days_pending": 2.0,
            "frequency_count": 1,
            "complaint_frequency": 3.0,
            "safety_risk": 0.65,
            "priority_score": 0.75,
            "priority_category": "HIGH",
        }
        database.save_complaint(complaint)
        database.save_work_order("WO-1", [complaint])

        self.assertTrue(database.mark_complaint_completed("complaint-1"))

        saved_complaint = database.get_complaint_by_key("complaint-1")
        saved_order = database.get_work_order("WO-1")
        self.assertEqual(saved_complaint["status"], "closed")
        self.assertIsNotNone(saved_complaint["closed_date"])
        self.assertEqual(saved_order["items"][0]["status"], "completed")
        self.assertEqual(saved_order["header"]["status"], "completed")
        self.assertFalse(database.mark_complaint_completed("complaint-1"))

    def test_completion_returns_false_for_unknown_complaint(self):
        self.assertFalse(database.mark_complaint_completed("missing"))


if __name__ == "__main__":
    unittest.main()
