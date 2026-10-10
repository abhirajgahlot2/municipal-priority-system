import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from data import database


class TestDatabase(unittest.TestCase):
    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_complaints.db"
        self.db_path_patch = patch.object(database, "DB_PATH", self.db_path)
        self.db_path_patch.start()
        database.init_db()

    def tearDown(self):
        self.db_path_patch.stop()
        self.temp_dir.cleanup()

    @staticmethod
    def make_complaint(unique_key="complaint-1", priority_score=0.75):
        return {
            "unique_key": unique_key,
            "created_date": "2026-10-10T09:00:00",
            "closed_date": None,
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
            "priority_score": priority_score,
            "priority_category": "HIGH",
        }

    def test_init_db_creates_database_and_required_tables(self):
        self.assertTrue(self.db_path.exists())

        connection = database.get_connection()
        try:
            table_names = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
        finally:
            connection.close()

        self.assertTrue(
            {"complaints", "work_orders", "work_order_items"}.issubset(table_names)
        )

    def test_save_and_get_complaint_round_trip(self):
        complaint = self.make_complaint()

        database.save_complaint(complaint)
        saved = database.get_complaint_by_key(complaint["unique_key"])

        self.assertIsNotNone(saved)
        self.assertEqual(saved["unique_key"], complaint["unique_key"])
        self.assertEqual(saved["complaint_type"], "Pothole")
        self.assertEqual(saved["priority_category"], "HIGH")
        self.assertAlmostEqual(saved["priority_score"], 0.75)

    def test_save_and_get_work_order_with_ordered_items(self):
        complaints = [
            self.make_complaint("complaint-1", 0.9),
            self.make_complaint("complaint-2", 0.6),
        ]
        for complaint in complaints:
            database.save_complaint(complaint)

        database.save_work_order("WO-1", complaints)
        saved = database.get_work_order("WO-1")

        self.assertIsNotNone(saved)
        self.assertEqual(saved["header"]["work_order_id"], "WO-1")
        self.assertEqual(saved["header"]["status"], "pending")
        self.assertEqual(
            [item["complaint_unique_key"] for item in saved["items"]],
            ["complaint-1", "complaint-2"],
        )
        self.assertEqual([item["queue_position"] for item in saved["items"]], [1, 2])

    def test_saving_same_complaint_replaces_existing_values(self):
        complaint = self.make_complaint()
        database.save_complaint(complaint)
        updated = {**complaint, "priority_score": 0.91, "priority_category": "CRITICAL"}

        database.save_complaint(updated)
        saved = database.get_complaint_by_key(complaint["unique_key"])

        self.assertAlmostEqual(saved["priority_score"], 0.91)
        self.assertEqual(saved["priority_category"], "CRITICAL")

    def test_missing_complaint_and_work_order_return_none(self):
        self.assertIsNone(database.get_complaint_by_key("missing"))
        self.assertIsNone(database.get_work_order("missing"))


if __name__ == "__main__":
    unittest.main()
