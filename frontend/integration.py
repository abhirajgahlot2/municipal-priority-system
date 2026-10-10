"""
integration.py – Member 4: Application Integration Layer
=========================================================
Coordinates Member 1 (Fuzzy Logic), Member 2 (ANN / Priority Prediction),
Member 3 (Genetic Algorithm Optimizer), and Member 5 (SQLite Database).

Flow:
    complaint
    -> calculate_safety_risk()
    -> predict_priority()
    -> optimize_work_order()
    -> save_complaint() / save_work_order()
    -> return enriched result
"""

from __future__ import annotations

import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Member 1: Fuzzy Engine
from backend.fuzzy_engine import calculate_safety_risk

# Member 2: ANN / Machine Learning prediction & feature engineering
from backend.pipeline import (
    _estimate_priority_score,
    _load_model_if_available,
    _predict_priority_score,
    _priority_category,
)
from backend.feature_engineering import (
    derive_severity,
    derive_traffic_impact,
    derive_public_impact,
    derive_weather_risk,
    derive_days_pending,
)

# Member 3: Genetic Algorithm
from backend.genetic_optimizer import optimize_work_order

# Member 5: Database
from data.database import (
    DB_PATH,
    get_connection,
    init_db,
    save_complaint,
    save_work_order,
    get_complaint_by_key,
    get_work_order,
)


def normalize_complaint_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure all fields conform to the COMMON DATA CONTRACT.
    Maps aliases:
        traffic_level -> traffic_impact
        people_affected -> public_impact
        location -> incident_zip
    """
    now_iso = datetime.now().isoformat(timespec="seconds")
    unique_key = (
        data.get("unique_key")
        or data.get("id")
        or f"CMP-{int(datetime.now().timestamp() * 1000)}"
    )

    complaint_type = str(data.get("complaint_type") or "Pothole").strip()
    descriptor = str(data.get("descriptor") or complaint_type).strip()
    incident_zip = str(
        data.get("incident_zip") or data.get("location") or "10001"
    ).strip()
    agency = str(data.get("agency") or "DOT").strip()
    status = str(data.get("status") or "Open").strip().lower()

    # Numerical fields with validation to 0-10 scale
    def _to_float_range(val, default, low=0.0, high=10.0):
        if val is None or val == "":
            return float(default)
        try:
            return float(np.clip(float(val), low, high))
        except (ValueError, TypeError):
            return float(default)

    severity = _to_float_range(data.get("severity"), 5.0)
    traffic_impact = _to_float_range(
        data.get("traffic_impact", data.get("traffic_level")), 5.0
    )
    public_impact = _to_float_range(
        data.get("public_impact", data.get("people_affected")), 5.0
    )
    weather_risk = _to_float_range(data.get("weather_risk"), 3.0)
    days_pending = _to_float_range(data.get("days_pending"), 1.0, low=0.0, high=365.0)
    complaint_frequency = _to_float_range(data.get("complaint_frequency"), 1.0)
    frequency_count = int(data.get("frequency_count") or int(round(complaint_frequency)))

    estimated_repair_time = _to_float_range(
        data.get("estimated_repair_time"), 4.0, low=0.5, high=120.0
    )

    return {
        "unique_key": unique_key,
        "created_date": data.get("created_date") or now_iso,
        "closed_date": data.get("closed_date"),
        "agency": agency,
        "complaint_type": complaint_type,
        "descriptor": descriptor,
        "incident_zip": incident_zip,
        "status": status,
        "severity": round(severity, 1),
        "traffic_impact": round(traffic_impact, 1),
        "traffic_level": round(traffic_impact, 1),
        "public_impact": round(public_impact, 1),
        "people_affected": round(public_impact, 1),
        "weather_risk": round(weather_risk, 1),
        "days_pending": round(days_pending, 1),
        "frequency_count": frequency_count,
        "complaint_frequency": round(complaint_frequency, 1),
        "estimated_repair_time": round(estimated_repair_time, 1),
    }


def predict_priority(complaint: Dict[str, Any], safety_risk: float) -> Tuple[float, str]:
    """
    Member 2 interface:
    Predicts priority_score (0.0 to 1.0) and assigns priority_category.
    """
    record = dict(complaint)
    record["safety_risk"] = float(safety_risk)
    priority_score = float(_predict_priority_score(record))
    category = _priority_category(priority_score)
    return round(priority_score, 4), category


def process_complaint(raw_complaint: Dict[str, Any], auto_save: bool = True) -> Dict[str, Any]:
    """
    Full Member 4 Pipeline Controller:
    Input: raw complaint dictionary
    Flow:
        1. Normalize fields (Common Data Contract)
        2. Member 1: calculate_safety_risk()
        3. Member 2: predict_priority()
        4. Member 5: save_complaint()
    Returns:
        Complete processed complaint record with derived features and predictions.
    """
    init_db()
    complaint = normalize_complaint_dict(raw_complaint)

    # Step 1: Member 1 Fuzzy Logic Engine
    fuzzy_input = {
        "severity": complaint["severity"],
        "traffic_impact": complaint["traffic_impact"],
        "public_impact": complaint["public_impact"],
        "weather_risk": complaint["weather_risk"],
        "days_pending": min(complaint["days_pending"], 30.0),
        "complaint_frequency": complaint["complaint_frequency"],
    }
    safety_risk = float(calculate_safety_risk(fuzzy_input))
    complaint["safety_risk"] = round(safety_risk, 4)

    # Step 2: Member 2 ANN / ML Inference
    priority_score, priority_category = predict_priority(complaint, safety_risk)
    complaint["priority_score"] = priority_score
    complaint["priority_category"] = priority_category

    # Step 3: Member 5 Database Persistence
    if auto_save:
        save_complaint(complaint)

    return complaint


def run_batch_work_order_optimization(
    complaints: List[Dict[str, Any]],
    constraints: Optional[Dict[str, Any]] = None,
    save_to_db: bool = True,
) -> Dict[str, Any]:
    """
    Member 3 & 4 Batch Optimization Controller:
    Runs the Genetic Algorithm across multiple complaints and creates a Work Order.
    """
    if not complaints:
        return {
            "work_order_id": None,
            "optimized_order": [],
            "ordered_complaints": [],
            "fitness": 0.0,
            "baseline_fitness": 0.0,
            "improvement_pct": 0.0,
            "notes": ["No complaints provided for optimization."],
        }

    init_db()

    # Pre-process all complaints to ensure safety_risk and priority_score exist
    processed_list: List[Dict[str, Any]] = []
    priority_scores: List[float] = []

    for c in complaints:
        if "safety_risk" not in c or "priority_score" not in c:
            proc = process_complaint(c, auto_save=save_to_db)
        else:
            proc = normalize_complaint_dict(c)
            proc["safety_risk"] = float(c["safety_risk"])
            proc["priority_score"] = float(c["priority_score"])
            proc["priority_category"] = c.get("priority_category") or _priority_category(
                proc["priority_score"]
            )
            if save_to_db:
                save_complaint(proc)
        processed_list.append(proc)
        priority_scores.append(proc["priority_score"])

    # GA hyperparameter constraints
    ga_params = {
        "population_size": 30,
        "n_generations": 40,
        "mutation_rate": 0.15,
        "crossover_rate": 0.85,
        "elite_size": 2,
        "random_seed": 42,
    }
    if constraints:
        ga_params.update(constraints)

    # Step 3: Member 3 Genetic Algorithm
    ga_result = optimize_work_order(processed_list, priority_scores, ga_params)

    optimized_keys = ga_result.get("optimized_order", [])
    complaint_map = {c["unique_key"]: c for c in processed_list}

    # Sequence of complaints according to the GA solution
    ordered_complaints = [
        complaint_map[key] for key in optimized_keys if key in complaint_map
    ]

    # Calculate percentage improvement over baseline
    base_fit = ga_result.get("baseline_fitness", 0.0)
    opt_fit = ga_result.get("fitness", 0.0)
    improvement_pct = 0.0
    if base_fit > 0:
        improvement_pct = max(0.0, ((base_fit - opt_fit) / base_fit) * 100.0)

    # Generate unique work order ID
    timestamp_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    work_order_id = f"WO-{timestamp_str}"

    # Step 5: Save to Database
    if save_to_db and ordered_complaints:
        save_work_order(work_order_id, ordered_complaints)

    return {
        "work_order_id": work_order_id,
        "optimized_order": optimized_keys,
        "ordered_complaints": ordered_complaints,
        "fitness": round(opt_fit, 4),
        "baseline_fitness": round(base_fit, 4),
        "improvement_pct": round(improvement_pct, 2),
        "notes": ga_result.get("notes", []),
        "excluded_complaints": ga_result.get("excluded_complaints", []),
    }


# ============================================================
# DATABASE QUERY & METRICS HELPERS FOR DASHBOARD
# ============================================================

def get_all_complaints_df() -> List[Dict[str, Any]]:
    """Fetch all complaints from Member 5's SQLite database."""
    init_db()
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM complaints ORDER BY id DESC")
        rows = [dict(r) for r in cursor.fetchall()]
        return rows
    finally:
        conn.close()


def get_all_work_orders_summary() -> List[Dict[str, Any]]:
    """Fetch all work orders and item counts from SQLite database."""
    init_db()
    conn = get_connection()
    try:
        cursor = conn.cursor()
        query = """
        SELECT w.work_order_id, w.created_at, w.status, w.notes,
               COUNT(i.id) as item_count,
               AVG(i.priority_score) as avg_priority,
               AVG(i.safety_risk) as avg_safety
        FROM work_orders w
        LEFT JOIN work_order_items i ON w.work_order_id = i.work_order_id
        GROUP BY w.work_order_id
        ORDER BY w.id DESC
        """
        cursor.execute(query)
        rows = [dict(r) for r in cursor.fetchall()]
        return rows
    finally:
        conn.close()


def get_work_order_details(work_order_id: str) -> Optional[Dict[str, Any]]:
    """Fetch full work order with ordered items."""
    return get_work_order(work_order_id)


def seed_sample_complaints(num_samples: int = 25) -> int:
    """
    Populates SQLite database with high quality sample records from
    the processed NYC 311 dataset so evaluators can immediately test
    the dashboard and GA scheduler without manual data entry.
    """
    csv_path = PROJECT_ROOT / "data" / "processed" / "processed_complaints.csv"
    if not csv_path.exists():
        csv_path = PROJECT_ROOT / "processed_complaints.csv"
    if not csv_path.exists():
        return 0

    import pandas as pd
    df = pd.read_csv(csv_path)
    sample_df = df.sample(min(num_samples, len(df)), random_state=42)

    count = 0
    for _, row in sample_df.iterrows():
        comp_dict = row.to_dict()
        process_complaint(comp_dict, auto_save=True)
        count += 1

    return count
