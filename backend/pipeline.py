from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from backend.feature_engineering import (
    combine_text,
    derive_complaint_frequency,
    derive_days_pending,
    derive_public_impact,
    derive_severity,
    derive_traffic_impact,
    derive_weather_risk,
)
from backend.fuzzy_engine import calculate_safety_risk
from backend.genetic_optimizer import optimize_work_order
from data.database import get_connection, init_db, save_complaint, save_work_order


def _safe_text(value, default=""):
    return str(value).strip() if value is not None and str(value).strip() else default


def _priority_category(score: float) -> str:
    if score >= 0.80:
        return "CRITICAL"
    if score >= 0.60:
        return "HIGH"
    if score >= 0.40:
        return "MEDIUM"
    return "LOW"


def _estimate_priority_score(complaint: dict) -> float:
    """Fallback priority model used when the saved Keras model is unavailable."""
    safety = float(complaint.get("safety_risk", 0.0))
    severity = float(complaint.get("severity", 0.0)) / 10.0
    traffic = float(complaint.get("traffic_impact", 0.0)) / 10.0
    public = float(complaint.get("public_impact", 0.0)) / 10.0
    weather = float(complaint.get("weather_risk", 0.0)) / 10.0
    pending = min(max(float(complaint.get("days_pending", 0.0)), 0.0), 30.0) / 30.0
    freq = float(complaint.get("complaint_frequency", 0.0)) / 10.0

    score = (
        0.35 * safety
        + 0.20 * severity
        + 0.15 * traffic
        + 0.10 * public
        + 0.10 * weather
        + 0.05 * pending
        + 0.05 * freq
    )
    return float(np.clip(score, 0.0, 1.0))


def _load_model_if_available():
    model_path = ROOT / "models" / "priority_model.keras"
    if not model_path.exists():
        return None

    try:
        import tensorflow as tf

        return tf.keras.models.load_model(str(model_path))
    except Exception:
        return None


def _predict_priority_score(complaint: dict) -> float:
    model = _load_model_if_available()
    if model is not None:
        try:
            feature_vector = np.array(
                [
                    float(complaint.get("severity", 0.0)),
                    float(complaint.get("traffic_impact", 0.0)),
                    float(complaint.get("public_impact", 0.0)),
                    float(complaint.get("weather_risk", 0.0)),
                    float(complaint.get("days_pending", 0.0)),
                    float(complaint.get("complaint_frequency", 0.0)),
                    float(complaint.get("safety_risk", 0.0)),
                ],
                dtype=float,
            ).reshape(1, -1)
            prediction = model.predict(feature_vector, verbose=0)
            value = float(np.asarray(prediction).reshape(-1)[0])
            if np.isfinite(value):
                return float(np.clip(value, 0.0, 1.0))
        except Exception:
            pass

    return _estimate_priority_score(complaint)


def _build_feature_row(complaint_input: dict) -> dict:
    complaint = {
        "unique_key": complaint_input.get("unique_key") or complaint_input.get("id") or f"complaint_{int(datetime.now().timestamp() * 1000)}",
        "created_date": complaint_input.get("created_date") or datetime.now().isoformat(timespec="seconds"),
        "closed_date": complaint_input.get("closed_date"),
        "agency": _safe_text(complaint_input.get("agency"), "USER"),
        "complaint_type": _safe_text(complaint_input.get("complaint_type"), "Unknown"),
        "descriptor": _safe_text(complaint_input.get("descriptor"), "Unknown"),
        "incident_zip": _safe_text(complaint_input.get("incident_zip"), "UNKNOWN"),
        "status": _safe_text(complaint_input.get("status"), "Open").lower(),
    }

    complaint["severity"] = float(derive_severity(complaint))
    complaint["traffic_impact"] = float(derive_traffic_impact(complaint))
    complaint["public_impact"] = float(derive_public_impact(complaint))
    complaint["weather_risk"] = float(derive_weather_risk(complaint))

    complaint["days_pending"] = float(
        complaint_input.get("days_pending")
        if complaint_input.get("days_pending") is not None
        else derive_days_pending(complaint)
    )

    complaint["frequency_count"] = int(complaint_input.get("frequency_count", 0))
    freq_value = complaint_input.get("complaint_frequency")
    complaint["complaint_frequency"] = float(freq_value) if freq_value is not None else 0.0

    complaint["safety_risk"] = float(calculate_safety_risk({
        "severity": complaint["severity"],
        "traffic_impact": complaint["traffic_impact"],
        "public_impact": complaint["public_impact"],
        "weather_risk": complaint["weather_risk"],
        "days_pending": complaint["days_pending"],
        "complaint_frequency": complaint["complaint_frequency"],
    }))

    complaint["priority_score"] = float(_predict_priority_score(complaint))
    complaint["priority_category"] = _priority_category(complaint["priority_score"])

    return complaint


def run_pipeline(complaint_input: dict) -> dict:
    """Run the full flow from raw user complaint to optimized work order and DB storage."""
    init_db()

    complaint = _build_feature_row(complaint_input)
    result = optimize_work_order([complaint], [complaint["priority_score"]], {"random_seed": 42})

    work_order_id = f"WO-{complaint['unique_key']}"

    complaint_record = {
        **complaint,
        "priority_score": complaint["priority_score"],
        "priority_category": complaint["priority_category"],
        "safety_risk": complaint["safety_risk"],
    }

    save_complaint(complaint_record)
    save_work_order(work_order_id, [complaint_record])

    return {
        "complaint_id": complaint["unique_key"],
        "work_order_id": work_order_id,
        "features": {
            "severity": complaint["severity"],
            "traffic_impact": complaint["traffic_impact"],
            "public_impact": complaint["public_impact"],
            "weather_risk": complaint["weather_risk"],
            "days_pending": complaint["days_pending"],
            "complaint_frequency": complaint["complaint_frequency"],
            "safety_risk": complaint["safety_risk"],
            "priority_score": complaint["priority_score"],
            "priority_category": complaint["priority_category"],
        },
        "optimized_order": result["optimized_order"],
        "fitness": result["fitness"],
        "baseline_fitness": result["baseline_fitness"],
        "notes": result["notes"],
        "saved_to_db": True,
    }


def _parse_cli_args(argv):
    if not argv:
        raise ValueError("Provide complaint_type, descriptor, incident_zip, and optional agency")

    complaint_type = argv[0]
    descriptor = argv[1] if len(argv) > 1 else "Unknown"
    incident_zip = argv[2] if len(argv) > 2 else "UNKNOWN"
    agency = argv[3] if len(argv) > 3 else "USER"

    return {
        "complaint_type": complaint_type,
        "descriptor": descriptor,
        "incident_zip": incident_zip,
        "agency": agency,
        "status": "Open",
    }


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    complaint_input = _parse_cli_args(args)
    output = run_pipeline(complaint_input)
    print(json.dumps(output, indent=2))
    return output


if __name__ == "__main__":
    main()
