# Municipal Complaint Priority System

This project processes municipal complaints, estimates their risk and priority, creates a work-order record, and stores the result in SQLite.

## What the Code Does

The main runnable workflow is `backend/pipeline.py`:

1. **Accepts a complaint.** The command-line interface takes the complaint type, descriptor, ZIP code, and agency. The Python API also accepts dates, status, an ID, pending days, and recurrence values.
2. **Builds features.** `backend/feature_engineering.py` estimates severity, traffic impact, public impact, weather risk, and days pending. These are domain-based estimates from complaint text and dates, not measurements from live sensors.
3. **Calculates fuzzy safety risk.** `backend/fuzzy_engine.py` combines severity, traffic impact, public impact, weather risk, days pending, and complaint frequency into a safety-risk score from 0 to 1.
4. **Predicts priority.** The pipeline attempts to load `models/priority_model.keras`. If the model is missing or cannot load, it uses a weighted fallback score. Scores are categorized as `LOW`, `MEDIUM`, `HIGH`, or `CRITICAL`.
5. **Optimizes an order.** `backend/genetic_optimizer.py` orders active complaints using priority, safety, severity, impact, age, weather, and recurrence signals. Distance and repair-time objectives are unavailable without location and repair-time data. The current pipeline sends one complaint to the optimizer per run, so each run creates a single-complaint work order.
6. **Persists and returns results.** The database layer creates `data/complaints.db` as needed and stores the complaint, work-order header, and work-order item. The pipeline prints a JSON result containing the IDs, features, priority, optimized order, fitness, and notes.

### Recurrence and Pending Days

When using `engineer_features()` with a dataframe, recurrence counts earlier complaints with the same type and ZIP code in the preceding 30 days. It converts counts to a 0-10 score: 0 maps to 0, 1 to 3, 2 to 6, 3 to 8, and 4 or more to 10.

The single-complaint pipeline does not look up complaint history to calculate recurrence. If `complaint_frequency` is not supplied through the Python API, it defaults to 0. `days_pending` is calculated from the creation and close dates unless explicitly supplied.

## Requirements

Use Python 3.11 or a compatible Python version. Install the core packages from the project root:

```powershell
py -m pip install numpy pandas scikit-fuzzy
```

To run the tests, also install pytest:

```powershell
py -m pip install pytest
```

TensorFlow is only needed to attempt loading the saved Keras model. The code falls back to its weighted score if TensorFlow is unavailable or the model fails to load. In the checked environment, `models/priority_model.keras` could not be deserialized by TensorFlow 2.20, so the fallback score was used.

## Run the Pipeline

From the repository root in PowerShell, pass the complaint type followed by the descriptor, ZIP code, and agency:

```powershell
py -m backend.pipeline "Pothole" "Large dangerous pothole affecting traffic" "10001" "DOT"
```

The first argument (complaint type) is required. The remaining arguments are optional and default to `Unknown`, `UNKNOWN`, and `USER` respectively. The command marks the complaint as open, prints the pipeline result as JSON, and stores the complaint and work order in `data/complaints.db`.

### Use the Python API

The API allows additional fields that are not available as command-line arguments:

```python
from backend.pipeline import run_pipeline

result = run_pipeline({
	"unique_key": "complaint-123",
	"created_date": "2026-10-10T09:00:00",
	"complaint_type": "Pothole",
	"descriptor": "Large dangerous pothole affecting traffic",
	"incident_zip": "10001",
	"agency": "DOT",
	"status": "Open",
	"days_pending": 1,
	"frequency_count": 1,
	"complaint_frequency": 3,
})

print(result)
```

If no unique key is provided, the pipeline generates one. Calls using an existing unique key replace that complaint record in SQLite.

## Database

The database file is created on the first pipeline run at `data/complaints.db`. Its schema is defined in `data/schema.sql`:

- `complaints`: input details, derived features, safety risk, and priority.
- `work_orders`: generated work-order ID, status, and notes.
- `work_order_items`: complaint membership, queue position, priority, and risk.

## Run Tests

From the repository root:

```powershell
py -m pytest tests -q
```

The tests cover feature engineering, priority and fuzzy-risk calculations, work-order behavior, database persistence, and genetic-optimizer behavior.

## Current Scope

The main workflow is currently a command-line/Python pipeline. `app.py` and `frontend/` do not currently provide a runnable user interface. Travel distance and repair-time estimates are not calculated because corresponding input data is not available.
