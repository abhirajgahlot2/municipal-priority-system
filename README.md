# Municipal Complaint Priority System

The Municipal Complaint Priority System accepts municipal service complaints, calculates risk and priority, and schedules work orders. The Streamlit dashboard is the main user interface; complaints and work orders are stored in SQLite.

## Requirements

- Python 3.10 or newer
- pip

Install the application dependencies from the repository root:

```powershell
py -m pip install numpy pandas scikit-fuzzy streamlit
```

TensorFlow is optional. If `models/priority_model.keras` is available and can be loaded, the pipeline uses it to predict priority. If TensorFlow or the model is unavailable, the pipeline uses its weighted fallback score.

Install pytest only if you want to run the tests:

```powershell
py -m pip install pytest
```

## Run the dashboard

From the repository root:

```powershell
py -m streamlit run app.py
```

Streamlit prints a local URL in the terminal; open that URL in your browser. Stop the dashboard with `Ctrl+C`.

### Check Complaint Priority

1. Choose one of the quick presets or enter a complaint directly.
2. Fill in the complaint type, description, ZIP code/location, and municipal agency.
3. Select **Run Priority Pipeline & Save Complaint**.
4. Review the priority score and category, safety risk, calculated feature values, work-order ID, and pipeline output.

The form sends those four complaint fields to `backend.pipeline.run_pipeline()`. The backend derives impact and hazard features from the complaint text when numeric feature values are not supplied. Those derived values are estimates, not live sensor readings. The pipeline stores the complaint and creates a single-complaint work order.

### Prioritized Work Orders

- View pending complaints and their descriptions, priority scores, and risk details.
- Select a complaint to see its row of details.
- Use **Mark selected complaint complete** to set its status to closed and record its completion date in SQLite. Linked work-order items are marked complete as well.
- View completed complaints in the completed list; they are no longer included in the pending dispatch queue.
- Set the batch size and select **Optimize & Generate Work Order** to schedule pending complaints. The dashboard can schedule up to 25 complaints per work order.

The genetic optimizer uses priority, safety risk, severity, traffic and public impact, pending days, weather risk, and recurrence. It does not optimize travel distance or repair duration.

### Load demonstration complaints

Use **Load 25 Sample Complaints** in the sidebar to load a sample from `data/processed/processed_complaints.csv`. If that file is not present, the app checks the repository-root `processed_complaints.csv`. The sample button requires one of these datasets to be available.

## Run the pipeline from the command line

From the repository root, pass complaint type followed by optional description, ZIP/location, and agency:

```powershell
py -m backend.pipeline "Pothole" "Large road hazard affecting traffic" "10001" "DOT"
```

The complaint type is required. Omitted optional arguments default to `Unknown`, `UNKNOWN`, and `USER`. The CLI prints the pipeline result as JSON and stores the complaint and work order in the SQLite database.

## Use the Python API

The pipeline can also be called directly from Python:

```python
from backend.pipeline import run_pipeline

result = run_pipeline({
    "unique_key": "complaint-123",
    "created_date": "2026-10-10T09:00:00",
    "complaint_type": "Pothole",
    "descriptor": "Large road hazard affecting traffic",
    "incident_zip": "10001",
    "agency": "DOT",
    "status": "Open",
    "days_pending": 1,
    "frequency_count": 1,
    "complaint_frequency": 3,
})

print(result)
```

The pipeline accepts optional derived features such as `severity`, `traffic_impact`, `public_impact`, `weather_risk`, `days_pending`, `complaint_frequency`, and `estimated_repair_time`. If impact values are omitted, feature engineering estimates them from the complaint details. If `unique_key` is omitted, the pipeline generates one. Saving a complaint with an existing key replaces its complaint record.

## Database

SQLite is initialized automatically when the app or pipeline first runs. The database file is `data/complaints.db`; its schema is in `data/schema.sql`.

- `complaints` stores complaint details, status, calculated features, safety risk, and priority.
- `work_orders` stores generated work-order records.
- `work_order_items` stores complaints in a work order, their queue position, and completion status.

## Run tests

From the repository root:

```powershell
py -m pytest tests -q
```

The test suite covers feature engineering, priority and safety-risk calculations, genetic work-order optimization, pipeline persistence, complaint completion, and database behavior.
