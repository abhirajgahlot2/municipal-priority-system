PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS complaints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    unique_key TEXT NOT NULL UNIQUE,
    created_date TEXT,
    closed_date TEXT,
    agency TEXT,
    complaint_type TEXT,
    descriptor TEXT,
    incident_zip TEXT,
    status TEXT,
    severity REAL,
    traffic_impact REAL,
    public_impact REAL,
    weather_risk REAL,
    days_pending REAL,
    frequency_count INTEGER,
    complaint_frequency REAL,
    safety_risk REAL,
    priority_score REAL,
    priority_category TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS work_orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    work_order_id TEXT NOT NULL UNIQUE,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    status TEXT DEFAULT 'pending',
    notes TEXT
);

CREATE TABLE IF NOT EXISTS work_order_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    work_order_id TEXT NOT NULL,
    complaint_unique_key TEXT NOT NULL,
    queue_position INTEGER NOT NULL,
    priority_score REAL,
    safety_risk REAL,
    status TEXT DEFAULT 'pending',
    FOREIGN KEY (work_order_id) REFERENCES work_orders(work_order_id),
    FOREIGN KEY (complaint_unique_key) REFERENCES complaints(unique_key)
);

CREATE INDEX IF NOT EXISTS idx_complaints_type_zip ON complaints(complaint_type, incident_zip);
CREATE INDEX IF NOT EXISTS idx_complaints_status ON complaints(status);
CREATE INDEX IF NOT EXISTS idx_work_order_items_order ON work_order_items(work_order_id, queue_position);
