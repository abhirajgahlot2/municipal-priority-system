"""
app.py – Municipal Complaint Priority System
============================================
AI-Based Priority Prediction & Work Order Scheduling
"""

import sqlite3
import sys
from pathlib import Path
import pandas as pd
import streamlit as st

# Setup Path so root modules are importable
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frontend.integration import (
    run_batch_work_order_optimization,
    get_all_complaints_df,
    mark_complaint_completed,
    seed_sample_complaints,
)
from backend.pipeline import run_pipeline
from frontend.ui_components import (
    load_custom_css,
    render_header,
    render_complaint_result_card,
    render_image_gallery,
)

# Page Configuration
st.set_page_config(
    page_title="Municipal Complaint Priority System",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply Clean Custom Styling
load_custom_css()

# Render Top Header
render_header()

# Quick Test Presets
PRESETS = {
    "Critical Pothole": {
        "type": "Pothole",
        "desc": "Deep cave-in crater on arterial road causing severe vehicle swerving",
        "zip": "10001",
        "agency": "DOT",
        "sev": 9.0,
        "traf": 9.5,
        "pub": 8.5,
        "weath": 7.0,
        "days": 4,
        "freq": 6.0,
        "time": 4.0,
    },
    "Water Main Break": {
        "type": "Water Main Break",
        "desc": "High pressure pipe burst flooding roadway and residential basements",
        "zip": "10025",
        "agency": "DEP",
        "sev": 9.5,
        "traf": 8.0,
        "pub": 9.0,
        "weath": 5.0,
        "days": 2,
        "freq": 4.0,
        "time": 8.0,
    },
    "Garbage Overflow": {
        "type": "Sanitation Condition",
        "desc": "Accumulated garbage blocking public sidewalk attracting pests",
        "zip": "11201",
        "agency": "DSNY",
        "sev": 6.0,
        "traf": 3.0,
        "pub": 7.0,
        "weath": 4.0,
        "days": 8,
        "freq": 8.0,
        "time": 3.0,
    },
    "Streetlight Out": {
        "type": "Damaged Streetlight",
        "desc": "Main street pole light not functioning creating dark intersection",
        "zip": "10451",
        "agency": "DOT",
        "sev": 4.5,
        "traf": 6.0,
        "pub": 5.0,
        "weath": 2.0,
        "days": 5,
        "freq": 2.0,
        "time": 2.5,
    },
    "Blocked Driveway": {
        "type": "Blocked Driveway",
        "desc": "Unauthorized vehicle parked across private driveway entrance",
        "zip": "11375",
        "agency": "NYPD",
        "sev": 3.0,
        "traf": 2.0,
        "pub": 2.0,
        "weath": 1.0,
        "days": 1,
        "freq": 1.0,
        "time": 1.0,
    },
}

if "selected_preset" not in st.session_state:
    st.session_state["selected_preset"] = PRESETS["Critical Pothole"]

if "evaluated_result" not in st.session_state:
    st.session_state["evaluated_result"] = None

if "last_work_order" not in st.session_state:
    st.session_state["last_work_order"] = None

# ============================================================
# SIDEBAR: QUICK CONTROLS
# ============================================================
with st.sidebar:
    st.markdown("### Municipal System")
    st.markdown("Automated complaint triage, safety risk assessment, and crew dispatch optimization.")

    st.markdown("---")
    st.markdown("### Demo Records")
    if st.button("Load 25 Sample Complaints", help="Quickly populates the database with real records", width="stretch"):
        with st.spinner("Loading sample complaints..."):
            count = seed_sample_complaints(25)
            if count:
                st.success(f"Loaded {count} sample complaints into database!")
                st.rerun()
            else:
                st.warning("Sample complaints could not be loaded: the dataset file was not found or is empty.")

    st.markdown("---")
    st.caption("AI Priority Prediction & Work Order System • 2026")


# ============================================================
# MAIN APPLICATION TABS
# ============================================================
tab_priority, tab_dispatch = st.tabs([
    "Check Complaint Priority",
    "Prioritized Work Orders",
])

# ------------------------------------------------------------
# TAB 1: FILE & PRIORITIZE COMPLAINT
# ------------------------------------------------------------
with tab_priority:
    st.markdown("### 1. Register or Select a Complaint")

    # One-click preset buttons for immediate testing
    st.markdown("##### Quick Test Presets:")
    p_cols = st.columns(5)
    for idx, (p_name, p_data) in enumerate(PRESETS.items()):
        with p_cols[idx]:
            if st.button(p_name, width="stretch"):
                st.session_state["selected_preset"] = p_data
                st.rerun()

    preset = st.session_state["selected_preset"]

    with st.form("complaint_form", clear_on_submit=False):
        st.markdown("#### Complaint Details")
        comp_types = [
            "Pothole",
            "Water Main Break",
            "Sanitation Condition",
            "Damaged Streetlight",
            "Blocked Driveway",
            "Illegal Parking",
            "Noise - Residential",
            "Heat/Hot Water",
            "Drainage Overflow",
            "Hazardous Tree Branch",
            "Other Municipal Issue",
        ]
        type_idx = comp_types.index(preset["type"]) if preset["type"] in comp_types else 0
        complaint_type = st.selectbox("Complaint Type", comp_types, index=type_idx)

        descriptor = st.text_area("Description", value=preset["desc"], height=80)

        sub_a, sub_b = st.columns(2)
        with sub_a:
            incident_zip = st.text_input("Zip Code / Location", value=preset["zip"])
        with sub_b:
            agencies = ["DOT", "DEP", "DSNY", "DOHMH", "NYPD", "Parks", "HPD"]
            agency_idx = agencies.index(preset["agency"]) if preset["agency"] in agencies else 0
            agency = st.selectbox("Municipal Agency", agencies, index=agency_idx)

        submit_btn = st.form_submit_button(
            "Run Priority Pipeline & Save Complaint",
            type="primary",
            width="stretch",
        )

    if submit_btn:
        st.session_state["evaluated_result"] = None
        st.session_state["pipeline_exchange"] = None
        with st.spinner("Running priority, safety-risk, and work-order pipeline..."):
            complaint_dict = {
                "complaint_type": complaint_type,
                "descriptor": descriptor,
                "incident_zip": incident_zip,
                "agency": agency,
            }
            try:
                pipeline_output = run_pipeline(complaint_dict)
            except (ValueError, sqlite3.Error) as error:
                st.error(f"Complaint pipeline could not complete: {error}")
            else:
                st.session_state["evaluated_result"] = {
                    **complaint_dict,
                    **pipeline_output["features"],
                    "unique_key": pipeline_output["complaint_id"],
                    "work_order_id": pipeline_output["work_order_id"],
                }
                st.session_state["pipeline_exchange"] = {
                    "input": complaint_dict,
                    "output": pipeline_output,
                }
                st.success("Complaint evaluated, optimized, and saved to the database.")

    # Display Result Card
    if st.session_state.get("evaluated_result"):
        render_complaint_result_card(st.session_state["evaluated_result"])
        pipeline_exchange = st.session_state.get("pipeline_exchange")
        if pipeline_exchange:
            st.caption(
                f"Generated work order: **{pipeline_exchange['output']['work_order_id']}** · "
                f"Optimizer fitness: **{pipeline_exchange['output']['fitness']:.4f}**"
            )
            with st.expander("View pipeline output"):
                st.markdown("#### Pipeline Output")
                output = pipeline_exchange["output"]
                output_rows = [
                    {"Result": "Complaint ID", "Value": str(output["complaint_id"])},
                    {"Result": "Work order ID", "Value": str(output["work_order_id"])},
                    {"Result": "Optimized order", "Value": ", ".join(map(str, output["optimized_order"]))},
                    {"Result": "Optimizer fitness", "Value": f"{output['fitness']:.4f}"},
                    {"Result": "Baseline fitness", "Value": f"{output['baseline_fitness']:.4f}"},
                    {"Result": "Saved to database", "Value": "Yes" if output["saved_to_db"] else "No"},
                ]
                output_rows.extend(
                    {"Result": name.replace("_", " ").title(), "Value": str(value)}
                    for name, value in output["features"].items()
                )
                st.table(pd.DataFrame(output_rows))
                if output["notes"]:
                    st.markdown("**Pipeline notes**")
                    for note in output["notes"]:
                        st.write(note)


# ------------------------------------------------------------
# TAB 2: PRIORITIZED WORK ORDERS
# ------------------------------------------------------------
with tab_dispatch:
    st.markdown("### 2. Prioritized Field Work Orders")
    st.markdown(
        "Orders active municipal complaints so repair crews can address the most critical hazards first."
    )

    all_complaints = get_all_complaints_df()

    if not all_complaints:
        st.info("No complaints found in database. Load sample complaints from the sidebar to test.")
    else:
        df = pd.DataFrame(all_complaints)
        closed_df = df[df["status"].astype(str).str.lower() == "closed"]
        open_df = df[df["status"].astype(str).str.lower() != "closed"]
        if open_df.empty:
            st.info("All recorded complaints are closed; no pending complaints are available to schedule.")

        st.markdown(f"**Pending Complaints in System:** {len(open_df)}")

        display_cols = [
            "unique_key",
            "complaint_type",
            "descriptor",
            "agency",
            "severity",
            "traffic_impact",
            "safety_risk",
            "priority_score",
            "priority_category",
        ]
        avail = [c for c in display_cols if c in open_df.columns]
        st.dataframe(open_df[avail].head(15), width="stretch", height=200)

        if not open_df.empty:
            complaint_records = open_df.to_dict("records")
            complaint_ids = [str(record["unique_key"]) for record in complaint_records]
            complaint_by_id = {
                str(record["unique_key"]): record for record in complaint_records
            }
            selected_complaint_id = st.selectbox(
                "Complaint details",
                complaint_ids,
                format_func=lambda key: (
                    f"{complaint_by_id[key].get('complaint_type', 'Complaint')} "
                    f"— {key}"
                ),
            )
            selected_complaint = complaint_by_id[selected_complaint_id]
            st.markdown(f"#### Selected complaint: {selected_complaint_id}")
            st.dataframe(
                pd.DataFrame([selected_complaint])[avail],
                width="stretch",
                hide_index=True,
            )
            if st.button(
                "Mark selected complaint complete",
                type="primary",
                key=f"complete_{selected_complaint_id}",
            ):
                if mark_complaint_completed(selected_complaint_id):
                    st.success("Complaint marked complete and saved to the database.")
                    st.rerun()
                else:
                    st.error("The complaint could not be found or was already completed.")

        if not closed_df.empty:
            with st.expander(f"Completed complaints ({len(closed_df)})"):
                completed_columns = [
                    "unique_key",
                    "complaint_type",
                    "agency",
                    "incident_zip",
                    "closed_date",
                    "priority_category",
                ]
                completed_available = [
                    column for column in completed_columns if column in closed_df.columns
                ]
                st.dataframe(
                    closed_df[completed_available],
                    width="stretch",
                    hide_index=True,
                )

        total_open = len(open_df)
        col_opt1, col_opt2 = st.columns([1, 1])
        with col_opt1:
            if total_open <= 1:
                batch_count = total_open
                if total_open:
                    st.info(f"Scheduling {batch_count} pending complaint(s) in this work order.")
            else:
                max_batch = min(total_open, 25)
                min_batch = 2 if max_batch > 2 else 1
                default_val = min(8, max_batch)
                batch_count = st.slider(
                    "Number of complaints to schedule in this work order:",
                    min_value=min_batch,
                    max_value=max_batch,
                    value=default_val,
                )
        with col_opt2:
            st.write("")
            st.write("")
            run_opt_btn = st.button("Optimize & Generate Work Order", type="primary", width="stretch")

        if run_opt_btn:
            if total_open == 0 or batch_count == 0:
                st.warning("No pending complaints available to optimize.")
            else:
                with st.spinner("Optimizing dispatch order by urgency and safety risk..."):
                    batch_records = open_df.sort_values(by="priority_score", ascending=False).head(batch_count).to_dict("records")
                    opt_result = run_batch_work_order_optimization(batch_records, save_to_db=True)
                    st.session_state["last_work_order"] = opt_result
                    st.success(f"Work Order Generated: **{opt_result['work_order_id']}** (Improvement: {opt_result['improvement_pct']}%)")

        current_wo = st.session_state.get("last_work_order")
        active_complaint_ids = set(open_df["unique_key"].astype(str))
        active_ordered_complaints = (
            [
                complaint
                for complaint in current_wo.get("ordered_complaints", [])
                if str(complaint.get("unique_key")) in active_complaint_ids
            ]
            if current_wo
            else []
        )
        if current_wo and active_ordered_complaints:
            st.markdown(f"#### Scheduled Dispatch Queue: `{current_wo['work_order_id']}`")
            if current_wo.get("improvement_pct", 0) > 0:
                st.caption(f"Genetic Algorithm optimization improvement over greedy baseline: **{current_wo['improvement_pct']}%**")
            table_data = []
            for idx, c in enumerate(active_ordered_complaints, start=1):
                table_data.append({
                    "Dispatch Order": f"#{idx}",
                    "Complaint ID": c.get("unique_key"),
                    "Complaint Type": c.get("complaint_type"),
                    "Location / Zip": c.get("incident_zip"),
                    "Priority Category": c.get("priority_category"),
                    "Priority Score": f"{float(c.get('priority_score', 0.0)):.4f}",
                    "Safety Risk": f"{float(c.get('safety_risk', 0.0)):.4f}",
                    "Severity": c.get("severity"),
                })

            st.dataframe(pd.DataFrame(table_data), width="stretch")

            if current_wo.get("notes"):
                with st.expander("Optimizer Notes & Constraints"):
                    for note in current_wo["notes"]:
                        st.write(f"- {note}")

# ============================================================
# GIVEN IMAGES AT THE END TO SHOW WORK
# ============================================================
render_image_gallery()
