"""
app.py – Municipal Complaint Priority System
============================================
AI-Based Priority Prediction & Work Order Scheduling
"""

import sys
from pathlib import Path
import pandas as pd
import streamlit as st

# Setup Path so root modules are importable
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frontend.integration import (
    process_complaint,
    run_batch_work_order_optimization,
    get_all_complaints_df,
    get_all_work_orders_summary,
    seed_sample_complaints,
)
from frontend.ui_components import (
    load_custom_css,
    render_header,
    render_complaint_result_card,
    render_image_gallery,
)

# Page Configuration
st.set_page_config(
    page_title="Municipal Complaint Priority System",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply Clean Custom Styling
load_custom_css()

# Render Top Header
render_header()

# Quick Test Presets
PRESETS = {
    "🚨 Critical Pothole": {
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
    "🌊 Water Main Break": {
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
    "🗑️ Garbage Overflow": {
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
    "💡 Streetlight Out": {
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
    "🚗 Blocked Driveway": {
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
    st.session_state["selected_preset"] = PRESETS["🚨 Critical Pothole"]

if "evaluated_result" not in st.session_state:
    st.session_state["evaluated_result"] = None

if "last_work_order" not in st.session_state:
    st.session_state["last_work_order"] = None

# ============================================================
# SIDEBAR: QUICK CONTROLS
# ============================================================
with st.sidebar:
    st.markdown("### 🏛️ Municipal System")
    st.markdown("Automated complaint triage, safety risk assessment, and crew dispatch optimization.")

    st.markdown("---")
    st.markdown("### 📥 Demo Records")
    if st.button("Load 25 Sample Complaints", help="Quickly populates the database with real records", use_container_width=True):
        with st.spinner("Loading sample complaints..."):
            count = seed_sample_complaints(25)
            st.success(f"Loaded {count} sample complaints into database!")
            st.rerun()

    st.markdown("---")
    st.caption("AI Priority Prediction & Work Order System • 2026")


# ============================================================
# MAIN APPLICATION TABS
# ============================================================
tab_priority, tab_dispatch = st.tabs([
    "📝 Check Complaint Priority",
    "📋 Prioritized Work Orders",
])

# ------------------------------------------------------------
# TAB 1: FILE & PRIORITIZE COMPLAINT
# ------------------------------------------------------------
with tab_priority:
    st.markdown("### 1. Register or Select a Complaint")

    # One-click preset buttons for immediate testing
    st.markdown("##### ⚡ Quick Test Presets:")
    p_cols = st.columns(5)
    for idx, (p_name, p_data) in enumerate(PRESETS.items()):
        with p_cols[idx]:
            if st.button(p_name, use_container_width=True):
                st.session_state["selected_preset"] = p_data
                st.rerun()

    preset = st.session_state["selected_preset"]

    with st.form("complaint_form", clear_on_submit=False):
        c1, c2 = st.columns(2)

        with c1:
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

            estimated_repair_time = st.number_input(
                "Estimated Repair Time (hours)",
                min_value=0.5,
                max_value=72.0,
                value=float(preset["time"]),
                step=0.5,
            )

        with c2:
            st.markdown("#### Impact & Hazard Factors (0 – 10)")
            severity = st.slider("Severity / Physical Hazard", 0.0, 10.0, float(preset["sev"]), 0.5)
            traffic_impact = st.slider("Traffic Congestion Impact", 0.0, 10.0, float(preset["traf"]), 0.5)
            public_impact = st.slider("Public Population Affected", 0.0, 10.0, float(preset["pub"]), 0.5)
            weather_risk = st.slider("Weather Risk (Rain/Snow/Freeze)", 0.0, 10.0, float(preset["weath"]), 0.5)

            sub_c, sub_d = st.columns(2)
            with sub_c:
                days_pending = st.number_input("Days Pending", min_value=0, max_value=60, value=int(preset["days"]))
            with sub_d:
                complaint_frequency = st.slider("Recurrence / Frequency", 0.0, 10.0, float(preset["freq"]), 0.5)

        submit_btn = st.form_submit_button(
            "⚡ Calculate Priority & Save Complaint",
            type="primary",
            use_container_width=True,
        )

    if submit_btn:
        with st.spinner("Calculating safety risk and priority category..."):
            complaint_dict = {
                "complaint_type": complaint_type,
                "descriptor": descriptor,
                "incident_zip": incident_zip,
                "agency": agency,
                "estimated_repair_time": estimated_repair_time,
                "severity": severity,
                "traffic_impact": traffic_impact,
                "traffic_level": traffic_impact,
                "public_impact": public_impact,
                "people_affected": public_impact,
                "weather_risk": weather_risk,
                "days_pending": days_pending,
                "complaint_frequency": complaint_frequency,
                "status": "Open",
            }
            res = process_complaint(complaint_dict, auto_save=True)
            st.session_state["evaluated_result"] = res
            st.success("✅ Complaint evaluated and saved to database!")

    # Display Result Card
    if st.session_state.get("evaluated_result"):
        render_complaint_result_card(st.session_state["evaluated_result"])


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
        open_df = df[df["status"].astype(str).str.lower() != "closed"]
        if open_df.empty:
            open_df = df

        st.markdown(f"**Pending Complaints in System:** {len(open_df)}")

        display_cols = [
            "unique_key",
            "complaint_type",
            "agency",
            "severity",
            "traffic_impact",
            "safety_risk",
            "priority_score",
            "priority_category",
        ]
        avail = [c for c in display_cols if c in open_df.columns]
        st.dataframe(open_df[avail].head(15), use_container_width=True, height=200)

        total_open = len(open_df)
        col_opt1, col_opt2 = st.columns([1, 1])
        with col_opt1:
            if total_open <= 1:
                batch_count = total_open
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
            run_opt_btn = st.button("🚀 Optimize & Generate Work Order", type="primary", use_container_width=True)

        if run_opt_btn:
            if total_open == 0 or batch_count == 0:
                st.warning("No pending complaints available to optimize.")
            else:
                with st.spinner("Optimizing dispatch order by urgency and safety risk..."):
                    batch_records = open_df.sort_values(by="priority_score", ascending=False).head(batch_count).to_dict("records")
                    opt_result = run_batch_work_order_optimization(batch_records, save_to_db=True)
                    st.session_state["last_work_order"] = opt_result
                    st.success(f"✅ Work Order Generated: **{opt_result['work_order_id']}** (Improvement: {opt_result['improvement_pct']}%)")

        current_wo = st.session_state.get("last_work_order")
        if current_wo and current_wo.get("ordered_complaints"):
            st.markdown(f"#### 📋 Scheduled Dispatch Queue: `{current_wo['work_order_id']}`")
            if current_wo.get("improvement_pct", 0) > 0:
                st.caption(f"🚀 Genetic Algorithm optimization improvement over greedy baseline: **{current_wo['improvement_pct']}%**")
            table_data = []
            for idx, c in enumerate(current_wo["ordered_complaints"], start=1):
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

            st.dataframe(pd.DataFrame(table_data), use_container_width=True)

            if current_wo.get("notes"):
                with st.expander("ℹ️ Optimizer Notes & Constraints"):
                    for note in current_wo["notes"]:
                        st.write(f"- {note}")

        saved_orders = get_all_work_orders_summary()
        if saved_orders:
            with st.expander(f"📁 Past Generated Work Orders in Database ({len(saved_orders)})"):
                orders_df = pd.DataFrame(saved_orders)
                st.dataframe(orders_df, use_container_width=True)


# ============================================================
# GIVEN IMAGES AT THE END TO SHOW WORK
# ============================================================
render_image_gallery()
