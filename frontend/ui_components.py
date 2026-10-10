"""
ui_components.py – Member 4: UI Design, Theming & Reusable Components
======================================================================
Provides modern styling, visual cards, charts, and responsive layouts
incorporating the project images:
  - logo.jpg
  - road_work.jpg
  - clean_city.jpg
  - swachh_bharat.jpg
"""

import base64
from pathlib import Path
from typing import Any, Dict, List, Optional
import streamlit as st
import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent


def get_image_as_base64(file_path: Path) -> str:
    """Read a local image and return its base64 string for inline HTML."""
    if not file_path.exists():
        return ""
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


def load_custom_css():
    """Inject modern municipal AI dashboard styles."""
    st.markdown(
        """
        <style>
        /* Import Google Font */
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        /* Hero Container */
        .hero-banner {
            background: linear-gradient(135deg, #09131e 0%, #0d2538 50%, #081e2b 100%);
            border: 1px solid rgba(0, 210, 255, 0.25);
            border-radius: 16px;
            padding: 24px 28px;
            margin-bottom: 24px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 20px;
        }

        .hero-text h1 {
            color: #ffffff;
            font-size: 2.1rem;
            font-weight: 800;
            margin: 0 0 8px 0;
            letter-spacing: -0.5px;
            background: linear-gradient(90deg, #38bdf8, #818cf8, #34d399);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .hero-text p {
            color: #94a3b8;
            font-size: 0.98rem;
            margin: 0;
            line-height: 1.5;
        }

        /* Glassmorphism Cards */
        .glass-card {
            background: rgba(15, 23, 42, 0.75);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 14px;
            padding: 20px;
            margin-bottom: 18px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
            transition: transform 0.2s ease, border-color 0.2s ease;
        }

        .glass-card:hover {
            border-color: rgba(56, 189, 248, 0.3);
            transform: translateY(-2px);
        }

        /* Priority Badges */
        .badge-critical {
            background: linear-gradient(135deg, #ef4444 0%, #b91c1c 100%);
            color: #ffffff;
            font-weight: 700;
            padding: 4px 12px;
            border-radius: 9999px;
            font-size: 0.82rem;
            letter-spacing: 0.5px;
            display: inline-block;
            box-shadow: 0 0 12px rgba(239, 68, 68, 0.45);
        }

        .badge-high {
            background: linear-gradient(135deg, #f97316 0%, #c2410c 100%);
            color: #ffffff;
            font-weight: 700;
            padding: 4px 12px;
            border-radius: 9999px;
            font-size: 0.82rem;
            letter-spacing: 0.5px;
            display: inline-block;
            box-shadow: 0 0 10px rgba(249, 115, 22, 0.35);
        }

        .badge-medium {
            background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
            color: #ffffff;
            font-weight: 700;
            padding: 4px 12px;
            border-radius: 9999px;
            font-size: 0.82rem;
            letter-spacing: 0.5px;
            display: inline-block;
            box-shadow: 0 0 10px rgba(59, 130, 246, 0.35);
        }

        .badge-low {
            background: linear-gradient(135deg, #10b981 0%, #047857 100%);
            color: #ffffff;
            font-weight: 700;
            padding: 4px 12px;
            border-radius: 9999px;
            font-size: 0.82rem;
            letter-spacing: 0.5px;
            display: inline-block;
            box-shadow: 0 0 10px rgba(16, 185, 129, 0.35);
        }

        /* Metric Display Blocks */
        .metric-block {
            background: rgba(30, 41, 59, 0.6);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 12px;
            padding: 16px;
            text-align: center;
        }

        .metric-title {
            color: #94a3b8;
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 6px;
        }

        .metric-value {
            color: #f8fafc;
            font-size: 1.8rem;
            font-weight: 800;
            line-height: 1;
        }

        /* Pipeline Badge */
        .pipeline-pill {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: rgba(14, 165, 233, 0.12);
            color: #38bdf8;
            border: 1px solid rgba(14, 165, 233, 0.3);
            border-radius: 20px;
            padding: 4px 10px;
            font-size: 0.75rem;
            font-weight: 600;
        }

        /* Mission Card */
        .mission-card {
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid rgba(255, 255, 255, 0.1);
            background: #0f172a;
            margin-bottom: 16px;
        }
        .mission-content {
            padding: 16px;
        }
        .mission-title {
            font-size: 1.05rem;
            font-weight: 700;
            color: #38bdf8;
            margin-bottom: 6px;
        }
        .mission-desc {
            font-size: 0.86rem;
            color: #cbd5e1;
            line-height: 1.45;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header():
    """Render top brand header with Municipal AI logo and project metadata."""
    logo_path = CURRENT_DIR / "logo.jpg"
    b64_logo = get_image_as_base64(logo_path) if logo_path.exists() else ""

    col_l, col_r = st.columns([1, 4])
    with col_l:
        if b64_logo:
            st.markdown(
                f"""
                <div style="text-align: center; padding: 6px;">
                    <img src="data:image/jpeg;base64,{b64_logo}"
                         style="max-width: 130px; border-radius: 50%; box-shadow: 0 0 25px rgba(56, 189, 248, 0.4); border: 2px solid #38bdf8;" />
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.write("🏛️ **Municipal AI**")

    with col_r:
        st.markdown(
            """
            <div style="padding-top: 10px;">
                <h1 style="margin: 0; font-size: 2.2rem; font-weight: 800; color: #f8fafc;">
                    Municipal Complaint <span style="color: #38bdf8;">Priority System</span>
                </h1>
                <p style="margin: 4px 0 0 0; color: #94a3b8; font-size: 1rem;">
                    AI-Based Priority Prediction & Work Order Scheduling
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("<hr style='border-color: rgba(255,255,255,0.08); margin: 16px 0 24px 0;'/>", unsafe_allow_html=True)


def render_priority_badge(category: str) -> str:
    """Return HTML string for styled category badge."""
    cat = str(category).upper()
    if cat == "CRITICAL":
        return '<span class="badge-critical">CRITICAL PRIORITY</span>'
    elif cat == "HIGH":
        return '<span class="badge-high">HIGH PRIORITY</span>'
    elif cat == "MEDIUM":
        return '<span class="badge-medium">MEDIUM PRIORITY</span>'
    else:
        return '<span class="badge-low">LOW PRIORITY</span>'


def render_complaint_result_card(result: Dict[str, Any]):
    """Display complaint evaluation results card."""
    safety_risk = float(result.get("safety_risk", 0.0))
    priority_score = float(result.get("priority_score", 0.0))
    category = result.get("priority_category", "LOW")
    badge_html = render_priority_badge(category)

    st.markdown(
        f"""
        <div class="glass-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
                <div>
                    <h3 style="margin: 0; color: #f8fafc; font-size: 1.3rem;">
                        📋 Complaint Evaluated: <span style="color: #38bdf8;">{result.get('unique_key')}</span>
                    </h3>
                    <p style="margin: 2px 0 0 0; color: #94a3b8; font-size: 0.9rem;">
                        Type: <strong>{result.get('complaint_type')}</strong> | Agency: <strong>{result.get('agency')}</strong> | Location/Zip: <strong>{result.get('incident_zip')}</strong>
                    </p>
                </div>
                <div>{badge_html}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(
            label="🛡️ Safety Hazard Risk",
            value=f"{safety_risk * 100:.1f}%",
            delta=f"Risk Score: {safety_risk:.3f}",
        )
    with c2:
        st.metric(
            label="⚡ AI Priority Score",
            value=f"{priority_score * 100:.1f}%",
            delta=f"Score: {priority_score:.3f}",
        )
    with c3:
        st.metric(
            label="🏷️ Priority Category",
            value=category,
        )
    with c4:
        st.metric(
            label="⏱️ Est. Crew Repair Time",
            value=f"{result.get('estimated_repair_time', 4.0)} hrs",
        )

    # Breakdown progress bars
    st.markdown("#### 🔬 Detailed Feature Attribution & Risk Signals")
    col_a, col_b = st.columns(2)
    with col_a:
        st.write(f"**Severity (0-10):** {result.get('severity', 0.0)}/10")
        st.progress(min(1.0, float(result.get("severity", 0.0)) / 10.0))

        st.write(f"**Traffic Impact (0-10):** {result.get('traffic_impact', 0.0)}/10")
        st.progress(min(1.0, float(result.get("traffic_impact", 0.0)) / 10.0))

        st.write(f"**Public Impact (0-10):** {result.get('public_impact', 0.0)}/10")
        st.progress(min(1.0, float(result.get("public_impact", 0.0)) / 10.0))

    with col_b:
        st.write(f"**Weather Risk (0-10):** {result.get('weather_risk', 0.0)}/10")
        st.progress(min(1.0, float(result.get("weather_risk", 0.0)) / 10.0))

        pending_days = float(result.get("days_pending", 0.0))
        st.write(f"**Days Pending:** {pending_days} days")
        st.progress(min(1.0, pending_days / 30.0))

        st.write(f"**Complaint Frequency (0-10):** {result.get('complaint_frequency', 0.0)}/10")
        st.progress(min(1.0, float(result.get("complaint_frequency", 0.0)) / 10.0))


def render_image_gallery():
    """Display the given municipal work images at the end of the page."""
    road_img = CURRENT_DIR / "road_work.jpg"
    clean_img = CURRENT_DIR / "clean_city.jpg"
    swachh_img = CURRENT_DIR / "swachh_bharat.jpg"

    st.markdown("---")
    st.markdown("### 📸 Municipal Work & Field Resolution")
    st.caption("Real-world municipal operations and field work prioritized by the system:")

    c1, c2, c3 = st.columns(3)
    with c1:
        if road_img.exists():
            st.image(
                str(road_img),
                use_container_width=True,
                caption="Road Infrastructure & Pothole Repair",
            )
    with c2:
        if swachh_img.exists():
            st.image(
                str(swachh_img),
                use_container_width=True,
                caption="Sanitation & Clean City Mission",
            )
    with c3:
        if clean_img.exists():
            st.image(
                str(clean_img),
                use_container_width=True,
                caption="Urban Maintenance & Public Wellbeing",
            )
