"""
Main Application Entrypoint for Intelligent Loan Risk Assessment & Underwriting Support System (LRAS).

Delivering:
- Landing Page
- Operational Executive Dashboard
- Comprehensive Interactive Risk Assessment & Prediction
- Holdout Model Evaluation & Diagnostic Insights
- Modeling Methodology & Responsible AI Governance
"""

from __future__ import annotations

import sys
from pathlib import Path
import streamlit as st

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pages.about import render_about_page
from pages.dashboard import render_dashboard_page
from pages.home import render_home_page
from pages.model_insights import render_model_insights_page
from pages.risk_assessment import render_risk_assessment_page
from utils.ui_helpers import load_custom_css, render_sidebar_header

# Streamlit Page Configuration (No emojis)
st.set_page_config(
    page_title="LRAS - Loan Risk Assessment",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply global styling and theme overrides
load_custom_css()

# Navigation Definitions - 5 Core Modules
PAGE_HOME = "Home"
PAGE_DASHBOARD = "Dashboard"
PAGE_RISK_ASSESSMENT = "Risk Assessment"
PAGE_MODEL_INSIGHTS = "Model Insights"
PAGE_REPORT_METHODOLOGY = "Report & Methodology"

PAGES = [
    PAGE_HOME,
    PAGE_DASHBOARD,
    PAGE_RISK_ASSESSMENT,
    PAGE_MODEL_INSIGHTS,
    PAGE_REPORT_METHODOLOGY,
]


def main():
    # Initialize session state for navigation
    if "nav_page" not in st.session_state or st.session_state["nav_page"] not in PAGES:
        st.session_state["nav_page"] = PAGE_HOME

    # If a page transition was requested (e.g. from in-page buttons), update radio key before widget creation
    if "pending_page" in st.session_state:
        pending = st.session_state.pop("pending_page")
        if pending in PAGES:
            st.session_state["navigation_radio"] = pending
        else:
            st.session_state["navigation_radio"] = PAGE_HOME
    elif "navigation_radio" not in st.session_state or st.session_state["navigation_radio"] not in PAGES:
        st.session_state["navigation_radio"] = st.session_state["nav_page"]

    # Sidebar Header
    render_sidebar_header()

    # Sidebar Navigation Menu (Clean SaaS text navigation without emojis)
    selected_page = st.sidebar.radio(
        "Navigation",
        options=PAGES,
        format_func=lambda p: p,
        key="navigation_radio",
        label_visibility="collapsed",
    )

    # Sync session state if user clicks sidebar radio
    st.session_state["nav_page"] = selected_page

    # Dispatch Page Routing
    current = st.session_state["nav_page"]

    if current == PAGE_HOME:
        render_home_page()
    elif current == PAGE_DASHBOARD:
        render_dashboard_page()
    elif current == PAGE_RISK_ASSESSMENT:
        render_risk_assessment_page()
    elif current == PAGE_MODEL_INSIGHTS:
        render_model_insights_page()
    elif current == PAGE_REPORT_METHODOLOGY:
        render_about_page()
    else:
        render_home_page()


if __name__ == "__main__":
    main()
