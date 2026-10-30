"""
UI Helpers and Reusable Styling Utilities for LRAS Streamlit Application.

Provides:
- Global typography and layout rules (Warm off-white #F7F5EE, dark forest olive #1F2E20, refined olive #3F5F3E, terracotta #D97732)
- Cached data loading functions
- Standardized metric cards, badges, and factor display components
- Clean SVG vector icons (Zero emojis)
- Indentation-safe HTML rendering
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "loan_risk_processed.csv"
METRICS_PATH = PROJECT_ROOT / "results" / "model_metrics.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
ASSETS_DIR = PROJECT_ROOT / "assets"


def clean_html(html_str: str) -> str:
    """
    Strip leading and trailing whitespace from every line of an HTML string.
    Ensures that CommonMark / Markdown parser NEVER interprets 4+ spaces of indentation as a code block.
    """
    return "\n".join(line.strip() for line in html_str.strip().splitlines())


def load_custom_css():
    """Inject global CSS rules to style the application according to the refined modern design."""
    css = """
    <style>
    /* Global Typography & Background */
    html, body, [class*="css"], .stApp {
        background-color: #F7F5EE;
        color: #202820;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        font-size: 15px;
        line-height: 1.55;
    }

    /* Page Heading & Subtitle Spacing */
    .page-header-container {
        margin-bottom: 24px;
    }
    .page-title {
        font-size: 32px;
        font-weight: 800;
        color: #202820;
        line-height: 1.2;
        letter-spacing: -0.5px;
        margin: 0 0 6px 0;
    }
    .page-subtitle {
        font-size: 14.5px;
        color: #687067;
        line-height: 1.5;
        margin: 0;
    }

    /* Sidebar Styling: Deep Dark Forest Olive */
    [data-testid="stSidebar"] {
        background-color: #1F2E20 !important;
        color: #E3E8E3 !important;
        border-right: 1px solid #2A3C2B;
    }
    [data-testid="stSidebar"] * {
        color: #E3E8E3 !important;
    }
    
    /* Modern Clean Sidebar Navigation */
    [data-testid="stSidebar"] [role="radiogroup"] {
        gap: 4px;
    }
    [data-testid="stSidebar"] [role="radiogroup"] > label {
        color: #C8D3C8 !important;
        font-weight: 500;
        font-size: 14px;
        padding: 10px 14px;
        border-radius: 8px;
        transition: all 0.15s ease-in-out;
        cursor: pointer;
        margin-bottom: 2px;
        display: flex;
        align-items: center;
    }
    /* Hide radio circle indicator for a polished SaaS navigation look */
    [data-testid="stSidebar"] [role="radiogroup"] > label > div:first-child {
        display: none !important;
    }
    [data-testid="stSidebar"] [role="radiogroup"] > label:hover {
        background-color: #2A3E2B !important;
        color: #FFFFFF !important;
    }
    /* Active nav item */
    [data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) {
        background-color: #3F5F3E !important;
        color: #FFFFFF !important;
        font-weight: 600;
    }

    /* Top header padding reduction */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2.5rem;
        max-width: 1200px;
    }

    /* Standard Cards */
    .lras-card {
        background-color: #FFFFFF;
        border: 1px solid #E3E5DD;
        border-radius: 14px;
        padding: 22px 24px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.025);
        margin-bottom: 20px;
    }

    .lras-card-header {
        font-size: 16px;
        font-weight: 700;
        color: #202820;
        margin-bottom: 14px;
        display: flex;
        align-items: center;
        gap: 10px;
        letter-spacing: -0.2px;
    }

    /* Metric Cards */
    .metric-card {
        background-color: #FFFFFF;
        border: 1px solid #E3E5DD;
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.02);
        display: flex;
        flex-direction: column;
        justify-content: center;
        min-height: 112px;
    }
    .metric-title {
        font-size: 12.5px;
        font-weight: 600;
        color: #687067;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 800;
        color: #202820;
        line-height: 1.15;
        letter-spacing: -0.5px;
    }
    .metric-sub {
        font-size: 12.5px;
        color: #687067;
        margin-top: 5px;
    }

    /* Risk Badges (Monochrome / Subdued FinTech Style) */
    .badge-risk-low {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background-color: #EEF4EE;
        color: #2B572E;
        border: 1px solid #C5DCC6;
        padding: 5px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 13.5px;
        letter-spacing: 0.2px;
    }
    .badge-risk-medium {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background-color: #FAF2EB;
        color: #B45309;
        border: 1px solid #F5D3B3;
        padding: 5px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 13.5px;
        letter-spacing: 0.2px;
    }
    .badge-risk-high {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background-color: #FDF2F2;
        color: #991B1B;
        border: 1px solid #FCA5A5;
        padding: 5px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 13.5px;
        letter-spacing: 0.2px;
    }

    /* Affordability Badges */
    .badge-afford-good {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background-color: #EEF4EE;
        color: #2B572E;
        border: 1px solid #C5DCC6;
        padding: 4px 12px;
        border-radius: 16px;
        font-weight: 600;
        font-size: 13px;
    }
    .badge-afford-moderate {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background-color: #FAF2EB;
        color: #C25E1A;
        border: 1px solid #F5D3B3;
        padding: 4px 12px;
        border-radius: 16px;
        font-weight: 600;
        font-size: 13px;
    }
    .badge-afford-strained {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background-color: #FDF2F2;
        color: #991B1B;
        border: 1px solid #FCA5A5;
        padding: 4px 12px;
        border-radius: 16px;
        font-weight: 600;
        font-size: 13px;
    }

    /* Factor Items */
    .factor-item-risk {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 9px 13px;
        margin-bottom: 6px;
        background-color: #FDF4F4;
        border-left: 3px solid #DC2626;
        border-radius: 6px;
        font-size: 13.5px;
        color: #202820;
    }
    .factor-item-protective {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 9px 13px;
        margin-bottom: 6px;
        background-color: #F2F7F2;
        border-left: 3px solid #3F5F3E;
        border-radius: 6px;
        font-size: 13.5px;
        color: #202820;
    }

    /* Primary Button Override */
    div.stButton > button:first-child {
        background-color: #3F5F3E;
        color: #FFFFFF !important;
        border: 1px solid #3F5F3E;
        border-radius: 8px;
        padding: 9px 24px;
        font-weight: 600;
        font-size: 14.5px;
        box-shadow: 0 1px 3px rgba(63, 95, 62, 0.2);
        transition: all 0.15s ease;
    }
    div.stButton > button:first-child:hover {
        background-color: #324D31;
        border-color: #324D31;
        color: #FFFFFF !important;
        box-shadow: 0 3px 8px rgba(63, 95, 62, 0.25);
    }
    div.stButton > button:first-child:active {
        background-color: #283D27;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid #E3E5DD;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 18px;
        font-weight: 600;
        color: #687067;
        border-radius: 6px 6px 0 0;
        font-size: 14px;
    }
    .stTabs [aria-selected="true"] {
        color: #3F5F3E !important;
        border-bottom: 2px solid #3F5F3E !important;
    }

    /* Streamlit DataFrame Font Styling */
    [data-testid="stDataFrame"] {
        font-size: 13.5px;
    }

    /* Form Label Styling */
    [data-testid="stWidgetLabel"] p {
        font-size: 14px !important;
        font-weight: 500 !important;
        color: #202820 !important;
    }

    </style>
    """
    st.markdown(clean_html(css), unsafe_allow_html=True)


@st.cache_data
def load_dataset() -> pd.DataFrame:
    """Load the processed loan risk dataset with caching."""
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_model_metrics() -> pd.DataFrame:
    """Load the evaluated model metrics CSV with caching."""
    df = pd.read_csv(METRICS_PATH)
    if "Unnamed: 0" in df.columns:
        df = df.rename(columns={"Unnamed: 0": "Model"})
    elif df.columns[0] == "":
        df = df.rename(columns={df.columns[0]: "Model"})
    return df


def render_page_header(title: str, subtitle: str):
    """Render consistent page title and subtitle across all 7 pages."""
    header_html = f"""
    <div class="page-header-container">
        <h1 class="page-title">{title}</h1>
        <p class="page-subtitle">{subtitle}</p>
    </div>
    """
    st.markdown(clean_html(header_html), unsafe_allow_html=True)


def render_sidebar_header():
    """Render the top logo and title in the sidebar with monochrome SVG vector."""
    sidebar_html = """
    <div style="display: flex; align-items: center; gap: 12px; padding: 6px 0 20px 0; border-bottom: 1px solid #2B3D2C; margin-bottom: 16px;">
        <div style="background-color: #3F5F3E; color: #FFFFFF; border-radius: 9px; width: 38px; height: 38px; display: flex; align-items: center; justify-content: center; flex-shrink: 0;">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#FFFFFF" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
            </svg>
        </div>
        <div>
            <div style="font-size: 16px; font-weight: 800; color: #FFFFFF; letter-spacing: 0.2px; line-height: 1.15;">
                Loan Risk
            </div>
            <div style="font-size: 12.5px; font-weight: 500; color: #A8B8A9; letter-spacing: 0.3px;">
                Assessment
            </div>
        </div>
    </div>
    """
    st.sidebar.markdown(clean_html(sidebar_html), unsafe_allow_html=True)


def render_metric_card(
    title: str,
    value: str,
    subtitle: str = "",
    icon_svg: str = "",
) -> str:
    """Return HTML string for a clean rounded metric card with optional SVG vector."""
    icon_html = (
        f'<div style="opacity: 0.7; display: flex; align-items: center;">{icon_svg}</div>'
        if icon_svg
        else ""
    )
    card_html = f"""
    <div class="metric-card">
        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
            <div class="metric-title">{title}</div>
            {icon_html}
        </div>
        <div class="metric-value">{value}</div>
        {f'<div class="metric-sub">{subtitle}</div>' if subtitle else ''}
    </div>
    """
    return clean_html(card_html)


def render_risk_badge_html(risk_level: str) -> str:
    """Return clean HTML badge for risk tier without emojis."""
    lvl = risk_level.upper()
    if "LOW" in lvl:
        return '<span class="badge-risk-low">Low Risk</span>'
    elif "MEDIUM" in lvl or "MODERATE" in lvl:
        return '<span class="badge-risk-medium">Medium Risk</span>'
    else:
        return '<span class="badge-risk-high">High Risk</span>'


def render_affordability_badge_html(status: str) -> str:
    """Return clean HTML badge for affordability status without emojis."""
    st_upper = status.upper()
    if "FAVORABLE" in st_upper or "GOOD" in st_upper:
        return '<span class="badge-afford-good">Favorable</span>'
    elif "MODERATE" in st_upper:
        return '<span class="badge-afford-moderate">Moderate Strain</span>'
    else:
        return '<span class="badge-afford-strained">High Burden</span>'


def render_prototype_disclaimer():
    """No-op: Disclaimer removed."""
    pass
