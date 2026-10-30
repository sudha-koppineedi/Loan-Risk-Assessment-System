"""
Home / Landing Page for LRAS Streamlit Application.
"""

from __future__ import annotations

from pathlib import Path
import streamlit as st
from utils.ui_helpers import clean_html

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HERO_SVG = PROJECT_ROOT / "assets" / "hero_illustration.svg"


def render_home_page():
    # Hero Section
    hero_col1, hero_col2 = st.columns([1.1, 0.9], gap="large")

    with hero_col1:
        hero_text_html = """
        <div style="padding-top: 10px;">
            <div style="font-size: 13px; font-weight: 700; color: #3F5F3E; text-transform: uppercase; letter-spacing: 1.2px; margin-bottom: 12px;">
                UNDERWRITING SUPPORT &amp; RISK ASSESSMENT
            </div>
            <h1 style="font-size: 33px; font-weight: 800; color: #202820; line-height: 1.2; margin-bottom: 18px; letter-spacing: -0.5px;">
                Smarter Loan Decisions<br>
                for a <span style="color: #D97732;">Safer Tomorrow</span>
            </h1>
            <p style="font-size: 15px; color: #687067; line-height: 1.65; margin-bottom: 28px;">
                An intelligent system that uses machine learning and comprehensive financial analysis 
                to assess loan default risk and provide objective, actionable underwriting insights.
            </p>
        </div>
        """
        st.markdown(clean_html(hero_text_html), unsafe_allow_html=True)

        btn_col1, btn_col2, _ = st.columns([1.1, 1.1, 1])
        with btn_col1:
            if st.button("Get Started →", use_container_width=True):
                st.session_state["pending_page"] = "Risk Assessment"
                st.rerun()
        with btn_col2:
            if st.button("Learn More", use_container_width=True):
                st.session_state["pending_page"] = "Report & Methodology"
                st.rerun()

    with hero_col2:
        # High-resolution vector illustration loaded cleanly as an image asset
        if HERO_SVG.exists():
            st.image(str(HERO_SVG), use_container_width=True)

    st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)

    # Core Capabilities Section (Clean 3-column layout, zero emojis)
    c1, c2, c3 = st.columns(3)

    with c1:
        card1_html = """
        <div class="lras-card" style="height: 100%; text-align: center; display: flex; flex-direction: column; align-items: center;">
            <div style="background-color: #F4F6F1; width: 48px; height: 48px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-bottom: 14px;">
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#3F5F3E" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                    <circle cx="12" cy="12" r="10"/>
                    <circle cx="12" cy="12" r="6"/>
                    <circle cx="12" cy="12" r="2"/>
                </svg>
            </div>
            <div style="font-weight: 700; font-size: 15.5px; color: #202820; margin-bottom: 8px;">
                Accurate Risk Prediction
            </div>
            <div style="font-size: 13.5px; color: #687067; line-height: 1.55;">
                Utilizes rigorously trained machine learning models to forecast default probabilities.
            </div>
        </div>
        """
        st.markdown(clean_html(card1_html), unsafe_allow_html=True)

    with c2:
        card2_html = """
        <div class="lras-card" style="height: 100%; text-align: center; display: flex; flex-direction: column; align-items: center;">
            <div style="background-color: #F4F6F1; width: 48px; height: 48px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-bottom: 14px;">
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#3F5F3E" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                    <line x1="18" y1="20" x2="18" y2="10"/>
                    <line x1="12" y1="20" x2="12" y2="4"/>
                    <line x1="6" y1="20" x2="6" y2="14"/>
                </svg>
            </div>
            <div style="font-weight: 700; font-size: 15.5px; color: #202820; margin-bottom: 8px;">
                Data-Driven Analysis
            </div>
            <div style="font-size: 13.5px; color: #687067; line-height: 1.55;">
                Evaluates DTI, LTV, and net cash-flow to furnish objective affordability indicators.
            </div>
        </div>
        """
        st.markdown(clean_html(card2_html), unsafe_allow_html=True)

    with c3:
        card3_html = """
        <div class="lras-card" style="height: 100%; text-align: center; display: flex; flex-direction: column; align-items: center;">
            <div style="background-color: #F4F6F1; width: 48px; height: 48px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-bottom: 14px;">
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#3F5F3E" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                    <rect x="2" y="3" width="20" height="14" rx="2" ry="2"/>
                    <line x1="8" y1="21" x2="16" y2="21"/>
                    <line x1="12" y1="17" x2="12" y2="21"/>
                </svg>
            </div>
            <div style="font-weight: 700; font-size: 15.5px; color: #202820; margin-bottom: 8px;">
                Model Insights
            </div>
            <div style="font-size: 13.5px; color: #687067; line-height: 1.55;">
                Benchmark evaluations, holdout test metrics, confusion matrices, and policy trade-offs.
            </div>
        </div>
        """
        st.markdown(clean_html(card3_html), unsafe_allow_html=True)
