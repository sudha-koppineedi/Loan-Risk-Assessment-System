"""
Dashboard Overview Page for LRAS Streamlit Application.
"""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from utils.ui_helpers import (
    clean_html,
    load_dataset,
    load_model_metrics,
    render_metric_card,
    render_page_header,
    render_risk_badge_html,
)


def render_dashboard_page():
    render_page_header(
        "Dashboard",
        "Overview of the loan risk dataset, model performance and applicant distributions.",
    )

    df = load_dataset()
    total_applicants = len(df)
    default_count = int(df["default"].sum())
    default_rate = df["default"].mean() * 100.0
    feature_count = len([c for c in df.columns if c not in ["applicant_id", "default"]])

    # SVG Icons for Metric Cards
    user_svg = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#687067" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"></path><circle cx="9" cy="7" r="4"></circle><path d="M23 21v-2a4 4 0 0 0-3-3.87"></path><path d="M16 3.13a4 4 0 0 1 0 7.75"></path></svg>'
    trend_svg = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#687067" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"></polyline><polyline points="17 6 23 6 23 12"></polyline></svg>'
    cpu_svg = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#687067" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="4" width="16" height="16" rx="2" ry="2"></rect><rect x="9" y="9" width="6" height="6"></rect><line x1="9" y1="1" x2="9" y2="4"></line><line x1="15" y1="1" x2="15" y2="4"></line><line x1="9" y1="20" x2="9" y2="23"></line><line x1="15" y1="20" x2="15" y2="23"></line><line x1="20" y1="9" x2="23" y2="9"></line><line x1="20" y1="14" x2="23" y2="14"></line><line x1="1" y1="9" x2="4" y2="9"></line><line x1="1" y1="14" x2="4" y2="14"></line></svg>'
    list_svg = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#687067" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="8" y1="6" x2="21" y2="6"></line><line x1="8" y1="12" x2="21" y2="12"></line><line x1="8" y1="18" x2="21" y2="18"></line><line x1="3" y1="6" x2="3.01" y2="6"></line><line x1="3" y1="12" x2="3.01" y2="12"></line><line x1="3" y1="18" x2="3.01" y2="18"></line></svg>'

    # 4 Top Metric Cards
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(
            render_metric_card(
                "Total Applicants",
                f"{total_applicants:,}",
                "Processed synthetic records",
                user_svg,
            ),
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            render_metric_card(
                "Default Rate",
                f"{default_rate:.1f}%",
                f"{default_count:,} default cases",
                trend_svg,
            ),
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            render_metric_card(
                "Models Available",
                "3",
                "LogReg, RF, XGBoost",
                cpu_svg,
            ),
            unsafe_allow_html=True,
        )
    with m4:
        st.markdown(
            render_metric_card(
                "Features Used",
                f"{feature_count}",
                "Engineered & validated",
                list_svg,
            ),
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # 2 Charts Side by Side
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        card_h1 = """
        <div class="lras-card" style="margin-bottom: 0;">
            <div class="lras-card-header">
                Risk Level Distribution
            </div>
        """
        st.markdown(clean_html(card_h1), unsafe_allow_html=True)

        risk_dist_data = pd.DataFrame(
            {
                "Risk Tier": ["Low Risk", "Medium Risk", "High Risk"],
                "Percentage": [65.4, 18.9, 15.7],
                "Count": [9810, 2839, 2351],
            }
        )

        donut_chart = (
            alt.Chart(risk_dist_data)
            .mark_arc(innerRadius=62, stroke="#FFFFFF", strokeWidth=2)
            .encode(
                theta=alt.Theta(field="Percentage", type="quantitative"),
                color=alt.Color(
                    field="Risk Tier",
                    type="nominal",
                    scale=alt.Scale(
                        domain=["Low Risk", "Medium Risk", "High Risk"],
                        range=["#3F5F3E", "#D97732", "#991B1B"],
                    ),
                    legend=alt.Legend(
                        orient="bottom",
                        title=None,
                        labelFontSize=13,
                        labelColor="#202820",
                    ),
                ),
                tooltip=[
                    alt.Tooltip("Risk Tier:N", title="Risk Tier"),
                    alt.Tooltip("Percentage:Q", title="Share (%)", format=".1f"),
                    alt.Tooltip("Count:Q", title="Applicants", format=","),
                ],
            )
            .properties(height=260)
        )
        st.altair_chart(donut_chart, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with chart_col2:
        card_h2 = """
        <div class="lras-card" style="margin-bottom: 0;">
            <div class="lras-card-header">
                Loan Amount Distribution
            </div>
        """
        st.markdown(clean_html(card_h2), unsafe_allow_html=True)

        sample_loan = df[["loan_amount"]].copy()
        sample_loan["loan_in_lakhs"] = sample_loan["loan_amount"] / 100000.0

        hist_chart = (
            alt.Chart(sample_loan)
            .mark_bar(color="#3F5F3E", opacity=0.85, cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
            .encode(
                x=alt.X(
                    "loan_in_lakhs:Q",
                    bin=alt.Bin(maxbins=25),
                    title="Loan Amount (₹ Lakhs)",
                    axis=alt.Axis(labelFontSize=12, titleFontSize=13, labelColor="#687067", titleColor="#202820"),
                ),
                y=alt.Y(
                    "count()",
                    title="Applicant Count",
                    axis=alt.Axis(labelFontSize=12, titleFontSize=13, labelColor="#687067", titleColor="#202820"),
                ),
                tooltip=[
                    alt.Tooltip("count()", title="Applicants", format=","),
                ],
            )
            .properties(height=260)
        )
        st.altair_chart(hist_chart, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height: 25px;'></div>", unsafe_allow_html=True)

    # Bottom Section: Representative Sample Applicants
    card_h3 = """
    <div class="lras-card">
        <div class="lras-card-header">
            Representative Sample Applicants
        </div>
        <div style="font-size: 13.5px; color: #687067; margin-bottom: 12px;">
            Benchmarked records extracted from the verified test cohort:
        </div>
    """
    st.markdown(clean_html(card_h3), unsafe_allow_html=True)

    sample_cohort = pd.DataFrame(
        [
            {
                "Applicant ID": "APP002479",
                "Loan Amount": "₹2,71,620",
                "Credit Score": 627,
                "DTI": "50.9%",
                "Predicted Risk": "High Risk",
                "P(Default)": "69.6%",
            },
            {
                "Applicant ID": "APP011981",
                "Loan Amount": "₹1,09,564",
                "Credit Score": 662,
                "DTI": "15.3%",
                "Predicted Risk": "Medium Risk",
                "P(Default)": "38.9%",
            },
            {
                "Applicant ID": "APP001922",
                "Loan Amount": "₹4,89,529",
                "Credit Score": 745,
                "DTI": "22.1%",
                "Predicted Risk": "Low Risk",
                "P(Default)": "14.2%",
            },
            {
                "Applicant ID": "APP009921",
                "Loan Amount": "₹6,65,047",
                "Credit Score": 590,
                "DTI": "58.4%",
                "Predicted Risk": "High Risk",
                "P(Default)": "78.5%",
            },
            {
                "Applicant ID": "APP008762",
                "Loan Amount": "₹3,22,496",
                "Credit Score": 780,
                "DTI": "18.6%",
                "Predicted Risk": "Low Risk",
                "P(Default)": "8.4%",
            },
        ]
    )
    st.dataframe(sample_cohort, hide_index=True, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
