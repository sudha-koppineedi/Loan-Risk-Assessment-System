"""
Report & Methodology Page for LRAS Streamlit Application.
"""

from __future__ import annotations

import streamlit as st
from utils.ui_helpers import (
    clean_html,
    render_page_header,
)


def render_about_page():
    render_page_header(
        "Report & Methodology",
        "System architecture, synthetic data provenance, modeling pipeline, and regulatory/ethical governance.",
    )

    c1, c2 = st.columns([1, 1], gap="large")

    with c1:
        # DATASET CARD
        dataset_html = """
        <div class="lras-card">
            <div class="lras-card-header">
                Dataset Provenance &amp; Scope
            </div>
            <ul style="font-size: 14px; color: #687067; line-height: 1.7; margin-bottom: 0;">
                <li><strong style="color: #202820;">Cohort Size:</strong> 15,000 unique fictional synthetic applicants.</li>
                <li><strong style="color: #202820;">Feature Dimensionality:</strong> 39 processed model inputs (demographics, financials, credit bureau history, and engineered cash-flow ratios).</li>
                <li><strong style="color: #202820;">Target Variable:</strong> <code>default</code> (0 = No Default, 1 = Default within observation window).</li>
                <li><strong style="color: #202820;">Default Prevalence:</strong> 15.67% default rate reflecting realistic retail credit delinquency skews.</li>
                <li><strong style="color: #202820;">Exclusions:</strong> <code>applicant_id</code> is strictly isolated as a database primary key and excluded from model features.</li>
            </ul>
        </div>
        """
        st.markdown(clean_html(dataset_html), unsafe_allow_html=True)

        # METHODOLOGY CARD
        method_html = """
        <div class="lras-card">
            <div class="lras-card-header">
                End-to-End Methodology
            </div>
            <ol style="font-size: 14px; color: #687067; line-height: 1.7; margin-bottom: 0; padding-left: 20px;">
                <li><strong style="color: #202820;">Data Preprocessing:</strong> Missing value handling (median for skewed financial figures, mode for categoricals) and percentile winsorization.</li>
                <li><strong style="color: #202820;">Feature Engineering:</strong> Mathematical formulation of amortized EMI, DTI, LTV, expense-to-income, and savings-to-income ratios.</li>
                <li><strong style="color: #202820;">Model Training:</strong> 80/20 stratified train/test split (12,000 train / 3,000 test) with class reweighting.</li>
                <li><strong style="color: #202820;">Calibrated Inference:</strong> Production of posterior default probabilities bounded in [0.0, 1.0].</li>
            </ol>
        </div>
        """
        st.markdown(clean_html(method_html), unsafe_allow_html=True)

        # MACHINE LEARNING MODELS CARD
        models_html = """
        <div class="lras-card">
            <div class="lras-card-header">
                Machine Learning Models
            </div>
            <div style="font-size: 14px; color: #687067; line-height: 1.65;">
                <div style="margin-bottom: 10px;">
                    <strong style="color: #202820;">1. Logistic Regression:</strong> Fitted with <code>RobustScaler</code> and balanced class weighting; provides optimal linear baseline transparency (ROC-AUC: 0.8213).
                </div>
                <div style="margin-bottom: 10px;">
                    <strong style="color: #202820;">2. Random Forest Classifier:</strong> 100 ensemble decision trees with depth pruning; excels at precision (0.4241) and overall accuracy (81.13%).
                </div>
                <div>
                    <strong style="color: #202820;">3. XGBoost Classifier:</strong> Gradient boosted decision trees utilizing <code>scale_pos_weight</code>; achieves top F1-Score (0.4899) and robust recall (72.13%).
                </div>
            </div>
        </div>
        """
        st.markdown(clean_html(models_html), unsafe_allow_html=True)

    with c2:
        # KEY CAPABILITIES CARD
        cap_html = """
        <div class="lras-card">
            <div class="lras-card-header">
                Core System Capabilities
            </div>
            <ul style="font-size: 14px; color: #687067; line-height: 1.7; margin-bottom: 0;">
                <li><strong style="color: #202820;">Probabilistic Default Forecasting:</strong> Calibrated risk likelihood estimates rather than ungrounded binary verdicts.</li>
                <li><strong style="color: #202820;">Comprehensive Affordability Analysis:</strong> Joint evaluation of debt commitments, net disposable cash flow, and collateral coverage.</li>
                <li><strong style="color: #202820;">Diagnostic Performance Benchmarking:</strong> Holdout test cohort evaluations with multi-metric comparison and classification trade-off analysis.</li>
            </ul>
        </div>
        """
        st.markdown(clean_html(cap_html), unsafe_allow_html=True)

        # LIMITATIONS CARD
        limit_html = """
        <div class="lras-card" style="border-left: 3px solid #D97732;">
            <div class="lras-card-header" style="color: #B45309;">
                Model Assumptions &amp; System Limitations
            </div>
            <ul style="font-size: 14px; color: #687067; line-height: 1.7; margin-bottom: 0;">
                <li><strong style="color: #202820;">Synthetic Data:</strong> No actual individual consumer financial records were accessed or utilized.</li>
                <li><strong style="color: #202820;">Macroeconomic Invariance:</strong> Models do not account for external macroeconomic fluctuations, inflation swings, or interest rate cycle shocks.</li>
                <li><strong style="color: #202820;">No Live Bureau Connection:</strong> Relies entirely on self-contained application inputs without external API verification.</li>
                <li><strong style="color: #202820;">Advisory Decision Support Only:</strong> Not a certified credit rating agency or formal underwriting engine.</li>
            </ul>
        </div>
        """
        st.markdown(clean_html(limit_html), unsafe_allow_html=True)

        # ETHICAL CONSIDERATIONS CARD
        ethical_html = """
        <div class="lras-card" style="border-left: 3px solid #3F5F3E;">
            <div class="lras-card-header" style="color: #202820;">
                Ethical AI &amp; Human Oversight
            </div>
            <div style="font-size: 14px; color: #687067; line-height: 1.65;">
                <p>
                    Algorithmic lending systems risk perpetuating historical disparities if unmonitored. LRAS adheres to core responsible AI principles:
                </p>
                <ul style="margin-bottom: 0;">
                    <li><strong style="color: #202820;">No Protected Class Discrimination:</strong> Protected demographic attributes (race, religion, gender) are excluded from the feature space.</li>
                    <li><strong style="color: #202820;">Objective Risk Transparency:</strong> Underwriting risk ratings are supported by clear quantitative cash-flow, leverage (DTI), and collateral coverage metrics.</li>
                    <li><strong style="color: #202820;">Human-in-the-Loop:</strong> The system is designed strictly as an underwriting <em>decision-support tool</em> to augment human judgment, not replace loan officers.</li>
                </ul>
            </div>
        </div>
        """
        st.markdown(clean_html(ethical_html), unsafe_allow_html=True)
