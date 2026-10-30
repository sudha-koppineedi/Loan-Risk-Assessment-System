"""
Model Insights Page for LRAS Streamlit Application.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import streamlit as st

from utils.ui_helpers import (
    clean_html,
    load_model_metrics,
    render_page_header,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"


def render_model_insights_page():
    render_page_header(
        "Model Insights",
        "Compare empirical model evaluation metrics, confusion matrices, and classification trade-offs on the holdout test cohort.",
    )

    tab1, tab2, tab3 = st.tabs(
        ["Model Performance", "Confusion Matrices", "Underwriting Trade-Offs"]
    )

    with tab1:
        card1_html = """
        <div class="lras-card">
            <div class="lras-card-header">
                Holdout Test Set Performance Comparison (N = 3,000)
            </div>
        """
        st.markdown(clean_html(card1_html), unsafe_allow_html=True)

        metrics_df = load_model_metrics()
        formatted_df = metrics_df.copy()
        num_cols = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "PR-AUC"]
        for col in num_cols:
            if col in formatted_df.columns:
                formatted_df[col] = formatted_df[col].map(lambda x: f"{float(x):.4f}")

        st.dataframe(formatted_df, hide_index=True, use_container_width=True)

        note_html = """
        <div style="font-size: 13px; color: #687067; margin-top: 10px; line-height: 1.5;">
            <em>Note: Models were trained on an 80/20 stratified split (12,000 train / 3,000 test, random_state=42). 
            No model is declared a universal "best"; choice depends on the underwriting policy's risk appetite and cost of false positives vs. false negatives.</em>
        </div>
        </div>
        """
        st.markdown(clean_html(note_html), unsafe_allow_html=True)

        st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

        # Comparative Bar Chart
        comp_img = RESULTS_DIR / "model_comparison.png"
        if comp_img.exists():
            comp_header_html = """
            <div class="lras-card">
                <div class="lras-card-header">
                    Multi-Metric Benchmark Comparison
                </div>
            """
            st.markdown(clean_html(comp_header_html), unsafe_allow_html=True)
            st.image(str(comp_img), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    with tab2:
        conf_img = RESULTS_DIR / "confusion_matrices.png"
        if conf_img.exists():
            conf_header_html = """
            <div class="lras-card">
                <div class="lras-card-header">
                    Test Set Confusion Matrices (Threshold = 0.50)
                </div>
            """
            st.markdown(clean_html(conf_header_html), unsafe_allow_html=True)
            st.image(str(conf_img), use_container_width=True)
            conf_sub_html = """
            <div style="font-size: 14px; color: #202820; margin-top: 14px; line-height: 1.6;">
                <strong>Underwriting Takeaways from Classification Distributions:</strong>
                <ul style="color: #687067; margin-top: 6px;">
                    <li><strong style="color: #202820;">Logistic Regression &amp; XGBoost:</strong> High recall on defaults (~72–74%), effectively catching bad loans at the cost of higher false rejection rates on borderline non-defaulters.</li>
                    <li><strong style="color: #202820;">Random Forest:</strong> Higher accuracy (81.1%) and precision (42.4%), reducing false alarms for non-defaulters, but missing more defaults (recall 57.0%).</li>
                </ul>
            </div>
            </div>
            """
            st.markdown(clean_html(conf_sub_html), unsafe_allow_html=True)

    with tab3:
        trade_html = """
        <div class="lras-card">
            <div class="lras-card-header">
                Underwriting Objective Alignment
            </div>
            <div style="font-size: 14px; color: #202820; line-height: 1.65;">
                <p style="color: #687067;">
                    In credit risk modeling, optimizing for <strong style="color: #202820;">pure accuracy is misleading</strong> due to class imbalance (~15.7% default prevalence in the LRAS dataset). 
                    A naive model predicting "No Default" for everyone would score 84.3% accuracy while offering zero underwriting protection.
                </p>
                <table style="width: 100%; border-collapse: collapse; margin: 18px 0; font-size: 13.5px;">
                    <tr style="background-color: #F8FAF7; text-align: left; color: #202820;">
                        <th style="padding: 10px 14px; border: 1px solid #E3E5DD;">Model Candidate</th>
                        <th style="padding: 10px 14px; border: 1px solid #E3E5DD;">Primary Strength</th>
                        <th style="padding: 10px 14px; border: 1px solid #E3E5DD;">Key Limitation</th>
                        <th style="padding: 10px 14px; border: 1px solid #E3E5DD;">Ideal Institutional Use-Case</th>
                    </tr>
                    <tr>
                        <td style="padding: 10px 14px; border: 1px solid #E3E5DD; font-weight: 600;">Logistic Regression</td>
                        <td style="padding: 10px 14px; border: 1px solid #E3E5DD; color: #687067;">Top ROC-AUC (0.8213), high recall (73.8%), full parameter transparency</td>
                        <td style="padding: 10px 14px; border: 1px solid #E3E5DD; color: #687067;">Linear decision boundary; cannot capture deep non-linear interactions</td>
                        <td style="padding: 10px 14px; border: 1px solid #E3E5DD; color: #687067;">Highly regulated lending tiers requiring audit-compliant linear scorecards</td>
                    </tr>
                    <tr style="background-color: #FAFBF9;">
                        <td style="padding: 10px 14px; border: 1px solid #E3E5DD; font-weight: 600;">Random Forest</td>
                        <td style="padding: 10px 14px; border: 1px solid #E3E5DD; color: #687067;">Highest accuracy (81.1%), highest precision (42.4%)</td>
                        <td style="padding: 10px 14px; border: 1px solid #E3E5DD; color: #687067;">Lower recall on defaults (57.0%)</td>
                        <td style="padding: 10px 14px; border: 1px solid #E3E5DD; color: #687067;">Growth-oriented lending with priority on minimizing false rejections</td>
                    </tr>
                    <tr>
                        <td style="padding: 10px 14px; border: 1px solid #E3E5DD; font-weight: 600;">XGBoost</td>
                        <td style="padding: 10px 14px; border: 1px solid #E3E5DD; color: #687067;">Highest F1-Score (0.4899), robust recall (72.1%), strong PR-AUC (0.5201)</td>
                        <td style="padding: 10px 14px; border: 1px solid #E3E5DD; color: #687067;">Sensitive to gradient boost hyperparameters</td>
                        <td style="padding: 10px 14px; border: 1px solid #E3E5DD; color: #687067;">Balanced loss mitigation seeking optimal precision-recall equilibrium</td>
                    </tr>
                </table>
            </div>
        </div>
        """
        st.markdown(clean_html(trade_html), unsafe_allow_html=True)
