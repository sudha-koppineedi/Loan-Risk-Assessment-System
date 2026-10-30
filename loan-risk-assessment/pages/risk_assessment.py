"""
Risk Assessment Page (Main Prediction Service Interface).
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from scripts.prediction_service import (
    MODEL_LOGISTIC_REGRESSION,
    MODEL_NAME_MAP,
    MODEL_RANDOM_FOREST,
    MODEL_XGBOOST,
    SUPPORTED_MODELS,
    ValidationError,
    predict_loan_risk,
)
from utils.ui_helpers import (
    clean_html,
    render_affordability_badge_html,
    render_page_header,
    render_risk_badge_html,
)

PRESET_PROFILES = {
    "Custom Input": None,
    "Prime Borrower (Low Risk Example)": {
        "age": 38,
        "dependents": 1,
        "employment_type": "Salaried",
        "employment_years": 10.0,
        "monthly_income": 110000.0,
        "monthly_expenses": 32000.0,
        "existing_emi": 8000.0,
        "savings": 650000.0,
        "outstanding_debt": 120000.0,
        "credit_score": 780,
        "credit_history_years": 12.0,
        "active_loans": 1,
        "previous_defaults": 0,
        "late_payments": 0,
        "credit_utilization": 18.0,
        "loan_amount": 800000.0,
        "loan_tenure_months": 60,
        "interest_rate": 9.5,
        "loan_purpose": "Home",
        "collateral_value": 1400000.0,
        "bank_balance_avg": 180000.0,
        "application_channel": "Online",
    },
    "Subprime Borrower (High Risk Example)": {
        "age": 27,
        "dependents": 2,
        "employment_type": "Contract",
        "employment_years": 1.5,
        "monthly_income": 28000.0,
        "monthly_expenses": 18000.0,
        "existing_emi": 9000.0,
        "savings": 12000.0,
        "outstanding_debt": 220000.0,
        "credit_score": 585,
        "credit_history_years": 2.5,
        "active_loans": 4,
        "previous_defaults": 1,
        "late_payments": 5,
        "credit_utilization": 82.0,
        "loan_amount": 350000.0,
        "loan_tenure_months": 36,
        "interest_rate": 18.5,
        "loan_purpose": "Personal",
        "collateral_value": 0.0,
        "bank_balance_avg": 3500.0,
        "application_channel": "Agent",
    },
    "Moderate Strain Borrower (Medium Risk Example)": {
        "age": 34,
        "dependents": 2,
        "employment_type": "Self-employed",
        "employment_years": 5.0,
        "monthly_income": 60000.0,
        "monthly_expenses": 28000.0,
        "existing_emi": 12000.0,
        "savings": 120000.0,
        "outstanding_debt": 250000.0,
        "credit_score": 670,
        "credit_history_years": 7.0,
        "active_loans": 2,
        "previous_defaults": 0,
        "late_payments": 2,
        "credit_utilization": 45.0,
        "loan_amount": 500000.0,
        "loan_tenure_months": 48,
        "interest_rate": 12.0,
        "loan_purpose": "Vehicle",
        "collateral_value": 500000.0,
        "bank_balance_avg": 45000.0,
        "application_channel": "Branch",
    },
}


def render_risk_assessment_page():
    # Standardized Page Header across all 7 pages
    render_page_header(
        "Risk Assessment",
        "Enter applicant details to predict loan default risk and evaluate affordability metrics.",
    )

    # Preset selection toolbar
    top_bar_col1, top_bar_col2 = st.columns([1.5, 1])
    with top_bar_col1:
        preset_choice = st.selectbox(
            "Quick Preset Borrower Profile (Optional Auto-Fill):",
            options=list(PRESET_PROFILES.keys()),
            index=0,
            help="Select an illustrative borrower profile to automatically populate the form fields below.",
        )
    with top_bar_col2:
        model_choice = st.selectbox(
            "Select ML Algorithm:",
            options=[MODEL_XGBOOST, MODEL_RANDOM_FOREST, MODEL_LOGISTIC_REGRESSION],
            format_func=lambda x: MODEL_NAME_MAP[x],
            index=0,
            help="Choose the model used for inference. No model is pre-judged as universally superior.",
        )

    preset_data = PRESET_PROFILES[preset_choice] or {}

    # Main Form & Results Layout
    form_col, result_col = st.columns([1.1, 0.9], gap="large")

    with form_col:
        form_card_header = """
        <div class="lras-card" style="padding-top: 18px;">
            <div class="lras-card-header">
                Applicant &amp; Underwriting Information
            </div>
        """
        st.markdown(clean_html(form_card_header), unsafe_allow_html=True)

        with st.form("risk_assessment_form"):
            # SECTION 1: Applicant Information
            s1_html = "<div style='font-weight: 700; color: #202820; font-size: 15px; margin-top: 4px; margin-bottom: 10px;'>Section 1: Applicant Information</div>"
            st.markdown(clean_html(s1_html), unsafe_allow_html=True)
            s1_c1, s1_c2 = st.columns(2)
            with s1_c1:
                age = st.number_input(
                    "Age (years)",
                    min_value=21,
                    max_value=65,
                    value=int(preset_data.get("age", 28)),
                    step=1,
                )
                emp_type = st.selectbox(
                    "Employment Type",
                    options=["Salaried", "Self-employed", "Business", "Contract", "Unemployed"],
                    index=["Salaried", "Self-employed", "Business", "Contract", "Unemployed"].index(
                        preset_data.get("employment_type", "Salaried")
                    ),
                )
            with s1_c2:
                dependents = st.number_input(
                    "Number of Dependents",
                    min_value=0,
                    max_value=5,
                    value=int(preset_data.get("dependents", 1)),
                    step=1,
                )
                emp_years = st.number_input(
                    "Employment Duration (years)",
                    min_value=0.0,
                    max_value=40.0,
                    value=float(preset_data.get("employment_years", 3.0)),
                    step=0.5,
                )

            st.markdown("<hr style='margin: 14px 0; border: none; border-top: 1px solid #E3E5DD;'>", unsafe_allow_html=True)

            # SECTION 2: Income & Expenses
            s2_html = "<div style='font-weight: 700; color: #202820; font-size: 15px; margin-bottom: 10px;'>Section 2: Income &amp; Expenses</div>"
            st.markdown(clean_html(s2_html), unsafe_allow_html=True)
            s2_c1, s2_c2 = st.columns(2)
            with s2_c1:
                monthly_income = st.number_input(
                    "Monthly Income (₹)",
                    min_value=15000.0,
                    max_value=500000.0,
                    value=float(preset_data.get("monthly_income", 80000.0)),
                    step=5000.0,
                )
                existing_emi = st.number_input(
                    "Existing EMI (₹)",
                    min_value=0.0,
                    max_value=150000.0,
                    value=float(preset_data.get("existing_emi", 15000.0)),
                    step=1000.0,
                )
                outstanding_debt = st.number_input(
                    "Outstanding Debt (₹)",
                    min_value=0.0,
                    max_value=5000000.0,
                    value=float(preset_data.get("outstanding_debt", 180000.0)),
                    step=10000.0,
                )
            with s2_c2:
                monthly_expenses = st.number_input(
                    "Monthly Expenses (₹)",
                    min_value=5000.0,
                    max_value=250000.0,
                    value=float(preset_data.get("monthly_expenses", 25000.0)),
                    step=2000.0,
                )
                savings = st.number_input(
                    "Liquid Savings (₹)",
                    min_value=5000.0,
                    max_value=5000000.0,
                    value=float(preset_data.get("savings", 250000.0)),
                    step=10000.0,
                )

            st.markdown("<hr style='margin: 14px 0; border: none; border-top: 1px solid #E3E5DD;'>", unsafe_allow_html=True)

            # SECTION 3: Credit Details
            s3_html = "<div style='font-weight: 700; color: #202820; font-size: 15px; margin-bottom: 10px;'>Section 3: Credit History &amp; Bureau Details</div>"
            st.markdown(clean_html(s3_html), unsafe_allow_html=True)
            s3_c1, s3_c2 = st.columns(2)
            with s3_c1:
                credit_score = st.number_input(
                    "Credit Score (CIBIL/FICO)",
                    min_value=300,
                    max_value=850,
                    value=int(preset_data.get("credit_score", 720)),
                    step=10,
                )
                active_loans = st.number_input(
                    "Active Loans Count",
                    min_value=0,
                    max_value=8,
                    value=int(preset_data.get("active_loans", 1)),
                    step=1,
                )
                late_payments = st.number_input(
                    "Late Payments (Past 12M)",
                    min_value=0,
                    max_value=15,
                    value=int(preset_data.get("late_payments", 0)),
                    step=1,
                )
            with s3_c2:
                credit_history_years = st.number_input(
                    "Credit History (years)",
                    min_value=0.0,
                    max_value=40.0,
                    value=float(preset_data.get("credit_history_years", 6.0)),
                    step=0.5,
                )
                previous_defaults = st.number_input(
                    "Previous Defaults Count",
                    min_value=0,
                    max_value=3,
                    value=int(preset_data.get("previous_defaults", 0)),
                    step=1,
                )
                credit_utilization = st.number_input(
                    "Credit Utilization (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=float(preset_data.get("credit_utilization", 25.0)),
                    step=5.0,
                )

            st.markdown("<hr style='margin: 14px 0; border: none; border-top: 1px solid #E3E5DD;'>", unsafe_allow_html=True)

            # SECTION 4: Proposed Loan Details
            s4_html = "<div style='font-weight: 700; color: #202820; font-size: 15px; margin-bottom: 10px;'>Section 4: Proposed Loan Details</div>"
            st.markdown(clean_html(s4_html), unsafe_allow_html=True)
            s4_c1, s4_c2 = st.columns(2)
            with s4_c1:
                loan_amount = st.number_input(
                    "Loan Amount Requested (₹)",
                    min_value=50000.0,
                    max_value=10000000.0,
                    value=float(preset_data.get("loan_amount", 800000.0)),
                    step=50000.0,
                )
                interest_rate = st.number_input(
                    "Interest Rate (% p.a.)",
                    min_value=6.0,
                    max_value=24.0,
                    value=float(preset_data.get("interest_rate", 10.5)),
                    step=0.25,
                )
                collateral_value = st.number_input(
                    "Collateral Value (₹, 0 if unsecured)",
                    min_value=0.0,
                    max_value=15000000.0,
                    value=float(preset_data.get("collateral_value", 1000000.0)),
                    step=50000.0,
                )
            with s4_c2:
                loan_tenure_months = st.number_input(
                    "Loan Tenure (months)",
                    min_value=12,
                    max_value=360,
                    value=int(preset_data.get("loan_tenure_months", 48)),
                    step=12,
                )
                loan_purpose = st.selectbox(
                    "Loan Purpose",
                    options=["Home", "Education", "Vehicle", "Personal", "Business", "Medical", "Other"],
                    index=["Home", "Education", "Vehicle", "Personal", "Business", "Medical", "Other"].index(
                        preset_data.get("loan_purpose", "Home")
                    ),
                )

            st.markdown("<hr style='margin: 14px 0; border: none; border-top: 1px solid #E3E5DD;'>", unsafe_allow_html=True)

            # SECTION 5: Banking & Application Channel
            s5_html = "<div style='font-weight: 700; color: #202820; font-size: 15px; margin-bottom: 10px;'>Section 5: Banking &amp; Application Channel</div>"
            st.markdown(clean_html(s5_html), unsafe_allow_html=True)
            s5_c1, s5_c2 = st.columns(2)
            with s5_c1:
                bank_balance_avg = st.number_input(
                    "Average Bank Balance (₹)",
                    min_value=1000.0,
                    max_value=3000000.0,
                    value=float(preset_data.get("bank_balance_avg", 60000.0)),
                    step=5000.0,
                )
            with s5_c2:
                application_channel = st.selectbox(
                    "Application Channel",
                    options=["Online", "Branch", "Agent"],
                    index=["Online", "Branch", "Agent"].index(
                        preset_data.get("application_channel", "Online")
                    ),
                )

            st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
            submit_btn = st.form_submit_button("Predict Risk", use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # Process Form Submission & Render Results
    with result_col:
        applicant_payload = {
            "age": age,
            "dependents": dependents,
            "employment_type": emp_type,
            "employment_years": emp_years,
            "monthly_income": monthly_income,
            "monthly_expenses": monthly_expenses,
            "existing_emi": existing_emi,
            "savings": savings,
            "outstanding_debt": outstanding_debt,
            "credit_score": credit_score,
            "credit_history_years": credit_history_years,
            "active_loans": active_loans,
            "previous_defaults": previous_defaults,
            "late_payments": late_payments,
            "credit_utilization": credit_utilization,
            "loan_amount": loan_amount,
            "loan_tenure_months": loan_tenure_months,
            "interest_rate": interest_rate,
            "collateral_value": collateral_value,
            "loan_purpose": loan_purpose,
            "bank_balance_avg": bank_balance_avg,
            "application_channel": application_channel,
        }

        # Run prediction if submitted or use cached session result
        if submit_btn or "last_prediction" not in st.session_state:
            with st.spinner("Executing model inference..."):
                try:
                    res = predict_loan_risk(applicant_payload, model_name=model_choice, top_k_factors=5)
                    st.session_state["last_prediction"] = res
                except ValidationError as e:
                    st.error(f"Validation Error: {e}")
                    return
                except Exception as e:
                    st.error(f"Inference Error: {e}")
                    return

        res = st.session_state["last_prediction"]
        fin = res["financial_metrics"]
        afford = res["affordability_assessment"]

        # CARD 1: Prediction Result
        pred_card_html = f"""
        <div class="lras-card" style="text-align: center; border-top: 4px solid #3F5F3E;">
            <div style="font-size: 12.5px; font-weight: 700; color: #687067; text-transform: uppercase; letter-spacing: 0.6px;">
                Prediction Result ({res['selected_model_display']})
            </div>
            <div style="font-size: 14px; color: #202820; margin-top: 4px;">
                Model Classification: <strong>{res['prediction_label']}</strong>
            </div>
            <div style="margin: 16px 0 8px 0;">
                <div style="font-size: 13px; color: #687067;">Estimated Default Probability</div>
                <div style="font-size: 34px; font-weight: 800; color: #202820; line-height: 1.1; letter-spacing: -1px;">
                    {res['probability_of_default'] * 100:.1f}%
                </div>
            </div>
            <div style="display: flex; justify-content: center; gap: 10px; margin: 16px 0 6px 0;">
                {render_risk_badge_html(res['risk_level'])}
                {render_affordability_badge_html(afford['status'])}
            </div>
        </div>
        """
        st.markdown(clean_html(pred_card_html), unsafe_allow_html=True)

        # CARD 2: Affordability Assessment Breakdown
        afford_card_html = f"""
        <div class="lras-card">
            <div class="lras-card-header">
                Affordability Assessment
            </div>
            <div style="font-size: 13.5px; color: #687067; margin-bottom: 14px; line-height: 1.45;">
                {afford['notes']}
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; font-size: 13.5px;">
                <div style="background-color: #F8FAF7; padding: 10px 12px; border-radius: 8px; border: 1px solid #EAECE4;">
                    <span style="color: #687067; font-size: 12px;">Estimated New EMI:</span><br>
                    <strong style="color: #202820; font-size: 15px;">₹{fin['estimated_new_emi']:,.2f}</strong>
                </div>
                <div style="background-color: #F8FAF7; padding: 10px 12px; border-radius: 8px; border: 1px solid #EAECE4;">
                    <span style="color: #687067; font-size: 12px;">Total Monthly EMI:</span><br>
                    <strong style="color: #202820; font-size: 15px;">₹{fin['total_emi']:,.2f}</strong>
                </div>
                <div style="background-color: #F8FAF7; padding: 10px 12px; border-radius: 8px; border: 1px solid #EAECE4;">
                    <span style="color: #687067; font-size: 12px;">Debt-to-Income (DTI):</span><br>
                    <strong style="color: #202820; font-size: 15px;">{fin['DTI'] * 100:.2f}%</strong>
                </div>
                <div style="background-color: #F8FAF7; padding: 10px 12px; border-radius: 8px; border: 1px solid #EAECE4;">
                    <span style="color: #687067; font-size: 12px;">Loan-to-Value (LTV):</span><br>
                    <strong style="color: #202820; font-size: 15px;">{fin['LTV'] * 100:.2f}%</strong>
                </div>
                <div style="background-color: #F8FAF7; padding: 10px 12px; border-radius: 8px; border: 1px solid #EAECE4;">
                    <span style="color: #687067; font-size: 12px;">Expense-to-Income:</span><br>
                    <strong style="color: #202820; font-size: 15px;">{fin['expense_to_income_ratio'] * 100:.2f}%</strong>
                </div>
                <div style="background-color: #F8FAF7; padding: 10px 12px; border-radius: 8px; border: 1px solid #EAECE4;">
                    <span style="color: #687067; font-size: 12px;">Net Disposable Income:</span><br>
                    <strong style="color: {'#991B1B' if fin['net_disposable_income'] < 0 else '#2B572E'}; font-size: 15px;">₹{fin['net_disposable_income']:,.2f}</strong>
                </div>
            </div>
        </div>
        """
        st.markdown(clean_html(afford_card_html), unsafe_allow_html=True)
