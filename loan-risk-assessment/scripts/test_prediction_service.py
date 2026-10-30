"""
Comprehensive Test Suite for LRAS Backend Prediction Service.

Tests:
1. Standalone financial calculation formulas (EMI, DTI, LTV, Expense/Income, Savings/Income)
2. Input validation (missing fields, invalid categoricals, out-of-range values)
3. Feature vector preparation & strict ordering (39 features, applicant_id exclusion)
4. Model prediction & SHAP explainability on existing records from data/loan_risk_processed.csv across all 3 models:
   - Logistic Regression
   - Random Forest
   - XGBoost
5. End-to-end prediction from a manually constructed raw applicant dictionary.
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.prediction_service import (
    ID_COL,
    PROCESSED_FEATURE_NAMES,
    SUPPORTED_MODELS,
    RiskThresholds,
    ValidationError,
    assess_risk_tier,
    calculate_dti,
    calculate_estimated_new_emi,
    calculate_expense_to_income_ratio,
    calculate_ltv,
    calculate_savings_to_income_ratio,
    compute_financial_summary,
    predict_loan_risk,
    prepare_feature_vector,
    validate_raw_applicant_dict,
)

DATA_PATH = PROJECT_ROOT / "data" / "loan_risk_processed.csv"


def test_financial_calculations():
    print("\n[TEST 1] Testing standalone financial calculations...")
    # Test 1: EMI Calculation
    # Principal=500,000, Rate=10.5% p.a., Tenure=60 months
    # Formula: p * r * (1+r)^n / ((1+r)^n - 1) where r = 0.105/12 = 0.00875
    emi = calculate_estimated_new_emi(500000.0, 10.5, 60)
    assert 10700.0 < emi < 10800.0, f"Unexpected EMI: {emi}"
    print(f"  [PASS] EMI calculation verified: {emi:.2f} (expected ~10,746.97)")

    # Test 2: Zero interest EMI
    zero_emi = calculate_estimated_new_emi(60000.0, 0.0, 12)
    assert zero_emi == 5000.0, f"Expected 5000.0, got {zero_emi}"
    print("  [PASS] Zero-interest EMI verified")

    # Test 3: DTI Calculation
    # existing_emi=5000, new_emi=10000, income=50000 -> DTI = 15000 / 50000 = 0.30
    dti = calculate_dti(5000.0, 10000.0, 50000.0)
    assert dti == 0.30, f"Expected 0.30, got {dti}"
    print("  [PASS] DTI calculation verified: 0.30")

    # Test 4: LTV Calculation
    # loan=400,000, collateral=500,000 -> 0.80
    ltv = calculate_ltv(400000.0, 500000.0)
    assert ltv == 0.80, f"Expected 0.80, got {ltv}"
    # Unsecured loan (collateral=0) -> 1.0
    ltv_unsecured = calculate_ltv(400000.0, 0.0)
    assert ltv_unsecured == 1.0, f"Expected 1.0, got {ltv_unsecured}"
    print("  [PASS] LTV calculation verified (secured=0.80, unsecured=1.0)")

    # Test 5: Expense-to-income and Savings-to-income
    exp_ratio = calculate_expense_to_income_ratio(25000.0, 50000.0)
    assert exp_ratio == 0.50, f"Expected 0.50, got {exp_ratio}"
    # savings=120,000, monthly_income=50,000 -> annual income=600,000 -> ratio=0.20
    sav_ratio = calculate_savings_to_income_ratio(120000.0, 50000.0)
    assert sav_ratio == 0.20, f"Expected 0.20, got {sav_ratio}"
    print("  [PASS] Expense & Savings ratios verified")


def test_input_validation():
    print("\n[TEST 2] Testing input validation & error handling...")

    valid_raw = {
        "age": 35,
        "dependents": 2,
        "employment_years": 8.0,
        "monthly_income": 65000.0,
        "monthly_expenses": 26000.0,
        "existing_emi": 6000.0,
        "savings": 250000.0,
        "outstanding_debt": 150000.0,
        "credit_score": 720,
        "credit_history_years": 10.0,
        "active_loans": 2,
        "previous_defaults": 0,
        "late_payments": 1,
        "loan_amount": 400000.0,
        "loan_tenure_months": 48,
        "interest_rate": 11.5,
        "collateral_value": 600000.0,
        "bank_balance_avg": 45000.0,
        "credit_utilization": 28.0,
        "employment_type": "Salaried",
        "loan_purpose": "Home",
        "application_channel": "Online",
    }

    # Case A: Valid input passes
    cleaned = validate_raw_applicant_dict(valid_raw)
    assert cleaned["employment_type"] == "Salaried"
    print("  [PASS] Valid applicant dictionary passed cleanly")

    # Case B: Missing field raises ValidationError
    invalid_missing = valid_raw.copy()
    del invalid_missing["monthly_income"]
    try:
        validate_raw_applicant_dict(invalid_missing)
        assert False, "Failed to raise ValidationError for missing field"
    except ValidationError as e:
        assert "monthly_income" in str(e)
        print("  [PASS] Missing field correctly caught by ValidationError")

    # Case C: Invalid categorical raises ValidationError
    invalid_cat = valid_raw.copy()
    invalid_cat["employment_type"] = "Astronaut"
    try:
        validate_raw_applicant_dict(invalid_cat)
        assert False, "Failed to raise ValidationError for invalid categorical"
    except ValidationError as e:
        assert "employment_type" in str(e)
        print("  [PASS] Invalid category correctly caught by ValidationError")

    # Case D: Out of bounds / negative numeric raises ValidationError
    invalid_num = valid_raw.copy()
    invalid_num["credit_score"] = -50
    try:
        validate_raw_applicant_dict(invalid_num)
        assert False, "Failed to raise ValidationError for negative score"
    except ValidationError as e:
        assert "credit_score" in str(e)
        print("  [PASS] Out-of-bounds numeric correctly caught by ValidationError")


def test_feature_vector_preparation():
    print("\n[TEST 3] Testing feature vector preparation & column ordering...")
    df_processed = pd.read_csv(DATA_PATH)
    sample_row = df_processed.iloc[[0]]

    # Prepare feature vector from processed row
    feat_df = prepare_feature_vector(sample_row)

    assert feat_df.shape == (1, 39), f"Expected shape (1, 39), got {feat_df.shape}"
    assert list(feat_df.columns) == PROCESSED_FEATURE_NAMES, "Column order mismatch"
    assert ID_COL not in feat_df.columns, "applicant_id was not excluded"
    assert "default" not in feat_df.columns, "default target was not excluded"
    assert np.isfinite(feat_df.to_numpy()).all(), "Feature vector contains non-finite values"
    print("  [PASS] Processed DataFrame converted to exact 39-feature vector, preserving ordering and excluding applicant_id")


def test_risk_tiers():
    print("\n[TEST 4] Testing risk tier categorization...")
    thresholds = RiskThresholds(low_risk_max=0.35, high_risk_min=0.60)

    low_tier = assess_risk_tier(0.20, thresholds)
    assert low_tier["risk_level"] == "Low Risk"

    med_tier = assess_risk_tier(0.48, thresholds)
    assert med_tier["risk_level"] == "Medium Risk"

    high_tier = assess_risk_tier(0.75, thresholds)
    assert high_tier["risk_level"] == "High Risk"
    print("  [PASS] Risk tiers correctly classified into Low (<0.35), Medium (0.35-0.60), and High (>=0.60)")


def test_model_predictions_on_processed_dataset():
    print("\n[TEST 5] Testing all three models with processed dataset records...")
    df_processed = pd.read_csv(DATA_PATH)

    # Pick 2 diverse test applicants: 1 actual default, 1 actual non-default
    def_row = df_processed[df_processed["default"] == 1].iloc[[0]]
    non_def_row = df_processed[df_processed["default"] == 0].iloc[[0]]

    for label, row in [("Actual Default", def_row), ("Actual Non-Default", non_def_row)]:
        app_id = row[ID_COL].iloc[0]
        actual_target = row["default"].iloc[0]
        print(f"\n  Evaluating Applicant: {app_id} ({label}, Ground Truth: {actual_target})")

        for model_name in SUPPORTED_MODELS:
            res = predict_loan_risk(row, model_name=model_name, top_k_factors=5)

            # Assertions
            assert res["prediction"] in [0, 1], f"Invalid prediction: {res['prediction']}"
            assert res["prediction_label"] in ["Default", "No Default"]
            assert 0.0 <= res["probability_of_default"] <= 1.0, f"Invalid prob: {res['probability_of_default']}"
            assert 0.0 <= res["probability_of_no_default"] <= 1.0
            assert abs(res["probability_of_default"] + res["probability_of_no_default"] - 1.0) < 1e-3
            assert res["risk_level"] in ["Low Risk", "Medium Risk", "High Risk"]
            assert res["selected_model"] == model_name

            # Financial metrics assertions
            fin = res["financial_metrics"]
            for m in ["monthly_income", "estimated_new_emi", "total_emi", "DTI", "LTV"]:
                assert m in fin, f"Missing metric {m}"
                assert np.isfinite(fin[m]), f"Non-finite metric {m}: {fin[m]}"

            # Affordability assessment
            afford = res["affordability_assessment"]
            assert afford["status"] in ["Favorable", "Moderate Strain", "High Burden / Strained"]
            assert np.isfinite(afford["net_disposable_income"])

            # SHAP assertions
            shap_dict = res["shap_values"]
            assert len(shap_dict) == 39, f"Expected 39 SHAP values, got {len(shap_dict)}"
            assert np.isfinite(list(shap_dict.values())).all(), "SHAP values contain non-finite numbers"
            assert len(res["risk_increasing_factors"]) > 0 or len(res["risk_reducing_factors"]) > 0

            print(
                f"    [{model_name.upper()}] Pred: {res['prediction_label']} | "
                f"P(Default)={res['probability_of_default']:.4f} | "
                f"Tier: {res['risk_level']} | "
                f"DTI: {fin['DTI']:.4f} | "
                f"Affordability: {afford['status']}"
            )
            top_risk = [f"{r['feature']} (+{r['shap_value']:.3f})" for r in res["risk_increasing_factors"][:2]]
            top_prot = [f"{p['feature']} ({p['shap_value']:.3f})" for p in res["risk_reducing_factors"][:2]]
            print(f"      Top Risk: {', '.join(top_risk)}")
            print(f"      Top Protective: {', '.join(top_prot)}")


def test_manually_constructed_applicant():
    print("\n[TEST 6] Testing manually constructed raw applicant input...")
    # Prime applicant: High income, great credit, reasonable loan
    prime_borrower = {
        "age": 42,
        "dependents": 1,
        "employment_years": 12.0,
        "monthly_income": 120000.0,
        "monthly_expenses": 35000.0,
        "existing_emi": 8000.0,
        "savings": 800000.0,
        "outstanding_debt": 150000.0,
        "credit_score": 790,
        "credit_history_years": 15.0,
        "active_loans": 1,
        "previous_defaults": 0,
        "late_payments": 0,
        "loan_amount": 1000000.0,
        "loan_tenure_months": 84,
        "interest_rate": 9.5,
        "collateral_value": 1800000.0,
        "bank_balance_avg": 250000.0,
        "credit_utilization": 15.0,
        "employment_type": "Salaried",
        "loan_purpose": "Home",
        "application_channel": "Online",
    }

    res_prime = predict_loan_risk(prime_borrower, model_name="xgboost")
    print(
        f"  Prime Borrower Result [XGBoost]:\n"
        f"    Pred: {res_prime['prediction_label']} (P(Default)={res_prime['probability_of_default']:.4f})\n"
        f"    Risk Tier: {res_prime['risk_level']}\n"
        f"    Estimated New EMI: INR {res_prime['financial_metrics']['estimated_new_emi']:,.2f}\n"
        f"    DTI: {res_prime['financial_metrics']['DTI']:.4f}\n"
        f"    LTV: {res_prime['financial_metrics']['LTV']:.4f}\n"
        f"    Affordability Status: {res_prime['affordability_assessment']['status']}"
    )
    assert res_prime["prediction"] == 0, "Expected Prime Borrower to be classified as No Default"
    assert res_prime["risk_level"] == "Low Risk", "Expected Prime Borrower to be Low Risk"

    # Subprime applicant: Lower income, high debt, prior late payments, lower credit score
    subprime_borrower = {
        "age": 28,
        "dependents": 3,
        "employment_years": 2.0,
        "monthly_income": 30000.0,
        "monthly_expenses": 20000.0,
        "existing_emi": 9000.0,
        "savings": 15000.0,
        "outstanding_debt": 250000.0,
        "credit_score": 580,
        "credit_history_years": 3.0,
        "active_loans": 4,
        "previous_defaults": 1,
        "late_payments": 6,
        "loan_amount": 350000.0,
        "loan_tenure_months": 36,
        "interest_rate": 18.0,
        "collateral_value": 0.0,
        "bank_balance_avg": 4000.0,
        "credit_utilization": 82.0,
        "employment_type": "Contract",
        "loan_purpose": "Personal",
        "application_channel": "Agent",
    }

    res_subprime = predict_loan_risk(subprime_borrower, model_name="xgboost")
    print(
        f"\n  Subprime Borrower Result [XGBoost]:\n"
        f"    Pred: {res_subprime['prediction_label']} (P(Default)={res_subprime['probability_of_default']:.4f})\n"
        f"    Risk Tier: {res_subprime['risk_level']}\n"
        f"    Estimated New EMI: INR {res_subprime['financial_metrics']['estimated_new_emi']:,.2f}\n"
        f"    DTI: {res_subprime['financial_metrics']['DTI']:.4f}\n"
        f"    LTV: {res_subprime['financial_metrics']['LTV']:.4f}\n"
        f"    Affordability Status: {res_subprime['affordability_assessment']['status']}"
    )
    assert res_subprime["probability_of_default"] > 0.50, "Expected high default probability for subprime borrower"
    assert res_subprime["risk_level"] in ["Medium Risk", "High Risk"]
    print("  [PASS] Manually constructed raw applicants successfully evaluated")


def main():
    print("=" * 60)
    print("LRAS BACKEND PREDICTION SERVICE TEST SUITE")
    print("=" * 60)

    test_financial_calculations()
    test_input_validation()
    test_feature_vector_preparation()
    test_risk_tiers()
    test_model_predictions_on_processed_dataset()
    test_manually_constructed_applicant()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED SUCCESSFULLY! BACKEND SERVICE IS OPERATIONAL.")
    print("=" * 60)


if __name__ == "__main__":
    main()
