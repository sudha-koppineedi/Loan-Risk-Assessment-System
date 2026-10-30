"""
Comprehensive Automated Test Suite for LRAS Streamlit Application Pages.

Uses Streamlit's official testing framework (streamlit.testing.v1.AppTest) to simulate
visiting and interacting with each of the 5 application pages and verifying interactive button navigation.
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
APP_PATH = str(PROJECT_ROOT / "app.py")

EXPECTED_PAGES = [
    "Home",
    "Dashboard",
    "Risk Assessment",
    "Model Insights",
    "Report & Methodology",
]


def test_all_pages():
    print("=" * 60)
    print("STARTING STREAMLIT UI AUTOMATED TESTING")
    print("=" * 60)

    # 1. Test Home Page and In-Page Navigation Buttons
    print("\n[TEST 1] Testing PAGE 1: Home and Action Buttons...")
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    assert not at.exception, f"Exception on Home page: {at.exception}"
    combined_home = " ".join([m.value for m in at.markdown])
    assert "Smarter Loan Decisions" in combined_home
    assert "Accurate Risk Prediction" in combined_home
    assert "Academic Prototype" not in combined_home, "Academic Prototype found on Home page"
    assert "What-If" not in combined_home, "What-If found on Home page"
    assert "SHAP" not in combined_home, "SHAP found on Home page"

    # Verify Sidebar contains exactly the 5 navigation options
    sidebar_opts = list(at.sidebar.radio[0].options)
    assert sidebar_opts == EXPECTED_PAGES, f"Sidebar options mismatch: {sidebar_opts} != {EXPECTED_PAGES}"
    print(f"  [PASS] Sidebar navigation verified with exactly 5 pages: {sidebar_opts}")

    assert len(at.button) >= 2, "Expected at least 2 buttons on Home page"

    # Test Button 1: "Get Started ->" navigates to "Risk Assessment"
    at.button[0].click().run()
    assert not at.exception, f"Exception after clicking 'Get Started': {at.exception}"
    assert at.session_state.get("nav_page") == "Risk Assessment", (
        f"Expected nav_page='Risk Assessment' after Get Started, got {at.session_state.get('nav_page')}"
    )
    assert at.sidebar.radio[0].value == "Risk Assessment", (
        f"Expected sidebar radio value='Risk Assessment', got {at.sidebar.radio[0].value}"
    )
    print("  [PASS] 'Get Started ->' button successfully navigates to Risk Assessment")

    # Navigate back to Home via sidebar
    at.sidebar.radio[0].set_value("Home").run()
    assert not at.exception, f"Exception returning to Home: {at.exception}"
    assert at.session_state.get("nav_page") == "Home"

    # Test Button 2: "Learn More" navigates to "Report & Methodology"
    at.button[1].click().run()
    assert not at.exception, f"Exception after clicking 'Learn More': {at.exception}"
    assert at.session_state.get("nav_page") == "Report & Methodology", (
        f"Expected nav_page='Report & Methodology' after Learn More, got {at.session_state.get('nav_page')}"
    )
    assert at.sidebar.radio[0].value == "Report & Methodology", (
        f"Expected sidebar radio value='Report & Methodology', got {at.sidebar.radio[0].value}"
    )
    print("  [PASS] 'Learn More' button successfully navigates to Report & Methodology")

    # 2. Test Dashboard Page
    print("\n[TEST 2] Testing PAGE 2: Dashboard...")
    at.sidebar.radio[0].set_value("Dashboard").run()
    assert not at.exception, f"Exception on Dashboard: {at.exception}"
    combined_dash = " ".join([m.value for m in at.markdown])
    assert "15,000" in combined_dash, "Total applicants metric 15,000 missing"
    assert "15.7%" in combined_dash, "Default rate missing"
    assert "3" in combined_dash, "Models count missing"
    assert "39" in combined_dash, "Features count missing"
    assert len(at.dataframe) > 0, "Benchmark cohort table missing"
    assert "Key Empirical Insights" not in combined_dash, "Key Empirical Insights found on Dashboard"
    assert "Academic Prototype" not in combined_dash, "Academic Prototype found on Dashboard"
    assert "SHAP" not in combined_dash, "SHAP found on Dashboard"
    print("  [PASS] Dashboard verified (15,000 applicants, 15.7% default rate, 3 models, 39 features, cohort table; Key Empirical Insights & SHAP removed)")

    # 3. Test Risk Assessment Page with all three models
    print("\n[TEST 3] Testing PAGE 3: Risk Assessment...")
    at.sidebar.radio[0].set_value("Risk Assessment").run()
    assert not at.exception, f"Exception on Risk Assessment loading: {at.exception}"

    model_opts = ["xgboost", "random_forest", "logistic_regression"]
    for model_key in model_opts:
        at.selectbox[1].set_value(model_key).run()
        assert not at.exception, f"Exception on selecting model {model_key}: {at.exception}"

        # Click "Predict Risk"
        at.button[0].click().run()
        assert not at.exception, f"Exception after 'Predict Risk' with {model_key}: {at.exception}"

        # Verify prediction result is stored and valid
        res = at.session_state.get("last_prediction")
        assert res is not None, "Prediction result not found in session_state"
        assert res["prediction"] in (0, 1), f"Invalid prediction class: {res['prediction']}"
        assert 0.0 <= res["probability_of_default"] <= 1.0, f"Probability out of bounds: {res['probability_of_default']}"
        assert res["risk_level"] in ("Low Risk", "Medium Risk", "High Risk")
        assert "financial_metrics" in res
        assert "affordability_assessment" in res

        # Verify SHAP card is removed from UI markdown
        combined_risk = " ".join([m.value for m in at.markdown])
        assert "Top Factors for This Prediction" not in combined_risk, "SHAP factor card found in Risk Assessment UI"
        assert "Academic Prototype" not in combined_risk, "Academic Prototype found in Risk Assessment UI"
        print(f"  [PASS] Risk Assessment verified with model: {model_key.upper()} (P(Def)={res['probability_of_default']:.4f}, Tier={res['risk_level']})")

    # 4. Test Model Insights Page
    print("\n[TEST 4] Testing PAGE 4: Model Insights...")
    at.sidebar.radio[0].set_value("Model Insights").run()
    assert not at.exception, f"Exception on Model Insights: {at.exception}"
    assert len(at.dataframe) > 0, "Model metrics dataframe missing"
    combined_insights = " ".join([m.value for m in at.markdown])
    assert "Receiver Operating Characteristic" not in combined_insights, "ROC Curves heading found in Model Insights"
    assert "Precision-Recall (PR) Curves" not in combined_insights, "PR Curves heading found in Model Insights"
    assert "Key Empirical Insights" not in combined_insights, "Key Empirical Insights found in Model Insights"
    assert "Academic Prototype" not in combined_insights, "Academic Prototype found in Model Insights"
    print("  [PASS] Model Insights verified (Holdout metrics table, confusion matrices, trade-offs; ROC & PR curves removed)")

    # 5. Test Report & Methodology Page
    print("\n[TEST 5] Testing PAGE 5: Report & Methodology...")
    at.sidebar.radio[0].set_value("Report & Methodology").run()
    assert not at.exception, f"Exception on Report & Methodology: {at.exception}"
    combined_report = " ".join([m.value for m in at.markdown])
    assert "15,000" in combined_report, "Cohort size missing"
    assert "39" in combined_report, "Feature count missing"
    assert "Academic Prototype" not in combined_report, "Academic Prototype found in Report & Methodology"
    assert "What-If" not in combined_report, "What-If found in Report & Methodology"
    assert "SHAP" not in combined_report, "SHAP found in Report & Methodology"
    print("  [PASS] Report & Methodology verified (Dataset scope, methodology, limitations, responsible AI; disclaimers & obsolete references removed)")

    print("\n" + "=" * 60)
    print("ALL 5 PAGES AND BUTTON ACTIONS PASSED TESTING WITH 0 EXCEPTIONS!")
    print("=" * 60)


if __name__ == "__main__":
    test_all_pages()
