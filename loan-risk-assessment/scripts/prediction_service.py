"""
Backend Prediction Service for Intelligent Loan Risk Assessment & Underwriting Support System (LRAS).

Provides a clean, modular, and reusable service layer that handles:
1. Applicant Input Handling & Strict Validation
2. Deterministic Financial Ratio Calculations (exact formulas matching preprocess.py)
3. Feature Vector Assembly & Strict Ordering (39 features, applicant_id excluded)
4. Model Loading & In-Memory Caching (Logistic Regression, Random Forest, XGBoost)
5. Model Inference (Predicted Class & Probability of Default)
6. Configurable Risk Tiers (Low, Medium, High Risk)
7. Illustrative Affordability Assessment
8. SHAP Explainability Integration (reusing scripts/shap_analysis.py)
9. Clean Structured Output for Downstream Consumers (e.g., Streamlit UI)

DISCLAIMER:
This system is designed for loan risk assessment and underwriting support.
It does not make real lending decisions.
All underlying training and evaluation data are synthetic.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd

# Import SHAP explainability utilities from verified existing module
from scripts.shap_analysis import explain_individual_prediction

# Constants & Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_PATH = PROJECT_ROOT / "data" / "loan_risk_processed.csv"

ID_COL = "applicant_id"
TARGET_COL = "default"

# Supported Model Keys
MODEL_LOGISTIC_REGRESSION = "logistic_regression"
MODEL_RANDOM_FOREST = "random_forest"
MODEL_XGBOOST = "xgboost"

SUPPORTED_MODELS = [
    MODEL_LOGISTIC_REGRESSION,
    MODEL_RANDOM_FOREST,
    MODEL_XGBOOST,
]

MODEL_NAME_MAP = {
    MODEL_LOGISTIC_REGRESSION: "Logistic Regression",
    MODEL_RANDOM_FOREST: "Random Forest",
    MODEL_XGBOOST: "XGBoost",
}

MODEL_FILE_MAP = {
    MODEL_LOGISTIC_REGRESSION: "logistic_regression.pkl",
    MODEL_RANDOM_FOREST: "random_forest.pkl",
    MODEL_XGBOOST: "xgboost.pkl",
}

# The exact 39 features in order of training
PROCESSED_FEATURE_NAMES: list[str] = [
    "age",
    "dependents",
    "employment_years",
    "monthly_income",
    "monthly_expenses",
    "existing_emi",
    "savings",
    "outstanding_debt",
    "credit_score",
    "credit_history_years",
    "active_loans",
    "previous_defaults",
    "late_payments",
    "loan_amount",
    "loan_tenure_months",
    "interest_rate",
    "collateral_value",
    "bank_balance_avg",
    "credit_utilization",
    "estimated_new_emi",
    "DTI",
    "LTV",
    "expense_to_income_ratio",
    "savings_to_income_ratio",
    "employment_type_Salaried",
    "employment_type_Self-employed",
    "employment_type_Business",
    "employment_type_Contract",
    "employment_type_Unemployed",
    "loan_purpose_Home",
    "loan_purpose_Education",
    "loan_purpose_Vehicle",
    "loan_purpose_Personal",
    "loan_purpose_Business",
    "loan_purpose_Medical",
    "loan_purpose_Other",
    "application_channel_Online",
    "application_channel_Branch",
    "application_channel_Agent",
]

# Categorical Categories (exact order as preprocess.py OneHotEncoder)
CATEGORICAL_OPTIONS: dict[str, list[str]] = {
    "employment_type": ["Salaried", "Self-employed", "Business", "Contract", "Unemployed"],
    "loan_purpose": ["Home", "Education", "Vehicle", "Personal", "Business", "Medical", "Other"],
    "application_channel": ["Online", "Branch", "Agent"],
}

# Standardization mappings for raw categorical strings
EMPLOYMENT_MAP = {
    "salaried": "Salaried",
    "self-employed": "Self-employed",
    "self employed": "Self-employed",
    "selfemployed": "Self-employed",
    "business": "Business",
    "contract": "Contract",
    "unemployed": "Unemployed",
}
PURPOSE_MAP = {
    "home": "Home",
    "education": "Education",
    "vehicle": "Vehicle",
    "personal": "Personal",
    "business": "Business",
    "medical": "Medical",
    "other": "Other",
}
CHANNEL_MAP = {
    "online": "Online",
    "branch": "Branch",
    "agent": "Agent",
}

# Documented input validation ranges (from preprocess.py)
NUMERIC_RANGES: dict[str, tuple[float, float]] = {
    "age": (21, 65),
    "dependents": (0, 5),
    "employment_years": (0.0, 40.0),
    "monthly_income": (15_000.0, 500_000.0),
    "monthly_expenses": (5_000.0, 250_000.0),
    "existing_emi": (0.0, 150_000.0),
    "savings": (5_000.0, 5_000_000.0),
    "outstanding_debt": (0.0, 5_000_000.0),
    "credit_score": (300, 850),
    "credit_history_years": (0.0, 40.0),
    "active_loans": (0, 8),
    "previous_defaults": (0, 3),
    "late_payments": (0, 15),
    "loan_amount": (50_000.0, 10_000_000.0),
    "loan_tenure_months": (12, 360),
    "interest_rate": (6.0, 24.0),
    "collateral_value": (0.0, 15_000_000.0),
    "bank_balance_avg": (1_000.0, 3_000_000.0),
    "credit_utilization": (0.0, 100.0),
}


class ValidationError(Exception):
    """Custom exception raised when applicant input validation fails."""

    def __init__(self, errors: list[str]):
        super().__init__("; ".join(errors))
        self.errors = errors


@dataclass
class RiskThresholds:
    """Configurable probability thresholds for illustrative prototype risk tiers."""

    low_risk_max: float = 0.35
    high_risk_min: float = 0.60

    def __post_init__(self):
        if not (0.0 <= self.low_risk_max < self.high_risk_min <= 1.0):
            raise ValueError(
                f"Invalid thresholds: low_risk_max ({self.low_risk_max}) must be less than "
                f"high_risk_min ({self.high_risk_min}), and both must be between 0.0 and 1.0."
            )


# ==============================================================================
# SECTION 1: FINANCIAL CALCULATIONS (Deterministic, exact preprocess.py formulas)
# ==============================================================================

def calculate_estimated_new_emi(
    principal: float,
    annual_rate_pct: float,
    tenure_months: float,
) -> float:
    """
    Compute Equated Monthly Installment (EMI) for proposed loan.

    Formula:
        r = (annual_rate_pct / 100.0) / 12.0
        n = max(tenure_months, 1.0)
        p = principal
        If r <= 1e-12: EMI = p / n
        Else: EMI = p * r * (1 + r)^n / ((1 + r)^n - 1)
    """
    p = float(principal)
    annual_pct = float(annual_rate_pct)
    n = max(float(tenure_months), 1.0)
    r = (annual_pct / 100.0) / 12.0

    if r <= 1e-12:
        return round(p / n, 2)

    comp = np.power(1.0 + r, n)
    emi = p * r * comp / (comp - 1.0)
    return round(float(emi), 2)


def calculate_dti(
    existing_emi: float,
    new_emi: float,
    monthly_income: float,
) -> float:
    """
    Compute Debt-to-Income (DTI) ratio.

    Formula: (existing_emi + new_emi) / max(monthly_income, 1.0)
    """
    income = max(float(monthly_income), 1.0)
    total_emi = float(existing_emi) + float(new_emi)
    return round(total_emi / income, 4)


def calculate_ltv(
    loan_amount: float,
    collateral_value: float,
) -> float:
    """
    Compute Loan-to-Value (LTV) ratio.

    Formula:
        If collateral_value > 0: loan_amount / collateral_value
        Else: 1.0 (unsecured loan modeling convention)
    """
    collateral = float(collateral_value)
    if collateral > 0.0:
        return round(float(loan_amount) / collateral, 4)
    return 1.0


def calculate_expense_to_income_ratio(
    monthly_expenses: float,
    monthly_income: float,
) -> float:
    """
    Compute Expense-to-Income ratio.

    Formula: monthly_expenses / max(monthly_income, 1.0)
    """
    income = max(float(monthly_income), 1.0)
    return round(float(monthly_expenses) / income, 4)


def calculate_savings_to_income_ratio(
    savings: float,
    monthly_income: float,
) -> float:
    """
    Compute Savings-to-Annual-Income ratio.

    Formula: savings / (max(monthly_income, 1.0) * 12.0)
    """
    income = max(float(monthly_income), 1.0)
    return round(float(savings) / (income * 12.0), 4)


def compute_financial_summary(
    monthly_income: float,
    monthly_expenses: float,
    existing_emi: float,
    loan_amount: float,
    loan_tenure_months: float,
    interest_rate: float,
    collateral_value: float,
    savings: float,
) -> dict[str, Any]:
    """
    Compute a consolidated financial summary and illustrative Affordability Assessment.
    """
    new_emi = calculate_estimated_new_emi(loan_amount, interest_rate, loan_tenure_months)
    total_emi = round(float(existing_emi) + new_emi, 2)
    dti = calculate_dti(existing_emi, new_emi, monthly_income)
    ltv = calculate_ltv(loan_amount, collateral_value)
    exp_inc_ratio = calculate_expense_to_income_ratio(monthly_expenses, monthly_income)
    sav_inc_ratio = calculate_savings_to_income_ratio(savings, monthly_income)

    net_disposable = round(float(monthly_income) - total_emi - float(monthly_expenses), 2)

    # Illustrative prototype heuristics for Affordability Assessment
    if net_disposable <= 0 or dti > 0.60:
        affordability_status = "High Burden / Strained"
        affordability_notes = (
            "Total debt commitments plus essential expenses exceed monthly income, "
            "or Debt-to-Income ratio exceeds 60%. Significant repayment strain indicated."
        )
    elif dti > 0.45 or exp_inc_ratio > 0.60 or net_disposable < (0.15 * monthly_income):
        affordability_status = "Moderate Strain"
        affordability_notes = (
            "Debt-to-Income ratio is between 45% and 60%, or living expenses leave a "
            "tight net disposable cash cushion (<15% of income)."
        )
    else:
        affordability_status = "Favorable"
        affordability_notes = (
            "Debt-to-Income and living expense ratios fall within conservative illustrative "
            "limits, maintaining healthy net disposable income."
        )

    return {
        "monthly_income": round(float(monthly_income), 2),
        "monthly_expenses": round(float(monthly_expenses), 2),
        "existing_emi": round(float(existing_emi), 2),
        "estimated_new_emi": new_emi,
        "total_emi": total_emi,
        "net_disposable_income": net_disposable,
        "DTI": dti,
        "LTV": ltv,
        "expense_to_income_ratio": exp_inc_ratio,
        "savings_to_income_ratio": sav_inc_ratio,
        "affordability_status": affordability_status,
        "affordability_notes": affordability_notes,
        "disclaimer": (
            "Affordability Assessment is an illustrative prototype metric based on simplified "
            "cash-flow heuristics and does not constitute a legal or institutional credit underwriting decision."
        ),
    }


# ==============================================================================
# SECTION 2: INPUT VALIDATION & FEATURE VECTOR ASSEMBLY
# ==============================================================================

def normalize_text_key(value: Any) -> str:
    text = str(value).strip().lower()
    text = " ".join(text.split())
    text = text.replace("_", " ")
    return text


def validate_raw_applicant_dict(data: dict[str, Any]) -> dict[str, Any]:
    """
    Validate raw applicant dictionary fields, data types, and sensible boundaries.
    Returns cleaned applicant dictionary.
    """
    errors: list[str] = []
    cleaned: dict[str, Any] = {}

    if not isinstance(data, dict):
        raise ValidationError(["Applicant input must be a dictionary"])

    # Disallow applicant_id from acting as a predictive feature
    if ID_COL in data:
        # We allow it to be passed for record tracking, but it will never be included in features
        pass

    # 1. Numeric Fields Validation
    for col, (low, high) in NUMERIC_RANGES.items():
        if col not in data or data[col] is None or str(data[col]).strip() == "":
            errors.append(f"Missing required numeric field: '{col}'")
            continue

        try:
            val = float(data[col])
        except (ValueError, TypeError):
            errors.append(f"Field '{col}' must be numeric, got: {data[col]}")
            continue

        if not np.isfinite(val):
            errors.append(f"Field '{col}' must be a finite number")
            continue

        # Sensible bounds check (allow slight flexibility but warn/reject nonsensical inputs)
        if val < 0 and low >= 0:
            errors.append(f"Field '{col}' cannot be negative (value: {val})")
        elif val < low * 0.5 or val > high * 2.0:
            errors.append(
                f"Field '{col}' value {val} is outside plausible validation bounds [{low}, {high}]"
            )

        # Cast integer counts to int
        if col in ["age", "dependents", "credit_score", "active_loans", "previous_defaults", "late_payments", "loan_tenure_months"]:
            cleaned[col] = int(round(val))
        else:
            cleaned[col] = float(val)

    # 2. Categorical Fields Validation
    cat_specs = [
        ("employment_type", EMPLOYMENT_MAP, CATEGORICAL_OPTIONS["employment_type"]),
        ("loan_purpose", PURPOSE_MAP, CATEGORICAL_OPTIONS["loan_purpose"]),
        ("application_channel", CHANNEL_MAP, CATEGORICAL_OPTIONS["application_channel"]),
    ]

    for field_name, norm_map, allowed_list in cat_specs:
        if field_name not in data or data[field_name] is None or str(data[field_name]).strip() == "":
            errors.append(f"Missing required categorical field: '{field_name}'")
            continue

        raw_val = str(data[field_name]).strip()
        norm_val = norm_map.get(normalize_text_key(raw_val))

        if norm_val not in allowed_list:
            errors.append(
                f"Invalid '{field_name}': '{raw_val}'. Allowed values: {allowed_list}"
            )
        else:
            cleaned[field_name] = norm_val

    if errors:
        raise ValidationError(errors)

    return cleaned


def assemble_feature_vector_from_raw(applicant_dict: dict[str, Any]) -> pd.DataFrame:
    """
    Assemble the exact 39-feature DataFrame from a validated raw applicant dictionary.
    Calculates derived ratios and one-hot encoded categories without leaking applicant_id.
    """
    cleaned = validate_raw_applicant_dict(applicant_dict)

    # 1. Derive the 5 engineered financial features
    new_emi = calculate_estimated_new_emi(
        cleaned["loan_amount"], cleaned["interest_rate"], cleaned["loan_tenure_months"]
    )
    dti = calculate_dti(cleaned["existing_emi"], new_emi, cleaned["monthly_income"])
    ltv = calculate_ltv(cleaned["loan_amount"], cleaned["collateral_value"])
    exp_inc_ratio = calculate_expense_to_income_ratio(
        cleaned["monthly_expenses"], cleaned["monthly_income"]
    )
    sav_inc_ratio = calculate_savings_to_income_ratio(
        cleaned["savings"], cleaned["monthly_income"]
    )

    feature_map: dict[str, float] = {
        "age": float(cleaned["age"]),
        "dependents": float(cleaned["dependents"]),
        "employment_years": float(cleaned["employment_years"]),
        "monthly_income": float(cleaned["monthly_income"]),
        "monthly_expenses": float(cleaned["monthly_expenses"]),
        "existing_emi": float(cleaned["existing_emi"]),
        "savings": float(cleaned["savings"]),
        "outstanding_debt": float(cleaned["outstanding_debt"]),
        "credit_score": float(cleaned["credit_score"]),
        "credit_history_years": float(cleaned["credit_history_years"]),
        "active_loans": float(cleaned["active_loans"]),
        "previous_defaults": float(cleaned["previous_defaults"]),
        "late_payments": float(cleaned["late_payments"]),
        "loan_amount": float(cleaned["loan_amount"]),
        "loan_tenure_months": float(cleaned["loan_tenure_months"]),
        "interest_rate": float(cleaned["interest_rate"]),
        "collateral_value": float(cleaned["collateral_value"]),
        "bank_balance_avg": float(cleaned["bank_balance_avg"]),
        "credit_utilization": float(cleaned["credit_utilization"]),
        "estimated_new_emi": float(new_emi),
        "DTI": float(dti),
        "LTV": float(ltv),
        "expense_to_income_ratio": float(exp_inc_ratio),
        "savings_to_income_ratio": float(sav_inc_ratio),
    }

    # 2. One-hot encode the 3 categorical variables (15 dummy columns)
    emp_type = cleaned["employment_type"]
    for opt in CATEGORICAL_OPTIONS["employment_type"]:
        feature_map[f"employment_type_{opt}"] = 1.0 if emp_type == opt else 0.0

    purpose = cleaned["loan_purpose"]
    for opt in CATEGORICAL_OPTIONS["loan_purpose"]:
        feature_map[f"loan_purpose_{opt}"] = 1.0 if purpose == opt else 0.0

    channel = cleaned["application_channel"]
    for opt in CATEGORICAL_OPTIONS["application_channel"]:
        feature_map[f"application_channel_{opt}"] = 1.0 if channel == opt else 0.0

    # 3. Assemble DataFrame strictly in training order
    row_df = pd.DataFrame([feature_map])[PROCESSED_FEATURE_NAMES]
    assert list(row_df.columns) == PROCESSED_FEATURE_NAMES, "Feature order mismatch"
    assert ID_COL not in row_df.columns, "applicant_id must not be in model features"
    return row_df


def prepare_feature_vector(
    input_data: Union[dict[str, Any], pd.DataFrame, pd.Series]
) -> pd.DataFrame:
    """
    Construct the validated 1-row feature DataFrame (39 features) from either:
    1. A raw applicant dictionary with domain fields (e.g. from UI form)
    2. An existing processed record / DataFrame row (e.g. from loan_risk_processed.csv)
    """
    if isinstance(input_data, pd.Series):
        input_data = input_data.to_frame().T

    if isinstance(input_data, pd.DataFrame):
        if len(input_data) != 1:
            raise ValidationError([f"Expected exactly 1 applicant record, got {len(input_data)}"])

        df_copy = input_data.copy()
        if ID_COL in df_copy.columns:
            df_copy = df_copy.drop(columns=[ID_COL])
        if TARGET_COL in df_copy.columns:
            df_copy = df_copy.drop(columns=[TARGET_COL])

        # If already in 39 processed features format:
        missing_features = [f for f in PROCESSED_FEATURE_NAMES if f not in df_copy.columns]
        if not missing_features:
            ordered_df = df_copy[PROCESSED_FEATURE_NAMES].astype(float)
            if not np.isfinite(ordered_df.to_numpy()).all():
                raise ValidationError(["Processed input contains non-finite or NaN values."])
            return ordered_df

        # If it has raw columns, convert row to dict and parse
        row_dict = input_data.iloc[0].to_dict()
        return assemble_feature_vector_from_raw(row_dict)

    elif isinstance(input_data, dict):
        # Check if dict already contains all 39 processed features
        missing_processed = [f for f in PROCESSED_FEATURE_NAMES if f not in input_data]
        if not missing_processed:
            clean_dict = {f: float(input_data[f]) for f in PROCESSED_FEATURE_NAMES}
            ordered_df = pd.DataFrame([clean_dict])[PROCESSED_FEATURE_NAMES]
            if not np.isfinite(ordered_df.to_numpy()).all():
                raise ValidationError(["Processed input contains non-finite or NaN values."])
            return ordered_df

        # Otherwise treat as raw applicant dictionary
        return assemble_feature_vector_from_raw(input_data)

    else:
        raise ValidationError([f"Unsupported input type: {type(input_data)}"])


# ==============================================================================
# SECTION 3: MODEL MANAGER & INFERENCE SERVICE
# ==============================================================================

class ModelManager:
    """
    Thread-safe, lazy-loading model registry and cache for trained ML artifacts.
    """

    def __init__(self, models_dir: Path = MODELS_DIR):
        self.models_dir = models_dir
        self._loaded_models: dict[str, object] = {}

    def get_model(self, model_key: str) -> object:
        canonical_key = model_key.lower().strip()
        if canonical_key not in SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model: '{model_key}'. Supported models: {SUPPORTED_MODELS}"
            )

        if canonical_key not in self._loaded_models:
            file_name = MODEL_FILE_MAP[canonical_key]
            model_path = self.models_dir / file_name
            if not model_path.exists():
                raise FileNotFoundError(f"Trained model artifact not found at: {model_path}")
            self._loaded_models[canonical_key] = joblib.load(model_path)

        return self._loaded_models[canonical_key]

    def clear_cache(self) -> None:
        self._loaded_models.clear()


# Module-level default model manager instance
_DEFAULT_MODEL_MANAGER = ModelManager()


def assess_risk_tier(
    probability_of_default: float,
    thresholds: Optional[RiskThresholds] = None,
) -> dict[str, Any]:
    """
    Classify predicted default probability into illustrative risk tiers.

    Categories:
    - Low Risk: prob < low_risk_max (default < 0.35)
    - Medium Risk: low_risk_max <= prob < high_risk_min (default 0.35 to 0.60)
    - High Risk: prob >= high_risk_min (default >= 0.60)
    """
    if thresholds is None:
        thresholds = RiskThresholds()

    prob = float(probability_of_default)

    if prob < thresholds.low_risk_max:
        risk_level = "Low Risk"
    elif prob < thresholds.high_risk_min:
        risk_level = "Medium Risk"
    else:
        risk_level = "High Risk"

    return {
        "risk_level": risk_level,
        "probability_of_default": round(prob, 4),
        "thresholds_used": asdict(thresholds),
        "disclaimer": (
            "Risk tiers ('Low', 'Medium', 'High') are illustrative application-level thresholds "
            "designed for decision support and do not reflect official bank underwriting policy."
        ),
    }


def predict_loan_risk(
    applicant_data: Union[dict[str, Any], pd.DataFrame, pd.Series],
    model_name: str = MODEL_XGBOOST,
    thresholds: Optional[RiskThresholds] = None,
    top_k_factors: int = 5,
    model_manager: Optional[ModelManager] = None,
) -> dict[str, Any]:
    """
    Primary prediction entrypoint for the LRAS backend.

    Parameters:
    - applicant_data: Raw applicant dictionary or processed feature row/DataFrame.
    - model_name: 'logistic_regression', 'random_forest', or 'xgboost'.
    - thresholds: Optional custom RiskThresholds instance.
    - top_k_factors: Number of top risk & protective SHAP factors to return.
    - model_manager: Optional custom ModelManager instance.

    Returns:
    Structured dictionary containing prediction, probability, risk tier,
    financial metrics, affordability assessment, and SHAP explainability.
    """
    canonical_model = model_name.lower().strip()
    if canonical_model not in SUPPORTED_MODELS:
        raise ValueError(
            f"Invalid model_name '{model_name}'. Choose from: {SUPPORTED_MODELS}"
        )

    mgr = model_manager or _DEFAULT_MODEL_MANAGER
    model = mgr.get_model(canonical_model)
    display_model_name = MODEL_NAME_MAP[canonical_model]

    # 1. Feature Engineering & Vector Preparation
    feature_df = prepare_feature_vector(applicant_data)

    # 2. Model Prediction & Probabilities
    # Preserves probability representation as float between 0.0 and 1.0
    pred_class = int(model.predict(feature_df)[0])
    prob_array = model.predict_proba(feature_df)[0]
    prob_no_default = round(float(prob_array[0]), 4)
    prob_default = round(float(prob_array[1]), 4)

    # 3. Illustrative Risk Tier Categorization
    risk_info = assess_risk_tier(prob_default, thresholds=thresholds)

    # 4. Financial Calculations & Affordability Assessment
    feat_row = feature_df.iloc[0]
    financial_metrics = compute_financial_summary(
        monthly_income=feat_row["monthly_income"],
        monthly_expenses=feat_row["monthly_expenses"],
        existing_emi=feat_row["existing_emi"],
        loan_amount=feat_row["loan_amount"],
        loan_tenure_months=feat_row["loan_tenure_months"],
        interest_rate=feat_row["interest_rate"],
        collateral_value=feat_row["collateral_value"],
        savings=feat_row["savings"],
    )

    # 5. SHAP Explainability Integration (calls verified scripts/shap_analysis.py)
    shap_res = explain_individual_prediction(
        model=model,
        applicant_row=feature_df,
        model_type=display_model_name,
        top_k=top_k_factors,
    )

    # 6. Structured Result
    return {
        "prediction": pred_class,
        "prediction_label": "Default" if pred_class == 1 else "No Default",
        "probability_of_default": prob_default,
        "probability_of_no_default": prob_no_default,
        "risk_level": risk_info["risk_level"],
        "risk_tier_details": risk_info,
        "selected_model": canonical_model,
        "selected_model_display": display_model_name,
        "financial_metrics": financial_metrics,
        "affordability_assessment": {
            "status": financial_metrics["affordability_status"],
            "notes": financial_metrics["affordability_notes"],
            "net_disposable_income": financial_metrics["net_disposable_income"],
            "dti": financial_metrics["DTI"],
            "ltv": financial_metrics["LTV"],
            "disclaimer": financial_metrics["disclaimer"],
        },
        "risk_increasing_factors": shap_res["top_risk_factors"],
        "risk_reducing_factors": shap_res["top_protective_factors"],
        "shap_values": shap_res["shap_values_dict"],
        "disclaimer": (
            "This system is designed for loan risk assessment and underwriting support. "
            "It does not make real lending decisions. Model outputs and risk levels are for decision support demonstration."
        ),
    }
