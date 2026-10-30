# LRAS Backend Prediction Service Report

**Project:** Intelligent Loan Risk Assessment & Underwriting Support System (LRAS)  
**Module:** Backend Prediction Service Layer (`scripts/prediction_service.py`)  
**Status:** Complete, Verified, and Operational  

---

> [!IMPORTANT]
> **Synthetic Data Notice:**  
> This system is designed for loan risk assessment and underwriting support. It does not make real lending decisions. All underlying training datasets, applicant records, and financial figures are synthetic and intended strictly for demonstration purposes.

---

## 1. Architectural Overview

The backend prediction service layer acts as the unified, decoupled bridge between incoming applicant data (raw user inputs or historical records), the trained machine learning models, and explainability subsystems. It was architected to be modular, robust, and directly consumable by the downstream Streamlit web application.

```
┌────────────────────────────────────────────────────────┐
│               Applicant Input Ingestion                │
│    (Raw Form Dictionary OR Processed Dataset Row)      │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│           Input Validation & Sanitization              │
│  - Checks required fields, types, and sensible bounds  │
│  - Standardizes text categoricals (maps variations)    │
│  - Rejects invalid inputs with clear ValidationError   │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│          Deterministic Financial Calculations          │
│  - Exact formulas from scripts/preprocess.py           │
│  - Estimated New EMI, DTI, LTV, Expense/Income,        │
│    Savings/Income                                      │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│         Feature Vector Assembly & Ordering             │
│  - Encodes 3 categoricals into 15 binary dummy columns │
│  - Preserves exact 39-feature sequence from training   │
│  - Strictly excludes 'applicant_id' and 'default'      │
└───────────────────────────┬────────────────────────────┘
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
┌──────────────────────────┐  ┌──────────────────────────┐
│   Model Inference Mgr    │  │    SHAP Explainability    │
│ - Lazy-loaded & cached   │  │ - scripts/shap_analysis   │
│ - LogReg / RF / XGBoost  │  │ - 39 feature attributions │
│ - Class & P(Default)     │  │ - Top Risk / Protective   │
└─────────────┬────────────┘  └───────────┬──────────────┘
              │                           │
              └─────────────┬─────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│              Structured Output Payload                 │
│  - prediction, probability_of_default, risk_level      │
│  - financial_metrics, affordability_assessment         │
│  - risk_increasing_factors, risk_reducing_factors      │
│  - full shap_values dictionary, prototype disclaimer   │
└────────────────────────────────────────────────────────┘
```

---

## 2. Input Handling & Validation

The service accepts borrower data via two supported pathways:
1. **Raw Applicant Dictionary:** Directly accepts primary applicant fields (e.g., from form inputs in the future Streamlit UI) without requiring the caller to manually compute ratios or one-hot encode categoricals.
2. **Processed Dataset Row:** Accepts an existing 1-row `pandas.DataFrame` or `Series` matching the 39 processed features (e.g., from `data/loan_risk_processed.csv`).

### Validation Rules:
- **Exclusion of Identifiers:** `applicant_id` is strictly excluded from predictive features to prevent data leakage and guarantee valid inference.
- **Missing Value Handling:** Missing required numerical or categorical attributes raise an explicit `ValidationError` listing every missing column.
- **Categorical Standardization:** Categorical values for `employment_type`, `loan_purpose`, and `application_channel` are normalized (trimming whitespace, lowercasing, resolving alias strings like `"self employed"` to `"Self-employed"`) and validated against documented categories.
- **Sensible Numeric Bounds:** Numeric features are checked against sensible ranges (e.g., non-negative debt/income, credit score within [300, 850], loan tenure within [12, 360] months).

---

## 3. Financial Calculations

All financial ratio calculations are deterministic, standalone functions that reuse the exact formulas established in `scripts/preprocess.py`:

| Metric | Formula | Notes |
| :--- | :--- | :--- |
| **Estimated New EMI** | $r = \frac{\text{annual\_rate\_pct}}{100 \times 12}$, $n = \max(\text{tenure}, 1)$<br>$\text{EMI} = P \cdot \frac{r(1+r)^n}{(1+r)^n - 1}$ (or $\frac{P}{n}$ if $r=0$) | Rounded to 2 decimal places. Standard amortized annuity formula. |
| **Debt-to-Income (DTI)** | $\text{DTI} = \frac{\text{existing\_emi} + \text{estimated\_new\_emi}}{\max(\text{monthly\_income}, 1)}$ | Total monthly debt service relative to monthly income. |
| **Loan-to-Value (LTV)** | $\text{LTV} = \frac{\text{loan\_amount}}{\text{collateral\_value}}$ if $\text{collateral} > 0$ else $1.0$ | Unsecured loans (zero collateral) default to $1.0$ modeling convention. |
| **Expense-to-Income** | $\frac{\text{monthly\_expenses}}{\max(\text{monthly\_income}, 1)}$ | Proportion of income consumed by non-debt living expenses. |
| **Savings-to-Income** | $\frac{\text{savings}}{\max(\text{monthly\_income}, 1) \times 12}$ | Liquid savings cushion relative to annual gross income. |

---

## 4. Model Selection & Inference

The service includes an in-memory cached `ModelManager` supporting lazy loading of the three trained models from disk without re-reading on each inference call:

1. **`logistic_regression`** (`models/logistic_regression.pkl`): Scikit-Learn `Pipeline` wrapping `RobustScaler` on continuous features and `LogisticRegression(class_weight='balanced')`.
2. **`random_forest`** (`models/random_forest.pkl`): `RandomForestClassifier(class_weight='balanced')` fitted directly on the 39 processed features.
3. **`xgboost`** (`models/xgboost.pkl`): `XGBClassifier(scale_pos_weight=...)` fitted directly on the 39 processed features.

### Probability Representation:
Model output probabilities represent the model's posterior probability estimate of loan default ($P(\text{Default} = 1)$), bounded in $[0.0, 1.0]$. The complementary probability of non-default ($P(\text{Default} = 0) = 1 - P(\text{Default} = 1)$) is also returned.

---

## 5. Configurable Risk Tiers

Risk tiers are mapped from $P(\text{Default})$ using a configurable `RiskThresholds` dataclass:

- **Low Risk:** $P(\text{Default}) < 0.35$
- **Medium Risk:** $0.35 \le P(\text{Default}) < 0.60$
- **High Risk:** $P(\text{Default}) \ge 0.60$

> [!NOTE]
> These thresholds are illustrative prototype boundaries designed for academic demonstration. They can be dynamically configured by callers without altering model weights or core inference code.

---

## 6. Affordability Assessment

The service calculates comprehensive affordability indicators and provides a prototype cash-flow status:
- **Total Monthly Debt Commitment:** $\text{existing\_emi} + \text{estimated\_new\_emi}$
- **Net Disposable Income:** $\text{monthly\_income} - \text{total\_emi} - \text{monthly\_expenses}$
- **Affordability Status Tiers:**
  - **Favorable:** $\text{DTI} \le 0.45$, $\text{Expense-to-Income} \le 0.60$, and healthy net disposable income ($> 15\%$ of monthly income).
  - **Moderate Strain:** $\text{DTI}$ between $0.45$ and $0.60$, or tight disposable cash cushion.
  - **High Burden / Strained:** $\text{DTI} > 0.60$ or negative net disposable cash flow (commitments exceed income).

*Note: Terminology uses "Affordability Assessment" rather than "Loan Approval" to maintain regulatory and prototype compliance.*

---

## 7. SHAP Explainability Integration

The service integrates directly with the verified [`scripts/shap_analysis.py`](file:///C:/Users/SUDHA/OneDrive/Documents/Attachments/Desktop/LRAS/loan-risk-assessment/scripts/shap_analysis.py) module.

For each individual prediction:
- Evaluates individual feature attributions across all 39 model inputs.
- Filters and ranks the top $k$ **Risk-Increasing Factors** ($\text{SHAP} > 0$, driving default probability higher).
- Filters and ranks the top $k$ **Risk-Reducing Factors** ($\text{SHAP} < 0$, driving default probability lower).
- All explanations are computed dynamically from actual SHAP attributions without hard-coding.

---

## 8. Structured Output Contract

The primary function `predict_loan_risk()` returns a standardized dictionary ready for UI consumption:

```python
{
    "prediction": 1,                               # int: 0 or 1
    "prediction_label": "Default",                 # str: "No Default" | "Default"
    "probability_of_default": 0.5122,              # float: [0.0, 1.0]
    "probability_of_no_default": 0.4878,           # float: [0.0, 1.0]
    "risk_level": "Medium Risk",                   # str: "Low Risk" | "Medium Risk" | "High Risk"
    "risk_tier_details": { ... },                  # dict: thresholds and metadata
    "selected_model": "xgboost",                   # str: model key
    "selected_model_display": "XGBoost",           # str: display name
    "financial_metrics": {                         # dict: all computed financial indicators
        "monthly_income": 38441.0,
        "monthly_expenses": 18824.0,
        "existing_emi": 10596.0,
        "estimated_new_emi": 8958.02,
        "total_emi": 19554.02,
        "net_disposable_income": 63.98,
        "DTI": 0.5087,
        "LTV": 1.0,
        "expense_to_income_ratio": 0.4897,
        "savings_to_income_ratio": 0.8043,
        "affordability_status": "Moderate Strain",
        "affordability_notes": "...",
        "disclaimer": "..."
    },
    "affordability_assessment": { ... },           # dict: streamlined affordability view
    "risk_increasing_factors": [                   # list of top positive SHAP drivers
        {"feature": "credit_score", "value": 627.0, "shap_value": 0.4525},
        {"feature": "DTI", "value": 0.5087, "shap_value": 0.1614}, ...
    ],
    "risk_reducing_factors": [                     # list of top negative SHAP drivers
        {"feature": "outstanding_debt", "value": 152992.0, "shap_value": -0.3627},
        {"feature": "LTV", "value": 1.0, "shap_value": -0.2780}, ...
    ],
    "shap_values": { ... },                        # dict: all 39 feature attributions
    "disclaimer": "This system is designed for loan risk assessment..."
}
```

---

## 9. Verification & Test Summary

The test suite in [`scripts/test_prediction_service.py`](file:///C:/Users/SUDHA/OneDrive/Documents/Attachments/Desktop/LRAS/loan-risk-assessment/scripts/test_prediction_service.py) was executed with 100% pass rate:
- **Test 1:** Standalone EMI, DTI, LTV, Expense/Income, and Savings/Income calculations verified against analytical benchmarks.
- **Test 2:** Input validation verified against missing attributes, invalid categorical values, and out-of-bounds numerics.
- **Test 3:** Feature vector assembly confirmed exact shape $(1, 39)$, exact column ordering, and `applicant_id` exclusion.
- **Test 4:** Risk tier threshold classifications verified.
- **Test 5:** End-to-end inference tested across all three models on historical processed applicants (Default & Non-Default).
- **Test 6:** Manually constructed Prime and Subprime borrower profiles evaluated, demonstrating expected low-risk and high-risk classifications.

---

## 10. Limitations & Disclaimers

1. **Synthetic Nature:** Trained entirely on 15,000 synthetic fictional records; models are not calibrated to real macroeconomic credit cycles.
2. **Simplified Tax & Cash Flow:** EMI and DTI computations assume pre-tax declared monthly income and do not account for jurisdictional tax brackets or variable-rate loans.
3. **Prototype Underwriting:** The affordability assessment and risk tiers are demonstrative heuristics and must not be used for actual credit underwriting or lending decisions.
