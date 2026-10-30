# Dataset Audit Report

## 1. Dataset Overview

- **Project Title:** Intelligent Loan Risk Assessment & Underwriting Support System (LRAS)
- **Dataset Nature:** Synthetic educational borrower dataset (reproducible seed 42)
- **Raw File:** `data/loan_risk_raw.csv` (15,105 rows, 24 columns, 15,000 unique applicants, 105 intentional duplicate rows)
- **Processed File:** `data/loan_risk_processed.csv` (15,000 rows, 41 columns, 15,000 unique applicants, 0 duplicate rows, 0 missing values)
- **Audit Date:** September 25, 2026

The objective of this audit is to evaluate data quality, feature distributions, target distribution, potential data leakage, feature relationships, and overall model readiness across both the RAW and PROCESSED datasets without modifying any existing data or code files.

---

## 2. Raw Dataset Quality

The raw dataset (`data/loan_risk_raw.csv`) contains exactly the 24 approved attributes in the expected order:

1. `applicant_id` (string, primary key, 100% complete)
2. `age` (integer, 21–65 years)
3. `dependents` (integer, 0–5)
4. `employment_type` (categorical: Salaried, Self-employed, Business, Contract, Unemployed with minor casing variants)
5. `employment_years` (float, 0–40 years)
6. `monthly_income` (float, ₹15,000–₹492,856)
7. `monthly_expenses` (float, ₹5,000–₹234,816)
8. `existing_emi` (float, ₹0–₹65,660)
9. `savings` (float, ₹5,000–₹5,000,000)
10. `outstanding_debt` (float, ₹499–₹5,000,000)
11. `credit_score` (integer, 410–850)
12. `credit_history_years` (float, 0.1–40 years)
13. `active_loans` (integer, 0–8)
14. `previous_defaults` (integer, 0–3)
15. `late_payments` (integer, 0–15)
16. `loan_amount` (float, ₹50,000–₹10,000,000)
17. `loan_tenure_months` (integer, 12–360 months)
18. `interest_rate` (float, 6.0%–19.55%)
19. `loan_purpose` (categorical: Home, Education, Vehicle, Personal, Business, Medical, Other)
20. `collateral_value` (float, ₹0–₹15,000,000)
21. `bank_balance_avg` (float, ₹13,078–₹2,550,072)
22. `credit_utilization` (float, 0.0%–100.0%)
23. `application_channel` (categorical: Online, Branch, Agent)
24. `default` (integer target, 0 or 1, 100% complete)

### Quality Checks on Raw Data:
- **Total Rows:** 15,105
- **Total Columns:** 24
- **Unique Applicant IDs:** 15,000
- **Duplicate Rows:** 105 exact row duplicates (0.69% duplicate rate, intentional noise)
- **Missing Values:** 3,743 total missing cells across 8 numeric features:
  - `employment_years`: 337 missing (2.23%)
  - `monthly_income`: 507 missing (3.36%)
  - `monthly_expenses`: 451 missing (2.99%)
  - `existing_emi`: 426 missing (2.82%)
  - `savings`: 627 missing (4.15%)
  - `outstanding_debt`: 377 missing (2.50%)
  - `credit_score`: 483 missing (3.20%)
  - `collateral_value`: 535 missing (3.54%)
- **Categorical Inconsistencies:** Intentionally injected casing and space variations present (e.g., `salaried`, `self employed`, `ONLINE`).
- **Unexpected / Missing Columns:** None. Exactly matches the required 24 attributes.

---

## 3. Processed Dataset Quality

The processed dataset (`data/loan_risk_processed.csv`) contains 15,000 cleaned rows and 41 engineered/encoded columns.

### Quality Verification Results:
- **Total Rows:** 15,000 (Exactly 1 record per unique applicant ID)
- **Total Columns:** 41
- **Missing Values:** 0 (100% complete; numeric missing values imputed using feature medians)
- **Duplicate Rows:** 0 (105 duplicate rows removed using `keep='first'`)
- **Duplicate Applicant IDs:** 0
- **Infinite / NaN Values:** 0 (Verified finite across all 41 columns)
- **Categorical Encoding:** Standardized and 1-hot encoded into 15 binary dummy columns across `employment_type` (5), `loan_purpose` (7), and `application_channel` (3).
- **Engineered Features:** 5 derived features successfully created and validated:
  - `estimated_new_emi`: Standard reducing-balance EMI calculation
  - `DTI`: Debt-to-Income ratio `(existing_emi + estimated_new_emi) / monthly_income`
  - `LTV`: Loan-to-Value ratio `loan_amount / collateral_value` (set to 1.0 for unsecured loans where collateral = 0)
  - `expense_to_income_ratio`: `monthly_expenses / monthly_income`
  - `savings_to_income_ratio`: `savings / (monthly_income * 12)`
- **Identity Column:** `applicant_id` retained for identity verification (MUST NOT be used as a predictor model feature).

---

## 4. Feature Distribution Findings

The empirical summary statistics for key numerical features in the processed dataset are as follows:

| Feature | Min | Q25 (25%) | Median | Mean | Q75 (75%) | Max | Std Dev | Skewness |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `age` | 21.00 | 31.00 | 38.00 | 38.11 | 45.00 | 65.00 | 10.01 | 0.23 |
| `dependents` | 0.00 | 0.00 | 1.00 | 1.55 | 2.00 | 5.00 | 1.43 | 0.75 |
| `employment_years` | 0.00 | 3.20 | 7.00 | 8.32 | 11.80 | 40.00 | 6.40 | 1.12 |
| `monthly_income` | 15,000.00 | 37,548.75 | 50,130.50 | 53,254.22 | 64,637.75 | 166,755.20 | 24,530.40 | 1.31 |
| `monthly_expenses` | 5,644.97 | 18,196.25 | 26,129.00 | 29,090.88 | 36,611.00 | 100,788.40 | 15,661.04 | 1.40 |
| `existing_emi` | 0.00 | 0.00 | 0.00 | 2,885.51 | 4,643.75 | 24,512.14 | 4,750.44 | 1.99 |
| `savings` | 5,000.00 | 162,040.50 | 371,029.00 | 617,889.44 | 761,180.75 | 5,000,000.00 | 761,541.87 | 2.86 |
| `outstanding_debt` | 13,619.67 | 171,954.50 | 342,253.50 | 520,308.07 | 663,051.50 | 3,322,949.00 | 536,729.20 | 2.25 |
| `credit_score` | 410.00 | 654.00 | 706.00 | 704.83 | 758.00 | 850.00 | 77.04 | -0.17 |
| `credit_history_years` | 0.10 | 5.20 | 9.70 | 10.91 | 15.10 | 40.00 | 7.22 | 0.87 |
| `active_loans` | 0.00 | 0.00 | 1.00 | 1.22 | 1.00 | 8.00 | 1.36 | 1.91 |
| `previous_defaults` | 0.00 | 0.00 | 0.00 | 0.21 | 0.00 | 3.00 | 0.47 | 2.27 |
| `late_payments` | 0.00 | 1.00 | 3.00 | 3.07 | 5.00 | 15.00 | 2.57 | 0.97 |
| `loan_amount` | 51,292.04 | 263,995.25 | 483,874.00 | 783,350.80 | 926,600.00 | 6,438,501.00 | 898,351.42 | 3.11 |
| `loan_tenure_months` | 12.00 | 34.00 | 49.00 | 71.03 | 76.00 | 360.00 | 61.95 | 1.89 |
| `interest_rate` | 6.00 | 11.40 | 12.75 | 12.67 | 14.04 | 19.55 | 1.95 | -0.18 |
| `collateral_value` | 0.00 | 0.00 | 107,904.00 | 567,052.35 | 683,282.75 | 6,814,216.00 | 1,028,834.71 | 3.14 |
| `bank_balance_avg` | 24,268.98 | 131,762.50 | 231,458.50 | 320,701.38 | 405,040.50 | 1,871,339.00 | 295,215.82 | 2.35 |
| `credit_utilization` | 0.00 | 28.30 | 38.20 | 38.18 | 48.10 | 100.00 | 14.53 | 0.05 |
| `estimated_new_emi` | 1,068.80 | 7,680.11 | 13,021.75 | 17,582.09 | 21,912.01 | 534,631.18 | 17,352.39 | 6.98 |
| `DTI` | 0.0263 | 0.2150 | 0.3240 | 0.3893 | 0.4791 | 12.4336 | 0.3226 | 10.27 |
| `LTV` | 0.4753 | 1.0000 | 1.0000 | 7.2036 | 2.8376 | 16,446.61 | 149.41 | 93.06 |
| `expense_to_income_ratio`| 0.0339 | 0.4339 | 0.5393 | 0.5523 | 0.6517 | 2.0105 | 0.1687 | 1.08 |
| `savings_to_income_ratio`| 0.0042 | 0.3169 | 0.6188 | 0.8763 | 1.1216 | 27.5522 | 0.9438 | 5.35 |

### Insights on Distributions:
1. **Bounded Features:** `age`, `credit_score`, `interest_rate`, `credit_utilization`, `dependents`, and `active_loans` lie within natural physiological/financial boundaries.
2. **Right-Skewed Rupee Amounts:** `monthly_income`, `savings`, `loan_amount`, and `outstanding_debt` exhibit natural right-skewness. Winsorization at the 0.5th and 99.5th percentiles in preprocessing successfully contained extreme tail noise while preserving variance.
3. **Heavy-Tailed Derived Ratios:** `LTV` (max 16,446.61) and `DTI` (max 12.43) contain extreme positive values caused by low non-zero collateral denominators or high EMI relative to income.

---

## 5. Target Distribution

The target variable `default` represents loan default outcome (0 = No Default, 1 = Default).

### Raw Dataset Target Distribution:
- **No Default (0):** 12,734 records (**84.30%**)
- **Default (1):** 2,371 records (**15.70%**)
- **Total:** 15,105 rows

### Processed Dataset Target Distribution:
- **No Default (0):** 12,649 records (**84.33%**)
- **Default (1):** 2,351 records (**15.67%**)
- **Total:** 15,000 rows

### Imbalance Assessment:
- The class imbalance ratio is **5.38 : 1** (84.33% non-default vs 15.67% default).
- This aligns with standard credit risk benchmarking where default rates naturally range between 10% and 25%.
- **Conclusion:** Target distribution is realistic, well-balanced for credit underwriting ML practice, and suitable for binary classification.

---

## 6. Data Leakage Analysis

A rigorous data leakage audit was conducted to verify whether any feature directly or indirectly leaks the ground truth target `default`.

### Empirical Predictor Correlations & ROC-AUC Scores against `default`:

| Feature | Pearson Correlation (r) | Spearman Correlation (r) | ROC-AUC Score | Mean (Default=0) | Mean (Default=1) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `credit_score` | -0.2949 | -0.2840 | 0.7255 | 714.63 | 652.13 |
| `late_payments` | +0.2509 | +0.2378 | 0.6871 | 2.80 | 4.57 |
| `credit_utilization` | +0.2042 | +0.1993 | 0.6582 | 36.90% | 45.06% |
| `interest_rate` | +0.1677 | +0.1641 | 0.6303 | 12.53% | 13.43% |
| `previous_defaults` | +0.2407 | +0.2213 | 0.6193 | 0.16 | 0.48 |
| `savings` | -0.0929 | -0.1327 | 0.6054 | ₹648,373.87 | ₹453,875.17 |
| `bank_balance_avg` | -0.1008 | -0.1288 | 0.6023 | ₹333,533.49 | ₹251,661.26 |
| `employment_years` | -0.1225 | -0.1285 | 0.6020 | 8.66 | 6.50 |
| `savings_to_income_ratio` | -0.0921 | -0.1273 | 0.6011 | 0.91 | 0.67 |
| `active_loans` | +0.1537 | +0.1250 | 0.5927 | 1.13 | 1.71 |
| `LTV` | +0.0047 | +0.1093 | 0.5857 | 6.90 | 8.84 |
| `existing_emi` | +0.0792 | +0.1130 | 0.5801 | ₹2,723.26 | ₹3,758.48 |
| `outstanding_debt` | +0.1176 | +0.0958 | 0.5761 | ₹493,089.77 | ₹666,749.73 |
| `monthly_income` | -0.0692 | -0.0829 | 0.5658 | ₹53,986.28 | ₹49,315.53 |
| `DTI` | +0.0679 | +0.0804 | 0.5639 | 0.38 | 0.44 |
| `unemployed_flag` | +0.1102 | +0.1102 | 0.5405 | 0.06 | 0.15 |

### Key Leakage Findings:
1. **No Single Dominant Feature:** The single highest ROC-AUC is `credit_score` at **0.7255** (Pearson r = **-0.2949**). No individual feature exhibits an AUC > 0.73 or absolute correlation > 0.30.
2. **Probabilistic Data Generation Mechanism:** Inspection of `scripts/generate_dataset.py` confirms that `default` is determined via a multi-factor logistic probability function combined with additive Gaussian noise (`rng.normal(0, 0.45)`) and random Bernoulli sampling (`rng.random < pd_hat`).
3. **No Outcome Fields:** There are no post-underwriting fields (such as collections, recovery amounts, or actual repayment records) included in the raw or processed attributes.
4. **Legitimate Predictive Signal vs Leakage:** The observed relationships between risk factors (`credit_score`, `previous_defaults`, `late_payments`, `utilization`) and `default` represent genuine financial domain signals rather than target leakage.

---

## 7. Feature Relationship Analysis

The synthetic data generation process preserves domain-logical relationships among borrower financial attributes:

### Feature-to-Feature Logical Correlations:
- **`monthly_income` vs `monthly_expenses`:** r = **+0.7962** (Higher income households naturally incur higher living expenses).
- **`monthly_income` vs `savings`:** r = **+0.5134** (Savings scale positively with monthly income).
- **`monthly_income` vs `loan_amount`:** r = **+0.3895** (Borrowers with higher incomes qualify for larger loan amounts).
- **`loan_amount` vs `collateral_value`:** r = **+0.8379** (Secured loans require proportional collateral backing).
- **`employment_years` vs `age`:** r = **+0.6276** (Older applicants have higher work tenure).
- **`credit_history_years` vs `age`:** r = **+0.7426** (Length of credit history expands with age).
- **`credit_score` vs `late_payments`:** r = **-0.6889** (Past late payments significantly suppress credit scores).
- **`credit_score` vs `credit_utilization`:** r = **-0.5476** (High credit line utilization reduces credit score).

### Feature-to-Target Stratification:
- **Default Rate by Credit Score Tier:**
  - 300–580 (Poor): **44.80%** default rate (895 applicants)
  - 580–670 (Fair): **24.89%** default rate (3,922 applicants)
  - 670–740 (Good): **13.80%** default rate (5,305 applicants)
  - 740–800 (Very Good): **6.51%** default rate (3,117 applicants)
  - 800–850 (Exceptional): **2.21%** default rate (1,761 applicants)
- **Default Rate by Prior Defaults:**
  - 0 prior defaults: **11.93%** default rate
  - 1 prior default: **28.26%** default rate
  - 2 prior defaults: **55.38%** default rate
  - 3 prior defaults: **75.00%** default rate
- **Default Rate by Late Payment Band:**
  - 0 late payments: **4.63%** default rate
  - 1–2 late payments: **10.32%** default rate
  - 3–5 late payments: **18.55%** default rate
  - 6–10 late payments: **29.57%** default rate
  - 11–15 late payments: **47.24%** default rate

---

## 8. Problems Found

| # | Problem | Evidence | Severity | Recommendation |
| :--- | :--- | :--- | :--- | :--- |
| 1 | **Heavy Right-Skewness & Extreme Values in Ratio Features (`LTV`, `DTI`, `estimated_new_emi`)** | `LTV` max = 16,446.61 (skewness = 93.06); `DTI` max = 12.43 (skewness = 10.27); `estimated_new_emi` max = ₹534,631.18 (skewness = 6.98). | **MEDIUM** | Tree-based models (Random Forest, XGBoost) handle unscaled skewed inputs seamlessly. For linear models (Logistic Regression), apply log transformation (`np.log1p`) or robust scaling (`RobustScaler`) during model pipeline setup. |
| 2 | **Class Imbalance in Target Variable (`default`)** | Negative class (0) = 84.33% (12,649); Positive class (1) = 15.67% (2,351). Imbalance ratio = 5.38 : 1. | **MEDIUM** | Evaluate models using PR-AUC, ROC-AUC, F1-Score, and Recall rather than standard Accuracy. Set `class_weight='balanced'` in Logistic Regression / Random Forest or `scale_pos_weight` in XGBoost. |
| 3 | **Data Quality Issues in RAW CSV (Missingness, Duplicates, Categoricals)** | 3,743 missing cells in raw data; 105 duplicate rows; non-canonical categorical strings (`salaried`, `self employed`). | **LOW** | Intended design for preprocessing pipeline demonstration. The processed dataset (`loan_risk_processed.csv`) has already resolved all duplicates, missing values, and formatting issues. Use `loan_risk_processed.csv` directly for ML. |
| 4 | **Presence of Non-Predictive Primary Key (`applicant_id`)** | Column `applicant_id` contains unique identifiers (`APP000001`–`APP015000`) in both raw and processed files. | **LOW** | Explicitly drop or set `applicant_id` as the dataframe index prior to feeding feature matrices into ML model fit functions. |

---

## 9. Model Readiness

The processed dataset `data/loan_risk_processed.csv` was evaluated for readiness against the candidate ML algorithms:

1. **Logistic Regression:**
   - **Status:** READY (Requires Standard / Robust Feature Scaling in ML pipeline).
   - **Notes:** Numerical features vary in scale (e.g. `age` 21–65 vs `savings` up to ₹5M). Standard scaling (`StandardScaler` or `RobustScaler`) should be fit on the training split before training Logistic Regression.
2. **Random Forest Classifier:**
   - **Status:** READY (Directly consumable).
   - **Notes:** Tree-based partitioning is invariant to monotonic feature scaling and handles skewed features (such as `LTV` and `DTI`) naturally without scaling.
3. **XGBoost Classifier:**
   - **Status:** READY (Directly consumable).
   - **Notes:** Gradient boosted decision trees excel on structured tabular data of this structure. Setting `scale_pos_weight = 5.38` during model instantiation will address class imbalance.

---

## 10. Final Conclusion

READY FOR MODELING
