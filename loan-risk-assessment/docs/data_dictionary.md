# Data Dictionary

**Project:** Intelligent Loan Risk Assessment & Underwriting Support System  
**Important:** This dictionary describes a **synthetic** educational dataset. Rows are fictional applicants. They do **not** represent real bank customers, and the internally used default-probability mechanism is **not** a real underwriting policy.

Currency amounts are Indian Rupees (₹) stored as numbers without a currency symbol.

---

## Raw attributes (24 columns)

These columns appear in `data/loan_risk_raw.csv`. Do not add, remove, or rename them without an explicit project decision.

| Column | Meaning | Data type | Realistic range / values | Raw or derived | Used as ML feature | Target |
|---|---|---|---|---|---|---|
| `applicant_id` | Unique fictional applicant identifier (`APP000001`, …) | string | Unique ID per generated applicant | Raw | **No** (identity only) | No |
| `age` | Applicant age in years | integer | 21–65 | Raw | Yes | No |
| `dependents` | Number of dependents | integer | 0–5 | Raw | Yes | No |
| `employment_type` | Type of employment | categorical | Salaried, Self-employed, Business, Contract, Unemployed | Raw | Yes (one-hot in processed file) | No |
| `employment_years` | Years in employment / work history | float | 0–40 | Raw | Yes | No |
| `monthly_income` | Stated monthly income (₹) | float | 15,000–500,000 | Raw | Yes | No |
| `monthly_expenses` | Typical monthly living expenses (₹) | float | 5,000–250,000 | Raw | Yes | No |
| `existing_emi` | Current EMI obligations (₹ / month) | float | 0–150,000 | Raw | Yes | No |
| `savings` | Liquid / near-liquid savings (₹) | float | 5,000–5,000,000 | Raw | Yes | No |
| `outstanding_debt` | Total outstanding debt excluding the new loan (₹) | float | 0–5,000,000 | Raw | Yes | No |
| `credit_score` | Bureau-style credit score | integer | 300–850 | Raw | Yes | No |
| `credit_history_years` | Length of credit history (years) | float | 0–40 | Raw | Yes | No |
| `active_loans` | Number of loans currently open | integer | 0–8 | Raw | Yes | No |
| `previous_defaults` | Count of prior default events | integer | 0–3 | Raw | Yes | No |
| `late_payments` | Count of late payments on record | integer | 0–15 | Raw | Yes | No |
| `loan_amount` | Requested loan amount (₹) | float | 50,000–10,000,000 | Raw | Yes | No |
| `loan_tenure_months` | Requested tenure in months | integer | 12–360 | Raw | Yes | No |
| `interest_rate` | Annual interest rate (percent) | float | 6–24 | Raw | Yes | No |
| `loan_purpose` | Stated purpose of the loan | categorical | Home, Education, Vehicle, Personal, Business, Medical, Other | Raw | Yes (one-hot in processed file) | No |
| `collateral_value` | Stated collateral value (₹); 0 means unsecured | float | 0–15,000,000 | Raw | Yes | No |
| `bank_balance_avg` | Average bank balance (₹) | float | 1,000–3,000,000 | Raw | Yes | No |
| `credit_utilization` | Credit utilization (percent) | float | 0–100 | Raw | Yes | No |
| `application_channel` | How the application was submitted | categorical | Online, Branch, Agent | Raw | Yes (one-hot in processed file) | No |
| `default` | Binary default outcome | integer | 0 = No Default, 1 = Default | Raw (label) | **No** (this is the target) | **Yes** |

The raw file may contain missing values, inconsistent categorical spellings, duplicate rows, and in-range outliers on purpose, so preprocessing can be demonstrated.

There is **no** `risk_level` column. Risk bands (Low / Medium / High) are intended to be applied later from a predicted probability of default, not stored in this dataset.

---

## Derived attributes (processed dataset only)

Created in `scripts/preprocess.py`. These are financial ratios, not invented “scores.”

| Column | Meaning | Data type | Construction | Raw or derived | Used as ML feature | Target |
|---|---|---|---|---|---|---|
| `estimated_new_emi` | EMI of the requested loan | float | Standard reducing-balance EMI using `loan_amount`, monthly rate = `interest_rate/12/100`, `n` = `loan_tenure_months` | Derived | Yes | No |
| `DTI` | Debt-to-income style burden | float | `(existing_emi + estimated_new_emi) / monthly_income` | Derived | Yes | No |
| `LTV` | Loan-to-value ratio | float | `loan_amount / collateral_value` when collateral > 0; **1.0** when collateral is 0 (keeps the value finite for unsecured loans) | Derived | Yes | No |
| `expense_to_income_ratio` | Expense burden | float | `monthly_expenses / monthly_income` | Derived | Yes | No |
| `savings_to_income_ratio` | Savings relative to annual income | float | `savings / (monthly_income * 12)` | Derived | Yes | No |

---

## Encoded categorical columns (processed dataset only)

After standardization, categoricals are one-hot encoded with `sklearn.preprocessing.OneHotEncoder` and fixed category orders:

- `employment_type_Salaried`, `employment_type_Self-employed`, `employment_type_Business`, `employment_type_Contract`, `employment_type_Unemployed`
- `loan_purpose_Home`, `loan_purpose_Education`, `loan_purpose_Vehicle`, `loan_purpose_Personal`, `loan_purpose_Business`, `loan_purpose_Medical`, `loan_purpose_Other`
- `application_channel_Online`, `application_channel_Branch`, `application_channel_Agent`

Original string categorical columns are not kept in the processed file.

---

## Modeling notes

- Train models on processed columns **excluding** `applicant_id` and `default`.
- Predict `default` (0/1). The model’s positive-class probability is the estimated **probability of default**.
- Convert that probability into Low / Medium / High risk only in a later application layer.
- Class imbalance is expected (about 10–25% defaults) and should be handled during modeling, not by inventing extra raw columns.
