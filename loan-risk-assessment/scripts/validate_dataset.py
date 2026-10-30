"""
Validate the synthetic loan-risk datasets against the academic data contract.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_PATH = DATA_DIR / "loan_risk_raw.csv"
PROCESSED_PATH = DATA_DIR / "loan_risk_processed.csv"

EXPECTED_RAW_COLUMNS = [
    "applicant_id",
    "age",
    "dependents",
    "employment_type",
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
    "loan_purpose",
    "collateral_value",
    "bank_balance_avg",
    "credit_utilization",
    "application_channel",
    "default",
]
EXPECTED_UNIQUE_GENERATED = 15_000
DERIVED_COLS = ["DTI", "LTV", "expense_to_income_ratio", "savings_to_income_ratio"]


def check(condition: bool, message: str, failures: list[str], passes: list[str]) -> None:
    if condition:
        passes.append(message)
    else:
        failures.append(message)


def validate() -> int:
    failures: list[str] = []
    passes: list[str] = []

    raw = pd.read_csv(RAW_PATH)
    processed = pd.read_csv(PROCESSED_PATH)

    check(
        list(raw.columns) == EXPECTED_RAW_COLUMNS,
        "Raw dataset has the 24 approved columns in the specified order",
        failures,
        passes,
    )
    check(
        raw["applicant_id"].nunique() == EXPECTED_UNIQUE_GENERATED,
        f"Exactly {EXPECTED_UNIQUE_GENERATED} unique applicant_id values exist before/after duplicate copies "
        f"(found {raw['applicant_id'].nunique()})",
        failures,
        passes,
    )
    check(
        len(raw) > EXPECTED_UNIQUE_GENERATED,
        f"Raw file includes intentional extra duplicate rows ({len(raw)} rows vs "
        f"{EXPECTED_UNIQUE_GENERATED} unique IDs)",
        failures,
        passes,
    )
    extra = len(raw) - raw["applicant_id"].nunique()
    check(
        75 <= extra <= 150,
        f"Duplicate extras are about 0.5-1% of 15000 (extra rows={extra})",
        failures,
        passes,
    )
    check(
        int(raw["applicant_id"].isna().sum()) == 0 and int(raw["default"].isna().sum()) == 0,
        "No missing applicant_id or default in the raw file",
        failures,
        passes,
    )

    # Range checks on processed data (cleaned, complete).
    age_ok = processed["age"].between(21, 65).all()
    check(bool(age_ok), "Processed age is within 21-65", failures, passes)

    score_ok = processed["credit_score"].between(300, 850).all()
    check(bool(score_ok), "Processed credit_score is within 300–850", failures, passes)

    income_ok = processed["monthly_income"].between(15_000, 5_00_000).all()
    check(bool(income_ok), "Processed monthly_income is within 15000–500000", failures, passes)

    loan_ok = processed["loan_amount"].between(50_000, 1_00_00_000).all()
    check(bool(loan_ok), "Processed loan_amount is within 50000–10000000", failures, passes)

    tenure_ok = processed["loan_tenure_months"].between(12, 360).all()
    check(bool(tenure_ok), "Processed loan_tenure_months is within 12–360", failures, passes)

    rate_ok = processed["interest_rate"].between(6, 24).all()
    check(bool(rate_ok), "Processed interest_rate is within 6–24", failures, passes)

    default_ok = set(processed["default"].unique()).issubset({0, 1})
    check(bool(default_ok), "Processed default contains only 0 and 1", failures, passes)

    check(
        int(processed.isna().sum().sum()) == 0,
        "Processed dataset has no missing values",
        failures,
        passes,
    )
    check(
        processed["applicant_id"].is_unique and int(processed.duplicated().sum()) == 0,
        "Processed dataset has unique applicant records (no duplicate IDs or rows)",
        failures,
        passes,
    )

    numeric_processed = processed.select_dtypes(include=[np.number])
    check(
        not bool(np.isinf(numeric_processed.to_numpy()).any()),
        "Processed dataset has no infinite values",
        failures,
        passes,
    )

    for col in DERIVED_COLS:
        check(col in processed.columns, f"Derived column present: {col}", failures, passes)
        if col in processed.columns:
            series = processed[col]
            finite = np.isfinite(series.to_numpy())
            check(bool(finite.all()), f"{col} has finite numeric values", failures, passes)
            check(bool(series.notna().all()), f"{col} has no missing values", failures, passes)

    check("estimated_new_emi" in processed.columns, "estimated_new_emi is present", failures, passes)
    check("applicant_id" in processed.columns, "applicant_id retained for identity, not as a model feature", failures, passes)
    check("default" in processed.columns, "Target column default is retained", failures, passes)

    dummy_prefixes = ("employment_type_", "loan_purpose_", "application_channel_")
    n_dummies = sum(c.startswith(dummy_prefixes) for c in processed.columns)
    check(n_dummies >= 3 + 7 + 3 - 0, f"One-hot encoded categoricals are present ({n_dummies} dummy columns)", failures, passes)

    rate = float(processed["default"].mean())
    check(
        0.10 <= rate <= 0.25,
        f"Processed default rate is in the 10-25% academic band ({rate:.4f})",
        failures,
        passes,
    )

    print("=" * 60)
    print("DATASET VALIDATION SUMMARY")
    print("Synthetic educational data — not real borrowers.")
    print("=" * 60)
    print(f"Raw shape:        {raw.shape}")
    print(f"Processed shape:  {processed.shape}")
    print(f"Unique raw IDs:   {raw['applicant_id'].nunique()}")
    print(f"Processed IDs:    {processed['applicant_id'].nunique()}")
    print(f"Default rate:     {rate:.4f}")
    print()
    print(f"PASSED ({len(passes)})")
    for item in passes:
        print(f"  [PASS] {item}")
    print()
    if failures:
        print(f"FAILED ({len(failures)})")
        for item in failures:
            print(f"  [FAIL] {item}")
        print()
        print("RESULT: VALIDATION FAILED")
        return 1

    print("RESULT: VALIDATION PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(validate())
