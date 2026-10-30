"""
Generate a synthetic borrower dataset for the educational project:
Intelligent Loan Risk Assessment & Underwriting Support System.

This data is entirely fictional. It does not represent real bank customers
or any real underwriting policy. It is designed for college-level ML
practice: preprocessing, feature engineering, and classification.

Reproducible with random seed 42.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_RECORDS = 15_000
RAW_COLUMNS = [
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

EMPLOYMENT_TYPES = [
    "Salaried",
    "Self-employed",
    "Business",
    "Contract",
    "Unemployed",
]
LOAN_PURPOSES = [
    "Home",
    "Education",
    "Vehicle",
    "Personal",
    "Business",
    "Medical",
    "Other",
]
APPLICATION_CHANNELS = ["Online", "Branch", "Agent"]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def clip_int(values: np.ndarray, low: int, high: int) -> np.ndarray:
    return np.clip(np.rint(values), low, high).astype(int)


def clip_float(values: np.ndarray, low: float, high: float, decimals: int = 2) -> np.ndarray:
    return np.round(np.clip(values, low, high), decimals)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def estimate_emi(principal: np.ndarray, annual_rate_pct: np.ndarray, tenure_months: np.ndarray) -> np.ndarray:
    """Standard reducing-balance EMI. Used only for internal risk, not stored in raw data."""
    r = (annual_rate_pct / 100.0) / 12.0
    n = np.maximum(tenure_months.astype(float), 1.0)
    p = principal.astype(float)
    # When rate is ~0, EMI is principal / n
    emi = np.where(
        r <= 1e-12,
        p / n,
        p * r * np.power(1.0 + r, n) / (np.power(1.0 + r, n) - 1.0),
    )
    return emi


def generate_base_population(rng: np.random.Generator, n: int) -> dict[str, np.ndarray]:
    """Create correlated borrower attributes. Columns are not drawn independently."""
    age = clip_int(rng.normal(38.0, 10.5, n), 21, 65)

    # Dependents tend to rise through mid-life, then slightly decline.
    dep_mean = np.clip((age - 22) / 10.0, 0.2, 2.8)
    dependents = clip_int(rng.poisson(dep_mean), 0, 5)

    # Employment mix by age: younger applicants more contract/unemployed.
    emp_probs = np.zeros((n, len(EMPLOYMENT_TYPES)))
    for i, a in enumerate(age):
        if a < 25:
            emp_probs[i] = [0.42, 0.10, 0.05, 0.28, 0.15]
        elif a < 35:
            emp_probs[i] = [0.52, 0.16, 0.10, 0.14, 0.08]
        elif a < 50:
            emp_probs[i] = [0.50, 0.20, 0.16, 0.08, 0.06]
        else:
            emp_probs[i] = [0.48, 0.18, 0.18, 0.08, 0.08]
    employment_type = np.array(
        [EMPLOYMENT_TYPES[rng.choice(len(EMPLOYMENT_TYPES), p=emp_probs[i])] for i in range(n)]
    )

    max_years = np.maximum(age - 18, 0)
    years_draw = rng.beta(2.2, 2.8, n) * max_years
    employment_years = np.where(
        employment_type == "Unemployed",
        rng.integers(0, 3, n) + rng.random(n) * 0.5,
        years_draw,
    )
    employment_years = clip_float(np.minimum(employment_years, max_years), 0, 40, 1)

    # Income: type, tenure, and age jointly shape earnings.
    type_base = {
        "Salaried": 42_000,
        "Self-employed": 38_000,
        "Business": 55_000,
        "Contract": 32_000,
        "Unemployed": 18_000,
    }
    type_spread = {
        "Salaried": 0.55,
        "Self-employed": 0.72,
        "Business": 0.90,
        "Contract": 0.50,
        "Unemployed": 0.28,
    }
    base = np.array([type_base[t] for t in employment_type], dtype=float)
    spread = np.array([type_spread[t] for t in employment_type], dtype=float)
    tenure_mult = 1.0 + 0.035 * np.minimum(employment_years, 25)
    age_mult = np.where(age < 25, 0.75, np.where(age > 55, 0.92, 1.0))
    log_income = np.log(base * tenure_mult * age_mult) + rng.normal(0, spread * 0.45, n)
    monthly_income = clip_float(np.exp(log_income), 15_000, 5_00_000, 0)

    # Expenses usually consume a share of income; some households overspend.
    expense_ratio = rng.beta(3.2, 3.8, n) * 0.75 + 0.18
    overspend = rng.random(n) < 0.12
    expense_ratio = np.where(overspend, expense_ratio + rng.uniform(0.10, 0.35, n), expense_ratio)
    monthly_expenses = clip_float(monthly_income * expense_ratio, 5_000, 2_50_000, 0)

    # Existing EMI burden is higher when income and active credit are higher.
    has_emi = rng.random(n) < np.clip(0.25 + 0.01 * (age - 21), 0.25, 0.72)
    emi_ratio = rng.beta(2.0, 5.0, n) * 0.45
    existing_emi = np.where(has_emi, monthly_income * emi_ratio, 0.0)
    existing_emi = clip_float(existing_emi, 0, 1_50_000, 0)

    # Savings grow with income, age, and leftover cash; unemployed/overspenders save less.
    leftover = np.maximum(monthly_income - monthly_expenses - existing_emi, 500)
    savings = leftover * (8 + employment_years * 1.8) * rng.lognormal(0.0, 0.55, n)
    savings = np.where(employment_type == "Unemployed", savings * 0.45, savings)
    savings = clip_float(savings, 5_000, 50_00_000, 0)

    debt_ratio_annual = rng.beta(1.8, 4.5, n) * 4.5
    outstanding_debt = monthly_income * 12 * debt_ratio_annual * np.where(has_emi, 1.0, 0.35)
    outstanding_debt = clip_float(outstanding_debt, 0, 50_00_000, 0)

    credit_history_years = clip_float(
        np.minimum(max_years, rng.beta(2.4, 2.0, n) * max_years),
        0,
        40,
        1,
    )

    # Latent repayment discipline drives score, utilization, lates, and prior defaults.
    discipline = rng.normal(0.0, 1.0, n)
    discipline += 0.025 * employment_years
    discipline += 0.02 * credit_history_years
    discipline -= 0.35 * (employment_type == "Unemployed").astype(float)
    discipline -= 0.20 * overspend.astype(float)

    credit_score = 680 + 70 * discipline + rng.normal(0, 35, n)
    credit_score = clip_int(credit_score, 300, 850)

    prev_lambda = np.clip(0.08 + 0.55 * (1.0 - (credit_score - 300) / 550.0), 0.02, 1.6)
    previous_defaults = clip_int(rng.poisson(prev_lambda), 0, 3)
    # Stronger scores almost never have multiple prior defaults.
    previous_defaults = np.where(credit_score >= 750, np.minimum(previous_defaults, 1), previous_defaults)
    previous_defaults = np.where(credit_score >= 800, 0, previous_defaults)

    late_lambda = np.clip(1.2 + 8.0 * (1.0 - (credit_score - 300) / 550.0) - 0.8 * discipline, 0.1, 10)
    late_payments = clip_int(rng.poisson(late_lambda), 0, 15)

    util_mean = np.clip(0.62 - 0.00055 * (credit_score - 300) - 0.04 * discipline, 0.08, 0.92)
    credit_utilization = clip_float(rng.normal(util_mean, 0.12, n) * 100.0, 0, 100, 1)

    active_loans = clip_int(
        rng.poisson(np.clip(0.6 + existing_emi / 18_000 + outstanding_debt / 8_00_000, 0.2, 5.5)),
        0,
        8,
    )
    active_loans = np.where(existing_emi == 0, np.minimum(active_loans, 1), active_loans)

    purpose_p = np.array([0.18, 0.12, 0.16, 0.22, 0.14, 0.10, 0.08])
    loan_purpose = rng.choice(LOAN_PURPOSES, size=n, p=purpose_p)

    # Requested amount scales with income and purpose; business/home skew larger.
    purpose_mult = {
        "Home": (28, 0.55),
        "Education": (10, 0.45),
        "Vehicle": (12, 0.40),
        "Personal": (6, 0.50),
        "Business": (18, 0.70),
        "Medical": (5, 0.55),
        "Other": (7, 0.50),
    }
    p_lo = np.array([purpose_mult[p][0] for p in loan_purpose])
    p_sd = np.array([purpose_mult[p][1] for p in loan_purpose])
    loan_amount = monthly_income * p_lo * rng.lognormal(0.0, p_sd)
    # Higher scores and savings modestly support larger sanctioned amounts.
    loan_amount *= 0.85 + 0.00025 * (credit_score - 300) + 0.00000002 * savings
    loan_amount = clip_float(loan_amount, 50_000, 1_00_00_000, 0)

    tenure_by_purpose = {
        "Home": (180, 60, 84, 360),
        "Education": (48, 18, 12, 120),
        "Vehicle": (48, 18, 12, 84),
        "Personal": (36, 12, 12, 72),
        "Business": (60, 24, 12, 120),
        "Medical": (24, 12, 12, 60),
        "Other": (36, 18, 12, 84),
    }
    tenure = np.zeros(n)
    for i, p in enumerate(loan_purpose):
        mean, sd, lo, hi = tenure_by_purpose[p]
        tenure[i] = np.clip(rng.normal(mean, sd), lo, hi)
    # Larger loans tend toward longer tenures.
    tenure += 12 * (loan_amount / 25_00_000)
    loan_tenure_months = clip_int(tenure, 12, 360)

    # Pricing: weaker credit and unsecured-style purposes get higher rates.
    rate = 14.5 - 0.012 * (credit_score - 500)
    rate += np.select(
        [
            loan_purpose == "Personal",
            loan_purpose == "Medical",
            loan_purpose == "Business",
            loan_purpose == "Home",
        ],
        [2.2, 1.6, 1.1, -1.8],
        default=0.4,
    )
    rate += rng.normal(0, 1.1, n)
    interest_rate = clip_float(rate, 6.0, 24.0, 2)

    # Collateral is common for home/vehicle/business and rare for personal/medical.
    coll_frac = np.select(
        [
            loan_purpose == "Home",
            loan_purpose == "Vehicle",
            loan_purpose == "Business",
            loan_purpose == "Education",
        ],
        [
            rng.uniform(0.9, 1.6, n),
            rng.uniform(0.7, 1.25, n),
            rng.uniform(0.4, 1.3, n),
            rng.uniform(0.0, 0.5, n),
        ],
        default=rng.uniform(0.0, 0.35, n),
    )
    unsecured = rng.random(n) < np.select(
        [
            np.isin(loan_purpose, ["Personal", "Medical", "Other"]),
            loan_purpose == "Education",
        ],
        [0.55, 0.35],
        default=0.08,
    )
    collateral_value = np.where(unsecured, 0.0, loan_amount * coll_frac)
    collateral_value = clip_float(collateral_value, 0, 1_50_00_000, 0)

    bank_balance_avg = 0.35 * savings + 1.8 * monthly_income * rng.lognormal(0, 0.4, n)
    bank_balance_avg = clip_float(bank_balance_avg, 1_000, 30_00_000, 0)

    channel_p = np.array([0.48, 0.34, 0.18])
    application_channel = rng.choice(APPLICATION_CHANNELS, size=n, p=channel_p)

    return {
        "age": age,
        "dependents": dependents,
        "employment_type": employment_type,
        "employment_years": employment_years,
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
        "loan_amount": loan_amount,
        "loan_tenure_months": loan_tenure_months,
        "interest_rate": interest_rate,
        "loan_purpose": loan_purpose,
        "collateral_value": collateral_value,
        "bank_balance_avg": bank_balance_avg,
        "credit_utilization": credit_utilization,
        "application_channel": application_channel,
    }


def calibrate_intercept(linear_predictor: np.ndarray, target_rate: float = 0.16) -> float:
    """Choose intercept so average PD is near the academic target (not a bank policy)."""

    def mean_pd(intercept: float) -> float:
        return float(sigmoid(intercept + linear_predictor).mean())

    lo, hi = -12.0, 8.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if mean_pd(mid) > target_rate:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2.0


def assign_default(features: dict[str, np.ndarray], rng: np.random.Generator) -> np.ndarray:
    """
    Internal probabilistic default mechanism.

    The linear combination below is a teaching device only. It is not stored
    as a column and is not a real-world underwriting scorecard.
    """
    income = np.maximum(features["monthly_income"], 1.0)
    new_emi = estimate_emi(
        features["loan_amount"],
        features["interest_rate"],
        features["loan_tenure_months"],
    )
    dti = (features["existing_emi"] + new_emi) / income
    expense_ratio = features["monthly_expenses"] / income
    collateral = features["collateral_value"]
    ltv = np.divide(
        features["loan_amount"],
        collateral,
        out=np.full(collateral.shape, 1.35, dtype=float),
        where=collateral > 0,
    )
    debt_to_annual = features["outstanding_debt"] / (income * 12.0)

    unemployed = (features["employment_type"] == "Unemployed").astype(float)
    contract = (features["employment_type"] == "Contract").astype(float)

    credit_weakness = (700.0 - features["credit_score"].astype(float)) / 100.0
    util = features["credit_utilization"] / 100.0

    lp = (
        0.85 * credit_weakness
        + 0.95 * features["previous_defaults"]
        + 0.09 * features["late_payments"]
        + 1.15 * util
        + 1.10 * np.clip(dti, 0.0, 2.5)
        + 0.55 * np.clip(ltv, 0.0, 3.0)
        + 0.70 * np.clip(debt_to_annual, 0.0, 4.0)
        + 0.85 * np.clip(expense_ratio, 0.0, 1.5)
        + 0.22 * features["active_loans"]
        + 0.70 * unemployed
        + 0.22 * contract
        - 0.028 * features["employment_years"]
        - 0.018 * features["credit_history_years"]
        + rng.normal(0.0, 0.45, len(income))
    )

    intercept = calibrate_intercept(lp, target_rate=0.16)
    pd_hat = sigmoid(intercept + lp)
    default = (rng.random(len(pd_hat)) < pd_hat).astype(int)
    return default


def inject_missing_values(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """About 2–5% missingness in selected numeric columns. IDs and target stay complete."""
    missing_rates = {
        "monthly_income": 0.035,
        "monthly_expenses": 0.030,
        "existing_emi": 0.028,
        "credit_score": 0.032,
        "savings": 0.040,
        "outstanding_debt": 0.025,
        "employment_years": 0.022,
        "collateral_value": 0.038,
    }
    out = df.copy()
    n = len(out)
    for col, rate in missing_rates.items():
        mask = rng.random(n) < rate
        out.loc[mask, col] = np.nan
    return out


def inject_categorical_inconsistencies(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Small share of messy categorical spellings for preprocessing practice."""
    out = df.copy()
    n = len(out)

    emp_variants = {
        "Salaried": ["salaried", "SALARIED", "Salaried "],
        "Self-employed": ["self employed", "Self Employed", "self-employed"],
        "Business": ["business", "BUSINESS"],
        "Contract": ["contract", "CONTRACT"],
        "Unemployed": ["unemployed", "UNEMPLOYED"],
    }
    emp_mask = rng.random(n) < 0.04
    emp_idx = np.where(emp_mask)[0]
    for i in emp_idx:
        canonical = out.at[i, "employment_type"]
        variants = emp_variants.get(canonical, [canonical])
        out.at[i, "employment_type"] = variants[rng.integers(0, len(variants))]

    ch_variants = {
        "Online": ["online", "ONLINE"],
        "Branch": ["branch", "BRANCH"],
        "Agent": ["agent", "AGENT"],
    }
    ch_mask = rng.random(n) < 0.035
    ch_idx = np.where(ch_mask)[0]
    for i in ch_idx:
        canonical = out.at[i, "application_channel"]
        variants = ch_variants.get(canonical, [canonical])
        out.at[i, "application_channel"] = variants[rng.integers(0, len(variants))]

    purpose_variants = {
        "Home": ["home", "HOME"],
        "Education": ["education", "EDUCATION"],
        "Vehicle": ["vehicle", "VEHICLE"],
        "Personal": ["personal", "PERSONAL"],
        "Business": ["business", "BUSINESS"],
        "Medical": ["medical", "MEDICAL"],
        "Other": ["other", "OTHER"],
    }
    p_mask = rng.random(n) < 0.025
    p_idx = np.where(p_mask)[0]
    for i in p_idx:
        canonical = out.at[i, "loan_purpose"]
        variants = purpose_variants.get(canonical, [canonical])
        out.at[i, "loan_purpose"] = variants[rng.integers(0, len(variants))]

    return out


def inject_outliers(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Extreme but in-range values for outlier-handling demonstrations."""
    out = df.copy()
    n = len(out)
    k = max(40, int(0.008 * n))
    idx = rng.choice(n, size=k, replace=False)

    split = np.array_split(idx, 5)
    out.loc[split[0], "monthly_income"] = rng.uniform(4_20_000, 5_00_000, len(split[0])).round(0)
    out.loc[split[1], "savings"] = rng.uniform(42_00_000, 50_00_000, len(split[1])).round(0)
    out.loc[split[2], "outstanding_debt"] = rng.uniform(42_00_000, 50_00_000, len(split[2])).round(0)
    out.loc[split[3], "late_payments"] = rng.integers(12, 16, len(split[3]))
    out.loc[split[4], "loan_amount"] = rng.uniform(80_00_000, 1_00_00_000, len(split[4])).round(0)
    return out


def add_duplicate_rows(df: pd.DataFrame, rng: np.random.Generator, rate: float = 0.007) -> pd.DataFrame:
    """Append copies of existing rows (same applicant_id) at about 0.5–1%."""
    n_dup = int(round(len(df) * rate))
    n_dup = int(np.clip(n_dup, 75, 150))
    dup_idx = rng.choice(len(df), size=n_dup, replace=False)
    duplicates = df.iloc[dup_idx].copy()
    combined = pd.concat([df, duplicates], ignore_index=True)
    return combined.sample(frac=1.0, random_state=SEED).reset_index(drop=True)


def build_raw_dataframe(rng: np.random.Generator) -> tuple[pd.DataFrame, dict]:
    features = generate_base_population(rng, N_RECORDS)
    features["default"] = assign_default(features, rng)

    applicant_id = [f"APP{str(i).zfill(6)}" for i in range(1, N_RECORDS + 1)]
    clean = pd.DataFrame({"applicant_id": applicant_id, **features})
    clean = clean[RAW_COLUMNS]

    meta = {
        "n_generated": N_RECORDS,
        "n_unique_ids": clean["applicant_id"].nunique(),
        "default_rate_before_noise": float(clean["default"].mean()),
    }

    dirty = inject_outliers(clean, rng)
    dirty = inject_missing_values(dirty, rng)
    dirty = inject_categorical_inconsistencies(dirty, rng)
    dirty = add_duplicate_rows(dirty, rng)

    assert list(dirty.columns) == RAW_COLUMNS
    meta["n_raw_rows"] = int(len(dirty))
    meta["n_duplicate_extra_rows"] = int(len(dirty) - N_RECORDS)
    return dirty, meta


def main() -> None:
    rng = np.random.default_rng(SEED)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    raw, meta = build_raw_dataframe(rng)
    raw_path = DATA_DIR / "loan_risk_raw.csv"
    sample_path = DATA_DIR / "raw_dataset_sample.csv"

    raw.to_csv(raw_path, index=False)
    raw.head(25).to_csv(sample_path, index=False)

    print("SYNTHETIC DATASET NOTICE")
    print("This file is fictional and created only for academic ML practice.")
    print("It does not represent real borrowers or a real bank policy.")
    print()
    print(f"Generated unique records (before duplicates): {meta['n_generated']}")
    print(f"Raw CSV rows (after duplicates): {meta['n_raw_rows']}")
    print(f"Extra duplicate rows: {meta['n_duplicate_extra_rows']}")
    print(f"Raw shape: {raw.shape}")
    print(f"Columns ({len(raw.columns)}): {list(raw.columns)}")
    print(f"Default rate in raw file: {raw['default'].mean():.4f}")
    print(f"Missing cells: {int(raw.isna().sum().sum())}")
    print(f"Wrote {raw_path}")
    print(f"Wrote {sample_path}")


if __name__ == "__main__":
    main()
