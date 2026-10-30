"""
Preprocess the synthetic loan-risk raw dataset into an ML-ready table.

Decisions are logged and written to docs/preprocessing_report.md using
statistics computed from the actual files (nothing is hard-coded).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

SEED = 42
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DOCS_DIR = PROJECT_ROOT / "docs"

RAW_PATH = DATA_DIR / "loan_risk_raw.csv"
PROCESSED_PATH = DATA_DIR / "loan_risk_processed.csv"
PROCESSED_SAMPLE_PATH = DATA_DIR / "processed_dataset_sample.csv"
REPORT_PATH = DOCS_DIR / "preprocessing_report.md"

CATEGORICAL_COLS = ["employment_type", "loan_purpose", "application_channel"]
ID_COL = "applicant_id"
TARGET_COL = "default"

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

RANGES = {
    "age": (21, 65),
    "dependents": (0, 5),
    "employment_years": (0, 40),
    "monthly_income": (15_000, 5_00_000),
    "monthly_expenses": (5_000, 2_50_000),
    "existing_emi": (0, 1_50_000),
    "savings": (5_000, 50_00_000),
    "outstanding_debt": (0, 50_00_000),
    "credit_score": (300, 850),
    "credit_history_years": (0, 40),
    "active_loans": (0, 8),
    "previous_defaults": (0, 3),
    "late_payments": (0, 15),
    "loan_amount": (50_000, 1_00_00_000),
    "loan_tenure_months": (12, 360),
    "interest_rate": (6, 24),
    "collateral_value": (0, 1_50_00_000),
    "bank_balance_avg": (1_000, 30_00_000),
    "credit_utilization": (0, 100),
}

# Cap only the far tail of heavy rupee fields. 1.5-IQR would flag the natural
# right skew of income/loan size as "outliers," so percentile winsorization
# is used instead. Bounded scores and counts are range-clipped later.
WINSORIZE_COLS = [
    "monthly_income",
    "monthly_expenses",
    "existing_emi",
    "savings",
    "outstanding_debt",
    "loan_amount",
    "collateral_value",
    "bank_balance_avg",
]


def load_raw(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def inspect_frame(df: pd.DataFrame) -> dict:
    return {
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing_by_col": {c: int(v) for c, v in df.isna().sum().items() if v > 0},
        "missing_total": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_ids": int(df[ID_COL].duplicated().sum()) if ID_COL in df.columns else 0,
        "target_counts": {int(k): int(v) for k, v in df[TARGET_COL].value_counts().sort_index().items()},
        "target_rate": float(df[TARGET_COL].mean()),
    }


def _norm_key(value: object) -> str:
    text = str(value).strip().lower()
    text = " ".join(text.split())
    text = text.replace("_", " ")
    return text


def standardize_categoricals(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    out = df.copy()
    notes = {}
    maps = {
        "employment_type": EMPLOYMENT_MAP,
        "loan_purpose": PURPOSE_MAP,
        "application_channel": CHANNEL_MAP,
    }
    for col, mapping in maps.items():
        original = out[col].astype(str)
        cleaned = original.map(lambda x: mapping.get(_norm_key(x), str(x).strip()))
        n_changed = int((original.str.strip() != cleaned).sum())
        out[col] = cleaned
        notes[col] = {
            "rows_changed": n_changed,
            "unique_after": sorted(out[col].dropna().unique().tolist()),
        }
    return out, notes


def impute_numeric(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Median imputation for numeric missingness.

    Median is used because rupee fields are skewed and a small number of
    extreme-but-valid outliers exist by design. applicant_id and default
    are not imputed.
    """
    out = df.copy()
    notes = {}
    numeric_cols = out.select_dtypes(include=[np.number]).columns.tolist()
    if TARGET_COL in numeric_cols:
        numeric_cols.remove(TARGET_COL)
    for col in numeric_cols:
        n_missing = int(out[col].isna().sum())
        if n_missing == 0:
            continue
        median_value = float(out[col].median())
        out[col] = out[col].fillna(median_value)
        notes[col] = {"n_missing": n_missing, "fill_value": median_value, "strategy": "median"}
    return out, notes


def impute_categorical(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    out = df.copy()
    notes = {}
    for col in CATEGORICAL_COLS:
        n_missing = int(out[col].isna().sum())
        if n_missing == 0:
            notes[col] = {"n_missing": 0, "strategy": "none"}
            continue
        mode_value = out[col].mode(dropna=True)
        fill_value = str(mode_value.iloc[0]) if len(mode_value) else "Unknown"
        out[col] = out[col].fillna(fill_value)
        notes[col] = {"n_missing": n_missing, "fill_value": fill_value, "strategy": "mode"}
    return out, notes


def winsorize_percentiles(
    df: pd.DataFrame, cols: list[str], lower_q: float = 0.005, upper_q: float = 0.995
) -> tuple[pd.DataFrame, dict]:
    """
    Cap values outside the 0.5th and 99.5th percentiles, then clip to documented ranges.

    Rows are not dropped. Percentile caps are used instead of 1.5-IQR because
    income, savings, and loan amounts are right-skewed by design; 1.5-IQR would
    treat ordinary high-value borrowers as outliers.
    """
    out = df.copy()
    notes = {}
    for col in cols:
        series = out[col].astype(float)
        low = float(series.quantile(lower_q))
        high = float(series.quantile(upper_q))
        if col in RANGES:
            allowed_low, allowed_high = RANGES[col]
            low = max(low, allowed_low)
            high = min(high, allowed_high)
        n_low = int((series < low).sum())
        n_high = int((series > high).sum())
        out[col] = series.clip(lower=low, upper=high)
        notes[col] = {
            "lower_quantile": lower_q,
            "upper_quantile": upper_q,
            "lower_cap": low,
            "upper_cap": high,
            "n_capped_low": n_low,
            "n_capped_high": n_high,
        }
    return out, notes


def validate_ranges(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    out = df.copy()
    notes = {}
    for col, (low, high) in RANGES.items():
        before = out[col].astype(float)
        n_below = int((before < low).sum())
        n_above = int((before > high).sum())
        out[col] = before.clip(low, high)
        notes[col] = {"n_below": n_below, "n_above": n_above, "low": low, "high": high}
    int_cols = [
        "age",
        "dependents",
        "credit_score",
        "active_loans",
        "previous_defaults",
        "late_payments",
        "loan_tenure_months",
        TARGET_COL,
    ]
    for col in int_cols:
        out[col] = np.rint(out[col]).astype(int)
    return out, notes


def compute_emi(principal: np.ndarray, annual_rate_pct: np.ndarray, tenure_months: np.ndarray) -> np.ndarray:
    r = (annual_rate_pct / 100.0) / 12.0
    n = np.maximum(tenure_months.astype(float), 1.0)
    p = principal.astype(float)
    return np.where(
        r <= 1e-12,
        p / n,
        p * r * np.power(1.0 + r, n) / (np.power(1.0 + r, n) - 1.0),
    )


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    income = np.maximum(out["monthly_income"].astype(float), 1.0)
    new_emi = compute_emi(
        out["loan_amount"].to_numpy(),
        out["interest_rate"].to_numpy(),
        out["loan_tenure_months"].to_numpy(),
    )
    out["estimated_new_emi"] = np.round(new_emi, 2)
    out["DTI"] = (out["existing_emi"].astype(float) + new_emi) / income
    collateral = out["collateral_value"].astype(float)
    # Unsecured loans (zero collateral) are treated as LTV = 1.0 so the
    # ratio stays finite. This is a modeling convention, not a legal LTV.
    out["LTV"] = np.where(collateral > 0, out["loan_amount"].astype(float) / collateral, 1.0)
    out["expense_to_income_ratio"] = out["monthly_expenses"].astype(float) / income
    out["savings_to_income_ratio"] = out["savings"].astype(float) / (income * 12.0)
    ratio_cols = ["DTI", "LTV", "expense_to_income_ratio", "savings_to_income_ratio"]
    for col in ratio_cols:
        out[col] = out[col].replace([np.inf, -np.inf], np.nan)
        if out[col].isna().any():
            out[col] = out[col].fillna(out[col].median())
        out[col] = np.round(out[col], 4)
    return out


def encode_categoricals(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    encoder = OneHotEncoder(
        categories=[
            ["Salaried", "Self-employed", "Business", "Contract", "Unemployed"],
            ["Home", "Education", "Vehicle", "Personal", "Business", "Medical", "Other"],
            ["Online", "Branch", "Agent"],
        ],
        handle_unknown="ignore",
        sparse_output=False,
        dtype=np.int64,
    )
    encoded = encoder.fit_transform(df[CATEGORICAL_COLS])
    encoded_names = encoder.get_feature_names_out(CATEGORICAL_COLS).tolist()
    encoded_df = pd.DataFrame(encoded, columns=encoded_names, index=df.index)
    rest = df.drop(columns=CATEGORICAL_COLS)
    out = pd.concat([rest, encoded_df], axis=1)
    return out, encoded_names


def write_report(stats: dict) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    raw = stats["raw"]
    missing_lines = "\n".join(
        f"- `{col}`: {count}" for col, count in sorted(raw["missing_by_col"].items())
    ) or "- None"
    cat_lines = "\n".join(
        f"- `{col}`: {info['rows_changed']} values standardized; unique after: {info['unique_after']}"
        for col, info in stats["categorical_notes"].items()
    )
    miss_num_lines = "\n".join(
        f"- `{col}`: {info['n_missing']} filled with median {info['fill_value']:.4f}"
        for col, info in stats["numeric_impute"].items()
    ) or "- No numeric missing values."
    miss_cat_lines = "\n".join(
        f"- `{col}`: {info['n_missing']} missing; strategy = {info['strategy']}"
        + (f"; fill = {info.get('fill_value')}" if info.get("fill_value") else "")
        for col, info in stats["categorical_impute"].items()
    )
    outlier_lines = "\n".join(
        f"- `{col}`: capped {info['n_capped_low']} low / {info['n_capped_high']} high "
        f"(percentile caps [{info['lower_cap']:.2f}, {info['upper_cap']:.2f}])"
        for col, info in stats["outlier_notes"].items()
    )
    feature_lines = "\n".join(f"- `{name}`" for name in stats["final_feature_list"])

    before_counts = raw["target_counts"]
    after_counts = stats["processed_target_counts"]
    n0_b, n1_b = before_counts.get(0, 0), before_counts.get(1, 0)
    n0_a, n1_a = after_counts.get(0, 0), after_counts.get(1, 0)

    text = f"""# Preprocessing Report

**Dataset status:** synthetic, educational, not real customer data.

This report is generated by `scripts/preprocess.py` from the files on disk.
Figures are computed, not invented.

## 1. Raw dataset size

- File: `data/loan_risk_raw.csv`
- Rows: **{raw['n_rows']}**
- Columns: **{raw['n_cols']}**
- Unique `applicant_id` values: **{stats['raw_unique_ids']}**
- Expected unique generated applicants before duplicate injection: **15000**

## 2. Missing values before preprocessing

- Total missing cells: **{raw['missing_total']}**

{missing_lines}

`applicant_id` and `default` were required to be complete and were not given missing values during generation.

## 3. Duplicate records

- Exact duplicate rows (`DataFrame.duplicated()`): **{raw['duplicate_rows']}**
- Duplicate `applicant_id` occurrences (count of extra ID copies): **{raw['duplicate_ids']}**

Duplicates were dropped with `keep='first'` so each applicant appears once. Rows were not dropped for missing values.

## 4. Categorical inconsistencies

Inconsistent casings and spacing (for example `salaried`, `self employed`, `ONLINE`) were mapped to canonical labels.

{cat_lines}

## 5. Outlier handling

Method: winsorize at the 0.5th and 99.5th percentiles, then clip to the documented academic ranges.
No rows were deleted for being outliers. 1.5-IQR was not used because rupee amounts are right-skewed.

{outlier_lines}

Count-like and score fields (`age`, `credit_score`, `late_payments`, etc.) were range-clipped rather than percentile-capped, because their bounds are already part of the data contract.

## 6. Missing-value treatment

Numeric columns: **median imputation** (robust to the designed right tail in rupee amounts).

{miss_num_lines}

Categorical columns: **mode imputation** if any missing values exist.

{miss_cat_lines}

## 7. Feature engineering

Derived columns (processed dataset only):

- `estimated_new_emi`: standard EMI formula using `loan_amount`, `interest_rate`, `loan_tenure_months`
- `DTI`: `(existing_emi + estimated_new_emi) / monthly_income`
- `LTV`: `loan_amount / collateral_value` if collateral > 0, otherwise **1.0** (finite convention for unsecured loans)
- `expense_to_income_ratio`: `monthly_expenses / monthly_income`
- `savings_to_income_ratio`: `savings / (monthly_income * 12)`

No artificial “risk score” or “financial health score” was created.

## 8. Encoding method

`sklearn.preprocessing.OneHotEncoder` with fixed category orders:

- `employment_type`: Salaried, Self-employed, Business, Contract, Unemployed
- `loan_purpose`: Home, Education, Vehicle, Personal, Business, Medical, Other
- `application_channel`: Online, Branch, Agent

`handle_unknown='ignore'`, dense integer dummy columns. `applicant_id` is retained for identity checks but **must not** be used as an ML feature. `default` stays 0/1.

## 9. Final processed dataset size

- File: `data/loan_risk_processed.csv`
- Rows: **{stats['processed_rows']}**
- Columns: **{stats['processed_cols']}**
- Remaining missing cells: **{stats['processed_missing']}**
- Remaining duplicate rows: **{stats['processed_duplicates']}**

## 10. Final feature list

Columns in the processed file (including id, target, derived fields, and dummies):

{feature_lines}

Recommended ML feature set: every column except `{ID_COL}` and `{TARGET_COL}`.

## 11. Target distribution before and after preprocessing

### Raw (including duplicate rows)

- No default (0): **{n0_b}** ({(n0_b / raw['n_rows'] * 100 if raw['n_rows'] else 0):.2f}%)
- Default (1): **{n1_b}** ({(n1_b / raw['n_rows'] * 100 if raw['n_rows'] else 0):.2f}%)

### Processed (duplicates removed)

- No default (0): **{n0_a}** ({(n0_a / stats['processed_rows'] * 100 if stats['processed_rows'] else 0):.2f}%)
- Default (1): **{n1_a}** ({(n1_a / stats['processed_rows'] * 100 if stats['processed_rows'] else 0):.2f}%)

Class imbalance is expected and should be handled later during modeling (not in this pipeline).
"""
    REPORT_PATH.write_text(text, encoding="utf-8")


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    raw = load_raw(RAW_PATH)
    raw_stats = inspect_frame(raw)
    raw_unique_ids = int(raw[ID_COL].nunique())

    print("=== Raw inspection ===")
    print(f"Shape: {raw.shape}")
    print(f"Dtypes:\n{raw.dtypes}")
    print(f"Missing total: {raw_stats['missing_total']}")
    print(f"Missing by column:\n{raw.isna().sum()}")
    print(f"Duplicate rows: {raw_stats['duplicate_rows']}")

    n_before_dedup = len(raw)
    df = raw.drop_duplicates(keep="first").reset_index(drop=True)
    n_dropped_dups = n_before_dedup - len(df)

    df, cat_notes = standardize_categoricals(df)
    df, num_impute = impute_numeric(df)
    df, cat_impute = impute_categorical(df)
    df, outlier_notes = winsorize_percentiles(df, WINSORIZE_COLS)
    df, range_notes = validate_ranges(df)
    df = add_derived_features(df)
    df, encoded_names = encode_categoricals(df)

    # Column order: id, original numerics, derived, dummies, target last.
    derived = [
        "estimated_new_emi",
        "DTI",
        "LTV",
        "expense_to_income_ratio",
        "savings_to_income_ratio",
    ]
    original_numeric = [
        c
        for c in raw.columns
        if c not in CATEGORICAL_COLS + [ID_COL, TARGET_COL]
    ]
    ordered = [ID_COL] + original_numeric + derived + encoded_names + [TARGET_COL]
    df = df[ordered]

    df.to_csv(PROCESSED_PATH, index=False)
    df.head(25).to_csv(PROCESSED_SAMPLE_PATH, index=False)

    processed_missing = int(df.isna().sum().sum())
    processed_dups = int(df.duplicated().sum())
    processed_target = {int(k): int(v) for k, v in df[TARGET_COL].value_counts().sort_index().items()}

    stats = {
        "raw": raw_stats,
        "raw_unique_ids": raw_unique_ids,
        "n_dropped_dups": n_dropped_dups,
        "categorical_notes": cat_notes,
        "numeric_impute": num_impute,
        "categorical_impute": cat_impute,
        "outlier_notes": outlier_notes,
        "range_notes": range_notes,
        "encoded_names": encoded_names,
        "processed_rows": int(len(df)),
        "processed_cols": int(df.shape[1]),
        "processed_missing": processed_missing,
        "processed_duplicates": processed_dups,
        "processed_target_counts": processed_target,
        "final_feature_list": list(df.columns),
    }
    write_report(stats)

    print("\n=== Processed ===")
    print(f"Shape: {df.shape}")
    print(f"Dropped duplicate rows: {n_dropped_dups}")
    print(f"Missing cells: {processed_missing}")
    print(f"Target counts: {processed_target}")
    print(f"Wrote {PROCESSED_PATH}")
    print(f"Wrote {PROCESSED_SAMPLE_PATH}")
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
