"""
SHAP Explainability Pipeline for LRAS Machine Learning Models.

Provides:
1. Global SHAP Feature Importance & Summary (Beeswarm) Analysis for:
   - Logistic Regression
   - Random Forest Classifier
   - XGBoost Classifier
2. CSV Exports of Mean Absolute SHAP Values.
3. Modular, Reusable Functionality for Individual Prediction Explanation.

Dataset: data/loan_risk_processed.csv
Sample Size: N = 1,000 stratified representative sample (random_state=42)
Output Directory: results/shap/
"""

from __future__ import annotations

from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.model_selection import train_test_split

# Constants
SEED = 42
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "loan_risk_processed.csv"
MODELS_DIR = PROJECT_ROOT / "models"
SHAP_RESULTS_DIR = PROJECT_ROOT / "results" / "shap"

ID_COL = "applicant_id"
TARGET_COL = "default"
SAMPLE_SIZE = 1000


def load_dataset_and_models() -> tuple[
    pd.DataFrame, pd.DataFrame, pd.Series, dict[str, object], list[str]
]:
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=[ID_COL, TARGET_COL])
    y = df[TARGET_COL]
    feature_names = list(X.columns)

    models = {
        "Logistic Regression": joblib.load(MODELS_DIR / "logistic_regression.pkl"),
        "Random Forest": joblib.load(MODELS_DIR / "random_forest.pkl"),
        "XGBoost": joblib.load(MODELS_DIR / "xgboost.pkl"),
    }

    return df, X, y, models, feature_names


def get_representative_sample(
    X: pd.DataFrame, y: pd.Series, n_samples: int = SAMPLE_SIZE
) -> tuple[pd.DataFrame, pd.Series]:
    if len(X) <= n_samples:
        return X.copy(), y.copy()
    test_ratio = n_samples / len(X)
    _, X_sample, _, y_sample = train_test_split(
        X, y, test_size=test_ratio, stratify=y, random_state=SEED
    )
    return X_sample, y_sample


def compute_shap_values(
    model: object,
    X_sample: pd.DataFrame,
    model_type: str,
    explainer: object | None = None,
    background_data: pd.DataFrame | None = None,
) -> tuple[np.ndarray, pd.DataFrame, object]:
    """
    Compute SHAP values using the appropriate explainer for each model type.
    Returns: (shap_matrix, X_transformed_or_raw, explainer)
    """
    if model_type == "Logistic Regression":
        preprocessor = model.named_steps["preprocessor"]
        classifier = model.named_steps["classifier"]
        X_scaled = preprocessor.transform(X_sample)
        X_scaled_df = pd.DataFrame(X_scaled, columns=X_sample.columns)

        if explainer is None:
            if background_data is not None:
                bg_scaled = preprocessor.transform(background_data)
                bg_df = pd.DataFrame(bg_scaled, columns=background_data.columns)
                masker = shap.maskers.Independent(bg_df, max_samples=min(100, len(bg_df)))
            elif len(X_sample) > 1:
                masker = shap.maskers.Independent(X_scaled_df, max_samples=len(X_sample))
            else:
                # Fallback background for single-row inference to avoid zero SHAP attributions
                df_bg = pd.read_csv(DATA_PATH).drop(columns=[ID_COL, TARGET_COL])
                bg_sample = df_bg.sample(min(100, len(df_bg)), random_state=SEED)
                bg_scaled = preprocessor.transform(bg_sample)
                bg_df = pd.DataFrame(bg_scaled, columns=bg_sample.columns)
                masker = shap.maskers.Independent(bg_df, max_samples=100)
            explainer = shap.LinearExplainer(classifier, masker)

        shap_vals = explainer.shap_values(X_scaled_df)

        if isinstance(shap_vals, list):
            shap_matrix = shap_vals[1]
        else:
            shap_matrix = shap_vals
        return np.asarray(shap_matrix), X_scaled_df, explainer

    elif model_type == "Random Forest":
        if explainer is None:
            explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
        shap_res = explainer(X_sample)

        if hasattr(shap_res, "values"):
            values = shap_res.values
            if len(values.shape) == 3:
                shap_matrix = values[:, :, 1]
            else:
                shap_matrix = values
        elif isinstance(shap_res, list):
            shap_matrix = shap_res[1]
        else:
            shap_matrix = np.asarray(shap_res)
        return np.asarray(shap_matrix), X_sample, explainer

    elif model_type == "XGBoost":
        if explainer is None:
            explainer = shap.TreeExplainer(model)
        shap_res = explainer(X_sample)

        if hasattr(shap_res, "values"):
            values = shap_res.values
            if len(values.shape) == 3:
                shap_matrix = values[:, :, 1]
            else:
                shap_matrix = values
        elif isinstance(shap_res, list):
            shap_matrix = shap_res[1]
        else:
            shap_matrix = np.asarray(shap_res)
        return np.asarray(shap_matrix), X_sample, explainer

    else:
        raise ValueError(f"Unsupported model_type: {model_type}")


def export_feature_importance_csv(
    shap_matrix: np.ndarray, feature_names: list[str], model_key: str
) -> pd.DataFrame:
    mean_abs_shap = np.abs(shap_matrix).mean(axis=0)
    df_imp = pd.DataFrame(
        {"feature": feature_names, "mean_abs_shap": mean_abs_shap}
    ).sort_values(by="mean_abs_shap", ascending=False).reset_index(drop=True)

    csv_fname = f"shap_feature_importance_{model_key.lower().replace(' ', '_')}.csv"
    csv_path = SHAP_RESULTS_DIR / csv_fname
    df_imp.to_csv(csv_path, index=False)
    print(f"Saved feature importance CSV: {csv_path}")
    return df_imp


def plot_global_bar_importance(
    df_imp: pd.DataFrame, model_name: str, file_suffix: str, top_k: int = 15
) -> None:
    plt.figure(figsize=(10, 6))
    top_df = df_imp.head(top_k).iloc[::-1]

    palette = sns.color_palette("mako", n_colors=top_k)
    bars = plt.barh(top_df["feature"], top_df["mean_abs_shap"], color=palette)

    plt.title(
        f"Global SHAP Feature Importance - {model_name}\n(Mean |SHAP Value| across N={SAMPLE_SIZE} Samples)",
        fontsize=13,
        fontweight="bold",
    )
    plt.xlabel("Mean |SHAP Value| (Impact on Model Output)", fontsize=11)
    plt.ylabel("Feature", fontsize=11)
    plt.grid(True, alpha=0.3, axis="x")

    for bar in bars:
        width = bar.get_width()
        plt.text(
            width + 0.001,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.4f}",
            va="center",
            fontsize=9,
        )

    plt.tight_layout()
    plot_path = SHAP_RESULTS_DIR / f"shap_{file_suffix}.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved bar plot: {plot_path}")


def plot_summary_beeswarm(
    shap_matrix: np.ndarray, X_data: pd.DataFrame, model_name: str, file_suffix: str, top_k: int = 15
) -> None:
    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        shap_matrix,
        X_data,
        feature_names=list(X_data.columns),
        max_display=top_k,
        show=False,
    )
    plt.title(
        f"SHAP Summary (Beeswarm) Plot - {model_name}\n(N={SAMPLE_SIZE} Representative Samples)",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    plt.tight_layout()
    summary_path = SHAP_RESULTS_DIR / f"shap_summary_{file_suffix}.png"
    plt.savefig(summary_path, dpi=300)
    plt.close()
    print(f"Saved summary beeswarm plot: {summary_path}")


def plot_consolidated_comparison(
    importance_dfs: dict[str, pd.DataFrame], top_k: int = 10
) -> None:
    all_top_features = set()
    for df_imp in importance_dfs.values():
        all_top_features.update(df_imp.head(top_k)["feature"].tolist())

    combined_rows = []
    for model_name, df_imp in importance_dfs.items():
        imp_map = dict(zip(df_imp["feature"], df_imp["mean_abs_shap"]))
        for feat in all_top_features:
            combined_rows.append(
                {
                    "model": model_name,
                    "feature": feat,
                    "mean_abs_shap": imp_map.get(feat, 0.0),
                }
            )

    comp_df = pd.DataFrame(combined_rows)

    feat_order = (
        comp_df.groupby("feature")["mean_abs_shap"]
        .max()
        .sort_values(ascending=False)
        .index.tolist()
    )

    plt.figure(figsize=(12, 7))
    sns.barplot(
        data=comp_df,
        y="feature",
        x="mean_abs_shap",
        hue="model",
        order=feat_order,
        palette="viridis",
    )
    plt.title(
        "Consolidated Global SHAP Feature Importance Comparison",
        fontsize=14,
        fontweight="bold",
    )
    plt.xlabel("Mean |SHAP Value|", fontsize=12)
    plt.ylabel("Feature", fontsize=12)
    plt.legend(title="Model Algorithm", frameon=True)
    plt.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()

    comp_path = SHAP_RESULTS_DIR / "shap_global_comparison.png"
    plt.savefig(comp_path, dpi=300)
    plt.close()
    print(f"Saved consolidated comparison plot: {comp_path}")


def explain_individual_prediction(
    model: object,
    applicant_row: pd.DataFrame,
    model_type: str,
    top_k: int = 5,
    explainer: object | None = None,
    background_data: pd.DataFrame | None = None,
) -> dict[str, object]:
    """
    Reusable function to compute individual SHAP explanation for a single applicant.

    Parameters:
    - model: Loaded model object (Pipeline, RandomForestClassifier, or XGBClassifier)
    - applicant_row: 1-row DataFrame containing the 39 processed features
    - model_type: 'Logistic Regression', 'Random Forest', or 'XGBoost'
    - top_k: Number of top risk & protective factors to return
    - explainer: Optional pre-fitted SHAP explainer instance
    - background_data: Optional background DataFrame for linear explainers
    """
    if len(applicant_row) != 1:
        raise ValueError("applicant_row must be a 1-row DataFrame")

    feature_names = list(applicant_row.columns)
    y_pred = int(model.predict(applicant_row)[0])
    y_prob = float(model.predict_proba(applicant_row)[0, 1])

    shap_matrix, X_trans, _ = compute_shap_values(
        model,
        applicant_row,
        model_type,
        explainer=explainer,
        background_data=background_data,
    )
    individual_shap = shap_matrix[0]  # Array of length 39

    factors = []
    for i, feat in enumerate(feature_names):
        val = applicant_row[feat].iloc[0]
        s_val = individual_shap[i]
        factors.append(
            {
                "feature": feat,
                "value": float(val) if isinstance(val, (int, float, np.number)) else str(val),
                "shap_value": float(s_val),
                "abs_shap": float(abs(s_val)),
            }
        )

    df_factors = pd.DataFrame(factors)

    # Risk factors: SHAP > 0 (increase default probability)
    risk_df = (
        df_factors[df_factors["shap_value"] > 0]
        .sort_values(by="shap_value", ascending=False)
        .head(top_k)
    )

    # Protective factors: SHAP < 0 (decrease default probability)
    protective_df = (
        df_factors[df_factors["shap_value"] < 0]
        .sort_values(by="shap_value", ascending=True)
        .head(top_k)
    )

    return {
        "prediction": y_pred,
        "prediction_label": "Default" if y_pred == 1 else "No Default",
        "probability_default": round(y_prob, 4),
        "probability_no_default": round(1.0 - y_prob, 4),
        "top_risk_factors": risk_df[["feature", "value", "shap_value"]].to_dict(orient="records"),
        "top_protective_factors": protective_df[["feature", "value", "shap_value"]].to_dict(orient="records"),
        "shap_values_dict": dict(zip(feature_names, individual_shap)),
    }


def main() -> None:
    SHAP_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("==================================================")
    print("LRAS SHAP EXPLAINABILITY PIPELINE")
    print("==================================================")

    df, X, y, models, feature_names = load_dataset_and_models()
    print(f"Loaded dataset: {X.shape[0]} rows, {X.shape[1]} features")

    # Sample representative N = 1000 subset for global SHAP calculation
    X_sample, y_sample = get_representative_sample(X, y, n_samples=SAMPLE_SIZE)
    print(f"Using representative sample of N = {len(X_sample)} for global SHAP analysis (random_state=42)")

    file_suffix_map = {
        "Logistic Regression": "logistic_regression",
        "Random Forest": "random_forest",
        "XGBoost": "xgboost",
    }

    importance_dfs = {}
    shap_matrices = {}

    print("\nComputing global SHAP feature importance...")
    for model_name, model in models.items():
        print(f"\n--- Model: {model_name} ---")
        suffix = file_suffix_map[model_name]

        shap_matrix, X_trans, explainer = compute_shap_values(model, X_sample, model_name)
        shap_matrices[model_name] = shap_matrix

        # Check finite
        assert np.isfinite(shap_matrix).all(), f"Non-finite SHAP values in {model_name}"

        # Export CSV
        df_imp = export_feature_importance_csv(shap_matrix, feature_names, model_name)
        importance_dfs[model_name] = df_imp

        # Bar plot
        plot_global_bar_importance(df_imp, model_name, suffix, top_k=15)

        # Beeswarm summary plot
        plot_summary_beeswarm(shap_matrix, X_trans, model_name, suffix, top_k=15)

    print("\nGenerating consolidated global feature comparison plot...")
    plot_consolidated_comparison(importance_dfs, top_k=10)

    # Test Individual Explanation functionality on sample applicants
    print("\n==================================================")
    print("TESTING INDIVIDUAL PREDICTION EXPLANATION FUNCTION")
    print("==================================================")

    def_idx = y_sample[y_sample == 1].index[0]
    non_def_idx = y_sample[y_sample == 0].index[0]

    for idx, label in [(non_def_idx, "Sample Non-Default Borrower"), (def_idx, "Sample Default Borrower")]:
        row = X.loc[[idx]]
        app_id = df.loc[idx, ID_COL]
        actual_target = y.loc[idx]

        print(f"\n--- Individual Explanation for Applicant {app_id} ({label}) ---")
        print(f"Actual Target: {actual_target}")

        for model_name, model in models.items():
            explanation = explain_individual_prediction(
                model, row, model_type=model_name, top_k=3
            )
            print(f"  [{model_name}] Pred: {explanation['prediction_label']} (P(Def)={explanation['probability_default']:.4f})")
            print("    Top Risk Factors:")
            for r in explanation["top_risk_factors"]:
                print(f"      + {r['feature']} = {r['value']} (SHAP: +{r['shap_value']:.4f})")
            print("    Top Protective Factors:")
            for p in explanation["top_protective_factors"]:
                print(f"      - {p['feature']} = {p['value']} (SHAP: {p['shap_value']:.4f})")

    print("\n==================================================")
    print("SHAP EXPLAINABILITY PIPELINE COMPLETED SUCCESSFULLY!")
    print("==================================================")


if __name__ == "__main__":
    main()
