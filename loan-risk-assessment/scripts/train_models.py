"""
Train, evaluate, and save machine learning models for Loan Risk Assessment.

Models:
1. Logistic Regression (with RobustScaler for numerical features & class_weight='balanced')
2. Random Forest Classifier (with class_weight='balanced')
3. XGBoost Classifier (with scale_pos_weight for class imbalance)

Dataset: data/loan_risk_processed.csv
Target: default
Exclude: applicant_id

Split: 80% train / 20% test, stratified by default, random_state=42.
Leakage Prevention: Preprocessing fitted ONLY on X_train.
"""

from __future__ import annotations

from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier

# Constants
SEED = 42
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "loan_risk_processed.csv"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_CSV = RESULTS_DIR / "model_metrics.csv"

ID_COL = "applicant_id"
TARGET_COL = "default"

DUMMY_PREFIXES = ("employment_type_", "loan_purpose_", "application_channel_")


def load_data(path: Path) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    df = pd.read_csv(path)
    X = df.drop(columns=[ID_COL, TARGET_COL])
    y = df[TARGET_COL]
    feature_names = list(X.columns)
    return X, y, feature_names


def get_feature_groups(feature_names: list[str]) -> tuple[list[str], list[str]]:
    dummy_cols = [c for c in feature_names if c.startswith(DUMMY_PREFIXES)]
    continuous_cols = [c for c in feature_names if c not in dummy_cols]
    return continuous_cols, dummy_cols


def build_models(
    continuous_cols: list[str], dummy_cols: list[str], scale_pos_weight: float
) -> dict[str, object]:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", RobustScaler(), continuous_cols),
            ("passthrough", "passthrough", dummy_cols),
        ]
    )

    log_reg_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=SEED,
                    solver="lbfgs",
                ),
            ),
        ]
    )

    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    )

    xgb_model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=SEED,
        eval_metric="logloss",
        n_jobs=-1,
    )

    return {
        "Logistic Regression": log_reg_pipeline,
        "Random Forest": rf_model,
        "XGBoost": xgb_model,
    }


def evaluate_model(
    model: object, X_test: pd.DataFrame, y_test: pd.Series
) -> tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    precision_pts, recall_pts, _ = precision_recall_curve(y_test, y_proba)

    metrics = {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc,
    }

    return metrics, cm, fpr, tpr, precision_pts, recall_pts


def generate_visualizations(
    results: dict[str, dict],
    cms: dict[str, np.ndarray],
    roc_curves: dict[str, tuple[np.ndarray, np.ndarray]],
    pr_curves: dict[str, tuple[np.ndarray, np.ndarray]],
) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")

    # 1. Model Comparison Chart
    metrics_df = pd.DataFrame(results).T
    plt.figure(figsize=(12, 6))
    ax = metrics_df.plot(kind="bar", figsize=(12, 6), width=0.8, colormap="viridis")
    plt.title("Model Comparison Across Performance Metrics (Test Set)", fontsize=14, fontweight="bold")
    plt.ylabel("Score", fontsize=12)
    plt.xlabel("Model", fontsize=12)
    plt.ylim(0.0, 1.05)
    plt.xticks(rotation=0, fontsize=11)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
    for p in ax.patches:
        height = p.get_height()
        if height > 0.05:
            ax.annotate(
                f"{height:.3f}",
                (p.get_x() + p.get_width() / 2.0, height),
                ha="center",
                va="bottom",
                fontsize=7.5,
                rotation=90,
                xytext=(0, 3),
                textcoords="offset points",
            )
    plt.tight_layout()
    comparison_path = RESULTS_DIR / "model_comparison.png"
    plt.savefig(comparison_path, dpi=300)
    plt.close()

    # 2. ROC Curves Comparison
    plt.figure(figsize=(8, 6))
    colors = {"Logistic Regression": "#1f77b4", "Random Forest": "#2ca02c", "XGBoost": "#ff7f0e"}
    for name, (fpr, tpr) in roc_curves.items():
        auc_val = results[name]["ROC-AUC"]
        plt.plot(fpr, tpr, label=f"{name} (ROC-AUC = {auc_val:.4f})", color=colors.get(name), linewidth=2)
    plt.plot([0, 1], [0, 1], "k--", label="Random Classifier (AUC = 0.5000)", linewidth=1.5)
    plt.title("ROC Curves Comparison (Test Set)", fontsize=14, fontweight="bold")
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    plt.ylabel("True Positive Rate (Recall)", fontsize=12)
    plt.legend(loc="lower right", frameon=True, fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    roc_path = RESULTS_DIR / "roc_curves.png"
    plt.savefig(roc_path, dpi=300)
    plt.close()

    # 3. Precision-Recall Curves Comparison
    plt.figure(figsize=(8, 6))
    for name, (precision_pts, recall_pts) in pr_curves.items():
        pr_auc_val = results[name]["PR-AUC"]
        plt.plot(
            recall_pts,
            precision_pts,
            label=f"{name} (PR-AUC = {pr_auc_val:.4f})",
            color=colors.get(name),
            linewidth=2,
        )
    plt.title("Precision-Recall Curves Comparison (Test Set)", fontsize=14, fontweight="bold")
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.legend(loc="lower left", frameon=True, fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    pr_path = RESULTS_DIR / "pr_curves.png"
    plt.savefig(pr_path, dpi=300)
    plt.close()

    # 4. Confusion Matrices (3 Subplots)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    model_names = list(cms.keys())
    for idx, name in enumerate(model_names):
        cm = cms[name]
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=axes[idx],
            annot_kws={"size": 14, "weight": "bold"},
            xticklabels=["No Default (0)", "Default (1)"],
            yticklabels=["No Default (0)", "Default (1)"],
        )
        axes[idx].set_title(f"{name}\nConfusion Matrix", fontsize=12, fontweight="bold")
        axes[idx].set_xlabel("Predicted Label", fontsize=11)
        axes[idx].set_ylabel("True Label", fontsize=11)
    plt.tight_layout()
    cm_path = RESULTS_DIR / "confusion_matrices.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()


def save_artifacts(models: dict[str, object], metrics_df: pd.DataFrame) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    filename_map = {
        "Logistic Regression": "logistic_regression.pkl",
        "Random Forest": "random_forest.pkl",
        "XGBoost": "xgboost.pkl",
    }

    for name, model in models.items():
        fname = filename_map[name]
        save_path = MODELS_DIR / fname
        joblib.dump(model, save_path)
        print(f"Saved model: {save_path}")

    metrics_df.to_csv(METRICS_CSV, index=True)
    print(f"Saved metrics CSV: {METRICS_CSV}")


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("==================================================")
    print("LRAS MACHINE LEARNING MODEL TRAINING & EVALUATION")
    print("==================================================")
    print(f"Loading processed dataset from: {DATA_PATH}")

    X, y, feature_names = load_data(DATA_PATH)
    print(f"Dataset shape: {X.shape[0]} rows, {X.shape[1]} features")
    print(f"Target distribution: 0={sum(y == 0)} ({(y == 0).mean():.2%}), 1={sum(y == 1)} ({(y == 1).mean():.2%})")

    # 1. Stratified 80/20 Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED
    )
    print(f"\nTrain set: {X_train.shape[0]} samples | Test set: {X_test.shape[0]} samples")

    # 2. Imbalance Ratio Calculation
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"Calculated scale_pos_weight for XGBoost: {scale_pos_weight:.4f}")

    continuous_cols, dummy_cols = get_feature_groups(feature_names)
    models = build_models(continuous_cols, dummy_cols, scale_pos_weight)

    results = {}
    cms = {}
    roc_curves = {}
    pr_curves = {}

    print("\nTraining and evaluating models...")

    for name, model in models.items():
        print(f"  Training {name}...")
        model.fit(X_train, y_train)

        metrics, cm, fpr, tpr, prec_pts, rec_pts = evaluate_model(model, X_test, y_test)
        results[name] = metrics
        cms[name] = cm
        roc_curves[name] = (fpr, tpr)
        pr_curves[name] = (prec_pts, rec_pts)

    metrics_df = pd.DataFrame(results).T

    print("\n" + "=" * 70)
    print("FINAL MODEL EVALUATION METRICS (TEST SET - N=3000)")
    print("=" * 70)
    print(metrics_df.to_string())

    print("\nGenerating evaluation charts...")
    generate_visualizations(results, cms, roc_curves, pr_curves)

    print("Saving model binaries and metrics...")
    save_artifacts(models, metrics_df)

    print("\n" + "=" * 70)
    print("MODEL TRAINING & EVALUATION COMPLETED SUCCESSFULLY!")
    print("==================================================")


if __name__ == "__main__":
    main()
