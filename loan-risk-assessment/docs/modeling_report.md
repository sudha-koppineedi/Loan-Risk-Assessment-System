# Machine Learning Modeling Report

## 1. Executive Summary

This report documents the machine learning modeling phase for the **Intelligent Loan Risk Assessment & Underwriting Support System (LRAS)**. 

Three candidate classification algorithms were trained and evaluated on the processed borrower dataset (`data/loan_risk_processed.csv`):
1. **Logistic Regression** (Linear baseline with `RobustScaler` preprocessing & balanced class weighting)
2. **Random Forest Classifier** (Ensemble bagging trees with balanced class weighting)
3. **XGBoost Classifier** (Gradient boosted decision trees with positive class scale weighting)

All model pipelines were trained strictly on an 80% stratified training split ($N = 12,000$) and evaluated on an isolated 20% holdout test split ($N = 3,000$).

---

## 2. Dataset Partitioning & Data Leakage Prevention

### Train / Test Split Specification:
- **Total Processed Dataset Size:** 15,000 records, 41 columns.
- **Excluded Features:** `applicant_id` (retained for row index identification only).
- **Target Feature:** `default` (0 = No Default, 1 = Default).
- **Split Ratio:** 80% Training ($N = 12,000$), 20% Testing ($N = 3,000$).
- **Stratification:** Stratified sampling applied on `default` to ensure identical positive class proportions in both sets (~15.67% default rate).
- **Random State:** `random_state = 42` for strict reproducibility.

### Strict Data Leakage Controls:
- **Preprocessing Scoping:** Feature scaling (`RobustScaler`) for Logistic Regression was fitted **ONLY** on the training split `X_train` and then applied to transform the test set `X_test`.
- **Parameter Calibration:** Imbalance ratios (e.g. XGBoost `scale_pos_weight`) were derived strictly from `y_train` distribution counts.

---

## 3. Class Imbalance Handling Strategy

The training dataset contains 10,119 non-defaults (84.33%) and 1,881 defaults (15.67%), yielding a positive-to-negative class ratio of **1 : 5.38**.

To prevent classifiers from favoring the majority non-default class:
1. **Logistic Regression:** Configured with `class_weight='balanced'`, which automatically assigns weights inversely proportional to class frequencies:
   $$w_0 = \frac{N}{2 \cdot N_0} \approx 0.59, \quad w_1 = \frac{N}{2 \cdot N_1} \approx 3.19$$
2. **Random Forest Classifier:** Configured with `class_weight='balanced'`, applying class weights across individual tree bootstraps.
3. **XGBoost Classifier:** Configured with positive class weight scalar:
   $$\text{scale\_pos\_weight} = \frac{\text{Count}(\text{default}=0)}{\text{Count}(\text{default}=1)} = \frac{10119}{1881} \approx 5.3796$$

---

## 4. Model Architecture & Hyperparameters

### Model 1: Logistic Regression Pipeline
- **Preprocessing:** `ColumnTransformer` applying `RobustScaler()` to continuous numerical attributes and `passthrough` to pre-encoded binary dummy attributes.
- **Classifier:** `LogisticRegression(class_weight='balanced', solver='lbfgs', max_iter=1000, random_state=42)`.

### Model 2: Random Forest Classifier
- **Parameters:** `RandomForestClassifier(n_estimators=100, max_depth=12, min_samples_split=5, min_samples_leaf=2, class_weight='balanced', random_state=42, n_jobs=-1)`.
- **Design Rationale:** `max_depth=12` prevents tree overfitting on synthetic noise while capturing non-linear interactions.

### Model 3: XGBoost Classifier
- **Parameters:** `XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=5.3796, eval_metric='logloss', random_state=42, n_jobs=-1)`.
- **Design Rationale:** Moderate depth (`max_depth=5`) combined with sub-sampling (`0.8`) and conservative learning rate (`0.05`) ensures stable gradient boosting gradient updates.

---

## 5. Evaluation Metrics & Performance Results

All models were evaluated on the $N = 3,000$ holdout test set (2,530 non-defaults, 470 defaults).

### Quantitative Summary Table (Test Set):

| Model Algorithm | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.7440 | 0.3498 | **0.7383** | 0.4747 | **0.8213** | **0.5240** |
| **Random Forest** | **0.8113** | **0.4241** | 0.5702 | 0.4864 | 0.8156 | 0.4803 |
| **XGBoost** | 0.7647 | 0.3709 | 0.7213 | **0.4899** | 0.8209 | 0.5201 |

### Confusion Matrix Breakdown ($N = 3,000$ Test Samples):

- **Logistic Regression:**
  - True Negatives (TN): **1,885**
  - False Positives (FP): **645**
  - False Negatives (FN): **123**
  - True Positives (TP): **347**
- **Random Forest Classifier:**
  - True Negatives (TN): **2,166**
  - False Positives (FP): **364**
  - False Negatives (FN): **202**
  - True Positives (TP): **268**
- **XGBoost Classifier:**
  - True Negatives (TN): **1,955**
  - False Positives (FP): **575**
  - False Negatives (FN): **131**
  - True Positives (TP): **339**

---

## 6. Objective Model Analysis & Business Trade-offs

Rather than declaring an arbitrary "best" model based solely on raw Accuracy, the selection of a model depends on underwriting policy priorities:

1. **Risk Minimization Priority (Maximizing Default Detection / High Recall):**
   - **Logistic Regression** achieves the highest Recall (**73.83%**, detecting 347 out of 470 defaults) and top ROC-AUC (**0.8213**).
   - **XGBoost** achieves nearly matching Recall (**72.13%**, detecting 339 out of 470 defaults) with higher overall Accuracy (76.47%).
   - *Trade-off:* Higher false positive rate (~645 or 575 false alarms), requiring additional manual underwriting review for flagged applicants.

2. **Precision & Efficiency Priority (Minimizing False Alarms / High Precision):**
   - **Random Forest** achieves the highest Overall Accuracy (**81.13%**) and Precision (**42.41%**), generating only 364 false positives.
   - *Trade-off:* Lower Recall (**57.02%**), missing 202 actual defaults (higher credit loss exposure).

3. **Balanced Operational Performance:**
   - **XGBoost** provides the highest overall F1-Score (**0.4899**) and PR-AUC (**0.5201**), offering a balanced compromise between default detection (72.13% Recall) and precision (37.09%).

---

## 7. Model Limitations

1. **Synthetic Data Boundaries:** Models were trained on synthetic data generated via probabilistic logistic functions. Real-world credit data contains macroeconomic shocks and non-stationary distributions not fully modeled here.
2. **Fixed Hyperparameters:** Models were trained using baseline hyperparameters without extensive grid/randomized search.
3. **Threshold Sensitivity:** Classification metrics reflect default decision thresholds at 0.50. Adjusting decision thresholds can further tailor Precision/Recall trade-offs for production deployment.

---

## 8. Saved Artifacts & Deliverables

- **Trained Model Binaries:**
  - `models/logistic_regression.pkl`
  - `models/random_forest.pkl`
  - `models/xgboost.pkl`
- **Evaluation Metrics Table:** `results/model_metrics.csv`
- **Visualization Artifacts:**
  - `results/model_comparison.png`
  - `results/roc_curves.png`
  - `results/pr_curves.png`
  - `results/confusion_matrices.png`
