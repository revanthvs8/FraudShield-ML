import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score,
    roc_auc_score,
)


# ==========================================
# 1. SET PROJECT PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "creditcard.csv"
MODEL_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. LOAD THE DATASET
# ==========================================

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# ==========================================
# 3. SEPARATE FEATURES AND TARGET
# ==========================================

X = df.drop(columns=["Class"])
y = df["Class"]

print("\nFeature shape:", X.shape)
print("Target shape:", y.shape)


# ==========================================
# 4. SPLIT TRAINING AND TEST DATA
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

print("\nFraud cases in training data:", int(y_train.sum()))
print("Fraud cases in testing data:", int(y_test.sum()))


# ==========================================
# 5. SCALE TIME AND AMOUNT
# ==========================================

# V1-V28 are already transformed features.
# Scale Time and Amount using training data only.

preprocessor = ColumnTransformer(
    transformers=[
        ("scale", StandardScaler(), ["Time", "Amount"]),
    ],
    remainder="passthrough",
)


# ==========================================
# 6. CREATE THE MODEL PIPELINE
# ==========================================

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=42,
            ),
        ),
    ]
)


# ==========================================
# 7. TRAIN THE MODEL
# ==========================================

print("\nTraining Logistic Regression model...")

model.fit(X_train, y_train)

print("Model training completed!")


# ==========================================
# 8. MAKE PREDICTIONS
# ==========================================

y_pred = model.predict(X_test)

# Probability estimates for the fraud class
y_proba = model.predict_proba(X_test)[:, 1]


# ==========================================
# 9. EVALUATE THE MODEL
# ==========================================

print("\n========== MODEL EVALUATION ==========")

print("\nClassification Report:")

report = classification_report(
    y_test,
    y_pred,
    target_names=["Legitimate", "Fraudulent"],
    zero_division=0,
)

print(report)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

pr_auc = average_precision_score(y_test, y_proba)
roc_auc = roc_auc_score(y_test, y_proba)

print(f"\nPR-AUC (Average Precision): {pr_auc:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")


# ==========================================
# 10. SAVE THE TRAINED MODEL
# ==========================================

model_path = MODEL_DIR / "fraud_detection_model.joblib"

joblib.dump(model, model_path)

print("\nModel saved successfully!")
print("Model location:", model_path)


# ==========================================
# 11. SAVE EVALUATION RESULTS
# ==========================================

report_path = REPORTS_DIR / "model_evaluation.txt"

with open(report_path, "w", encoding="utf-8") as file:
    file.write("FRAUDSHIELD ML - MODEL EVALUATION\n\n")
    file.write("Model: Logistic Regression\n")
    file.write("Test size: 20%\n")
    file.write("Class weighting: balanced\n\n")
    file.write("Classification Report:\n")
    file.write(report)
    file.write("\n\nConfusion Matrix:\n")
    file.write(str(confusion_matrix(y_test, y_pred)))
    file.write(f"\n\nPR-AUC: {pr_auc:.4f}")
    file.write(f"\nROC-AUC: {roc_auc:.4f}")

print("Evaluation report saved:", report_path)

print("\nFraudShield ML training stage completed!")