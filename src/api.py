from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel, Field


# Locate the project directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Locate the trained model
MODEL_PATH = PROJECT_ROOT / "models" / "fraud_detection_model.joblib"


# Check that the model exists
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "Model not found. Run python src/train_model.py first."
    )


# Load the trained model
model = joblib.load(MODEL_PATH)


# Create the FastAPI application
app = FastAPI(
    title="FraudShield ML",
    description="Machine Learning Based Fraud Detection API",
    version="1.0.0",
)


# Define transaction input
class Transaction(BaseModel):
    Time: float = Field(ge=0)

    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float

    Amount: float = Field(ge=0)


# Home endpoint
@app.get("/")
def home():
    return {
        "project": "FraudShield ML",
        "message": "Fraud Detection API is running!"
    }


# Health check endpoint
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True
    }


# Fraud prediction endpoint
@app.post("/predict")
def predict_transaction(transaction: Transaction):

    # Convert input into a DataFrame
    transaction_df = pd.DataFrame(
        [transaction.model_dump()],
        columns=[
            "Time",
            *[f"V{i}" for i in range(1, 29)],
            "Amount"
        ]
    )

    # Predict fraud probability
    fraud_probability = float(
        model.predict_proba(transaction_df)[0][1]
    )

    # Predict the class using the trained model's threshold
    prediction = int(
        model.predict(transaction_df)[0]
    )

    # Convert prediction to readable output
    if prediction == 1:
        label = "Potentially Fraudulent"
    else:
        label = "Likely Legitimate"

    return {
        "prediction": label,
        "fraud_probability": round(fraud_probability, 4),
        "risk_score": round(fraud_probability * 100, 2),
        "disclaimer": "Model estimate only; not proof of fraud."
    }