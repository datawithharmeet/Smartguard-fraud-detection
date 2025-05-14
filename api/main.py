import pandas as pd
import pickle
import shap
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

# --- Config ---
MODEL_PATH = "artifacts/fraud_model.pkl"
SAMPLE_DATA_PATH = "artifacts/model_input.parquet"
TOP_N_FEATURES = 3

# --- Load model and sample structure ---
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

df_sample = pd.read_parquet(SAMPLE_DATA_PATH).drop(columns=["fraud_label"]).sample(n=100, random_state=42)
expected_columns = df_sample.columns.tolist()
explainer = shap.TreeExplainer(model)

# --- FastAPI App ---
app = FastAPI(title="SmartGuard Fraud Detection API (Polished)")

# --- Define user input schema (raw, user-friendly format) ---
class TransactionInput(BaseModel):
    amount: float
    use_chip: str
    card_type: str
    card_brand: str
    merchant_state: str
    credit_score: float
    yearly_income: float
    num_credit_cards: float

# --- Preprocessing Function ---
def preprocess_input(raw: TransactionInput):
    """Convert raw input into model-ready DataFrame."""
    data = {}

    # Log-transformed numerical features
    data["log_amount"] = np.log1p(raw.amount)
    data["log_income"] = np.log1p(raw.yearly_income)
    data["credit_score"] = raw.credit_score
    data["num_credit_cards"] = raw.num_credit_cards

    # Derived feature
    threshold = df_sample["log_amount"].quantile(0.95)
    data["is_high_amount"] = int(data["log_amount"] > threshold)

    # One-hot encoding handling
    for col in expected_columns:
        if col.startswith("use_chip_"):
            data[col] = 1 if col == f"use_chip_{raw.use_chip}" else 0
        elif col.startswith("card_type_"):
            data[col] = 1 if col == f"card_type_{raw.card_type}" else 0
        elif col.startswith("card_brand_"):
            data[col] = 1 if col == f"card_brand_{raw.card_brand}" else 0
        elif col.startswith("merchant_state_"):
            data[col] = 1 if col == f"merchant_state_{raw.merchant_state}" else 0
        elif col not in data:
            data[col] = 0  # fill remaining columns

    df_final = pd.DataFrame([data])[expected_columns]
    return df_final

# --- Prediction Endpoint ---
@app.post("/predict")
def predict(input_data: TransactionInput):
    df_input = preprocess_input(input_data)

    prob = model.predict_proba(df_input)[0][1]
    pred = model.predict(df_input)[0]

    # Explain with SHAP
    shap_vals = explainer.shap_values(df_input)

    # Handle binary classification (shap_vals is just an array)
    if isinstance(shap_vals, list): # Multi-class case
        shap_array = shap_vals[1]    # Use class 1 (fraud)
    else:
        shap_array = shap_vals       # Binary case

    top_indices = np.argsort(np.abs(shap_array[0]))[::-1][:3]
    top_features = {
               df_input.columns[i]: float(shap_array[0][i])
               for i in top_indices
    }


    return {
        "fraud_prediction": int(pred),
        "risk_score": round(float(prob), 4),
        "top_contributing_features": top_features
    }
