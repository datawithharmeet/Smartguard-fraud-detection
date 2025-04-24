from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from api.model_loader import load_model, get_feature_names

app = FastAPI()
model = load_model()
feature_names = get_feature_names()

class Transaction(BaseModel):
    data: dict

@app.get("/")
def root():
    return {"message": "Welcome to SmartGuard Fraud Detection API"}

@app.post("/predict")
def predict(transaction: Transaction):
    try:
        input_df = pd.DataFrame([transaction.data])
        input_df = input_df.reindex(columns=feature_names, fill_value=0)  # ensure column order
        proba = model.predict_proba(input_df)[0][1]
        label = int(proba > 0.5)
        return {
            "fraud_probability": round(proba, 4),
            "is_fraud": label
        }
    except Exception as e:
        return {"error": str(e)}
