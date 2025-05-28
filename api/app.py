import os
import sys
import json
import pandas as pd
import shap
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List,Optional

from frauddetection.utils.main_utils import load_object
from frauddetection.streaming.feature_engineering import RealTimeFeatureEngineer
from frauddetection.constants import pipeline_constants as pc
from frauddetection.exception.exception import SmartGuardException


app = FastAPI(title="SmartGuard API")

model = load_object(os.path.join(pc.ARTIFACT_DIR, pc.MODEL_TRAINER_DIR_NAME, pc.MODEL_FILE_NAME))
sg_preprocessor = load_object(os.path.join(pc.ARTIFACT_DIR, pc.DATA_TRANSFORMATION_DIR_NAME, pc.PREPROCESSING_OBJECT_FILE_NAME))
explainer = load_object(os.path.join(pc.ARTIFACT_DIR, pc.SHAP_OUTPUT_DIR, pc.SHAP_EXPLAINER_FILE))
with open(os.path.join(pc.ARTIFACT_DIR, pc.MODEL_TRAINER_DIR_NAME, pc.FEATURE_LIST_FILE)) as f:
    feature_list = [line.strip() for line in f]

feature_engineer = RealTimeFeatureEngineer()

class Transaction(BaseModel):
    id: str
    date: str
    client_id: int
    card_id: int
    amount: str
    use_chip: str
    merchant_id: int
    merchant_city: str
    merchant_state: Optional[str] = None
    zip: Optional[float] = None
    mcc: int
    errors: Optional[str] = None

@app.post("/predict")
def predict(transaction: Transaction):
    try:
        raw_data = transaction.dict()

        # Feature engineering
        merged_df = feature_engineer.merge_and_clean(raw_data)
        transformed_df = sg_preprocessor.transform(merged_df)
        transformed_df = transformed_df.reindex(columns=feature_list, fill_value=0)

        # Ensure numerical data
        final_df = pd.DataFrame([transformed_df.iloc[0]]).astype("float32")

        # Prediction
        risk_score = float(model.predict_proba(final_df)[:, 1][0])
        prediction = "FRAUD" if risk_score >= pc.MODEL_TRAINER_THRESHOLD else "LEGIT"

        # SHAP explanations
        shap_values = explainer(final_df)
        shap_contribs = shap_values[0].values
        top_features = sorted(
            [(feat, val) for feat, val in zip(feature_list, shap_contribs) if final_df.iloc[0][feat] != 0],
            key=lambda x: abs(x[1]),
            reverse=True
        )[:pc.SHAP_TOP_N_FEATURES]

        
        return {
            "risk_score": float(risk_score),
            "prediction": prediction,
            "top_features": [(str(f), float(v)) for f, v in top_features]
        }


    except Exception as e:
        raise SmartGuardException(e, sys)

    