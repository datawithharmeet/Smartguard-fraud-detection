import os
import pandas as pd
import shap
import pickle
import numpy as np
from frauddetection.utils.main_utils import load_object

MODEL_PATH = "artifacts/model_trainer/fraud_model.pkl"
PREPROCESSOR_PATH = "artifacts/data_transformation/preprocessing.pkl"
THRESHOLD = 0.8

def load_model_and_preprocessor():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(PREPROCESSOR_PATH):
        os.system(f"aws s3 cp s3://smartguard-artifacts/fraud_model.pkl {MODEL_PATH}")
        os.system(f"aws s3 cp s3://smartguard-artifacts/preprocessing.pkl {PREPROCESSOR_PATH}")
    model = load_object(MODEL_PATH)
    preprocessor = load_object(PREPROCESSOR_PATH)
    return model, preprocessor

def predict_transaction(transaction_df, model, preprocessor, explain=False):
    X = preprocessor.transform(transaction_df)
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= THRESHOLD).astype(int)

    shap_features = None
    if explain:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        mean_shap = np.abs(shap_values).flatten()
        top_indices = np.argsort(mean_shap)[::-1][:3]
        shap_features = {X.columns[i]: round(mean_shap[i], 4) for i in top_indices}

    return y_pred[0], y_prob[0], shap_features
