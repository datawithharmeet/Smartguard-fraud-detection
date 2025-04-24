import pandas as pd
import lightgbm as lgb
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)

# --- Config ---
INPUT_PATH = "artifacts/model_input.parquet"
MODEL_PATH = "artifacts/fraud_model.pkl"
TARGET_COL = "fraud_label"

def load_data():
    print("🔹 Loading data...")
    return pd.read_parquet(INPUT_PATH)

def prepare_data(df):
    print("🔹 Splitting train/test...")
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

def compute_class_weight(y_train):
    print("Computing class imbalance weight...")
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    weight = n_neg / n_pos
    print(f"scale_pos_weight = {weight:.2f}")
    return weight

def train_model(X_train, y_train, scale_pos_weight):
    print("🔹 Training LightGBM model...")
    model = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        class_weight=None,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train, eval_metric="auc")
    return model

def evaluate(model, X_test, y_test):
    print("🔹 Evaluating model...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\n Classification Report:\n", classification_report(y_test, y_pred))
    print(" Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print(f" ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")
    print(f" Precision: {precision_score(y_test, y_pred):.4f}")
    print(f" Recall: {recall_score(y_test, y_pred):.4f}")
    print(f" F1 Score: {f1_score(y_test, y_pred):.4f}")

def save_model(model):
    os.makedirs("artifacts", exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f" Model saved to {MODEL_PATH}")

def main():
    df = load_data()
    X_train, X_test, y_train, y_test = prepare_data(df)
    weight = compute_class_weight(y_train)
    model = train_model(X_train, y_train, weight)
    evaluate(model, X_test, y_test)
    save_model(model)

if __name__ == "__main__":
    main()
