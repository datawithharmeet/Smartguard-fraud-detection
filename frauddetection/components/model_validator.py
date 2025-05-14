import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from frauddetection.utils.main_utils import load_object

class ModelValidator:
    def __init__(self, model_path: str, data_path: str, threshold: float):
        self.model_path = model_path
        self.data_path = data_path
        self.threshold = threshold

    def validate(self):
        model = load_object(self.model_path)
        df = pd.read_parquet(self.data_path)

        X = df.drop(columns=["fraud_label"])
        y = df["fraud_label"]

        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=21)
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= self.threshold).astype(int)

        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)

        print("Final Validation at Threshold =", self.threshold)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)
        print("ROC AUC:", auc)
        print("Confusion Matrix:\n", cm)

        # Optional: Plot score distribution
        plt.figure(figsize=(8, 4))
        plt.hist(y_prob, bins=100, alpha=0.7, label='Risk Scores')
        plt.axvline(self.threshold, color='red', linestyle='--', label=f'Threshold = {self.threshold}')
        plt.title("Risk Score Distribution on Test Set")
        plt.xlabel("Fraud Risk Score")
        plt.ylabel("Frequency")
        plt.legend()
        plt.tight_layout()
        plt.show()

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": auc,
            "confusion_matrix": cm
        }
