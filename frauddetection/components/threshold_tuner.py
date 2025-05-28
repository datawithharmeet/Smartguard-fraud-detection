import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

from frauddetection.constants import pipeline_constants as pc
from frauddetection.utils.main_utils import load_object


class ThresholdTuner:
    def __init__(self, model_path: str, data_path: str, threshold_range=(0.5, 0.91, 0.05)):
        self.model_path = model_path
        self.data_path = data_path
        self.thresholds = np.arange(*threshold_range)

    def evaluate_thresholds(self):
        model = load_object(self.model_path)
        df = pd.read_parquet(self.data_path)

        X = df.drop(columns=["fraud_label"])
        y = df["fraud_label"]

        # Align features with training
        with open("artifacts/model_trainer/final_features.txt", "r") as f:
            required_features = [line.strip() for line in f.readlines()]
    
        for col in required_features:
            if col not in X.columns:
                X[col] = 0  # Add missing columns with default zero
  
        X = X[required_features]  # Ensure correct column order

        y_prob = model.predict_proba(X)[:, 1]

        results = []
        for t in self.thresholds:
            y_pred = (y_prob >= t).astype(int)
            precision = precision_score(y, y_pred, zero_division=0)
            recall = recall_score(y, y_pred)
            f1 = f1_score(y, y_pred)
            results.append({
                "threshold": round(t, 2),
                "precision": precision,
                "recall": recall,
                "f1": f1
            })

        df_results = pd.DataFrame(results)
        return df_results

    def find_best_threshold(self, df_results: pd.DataFrame, min_recall=0.8):
        candidates = df_results[df_results["recall"] >= min_recall]
        if not candidates.empty:
            best_row = candidates.sort_values(by="precision", ascending=False).iloc[0]
        else:
            best_row = df_results.sort_values(by="recall", ascending=False).iloc[0]
        return best_row["threshold"], best_row

    def tune_and_return(self):
        df_results = self.evaluate_thresholds()
        best_threshold, best_row = self.find_best_threshold(df_results)

        # Save plot
        os.makedirs("artifacts/threshold_tuning", exist_ok=True)
        plt.figure(figsize=(8, 5))
        plt.plot(df_results["threshold"], df_results["precision"], label="Precision")
        plt.plot(df_results["threshold"], df_results["recall"], label="Recall")
        plt.plot(df_results["threshold"], df_results["f1"], label="F1 Score")
        plt.axvline(x=best_threshold, color="gray", linestyle="--", label=f"Best Threshold = {best_threshold:.2f}")
        plt.xlabel("Threshold")
        plt.ylabel("Score")
        plt.title("Threshold vs. Metrics")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("artifacts/threshold_tuning/threshold_metrics_plot.png")

        # Save best threshold
        with open("artifacts/threshold_tuning/best_threshold.txt", "w") as f:
            f.write(f"{best_threshold:.2f}")

        print(" Best Threshold Found:")
        print(best_row)

        return best_threshold, best_row


