import os
import shap
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from frauddetection.utils.main_utils import load_object

class SHAPExplainer:
    def __init__(self, model_path: str, data_path: str, output_dir: str = "artifacts/shap_outputs", sample_size: int = 100_000, top_n: int = 3):
        self.model_path = model_path
        self.data_path = data_path
        self.output_dir = output_dir
        self.sample_size = sample_size
        self.top_n = top_n
        os.makedirs(self.output_dir, exist_ok=True)

    def load_data(self):
        df = pd.read_parquet(self.data_path)
        return df.drop(columns=["fraud_label"])

    def load_model(self):
        return load_object(self.model_path)

    def generate_shap_values(self, model, X_sample):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        return explainer, shap_values

    def plot_global_summary(self, shap_values, X_sample):
        shap.summary_plot(shap_values, X_sample, show=False)
        plt.title("Global Feature Importance (SHAP Summary)")
        plt.savefig(os.path.join(self.output_dir, "global_shap_summary.png"), bbox_inches='tight')
        plt.close()

    def plot_dependence_top_features(self, shap_values, X_sample):
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        top_features = X_sample.columns[np.argsort(mean_abs_shap)[-self.top_n:]]

        for feature in top_features:
            shap.dependence_plot(feature, shap_values, X_sample, show=False)
            plt.title(f"Dependence Plot: {feature}")
            plt.savefig(os.path.join(self.output_dir, f"dependence_{feature}.png"), bbox_inches='tight')
            plt.close()

    def run(self):
        print("Loading model and data")
        model = self.load_model()
        X = self.load_data()

        print(f"Sampling {self.sample_size} rows for SHAP analysis")
        X_sample = X.sample(n=min(self.sample_size, len(X)), random_state=42)

        print("Generating SHAP explanations")
        explainer, shap_values = self.generate_shap_values(model, X_sample)

        print("Plotting global summary")
        self.plot_global_summary(shap_values, X_sample)

        print(f"Plotting top {self.top_n} feature dependence plots")
        self.plot_dependence_top_features(shap_values, X_sample)

        print(f"SHAP visualizations saved to: {self.output_dir}")
