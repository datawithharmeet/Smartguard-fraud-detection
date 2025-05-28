import os
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from frauddetection.entity.config_entity import SHAPExplainerConfig
from frauddetection.entity.artifact_entity import SHAPExplainerArtifact
from frauddetection.utils.main_utils import load_object, save_object


class SHAPExplainer:
    def __init__(self, config: SHAPExplainerConfig):
        self.config = config
        os.makedirs(self.config.output_dir, exist_ok=True)

    def load_data(self):
        df = pd.read_parquet(self.config.data_path)
        df = df.drop(columns=["fraud_label"])

    # Load final feature list
        with open(os.path.join("artifacts", "model_trainer", "final_features.txt")) as f:
            feature_list = [line.strip() for line in f]

    # Filter columns to match training features
        df = df[feature_list]
        return df
    
    def load_model(self):
        return load_object(self.config.model_path)

    def generate_shap_values(self, model, X_sample):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        return explainer, shap_values

    def plot_global_summary(self, shap_values, X_sample):
        summary_path = os.path.join(self.config.output_dir, "global_shap_summary.png")
        shap.summary_plot(shap_values, X_sample, show=False)
        plt.title("Global Feature Importance (SHAP Summary)")
        plt.savefig(summary_path, bbox_inches='tight')
        plt.close()
        return summary_path

    def plot_dependence_top_features(self, shap_values, X_sample):
        dependence_paths = []
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        top_features = X_sample.columns[np.argsort(mean_abs_shap)[-self.config.top_n:]]

        for feature in top_features:
            shap.dependence_plot(feature, shap_values, X_sample, show=False)
            file_path = os.path.join(self.config.output_dir, f"dependence_{feature}.png")
            plt.title(f"Dependence Plot: {feature}")
            plt.savefig(file_path, bbox_inches='tight')
            plt.close()
            dependence_paths.append(file_path)

        return dependence_paths

    def run(self) -> SHAPExplainerArtifact:
        model = self.load_model()
        X = self.load_data()
        X_sample = X.sample(n=min(self.config.sample_size, len(X)), random_state=42)

        explainer, shap_values = self.generate_shap_values(model, X_sample)

        summary_path = self.plot_global_summary(shap_values, X_sample)
        dependence_paths = self.plot_dependence_top_features(shap_values, X_sample)

        # Save SHAP explainer object
        explainer_path = os.path.join(self.config.output_dir, "shap_explainer.pkl")
        save_object(explainer, explainer_path)

        return SHAPExplainerArtifact(
            output_dir=self.config.output_dir,
            summary_plot_path=summary_path,
            dependence_plot_paths=dependence_paths
        )




