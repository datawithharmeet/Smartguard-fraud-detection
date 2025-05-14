import os
import sys
import pandas as pd
import numpy as np
import mlflow
import dagshub
import pickle

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split

from frauddetection.entity.config_entity import ModelTrainerConfig
from frauddetection.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact, ClassificationMetricArtifact
from frauddetection.constants import pipeline_constants as pc
from frauddetection.utils.main_utils import save_object

dagshub.init(repo_owner='datawithharmeet', repo_name='Smartguard-fraud-detection', mlflow=True)

class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig, data_transformation_artifact: DataTransformationArtifact):
        self.config = model_trainer_config
        self.transformation_artifact = data_transformation_artifact

    def evaluate(self, model, X_test, y_test, threshold=0.5):
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)

        return ClassificationMetricArtifact(
            precision=precision_score(y_test, y_pred),
            recall=recall_score(y_test, y_pred),
            f1_score=f1_score(y_test, y_pred),
            roc_auc=roc_auc_score(y_test, y_prob)
        )

    def get_model(self, name, weight):
        if name == "lightgbm":
            return LGBMClassifier(n_estimators=1000, learning_rate=0.05, scale_pos_weight=weight, random_state=42, n_jobs=-1)
        elif name == "xgboost":
            return XGBClassifier(n_estimators=1000, learning_rate=0.1, subsample=0.6, max_depth=3,
                                 min_child_weight=1, gamma=0, colsample_bytree=1.0,
                                 scale_pos_weight=weight, random_state=42,
                                 use_label_encoder=False, eval_metric='logloss', n_jobs=-1)
        elif name == "randomforest":
            return RandomForestClassifier(n_estimators=100, class_weight={0: 1, 1: weight}, random_state=42, n_jobs=-1)
        else:
            raise ValueError(f"Unsupported model: {name}")

    def train_model(self, X_train, y_train, X_test, y_test):
        best_model = None
        best_metrics = None
        best_model_name = None
        best_recall = 0
        best_precision = 0

        weight = (len(y_train) - sum(y_train)) / sum(y_train)

        for model_name in ["xgboost"]:  # Only run XGBoost
            model = self.get_model(model_name, weight)

            with mlflow.start_run(run_name=model_name):
                mlflow.log_param("model_type", model_name)
                mlflow.log_param("scale_pos_weight", weight)
                mlflow.log_param("threshold", pc.MODEL_TRAINER_THRESHOLD)

                model.fit(X_train, y_train)
                metrics = self.evaluate(model, X_test, y_test, threshold=pc.MODEL_TRAINER_THRESHOLD)

                mlflow.log_metric("precision", metrics.precision)
                mlflow.log_metric("recall", metrics.recall)
                mlflow.log_metric("f1", metrics.f1_score)
                mlflow.log_metric("roc_auc", metrics.roc_auc)

                print(f"\n {model_name.upper()} completed: Precision={metrics.precision:.4f}, Recall={metrics.recall:.4f}, F1={metrics.f1_score:.4f}, AUC={metrics.roc_auc:.4f}")

                is_better = False
                if metrics.recall > best_recall:
                    is_better = True
                elif abs(metrics.recall - best_recall) <= 0.01 and metrics.precision > best_precision:
                    is_better = True

                if is_better:
                    best_model = model
                    best_model_name = model_name
                    best_metrics = metrics
                    best_recall = metrics.recall
                    best_precision = metrics.precision

        print(f"\n Best model: {best_model_name.upper()} | Recall: {best_recall:.4f}, Precision: {best_precision:.4f}")
        return best_model, best_metrics, best_model_name

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        # Load data
        df = pd.read_parquet(self.transformation_artifact.transformed_file_path)
        X = df.drop(columns=["fraud_label"])
        y = df["fraud_label"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

        # Train & get best model
        best_model, best_metrics, best_model_name = self.train_model(X_train, y_train, X_test, y_test)

        # Save model
        os.makedirs(os.path.dirname(self.config.trained_model_path), exist_ok=True)
        save_object(best_model, self.config.trained_model_path)

        # Save feature list
        with open(self.config.feature_list_path, "w") as f:
            for col in X_train.columns:
                f.write(f"{col}\n")

        return ModelTrainerArtifact(
            trained_model_path=self.config.trained_model_path,
            feature_list_path=self.config.feature_list_path,
            train_metric=best_metrics,
            test_metric=best_metrics,
            model_type=best_model_name
        )
