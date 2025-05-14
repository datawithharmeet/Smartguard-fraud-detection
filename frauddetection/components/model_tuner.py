import os
import pandas as pd
import numpy as np
import xgboost as xgb
import mlflow
import dagshub

from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

from frauddetection.entity.config_entity import ModelTrainerConfig
from frauddetection.entity.artifact_entity import DataTransformationArtifact
from frauddetection.constants import pipeline_constants as pc
from frauddetection.utils.main_utils import save_object, load_object


dagshub.init(repo_owner="datawithharmeet", repo_name="Smartguard-fraud-detection", mlflow=True)

class HyperparameterTuner:
    def __init__(self, model_trainer_config: ModelTrainerConfig, data_transformation_artifact: DataTransformationArtifact):
        self.config = model_trainer_config
        self.transformation_artifact = data_transformation_artifact

    def sample_data(self):
        df = pd.read_parquet(self.transformation_artifact.transformed_file_path)
        df_sample = df.sample(n=1_000_000, random_state=42) if len(df) > 1_000_000 else df
        X = df_sample.drop(columns=["fraud_label"])
        y = df_sample["fraud_label"]
        return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    def run_random_search(self, X_train, y_train):
        param_grid = {
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5, 7],
            'min_child_weight': [1, 5, 10],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'n_estimators': [300, 500, 1000],
            'gamma': [0, 0.1, 0.2],
            'scale_pos_weight': [650, 667, 700]
        }

        xgb_clf = xgb.XGBClassifier(
            objective='binary:logistic',
            use_label_encoder=False,
            eval_metric='auc',
            n_jobs=-1,
            random_state=42,
            verbosity=0
        )

        search = RandomizedSearchCV(
            estimator=xgb_clf,
            param_distributions=param_grid,
            n_iter=30,
            scoring='roc_auc',
            n_jobs=-1,
            cv=5,
            verbose=2,
            random_state=42
        )

        print("Running RandomizedSearchCV...")
        search.fit(X_train, y_train)
        return search

    def evaluate(self, model, X_test, y_test):
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        return {
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_prob)
        }

    def tune_model(self):
        X_train, X_test, y_train, y_test = self.sample_data()

        with mlflow.start_run(run_name="XGBoost_Tuning_Stage3"):
            search = self.run_random_search(X_train, y_train)
            best_model = search.best_estimator_
            best_params = search.best_params_
            mlflow.log_params(best_params)

            metrics = self.evaluate(best_model, X_test, y_test)
            mlflow.log_metrics(metrics)

            print(" Best Parameters:", best_params)
            print(" Evaluation:", metrics)

            os.makedirs(os.path.dirname(self.config.trained_model_path), exist_ok=True)
            save_object(best_model, os.path.join("artifacts/model_trainer", "xgboost_tuned_model.pkl"))

            return best_model, best_params, metrics
