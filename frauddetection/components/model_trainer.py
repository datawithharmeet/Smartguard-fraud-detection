import os
import pandas as pd
import mlflow
import dagshub
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from frauddetection.entity.config_entity import ModelTrainerConfig,DataTransformationConfig
from frauddetection.entity.artifact_entity import ModelTrainerArtifact, ClassificationMetricArtifact, DataTransformationArtifact
from frauddetection.constants import pipeline_constants as pc
from frauddetection.utils.main_utils import save_object

dagshub.init(repo_owner='datawithharmeet', repo_name='Smartguard-fraud-detection', mlflow=True)

class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig, data_transformation_artifact: DataTransformationArtifact):
        self.config = model_trainer_config
        self.transformation_artifact = data_transformation_artifact

    def evaluate(self, model, X, y, threshold=0.5):
        y_prob = model.predict_proba(X)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)
        return ClassificationMetricArtifact(
            precision=precision_score(y, y_pred),
            recall=recall_score(y, y_pred),
            f1_score=f1_score(y, y_pred),
            auc_pr=average_precision_score(y, y_prob)
        )

    def get_model(self, name, weight):
        if name == "lightgbm":
            return LGBMClassifier(n_estimators=1000, learning_rate=0.05, subsample=0.6, max_depth=-1,
                                  num_leaves=31, colsample_bytree=1.0, scale_pos_weight=weight,
                                  random_state=42, n_jobs=-1)
        elif name == "xgboost":
            return XGBClassifier(n_estimators=1000, learning_rate=0.01, subsample=0.6, max_depth=3,
                                 min_child_weight=5, gamma=0.1, colsample_bytree=0.8,
                                 scale_pos_weight=weight, random_state=42,
                                 use_label_encoder=False, eval_metric='logloss', n_jobs=-1)
        elif name == "randomforest":
            return RandomForestClassifier(n_estimators=100, class_weight={0: 1, 1: weight},
                                          random_state=42, n_jobs=-1)
        else:
            raise ValueError(f"Unsupported model: {name}")

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        train_df = pd.read_parquet(self.transformation_artifact.transformed_train_file_path)
        test_df = pd.read_parquet(self.transformation_artifact.transformed_test_file_path)

        X = train_df.drop(columns=["fraud_label"])
        y = train_df["fraud_label"]
        X_test = test_df.drop(columns=["fraud_label"])
        y_test = test_df["fraud_label"]

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        X_train, X_val = X_train.align(X_val, join="left", axis=1, fill_value=0)

        best_model, best_model_name = self.select_best_model(X_train, y_train, X_val, y_val)
        #final_model = self.tune_model(best_model, X, y)
        final_model = best_model
        X_test = X_test.reindex(columns=X.columns, fill_value=0)
        # MLflow logging for final tuned model
        with mlflow.start_run(run_name=f"{best_model_name}_final_tuned"):
            mlflow.log_param("threshold", self.config.threshold)

            final_metrics = self.evaluate(final_model, X_test, y_test, threshold=self.config.threshold)

            mlflow.log_metric("precision", final_metrics.precision)
            mlflow.log_metric("recall", final_metrics.recall)
            mlflow.log_metric("f1", final_metrics.f1_score)
            mlflow.log_metric("auc_pr", final_metrics.auc_pr)

            mlflow.sklearn.log_model(final_model, "model")


        os.makedirs(os.path.dirname(self.config.trained_model_path), exist_ok=True)
        save_object(final_model, self.config.trained_model_path)

        with open(self.config.feature_list_path, "w") as f:
            for col in X_test.columns:
                f.write(f"{col}\n")

        print(f"Final test set performance for {best_model_name.upper()}:")
        print(f"Precision: {final_metrics.precision:.4f}, Recall: {final_metrics.recall:.4f}, "
              f"F1: {final_metrics.f1_score:.4f}, AUC_PR: {final_metrics.auc_pr:.4f}")

        return ModelTrainerArtifact(
            trained_model_path=self.config.trained_model_path,
            feature_list_path=self.config.feature_list_path,
            train_metric=None,
            test_metric=final_metrics,
            model_type=best_model_name
        )

    def select_best_model(self, X_train, y_train, X_val, y_val):
        best_model = None
        best_model_name = None
        best_recall = 0
        best_precision = 0
        best_aucpr = 0

        weight = (len(y_train) - sum(y_train)) / sum(y_train)

        for model_name in ["xgboost"]:
            print(f"\nTraining baseline model: {model_name.upper()}")
            model = self.get_model(model_name, weight)

            with mlflow.start_run(run_name=f"{model_name}_baseline"):
                model.fit(X_train, y_train)
                val_metrics = self.evaluate(model, X_val, y_val, threshold=self.config.threshold)

                mlflow.log_param("model_type", model_name)
                mlflow.log_param("scale_pos_weight", weight)
                mlflow.log_param("threshold", self.config.threshold)
                mlflow.log_metric("precision", val_metrics.precision)
                mlflow.log_metric("recall", val_metrics.recall)
                mlflow.log_metric("f1", val_metrics.f1_score)
                mlflow.log_metric("auc_pr", val_metrics.auc_pr)

                print(f"{model_name.upper()} - Precision: {val_metrics.precision:.4f}, "
                      f"Recall: {val_metrics.recall:.4f}, F1: {val_metrics.f1_score:.4f}, "
                      f"AUC_PR: {val_metrics.auc_pr:.4f}")

                is_better = False
                if val_metrics.recall > best_recall:
                    is_better = True
                elif abs(val_metrics.recall - best_recall) <= 0.01 and val_metrics.precision > best_precision:
                    is_better = True

                if is_better:
                    best_model = model
                    best_model_name = model_name
                    best_recall = val_metrics.recall
                    best_precision = val_metrics.precision
                    best_aucpr = val_metrics.auc_pr

        print(f"\nBest model after validation: {best_model_name.upper()} | "
              f"Recall: {best_recall:.4f}, Precision: {best_precision:.4f}, AUC_PR: {best_aucpr:.4f}")
        return best_model, best_model_name

    def tune_model(self, base_model, X, y):
        print("\nStarting hyperparameter tuning for the best model...")

        sample_size = 100000
        if len(X) > sample_size:
            X_sample = X.sample(sample_size, random_state=42)
            y_sample = y.loc[X_sample.index]
        else:
            X_sample = X
            y_sample = y
        
        param_dist = {
            "n_estimators": [300, 500, 1000],
            "max_depth": [3, 4, 6],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "min_child_weight": [1, 3, 5],
            "gamma": [0, 0.1, 0.3],
        }

        clf = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_dist,
            scoring="average_precision",
            cv=5,
            n_iter=20,
            n_jobs=-1,
            verbose=1,
            random_state=42
        )
        clf.fit(X_sample, y_sample)
        print("\nBest hyperparameters found:")
        print(clf.best_params_)
        return clf.best_estimator_


if __name__ == "__main__":
    from frauddetection.entity.config_entity import TrainingPipelineConfig
    from frauddetection.components.data_transformation import DataTransformation

    pipeline_config = TrainingPipelineConfig()
    transformation_config = DataTransformationConfig(training_pipeline_config=pipeline_config)
    transformation = DataTransformation(config=transformation_config)
    transformation_artifact = transformation.initiate_data_transformation()

    trainer_config = ModelTrainerConfig(training_pipeline_config=pipeline_config)
    trainer = ModelTrainer(model_trainer_config=trainer_config, data_transformation_artifact=transformation_artifact)
    artifact = trainer.initiate_model_trainer()

    print("\nModel training, hyperparameter tuning, and final evaluation complete.")
    print(f"Final test metrics: {artifact.test_metric}")
