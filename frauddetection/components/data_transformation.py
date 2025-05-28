import os
import yaml
import pandas as pd

from frauddetection.entity.config_entity import DataTransformationConfig
from frauddetection.entity.artifact_entity import DataTransformationArtifact
from frauddetection.utils.main_utils import save_parquet, save_object
from frauddetection.components.smartgaurd_preproceesor import SmartGuardPreprocessor


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        # Load feature config
        with open("config/feature_config.yaml", "r") as f:
            feature_cfg = yaml.safe_load(f)

        # Load training and test data
        train_df = pd.read_parquet(self.config.train_file_path)
        test_df = pd.read_parquet(self.config.test_file_path)

        # Fit on training set only
        transformer = SmartGuardPreprocessor(feature_cfg)
        transformer.fit(train_df)

        # Transform both sets
        train_transformed = transformer.transform(train_df)
        test_transformed = transformer.transform(test_df)

        # Select encoded + included features
        include = feature_cfg["features"]["include"]
        additional = [
            col for col in train_transformed.columns
            for f in feature_cfg["encoding"]
            if feature_cfg["encoding"][f] == "one_hot" and f in col
        ]
        selected_cols = include + additional + ["fraud_label"]
        train_transformed = train_transformed[[c for c in selected_cols if c in train_transformed.columns]]
        test_transformed = test_transformed[[c for c in selected_cols if c in test_transformed.columns]]

        # Drop intermediate columns (if still present)
        cols_to_drop = [
            'amount', 'credit_limit', 'yearly_income', 'total_debt',
            'amount_clean', 'credit_limit_clean', 'yearly_income_clean', 'total_debt_clean'
        ]
        train_transformed.drop(columns=cols_to_drop, errors="ignore", inplace=True)
        test_transformed.drop(columns=cols_to_drop, errors="ignore", inplace=True)

        # Save transformed files
        os.makedirs(os.path.dirname(self.config.transformed_train_file_path), exist_ok=True)
        save_parquet(train_transformed, self.config.transformed_train_file_path)
        save_parquet(test_transformed, self.config.transformed_test_file_path)

        # Save transformer
        os.makedirs(os.path.dirname(self.config.transformer_object_path), exist_ok=True)
        save_object(transformer, self.config.transformer_object_path)

        print("Data transformation completed.")
        return DataTransformationArtifact(
            transformed_train_file_path=self.config.transformed_train_file_path,
            transformed_test_file_path=self.config.transformed_test_file_path,
            transformer_object_path=self.config.transformer_object_path
        )

if __name__ == "__main__":
    from frauddetection.entity.config_entity import TrainingPipelineConfig, DataTransformationConfig

    # Initialize pipeline config
    pipeline_config = TrainingPipelineConfig()
    transformation_config = DataTransformationConfig(training_pipeline_config=pipeline_config)

    # Run transformation
    transformation = DataTransformation(config=transformation_config)
    artifact = transformation.initiate_data_transformation()

    print("\n✅ Transformation completed.")
    print(f"• Transformed Train: {artifact.transformed_train_file_path}")
    print(f"• Transformed Test : {artifact.transformed_test_file_path}")
    print(f"• Transformer Path : {artifact.transformer_object_path}")
