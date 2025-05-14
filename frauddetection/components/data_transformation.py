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

        # Load raw processed data
        df = pd.read_parquet(self.config.processed_file_path)

        # Fit and transform using SmartGuardPreprocessor
        transformer = SmartGuardPreprocessor(feature_cfg)
        transformer.fit(df)
        df_transformed = transformer.transform(df)

        # Select columns
        include = feature_cfg["features"]["include"]
        additional = [
            col for col in df_transformed.columns
            for f in feature_cfg["encoding"]
            if feature_cfg["encoding"][f] == "one_hot" and f in col
        ] 
        selected_cols = include + additional + ["fraud_label"]
        df_transformed = df_transformed[[c for c in selected_cols if c in df_transformed.columns]]

        # Drop intermediate columns
        df_transformed.drop(columns=[
            'amount', 'credit_limit', 'yearly_income', 'total_debt',
            'amount_clean', 'credit_limit_clean', 'yearly_income_clean', 'total_debt_clean'
        ], errors='ignore', inplace=True)

        # Save transformed dataset
        os.makedirs(os.path.dirname(self.config.transformed_file_path), exist_ok=True)
        save_parquet(df_transformed, self.config.transformed_file_path)

        # Save the transformer object
        os.makedirs(os.path.dirname(self.config.transformer_object_path), exist_ok=True)
        save_object(transformer, self.config.transformer_object_path)

        print(f" Data transformation completed. Saved to: {self.config.transformed_file_path}")

        return DataTransformationArtifact(
            transformed_file_path=self.config.transformed_file_path,
            transformer_object_path=self.config.transformer_object_path
        )
