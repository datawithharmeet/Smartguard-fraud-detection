import os
import numpy as np

# Global constants
PIPELINE_NAME: str = "SmartGuardFraudDetection"
ARTIFACT_DIR: str = "artifacts"
SAVED_MODEL_DIR: str = "saved_models"

# File names
TRAIN_FILE_NAME: str = "train.parquet"
TEST_FILE_NAME: str = "test.parquet"
FILE_NAME: str = "processed_data.parquet"
MODEL_FILE_NAME: str = "fraud_model.pkl"
FEATURE_LIST_FILE: str = "final_features.txt"

# Data Ingestion
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2
DATA_INGESTION_COLLECTION_NAME: str = "fraud_collection"
DATA_INGESTION_DATABASE_NAME: str = "fraud_detection"

# Data Transformation
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformer"
PREPROCESSING_OBJECT_FILE_NAME: str = "preprocessing.pkl"