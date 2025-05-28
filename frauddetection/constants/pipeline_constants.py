import os

# Global pipeline constants
PIPELINE_NAME: str = "SmartGuardFraudDetection"
ARTIFACT_DIR: str = "artifacts"
SAVED_MODEL_DIR: str = "saved_models"

# File names
FILE_NAME: str = "processed_data.parquet"
TRAIN_FILE_NAME: str = "train.parquet"
TEST_FILE_NAME: str = "test.parquet"
MODEL_FILE_NAME: str = "fraud_model.pkl"
FEATURE_LIST_FILE: str = "final_features.txt"
PREPROCESSING_OBJECT_FILE_NAME: str = "preprocessing.pkl"
PREPROCESSED_FILE_NAME: str = "model_input.parquet"

# ------------------------------
# Data Ingestion
# ------------------------------
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2

# ------------------------------
# Data Transformation
# ------------------------------
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformer"

# ------------------------------
# Model Trainer
# ------------------------------
MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_MODEL_DIR: str = "trained_model"
MODEL_TRAINER_THRESHOLD: float = 0.7

# ------------------------------
# SHAP Explanation
# ------------------------------
SHAP_OUTPUT_DIR: str = "shap_outputs"
SHAP_EXPLAINER_FILE = "shap_explainer.pkl"
SHAP_SAMPLE_SIZE: int = 100_000
SHAP_TOP_N_FEATURES: int = 3

# ------------------------------
# Kafka Streaming
# ------------------------------
KAFKA_TOPIC = "transaction_stream"
KAFKA_BROKER = "localhost:9092"
KAFKA_SAMPLE_SIZE = 10
KAFKA_DELAY = 0.1
KAFKA_STREAMING_DIR_NAME = os.path.join("frauddetection", "streaming")
KAFKA_OUTPUT_LOG_FILE_NAME = "stream_predictions.csv"
MODEL_TRAINER_PATH = os.path.join(ARTIFACT_DIR, MODEL_TRAINER_DIR_NAME)
DATA_TRANSFORMATION_PATH = os.path.join(ARTIFACT_DIR, DATA_TRANSFORMATION_DIR_NAME)
SHAP_OUTPUT_PATH = os.path.join(ARTIFACT_DIR, SHAP_OUTPUT_DIR)

