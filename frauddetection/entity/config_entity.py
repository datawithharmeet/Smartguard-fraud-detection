import os
from frauddetection.constants import pipeline_constants as pc


class TrainingPipelineConfig:
    def __init__(self):
        self.pipeline_name = pc.PIPELINE_NAME
        self.artifact_name = pc.ARTIFACT_DIR
        self.artifact_dir = pc.ARTIFACT_DIR
        self.model_dir = pc.SAVED_MODEL_DIR


class DataIngestionConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.data_ingestion_dir = os.path.join(
            training_pipeline_config.artifact_dir,
            pc.DATA_INGESTION_DIR_NAME
        )

        self.feature_store_file_path = os.path.join(
            self.data_ingestion_dir,
            pc.DATA_INGESTION_FEATURE_STORE_DIR,
            pc.FILE_NAME
        )

        self.training_file_path = os.path.join(
            self.data_ingestion_dir,
            pc.DATA_INGESTION_INGESTED_DIR,
            pc.TRAIN_FILE_NAME
        )

        self.testing_file_path = os.path.join(
            self.data_ingestion_dir,
            pc.DATA_INGESTION_INGESTED_DIR,
            pc.TEST_FILE_NAME
        )

        self.train_test_split_ratio = pc.DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO

class DataTransformationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.data_transformation_dir = os.path.join(
            training_pipeline_config.artifact_dir,
            pc.DATA_TRANSFORMATION_DIR_NAME
        )

        self.train_file_path = os.path.join(
            training_pipeline_config.artifact_dir,
            pc.DATA_INGESTION_DIR_NAME,
            pc.DATA_INGESTION_INGESTED_DIR,
            pc.TRAIN_FILE_NAME
        )

        self.test_file_path = os.path.join(
            training_pipeline_config.artifact_dir,
            pc.DATA_INGESTION_DIR_NAME,
            pc.DATA_INGESTION_INGESTED_DIR,
            pc.TEST_FILE_NAME
        )

        self.transformed_train_file_path = os.path.join(
            self.data_transformation_dir,
            pc.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
            pc.TRAIN_FILE_NAME.replace("parquet", "parquet")
        )

        self.transformed_test_file_path = os.path.join(
            self.data_transformation_dir,
            pc.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
            pc.TEST_FILE_NAME.replace("parquet", "parquet")
        )

        self.transformer_object_path = os.path.join(
            self.data_transformation_dir,
            pc.DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
            pc.PREPROCESSING_OBJECT_FILE_NAME
        )


class ModelTrainerConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.model_trainer_dir = os.path.join(
            training_pipeline_config.artifact_dir,
            pc.MODEL_TRAINER_DIR_NAME
        )
        self.trained_model_path = os.path.join(
            self.model_trainer_dir,
            pc.MODEL_FILE_NAME  
        )
        self.feature_list_path = os.path.join(
            self.model_trainer_dir,
            pc.FEATURE_LIST_FILE 
        )
        self.train_data_path = os.path.join(
            training_pipeline_config.artifact_dir,
            pc.DATA_TRANSFORMATION_DIR_NAME,
            pc.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
            pc.TRAIN_FILE_NAME
        )

        self.test_data_path = os.path.join(
            training_pipeline_config.artifact_dir,
            pc.DATA_TRANSFORMATION_DIR_NAME,
            pc.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
            pc.TEST_FILE_NAME
        )
        self.threshold = pc.MODEL_TRAINER_THRESHOLD
    
class SHAPExplainerConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.model_path = os.path.join(
            training_pipeline_config.artifact_dir,
            pc.MODEL_TRAINER_DIR_NAME,
            pc.MODEL_FILE_NAME
        )
        self.data_path = os.path.join(
            training_pipeline_config.artifact_dir,
            pc.DATA_TRANSFORMATION_DIR_NAME,
            pc.PREPROCESSED_FILE_NAME
        )
        self.output_dir = os.path.join(
            training_pipeline_config.artifact_dir,
            pc.SHAP_OUTPUT_DIR
        )

        self.sample_size = pc.SHAP_SAMPLE_SIZE
        self.top_n = pc.SHAP_TOP_N_FEATURES


class KafkaProducerConfig:
    def __init__(self):
        self.data_path = os.path.join("data", "unlabeled_transactions.parquet")
        self.kafka_topic = pc.KAFKA_TOPIC
        self.kafka_broker = pc.KAFKA_BROKER
        self.delay = pc.KAFKA_DELAY
        self.sample_size = pc.KAFKA_SAMPLE_SIZE
    
        

class KafkaConsumerConfig:
    def __init__(self):
        self.kafka_topic = pc.KAFKA_TOPIC
        self.kafka_broker = pc.KAFKA_BROKER

        self.model_path = os.path.join(pc.ARTIFACT_DIR, pc.MODEL_TRAINER_DIR_NAME, pc.MODEL_FILE_NAME)
        self.transformer_path = os.path.join(pc.ARTIFACT_DIR, pc.DATA_TRANSFORMATION_DIR_NAME, pc.PREPROCESSING_OBJECT_FILE_NAME)
        self.feature_list_path = os.path.join(pc.ARTIFACT_DIR, pc.MODEL_TRAINER_DIR_NAME, pc.FEATURE_LIST_FILE)
        self.shap_explainer_path = os.path.join(pc.ARTIFACT_DIR, pc.SHAP_OUTPUT_DIR, pc.SHAP_EXPLAINER_FILE)


        self.output_csv_path = os.path.join(pc.KAFKA_STREAMING_DIR_NAME, pc.KAFKA_OUTPUT_LOG_FILE_NAME)
        
        self.shap_top_n = pc.SHAP_TOP_N_FEATURES
        self.threshold = pc.MODEL_TRAINER_THRESHOLD

