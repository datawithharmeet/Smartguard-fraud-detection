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
        self.processed_file_path = os.path.join(
            self.data_ingestion_dir,
            pc.FILE_NAME 
        )
        self.collection_name = pc.DATA_INGESTION_COLLECTION_NAME
        self.database_name = pc.DATA_INGESTION_DATABASE_NAME


class DataTransformationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.data_transformation_dir = os.path.join(
            training_pipeline_config.artifact_dir,
            pc.DATA_TRANSFORMATION_DIR_NAME
        )
        self.processed_file_path = os.path.join(
            training_pipeline_config.artifact_dir,
            pc.DATA_INGESTION_DIR_NAME,
            pc.FILE_NAME  
        )    
        self.transformed_file_path = os.path.join(
            self.data_transformation_dir,
            pc.PREPROCESSED_FILE_NAME  #
        )
        self.transformer_object_path = os.path.join(
            self.data_transformation_dir,
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
        self.input_data_path = os.path.join(
            training_pipeline_config.artifact_dir,
            pc.DATA_TRANSFORMATION_DIR_NAME,
            pc.PREPROCESSED_FILE_NAME  
        )
        self.threshold = pc.MODEL_TRAINER_THRESHOLD
