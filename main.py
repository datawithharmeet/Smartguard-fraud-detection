import sys
from frauddetection.entity.config_entity import (
    TrainingPipelineConfig,
    DataIngestionConfig,
    DataTransformationConfig,
    ModelTrainerConfig
)
from frauddetection.components.data_ingestion import DataIngestion
from frauddetection.components.data_transformation import DataTransformation
from frauddetection.components.model_trainer import ModelTrainer
from frauddetection.logging.logger import logger
from frauddetection.exception.exception import SmartGuardException

def start_training_pipeline():
    try:
        print("\n🔁 Starting SmartGuard training pipeline...\n")

        # === 1. Training pipeline config
        pipeline_config = TrainingPipelineConfig()

        # === 2. Data Ingestion
        print("Step 1: Running Data Ingestion...")
        ingestion_config = DataIngestionConfig(training_pipeline_config=pipeline_config)
        ingestion = DataIngestion(config=ingestion_config)
        ingestion_artifact = ingestion.ingest_data()

        print(f"Data Ingestion complete.")
    
        # - Data Transformation
        logger.info(" Step 2: Starting Data Transformation...")
        transformation_config = DataTransformationConfig(training_pipeline_config=pipeline_config)
        transformation = DataTransformation(config=transformation_config)
        transformation_artifact = transformation.initiate_data_transformation()

        logger.info("Data Transformation completed successfully.")

        # - Model Trainer
        logger.info("Step 3: Running Model Training...")
        trainer_config = ModelTrainerConfig(training_pipeline_config=pipeline_config)
        trainer = ModelTrainer(model_trainer_config=trainer_config, data_transformation_artifact=transformation_artifact)
        trainer_artifact = trainer.initiate_model_trainer()

        logger.info("Model Training Completed Successfully.")
        logger.info(f"Model Path: {trainer_artifact.trained_model_path}")
        logger.info(f"Train Metrics: {trainer_artifact.train_metric}")
        logger.info(f"Test Metrics : {trainer_artifact.test_metric}")
        # - Model Validator
        # - SHAP
        # - Deployment
        # etc.

    except Exception as e:
        logger.error(SmartGuardException(e, sys))
        sys.exit(1)


if __name__ == "__main__":
    start_training_pipeline()
