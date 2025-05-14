from frauddetection.entity.config_entity import (
    TrainingPipelineConfig,
    DataIngestionConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
)
from frauddetection.components.data_ingestion import DataIngestion
from frauddetection.components.data_transformation import DataTransformation
from frauddetection.components.model_trainer import ModelTrainer
from frauddetection.components.model_validator import ModelValidator
from frauddetection.constants import pipeline_constants as pc


def run_pipeline():
    print("Starting SmartGuard Fraud Detection Pipeline...")

    # Step 1: Set up pipeline config
    pipeline_config = TrainingPipelineConfig()

    # Step 2: Data Ingestion
    ingestion_config = DataIngestionConfig(training_pipeline_config=pipeline_config)
    ingestion = DataIngestion(config=ingestion_config)
    ingestion_artifact = ingestion.ingest_data()
    print(" Data Ingestion completed.\n")

    # Step 3: Data Transformation
    transformation_config = DataTransformationConfig(training_pipeline_config=pipeline_config)
    transformation = DataTransformation(config=transformation_config)
    transformation_artifact = transformation.initiate_data_transformation()
    print(" Data Transformation completed.\n")

    # Step 4: Model Training
    trainer_config = ModelTrainerConfig(training_pipeline_config=pipeline_config)
    trainer = ModelTrainer(model_trainer_config=trainer_config, data_transformation_artifact=transformation_artifact)
    model_artifact = trainer.initiate_model_trainer()
    print("Model Training completed.")
    print(f" Saved model to: {model_artifact.trained_model_path}")
    print(f" Feature list at: {model_artifact.feature_list_path}\n")

    # Step 5: Final Validation
    validator = ModelValidator(
        model_path=model_artifact.trained_model_path,
        data_path=transformation_config.transformed_file_path,
        threshold=pc.MODEL_TRAINER_THRESHOLD
    )
    validator.validate()
    print("Final Model Validation completed.\n")


if __name__ == "__main__":
    run_pipeline()
