from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    processed_file_path: str

@dataclass
class DataTransformationArtifact:
    transformer_object_path: str
    transformed_file_path: str

@dataclass
class ClassificationMetricArtifact:
    precision: float
    recall: float
    f1_score: float
    roc_auc: float

@dataclass
class ModelTrainerArtifact:
    trained_model_path: str
    feature_list_path: str
    train_metric: ClassificationMetricArtifact
    test_metric: ClassificationMetricArtifact
    model_type: str