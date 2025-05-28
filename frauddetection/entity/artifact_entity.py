from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class DataIngestionArtifact:
    feature_store_file_path: str
    training_file_path: str
    testing_file_path: str


@dataclass
class DataTransformationArtifact:
    transformed_train_file_path: str
    transformed_test_file_path: str
    transformer_object_path: str

@dataclass
class ClassificationMetricArtifact:
    precision: float
    recall: float
    f1_score: float
    auc_pr: float

@dataclass
class ModelTrainerArtifact:
    trained_model_path: str
    feature_list_path: str
    train_metric: ClassificationMetricArtifact
    test_metric: ClassificationMetricArtifact
    model_type: str
    
@dataclass
class SHAPExplainerArtifact:
    output_dir: str
    summary_plot_path: str
    dependence_plot_paths: list
    
@dataclass
class KafkaConsumerArtifact:
    raw_transaction: dict
    risk_score: float
    prediction: int
    top_features: List[Tuple[str, float]]
