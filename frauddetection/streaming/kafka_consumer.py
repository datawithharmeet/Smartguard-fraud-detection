import os
import json
import pandas as pd
import shap
from kafka import KafkaConsumer

from frauddetection.utils.main_utils import load_object
from frauddetection.entity.config_entity import KafkaConsumerConfig
from frauddetection.entity.artifact_entity import KafkaConsumerArtifact
from frauddetection.streaming.feature_engineering import RealTimeFeatureEngineer


class TransactionKafkaConsumer:
    def __init__(self, config: KafkaConsumerConfig):
        self.config = config

        self.consumer = KafkaConsumer(
            self.config.kafka_topic,
            bootstrap_servers=self.config.kafka_broker,
            auto_offset_reset="latest",
            enable_auto_commit=True,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        )

        self.model = load_object(self.config.model_path)
        self.sg_preprocessor = load_object(self.config.transformer_path)
        with open(self.config.feature_list_path) as f:
            self.feature_list = [line.strip() for line in f]

        self.explainer = load_object(self.config.shap_explainer_path)

        self.feature_engineer = RealTimeFeatureEngineer()

        self._init_csv_log()

    def _init_csv_log(self):
        #  Always overwrite the CSV for a fresh demo
        with open(self.config.output_csv_path, "w") as f:
            f.write("raw,risk_score,prediction,top_features\n")


    def consume(self):
        print("Listening for transactions...")
        for message in self.consumer:
            transaction = message.value
            try:
                artifact = self._process_transaction(transaction)
                self._log_and_output(artifact)
            except Exception as e:
                print(f"Error processing transaction: {e}")

    def _process_transaction(self, transaction: dict) -> KafkaConsumerArtifact:
        merged_df = self.feature_engineer.merge_and_clean(transaction)
        engineered_df = self.sg_preprocessor.transform(merged_df)
        engineered_df = engineered_df.reindex(columns=self.feature_list, fill_value=0)

        final_df = pd.DataFrame([engineered_df.iloc[0]])
        risk_score = self.model.predict_proba(final_df)[:, 1][0]
        prediction = int(risk_score >= self.config.threshold)

        shap_values = self.explainer(final_df)
        shap_contribs = shap_values[0].values

        active_features = [
            (feat, val) for feat, val in zip(self.feature_list, shap_contribs)
            if engineered_df.iloc[0][feat] != 0
        ]

        top_features = sorted(
            active_features,
            key=lambda x: abs(x[1]),
            reverse=True
        )[:self.config.shap_top_n]


        return KafkaConsumerArtifact(
            raw_transaction=transaction,
            risk_score=risk_score,
            prediction=prediction,
            top_features=top_features
        )

    def _log_and_output(self, artifact: KafkaConsumerArtifact):
        print("New Transaction:")
        print(json.dumps(artifact.raw_transaction, indent=2))
        print(f"Risk Score: {artifact.risk_score:.4f}")
        print(f"Prediction: {'FRAUD' if artifact.prediction else 'LEGIT'}")
        print("Top SHAP Features:")
        for feat, val in artifact.top_features:
            print(f"  - {feat}: {val:.4f}")

        log_df = pd.DataFrame([{
            "raw": json.dumps(artifact.raw_transaction),
            "risk_score": artifact.risk_score,
            "prediction": artifact.prediction,
            "top_features": json.dumps([
                (feat, float(val)) for feat, val in artifact.top_features
            ])
        }])

        log_df.to_csv(self.config.output_csv_path, mode="a", header=False, index=False)
        
        #  Force flush to disk immediately
        with open(self.config.output_csv_path, "a") as f:
            f.flush()
            os.fsync(f.fileno())

if __name__ == "__main__":
    from frauddetection.entity.config_entity import KafkaConsumerConfig

    config = KafkaConsumerConfig()
    consumer = TransactionKafkaConsumer(config)
    consumer.consume()
