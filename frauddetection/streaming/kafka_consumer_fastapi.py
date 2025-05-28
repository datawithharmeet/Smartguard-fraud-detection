import os
import json
import time
import requests
import pandas as pd
from kafka import KafkaConsumer
from frauddetection.entity.config_entity import KafkaConsumerConfig

class FastAPIBasedKafkaConsumer:
    def __init__(self, config: KafkaConsumerConfig):
        self.config = config
        self.api_url = "http://127.0.0.1:8000/predict"

        self.consumer = KafkaConsumer(
            self.config.kafka_topic,
            bootstrap_servers=self.config.kafka_broker,
            auto_offset_reset="latest",
            enable_auto_commit=True,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        )

        self._init_csv_log()

    def _init_csv_log(self):
        with open(self.config.output_csv_path, "w") as f:
            f.write("raw,risk_score,prediction,top_features\n")

    def consume(self):
        print("Listening for transactions (FastAPI)...")
        for message in self.consumer:
            transaction = message.value
            try:
                response = requests.post(self.api_url, json=transaction, timeout=5)
                if response.status_code == 200:
                    result = response.json()
                    self._log_result(transaction, result)
                else:
                    print(f"FastAPI error: {response.status_code} {response.text}")
            except Exception as e:
                print(f"Error processing transaction: {e}")

    def _log_result(self, transaction, result):
        print("New Transaction:")
        print(json.dumps(transaction, indent=2))
        print(f"Risk Score: {result['risk_score']:.4f}")
        print(f"Prediction: {result['prediction']}")
        print("Top SHAP Features:")
        for feat, val in result["top_features"]:
            print(f"  - {feat}: {val:.4f}")

        log_df = pd.DataFrame([{
            "raw": json.dumps(transaction),
            "risk_score": result["risk_score"],
            "prediction": result["prediction"],
            "top_features": json.dumps(result["top_features"])
        }])
        log_df.to_csv(self.config.output_csv_path, mode="a", header=False, index=False)
        with open(self.config.output_csv_path, "a") as f:
            f.flush()
            os.fsync(f.fileno())

if __name__ == "__main__":
    from frauddetection.entity.config_entity import KafkaConsumerConfig
    config = KafkaConsumerConfig()
    consumer = FastAPIBasedKafkaConsumer(config)
    consumer.consume()
