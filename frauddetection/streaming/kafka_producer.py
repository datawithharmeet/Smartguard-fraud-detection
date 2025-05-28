import time
import json
from kafka import KafkaProducer
import pandas as pd
import numpy as np

from frauddetection.entity.config_entity import KafkaProducerConfig
from frauddetection.utils.main_utils import read_parquet

class TransactionKafkaProducer:
    def __init__(self, config: KafkaProducerConfig):
        self.config = config
        self.producer = KafkaProducer(
            bootstrap_servers=[self.config.kafka_broker],
            value_serializer=lambda record: json.dumps(record).encode("utf-8")
        )

    def load_data(self) -> pd.DataFrame:
        df = read_parquet(self.config.data_path)
        if self.config.sample_size:
            df = df.head(self.config.sample_size)
        return df

    def stream(self):
        df = self.load_data()
        print(f"Streaming to Kafka topic: {self.config.kafka_topic}")
        for i, row in df.iterrows():
            row = row.replace({np.nan: None, np.inf: None, -np.inf: None})
            self.producer.send(self.config.kafka_topic, value=row.to_dict())
            print(f"Sent transaction {i+1}/{len(df)}: ID={row.get('id')}")
            time.sleep(self.config.delay)
        print("Completed streaming.")


if __name__ == "__main__":
    config = KafkaProducerConfig()
    producer = TransactionKafkaProducer(config)
    producer.stream()