import os
import pandas as pd
from sklearn.model_selection import train_test_split

from frauddetection.entity.config_entity import DataIngestionConfig
from frauddetection.entity.artifact_entity import DataIngestionArtifact
from frauddetection.utils.main_utils import read_csv, read_json, save_parquet


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def ingest_data(self) -> DataIngestionArtifact:
        # Load raw data
        transactions = read_csv(os.path.join("data", "transactions_data.csv"))
        cards = read_csv(os.path.join("data", "cards_data.csv"))
        users = read_csv(os.path.join("data", "users_data.csv"))
        fraud_raw = read_json(os.path.join("data", "train_fraud_labels.json"))
        mcc_dict = read_json(os.path.join("data", "mcc_codes.json"))

        # Process fraud labels
        fraud_df = pd.DataFrame(list(fraud_raw.get("target", {}).items()), columns=["id", "fraud_label"])
        fraud_df["id"] = fraud_df["id"].astype(int)
        fraud_df["fraud_label"] = fraud_df["fraud_label"].map({"Yes": 1, "No": 0})

        # Merge transactions with fraud labels
        transactions = transactions.merge(fraud_df, on="id", how="inner")

        # Merge with cards
        cards = cards.rename(columns={"id": "card_id_actual"})
        transactions = transactions.merge(cards, left_on="card_id", right_on="card_id_actual", how="left")
        transactions.drop(columns=["card_id_actual"], inplace=True)

        # Fix duplicate client_id
        if "client_id_x" in transactions.columns and "client_id_y" in transactions.columns:
            transactions.rename(columns={"client_id_x": "client_id"}, inplace=True)
            transactions.drop(columns=["client_id_y"], inplace=True)

        # Merge with users
        users = users.rename(columns={"id": "user_id"})
        transactions = transactions.merge(users, left_on="client_id", right_on="user_id", how="left")
        transactions.drop(columns=["user_id"], inplace=True)

        # MCC mapping
        transactions["mcc_desc"] = transactions["mcc"].astype(str).map(mcc_dict)

        # Save full feature store
        os.makedirs(os.path.dirname(self.config.feature_store_file_path), exist_ok=True)
        save_parquet(transactions, self.config.feature_store_file_path)

        # Split
        train_df, test_df = train_test_split(
            transactions,
            test_size=self.config.train_test_split_ratio,
            stratify=transactions["fraud_label"],
            random_state=42
        )

        # Save train and test sets
        os.makedirs(os.path.dirname(self.config.training_file_path), exist_ok=True)
        save_parquet(train_df, self.config.training_file_path)
        save_parquet(test_df, self.config.testing_file_path)

        print("✅ Data ingestion complete with train-test split.")
        return DataIngestionArtifact(
            feature_store_file_path=self.config.feature_store_file_path,
            training_file_path=self.config.training_file_path,
            testing_file_path=self.config.testing_file_path
        )
