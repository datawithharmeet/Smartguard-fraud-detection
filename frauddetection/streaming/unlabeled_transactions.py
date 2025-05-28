import os
import pandas as pd
import json

from frauddetection.utils.main_utils import read_csv, read_json, save_parquet

# Configs
TRANSACTION_FILE = "data/transactions_data.csv"
LABEL_FILE = "data/train_fraud_labels.json"
OUTPUT_PATH = "data/unlabeled_transactions.parquet"

def extract_unlabeled():
    print("Reading transaction data...")
    df = read_csv(TRANSACTION_FILE)

    print("Reading labeled transaction IDs...")
    labels = read_json(LABEL_FILE)        # This gives a dict with 'target' key
    labeled_ids = set(labels["target"].keys())  # Keys are string transaction_ids

    print("Filtering unlabeled transactions...")
    df["id"] = df["id"].astype(str)
    unlabeled_df = df[~df["id"].isin(labeled_ids)]

    print(f"Unlabeled transactions found: {len(unlabeled_df)}")
    save_parquet(unlabeled_df, OUTPUT_PATH)
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    extract_unlabeled()
