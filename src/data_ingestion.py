import pandas as pd
import os
import json

def load_csv(path):
    print(f"Loading CSV: {path}")
    return pd.read_csv(path)

def load_json(path):
    print(f"Loading JSON: {path}")
    with open(path, 'r') as f:
        return json.load(f)

def run_data_ingestion(data_dir="data", output_path="artifacts/processed_data.parquet"):
    # Load datasets
    transactions = load_csv(os.path.join(data_dir, "transactions_data.csv"))
    cards = load_csv(os.path.join(data_dir, "cards_data.csv"))
    users = load_csv(os.path.join(data_dir, "users_data.csv"))

    # Load and process fraud labels
    fraud_raw = load_json(os.path.join(data_dir, "train_fraud_labels.json"))
    fraud_dict = fraud_raw.get("target", {})
    fraud_df = pd.DataFrame(list(fraud_dict.items()), columns=["id", "fraud_label"])
    fraud_df["id"] = fraud_df["id"].astype(int)
    fraud_df["fraud_label"] = fraud_df["fraud_label"].map({"Yes": 1, "No": 0})

    # Load MCC code mapping
    mcc_dict = load_json(os.path.join(data_dir, "mcc_codes.json"))

    # Merge transactions with fraud labels
    transactions = transactions.merge(fraud_df, on="id", how="inner")

    # Merge transactions with card data
    cards = cards.rename(columns={"id": "card_id_actual"})
    transactions = transactions.merge(cards, left_on="card_id", right_on="card_id_actual", how="left")
    transactions.drop(columns=["card_id_actual"], inplace=True)
    
    # Fix duplicated client_id before merging with users
    transactions.rename(columns={"client_id_x": "client_id"}, inplace=True)
    transactions.drop(columns=["client_id_y"], inplace=True)

    # Merge transactions with user data
    users = users.rename(columns={"id": "user_id"})
    transactions = transactions.merge(users, left_on="client_id", right_on="user_id", how="left")
    transactions.drop(columns=["user_id"], inplace=True)

    # Map MCC code descriptions
    transactions["mcc_desc"] = transactions["mcc"].astype(str).map(mcc_dict)

    # Save as Parquet
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    transactions.to_parquet(output_path, index=False)
    print(f"✅ Data ingestion complete. Output saved to: {output_path}")

if __name__ == "__main__":
    run_data_ingestion()
