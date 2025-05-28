import pandas as pd
import json
import os
from frauddetection.utils.main_utils import read_csv, read_json

class RealTimeFeatureEngineer:
    def __init__(self, base_data_path="data"):
        self.cards = read_csv(os.path.join(base_data_path, "cards_data.csv"))
        self.users = read_csv(os.path.join(base_data_path, "users_data.csv"))
        self.mcc_dict = read_json(os.path.join(base_data_path, "mcc_codes.json"))

        # Pre-clean
        self.cards.rename(columns={"id": "card_id_actual"}, inplace=True)
        self.users.rename(columns={"id": "user_id"}, inplace=True)

    def merge_and_clean(self, transaction: dict) -> pd.DataFrame:
        tx = pd.DataFrame([transaction])

        # Merge with cards
        tx = tx.merge(self.cards, left_on="card_id", right_on="card_id_actual", how="left")
        tx.drop(columns=["card_id_actual"], inplace=True, errors="ignore")

        # Handle duplicate client_id columns
        if "client_id_x" in tx.columns and "client_id_y" in tx.columns:
            tx.rename(columns={"client_id_x": "client_id"}, inplace=True)
            tx.drop(columns=["client_id_y"], inplace=True)

        # Merge with users
        tx = tx.merge(self.users, left_on="client_id", right_on="user_id", how="left")
        tx.drop(columns=["user_id"], inplace=True, errors="ignore")

        # MCC mapping
        tx["mcc_desc"] = tx["mcc"].astype(str).map(self.mcc_dict)

        return tx
