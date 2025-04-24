import pandas as pd
import numpy as np
import yaml
import os

CONFIG_PATH = "config/feature_config.yaml"

def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def preprocess(df, config):
    # ---------- 1. Log Transforms ----------
    if config["flags"]["apply_log_transforms"]:
        df["amount_clean"] = df["amount"].replace('[\$,]', '', regex=True).astype(float)
        df["log_amount"] = df["amount_clean"].apply(lambda x: np.log1p(x) if x > 0 else 0)

        df["yearly_income_clean"] = df["yearly_income"].replace('[\$,]', '', regex=True).astype(float)
        df["log_income"] = df["yearly_income_clean"].apply(lambda x: np.log1p(x) if x > 0 else 0)

    # ---------- 2. Derived Features ----------
    if config["flags"]["add_debt_to_income"]:
        df["total_debt_clean"] = df["total_debt"].replace('[\$,]', '', regex=True).astype(float)
        df["debt_to_income"] = df["total_debt_clean"] / (df["yearly_income_clean"] + 1e-6)

    if config["flags"]["generate_high_amount_flag"]:
        threshold = df["log_amount"].quantile(0.95)
        df["is_high_amount"] = (df["log_amount"] > threshold).astype(int)

    if "credit_limit" in df.columns and "amount_clean" in df.columns:
        df["credit_limit_clean"] = df["credit_limit"].replace('[\$,]', '', regex=True).astype(float)
        df["amount_to_limit_ratio"] = df["amount_clean"] / (df["credit_limit_clean"] + 1e-6)

    # ---------- 3. Encoding Categorical ----------
    for feature, method in config["encoding"].items():
        if method == "one_hot":
            dummies = pd.get_dummies(df[feature], prefix=feature)
            df = pd.concat([df, dummies], axis=1)
        elif method == "target":
            fraud_means = df.groupby(feature)["fraud_label"].mean()
            df[f"{feature}_target_encoded"] = df[feature].map(fraud_means)
    
    df.drop(columns=list(config["encoding"].keys()), inplace=True, errors="ignore")

    # ---------- 4. Binning ----------
    if config["flags"].get("apply_feature_binning"):
        for feature, bins in config.get("buckets", {}).items():
            labels = [f"{feature}_bin_{i}" for i in range(len(bins)-1)]
            df[f"{feature}_binned"] = pd.cut(df[feature], bins=bins, labels=labels, include_lowest=True)

    return df

def select_features(df, config):
    include = config["features"]["include"]
    additional_cols = [col for col in df.columns if any(f in col for f in config["encoding"] if config["encoding"][f] == "one_hot")]
    selected = include + additional_cols + ["fraud_label"]
    return df[[col for col in selected if col in df.columns]]

def drop_intermediate_columns(df):
    cols_to_drop = [
        'amount', 'credit_limit', 'yearly_income', 'total_debt',
        'amount_clean', 'credit_limit_clean', 'yearly_income_clean', 'total_debt_clean'
    ]
    return df.drop(columns=[col for col in cols_to_drop if col in df.columns])

def main():
    # Load config
    config = load_config(CONFIG_PATH)

    # Load processed data
    df = pd.read_parquet("artifacts/processed_data.parquet")

    # Apply transformations
    df_processed = preprocess(df, config)

    # Select final features
    df_final = select_features(df_processed, config)

    # Drop unused raw/intermediate columns
    df_final = drop_intermediate_columns(df_final)

    # Ensure output folder exists
    os.makedirs("artifacts", exist_ok=True)

    # Save full data
    full_path = "artifacts/model_input.parquet"
    df_final.to_parquet(full_path, index=False)
    print(f"Full feature-engineered data saved to: {full_path}")
    print(f"Shape: {df_final.shape}")

    # Save 10% sample
    # sample_path = "artifacts/model_input_sample.parquet"
    # df_final.sample(frac=0.1, random_state=42).to_parquet(sample_path, index=False)
    # print(f"Sample (10%) saved to: {sample_path}")

if __name__ == "__main__":
    main()
