import pandas as pd
import numpy as np

class SmartGuardPreprocessor:
    def __init__(self, config: dict):
        self.config = config
        self.high_amount_threshold = None
        self.fraud_target_means = {}

    def fit(self, df: pd.DataFrame):
        if self.config["flags"]["generate_high_amount_flag"]:
            self.high_amount_threshold = (
                df["amount"]
                .replace('[\$,]', '', regex=True)
                .astype(float)
                .apply(np.log1p)
                .quantile(0.95)
            )

        for feature, method in self.config["encoding"].items():
            if method == "target":
                self.fraud_target_means[feature] = df.groupby(feature)["fraud_label"].mean().to_dict()

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        cfg = self.config

        if cfg["flags"]["apply_log_transforms"]:
            df["amount_clean"] = df["amount"].replace('[\$,]', '', regex=True).astype(float)
            df["log_amount"] = df["amount_clean"].apply(lambda x: np.log1p(x) if x > 0 else 0)

            df["yearly_income_clean"] = df["yearly_income"].replace('[\$,]', '', regex=True).astype(float)
            df["log_income"] = df["yearly_income_clean"].apply(lambda x: np.log1p(x) if x > 0 else 0)

        if cfg["flags"]["add_debt_to_income"]:
            df["total_debt_clean"] = df["total_debt"].replace('[\$,]', '', regex=True).astype(float)
            df["debt_to_income"] = df["total_debt_clean"] / (df["yearly_income_clean"] + 1e-6)

        if cfg["flags"]["generate_high_amount_flag"]:
            df["is_high_amount"] = (df["log_amount"] > self.high_amount_threshold).astype(int)

        if "credit_limit" in df.columns and "amount_clean" in df.columns:
            df["credit_limit_clean"] = df["credit_limit"].replace('[\$,]', '', regex=True).astype(float)
            df["amount_to_limit_ratio"] = df["amount_clean"] / (df["credit_limit_clean"] + 1e-6)

        for feature, method in cfg["encoding"].items():
            if method == "one_hot":
                dummies = pd.get_dummies(df[feature], prefix=feature)
                df = pd.concat([df, dummies], axis=1)
            elif method == "target":
                df[f"{feature}_target_encoded"] = df[feature].map(self.fraud_target_means.get(feature, {}))

        df.drop(columns=list(cfg["encoding"].keys()), inplace=True, errors="ignore")

        if cfg["flags"].get("apply_feature_binning"):
            for feature, bins in cfg.get("buckets", {}).items():
                labels = [f"{feature}_bin_{i}" for i in range(len(bins)-1)]
                df[f"{feature}_binned"] = pd.cut(df[feature], bins=bins, labels=labels, include_lowest=True)

        return df
