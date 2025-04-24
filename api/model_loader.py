import pickle
import pandas as pd

MODEL_PATH = "artifacts/fraud_model.pkl"

def load_model():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return model

# You can optionally load feature list from a file or hardcode it
def get_feature_names():
    df = pd.read_parquet("artifacts/model_input.parquet", engine='pyarrow')
    return [col for col in df.columns if col != "fraud_label"]
