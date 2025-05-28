import os
import pandas as pd
import json
import pickle

def read_csv(path):
    return pd.read_csv(path)

def read_json(path):
    with open(path, "r") as f:
        return json.load(f)
    
def read_parquet(path):
    return pd.read_parquet(path)   

def save_parquet(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True) 
    df.to_parquet(path, index=False)
    
def save_object(obj, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as file:
        pickle.dump(obj, file)
        
def load_object(filepath):
    with open(filepath, "rb") as file:
        return pickle.load(file)
  