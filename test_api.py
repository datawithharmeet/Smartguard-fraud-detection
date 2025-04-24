import pandas as pd
import requests
import json

# Load one row from your model input
df = pd.read_parquet("artifacts/model_input.parquet")

# Grab one sample row as input
sample_input = df.drop(columns=["fraud_label"]).iloc[0].to_dict()

# API endpoint
url = "http://127.0.0.1:8000/predict"

# Send POST request
response = requests.post(url, json={"data": sample_input})

# Print response
print("✅ Response:", response.json())
