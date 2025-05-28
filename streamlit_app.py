import streamlit as st
import pandas as pd
import json

# Set path to your prediction logs
LOG_PATH = "frauddetection/streaming/stream_predictions.csv"  # update this

# Streamlit layout
st.set_page_config(page_title="SmartGuard: Real-Time Fraud Dashboard", layout="wide")
st.title(" SmartGuard: Real-Time Transaction Risk Scoring")

# Auto-refresh every 5 seconds
st.experimental_rerun_interval = 5

# Read predictions
try:
    df = pd.read_csv(LOG_PATH)
    df["risk_score"] = df["risk_score"].round(4)

    # Decode JSON columns
    df["raw"] = df["raw"].apply(json.loads)
    df["top_features"] = df["top_features"].apply(json.loads)

    # Display latest 20
    st.subheader("Latest Predictions")
    for i, row in df.tail(20).iterrows():
        st.markdown(f"**Transaction ID:** `{row['raw'].get('id')}`")
        st.write(f" **Risk Score**: `{row['risk_score']}` | 🧾 **Prediction**: `{row['prediction']}`")
        st.write(f" **Location:** {row['raw'].get('merchant_city')}, {row['raw'].get('merchant_state')}")
        
        st.write(" **Top SHAP Features:**")
        for feat, val in row["top_features"]:
            st.markdown(f"- `{feat}`: {val:.4f}")
        
        st.markdown("---")

except Exception as e:
    st.error(f"Could not load predictions. Error: {e}")
