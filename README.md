# SmartGuard: Real-Time Transaction Fraud Detection System

**SmartGuard** is a production-ready fraud detection system designed to score financial transactions in real-time. The project combines data engineering, supervised machine learning, threshold optimization, SHAP-based explainability, and cloud deployment components, making it a scalable, business-impact-driven solution. Trained on 9 million labeled transactions and supported by 4 million unlabeled transactions for streaming simulation, SmartGuard mimics enterprise-grade fraud detection pipelines used in the financial sector.

---

##  Project Highlights

*  End-to-end pipeline covering data ingestion, transformation, model training, tuning, validation, and deployment
*  Exploratory Data Analysis and statistical testing to validate data assumptions and shape transformation strategy
*  Feature engineering pipeline using YAML config, including log transforms, ratios, encodings, and business rules
*  Trained multiple models (XGBoost, LightGBM, RandomForest) with recall-prioritized threshold tuning
*  Final model achieved **82% recall** and **3.8% precision** at 0.8 threshold — balancing fraud catch rate with customer experience
*  SHAP explainability integrated for both global summaries and local transaction-level force plots
*  Real-time risk scoring API deployed using FastAPI and models stored on AWS S3
*  **Kafka-based streaming (in progress)** for simulating real-time transaction ingestion and scoring

---

##  Architecture Overview

```
                +-------------------------+
                |  Raw Transaction Data   |
                +-----------+-------------+
                            ↓
                +-----------v-------------+
                |    Data Ingestion       |
                +-----------+-------------+
                            ↓
                +-----------v-------------+
                | Data Transformation     | ← YAML-based feature engineering
                +-----------+-------------+
                            ↓
                +-----------v-------------+
                |   Model Training (CV)   |
                +-----------+-------------+
                            ↓
                +-----------v-------------+
                | Threshold Optimization  | ← Maximize recall, tune precision
                +-----------+-------------+
                            ↓
                +-----------v-------------+
                | Final Model Validation  | ← Held-out test set
                +-----------+-------------+
                            ↓
                +-----------v-------------+
                |   SHAP Explainability   |
                +-----------+-------------+
                            ↓
                +-----------v-------------+
                |  Real-time Scoring API  | ← FastAPI + S3-deployed model
                +-------------------------+
```


---

##  Key Results

| Metric    | Value |
| --------- | ----- |
| Precision | 3.8%  |
| Recall    | 82%   |
| F1 Score  | 7.4%  |
| ROC AUC   | 97.6% |
| Threshold | 0.8   |

---

##  Explainability

* SHAP global summary plots for top feature importances
* SHAP dependence plots for most influential features
* SHAP force plots returned per transaction in API

---

##  Streaming Simulation (In Progress)

* Kafka producer will stream 4M unlabeled transactions to simulate live ingestion
* Kafka consumer will pull events, forward to scoring API, and log responses
* Designed to reflect a production-grade fraud detection feedback loop

---

##  Technologies Used

* **Languages:** Python, SQL
* **Libraries:** Pandas, NumPy, Scikit-learn, XGBoost, LightGBM, SHAP, Matplotlib, Seaborn
* **Deployment:** FastAPI, AWS EC2/S3, Docker
* **Tracking:** MLflow, DagsHub
* **Automation:** GitHub Actions, YAML-based configs
* **Streaming (WIP):** Apache Kafka (Producer/Consumer)

---


