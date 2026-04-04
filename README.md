# ✈️ Aircraft Predictive Maintenance using NASA CMAPSS (FD003)

## Overview

This project presents an **end‑to‑end predictive maintenance system for aircraft engines** built using the **NASA CMAPSS FD003 turbofan degradation dataset**. The solution predicts Remaining Useful Life (RUL), classifies near‑term failure risk, clusters engine health conditions, and deploys an interactive dashboard using AWS infrastructure.

The objective aligns with a **Transportation Demand Analysis theme**, supporting airlines in optimizing maintenance scheduling, improving fleet availability, and reducing operational disruptions.

---

# Project Objectives

This system enables:

• Remaining Useful Life prediction (regression)
• Failure risk prediction within 30 cycles (classification)
• Engine health clustering (unsupervised learning)
• Maintenance priority scoring
• Cloud‑based deployment using AWS
• Interactive Streamlit dashboard for decision support

---

# Dataset Information

Dataset Source:
NASA CMAPSS Turbofan Engine Degradation Simulation Dataset

Dataset Variant Used:
FD003

Dataset Characteristics:

| Attribute             | Value                     |
| --------------------- | ------------------------- |
| Training trajectories | 100 engines               |
| Test trajectories     | 100 engines               |
| Operating conditions  | 1 (Sea Level)             |
| Fault modes           | 2 (Fan + HPC degradation) |
| Sensors               | 21                        |
| Operational settings  | 3                         |

Dataset Link:
[https://www.kaggle.com/datasets/bishals098/nasa-turbofan-engine-degradation-simulation/data](https://www.kaggle.com/datasets/bishals098/nasa-turbofan-engine-degradation-simulation/data)

---

# System Architecture

The architecture follows a complete machine learning lifecycle:

NASA Dataset → Data Processing → Feature Engineering → Clustering → Regression Model → Classification Model → AWS Deployment → Streamlit Dashboard

AWS Services Used:

• Amazon S3 – dataset + processed features + model storage
• Amazon SageMaker – model training
• Amazon EC2 – dashboard deployment
• Security Groups – controlled access (Port 8501)

---

# Feature Engineering Pipeline

Feature engineering transforms raw time‑series sensor data into predictive degradation indicators.

Steps include:

1. Removal of low‑variance sensors
2. Rolling statistical features (window size = 5)
3. Delta‑from‑start degradation indicators
4. Lifecycle progression features
5. Remaining Useful Life calculation
6. Failure risk label generation

Key engineered features:

• engine_age
• cycle_norm
• sensor_11_mean_5
• sensor_9_mean_5
• sensor_12_delta_start

Final engineered dataset size:

75 predictive features

---

# Clustering Analysis

K‑Means clustering identifies degradation states of engines.

Pipeline:

Feature engineered dataset → Scaling → Silhouette evaluation → Cluster selection (k=3)

Clusters Identified:

| Cluster   | Meaning              |
| --------- | -------------------- |
| Cluster 0 | Healthy engines      |
| Cluster 1 | Moderate degradation |
| Cluster 2 | High failure risk    |

These clusters support maintenance prioritization decisions.

---

# Regression Model – Remaining Useful Life Prediction

Model Used:
XGBoost Regressor

Performance Metrics:

| Metric     | Value   |
| ---------- | ------- |
| RMSE       | 13.557  |
| MAE        | 9.619   |
| R²         | 0.880   |
| NASA Score | 366.492 |

Interpretation:

The model accurately estimates engine lifetime degradation patterns and supports maintenance scheduling optimization.

---

# Classification Model – Failure Risk Prediction

Model Used:
XGBoost Classifier

Target:
Predict whether engine failure will occur within 30 cycles

Confusion Matrix:

|             | Predicted Safe | Predicted Risk |
| ----------- | -------------- | -------------- |
| Actual Safe | 3              | 77             |
| Actual Risk | 0              | 20             |

Performance Metrics:

| Metric    | Value |
| --------- | ----- |
| Accuracy  | 0.23  |
| Precision | 0.206 |
| Recall    | 1.00  |
| F1 Score  | 0.341 |
| ROC‑AUC   | 0.966 |

Interpretation:

The classifier prioritizes **capturing all failure‑risk engines**, making it suitable for safety‑critical aviation applications.

---

# Model Explainability (SHAP Analysis)

Top regression predictors:

• sensor_11_mean_5
• engine_age
• sensor_9_mean_5
• sensor_3_mean_5
• sensor_12_delta_start

Top classification predictors:

• cycle_norm
• engine_age
• sensor_11_mean_5
• sensor_4_mean_5
• sensor_11_delta_start

These confirm that lifecycle progression and degradation trend indicators strongly influence predictions.

---

# Maintenance Priority Scoring Logic

Priority Score combines:

Remaining Useful Life
Failure Probability
Cluster Label

Higher scores indicate urgent maintenance requirement.

Used for:

Fleet scheduling optimization
Maintenance planning
Resource allocation decisions

---

# Interactive Dashboard (Streamlit)

The deployed dashboard enables:

• Fleet overview monitoring
• Engine‑level diagnostics
• Maintenance priority ranking
• CSV upload prediction interface
• Model performance visualization

Dashboard Modules:

Dashboard Overview
Engine Insights
Predict New Engine
Model Performance

---

# AWS Deployment Architecture

Deployment workflow:

Dataset → S3 Storage → SageMaker Training → Model Export → EC2 Deployment → Streamlit Dashboard

Security configuration:

Port 8501 enabled for dashboard access
SSH enabled for remote administration

---

# Business Impact (Transportation Demand Analysis Alignment)

This solution supports airlines by:

Reducing unexpected engine failures
Improving aircraft availability
Optimizing maintenance schedules
Supporting fleet demand forecasting
Lowering operational downtime
Improving spare‑parts planning

Result:

Better aircraft utilization and improved transportation network efficiency.

---

# Project Structure

```
project/
│
├── notebooks/
│   ├── ingest.py
│   ├── EDA_and_Preprocessing.ipynb
│   ├── Clustering_Analysis.ipynb
│   ├── Model_Evaluation_SHAP.ipynb
│
├── training/
│   ├── sagemaker_xgboost_rul.py
│   ├── sagemaker_xgboost_classifier.py
│
├── deployment/
│   └── app.py
│
├── requirements.txt
│
└── README.md
```

---

# Technologies Used

Python
Pandas
NumPy
Scikit‑learn
XGBoost
Matplotlib
Seaborn
SHAP
AWS S3
AWS SageMaker
AWS EC2
Streamlit

---

# Future Improvements

Improve classification calibration
Add LSTM‑based temporal modeling
Integrate real‑time sensor ingestion
Deploy REST API endpoints
Automate retraining pipeline
Add airline‑level fleet dashboards

---

# Author

Chamali Basnayake