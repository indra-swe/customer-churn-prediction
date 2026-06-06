# Enterprise Customer Churn Prediction Engine
[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4-orange.svg)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/Framework-XGBoost%20%7C%20LightGBM-green.svg)](https://github.com/dmlc/xgboost)

An end-to-end modular machine learning pipeline built to predict customer attrition and isolate high-value revenue risks using tree-based ensemble frameworks. This project transitions away from monolithic notebook structures into a production-ready, decoupled software architecture designed to prevent data leakage and handle class imbalances natively.

---

## 📈 Business Case & Strategic Impact
In subscription-based industries, customer retention is directly tied to profitability. Acquiring a new customer costs **5x more** than retaining an existing one. 

### Financial Optimization Framework
* **The Metric Catch:** Our production ensemble achieves a **Churn Recall of 80%**, meaning the system proactively flags 80% of all actual churning accounts before they exit.
* **Marketing Budget Optimization:** With a **Precision of 52%**, roughly half of our flagged targets represent false alarms. However, because proactive customer retention actions (such as targeted automated email incentives or localized digital outreach) are low-cost, casting a wider operational net is highly optimized. 
* **ROI Projection:** By proactively protecting 80% of at-risk accounts, marketing teams can deploy defensive retention campaigns that systematically safeguard annualized recurring revenue (ARR) while eliminating manual guesswork.

---

## 🛠️ System Architecture & Directory Structure
This project enforces production software engineering standards by isolating functional responsibilities into decoupled Python modules:

```text
customer-churn-prediction/
│
├── data/
│   ├── raw/            # Original, unmutated data source dump (churn_data.csv)
│   └── processed/      # Stratified, isolated splits (train.csv, test.csv)
│
├── src/                # Modular production engine
│   ├── __init__.py
│   ├── data_ingestion.py   # Secure train/test isolation layer
│   ├── preprocessing.py    # Immutable transformer pipeline definitions
│   ├── model_training.py   # Soft-voting XGBoost + LightGBM architecture
│   └── evaluation.py       # Enterprise metric reporting and plotting
│
├── outputs/            # Saved production performance visualizations
│   └── production_confusion_matrix.png
│
├── main.py             # Core pipeline automation orchestrator
├── requirements.txt    # Pinned dependency manifests
└── README.md