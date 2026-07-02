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
```

## 🔬 Advanced Data Handling & Engineering
1. Data Leakage PreventionStandard analytics workflows frequently apply scaling or encoding across an entire dataset globally, which leaks test distribution statistics into training computations. This architecture resolves this by running a strict stratified train/test split prior to any data transformation, ensuring the model generalizes reliably to pristine unseen data.

2. Immutability via ColumnTransformersNumeric features (tenure, MonthlyCharges, TotalCharges) undergo median imputation and standardization via StandardScaler. Categorical string objects are encoded into numerical vectors utilizing OneHotEncoder(drop='first', handle_unknown='ignore'). This prevents multi-collinearity and keeps the pipeline from crashing when encountering unexpected data variations in production.

3. Native Imbalance CorrectionCustomer churn datasets are naturally skewed toward loyal users (Class Imbalance). Our orchestrator calculates an explicit target scaling factor ($2.77$) directly from the training label distributions:

$$\text{Scale Pos Weight} = \frac{\text{Total Negative Class Count}}{\text{Total Positive Class Count}}$$

This factor is dynamically injected into the objective functions of our gradient-boosting classifiers to heavily penalize missing a true churn event.

## 📊 Performance Metrics (Test Partition Evaluation)
Upon running the end-to-end pipeline against completely untouched evaluation subsets, the system yielded the following enterprise classification parameters:Target Class ProfilePrecisionRecall (Sensitivity)F1-ScoreEvaluation SupportLoyal Accounts (0)0.910.730.811035 Churn Risk Segment (1)0.520.800.63374Global Accuracy0.751409Statistical ROC-AUC Score: 0.8441

## 🚀 Local Installation & Execution
Clone the repository and step inside the workspace root:
- Bashgit clone [https://github.com/indra-swe/customer-churn-prediction.git](https://github.com/indra-swe/customer-churn-prediction.git)

- Initialize and activate a localized virtual environment:
cd customer-churn-prediction
python -m venv venv
.\venv\Scripts\Activate.ps1

- Install the pinned production dependencies: 
pip install -r requirements.txt

- Trigger the end-to-end execution pipeline: 
python main.py
