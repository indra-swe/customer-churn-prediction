# main.py
import os
import pandas as pd
import numpy as np
from src.data_ingestion import ingest_and_stratify_data
from src.preprocessing import get_preprocessor
from src.model_training import get_ensemble_pipeline  # Import our new module

def run_pipeline():
    RAW_PATH = r"D:\Data Analytics Projects\customer-churn-prediction\data\raw\churn_data.csv"
    PROCESSED_DIR = r"D:\Data Analytics Projects\customer-churn-prediction\data\processed"
    TARGET = "Churn"
    
    if not os.path.exists(os.path.join(PROCESSED_DIR, "train.csv")):
        ingest_and_stratify_data(RAW_PATH, PROCESSED_DIR, TARGET)
        
    train_df = pd.read_csv(os.path.join(PROCESSED_DIR, "train.csv"))
    train_df['TotalCharges'] = pd.to_numeric(train_df['TotalCharges'].replace(r'^\s*$', np.nan, regex=True))
    
    X_train = train_df.drop(columns=['customerID', TARGET])
    y_train = train_df[TARGET].map({'No': 0, 'Yes': 1})
    
    numerical_features = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']
    categorical_features = [
        'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
        'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
        'PaperlessBilling', 'PaymentMethod'
    ]
    
    # 1. Execute Preprocessing Transformation
    preprocessor = get_preprocessor(numerical_features, categorical_features)
    X_train_processed = preprocessor.fit_transform(X_train)
    print(f"[✓] Data preprocessing pipeline complete. Matrix shape: {X_train_processed.shape}")
    
    # 2. Calculate explicit class imbalance scaling weight ratio
    # Formula: total_negative_class_count / total_positive_class_count
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    imbalance_ratio = neg_count / pos_count
    print(f"[-] Data imbalance analysis: {neg_count} Loyal / {pos_count} Churned. Target Ratio: {imbalance_ratio:.2f}")
    
    # 3. Fetch and Train our Weighted Ensemble Model
    print("[!] Initializing and fitting XGBoost + LightGBM Ensemble Engine...")
    model_engine = get_ensemble_pipeline(scale_pos_weight=imbalance_ratio)
    model_engine.fit(X_train_processed, y_train)
    print("[✓] Model ensemble training phase executed successfully!")

if __name__ == "__main__":
    run_pipeline()