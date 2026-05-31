import os
import pandas as pd
import numpy as np
from src.data_ingestion import ingest_and_stratify_data
from src.preprocessing import get_preprocessor

def run_pipeline():
    # Define absolute workspace paths
    RAW_PATH = r"D:\Data Analytics Projects\customer-churn-prediction\data\raw\churn_data.csv"
    PROCESSED_DIR = r"D:\Data Analytics Projects\customer-churn-prediction\data\processed"
    TARGET = "Churn"
    
    # Trigger Data Ingestion if split files don't exist yet
    if not os.path.exists(os.path.join(PROCESSED_DIR, "train.csv")):
        ingest_and_stratify_data(RAW_PATH, PROCESSED_DIR, TARGET)
        
    # Load training split data
    train_df = pd.read_csv(os.path.join(PROCESSED_DIR, "train.csv"))
    
    # Clean the TotalCharges type trap
    train_df['TotalCharges'] = pd.to_numeric(train_df['TotalCharges'].replace(r'^\s*$', np.nan, regex=True))
    
    # Separate Features and Labels
    X_train = train_df.drop(columns=['customerID', TARGET])
    y_train = train_df[TARGET].map({'No': 0, 'Yes': 1})
    
    # Explicitly define feature arrays by statistical type
    numerical_features = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']
    categorical_features = [
        'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
        'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
        'PaperlessBilling', 'PaymentMethod'
    ]
    
    # Initialize preprocessing pipeline engine
    preprocessor = get_preprocessor(numerical_features, categorical_features)
    
    # Fit and transform the raw training matrix
    X_train_processed = preprocessor.fit_transform(X_train)
    print(f"\nPipeline execution successful!")
    print(f"Raw feature shape: {X_train.shape}")
    print(f"Cleaned pipeline matrix shape: {X_train_processed.shape}")

if __name__ == "__main__":
    run_pipeline()