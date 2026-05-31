import os
import pandas as pd
from sklearn.model_selection import train_test_split

def ingest_and_stratify_data(raw_path, processed_dir, target_column, test_size=0.2):
    print(f"Loading raw dataset from: {raw_path}")
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw data file not found at {raw_path}. Please verify your file name.")
        
    #Read raw dataset
    df = pd.read_csv(raw_path)
    print(f"Successfully loaded dataset. Shape: {df.shape}")
    
    #Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    #Perform Stratified Split
    #'stratify=y' ensures the minority churn class ratio is identical in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    #Recombine features and targets into structured sets
    train_set = pd.concat([X_train, y_train], axis=1)
    test_set = pd.concat([X_test, y_test], axis=1)
    
    #Ensure processed storage directories exist
    os.makedirs(processed_dir, exist_ok=True)
    
    #Save processed splits as production-ready reference data
    train_path = os.path.join(processed_dir, "train.csv")
    test_path = os.path.join(processed_dir, "test.csv")
    
    train_set.to_csv(train_path, index=False)
    test_set.to_csv(test_path, index=False)
    
    print(f"Stratified training set saved to: {train_path} ({train_set.shape})")
    print(f"Stratified test set saved to: {test_path} ({test_set.shape})")

if __name__ == "__main__":
    #Define absolute localized path structures
    RAW_DATA_FILE = r"D:\Data Analytics Projects\customer-churn-prediction\data\raw\churn_data.csv"
    PROCESSED_DATA_DIR = r"D:\Data Analytics Projects\customer-churn-prediction\data\processed"
    TARGET = "Churn"
    
    #Execute the ingestion pipeline
    ingest_and_stratify_data(RAW_DATA_FILE, PROCESSED_DATA_DIR, TARGET)