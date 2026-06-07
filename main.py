import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
from src.data_ingestion import ingest_and_stratify_data
from src.preprocessing import get_preprocessor
from src.model_training import get_ensemble_pipeline
from src.evaluation import evaluate_production_model
from src.feature_importance import plot_ensemble_feature_importance

def run_pipeline():
    RAW_PATH = r"D:\Data Analytics Projects\customer-churn-prediction\data\raw\churn_data.csv"
    PROCESSED_DIR = r"D:\Data Analytics Projects\customer-churn-prediction\data\processed"
    TARGET = "Churn"
    
    if not os.path.exists(os.path.join(PROCESSED_DIR, "train.csv")):
        ingest_and_stratify_data(RAW_PATH, PROCESSED_DIR, TARGET)
        
    # 1. DATA LOADING & TYPE CLEANING
    train_df = pd.read_csv(os.path.join(PROCESSED_DIR, "train.csv"))
    test_df = pd.read_csv(os.path.join(PROCESSED_DIR, "test.csv"))
    
    # Clean the TotalCharges formatting trap on both sets
    for dataframe in [train_df, test_df]:
        dataframe['TotalCharges'] = pd.to_numeric(dataframe['TotalCharges'].replace(r'^\s*$', np.nan, regex=True))
    
    # Separate Features and Targets
    X_train = train_df.drop(columns=['customerID', TARGET])
    y_train = train_df[TARGET].map({'No': 0, 'Yes': 1})
    
    X_test = test_df.drop(columns=['customerID', TARGET])
    y_test = test_df[TARGET].map({'No': 0, 'Yes': 1})
    
    # Feature configurations
    numerical_features = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']
    categorical_features = [
        'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
        'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
        'PaperlessBilling', 'PaymentMethod'
    ]
    
    # 2. PIPELINE PREPROCESSING
    preprocessor = get_preprocessor(numerical_features, categorical_features)
    
    # CRUCIAL RULE: Fit on train, transform on BOTH separately
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # 3. MODEL TRAINING
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    imbalance_ratio = neg_count / pos_count
    
    model_engine = get_ensemble_pipeline(scale_pos_weight=imbalance_ratio)
    model_engine.fit(X_train_processed, y_train)
    print(" Model ensemble pipeline successfully trained.")
    
    # 4. PRODUCTION EVALUATION LAYER
    # Generate predictions on the completely unseen testing set
    OPTIMIZED_THRESHOLD = 0.65
    test_predictions = (test_probabilities >= OPTIMIZED_THRESHOLD).astype(int)    
    test_probabilities = model_engine.predict_proba(X_test_processed)[:, 1]
    
    # Run the comprehensive evaluation report
    evaluate_production_model(y_test, test_predictions, test_probabilities)
    # Generate and export the feature importance visualization
    plot_ensemble_feature_importance(model_engine, preprocessor, numerical_features, categorical_features)

if __name__ == "__main__":
    run_pipeline()