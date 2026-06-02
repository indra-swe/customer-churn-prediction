from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier

def get_ensemble_pipeline(scale_pos_weight=1.0):
    """
    Constructs an enterprise-grade weighted soft-voting ensemble model.
    Injects class weight adjustments to directly counter dataset imbalance.
    """
    # 1. Initialize XGBoost with imbalance weighting and depth constraints to prevent overfitting
    xgb_model = XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    # 2. Initialize LightGBM mirroring identical core training restrictions
    lgb_model = LGBMClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=-1
    )
    
    # 3. Combine both structural models into a soft-voting classifier
    ensemble_model = VotingClassifier(
        estimators=[
            ('xgb', xgb_model),
            ('lgbm', lgb_model)
        ],
        voting='soft'
    )
    
    return ensemble_model