import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

def evaluate_production_model(y_true, y_pred, y_prob):
    """
    Executes a high-fidelity business and statistical evaluation of the 
    deployed model, outputting critical classification matrices.
    """
    print("\n" + "="*20 + " ENTERPRISE EVALUATION REPORT " + "="*20)
    
    # 1. Classification Report (Precision vs. Recall)
    print("\n[📊] Core Classification Metrics:")
    print(classification_report(y_true, y_pred, target_names=['Loyal (0)', 'Churn Risk (1)']))
    
    # 2. ROC-AUC Performance
    auc_score = roc_auc_score(y_true, y_prob)
    print(f"[📈] Statistical ROC-AUC Score: {auc_score:.4f}")
    
    # 3. Visual Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Predict Loyal', 'Predict Churn'],
                yticklabels=['Actual Loyal', 'Actual Churn'])
    plt.title("Production Model Confusion Matrix")
    plt.ylabel("Actual Customer State")
    plt.xlabel("Predicted Customer State")
    plt.savefig("outputs/production_confusion_matrix.png", bbox_inches='tight')
    plt.close()
    print(" Visual confusion matrix matrix plot exported to: outputs/production_confusion_matrix.png")