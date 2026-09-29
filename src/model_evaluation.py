"""
Model Evaluation & Diagnostic Interpretation Module.
Calculates:
- Full Per-Class Classification Report (Precision, Recall, F1, Support)
- Normalized Confusion Matrix (Test Set)
- Permutation Feature Importance via sklearn.inspection.permutation_importance
  (Used strictly for post-hoc model interpretation, never for feature selection).
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import logging
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.inspection import permutation_importance

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

from config.config import MODEL_METADATA_PATH, BEST_MODEL_PATH, RANDOM_STATE
from src.model_training import load_trained_model
from src.data_loader import load_processed_data
from src.feature_engineering import engineer_features_pipeline
from src.preprocessing import prepare_ml_data


def evaluate_best_model_details() -> Dict[str, Any]:
    """
    Computes in-depth evaluation and interpretation artifacts for the winning model pipeline:
    - Classification report dataframe
    - Confusion matrix dataframe
    - Permutation feature importance (mean and std across 10 repeats)
    """
    pipeline, le = load_trained_model()
    
    # Prepare test split
    df_students_clean, _ = load_processed_data()
    df_featured = engineer_features_pipeline(df_students_clean, save=False)
    _, X_test, _, y_test, _, feature_names = prepare_ml_data(df_featured)
    
    y_pred = pipeline.predict(X_test)
    
    # 1. Classification Report
    class_names = [str(c) for c in le.classes_]
    report_dict = classification_report(y_test, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    df_report = pd.DataFrame(report_dict).transpose().reset_index()
    df_report.rename(columns={"index": "Class"}, inplace=True)
    
    # 2. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    
    # Normalized Confusion Matrix (Percentages)
    cm_norm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)
    df_cm_norm = pd.DataFrame(np.round(cm_norm * 100, 1), index=class_names, columns=class_names)
    
    # 3. Permutation Feature Importance (10 repeats on test set)
    logger.info("Computing Permutation Feature Importance (10 repeats on test split)...")
    perm_result = permutation_importance(
        pipeline,
        X_test,
        y_test,
        n_repeats=10,
        random_state=RANDOM_STATE,
        scoring="f1_weighted",
        n_jobs=-1
    )
    
    feat_importances = []
    for name, mean_imp, std_imp in zip(feature_names, perm_result.importances_mean, perm_result.importances_std):
        feat_importances.append({
            "Feature": name,
            "Mean_Importance": round(float(mean_imp * 100), 2),
            "Std_Importance": round(float(std_imp * 100), 2),
            "Importance": round(float(max(0.0, mean_imp * 100)), 2)
        })
        
    df_feat_imp = pd.DataFrame(feat_importances).sort_values(by="Mean_Importance", ascending=False).reset_index(drop=True)
    
    return {
        "df_report": df_report,
        "df_cm": df_cm,
        "df_cm_norm": df_cm_norm,
        "df_feature_importance": df_feat_imp,
        "overall_accuracy": round(float(report_dict.get("accuracy", 0.0) * 100), 2),
        "macro_f1": round(float(report_dict.get("macro avg", {}).get("f1-score", 0.0) * 100), 2),
        "weighted_f1": round(float(report_dict.get("weighted avg", {}).get("f1-score", 0.0) * 100), 2)
    }


if __name__ == "__main__":
    eval_res = evaluate_best_model_details()
    print("Model Evaluation Accuracy:", eval_res["overall_accuracy"], "%")
    print("Model Evaluation Weighted F1:", eval_res["weighted_f1"], "%")
    print("\nTop 5 Most Influential Features (Permutation Importance):")
    print(eval_res["df_feature_importance"].head(5).to_string(index=False))
