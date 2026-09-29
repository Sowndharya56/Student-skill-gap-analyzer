"""
Model Training, Cross-Validation & Model Selection Pipeline.
Methodology:
1. Stratified Train/Test split (held-out test set untouched during model selection).
2. Leakage-free preprocessing via sklearn Pipeline (SimpleImputer and StandardScaler encapsulated).
3. 5-Fold Stratified Cross-Validation on Training Set (X_train, y_train) only.
4. Candidate Model Selection based exclusively on Mean CV Weighted F1-Score.
5. Single final evaluation of winning pipeline on untouched test set (X_test, y_test).
6. Comprehensive model persistence with metadata and exact split indices.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import logging
from datetime import datetime
from typing import Dict, Any, Tuple, Optional
import joblib
import numpy as np
import pandas as pd

import sklearn
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

from config.config import (
    BEST_MODEL_PATH,
    MODEL_METADATA_PATH,
    SPLIT_INDICES_PATH,
    LABEL_ENCODER_PATH,
    MODELS_DIR,
    RANDOM_STATE,
    TEST_SIZE,
    DATASET_PROVENANCE
)
from src.data_loader import load_processed_data
from src.feature_engineering import engineer_features_pipeline
from src.preprocessing import prepare_ml_data, load_label_encoder


def get_candidate_pipelines(random_state: int = RANDOM_STATE) -> Dict[str, Pipeline]:
    """
    Constructs leak-free pipelines for candidate multi-class classification algorithms.
    Imputation and scaling are encapsulated inside pipelines to prevent fold leakage.
    """
    pipelines = {
        "Support Vector Machine": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", SVC(
                C=2.0,
                kernel="rbf",
                probability=True,
                random_state=random_state
            ))
        ]),
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", RandomForestClassifier(
                n_estimators=150,
                max_depth=12,
                min_samples_split=4,
                random_state=random_state,
                n_jobs=-1
            ))
        ]),
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                max_iter=1000,
                C=1.0,
                random_state=random_state
            ))
        ]),
        "K-Nearest Neighbors": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier(
                n_neighbors=7,
                weights="distance",
                n_jobs=-1
            ))
        ]),
        "Gradient Boosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=4,
                random_state=random_state
            ))
        ]),
        "Decision Tree": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", DecisionTreeClassifier(
                max_depth=10,
                min_samples_split=6,
                random_state=random_state
            ))
        ])
    }
    return pipelines


def cross_validate_candidates(
    pipelines: Dict[str, Pipeline],
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv_splits: int = 5
) -> pd.DataFrame:
    """
    Evaluates candidate models using 5-Fold Stratified Cross-Validation on Training data ONLY.
    No test set data is accessed during this comparison and selection phase.
    """
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)
    scoring_metrics = {
        "f1_weighted": "f1_weighted",
        "accuracy": "accuracy",
        "precision_weighted": "precision_weighted",
        "recall_weighted": "recall_weighted"
    }
    
    cv_results = []
    logger.info("Executing 5-Fold Stratified CV on %d training samples across %d candidate models...", len(X_train), len(pipelines))
    
    for name, pipeline in pipelines.items():
        logger.info("Evaluating candidate: %s...", name)
        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring_metrics,
            n_jobs=-1
        )
        
        mean_f1 = float(np.mean(scores["test_f1_weighted"]))
        std_f1 = float(np.std(scores["test_f1_weighted"]))
        mean_acc = float(np.mean(scores["test_accuracy"]))
        std_acc = float(np.std(scores["test_accuracy"]))
        mean_prec = float(np.mean(scores["test_precision_weighted"]))
        mean_rec = float(np.mean(scores["test_recall_weighted"]))
        
        cv_results.append({
            "Model": name,
            "CV Mean F1 (Weighted)": round(mean_f1, 4),
            "CV Std F1": round(std_f1, 4),
            "CV Mean Accuracy": round(mean_acc, 4),
            "CV Std Accuracy": round(std_acc, 4),
            "CV Mean Precision": round(mean_prec, 4),
            "CV Mean Recall": round(mean_rec, 4)
        })
        
    df_cv = pd.DataFrame(cv_results)
    # Model selection based strictly on Mean CV Weighted F1 on training set
    df_cv.sort_values(by=["CV Mean F1 (Weighted)", "CV Mean Accuracy"], ascending=False, inplace=True)
    df_cv.reset_index(drop=True, inplace=True)
    return df_cv


def train_and_select_best_model() -> Tuple[Pipeline, pd.DataFrame, Dict[str, Any]]:
    """
    Complete Leakage-Free Machine Learning Workflow:
    1. Load data & engineer features
    2. Split into Stratified Train and Held-Out Test sets
    3. Perform 5-Fold Stratified CV on Train set only
    4. Select winning candidate by Mean CV Weighted F1
    5. Fit winning pipeline on full Train set
    6. Evaluate once on held-out Test set
    7. Save winning pipeline and audit metadata
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Load cleaned data and engineer features
    df_students_clean, _ = load_processed_data()
    df_featured = engineer_features_pipeline(df_students_clean, save=True)
    
    # 2. Extract X, y and split into Train & Test (Test set remains untouched)
    X_train, X_test, y_train, y_test, label_encoder, feature_names, train_idx, test_idx = prepare_ml_data(df_featured)
    
    # 3. Benchmark candidate models on Training Set only
    candidate_pipelines = get_candidate_pipelines()
    df_cv_comparison = cross_validate_candidates(candidate_pipelines, X_train, y_train, cv_splits=5)
    
    # 4. Select best candidate based on CV Weighted F1
    best_model_name = df_cv_comparison.iloc[0]["Model"]
    best_cv_f1 = df_cv_comparison.iloc[0]["CV Mean F1 (Weighted)"]
    best_cv_acc = df_cv_comparison.iloc[0]["CV Mean Accuracy"]
    
    logger.info("=" * 65)
    logger.info("WINNING MODEL SELECTED (CV Metric): %s", best_model_name)
    logger.info("CV Mean Weighted F1: %.4f | CV Mean Accuracy: %.4f", best_cv_f1, best_cv_acc)
    logger.info("=" * 65)
    
    # 5. Fit the winning pipeline on full Training data
    best_pipeline = candidate_pipelines[best_model_name]
    best_pipeline.fit(X_train, y_train)
    
    # 6. Single Final Evaluation on strictly held-out Test Set
    y_test_pred = best_pipeline.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_prec = precision_score(y_test, y_test_pred, average="weighted", zero_division=0)
    test_rec = recall_score(y_test, y_test_pred, average="weighted", zero_division=0)
    test_f1 = f1_score(y_test, y_test_pred, average="weighted", zero_division=0)
    
    logger.info("Final Held-Out Test Evaluation:")
    logger.info("Test Accuracy: %.4f | Test F1: %.4f | Test Precision: %.4f | Test Recall: %.4f", test_acc, test_f1, test_prec, test_rec)
    
    # 7. Save winning pipeline
    joblib.dump(best_pipeline, BEST_MODEL_PATH)
    logger.info("Saved fitted winning pipeline to: %s", BEST_MODEL_PATH)
    
    # 8. Save complete train and test split indices for exact reproducibility
    split_indices = {
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "train_count": len(train_idx),
        "test_count": len(test_idx),
        "train_indices": [int(i) for i in train_idx],
        "test_indices": [int(i) for i in test_idx]
    }
    with open(SPLIT_INDICES_PATH, "w", encoding="utf-8") as f:
        json.dump(split_indices, f, indent=4)
    logger.info("Saved complete split indices to: %s", SPLIT_INDICES_PATH)
    
    # 9. Save structured metadata JSON
    metadata = {
        "dataset_name": DATASET_PROVENANCE["student_profiles"]["name"],
        "dataset_source": DATASET_PROVENANCE["student_profiles"]["source"],
        "model_name": best_model_name,
        "selection_metric": "5-Fold Stratified CV Mean Weighted F1 (Training Data Only)",
        "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "cv_mean_weighted_f1": best_cv_f1,
        "cv_mean_accuracy": best_cv_acc,
        "final_test_metrics": {
            "test_accuracy": round(float(test_acc), 4),
            "test_precision_weighted": round(float(test_prec), 4),
            "test_recall_weighted": round(float(test_rec), 4),
            "test_f1_weighted": round(float(test_f1), 4)
        },
        "target_variable": "Target_Career",
        "classes": [str(c) for c in label_encoder.classes_],
        "feature_names": feature_names,
        "num_features": len(feature_names),
        "train_samples_count": int(X_train.shape[0]),
        "test_samples_count": int(X_test.shape[0]),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "cv_folds": 5,
        "cv_strategy": "StratifiedKFold(n_splits=5, shuffle=True, random_state=42)",
        "cv_benchmarks_table": df_cv_comparison.to_dict(orient="records"),
        "environment": {
            "python_version": sys.version.split()[0],
            "scikit_learn_version": sklearn.__version__,
            "pandas_version": pd.__version__,
            "numpy_version": np.__version__,
            "joblib_version": joblib.__version__
        }
    }
    with open(MODEL_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    logger.info("Saved comprehensive metadata to: %s", MODEL_METADATA_PATH)
    
    return best_pipeline, df_cv_comparison, metadata


def load_trained_model() -> Tuple[Optional[Pipeline], Optional[Any]]:
    """
    Loads saved Winning Pipeline and LabelEncoder for UI inference.
    Does NOT automatically trigger model training if artifacts are missing;
    instead returns (None, None) so callers can display user-friendly guidance.
    """
    if not BEST_MODEL_PATH.exists() or not MODEL_METADATA_PATH.exists() or not LABEL_ENCODER_PATH.exists():
        logger.warning("Trained model pipeline artifact not found at %s.", BEST_MODEL_PATH)
        return None, None
        
    pipeline = joblib.load(BEST_MODEL_PATH)
    le = load_label_encoder()
    return pipeline, le


if __name__ == "__main__":
    best_pipe, df_comp, meta = train_and_select_best_model()
    print("\n--- 5-Fold Stratified Cross-Validation Benchmark Results ---")
    print(df_comp.to_string(index=False))
    print(f"\nFinal Test Set Accuracy: {meta['final_test_metrics']['test_accuracy']*100:.2f}%")
    print(f"Final Test Set F1: {meta['final_test_metrics']['test_f1_weighted']*100:.2f}%")
