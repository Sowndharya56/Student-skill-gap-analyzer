"""
Data Preprocessing Module.
Handles:
- Feature and target extraction
- Target label encoding
- Stratified Train/Test split without preprocessing leakage
- Index tracking for exact reproducibility
- Transformation utilities for inference
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import logging
from typing import Tuple, List, Optional, Dict, Any
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

from config.config import (
    CORE_SKILLS,
    MODELS_DIR,
    LABEL_ENCODER_PATH,
    RANDOM_STATE,
    TEST_SIZE
)
from src.schema_validator import detect_target_leakage

# Canonical Feature set definition for Machine Learning
ML_FEATURE_COLUMNS = CORE_SKILLS + [
    "Prog_Core_Index",
    "Data_Analytics_Index",
    "AI_Intelligent_Index",
    "Cloud_DevOps_Index",
    "Overall_Skill_Score",
    "Skill_Variability",
    "Advanced_Skills_Count",
    "Developing_Skills_Count",
    "Employability_Index"
]


def prepare_ml_data(
    df: pd.DataFrame,
    target_col: str = "Target_Career",
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, LabelEncoder, List[str], np.ndarray, np.ndarray]:
    """
    Prepares dataset for machine learning:
    1. Extracts feature matrix X and target vector y
    2. Runs automated target leakage verification
    3. Encodes target career labels
    4. Performs reproducible stratified train-test split (test set held out untouched)
    5. Returns train and test indices for reproducible tracking
    
    Returns:
        X_train, X_test, y_train, y_test, label_encoder, feature_names, train_indices, test_indices
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Feature verification
    available_features = [col for col in ML_FEATURE_COLUMNS if col in df.columns]
    if len(available_features) < len(CORE_SKILLS):
        raise ValueError(f"Missing core skill features in DataFrame. Found {len(available_features)} of {len(CORE_SKILLS)}")
    
    # 2. Target leakage check
    leakage_report = detect_target_leakage(df, target_col=target_col, feature_cols=available_features)
    if leakage_report["leakage_detected"]:
        raise ValueError(f"Target leakage detected during preprocessing: {leakage_report['suspicious_features']}")
        
    X = df[available_features].values
    y_raw = df[target_col].values
    indices = np.arange(len(df))
    
    # 3. Label Encoding for multi-class target
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    
    # 4. Stratified Train-Test Split
    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
        X, y, indices,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    # Save LabelEncoder artifact
    joblib.dump(label_encoder, LABEL_ENCODER_PATH)
    logger.info("Saved LabelEncoder to: %s", LABEL_ENCODER_PATH)
    
    return X_train, X_test, y_train, y_test, label_encoder, available_features, train_idx, test_idx


def load_label_encoder() -> LabelEncoder:
    """Loads saved LabelEncoder artifact from disk."""
    if not LABEL_ENCODER_PATH.exists():
        raise FileNotFoundError(f"LabelEncoder artifact not found at: {LABEL_ENCODER_PATH}")
    return joblib.load(LABEL_ENCODER_PATH)


def build_student_feature_vector(student_features_dict: dict) -> np.ndarray:
    """Constructs a 2D numpy feature array for a single student profile."""
    feature_vector = []
    for col in ML_FEATURE_COLUMNS:
        feature_vector.append(float(student_features_dict.get(col, 50.0)))
    return np.array([feature_vector])
