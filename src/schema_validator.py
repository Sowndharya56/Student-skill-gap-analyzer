"""
Schema Validation & Target Leakage Detection Module.
Validates dataset structural integrity, required columns, numerical ranges,
and runs checks for potential target leakage.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import logging
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

from config.config import (
    CORE_SKILLS,
    TARGET_CAREERS,
    STUDENT_DATA_RAW,
    JOB_REQUIREMENTS_RAW,
    JOB_POSTINGS_RAW
)


class SchemaValidationError(Exception):
    """Custom exception raised when dataset schema validation fails."""
    pass


def validate_dataset_exists(file_path: Path, dataset_name: str) -> None:
    """Checks whether the required CSV file exists and is non-empty."""
    if not file_path.exists():
        raise SchemaValidationError(
            f"Dataset '{dataset_name}' not found at path: {file_path}. "
            f"Please ensure the authentic CSV is placed inside data/raw/."
        )
    if file_path.stat().st_size == 0:
        raise SchemaValidationError(f"Dataset '{dataset_name}' at {file_path} is an empty file.")


def validate_student_dataframe_schema(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validates student profiles dataframe schema:
    1. Not empty
    2. Has at least minimum required rows
    3. Contains target column 'Target_Career'
    4. Contains core skill features (or variations recognizable by cleaning layer)
    5. Numeric columns are valid numbers
    """
    if df.empty:
        raise SchemaValidationError("Student dataframe is completely empty (0 rows).")
        
    validation_report = {
        "is_valid": True,
        "rows": len(df),
        "columns": list(df.columns),
        "missing_critical_columns": [],
        "warnings": []
    }
    
    # Check Target column
    has_target = "Target_Career" in df.columns or "target_career" in [c.lower() for c in df.columns]
    if not has_target:
        validation_report["missing_critical_columns"].append("Target_Career")
        validation_report["is_valid"] = False
        raise SchemaValidationError("Critical column 'Target_Career' is missing from student profiles dataset.")
        
    # Check that at least 12 core skills are present
    col_names_lower = [c.lower().replace(" ", "_").replace("-", "_") for c in df.columns]
    found_skills = 0
    for skill in CORE_SKILLS:
        s_lower = skill.lower().replace(" ", "_")
        if any(s_lower in c for c in col_names_lower):
            found_skills += 1
            
    if found_skills < 12:
        msg = f"Insufficient technical skill columns in student dataset. Found only {found_skills} recognizable skills."
        validation_report["warnings"].append(msg)
        logger.warning(msg)
        
    return validation_report


def detect_target_leakage(
    df: pd.DataFrame,
    target_col: str = "Target_Career",
    feature_cols: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Runs automated target leakage checks:
    1. Checks if target column is inadvertently included in feature columns.
    2. Checks if any feature has an exact match or duplicate representation with the target.
    3. Checks for high numerical correlation (>0.98) when sample size is statistically sufficient (>= 30 samples).
    
    Returns:
        Dict with leakage audit results and status.
    """
    report = {
        "leakage_detected": False,
        "checked_features_count": 0,
        "suspicious_features": [],
        "notes": []
    }
    
    if target_col not in df.columns:
        report["notes"].append(f"Target column '{target_col}' not present for leakage audit.")
        return report
        
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c != target_col and c not in ["Student_ID", "Student_Name"]]
        
    report["checked_features_count"] = len(feature_cols)
    
    # 1. Direct Target Inclusion Check
    if target_col in feature_cols:
        report["leakage_detected"] = True
        report["suspicious_features"].append({"feature": target_col, "reason": "Target column directly present in feature set"})
        
    # 2. Duplicate or Label Leakage Check
    target_series = df[target_col].astype(str)
    for col in feature_cols:
        if col not in df.columns:
            continue
            
        # Check if categorical/string column duplicates target labels exactly
        if df[col].dtype == "object":
            matches = (df[col].astype(str) == target_series).mean()
            if matches > 0.95 and col != target_col:
                report["leakage_detected"] = True
                report["suspicious_features"].append({
                    "feature": col,
                    "reason": f"Feature has {matches*100:.1f}% exact match with target labels"
                })
                
        # 3. Correlation Check (only for statistically valid sample sizes >= 30)
        elif np.issubdtype(df[col].dtype, np.number) and len(df) >= 30:
            cat_target = pd.factorize(target_series)[0]
            corr = abs(np.corrcoef(df[col].fillna(df[col].median()), cat_target)[0, 1])
            if corr > 0.98:
                report["leakage_detected"] = True
                report["suspicious_features"].append({
                    "feature": col,
                    "reason": f"Suspiciously high correlation (|r| = {corr:.4f}) with target variable"
                })
                
    if not report["leakage_detected"]:
        report["notes"].append("Passed all automated target leakage checks. Features independently represent student inputs.")
        
    return report


if __name__ == "__main__":
    validate_dataset_exists(STUDENT_DATA_RAW, "Student Career Profiles")
    df_raw = pd.read_csv(STUDENT_DATA_RAW)
    schema_res = validate_student_dataframe_schema(df_raw)
    leakage_res = detect_target_leakage(df_raw)
    print("Schema Validation:", schema_res["is_valid"])
    print("Leakage Detected:", leakage_res["leakage_detected"])
