"""
Data Cleaning & Normalization Module.
Standardizes headers, removes duplicate rows, and enforces domain bounds [0, 100].
NOTE: Statistical parameters (e.g. medians for imputation) are NOT calculated here
to prevent data leakage; imputation is handled inside sklearn Pipeline on X_train.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import logging
from typing import Dict, Tuple
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

from config.config import (
    CORE_SKILLS,
    PROCESSED_DATA_DIR,
    STUDENT_DATA_CLEANED,
    ROLE_SKILL_MATRIX
)
from src.data_loader import load_raw_data

COLUMN_MAPPING = {
    "python_skill": "Python",
    "python": "Python",
    "PYTHON": "Python",
    "ml_proficiency": "Machine_Learning",
    "machine_learning": "Machine_Learning",
    "powerbi_tableau_visuals": "Data_Visualization",
    "power_bi": "Data_Visualization",
    "data_visualization": "Data_Visualization",
    "cloud_aws_azure": "Cloud_Computing",
    "cloud_computing": "Cloud_Computing",
    "devops_git_cicd": "DevOps_CICD",
    "devops_cicd": "DevOps_CICD",
    "docker_kubernetes": "Docker_Kubernetes",
    "data_analysis": "Data_Analysis",
    "deep_learning": "Deep_Learning",
    "nlp": "NLP",
    "database_design": "Database_Design",
    "big_data": "Big_Data",
    "cybersecurity_basics": "Cybersecurity_Basics",
    "statistics_math": "Statistics_Math",
    "problem_solving": "Problem_Solving",
    "communication": "Communication"
}


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Maps varied raw column names to canonical system names."""
    df = df.copy()
    new_cols = {}
    for col in df.columns:
        clean_col = col.strip()
        lookup_key = clean_col.lower().replace(" ", "_").replace("-", "_")
        if lookup_key in COLUMN_MAPPING:
            new_cols[col] = COLUMN_MAPPING[lookup_key]
        else:
            new_cols[col] = clean_col
    df.rename(columns=new_cols, inplace=True)
    return df


def clean_student_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Cleans raw student dataframe:
    - Standardizes column names
    - Deduplicates on Student_ID
    - Clamps skill values to [0, 100]
    - Clamps academic indicators (GPA to [0, 10], experience to [0, 120])
    """
    stats = {
        "initial_rows": len(df),
        "duplicates_removed": 0,
        "missing_values_count": int(df.isnull().sum().sum())
    }
    
    df_clean = df.copy()
    df_clean = standardize_column_names(df_clean)
    
    # 1. Deduplication
    dups_count = int(df_clean.duplicated(subset=["Student_ID"] if "Student_ID" in df_clean.columns else None).sum())
    if dups_count > 0:
        df_clean.drop_duplicates(subset=["Student_ID"] if "Student_ID" in df_clean.columns else None, keep="first", inplace=True)
        stats["duplicates_removed"] = dups_count
        logger.info("Removed %d duplicate student records.", dups_count)
        
    # 2. Clip numeric skill levels strictly between 0 and 100
    for skill in CORE_SKILLS:
        if skill in df_clean.columns:
            # Clip non-null numeric values to [0, 100]
            df_clean[skill] = df_clean[skill].apply(
                lambda x: np.clip(float(x), 0.0, 100.0) if pd.notnull(x) else x
            )
            
    # 3. Clip other numeric features
    if "GPA" in df_clean.columns:
        df_clean["GPA"] = df_clean["GPA"].apply(
            lambda x: np.clip(float(x), 0.0, 10.0) if pd.notnull(x) else x
        )
    if "Experience_Months" in df_clean.columns:
        df_clean["Experience_Months"] = df_clean["Experience_Months"].apply(
            lambda x: np.clip(float(x), 0.0, 120.0) if pd.notnull(x) else x
        )
    if "Projects_Count" in df_clean.columns:
        df_clean["Projects_Count"] = df_clean["Projects_Count"].apply(
            lambda x: np.clip(float(x), 0.0, 50.0) if pd.notnull(x) else x
        )
    if "Certifications_Count" in df_clean.columns:
        df_clean["Certifications_Count"] = df_clean["Certifications_Count"].apply(
            lambda x: np.clip(float(x), 0.0, 30.0) if pd.notnull(x) else x
        )
        
    stats["final_rows"] = len(df_clean)
    return df_clean, stats


def clean_job_requirements(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans and standardizes benchmark role-skill requirements matrix."""
    df_clean = df.copy()
    df_clean = standardize_column_names(df_clean)
    
    if "Job_Role" not in df_clean.columns and "job_role" in df_clean.columns:
        df_clean.rename(columns={"job_role": "Job_Role"}, inplace=True)
        
    for skill in CORE_SKILLS:
        if skill in df_clean.columns:
            df_clean[skill] = df_clean[skill].apply(
                lambda x: int(np.clip(float(x), 0.0, 100.0)) if pd.notnull(x) else 50
            )
        else:
            df_clean[skill] = 50
            
    return df_clean


def clean_and_save_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Runs data cleaning on raw datasets and saves processed CSVs."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df_students_raw, df_job_requirements_raw, _ = load_raw_data()
    
    df_students_clean, _ = clean_student_data(df_students_raw)
    df_matrix_clean = clean_job_requirements(df_job_requirements_raw)
    
    df_students_clean.to_csv(STUDENT_DATA_CLEANED, index=False)
    df_matrix_clean.to_csv(ROLE_SKILL_MATRIX, index=False)
    
    logger.info("Saved cleaned student profiles to: %s (Rows: %d)", STUDENT_DATA_CLEANED, len(df_students_clean))
    logger.info("Saved cleaned role-skill matrix to: %s", ROLE_SKILL_MATRIX)
    
    return df_students_clean, df_matrix_clean


if __name__ == "__main__":
    df_clean, df_matrix = clean_and_save_data()
    print("Data cleaning completed successfully.")
