"""
Data Loader Module.
Provides robust functions for loading datasets, verifying schemas, and handling missing files.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import logging
from typing import Tuple, Dict, Any
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

from config.config import (
    STUDENT_DATA_RAW,
    JOB_REQUIREMENTS_RAW,
    JOB_POSTINGS_RAW,
    STUDENT_DATA_CLEANED,
    ROLE_SKILL_MATRIX,
    DATASET_PROVENANCE
)
from src.schema_validator import (
    validate_dataset_exists,
    validate_student_dataframe_schema,
    detect_target_leakage
)


def load_raw_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads all raw datasets from disk with schema validation.
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            (df_students, df_job_requirements, df_job_postings)
    """
    validate_dataset_exists(STUDENT_DATA_RAW, "Student Career Profiles")
    validate_dataset_exists(JOB_REQUIREMENTS_RAW, "Industry Job Requirements")
    validate_dataset_exists(JOB_POSTINGS_RAW, "Tech Job Postings")

    logger.info("Loading verified datasets from: %s", STUDENT_DATA_RAW.parent)
    df_students = pd.read_csv(STUDENT_DATA_RAW)
    df_job_requirements = pd.read_csv(JOB_REQUIREMENTS_RAW)
    df_job_postings = pd.read_csv(JOB_POSTINGS_RAW)

    # Perform structural schema validation
    validate_student_dataframe_schema(df_students)

    logger.info("Loaded Student Profiles: %d rows, %d cols", df_students.shape[0], df_students.shape[1])
    logger.info("Loaded Job Requirements: %d rows, %d cols", df_job_requirements.shape[0], df_job_requirements.shape[1])
    logger.info("Loaded Job Postings: %d rows, %d cols", df_job_postings.shape[0], df_job_postings.shape[1])

    return df_students, df_job_requirements, df_job_postings


def get_dataset_summary(df: pd.DataFrame, name: str = "Dataset") -> Dict[str, Any]:
    """
    Computes structural and statistical summary of a given DataFrame.
    """
    summary = {
        "name": name,
        "shape": df.shape,
        "columns": list(df.columns),
        "data_types": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "total_missing": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "numeric_summary": df.describe().to_dict() if not df.empty else {}
    }
    return summary


def load_processed_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads pre-cleaned and standardized datasets for fast inference and reporting.
    """
    if not STUDENT_DATA_CLEANED.exists() or not ROLE_SKILL_MATRIX.exists():
        from src.data_cleaning import clean_and_save_data
        logger.info("Processed datasets not found. Running data cleaning pipeline...")
        clean_and_save_data()

    df_students = pd.read_csv(STUDENT_DATA_CLEANED)
    df_matrix = pd.read_csv(ROLE_SKILL_MATRIX)
    return df_students, df_matrix


def get_provenance_info() -> Dict[str, Any]:
    """Returns dataset provenance metadata."""
    return DATASET_PROVENANCE


if __name__ == "__main__":
    df_stu, df_req, df_post = load_raw_data()
    summary = get_dataset_summary(df_stu, "Student Profiles")
    print(f"Data Summary: Shape={summary['shape']}, Duplicates={summary['duplicate_rows']}, Missing={summary['total_missing']}")
