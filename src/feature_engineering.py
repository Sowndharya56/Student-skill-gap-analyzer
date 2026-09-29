"""
Feature Engineering Module.
Derives domain-specific competency indices and heuristic composite indicators.
Features engineered:
- Domain Competency Indices (Prog_Core_Index, Data_Analytics_Index, AI_Intelligent_Index, Cloud_DevOps_Index)
- Distribution Metrics (Overall_Skill_Score, Skill_Variability, Advanced_Skills_Count, Developing_Skills_Count)
- Heuristic Employability Index (Weighted composite design formula)

NOTE: Engineered features are derived directly from input skill metrics and academic indicators.
They are correlated with raw features and are strictly independent of the target variable.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import logging
from typing import List, Dict
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

from config.config import (
    CORE_SKILLS,
    SKILL_DOMAINS,
    ENGINEERED_FEATURES_DATA
)


def compute_domain_indices(df: pd.DataFrame) -> pd.DataFrame:
    """Computes composite domain indices from raw skill ratings."""
    df_feat = df.copy()
    
    prog_skills = [s for s in SKILL_DOMAINS["Programming & Core CS"] if s in df_feat.columns]
    if prog_skills:
        df_feat["Prog_Core_Index"] = df_feat[prog_skills].mean(axis=1).round(2)
        
    data_skills = [s for s in SKILL_DOMAINS["Data & Business Analytics"] if s in df_feat.columns]
    if data_skills:
        df_feat["Data_Analytics_Index"] = df_feat[data_skills].mean(axis=1).round(2)
        
    ai_skills = [s for s in SKILL_DOMAINS["AI & Intelligent Systems"] if s in df_feat.columns]
    if ai_skills:
        df_feat["AI_Intelligent_Index"] = df_feat[ai_skills].mean(axis=1).round(2)
        
    cloud_skills = [s for s in SKILL_DOMAINS["Cloud, DevOps & Security"] if s in df_feat.columns]
    if cloud_skills:
        df_feat["Cloud_DevOps_Index"] = df_feat[cloud_skills].mean(axis=1).round(2)
        
    avail_skills = [s for s in CORE_SKILLS if s in df_feat.columns]
    if avail_skills:
        df_feat["Overall_Skill_Score"] = df_feat[avail_skills].mean(axis=1).round(2)
        df_feat["Skill_Variability"] = df_feat[avail_skills].std(axis=1).round(2)
        df_feat["Advanced_Skills_Count"] = (df_feat[avail_skills] >= 75).sum(axis=1)
        df_feat["Developing_Skills_Count"] = (df_feat[avail_skills] < 40).sum(axis=1)
        
    return df_feat


def compute_employability_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the Heuristic Employability Index.
    
    Formula (Architectural Design Choice):
        Employability Index = (0.50 * Skill_Score) + (0.20 * GPA_Score) + (0.15 * Project_Score) + (0.15 * Exp_Score)
        
    DISCLAIMER: This is a rule-based heuristic indicator designed for student feedback.
    It is NOT an ML-learned statistical outcome.
    """
    df_feat = df.copy()
    
    skill_comp = df_feat["Overall_Skill_Score"] if "Overall_Skill_Score" in df_feat.columns else pd.Series(50.0, index=df_feat.index)
    gpa_comp = (df_feat["GPA"] / 10.0) * 100.0 if "GPA" in df_feat.columns else pd.Series(75.0, index=df_feat.index)
    proj_comp = (df_feat["Projects_Count"].clip(upper=10) / 10.0) * 100.0 if "Projects_Count" in df_feat.columns else pd.Series(50.0, index=df_feat.index)
    exp_comp = (df_feat["Experience_Months"].clip(upper=24) / 24.0) * 100.0 if "Experience_Months" in df_feat.columns else pd.Series(30.0, index=df_feat.index)
    
    employability = (0.50 * skill_comp) + (0.20 * gpa_comp) + (0.15 * proj_comp) + (0.15 * exp_comp)
    df_feat["Employability_Index"] = employability.clip(0.0, 100.0).round(2)
    
    return df_feat


def extract_features_for_student_profile(
    student_skills_dict: Dict[str, float],
    gpa: float = 8.0,
    projects: int = 4,
    experience: int = 6
) -> Dict[str, float]:
    """Extracts and computes engineered feature dictionary for a single student profile."""
    row = {k: float(v) for k, v in student_skills_dict.items()}
    row["GPA"] = float(gpa)
    row["Projects_Count"] = int(projects)
    row["Experience_Months"] = int(experience)
    
    df_single = pd.DataFrame([row])
    df_single = compute_domain_indices(df_single)
    df_single = compute_employability_index(df_single)
    
    return df_single.iloc[0].to_dict()


def engineer_features_pipeline(df_cleaned: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    """Executes the full feature engineering transformation pipeline."""
    logger.info("Executing Feature Engineering pipeline...")
    df_feat = compute_domain_indices(df_cleaned)
    df_feat = compute_employability_index(df_feat)
    
    if save:
        df_feat.to_csv(ENGINEERED_FEATURES_DATA, index=False)
        logger.info("Saved engineered features dataset to: %s", ENGINEERED_FEATURES_DATA)
        
    return df_feat


if __name__ == "__main__":
    from src.data_cleaning import clean_and_save_data
    df_clean, _ = clean_and_save_data()
    df_features = engineer_features_pipeline(df_clean, save=True)
    print(f"Feature Engineering complete. Shape: {df_features.shape}")
