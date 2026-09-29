"""
Script to create the 4 Jupyter Notebooks with complete Data Science Life Cycle:
1. notebooks/01_data_understanding.ipynb
2. notebooks/02_eda.ipynb
3. notebooks/03_preprocessing.ipynb
4. notebooks/04_model_training.ipynb
"""

import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)

def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

def md_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.split("\n")]
    }

def code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.split("\n")]
    }

# 1. 01_data_understanding.ipynb
nb1_cells = [
    md_cell("""# Step 1, 2 & 3: Problem Definition, Data Ingestion & Data Understanding
## Project: AI-Based Student Skill Gap Analyzer & Career Recommendation System

### Objectives
- Define the student employability and skill-gap quantification problem.
- Ingest the core datasets from `data/raw/` with schema validation:
  1. Student Career Profiles
  2. Industry Job Requirements Benchmark Matrix
  3. Tech Job Postings Corpus
- Inspect dataset provenance, shapes, columns, data types, missing values, duplicates, and statistical summaries.
- Verify zero target leakage in feature spaces.

> **Dataset Status Note:** Currently operating on the development benchmark dataset fixtures. The pipeline is structurally validated and ready to ingest verified external production datasets once provided.
"""),
    code_cell("""import sys
from pathlib import Path
import pandas as pd
import numpy as np

project_root = Path("..").resolve()
sys.path.insert(0, str(project_root))

from config.config import STUDENT_DATA_RAW, JOB_REQUIREMENTS_RAW, JOB_POSTINGS_RAW, DATASET_PROVENANCE
from src.data_loader import load_raw_data, get_dataset_summary
from src.schema_validator import validate_student_dataframe_schema, detect_target_leakage"""),
    md_cell("### 1. Dataset Provenance & Verification Status"),
    code_cell("""for key, prov in DATASET_PROVENANCE.items():
    print(f"Dataset: {prov['name']}")
    print(f" - Source: {prov['source']}")
    print(f" - Verification Status: {prov['verified_status']}")
    print(f" - License: {prov['license']}")
    print(f" - Purpose: {prov['purpose']}")
    print(f" - Limitations: {prov['limitations']}\\n")"""),
    md_cell("### 2. Ingest Datasets with Schema Validation"),
    code_cell("""df_students_raw, df_job_requirements_raw, df_job_postings_raw = load_raw_data()
print(f"Student Profiles Shape: {df_students_raw.shape}")
print(f"Job Requirements Shape: {df_job_requirements_raw.shape}")
print(f"Job Postings Shape: {df_job_postings_raw.shape}")"""),
    md_cell("### 3. Inspect Data Types and Structural Information"),
    code_cell("""df_students_raw.info()"""),
    code_cell("""df_students_raw.head()"""),
    md_cell("### 4. Missing Values, Duplicates & Summary Statistics"),
    code_cell("""summary = get_dataset_summary(df_students_raw, "Raw Student Profiles")
print(f"Total Missing Values: {summary['total_missing']}")
print(f"Duplicate Rows: {summary['duplicate_rows']}")
df_students_raw.describe()"""),
    md_cell("### 5. Target Leakage Verification"),
    code_cell("""leakage_report = detect_target_leakage(df_students_raw)
print("Target Leakage Detected:", leakage_report["leakage_detected"])
print("Audit Notes:", leakage_report["notes"])""")
]

# 2. 02_eda.ipynb
nb2_cells = [
    md_cell("""# Step 4: Exploratory Data Analysis (EDA)
## Project: AI-Based Student Skill Gap Analyzer & Career Recommendation System

This notebook covers:
- Distribution of student skill levels across 20 technical categories
- Distribution of target career roles and academic degrees
- Inter-skill correlation analysis
- Numerical boxplots and distribution diagnostics
"""),
    code_cell("""import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

project_root = Path("..").resolve()
sys.path.insert(0, str(project_root))

from config.config import CORE_SKILLS
from src.data_loader import load_processed_data

df_students, df_matrix = load_processed_data()
df_students.head()"""),
    md_cell("### 1. Target Career Role Distribution"),
    code_cell("""plt.figure(figsize=(10, 5))
sns.countplot(data=df_students, y='Target_Career', palette='viridis', order=df_students['Target_Career'].value_counts().index)
plt.title('Distribution of Student Target Career Archetypes')
plt.xlabel('Number of Students')
plt.ylabel('Career Role')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.show()"""),
    md_cell("### 2. Skill Proficiency Distributions (Boxplots)"),
    code_cell("""key_skills = ['Python', 'SQL', 'Data_Analysis', 'Machine_Learning', 'Cloud_Computing', 'Problem_Solving']
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_students[key_skills], palette='Set2')
plt.title('Proficiency Score Distributions across Key Skills (0-100)')
plt.ylabel('Proficiency Level')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()"""),
    md_cell("### 3. Inter-Skill Correlation Heatmap"),
    code_cell("""plt.figure(figsize=(12, 10))
corr = df_students[CORE_SKILLS].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', cbar=True, square=True)
plt.title('Correlation Heatmap of 20 Core Technical Skills')
plt.show()""")
]

# 3. 03_preprocessing.ipynb
nb3_cells = [
    md_cell("""# Step 5 & 6: Data Cleaning, Preprocessing & Feature Engineering
## Project: AI-Based Student Skill Gap Analyzer & Career Recommendation System

This notebook covers:
- Standardizing column names and noisy headers
- Deduplication and boundary clamping to [0, 100]
- Computing composite domain indices (Programming, Data Analytics, AI, Cloud/DevOps)
- Deriving the Heuristic Employability Index
- Stratified Train-Test splitting (preventing data leakage)
"""),
    code_cell("""import sys
from pathlib import Path
import pandas as pd
import numpy as np

project_root = Path("..").resolve()
sys.path.insert(0, str(project_root))

from src.data_loader import load_raw_data
from src.data_cleaning import clean_student_data, clean_job_requirements
from src.feature_engineering import compute_domain_indices, compute_employability_index
from src.preprocessing import prepare_ml_data"""),
    md_cell("### 1. Data Cleaning & Header Standardization"),
    code_cell("""df_stu_raw, df_req_raw, _ = load_raw_data()
df_stu_clean, clean_stats = clean_student_data(df_stu_raw)
print(f"Cleaning Statistics: {clean_stats}")
df_stu_clean.head()"""),
    md_cell("### 2. Feature Engineering & Heuristic Employability Index"),
    code_cell("""df_featured = compute_domain_indices(df_stu_clean)
df_featured = compute_employability_index(df_featured)
print(f"Engineered Dataset Shape: {df_featured.shape}")
df_featured[['Student_Name', 'Prog_Core_Index', 'Data_Analytics_Index', 'AI_Intelligent_Index', 'Cloud_DevOps_Index', 'Employability_Index']].head()"""),
    md_cell("### 3. Stratified Train-Test Split (Held-Out Test Set Isolation)"),
    code_cell("""X_train, X_test, y_train, y_test, label_encoder, feature_names, train_idx, test_idx = prepare_ml_data(df_featured)
print(f"X_train Shape: {X_train.shape}, X_test Shape: {X_test.shape}")
print(f"Target Classes ({len(label_encoder.classes_)}): {label_encoder.classes_}")
print(f"Recorded Train Indices Count: {len(train_idx)}, Test Indices Count: {len(test_idx)}")""")
]

# 4. 04_model_training.ipynb
nb4_cells = [
    md_cell("""# Step 7 & 8: Machine Learning Pipelines, 5-Fold Stratified CV & Permutation Importance
## Project: AI-Based Student Skill Gap Analyzer & Career Recommendation System

Methodology:
- Wrap all estimators in `sklearn.pipeline.Pipeline` with `SimpleImputer` and `StandardScaler` to prevent preprocessing leakage.
- Run 5-Fold Stratified Cross-Validation on the Training Set (`X_train`, `y_train`) only.
- Model selection based strictly on **Mean CV Weighted F1-Score**.
- Single final evaluation of winning pipeline on held-out test set (`X_test`, `y_test`).
- Post-hoc Permutation Feature Importance computed on held-out test split.

> **Academic Note:** Final model metrics will reflect the dataset loaded in `data/raw/`. Retraining is required whenever a new verified production dataset is supplied.
"""),
    code_cell("""import sys
from pathlib import Path
import pandas as pd
import numpy as np

project_root = Path("..").resolve()
sys.path.insert(0, str(project_root))

from src.model_training import train_and_select_best_model, load_trained_model
from src.model_evaluation import evaluate_best_model_details"""),
    md_cell("### 1. 5-Fold Stratified Cross-Validation & Model Selection"),
    code_cell("""best_pipeline, df_cv_comparison, metadata = train_and_select_best_model()
print("--- 5-Fold Stratified CV Benchmark Comparison (Training Data Only) ---")
df_cv_comparison"""),
    md_cell("### 2. Final Evaluation on Held-Out Test Set"),
    code_cell("""eval_results = evaluate_best_model_details()
print(f"Winning Model: {metadata['model_name']}")
print(f"Test Accuracy: {eval_results['overall_accuracy']}%")
print(f"Test Weighted F1: {eval_results['weighted_f1']}%")"""),
    md_cell("### 3. Classification Report per Career Role"),
    code_cell("""eval_results['df_report']"""),
    md_cell("### 4. Normalized Confusion Matrix (%)"),
    code_cell("""eval_results['df_cm_norm']"""),
    md_cell("### 5. Permutation Feature Importance (10 Repeats on Test Set)"),
    code_cell("""eval_results['df_feature_importance'].head(15)""")
]

# Write all 4 notebooks to disk
with open(NOTEBOOKS_DIR / "01_data_understanding.ipynb", "w", encoding="utf-8") as f:
    json.dump(make_notebook(nb1_cells), f, indent=2)

with open(NOTEBOOKS_DIR / "02_eda.ipynb", "w", encoding="utf-8") as f:
    json.dump(make_notebook(nb2_cells), f, indent=2)

with open(NOTEBOOKS_DIR / "03_preprocessing.ipynb", "w", encoding="utf-8") as f:
    json.dump(make_notebook(nb3_cells), f, indent=2)

with open(NOTEBOOKS_DIR / "04_model_training.ipynb", "w", encoding="utf-8") as f:
    json.dump(make_notebook(nb4_cells), f, indent=2)

print("Created all 4 updated Jupyter Notebooks successfully.")
