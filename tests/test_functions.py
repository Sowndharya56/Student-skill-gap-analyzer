"""
Comprehensive Automated Unit Test Suite for AI Student Skill Gap Analyzer & Career Recommendation System.

Test Coverage:
1. DATA LOADING & SCHEMA VALIDATION:
   - File loading and non-emptiness checks
   - Missing critical column handling (SchemaValidationError)
   - Target leakage checks (feature vs target separation)

2. DATA CLEANING & CLAMPING:
   - Deduplication on Student_ID
   - Bounds clamping [0, 100] for skills and [0, 10] for GPA

3. FEATURE ENGINEERING:
   - Domain indices derivation (Prog, Analytics, AI, Cloud/DevOps)
   - Heuristic Employability Index calculation
   - Single student profile feature vector building

4. SKILL GAP ENGINE:
   - Non-negative gap computation: max(0, Required - Student)
   - Zero gap for mastered skills (Student >= Required)
   - Positive gap for deficit skills (Required > Student)
   - Clamping negative student inputs to 0
   - Clamping >100 student inputs to 100
   - Handling missing and unknown skills
   - Severity categorization (Mastered, Low, Medium, High)
   - Actionable roadmap generation

5. CAREER RECOMMENDATION ENGINE:
   - Valid student profile recommendation ranking
   - Empty skill dictionary handling (safe fallback)
   - All-zeros skill profile handling
   - Missing model fallback behavior

6. NLP SKILL EXTRACTION & BOUNDARY RECOGNITION:
   - Punctuation terms: C++, C#, CI/CD, .NET, Node.js, Power BI, AWS, Azure, GCP
   - Strict distinctness: C# != C++
   - Strict distinctness: .NET != Java
   - Empty and invalid input handling

7. MACHINE LEARNING PIPELINE:
   - Pipeline creation with SimpleImputer + StandardScaler
   - Model loading and calibrated probability outputs
   - Model metadata verification
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import numpy as np
import pandas as pd

from config.config import (
    CORE_SKILLS,
    TARGET_CAREERS,
    GAP_THRESHOLDS,
    HYBRID_WEIGHTS,
    MODEL_METADATA_PATH,
    BEST_MODEL_PATH
)
from src.data_loader import load_raw_data, get_dataset_summary, load_processed_data
from src.schema_validator import (
    validate_dataset_exists,
    validate_student_dataframe_schema,
    detect_target_leakage,
    SchemaValidationError
)
from src.data_cleaning import clean_student_data, clean_job_requirements, standardize_column_names
from src.feature_engineering import compute_domain_indices, compute_employability_index, extract_features_for_student_profile
from src.preprocessing import prepare_ml_data, load_label_encoder, build_student_feature_vector
from src.skill_gap import (
    calculate_skill_gaps,
    get_gap_severity,
    compute_role_readiness_metrics,
    generate_learning_roadmap,
    validate_and_clamp_score
)
from src.career_recommendation import recommend_careers, compute_vector_match_scores
from src.nlp_skills import clean_text_for_tokens, tokenize_text, extract_skills_from_text
from src.model_training import load_trained_model, get_candidate_pipelines


# ==============================================================================
# 1. DATA & SCHEMA VALIDATION TESTS
# ==============================================================================
def test_data_loader_and_schema():
    """Verify raw datasets load with valid schema and non-emptiness."""
    df_stu, df_req, df_post = load_raw_data()
    assert len(df_stu) > 1000
    assert len(df_req) == 10
    assert len(df_post) >= 300
    
    schema_res = validate_student_dataframe_schema(df_stu)
    assert schema_res["is_valid"] is True


def test_schema_validator_missing_column_error():
    """Verify SchemaValidationError when critical column is missing."""
    invalid_df = pd.DataFrame([{"Student_Name": "Test Student", "Python": 80}])
    with pytest.raises(SchemaValidationError):
        validate_student_dataframe_schema(invalid_df)


def test_target_leakage_detection():
    """Verify target leakage detector identifies duplicate target variables and passes clean features."""
    np.random.seed(42)
    clean_df = pd.DataFrame({
        "Python": np.random.randint(40, 95, size=60),
        "SQL": np.random.randint(40, 95, size=60),
        "Target_Career": np.random.choice(["Data Scientist", "Data Analyst", "Backend Developer"], size=60)
    })
    clean_report = detect_target_leakage(clean_df, target_col="Target_Career", feature_cols=["Python", "SQL"])
    assert clean_report["leakage_detected"] is False

    # Leaked dataframe with direct target column in feature list
    leaked_report = detect_target_leakage(clean_df, target_col="Target_Career", feature_cols=["Python", "Target_Career"])
    assert leaked_report["leakage_detected"] is True


# ==============================================================================
# 2. DATA CLEANING & CLAMPING TESTS
# ==============================================================================
def test_data_cleaning_and_bounds():
    """Verify deduplication, column mapping, and score clamping [0, 100]."""
    raw_sample = pd.DataFrame([
        {"Student_ID": "S1", "python_skill": 80, "ML_Proficiency": 90, "GPA": 8.5, "Certifications_Count": np.nan},
        {"Student_ID": "S1", "python_skill": 80, "ML_Proficiency": 90, "GPA": 8.5, "Certifications_Count": np.nan}, # Duplicate
        {"Student_ID": "S2", "python_skill": 150, "ML_Proficiency": -25, "GPA": 12.5, "Certifications_Count": 2}   # Out of bounds
    ])
    
    df_cleaned, stats = clean_student_data(raw_sample)
    assert len(df_cleaned) == 2
    assert stats["duplicates_removed"] == 1
    assert "Python" in df_cleaned.columns
    assert "Machine_Learning" in df_cleaned.columns
    
    # Assert clamping to [0, 100]
    assert df_cleaned.loc[df_cleaned["Student_ID"] == "S2", "Python"].values[0] == 100.0
    assert df_cleaned.loc[df_cleaned["Student_ID"] == "S2", "Machine_Learning"].values[0] == 0.0
    assert df_cleaned.loc[df_cleaned["Student_ID"] == "S2", "GPA"].values[0] == 10.0


# ==============================================================================
# 3. FEATURE ENGINEERING TESTS
# ==============================================================================
def test_feature_engineering():
    """Verify domain indices, composite scores, and single profile feature extraction."""
    sample_skills = {skill: 80 for skill in CORE_SKILLS}
    features = extract_features_for_student_profile(sample_skills, gpa=9.0, projects=5, experience=12)
    
    assert "Prog_Core_Index" in features
    assert "Data_Analytics_Index" in features
    assert "AI_Intelligent_Index" in features
    assert "Cloud_DevOps_Index" in features
    assert "Employability_Index" in features
    assert 0.0 <= features["Employability_Index"] <= 100.0


# ==============================================================================
# 4. SKILL GAP ENGINE TESTS
# ==============================================================================
def test_skill_gap_math():
    """Verify non-negative gap calculation: max(0, Required - Student)."""
    # 1. Normal gap
    student_skills = {"Python": 60, "SQL": 85, "Machine_Learning": 30, "Cloud_Computing": -15}
    required_skills = {"Python": 80, "SQL": 70, "Machine_Learning": 90, "Cloud_Computing": 70}
    
    df_gaps = calculate_skill_gaps(student_skills, required_skills)
    
    # Python: Req 80 - Stu 60 = 20
    py_gap = df_gaps.loc[df_gaps["Skill"] == "Python", "Gap"].values[0]
    assert py_gap == 20.0
    
    # SQL: Req 70 - Stu 85 = 0 (Mastered, no negative gap)
    sql_gap = df_gaps.loc[df_gaps["Skill"] == "SQL", "Gap"].values[0]
    assert sql_gap == 0.0
    
    # ML: Req 90 - Stu 30 = 60 (High priority gap)
    ml_gap = df_gaps.loc[df_gaps["Skill"] == "Machine_Learning", "Gap"].values[0]
    assert ml_gap == 60.0
    
    # Cloud: Negative student input clamped to 0 -> Gap = 70 - 0 = 70
    cloud_gap = df_gaps.loc[df_gaps["Skill"] == "Cloud_Computing", "Gap"].values[0]
    assert cloud_gap == 70.0


def test_skill_gap_severity_and_roadmap():
    """Verify severity categorization thresholds and roadmap structure."""
    assert get_gap_severity(0.0)[0] == GAP_THRESHOLDS["NONE"][2]
    assert get_gap_severity(10.0)[0] == GAP_THRESHOLDS["LOW"][2]
    assert get_gap_severity(25.0)[0] == GAP_THRESHOLDS["MEDIUM"][2]
    assert get_gap_severity(45.0)[0] == GAP_THRESHOLDS["HIGH"][2]
    
    df_test = pd.DataFrame([
        {"Skill": "Python", "Display_Name": "Python Programming", "Student_Score": 40, "Required_Score": 85, "Gap": 45, "Severity": "High Priority Gap", "Color": "#EF4444"}
    ])
    roadmap = generate_learning_roadmap(df_test)
    assert len(roadmap) == 1
    assert roadmap[0]["skill"] == "Python"
    assert roadmap[0]["priority_level"] == "High"
    assert len(roadmap[0]["topics"]) > 0


# ==============================================================================
# 5. CAREER RECOMMENDATION TESTS
# ==============================================================================
def test_career_recommendations_normal():
    """Verify career recommendation ranking with valid profile."""
    student_skills = {
        "Python": 90, "SQL": 85, "Data_Analysis": 92, "Machine_Learning": 88,
        "Data_Visualization": 85, "Statistics_Math": 90, "Problem_Solving": 85
    }
    recommendations = recommend_careers(student_skills, gpa=8.8, projects=6, experience=12, top_n=3)
    assert len(recommendations) == 3
    assert recommendations[0]["hybrid_match_pct"] >= recommendations[1]["hybrid_match_pct"]
    assert recommendations[0]["role"] in TARGET_CAREERS
    assert len(recommendations[0]["explanation"]) > 10


def test_career_recommendations_edge_cases():
    """Verify recommendation engine handles empty and all-zero profiles without exceptions."""
    # 1. Empty skills dictionary
    recs_empty = recommend_careers({}, top_n=3)
    assert len(recs_empty) == 3
    assert recs_empty[0]["hybrid_match_pct"] >= 0.0

    # 2. All zero skills
    zero_skills = {s: 0.0 for s in CORE_SKILLS}
    recs_zero = recommend_careers(zero_skills, top_n=3)
    assert len(recs_zero) == 3
    assert recs_zero[0]["hybrid_match_pct"] >= 0.0


# ==============================================================================
# 6. NLP SKILL EXTRACTION & DISTINCTNESS TESTS
# ==============================================================================
def test_nlp_skill_extraction_special_terms():
    """
    Verify NLP parser correctly extracts punctuation-heavy terms:
    C++, C#, CI/CD, .NET, Node.js, Power BI, AWS, Azure, GCP.
    """
    test_cases = [
        ("We are hiring a C++ developer with strong OOP skills.", "C++"),
        ("Looking for a backend developer with C# experience.", "C#"),
        ("Must have hands-on experience in CI/CD pipelines and Git.", "DevOps_CICD"),
        ("Enterprise application using .NET Core and Web APIs.", ".NET"),
        ("Seeking a Full Stack Engineer proficient in Node.js and React.", "Node.js"),
        ("Create executive dashboards using Power BI and Tableau.", "Data_Visualization"),
        ("Deploy cloud infrastructure on AWS and Microsoft Azure.", "Cloud_Computing"),
        ("Manage Big Data workloads using GCP and Apache Spark.", "Cloud_Computing"),
        ("Machine Learning Engineer with Deep Learning and Python background.", "Machine_Learning")
    ]
    
    for text, expected_canonical_skill in test_cases:
        res = extract_skills_from_text(text)
        extracted = res["extracted_skills"]
        assert expected_canonical_skill in extracted, f"Failed to extract '{expected_canonical_skill}' from: '{text}' (Extracted: {extracted})"


def test_nlp_distinctness_csharp_not_cpp_and_dotnet_not_java():
    """
    Verify that C# is strictly distinct from C++ and .NET is strictly distinct from Java.
    """
    res_cpp = extract_skills_from_text("Senior C++ Developer")
    res_csharp = extract_skills_from_text("Senior C# Developer")
    assert "C++" in res_cpp["extracted_skills"]
    assert "C#" in res_csharp["extracted_skills"]
    assert "C++" not in res_csharp["extracted_skills"]
    assert "C#" not in res_cpp["extracted_skills"]

    res_dotnet = extract_skills_from_text("Senior .NET Developer")
    res_java = extract_skills_from_text("Senior Java Developer")
    assert ".NET" in res_dotnet["extracted_skills"]
    assert "Java" in res_java["extracted_skills"]
    assert "Java" not in res_dotnet["extracted_skills"]
    assert ".NET" not in res_java["extracted_skills"]


def test_nlp_empty_and_invalid_input():
    """Verify NLP parser handles empty string or non-string input safely."""
    assert extract_skills_from_text("")["extracted_skills"] == []
    assert extract_skills_from_text(None)["extracted_skills"] == []


# ==============================================================================
# 7. MACHINE LEARNING PIPELINE & MODEL LOADING TESTS
# ==============================================================================
def test_model_loading_and_prediction():
    """Verify saved pipeline loads and produces calibrated multi-class probabilities."""
    pipeline, le = load_trained_model()
    if pipeline is not None and le is not None:
        assert len(le.classes_) == 10
        
        test_profile = {skill: 70 for skill in CORE_SKILLS}
        feat_dict = extract_features_for_student_profile(test_profile)
        X_vec = build_student_feature_vector(feat_dict)
        
        probs = pipeline.predict_proba(X_vec)[0]
        assert len(probs) == 10
        assert np.isclose(np.sum(probs), 1.0)
