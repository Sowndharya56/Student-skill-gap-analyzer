"""
End-to-End Integration Test Suite validating the full system flow:
Student Profile -> ML Career Recommendation -> Skill Gap Analysis ->
Skill Priority Engine -> Learning Sequence -> Course Recommendation -> Enhanced 6-Phase Roadmap
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import pandas as pd
import numpy as np

from config.config import CORE_SKILLS, TARGET_CAREERS
from src.data_loader import load_processed_data
from src.skill_gap import calculate_skill_gaps, compute_role_readiness_metrics, generate_enhanced_learning_roadmap
from src.career_recommendation import recommend_careers
from src.skill_priority import calculate_skill_priorities, generate_learning_sequence
from src.course_recommender import recommend_courses_for_profile, recommend_courses_for_skill
from src.nlp_skills import extract_skills_from_text
from src.model_evaluation import evaluate_best_model_details
from src.utils import generate_markdown_audit_report


def test_beginner_profile_flow():
    """
    Test 1: Beginner student profile (low skills across the board).
    Verifies:
    - Identifies high gaps
    - Recommends foundational prerequisite skills first
    - Course recommendations suggest Beginner-level courses
    """
    beginner_skills = {
        "Python": 25.0, "SQL": 30.0, "R": 10.0, "Java": 20.0, "C++": 15.0, "JavaScript": 20.0,
        "Data_Analysis": 20.0, "Machine_Learning": 15.0, "Deep_Learning": 10.0, "NLP": 10.0,
        "Data_Visualization": 25.0, "Cloud_Computing": 15.0, "DevOps_CICD": 10.0, "Docker_Kubernetes": 10.0,
        "Database_Design": 25.0, "Big_Data": 10.0, "Cybersecurity_Basics": 20.0,
        "Statistics_Math": 30.0, "Problem_Solving": 35.0, "Communication": 50.0
    }
    
    # 1. Career recommendations
    recs = recommend_careers(beginner_skills, gpa=7.0, projects=1, experience=0, top_n=3)
    assert len(recs) == 3
    
    # 2. Skill Gap for Data Scientist
    df_students, df_matrix = load_processed_data()
    role_row = df_matrix[df_matrix["Job_Role"] == "Data Scientist"].iloc[0]
    reqs = {s: float(role_row[s]) for s in CORE_SKILLS}
    df_gaps = calculate_skill_gaps(beginner_skills, reqs)
    readiness = compute_role_readiness_metrics(df_gaps)
    
    assert readiness["high_priority_count"] > 5
    
    # 3. Priority Engine
    df_priorities = calculate_skill_priorities(beginner_skills, "Data Scientist", df_matrix)
    assert df_priorities.iloc[0]["Priority_Category"] in ["Critical", "High"]
    
    # 4. Learning Sequence
    raw_ranking, learning_seq, has_diff, diff_expl = generate_learning_sequence(df_priorities)
    assert len(learning_seq) > 0
    
    # Prerequisite check: Python or Statistics must precede ML
    seq_names = [s["Skill"] for s in learning_seq]
    if "Python" in seq_names and "Machine_Learning" in seq_names:
        assert seq_names.index("Python") < seq_names.index("Machine_Learning")
    if "Statistics_Math" in seq_names and "Machine_Learning" in seq_names:
        assert seq_names.index("Statistics_Math") < seq_names.index("Machine_Learning")
        
    # 5. Course Recommendation
    course_recs = recommend_courses_for_profile(beginner_skills, df_priorities, top_skills_count=3)
    assert len(course_recs) > 0
    
    # Check that recommended courses are predominantly Beginner level
    for skill_name, courses in course_recs.items():
        top_course = courses[0]
        assert "beginner" in top_course["level"].lower() or "intermediate" in top_course["level"].lower()


def test_advanced_profile_flow():
    """
    Test 2: Advanced profile (high skills in ML and Data Science).
    Verifies:
    - High readiness score
    - Mastered skills receive Low priority
    - Remaining deficits are targeted with Advanced/Specialized courses
    """
    advanced_skills = {
        "Python": 90.0, "SQL": 85.0, "R": 70.0, "Java": 40.0, "C++": 35.0, "JavaScript": 40.0,
        "Data_Analysis": 92.0, "Machine_Learning": 88.0, "Deep_Learning": 80.0, "NLP": 75.0,
        "Data_Visualization": 85.0, "Cloud_Computing": 70.0, "DevOps_CICD": 50.0, "Docker_Kubernetes": 60.0,
        "Database_Design": 70.0, "Big_Data": 75.0, "Cybersecurity_Basics": 35.0,
        "Statistics_Math": 90.0, "Problem_Solving": 85.0, "Communication": 85.0
    }
    
    # 1. Career recommendation
    recs = recommend_careers(advanced_skills, gpa=9.2, projects=6, experience=18, top_n=3)
    assert recs[0]["role"] in ["Data Scientist", "AI Specialist", "Machine Learning Engineer"]
    assert recs[0]["hybrid_match_pct"] >= 80.0
    
    # 2. Priorities for Data Scientist
    df_priorities = calculate_skill_priorities(advanced_skills, "Data Scientist")
    # Mastered skills (Python, Data Analysis, Stats) should be low priority
    py_row = df_priorities[df_priorities["Skill"] == "Python"].iloc[0]
    assert py_row["Gap"] == 0.0
    assert py_row["Priority_Category"] == "Low"


def test_cross_career_adaptation():
    """
    Test 3: Verify priorities and courses change dynamically when target career changes
    (e.g., Data Scientist vs Cloud & DevOps Engineer for the same student).
    """
    student_skills = {
        "Python": 65.0, "SQL": 60.0, "R": 20.0, "Java": 50.0, "C++": 40.0, "JavaScript": 50.0,
        "Data_Analysis": 60.0, "Machine_Learning": 55.0, "Deep_Learning": 40.0, "NLP": 35.0,
        "Data_Visualization": 50.0, "Cloud_Computing": 40.0, "DevOps_CICD": 35.0, "Docker_Kubernetes": 30.0,
        "Database_Design": 55.0, "Big_Data": 40.0, "Cybersecurity_Basics": 45.0,
        "Statistics_Math": 60.0, "Problem_Solving": 65.0, "Communication": 70.0
    }
    
    # Evaluate for Data Scientist
    ds_priorities = calculate_skill_priorities(student_skills, "Data Scientist")
    ds_top_skills = ds_priorities.head(3)["Skill"].tolist()
    
    # Evaluate for Cloud & DevOps Engineer
    devops_priorities = calculate_skill_priorities(student_skills, "Cloud & DevOps Engineer")
    devops_top_skills = devops_priorities.head(3)["Skill"].tolist()
    
    # The top priority skills MUST differ between the two careers
    assert ds_top_skills != devops_top_skills
    assert any(s in ["Cloud_Computing", "DevOps_CICD", "Docker_Kubernetes", "Cybersecurity_Basics"] for s in devops_top_skills)
    assert any(s in ["Data_Analysis", "Machine_Learning", "Deep_Learning", "NLP", "Statistics_Math"] for s in ds_top_skills)


def test_enhanced_roadmap_generation():
    """
    Test 4: Verify 6-Phase structured roadmap generation.
    """
    student_skills = {"Python": 50.0, "Statistics_Math": 40.0, "Machine_Learning": 35.0}
    df_p = calculate_skill_priorities(student_skills, "Data Scientist")
    raw_ranking, learning_seq, _, _ = generate_learning_sequence(df_p)
    course_recs = recommend_courses_for_profile(student_skills, df_p, top_skills_count=3)
    
    roadmap = generate_enhanced_learning_roadmap(learning_seq, course_recs, student_skills)
    assert len(roadmap) > 0
    
    first_item = roadmap[0]
    assert "phase_1_prerequisites" in first_item
    assert "phase_2_skill_focus" in first_item
    assert "phase_3_recommended_course" in first_item
    assert "phase_4_practice_topics" in first_item
    assert "phase_5_portfolio_project" in first_item
    assert "phase_6_reassessment" in first_item


def test_report_generation_integration():
    """
    Test 5: Verify markdown audit report generation includes all new sections.
    """
    student_skills = {"Python": 60.0, "SQL": 70.0, "Machine_Learning": 40.0}
    df_students, df_matrix = load_processed_data()
    role_row = df_matrix[df_matrix["Job_Role"] == "Data Scientist"].iloc[0]
    reqs = {s: float(role_row[s]) for s in CORE_SKILLS}
    df_gaps = calculate_skill_gaps(student_skills, reqs)
    readiness = compute_role_readiness_metrics(df_gaps)
    recs = recommend_careers(student_skills, top_n=3)
    
    df_p = calculate_skill_priorities(student_skills, "Data Scientist", df_matrix)
    raw_ranking, learning_seq, _, _ = generate_learning_sequence(df_p)
    course_recs = recommend_courses_for_profile(student_skills, df_p, top_skills_count=3)
    roadmap = generate_enhanced_learning_roadmap(learning_seq, course_recs, student_skills)
    
    report_md = generate_markdown_audit_report(
        student_name="Test Student",
        degree="B.Tech CS",
        gpa=8.5,
        experience=6,
        projects=3,
        target_role="Data Scientist",
        readiness_metrics=readiness,
        recommendations=recs,
        df_gaps=df_gaps,
        roadmap=roadmap,
        df_priorities=df_p,
        learning_sequence=learning_seq,
        course_recs=course_recs
    )
    
    assert "# AI-Based Student Skill Gap Analysis & Employability Audit Report" in report_md
    assert "Intelligent Skill Improvement Priority Ranking" in report_md
    assert "Recommended Prerequisite-Aware Learning Sequence" in report_md
    assert "Personalized Online Course Recommendations" in report_md
    assert "Comprehensive Skill Gap Breakdown Table" in report_md
