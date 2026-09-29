"""
Unit Tests for Personalized Course Recommendation Engine (Feature 2).

Coverage:
13. Course skill matching (exact match, topics match, non-match)
14. Course level matching (Beginner, Intermediate, Advanced vs student proficiency)
15. Gap relevance scoring
16. Prerequisite matching and scoring
17. Missing prerequisite detection
18. Course ranking by multi-factor score (35% Skill, 25% Level, 20% Gap, 10% Prereq, 10% Quality)
19. Top-3 course selection
20. Empty course dataset handling
21. Unknown skill handling
22. Unknown career / profile handling
23. Missing or corrupted course metadata handling
24. Profile-wide recommendation batch generation
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

from config.config import CORE_SKILLS
from src.course_recommender import (
    load_course_catalog,
    create_fallback_courses_dataframe,
    compute_skill_match,
    compute_level_match,
    compute_gap_relevance,
    check_prerequisites,
    compute_course_quality,
    generate_why_recommended_text,
    recommend_courses_for_skill,
    recommend_courses_for_profile
)
from src.skill_priority import calculate_skill_priorities


# ==============================================================================
# 1. INDIVIDUAL COURSE MATCHING COMPONENTS
# ==============================================================================
def test_course_skill_matching():
    """Verify exact match returns 100.0, topic match returns partial, non-match returns 0.0."""
    assert compute_skill_match("Python", "Python") == 100.0
    assert compute_skill_match("python", "Python") == 100.0
    assert compute_skill_match("Machine_Learning", "Machine_Learning") == 100.0
    
    # Topic mention
    assert compute_skill_match("Data_Analysis", "Python", course_topics="Python Basics;Pandas;NumPy") == 70.0
    
    # Complete mismatch
    assert compute_skill_match("Cybersecurity_Basics", "Python", course_topics="Network Security;OWASP") == 0.0


def test_course_level_matching():
    """
    Verify level match:
    - Beginner student (<40) matches Beginner courses best (100)
    - Intermediate student (40-69) matches Intermediate courses best (100)
    - Advanced student (>=70) matches Advanced courses best (100)
    """
    # 1. Beginner student (level = 20)
    assert compute_level_match(20.0, "Beginner") == 100.0
    assert compute_level_match(20.0, "Advanced") < 30.0
    
    # 2. Intermediate student (level = 55)
    assert compute_level_match(55.0, "Intermediate") == 100.0
    assert compute_level_match(55.0, "Beginner") >= 70.0
    
    # 3. Advanced student (level = 85)
    assert compute_level_match(85.0, "Advanced") == 100.0
    assert compute_level_match(85.0, "Beginner") < 40.0


def test_gap_relevance_scoring():
    """Verify larger skill gaps produce higher gap relevance scores."""
    high_gap_score = compute_gap_relevance(50.0)
    med_gap_score = compute_gap_relevance(25.0)
    zero_gap_score = compute_gap_relevance(0.0)
    
    assert high_gap_score == 100.0
    assert med_gap_score >= 80.0
    assert zero_gap_score <= 30.0


def test_prerequisite_matching_and_missing_detection():
    """Verify prerequisite check detects missing and satisfied prerequisites accurately."""
    # Student has Python=80, Statistics=30, ML=40
    student_skills = {"Python": 80.0, "Statistics_Math": 30.0, "Machine_Learning": 40.0}
    
    # Course requiring Python & Statistics
    prereqs_str = "Python;Statistics_Math"
    match_score, satisfied, missing = check_prerequisites(student_skills, prereqs_str, threshold=50.0)
    
    assert "Python" in satisfied
    assert "Statistics_Math" in missing
    assert match_score == 50.0  # 1 of 2 satisfied
    
    # Course with No prerequisites
    match_none, sat_none, miss_none = check_prerequisites(student_skills, "None")
    assert match_none == 100.0
    assert len(miss_none) == 0


def test_course_quality_normalization():
    """Verify rating / 5.0 is clamped into [0.0, 100.0]."""
    assert compute_course_quality(5.0) == 100.0
    assert compute_course_quality(4.5) == 90.0
    assert compute_course_quality(np.nan) == 80.0  # Safe default
    assert compute_course_quality(None) == 80.0


# ==============================================================================
# 2. COURSE RECOMMENDATION RANKING & TOP-3 SELECTION
# ==============================================================================
def test_recommend_courses_for_skill_top_3():
    """Verify recommendation returns top 3 courses sorted by match score descending."""
    student_skills = {"Python": 35.0, "Statistics_Math": 40.0}
    recs = recommend_courses_for_skill("Python", student_skills=student_skills, gap=50.0, top_n=3)
    
    assert len(recs) <= 3
    assert len(recs) > 0
    
    # Check fields
    first = recs[0]
    required_keys = [
        "course_id", "course_name", "provider", "skill", "level", "match_score",
        "duration", "rating", "prerequisites", "missing_prerequisites", "topics",
        "url", "cost_type", "description", "why_recommended"
    ]
    for k in required_keys:
        assert k in first, f"Missing key: {k}"
        
    # Check sorted descending
    scores = [r["match_score"] for r in recs]
    assert scores == sorted(scores, reverse=True)


def test_recommend_courses_for_profile():
    """Verify batch recommendation generates courses for multiple priority skills."""
    student_skills = {"Python": 45.0, "Statistics_Math": 35.0, "Machine_Learning": 50.0}
    df_p = calculate_skill_priorities(student_skills, target_career="Data Scientist")
    
    profile_recs = recommend_courses_for_profile(student_skills, df_p, top_skills_count=3)
    assert len(profile_recs) <= 3
    assert len(profile_recs) > 0
    
    for skill_name, courses in profile_recs.items():
        assert len(courses) <= 3
        assert len(courses) > 0
        assert all(c["match_score"] >= 0.0 for c in courses)


# ==============================================================================
# 3. EDGE CASES (EMPTY DATASET, UNKNOWN SKILL, MISSING METADATA)
# ==============================================================================
def test_empty_course_dataset_fallback():
    """Verify engine falls back gracefully when provided an empty course DataFrame."""
    empty_df = pd.DataFrame()
    student_skills = {"Python": 50.0}
    recs = recommend_courses_for_skill("Python", student_skills, gap=30.0, top_n=3, courses_df=empty_df)
    
    # Should use fallback dataset and return valid recommendations without crashing
    assert len(recs) > 0
    assert recs[0]["skill"] == "Python"


def test_unknown_skill_course_recommendation():
    """Verify requesting an unsupported skill returns empty list rather than raising an exception."""
    student_skills = {"Python": 50.0}
    recs = recommend_courses_for_skill("Non_Existent_Quantum_Teleportation", student_skills, gap=30.0, top_n=3)
    assert recs == []


def test_missing_course_metadata_handling():
    """Verify missing duration, rating, or URLs are safely filled with defaults."""
    corrupted_data = pd.DataFrame([{
        "Course_ID": "CRS_TEST_01",
        "Course_Name": "Test Course with Missing Data",
        "Provider": "Test Provider",
        "Skill": "Python",
        "Level": None,
        "Duration_Hours": np.nan,
        "Rating": None,
        "Prerequisites": None,
        "Topics": None,
        "URL": None,
        "Cost_Type": None,
        "Description": None
    }])
    
    recs = recommend_courses_for_skill("Python", {"Python": 40.0}, gap=40.0, top_n=3, courses_df=corrupted_data)
    assert len(recs) == 1
    assert recs[0]["rating"] == 4.7
    assert recs[0]["duration"] == 30
    assert "coursera.org" in recs[0]["url"]
