"""
Unit Tests for Intelligent Skill Improvement Priority Engine (Feature 1).

Coverage:
1. Gap severity calculation & normalization
2. Career importance calculation from benchmark
3. Dependency impact calculation using DAG
4. Current weakness calculation
5. Multi-factor priority score calculation (40% Gap, 30% Importance, 20% Dependency, 10% Weakness)
6. Priority category thresholds (Critical, High, Medium, Low)
7. Priority ranking ordering (descending)
8. Learning sequence generation
9. Prerequisite topological ordering (e.g. Statistics before ML when both weak)
10. Empty student profile handling
11. Zero-skill profile handling
12. Fully-skilled / mastered profile handling
13. Unknown target career handling
14. Transitive downstream dependency calculation
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
    PRIORITY_WEIGHTS,
    PRIORITY_CATEGORIES,
    SKILL_DEPENDENCY_GRAPH
)
from src.skill_priority import (
    compute_gap_severity,
    compute_career_importance,
    compute_current_weakness,
    compute_dependency_impact,
    get_transitive_downstream_skills,
    get_priority_category,
    generate_priority_reason,
    calculate_skill_priorities,
    generate_learning_sequence
)


# ==============================================================================
# 1. INDIVIDUAL FACTOR CALCULATIONS
# ==============================================================================
def test_gap_severity_calculation():
    """Verify gap severity strictly clamped in [0.0, 100.0]."""
    assert compute_gap_severity(0.0) == 0.0
    assert compute_gap_severity(25.5) == 25.5
    assert compute_gap_severity(100.0) == 100.0
    assert compute_gap_severity(150.0) == 100.0  # Clamping above
    assert compute_gap_severity(-20.0) == 0.0    # Clamping below


def test_career_importance_calculation():
    """Verify career importance maps directly to required benchmark in [0.0, 100.0]."""
    assert compute_career_importance(95.0) == 95.0
    assert compute_career_importance(50.0) == 50.0
    assert compute_career_importance(0.0) == 0.0
    assert compute_career_importance(120.0) == 100.0


def test_current_weakness_calculation():
    """Verify lower student proficiency produces higher weakness (100 - student_score)."""
    assert compute_current_weakness(20.0) == 80.0
    assert compute_current_weakness(90.0) == 10.0
    assert compute_current_weakness(0.0) == 100.0
    assert compute_current_weakness(100.0) == 0.0
    assert compute_current_weakness(-10.0) == 100.0
    assert compute_current_weakness(120.0) == 0.0


def test_transitive_downstream_dependencies():
    """Verify recursive downstream skill traversal through DAG."""
    # Python -> Data_Analysis -> Machine_Learning -> Deep_Learning -> NLP
    py_downstream = get_transitive_downstream_skills("Python")
    assert "Data_Analysis" in py_downstream
    assert "Machine_Learning" in py_downstream
    assert "Deep_Learning" in py_downstream
    assert "NLP" in py_downstream
    
    # NLP is a leaf node in the DAG
    nlp_downstream = get_transitive_downstream_skills("NLP")
    assert len(nlp_downstream) == 0


def test_dependency_impact_calculation():
    """Verify dependency impact score increases when a weak skill blocks required downstream skills."""
    all_reqs = {
        "Python": 85.0,
        "Data_Analysis": 90.0,
        "Machine_Learning": 88.0,
        "Deep_Learning": 75.0,
        "NLP": 70.0,
        "Big_Data": 70.0,
        "DevOps_CICD": 50.0
    }
    
    # 1. Weak in Python (score = 20) -> should have substantial dependency impact
    weak_py_impact = compute_dependency_impact("Python", student_score=20.0, required_score=85.0, all_required_skills=all_reqs)
    assert weak_py_impact > 40.0
    
    # 2. Mastered Python (score = 90) -> dependency impact should be 0 (not blocking)
    mastered_py_impact = compute_dependency_impact("Python", student_score=90.0, required_score=85.0, all_required_skills=all_reqs)
    assert mastered_py_impact == 0.0
    
    # 3. Leaf skill (Communication or NLP) with no downstream -> dependency impact should be 0
    leaf_impact = compute_dependency_impact("Communication", student_score=20.0, required_score=80.0, all_required_skills=all_reqs)
    assert leaf_impact == 0.0


# ==============================================================================
# 2. PRIORITY CATEGORY THRESHOLDS
# ==============================================================================
def test_priority_categories():
    """Verify category thresholds: Critical (80-100), High (60-79), Medium (40-59), Low (0-39)."""
    cat_85, badge_85, col_85 = get_priority_category(85.0)
    assert cat_85 == "Critical"
    assert "Critical" in badge_85
    
    cat_65, badge_65, col_65 = get_priority_category(65.0)
    assert cat_65 == "High"
    assert "High" in badge_65
    
    cat_45, badge_45, col_45 = get_priority_category(45.0)
    assert cat_45 == "Medium"
    assert "Medium" in badge_45
    
    cat_20, badge_20, col_20 = get_priority_category(20.0)
    assert cat_20 == "Low"
    assert "Low" in badge_20


# ==============================================================================
# 3. PRIORITY SCORE & RANKING
# ==============================================================================
def test_calculate_skill_priorities_structure_and_ordering():
    """Verify calculate_skill_priorities returns all 20 skills sorted by Priority_Score descending."""
    student_skills = {
        "Python": 40,
        "SQL": 75,
        "Statistics_Math": 30,
        "Machine_Learning": 35,
        "Data_Analysis": 45
    }
    df_p = calculate_skill_priorities(student_skills, target_career="Data Scientist")
    
    assert len(df_p) == len(CORE_SKILLS)
    required_cols = [
        "Skill", "Display_Name", "Student_Score", "Required_Score", "Gap",
        "Gap_Severity", "Career_Importance", "Dependency_Impact", "Current_Weakness",
        "Priority_Score", "Priority_Category", "Priority_Badge", "Priority_Color", "Reason"
    ]
    for col in required_cols:
        assert col in df_p.columns
        
    # Verify descending sort
    scores = df_p["Priority_Score"].tolist()
    assert scores == sorted(scores, reverse=True)
    
    # Verify all scores strictly in [0.0, 100.0]
    assert all(0.0 <= s <= 100.0 for s in scores)


def test_priority_reason_generation():
    """Verify generated reasons contain specific numbers and context."""
    reason = generate_priority_reason(
        skill="Machine_Learning",
        gap=45.0,
        required_score=90.0,
        student_score=45.0,
        dependency_impact=60.0,
        target_career="Machine Learning Engineer"
    )
    assert "gap of 45" in reason or "45 points" in reason
    assert "Machine Learning Engineer" in reason
    assert len(reason) > 20


# ==============================================================================
# 4. LEARNING SEQUENCE & PREREQUISITE ORDERING
# ==============================================================================
def test_learning_sequence_prerequisite_ordering():
    """
    Verify prerequisite ordering: If Statistics is a prerequisite for Machine Learning,
    and both have gaps, Statistics should precede Machine Learning in the learning sequence.
    """
    student_skills = {
        "Statistics_Math": 35.0,
        "Machine_Learning": 40.0,
        "Python": 50.0
    }
    df_p = calculate_skill_priorities(student_skills, target_career="Data Scientist")
    raw_ranking, learning_seq, has_diff, diff_expl = generate_learning_sequence(df_p)
    
    seq_skill_names = [item["Skill"] for item in learning_seq]
    
    # Check that Statistics precedes Machine Learning
    if "Statistics_Math" in seq_skill_names and "Machine_Learning" in seq_skill_names:
        stat_idx = seq_skill_names.index("Statistics_Math")
        ml_idx = seq_skill_names.index("Machine_Learning")
        assert stat_idx < ml_idx, f"Statistics (index {stat_idx}) must precede Machine Learning (index {ml_idx}) in learning sequence"
        
    # Check that Python precedes Data_Analysis
    if "Python" in seq_skill_names and "Data_Analysis" in seq_skill_names:
        py_idx = seq_skill_names.index("Python")
        da_idx = seq_skill_names.index("Data_Analysis")
        assert py_idx < da_idx, f"Python (index {py_idx}) must precede Data Analysis (index {da_idx}) in learning sequence"


# ==============================================================================
# 5. EDGE CASES (EMPTY, ZERO, FULLY SKILLED, UNKNOWN CAREER)
# ==============================================================================
def test_empty_student_profile():
    """Verify priority engine handles empty skills dict gracefully."""
    df_p = calculate_skill_priorities({}, target_career="Data Scientist")
    assert len(df_p) == len(CORE_SKILLS)
    assert all(df_p["Student_Score"] == 0.0)
    assert all(df_p["Priority_Score"] >= 0.0)


def test_zero_skills_profile():
    """Verify priority engine handles all-zero proficiencies."""
    zero_profile = {s: 0.0 for s in CORE_SKILLS}
    df_p = calculate_skill_priorities(zero_profile, target_career="Cloud & DevOps Engineer")
    assert len(df_p) == len(CORE_SKILLS)
    # Foundational prerequisites like Cloud Computing & DevOps should receive high priority
    top_skills = df_p.head(3)["Skill"].tolist()
    assert any(s in ["Cloud_Computing", "DevOps_CICD", "Docker_Kubernetes", "Problem_Solving"] for s in top_skills)


def test_fully_skilled_mastered_profile():
    """Verify profile with all skills at 100 has 0 gaps and low priority scores."""
    mastered_profile = {s: 100.0 for s in CORE_SKILLS}
    df_p = calculate_skill_priorities(mastered_profile, target_career="Data Scientist")
    assert len(df_p) == len(CORE_SKILLS)
    assert all(df_p["Gap"] == 0.0)
    assert all(df_p["Priority_Category"] == "Low")


def test_unknown_target_career():
    """Verify fallback when target career is not in benchmark matrix."""
    df_p = calculate_skill_priorities({"Python": 60}, target_career="Quantum Computing Architect")
    assert len(df_p) == len(CORE_SKILLS)
    assert all(df_p["Priority_Score"] >= 0.0)
