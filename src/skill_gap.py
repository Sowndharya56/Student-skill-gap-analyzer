"""
Skill Gap Analysis & Priority Recommendation Engine.
Formulation:
    Skill Gap = max(0.0, Required_Skill_Level - Student_Skill_Level)
    
Severity Categorization:
    - Mastered / Sufficient (Gap = 0 pts)
    - Low Priority Gap (1 <= Gap <= 15 pts)
    - Medium Priority Gap (16 <= Gap <= 35 pts)
    - High Priority Gap (Gap > 35 pts)
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import logging
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

from config.config import (
    CORE_SKILLS,
    SKILL_DISPLAY_NAMES,
    GAP_THRESHOLDS,
    LEARNING_RESOURCES,
    SKILL_DEPENDENCY_GRAPH
)


def get_gap_severity(gap: float) -> Tuple[str, str]:
    """
    Returns the severity label and color hex code based on configurable GAP_THRESHOLDS.
    """
    if gap <= 0.0:
        return GAP_THRESHOLDS["NONE"][2], GAP_THRESHOLDS["NONE"][3]
    elif gap <= 15.0:
        return GAP_THRESHOLDS["LOW"][2], GAP_THRESHOLDS["LOW"][3]
    elif gap <= 35.0:
        return GAP_THRESHOLDS["MEDIUM"][2], GAP_THRESHOLDS["MEDIUM"][3]
    else:
        return GAP_THRESHOLDS["HIGH"][2], GAP_THRESHOLDS["HIGH"][3]


def validate_and_clamp_score(value: Any, default: float = 0.0) -> float:
    """Safely parses and clamps any score value strictly into [0.0, 100.0]."""
    try:
        val = float(value)
        if np.isnan(val):
            return default
        return float(np.clip(val, 0.0, 100.0))
    except (ValueError, TypeError):
        return default


def calculate_skill_gaps(
    student_skills: Dict[str, Any],
    required_skills: Dict[str, Any]
) -> pd.DataFrame:
    """
    Calculates non-negative skill gap for all required skills.
    
    Formula:
        Skill_Gap = max(0.0, Required_Skill - Student_Skill)
        
    Args:
        student_skills: Dict of {skill_name: student_score (0-100)}
        required_skills: Dict of {skill_name: required_score (0-100)}
        
    Returns:
        pd.DataFrame sorted by Gap descending with columns:
            ['Skill', 'Display_Name', 'Student_Score', 'Required_Score', 'Gap', 'Severity', 'Color']
    """
    records = []
    
    if not required_skills:
        return pd.DataFrame(columns=["Skill", "Display_Name", "Student_Score", "Required_Score", "Gap", "Severity", "Color"])
        
    for skill, req_raw in required_skills.items():
        req_score = validate_and_clamp_score(req_raw, default=50.0)
        stu_raw = student_skills.get(skill, 0.0) if student_skills else 0.0
        stu_score = validate_and_clamp_score(stu_raw, default=0.0)
        
        # Core Rule: Non-negative gap
        gap = max(0.0, req_score - stu_score)
        severity_label, color_code = get_gap_severity(gap)
        
        records.append({
            "Skill": skill,
            "Display_Name": SKILL_DISPLAY_NAMES.get(skill, skill),
            "Student_Score": round(stu_score, 1),
            "Required_Score": round(req_score, 1),
            "Gap": round(gap, 1),
            "Severity": severity_label,
            "Color": color_code
        })
        
    df_gaps = pd.DataFrame(records)
    if not df_gaps.empty:
        df_gaps.sort_values(by="Gap", ascending=False, inplace=True)
        df_gaps.reset_index(drop=True, inplace=True)
    return df_gaps


def generate_learning_roadmap(df_gaps: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Generates actionable, prioritized learning roadmap for skills that have positive gaps.
    """
    roadmap = []
    
    if df_gaps.empty:
        return roadmap
        
    for _, row in df_gaps.iterrows():
        skill = row["Skill"]
        gap = row["Gap"]
        
        resources = LEARNING_RESOURCES.get(skill, {
            "topics": ["Core Fundamentals", "Intermediate Practice", "Applied Portfolio Project"],
            "recommended_courses": ["Online Specialization Track", "Official Documentation & Tutorials"],
            "suggested_project": f"Build an End-to-End {skill} Application"
        })
        
        est_hours = max(5, int((gap / 10.0) * 12)) if gap > 0 else 0
        
        item = {
            "skill": skill,
            "display_name": row["Display_Name"],
            "student_score": row["Student_Score"],
            "required_score": row["Required_Score"],
            "gap": gap,
            "severity": row["Severity"],
            "color": row["Color"],
            "estimated_hours": est_hours,
            "priority_level": "High" if gap > 35 else ("Medium" if gap > 15 else ("Low" if gap > 0 else "None")),
            "topics": resources["topics"],
            "courses": resources["recommended_courses"],
            "suggested_project": resources["suggested_project"]
        }
        roadmap.append(item)
        
    return roadmap


def generate_enhanced_learning_roadmap(
    learning_sequence: List[Dict[str, Any]],
    course_recs: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    student_skills: Optional[Dict[str, Any]] = None,
    dependency_graph: Optional[Dict[str, List[str]]] = None
) -> List[Dict[str, Any]]:
    """
    Generates an enhanced, structured 6-Phase actionable roadmap for prioritized skills.
    
    Phases:
        PHASE 1: Prerequisite / Foundation Skills
        PHASE 2: Highest Priority Skill (Core Focus)
        PHASE 3: Recommended Course
        PHASE 4: Practice (Key Topics & Exercises)
        PHASE 5: Portfolio Project
        PHASE 6: Re-assessment Checkpoint
        
    Args:
        learning_sequence: List of prioritized skill records in recommended learning order
        course_recs: Dict mapping skill_name -> list of recommended course dicts
        student_skills: Dict of {skill_name: student_score}
        dependency_graph: Optional DAG of skill prerequisites
        
    Returns:
        List of 6-Phase structured roadmap items.
    """
    enhanced_roadmap = []
    if not learning_sequence:
        return enhanced_roadmap
        
    graph = dependency_graph or SKILL_DEPENDENCY_GRAPH
    # Reverse graph: find which skills are prerequisites FOR skill S
    prereqs_for_skill = defaultdict(list)
    for u, downstreams in graph.items():
        for v in downstreams:
            prereqs_for_skill[v].append(u)
            
    for step_num, item in enumerate(learning_sequence, 1):
        skill = item.get("Skill", item.get("skill", ""))
        gap = item.get("Gap", item.get("gap", 0.0))
        stu_score = item.get("Student_Score", item.get("student_score", 0.0))
        req_score = item.get("Required_Score", item.get("required_score", 0.0))
        disp_name = item.get("Display_Name", item.get("display_name", SKILL_DISPLAY_NAMES.get(skill, skill)))
        priority_label = item.get("Priority_Category", item.get("priority_level", "Medium"))
        priority_badge = item.get("Priority_Badge", f"📌 {priority_label}")
        reason = item.get("Reason", f"Targeted focus skill with a gap of {gap:.0f} points.")
        
        # 1. Phase 1: Prerequisites Check
        prereqs = prereqs_for_skill.get(skill, [])
        prereq_status = []
        for p in prereqs:
            p_disp = SKILL_DISPLAY_NAMES.get(p, p)
            p_score = student_skills.get(p, 0.0) if student_skills else 0.0
            if p_score >= 60.0:
                prereq_status.append(f"✅ {p_disp} (Proficiency: {p_score:.0f}/100 — Satisfied)")
            else:
                prereq_status.append(f"⚠️ {p_disp} (Proficiency: {p_score:.0f}/100 — Strengthen First)")
                
        if not prereq_status:
            prereq_status = ["✅ No strict prerequisites required (Beginner-accessible foundational domain)."]
            
        # 2. Phase 2: Skill Deficit Focus Details
        est_hours = max(5, int((gap / 10.0) * 12)) if gap > 0 else 0
        
        # 3. Phase 3: Recommended Course
        skill_courses = course_recs.get(skill, []) if course_recs else []
        top_course = skill_courses[0] if skill_courses else None
        
        # 4. Phase 4 & 5: Topics & Project
        resources = LEARNING_RESOURCES.get(skill, {
            "topics": ["Core Fundamentals", "Intermediate Practice", "Applied Portfolio Project"],
            "recommended_courses": ["Online Specialization Track"],
            "suggested_project": f"Build an End-to-End {disp_name} Application"
        })
        
        roadmap_entry = {
            "step_number": step_num,
            "skill": skill,
            "display_name": disp_name,
            "student_score": stu_score,
            "required_score": req_score,
            "gap": gap,
            "priority_label": priority_label,
            "priority_badge": priority_badge,
            "estimated_hours": est_hours,
            "reason": reason,
            
            # 6 Formal Roadmap Phases
            "phase_1_prerequisites": prereq_status,
            "phase_2_skill_focus": {
                "skill": disp_name,
                "current_score": stu_score,
                "required_score": req_score,
                "gap": gap,
                "priority": priority_label,
                "reason": reason
            },
            "phase_3_recommended_course": top_course,
            "phase_4_practice_topics": resources["topics"],
            "phase_5_portfolio_project": resources["suggested_project"],
            "phase_6_reassessment": f"After completing the recommended course and portfolio project, reassess your {disp_name} score to re-evaluate your career readiness."
        }
        enhanced_roadmap.append(roadmap_entry)
        
    return enhanced_roadmap


def compute_role_readiness_metrics(df_gaps: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculates summary readiness metrics for a target job role.
    """
    if df_gaps.empty:
        return {
            "career_match_percentage": 0.0,
            "total_gap_points": 0.0,
            "average_gap_points": 0.0,
            "high_priority_count": 0,
            "medium_priority_count": 0,
            "low_priority_count": 0,
            "mastered_count": 0,
            "total_skills_evaluated": 0
        }
        
    total_required = df_gaps["Required_Score"].sum()
    total_gap = df_gaps["Gap"].sum()
    
    match_pct = max(0.0, 100.0 - (total_gap / (total_required + 1e-9) * 100.0))
    
    high_gaps = (df_gaps["Gap"] > 35).sum()
    med_gaps = ((df_gaps["Gap"] <= 35) & (df_gaps["Gap"] > 15)).sum()
    low_gaps = ((df_gaps["Gap"] <= 15) & (df_gaps["Gap"] > 0)).sum()
    mastered = (df_gaps["Gap"] == 0).sum()
    
    return {
        "career_match_percentage": round(float(match_pct), 1),
        "total_gap_points": round(float(total_gap), 1),
        "average_gap_points": round(float(df_gaps["Gap"].mean()), 1),
        "high_priority_count": int(high_gaps),
        "medium_priority_count": int(med_gaps),
        "low_priority_count": int(low_gaps),
        "mastered_count": int(mastered),
        "total_skills_evaluated": len(df_gaps)
    }


if __name__ == "__main__":
    test_stu = {"Python": 60, "SQL": 80, "Machine_Learning": -10} # Negative test
    test_req = {"Python": 85, "SQL": 75, "Machine_Learning": 88}
    gaps_df = calculate_skill_gaps(test_stu, test_req)
    print(gaps_df)
