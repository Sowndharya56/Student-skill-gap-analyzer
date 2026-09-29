"""
Personalized Course Recommendation Engine.

Answers:
"What should this student learn to close their highest-priority skill gap?"

Course Matching Algorithm:
    Course Match Score =
        (0.35 * Skill_Match)
      + (0.25 * Level_Match)
      + (0.20 * Gap_Relevance)
      + (0.10 * Prerequisite_Match)
      + (0.10 * Course_Quality)

Normalized to 0–100 scale.
Recommends the Top 3 best-fitting courses per priority skill.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import logging
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

from config.config import (
    CORE_SKILLS,
    SKILL_DISPLAY_NAMES,
    COURSES_DATA,
    COURSE_MATCH_WEIGHTS,
    LEARNING_RESOURCES
)
from src.skill_gap import validate_and_clamp_score


def load_course_catalog(courses_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Loads the course catalog from disk with schema validation and fallback.
    """
    path = courses_path or COURSES_DATA
    if path.exists():
        try:
            df_courses = pd.read_csv(path)
            required_cols = ["Course_ID", "Course_Name", "Provider", "Skill", "Level", "Duration_Hours", "Rating"]
            if all(col in df_courses.columns for col in required_cols):
                return df_courses
            else:
                logger.warning("Course dataset missing required columns. Creating fallback.")
        except Exception as e:
            logger.error("Error reading courses dataset at %s: %s", path, e)
            
    # Fallback minimal course dataset if file is unavailable
    return create_fallback_courses_dataframe()


def create_fallback_courses_dataframe() -> pd.DataFrame:
    """
    Creates an in-memory fallback course dataframe from LEARNING_RESOURCES
    to ensure zero application crashes even if courses.csv is missing.
    """
    records = []
    for skill in CORE_SKILLS:
        res = LEARNING_RESOURCES.get(skill, {
            "topics": ["Foundations", "Practical Projects"],
            "recommended_courses": [f"Online {skill} Specialization"],
            "suggested_project": f"Build a {skill} Project"
        })
        topics_str = ";".join(res.get("topics", ["Core Concepts"]))
        courses_list = res.get("recommended_courses", [f"{skill} Masterclass"])
        
        for idx, c_name in enumerate(courses_list):
            lvl = "Beginner" if idx == 0 else "Intermediate"
            records.append({
                "Course_ID": f"FB_{skill}_{idx+1}",
                "Course_Name": c_name,
                "Provider": "Verified Online Provider (Demo)",
                "Skill": skill,
                "Level": lvl,
                "Duration_Hours": 30 + (idx * 10),
                "Rating": 4.7,
                "Prerequisites": "None" if idx == 0 else "Python",
                "Topics": topics_str,
                "URL": "https://www.coursera.org",
                "Cost_Type": "Free Audit",
                "Description": f"Comprehensive {lvl.lower()} course focused on {SKILL_DISPLAY_NAMES.get(skill, skill)}."
            })
    return pd.DataFrame(records)


def compute_skill_match(course_skill: str, target_skill: str, course_topics: str = "") -> float:
    """
    Evaluates skill alignment between the course and the target skill deficit.
    - Exact primary skill match = 100.0
    - Related topic match = 70.0
    - Otherwise = 0.0
    """
    if not course_skill or not target_skill:
        return 0.0
    if str(course_skill).strip().lower() == str(target_skill).strip().lower():
        return 100.0
        
    # Check if target skill is mentioned in course topics
    clean_target = target_skill.replace("_", " ").lower()
    if course_topics and clean_target in str(course_topics).lower():
        return 70.0
        
    return 0.0


def compute_level_match(student_level: float, course_level: str) -> float:
    """
    Evaluates how appropriately the course difficulty matches the student's current proficiency.
    
    Student proficiency tiers:
    - Beginner: < 40
    - Intermediate: 40 to 69
    - Advanced: >= 70
    """
    stu_val = float(np.clip(student_level, 0.0, 100.0))
    lvl = str(course_level).strip().lower()
    
    if stu_val < 40.0:  # Beginner student
        if "beginner" in lvl and "intermediate" not in lvl:
            return 100.0
        elif "beginner" in lvl or "all" in lvl:
            return 85.0
        elif "intermediate" in lvl:
            return 50.0
        else:  # Advanced
            return 15.0
    elif stu_val < 70.0:  # Intermediate student
        if "intermediate" in lvl and "beginner" not in lvl and "advanced" not in lvl:
            return 100.0
        elif "intermediate" in lvl:
            return 90.0
        elif "beginner" in lvl:
            return 75.0
        else:  # Advanced
            return 60.0
    else:  # Advanced student (>= 70)
        if "advanced" in lvl:
            return 100.0
        elif "intermediate" in lvl:
            return 75.0
        elif "beginner" in lvl:
            return 30.0
        else:
            return 70.0


def compute_gap_relevance(gap: float) -> float:
    """
    Evaluates relevance based on the size of the student's skill deficit.
    Larger gap = higher urgency to take a dedicated course.
    """
    gap_val = float(np.clip(gap, 0.0, 100.0))
    if gap_val >= 40.0:
        return 100.0
    elif gap_val >= 20.0:
        return 80.0
    elif gap_val > 0.0:
        return 60.0
    else:
        return 30.0


def check_prerequisites(
    student_skills: Dict[str, Any],
    prerequisites_str: str,
    threshold: float = 50.0
) -> Tuple[float, List[str], List[str]]:
    """
    Checks if the student satisfies the course prerequisites.
    
    Args:
        student_skills: Dict of {skill_name: student_score}
        prerequisites_str: Semicolon-separated string of prerequisite skill names, or "None"
        threshold: Minimum score to consider prerequisite satisfied (default: 50.0)
        
    Returns:
        (prerequisite_match_score_0_100, satisfied_prereqs_list, missing_prereqs_list)
    """
    if not prerequisites_str or str(prerequisites_str).strip().lower() in ["none", "nan", ""]:
        return 100.0, [], []
        
    prereq_tokens = [p.strip() for p in str(prerequisites_str).split(";") if p.strip()]
    if not prereq_tokens:
        return 100.0, [], []
        
    satisfied = []
    missing = []
    
    for token in prereq_tokens:
        # Check if prerequisite has a custom threshold syntax (e.g. Python>=60)
        if ">=" in token:
            parts = token.split(">=")
            req_skill = parts[0].strip()
            min_score = float(parts[1].strip())
        elif ">" in token:
            parts = token.split(">")
            req_skill = parts[0].strip()
            min_score = float(parts[1].strip())
        else:
            req_skill = token.strip()
            min_score = threshold
            
        stu_score = validate_and_clamp_score(student_skills.get(req_skill, 0.0)) if student_skills else 0.0
        
        if stu_score >= min_score:
            satisfied.append(req_skill)
        else:
            missing.append(req_skill)
            
    match_score = (len(satisfied) / len(prereq_tokens)) * 100.0
    return float(round(match_score, 1)), satisfied, missing


def compute_course_quality(rating: Any) -> float:
    """
    Normalizes course rating (out of 5.0) to a 0–100 scale.
    """
    try:
        r = float(rating)
        if np.isnan(r) or r <= 0:
            return 80.0
        return float(np.clip((r / 5.0) * 100.0, 0.0, 100.0))
    except (ValueError, TypeError):
        return 80.0


def generate_why_recommended_text(
    course_name: str,
    skill: str,
    level: str,
    match_score: float,
    gap: float,
    missing_prereqs: List[str]
) -> str:
    """
    Generates an actionable, personalized explanation for why this course was recommended.
    """
    disp_skill = SKILL_DISPLAY_NAMES.get(skill, skill)
    points = []
    
    if gap >= 35.0:
        points.append(f"Directly targets your critical {gap:.0f}-point deficit in {disp_skill}")
    elif gap > 0.0:
        points.append(f"Structured to bridge your {gap:.0f}-point gap in {disp_skill}")
    else:
        points.append(f"Great resource for maintaining expertise in {disp_skill}")
        
    points.append(f"tailored to the {level} difficulty tier")
    
    if missing_prereqs:
        missing_names = [SKILL_DISPLAY_NAMES.get(m, m) for m in missing_prereqs]
        return f"{' and '.join(points)}. Note: Consider strengthening {', '.join(missing_names)} alongside this course."
    else:
        return f"{' and '.join(points)} with all prerequisites currently satisfied."


def recommend_courses_for_skill(
    skill: str,
    student_skills: Dict[str, Any],
    gap: float = 0.0,
    top_n: int = 3,
    courses_df: Optional[pd.DataFrame] = None
) -> List[Dict[str, Any]]:
    """
    Recommends the Top N courses for a single prioritized technical skill.
    
    Formula:
        Match Score = (0.35 * Skill_Match) + (0.25 * Level_Match) + (0.20 * Gap_Relevance)
                    + (0.10 * Prereq_Match) + (0.10 * Quality)
                    
    Returns:
        List of structured course recommendation dictionaries.
    """
    df_courses = courses_df if courses_df is not None and not courses_df.empty else load_course_catalog()
    
    if df_courses.empty:
        df_courses = create_fallback_courses_dataframe()
        
    stu_score = validate_and_clamp_score(student_skills.get(skill, 0.0)) if student_skills else 0.0
    safe_gap = validate_and_clamp_score(gap, default=0.0)
    
    w_skill = COURSE_MATCH_WEIGHTS.get("skill_match", 0.35)
    w_lvl = COURSE_MATCH_WEIGHTS.get("level_match", 0.25)
    w_gap = COURSE_MATCH_WEIGHTS.get("gap_relevance", 0.20)
    w_pre = COURSE_MATCH_WEIGHTS.get("prerequisite_match", 0.10)
    w_qual = COURSE_MATCH_WEIGHTS.get("course_quality", 0.10)
    
    candidates = []
    
    for _, row in df_courses.iterrows():
        c_skill = str(row.get("Skill", ""))
        c_topics = str(row.get("Topics", ""))
        c_level = str(row.get("Level", "Beginner"))
        c_rating = row.get("Rating", 4.7)
        c_prereqs = str(row.get("Prerequisites", "None"))
        
        # 1. Skill Match
        skill_score = compute_skill_match(c_skill, skill, c_topics)
        if skill_score == 0.0:
            continue  # Skip courses unrelated to this skill
            
        # 2. Level Match
        level_score = compute_level_match(stu_score, c_level)
        
        # 3. Gap Relevance
        gap_score = compute_gap_relevance(safe_gap)
        
        # 4. Prerequisite Match
        prereq_score, satisfied_p, missing_p = check_prerequisites(student_skills, c_prereqs)
        
        # 5. Course Quality
        quality_score = compute_course_quality(c_rating)
        
        # Composite Course Match Score
        match_score = (
            (w_skill * skill_score)
          + (w_lvl * level_score)
          + (w_gap * gap_score)
          + (w_pre * prereq_score)
          + (w_qual * quality_score)
        )
        match_score = float(np.clip(round(match_score, 1), 0.0, 100.0))
        
        # Parse topics list
        topics_list = [t.strip() for t in c_topics.split(";") if t.strip()] if c_topics else ["Practical Applications"]
        prereqs_list = [p.strip() for p in c_prereqs.split(";") if p.strip()] if c_prereqs and c_prereqs.lower() != "none" else []
        
        why_text = generate_why_recommended_text(
            course_name=str(row.get("Course_Name", "")),
            skill=skill,
            level=c_level,
            match_score=match_score,
            gap=safe_gap,
            missing_prereqs=missing_p
        )
        
        # Safe extraction of string fields with fallbacks
        c_id = str(row.get("Course_ID", f"CRS_{skill}")) if pd.notnull(row.get("Course_ID")) and str(row.get("Course_ID")).lower() != "none" else f"CRS_{skill}"
        c_name = str(row.get("Course_Name", "Specialized Course")) if pd.notnull(row.get("Course_Name")) and str(row.get("Course_Name")).lower() != "none" else f"{skill} Specialization"
        c_prov = str(row.get("Provider", "Online Platform")) if pd.notnull(row.get("Provider")) and str(row.get("Provider")).lower() != "none" else "Online Platform"
        c_url = str(row.get("URL", "https://www.coursera.org")) if pd.notnull(row.get("URL")) and str(row.get("URL")).lower() not in ["none", "nan", ""] else "https://www.coursera.org"
        c_cost = str(row.get("Cost_Type", "Free Audit")) if pd.notnull(row.get("Cost_Type")) and str(row.get("Cost_Type")).lower() not in ["none", "nan", ""] else "Free Audit"
        c_desc = str(row.get("Description", f"Comprehensive course in {skill}")) if pd.notnull(row.get("Description")) and str(row.get("Description")).lower() not in ["none", "nan", ""] else f"Comprehensive practical course covering {SKILL_DISPLAY_NAMES.get(skill, skill)}."
        c_dur = int(row.get("Duration_Hours", 30)) if pd.notnull(row.get("Duration_Hours")) and not np.isnan(float(row.get("Duration_Hours", 30))) else 30
        c_rat = float(c_rating) if pd.notnull(c_rating) and not np.isnan(float(c_rating)) else 4.7
        
        candidates.append({
            "course_id": c_id,
            "course_name": c_name,
            "provider": c_prov,
            "skill": skill,
            "display_name": SKILL_DISPLAY_NAMES.get(skill, skill),
            "level": c_level if c_level and c_level.lower() != "none" else "Beginner",
            "match_score": match_score,
            "duration": c_dur,
            "rating": c_rat,
            "prerequisites": prereqs_list,
            "missing_prerequisites": missing_p,
            "topics": topics_list,
            "url": c_url,
            "cost_type": c_cost,
            "description": c_desc,
            "why_recommended": why_text
        })
        
    candidates.sort(key=lambda x: x["match_score"], reverse=True)
    return candidates[:top_n]


def recommend_courses_for_profile(
    student_skills: Dict[str, Any],
    df_priorities: pd.DataFrame,
    top_skills_count: int = 3,
    courses_df: Optional[pd.DataFrame] = None
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Generates personalized course recommendations for the student's highest-priority skill gaps.
    
    Args:
        student_skills: Student proficiency dictionary
        df_priorities: Priority DataFrame from calculate_skill_priorities
        top_skills_count: Number of highest-priority skills to generate course recommendations for
        courses_df: Optional custom courses catalog DataFrame
        
    Returns:
        Dict mapping skill_name -> list of top-3 recommended course dicts
    """
    if df_priorities.empty:
        return {}
        
    # Focus on skills that have a deficit (Gap > 0), otherwise take top prioritized
    gap_priorities = df_priorities[df_priorities["Gap"] > 0]
    target_skills_df = gap_priorities if not gap_priorities.empty else df_priorities
    
    top_skills = target_skills_df.head(top_skills_count)
    results = {}
    
    for _, row in top_skills.iterrows():
        skill = row["Skill"]
        gap = row["Gap"]
        recs = recommend_courses_for_skill(
            skill=skill,
            student_skills=student_skills,
            gap=gap,
            top_n=3,
            courses_df=courses_df
        )
        results[skill] = recs
        
    return results


if __name__ == "__main__":
    from src.skill_priority import calculate_skill_priorities
    test_skills = {"Python": 45, "Statistics_Math": 35, "Machine_Learning": 50, "SQL": 60}
    df_priorities = calculate_skill_priorities(test_skills, "Data Scientist")
    
    print("\n--- Testing Course Recommendations for Top Skills ---")
    course_recs = recommend_courses_for_profile(test_skills, df_priorities, top_skills_count=2)
    for sk, courses in course_recs.items():
        print(f"\nSkill: {sk}")
        for c in courses:
            print(f"  - [{c['match_score']}% Match] {c['course_name']} ({c['provider']}) | Level: {c['level']} | Missing Prereqs: {c['missing_prerequisites']}")
