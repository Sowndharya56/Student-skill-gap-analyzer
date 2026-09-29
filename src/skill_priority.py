"""
Intelligent Skill Improvement Priority Engine.

Answers:
"Which skill should this student improve FIRST for their target career?"

Formulation:
    Priority Score =
        (0.40 * Gap_Severity)
      + (0.30 * Career_Importance)
      + (0.20 * Dependency_Impact)
      + (0.10 * Current_Weakness)

Categories:
    80 – 100 -> 🔴 Critical
    60 – 79  -> 🟠 High
    40 – 59  -> 🟡 Medium
     0 – 39  -> 🟢 Low

Also generates a topological, prerequisite-aware recommended learning sequence.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from collections import defaultdict, deque
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

from config.config import (
    CORE_SKILLS,
    SKILL_DISPLAY_NAMES,
    PRIORITY_WEIGHTS,
    PRIORITY_CATEGORIES,
    SKILL_DEPENDENCY_GRAPH
)
from src.skill_gap import calculate_skill_gaps, validate_and_clamp_score
from src.data_loader import load_processed_data


def compute_gap_severity(gap: float) -> float:
    """
    Normalizes skill gap to a 0–100 scale.
    Formula: Clamped to [0.0, 100.0].
    """
    return float(np.clip(gap, 0.0, 100.0))


def compute_career_importance(required_score: float) -> float:
    """
    Normalizes career benchmark requirement to 0–100 scale.
    Higher benchmark requirement = greater career importance.
    """
    return float(np.clip(required_score, 0.0, 100.0))


def compute_current_weakness(student_score: float) -> float:
    """
    Calculates current weakness on a 0–100 scale.
    Lower skill score = higher weakness.
    Formula: 100.0 - student_score (clamped to [0.0, 100.0]).
    """
    clamped_score = float(np.clip(student_score, 0.0, 100.0))
    return float(np.clip(100.0 - clamped_score, 0.0, 100.0))


def get_transitive_downstream_skills(
    skill: str,
    dependency_graph: Optional[Dict[str, List[str]]] = None
) -> Set[str]:
    """
    Finds all direct and indirect downstream skills that depend on the given skill.
    """
    graph = dependency_graph or SKILL_DEPENDENCY_GRAPH
    visited: Set[str] = set()
    queue = deque(graph.get(skill, []))
    
    while queue:
        current = queue.popleft()
        if current not in visited:
            visited.add(current)
            for downstream in graph.get(current, []):
                if downstream not in visited:
                    queue.append(downstream)
                    
    return visited


def compute_dependency_impact(
    skill: str,
    student_score: float,
    required_score: float,
    all_required_skills: Dict[str, float],
    dependency_graph: Optional[Dict[str, List[str]]] = None
) -> float:
    """
    Calculates how severely a weakness in this skill blocks critical downstream skills.
    
    Factors:
    - Direct and transitive downstream skills in the DAG
    - Target career importance of those downstream skills
    - Student's current deficit/weakness in this foundational skill
    
    If the skill has no downstream dependents or the student has already mastered
    the skill (score >= required), dependency impact is 0.
    
    Returns:
        Score normalized between 0.0 and 100.0.
    """
    graph = dependency_graph or SKILL_DEPENDENCY_GRAPH
    downstream_skills = get_transitive_downstream_skills(skill, graph)
    
    if not downstream_skills:
        return 0.0
        
    # If student has mastered this skill (score meets or exceeds requirement and is >= 75)
    gap = max(0.0, required_score - student_score)
    if gap <= 0.0 and student_score >= 75.0:
        return 0.0
        
    # Calculate downstream importance based on career requirements
    downstream_reqs = [
        all_required_skills.get(d, 50.0)
        for d in downstream_skills
        if d in CORE_SKILLS
    ]
    
    if not downstream_reqs:
        return 0.0
        
    avg_downstream_importance = float(np.mean(downstream_reqs))
    
    # Weakness / blocking degree of current skill (0.0 to 1.0)
    # If student has a deficit, blocking degree is proportional to gap and low score
    weakness_factor = (100.0 - float(np.clip(student_score, 0.0, 100.0))) / 100.0
    
    # Scale with breadth of downstream impact (more dependents = higher urgency)
    count_multiplier = min(1.2, 0.7 + (0.1 * len(downstream_skills)))
    
    impact = weakness_factor * avg_downstream_importance * count_multiplier
    return float(np.clip(round(impact, 1), 0.0, 100.0))


def get_priority_category(priority_score: float) -> Tuple[str, str, str]:
    """
    Maps priority score to priority category, emoji badge, and hex color code.
    
    Returns:
        (Category Name, Badge String, Color Hex)
    """
    score = float(np.clip(priority_score, 0.0, 100.0))
    if score >= PRIORITY_CATEGORIES["CRITICAL"][0]:
        cat = PRIORITY_CATEGORIES["CRITICAL"]
    elif score >= PRIORITY_CATEGORIES["HIGH"][0]:
        cat = PRIORITY_CATEGORIES["HIGH"]
    elif score >= PRIORITY_CATEGORIES["MEDIUM"][0]:
        cat = PRIORITY_CATEGORIES["MEDIUM"]
    else:
        cat = PRIORITY_CATEGORIES["LOW"]
        
    return cat[2], f"{cat[3]} {cat[2]}", cat[4]


def generate_priority_reason(
    skill: str,
    gap: float,
    required_score: float,
    student_score: float,
    dependency_impact: float,
    target_career: str,
    dependency_graph: Optional[Dict[str, List[str]]] = None
) -> str:
    """
    Generates a personalized, human-readable reason for why this skill received its priority.
    Uses exact student numbers and career benchmark context.
    """
    display_name = SKILL_DISPLAY_NAMES.get(skill, skill)
    graph = dependency_graph or SKILL_DEPENDENCY_GRAPH
    direct_dependents = graph.get(skill, [])
    dep_names = [SKILL_DISPLAY_NAMES.get(d, d) for d in direct_dependents[:2]]
    
    reasons = []
    
    # 1. Gap Severity context
    if gap >= 35.0:
        reasons.append(f"a substantial skill gap of {gap:.0f} points (Current: {student_score:.0f} vs Benchmark: {required_score:.0f})")
    elif gap > 15.0:
        reasons.append(f"a moderate gap of {gap:.0f} points to bridge")
    elif gap > 0.0:
        reasons.append(f"a minor deficit of {gap:.0f} points")
    else:
        reasons.append(f"proficient standing meeting the benchmark of {required_score:.0f}")

    # 2. Career Importance context
    if required_score >= 85.0:
        reasons.append(f"is mission-critical for {target_career} (Benchmark: {required_score:.0f}/100)")
    elif required_score >= 70.0:
        reasons.append(f"is an important standard competency for {target_career}")
        
    # 3. Prerequisite / Dependency context
    if dependency_impact >= 40.0 and dep_names:
        deps_str = " and ".join(dep_names)
        reasons.append(f"serves as a foundational prerequisite for {deps_str}")
        
    if gap <= 0.0:
        return f"Low priority: You currently meet or exceed the required benchmark ({required_score:.0f}/100) for {display_name}."
        
    if len(reasons) >= 3:
        return f"High priority because this skill has {reasons[0]}, {reasons[1]}, and {reasons[2]}."
    elif len(reasons) == 2:
        return f"Prioritized because this skill has {reasons[0]} and {reasons[1]}."
    else:
        return f"Targeted for improvement with {reasons[0]} for {target_career}."


def calculate_skill_priorities(
    student_skills: Dict[str, Any],
    target_career: str,
    df_requirements: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Computes multi-factor priority scores for all 20 skills for the given target career.
    
    Formula:
        Priority Score =
            (0.40 * Gap_Severity)
          + (0.30 * Career_Importance)
          + (0.20 * Dependency_Impact)
          + (0.10 * Current_Weakness)
          
    Args:
        student_skills: Dict of {skill_name: student_score (0-100)}
        target_career: Name of the target role (e.g. "Data Scientist")
        df_requirements: Optional DataFrame containing role-skill benchmark requirements
        
    Returns:
        pd.DataFrame sorted by Priority_Score descending containing:
            ['Skill', 'Display_Name', 'Student_Score', 'Required_Score', 'Gap',
             'Gap_Severity', 'Career_Importance', 'Dependency_Impact', 'Current_Weakness',
             'Priority_Score', 'Priority_Category', 'Priority_Badge', 'Priority_Color', 'Reason']
    """
    # Safe fallback if empty or None
    safe_skills = {s: validate_and_clamp_score(student_skills.get(s, 0.0)) for s in CORE_SKILLS} if student_skills else {s: 0.0 for s in CORE_SKILLS}
    
    # Retrieve required benchmark for the target career
    required_skills_dict = {}
    if df_requirements is None or df_requirements.empty:
        try:
            _, df_matrix = load_processed_data()
            if not df_matrix.empty and target_career in df_matrix["Job_Role"].values:
                role_row = df_matrix[df_matrix["Job_Role"] == target_career].iloc[0]
                required_skills_dict = {s: float(role_row[s]) for s in CORE_SKILLS if s in role_row}
        except Exception as e:
            logger.warning("Could not load role benchmark matrix: %s", e)
            
    if not required_skills_dict and df_requirements is not None and not df_requirements.empty:
        if target_career in df_requirements["Job_Role"].values:
            role_row = df_requirements[df_requirements["Job_Role"] == target_career].iloc[0]
            required_skills_dict = {s: float(role_row[s]) for s in CORE_SKILLS if s in role_row}
            
    # Default fallback benchmark if career not found
    if not required_skills_dict:
        required_skills_dict = {s: 70.0 for s in CORE_SKILLS}

    w_gap = PRIORITY_WEIGHTS.get("gap_severity", 0.40)
    w_imp = PRIORITY_WEIGHTS.get("career_importance", 0.30)
    w_dep = PRIORITY_WEIGHTS.get("dependency_impact", 0.20)
    w_weak = PRIORITY_WEIGHTS.get("current_weakness", 0.10)
    
    records = []
    for skill in CORE_SKILLS:
        stu_score = safe_skills.get(skill, 0.0)
        req_score = validate_and_clamp_score(required_skills_dict.get(skill, 50.0), default=50.0)
        
        # 1. Gap Severity
        gap = max(0.0, req_score - stu_score)
        gap_sev = compute_gap_severity(gap)
        
        # 2. Career Importance
        career_imp = compute_career_importance(req_score)
        
        # 3. Dependency Impact
        dep_impact = compute_dependency_impact(
            skill=skill,
            student_score=stu_score,
            required_score=req_score,
            all_required_skills=required_skills_dict
        )
        
        # 4. Current Weakness
        weakness = compute_current_weakness(stu_score)
        
        # Multi-factor priority score calculation
        if gap == 0.0:
            # If skill is already mastered/sufficient, priority score is scaled down
            # but preserves minor maintenance value if benchmark is critical
            raw_priority = (0.15 * career_imp) + (0.05 * weakness)
        else:
            raw_priority = (w_gap * gap_sev) + (w_imp * career_imp) + (w_dep * dep_impact) + (w_weak * weakness)
            
        priority_score = float(np.clip(round(raw_priority, 1), 0.0, 100.0))
        category, badge, color = get_priority_category(priority_score)
        
        reason = generate_priority_reason(
            skill=skill,
            gap=gap,
            required_score=req_score,
            student_score=stu_score,
            dependency_impact=dep_impact,
            target_career=target_career
        )
        
        records.append({
            "Skill": skill,
            "Display_Name": SKILL_DISPLAY_NAMES.get(skill, skill),
            "Student_Score": round(stu_score, 1),
            "Required_Score": round(req_score, 1),
            "Gap": round(gap, 1),
            "Gap_Severity": round(gap_sev, 1),
            "Career_Importance": round(career_imp, 1),
            "Dependency_Impact": round(dep_impact, 1),
            "Current_Weakness": round(weakness, 1),
            "Priority_Score": priority_score,
            "Priority_Category": category,
            "Priority_Badge": badge,
            "Priority_Color": color,
            "Reason": reason
        })
        
    df_priorities = pd.DataFrame(records)
    df_priorities.sort_values(by="Priority_Score", ascending=False, inplace=True)
    df_priorities.reset_index(drop=True, inplace=True)
    return df_priorities


def generate_learning_sequence(
    df_priorities: pd.DataFrame,
    dependency_graph: Optional[Dict[str, List[str]]] = None
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], bool, str]:
    """
    Generates a prerequisite-aware learning sequence from priority rankings.
    
    Algorithm:
    - Identifies skills that have positive gaps.
    - Uses a topological sort prioritizing foundational prerequisite skills
      when downstream skills require them.
    - If Skill A is a prerequisite for Skill B and Skill A has a deficit,
      Skill A is sequenced BEFORE Skill B.
    - Skills with equal readiness/dependency tier are ordered by Priority_Score descending.
    
    Returns:
        Tuple of:
        - raw_priority_ranking: List of skill dicts sorted purely by Priority Score
        - learning_sequence: List of skill dicts ordered by prerequisite-aware sequence
        - has_difference: Boolean indicating if learning sequence differs from raw priority order
        - difference_explanation: Human-readable explanation of why prerequisite reordering occurred
    """
    if df_priorities.empty:
        return [], [], False, "No skills to sequence."
        
    graph = dependency_graph or SKILL_DEPENDENCY_GRAPH
    
    # Filter to deficit skills (Gap > 0)
    gap_skills_df = df_priorities[df_priorities["Gap"] > 0].copy()
    if gap_skills_df.empty:
        # All mastered, fallback to top priority skills
        gap_skills_df = df_priorities.head(5).copy()
        
    raw_ranking = gap_skills_df.to_dict(orient="records")
    gap_skill_set = set(gap_skills_df["Skill"].values)
    
    # Build in-degree map restricted to skills needing improvement
    # Edge A -> B means A is prerequisite for B (A must be learned before B)
    in_degree = {s: 0 for s in gap_skill_set}
    dependents = defaultdict(list)
    
    for u in gap_skill_set:
        for v in graph.get(u, []):
            if v in gap_skill_set:
                dependents[u].append(v)
                in_degree[v] += 1
                
    # Priority-aware Kahn's topological sort
    # In each step, pick available node with highest Priority_Score
    available = [s for s in gap_skill_set if in_degree[s] == 0]
    
    # Helper to get priority score for a skill
    score_map = {row["Skill"]: row["Priority_Score"] for row in raw_ranking}
    available.sort(key=lambda s: score_map.get(s, 0.0), reverse=True)
    
    ordered_skills = []
    while available:
        # Pop skill with highest priority among currently available (zero in-degree)
        curr = available.pop(0)
        ordered_skills.append(curr)
        
        for neighbor in dependents[curr]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                available.append(neighbor)
                available.sort(key=lambda s: score_map.get(s, 0.0), reverse=True)
                
    # Any remaining nodes (in case of cycles, though DAG is acyclic)
    for s in gap_skill_set:
        if s not in ordered_skills:
            ordered_skills.append(s)
            
    # Build ordered list of full skill records
    record_map = {row["Skill"]: row for row in raw_ranking}
    learning_sequence = [record_map[s] for s in ordered_skills if s in record_map]
    
    # Detect if order differs from raw priority ranking
    raw_order_names = [r["Skill"] for r in raw_ranking]
    seq_order_names = [r["Skill"] for r in learning_sequence]
    has_difference = (raw_order_names != seq_order_names)
    
    difference_explanation = ""
    if has_difference:
        # Find the specific prerequisite pairs that caused reordering
        reordered_pairs = []
        for i, s1 in enumerate(seq_order_names):
            for s2 in seq_order_names[i+1:]:
                # If s2 was ranked higher in raw priority, but s1 comes before s2
                raw_idx_1 = raw_order_names.index(s1)
                raw_idx_2 = raw_order_names.index(s2)
                if raw_idx_1 > raw_idx_2:
                    s1_disp = SKILL_DISPLAY_NAMES.get(s1, s1)
                    s2_disp = SKILL_DISPLAY_NAMES.get(s2, s2)
                    reordered_pairs.append(
                        f"**{s1_disp}** is sequenced before **{s2_disp}** because it is a critical prerequisite, "
                        f"even though {s2_disp} had a higher standalone priority score."
                    )
        if reordered_pairs:
            difference_explanation = " ".join(reordered_pairs[:2])
        else:
            difference_explanation = "The learning sequence adjusts for prerequisite dependencies to ensure foundational skills are built before advanced concepts."
    else:
        difference_explanation = "The learning sequence matches the raw priority ranking, as all prerequisite skills are already ordered foundationally."
        
    return raw_ranking, learning_sequence, has_difference, difference_explanation


if __name__ == "__main__":
    test_stu = {"Python": 45, "Statistics_Math": 35, "Machine_Learning": 50, "SQL": 60}
    df_p = calculate_skill_priorities(test_stu, target_career="Data Scientist")
    print("\n--- Priority Ranking ---")
    print(df_p[["Skill", "Student_Score", "Required_Score", "Gap", "Priority_Score", "Priority_Category"]].head(6).to_string(index=False))
    
    raw_r, l_seq, diff, expl = generate_learning_sequence(df_p)
    print("\n--- Learning Sequence ---")
    for idx, s in enumerate(l_seq[:5], 1):
        print(f"{idx}. {s['Display_Name']} (Priority Score: {s['Priority_Score']})")
    print(f"\nSequence Difference: {diff}")
    print(f"Explanation: {expl}")
