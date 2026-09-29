"""
Career Recommendation Engine.
Methodology:
Combines:
1. Machine Learning Prediction Probabilities (Calibrated Multi-Class Classifier Pipeline)
2. Cosine Vector Skill Profile Similarity against Industry Benchmarks
3. Role Readiness Match based on evaluated skill deficits

Formula:
    Hybrid Match = (w_ml * ML_Prob) + (w_vec * Cosine_Sim) + (w_readiness * Readiness_Match)

NOTE: The hybrid weights (0.40 ML, 0.35 Vector Similarity, 0.25 Readiness) are heuristic design choices
selected for the multi-criteria decision architecture. They are not statistically learned model parameters.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import logging
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

from config.config import (
    CORE_SKILLS,
    TARGET_CAREERS,
    SKILL_DISPLAY_NAMES,
    HYBRID_WEIGHTS
)
from src.data_loader import load_processed_data
from src.feature_engineering import extract_features_for_student_profile
from src.preprocessing import build_student_feature_vector
from src.model_training import load_trained_model
from src.skill_gap import calculate_skill_gaps, compute_role_readiness_metrics


def compute_vector_match_scores(
    student_skills: Dict[str, float],
    df_requirements: pd.DataFrame
) -> Dict[str, float]:
    """Computes Cosine Similarity between student skill vector and industry benchmark vectors."""
    student_vec = np.array([float(student_skills.get(s, 0.0)) for s in CORE_SKILLS]).reshape(1, -1)
    norm = np.linalg.norm(student_vec)
    
    match_scores = {}
    for _, row in df_requirements.iterrows():
        role = row["Job_Role"]
        role_vec = np.array([float(row[s]) for s in CORE_SKILLS]).reshape(1, -1)
        role_norm = np.linalg.norm(role_vec)
        
        if norm == 0 or role_norm == 0:
            match_scores[role] = 0.0
        else:
            sim = cosine_similarity(student_vec, role_vec)[0][0]
            match_scores[role] = round(float(sim * 100.0), 1)
            
    return match_scores


def recommend_careers(
    student_skills: Dict[str, Any],
    gpa: float = 8.0,
    projects: int = 4,
    experience: int = 6,
    top_n: int = 5
) -> List[Dict[str, Any]]:
    """
    Produces ranked career recommendations with safe edge-case handling for missing
    models, unknown careers, empty input, or all-zero proficiencies.
    """
    # Safe fallback if empty or all zero
    safe_skills = {s: float(student_skills.get(s, 0.0)) for s in CORE_SKILLS} if student_skills else {s: 0.0 for s in CORE_SKILLS}
    
    # 1. Load benchmark matrix
    try:
        _, df_matrix = load_processed_data()
    except Exception as e:
        logger.error("Failed to load processed role benchmark matrix: %s", e)
        return []
        
    # 2. Compute Cosine Vector similarities
    vector_scores = compute_vector_match_scores(safe_skills, df_matrix)
    
    # 3. Compute Machine Learning Model probabilities
    ml_probs = {}
    try:
        pipeline, le = load_trained_model()
        if pipeline is not None and le is not None:
            features_dict = extract_features_for_student_profile(safe_skills, gpa=gpa, projects=projects, experience=experience)
            X_feature_vec = build_student_feature_vector(features_dict)
            if hasattr(pipeline, "predict_proba"):
                probs = pipeline.predict_proba(X_feature_vec)[0]
                for idx, class_label in enumerate(le.classes_):
                    ml_probs[class_label] = probs[idx] * 100.0
            else:
                pred_cls = pipeline.predict(X_feature_vec)[0]
                pred_label = le.inverse_transform([pred_cls])[0]
                for role in TARGET_CAREERS:
                    ml_probs[role] = 100.0 if role == pred_label else 0.0
        else:
            for role in TARGET_CAREERS:
                ml_probs[role] = vector_scores.get(role, 0.0)
    except Exception as e:
        logger.warning("ML pipeline probability prediction unavailable, using vector fallback: %s", e)
        for role in TARGET_CAREERS:
            ml_probs[role] = vector_scores.get(role, 0.0)
            
    # 4. Synthesize Hybrid Recommendation Score using configured heuristic weights
    w_ml = HYBRID_WEIGHTS.get("ml_probability", 0.40)
    w_vec = HYBRID_WEIGHTS.get("vector_similarity", 0.35)
    w_read = HYBRID_WEIGHTS.get("readiness_match", 0.25)
    
    recommendations = []
    
    for _, row in df_matrix.iterrows():
        role = row["Job_Role"]
        role_reqs = {s: float(row[s]) for s in CORE_SKILLS}
        
        vec_sim = vector_scores.get(role, 0.0)
        ml_prob = ml_probs.get(role, 0.0)
        
        df_gaps = calculate_skill_gaps(safe_skills, role_reqs)
        readiness = compute_role_readiness_metrics(df_gaps)
        
        hybrid_score = (w_ml * ml_prob) + (w_vec * vec_sim) + (w_read * readiness["career_match_percentage"])
        hybrid_score = round(float(np.clip(hybrid_score, 0.0, 100.0)), 1)
        
        strengths = []
        for _, g_row in df_gaps.iterrows():
            if g_row["Student_Score"] >= (g_row["Required_Score"] - 5) and g_row["Required_Score"] >= 60 and g_row["Student_Score"] > 0:
                strengths.append(g_row["Display_Name"])
        
        top_missing = []
        for _, g_row in df_gaps.head(3).iterrows():
            if g_row["Gap"] > 0:
                top_missing.append(f"{g_row['Display_Name']} (Gap: {int(g_row['Gap'])})")
                
        if hybrid_score >= 80:
            fit_level = "Strong Alignment"
            explanation = f"Profile exhibits solid competencies in {', '.join(strengths[:2]) if strengths else 'core technical areas'}, aligning well with industry requirements for {role}."
        elif hybrid_score >= 60:
            fit_level = "Moderate Alignment"
            explanation = f"Profile shows moderate alignment with {role}. Targeted practice in {', '.join([m.split(' (')[0] for m in top_missing[:2]]) if top_missing else 'specialized skills'} will bridge remaining gaps."
        else:
            fit_level = "Developing Match"
            explanation = f"Requires foundational upskilling in {', '.join([m.split(' (')[0] for m in top_missing[:2]]) if top_missing else 'key domain skills'} to reach industry benchmark readiness for {role}."
            
        recommendations.append({
            "role": role,
            "hybrid_match_pct": hybrid_score,
            "ml_confidence_pct": round(ml_prob, 1),
            "vector_similarity_pct": round(vec_sim, 1),
            "readiness_match_pct": readiness["career_match_percentage"],
            "fit_level": fit_level,
            "supporting_strengths": strengths[:4],
            "critical_gaps": top_missing,
            "explanation": explanation,
            "df_gaps": df_gaps,
            "readiness_metrics": readiness
        })
        
    recommendations.sort(key=lambda x: x["hybrid_match_pct"], reverse=True)
    return recommendations[:top_n]


if __name__ == "__main__":
    test_skills = {"Python": 90, "SQL": 85, "Data_Analysis": 90, "Machine_Learning": 85}
    recs = recommend_careers(test_skills, top_n=3)
    for r in recs:
        print(f"{r['role']}: {r['hybrid_match_pct']}% (ML: {r['ml_confidence_pct']}%, Vector: {r['vector_similarity_pct']}%)")
