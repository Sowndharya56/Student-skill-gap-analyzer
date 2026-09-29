"""
Utility Module for Clean Plotly Visualizations & Report Generation.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from config.config import (
    CORE_SKILLS,
    SKILL_DISPLAY_NAMES,
    GAP_THRESHOLDS
)


def plot_student_vs_required_bars(df_gaps: pd.DataFrame) -> go.Figure:
    """
    Renders a clear, paired horizontal bar chart comparing Student Score vs Required Score.
    """
    if df_gaps.empty:
        return go.Figure()

    df_sorted = df_gaps.sort_values(by="Gap", ascending=True).copy()
    
    fig = go.Figure()
    
    # Required Benchmark Bars
    fig.add_trace(go.Bar(
        y=df_sorted["Display_Name"],
        x=df_sorted["Required_Score"],
        name="Required Benchmark",
        orientation="h",
        marker=dict(color="#CBD5E1", line=dict(color="#94A3B8", width=1)),
        text=[f"{val:.0f}" for val in df_sorted["Required_Score"]],
        textposition="auto",
        hoverinfo="text",
        hovertext=[f"<b>{name}</b><br>Required Level: {req:.0f}" for name, req in zip(df_sorted["Display_Name"], df_sorted["Required_Score"])]
    ))
    
    # Student Proficiency Bars
    fig.add_trace(go.Bar(
        y=df_sorted["Display_Name"],
        x=df_sorted["Student_Score"],
        name="Student Proficiency",
        orientation="h",
        marker=dict(color="#2563EB", line=dict(color="#1D4ED8", width=1)),
        text=[f"{val:.0f}" for val in df_sorted["Student_Score"]],
        textposition="auto",
        hoverinfo="text",
        hovertext=[f"<b>{name}</b><br>Student Level: {stu:.0f}<br>Gap: {gap:.0f} pts" for name, stu, gap in zip(df_sorted["Display_Name"], df_sorted["Student_Score"], df_sorted["Gap"])]
    ))
    
    fig.update_layout(
        title=dict(text="<b>Student Proficiency vs. Required Benchmark by Skill</b>", font=dict(size=15)),
        barmode="group",
        xaxis=dict(title="Proficiency Score (0 – 100)", range=[0, 105], showgrid=True, gridcolor="#E2E8F0"),
        yaxis=dict(tickfont=dict(size=11)),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        margin=dict(l=150, r=30, t=50, b=50),
        height=540
    )
    return fig


def plot_radar_comparison(
    student_skills: Dict[str, float],
    required_skills: Dict[str, float],
    role_name: str = "Target Role"
) -> go.Figure:
    """
    Renders an interactive polar radar chart comparing Student Skills vs Role Benchmark.
    """
    categories = [SKILL_DISPLAY_NAMES.get(s, s) for s in CORE_SKILLS]
    stu_vals = [student_skills.get(s, 0.0) for s in CORE_SKILLS]
    req_vals = [required_skills.get(s, 0.0) for s in CORE_SKILLS]
    
    categories_closed = categories + [categories[0]]
    stu_vals_closed = stu_vals + [stu_vals[0]]
    req_vals_closed = req_vals + [req_vals[0]]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=req_vals_closed,
        theta=categories_closed,
        fill="toself",
        name=f"Required Benchmark ({role_name})",
        line=dict(color="#DC2626", width=2, dash="dash"),
        fillcolor="rgba(220, 38, 38, 0.12)"
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=stu_vals_closed,
        theta=categories_closed,
        fill="toself",
        name="Student Proficiency",
        line=dict(color="#2563EB", width=2.5),
        fillcolor="rgba(37, 99, 235, 0.25)"
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(size=9),
                gridcolor="#E2E8F0"
            ),
            angularaxis=dict(
                tickfont=dict(size=10),
                rotation=90,
                direction="clockwise",
                gridcolor="#E2E8F0"
            )
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.22,
            xanchor="center",
            x=0.5,
            font=dict(size=11)
        ),
        margin=dict(l=60, r=60, t=30, b=50),
        height=500
    )
    return fig


def plot_gap_bar_chart(df_gaps: pd.DataFrame) -> go.Figure:
    """
    Renders a clean horizontal bar chart showing Skill Gaps sorted by severity.
    """
    if df_gaps.empty:
        return go.Figure()
        
    df_sorted = df_gaps.sort_values(by="Gap", ascending=True).copy()
    
    # Accessible clean colors: green for 0, blue for low, amber for medium, red for high
    clean_colors = []
    for gap in df_sorted["Gap"]:
        if gap == 0:
            clean_colors.append("#10B981") # Green
        elif gap <= 15:
            clean_colors.append("#3B82F6") # Blue
        elif gap <= 35:
            clean_colors.append("#F59E0B") # Amber
        else:
            clean_colors.append("#EF4444") # Red
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=df_sorted["Display_Name"],
        x=df_sorted["Gap"],
        orientation="h",
        marker=dict(color=clean_colors, line=dict(width=1, color="rgba(0,0,0,0.1)")),
        text=[f"{val:.0f} pts ({sev})" if val > 0 else "0 pts (Mastered)" for val, sev in zip(df_sorted["Gap"], df_sorted["Severity"])],
        textposition="auto",
        hoverinfo="text",
        hovertext=[
            f"<b>{name}</b><br>Student Score: {stu:.0f}<br>Required Score: {req:.0f}<br>Skill Gap: {gap:.0f} pts<br>Severity: {sev}"
            for name, stu, req, gap, sev in zip(
                df_sorted["Display_Name"], df_sorted["Student_Score"],
                df_sorted["Required_Score"], df_sorted["Gap"], df_sorted["Severity"]
            )
        ]
    ))
    
    fig.update_layout(
        title=dict(text="<b>Identified Skill Deficits (Gap Magnitude = Required – Student)</b>", font=dict(size=15)),
        xaxis=dict(title="Gap Points", range=[0, max(50, df_sorted['Gap'].max() + 10)], showgrid=True, gridcolor="#E2E8F0"),
        yaxis=dict(tickfont=dict(size=11)),
        margin=dict(l=150, r=30, t=50, b=40),
        height=520
    )
    return fig


def plot_career_match_ranking(recommendations: List[Dict[str, Any]]) -> go.Figure:
    """
    Horizontal bar chart showing ranked career alignment percentages.
    """
    if not recommendations:
        return go.Figure()
        
    df_rec = pd.DataFrame(recommendations).sort_values(by="hybrid_match_pct", ascending=True)
    
    colors = []
    for val in df_rec["hybrid_match_pct"]:
        if val >= 80:
            colors.append("#10B981") # Green
        elif val >= 60:
            colors.append("#2563EB") # Blue
        else:
            colors.append("#F59E0B") # Amber
            
    fig = go.Figure(go.Bar(
        x=df_rec["hybrid_match_pct"],
        y=df_rec["role"],
        orientation="h",
        marker=dict(color=colors, line=dict(color="rgba(0,0,0,0.1)", width=1)),
        text=[f"{val:.1f}% ({fit})" for val, fit in zip(df_rec["hybrid_match_pct"], df_rec["fit_level"])],
        textposition="auto",
        hoverinfo="text",
        hovertext=[
            f"<b>{role}</b><br>Overall Match Score: {m:.1f}%<br>ML Model Probability: {ml:.1f}%<br>Skill Cosine Similarity: {vec:.1f}%<br>Readiness Status: {fit}"
            for role, m, ml, vec, fit in zip(
                df_rec["role"], df_rec["hybrid_match_pct"],
                df_rec["ml_confidence_pct"], df_rec["vector_similarity_pct"], df_rec["fit_level"]
            )
        ]
    ))
    
    fig.update_layout(
        title=dict(text="<b>Career Compatibility Ranking (Hybrid Match Score)</b>", font=dict(size=15)),
        xaxis=dict(title="Compatibility Score (%)", range=[0, 105], showgrid=True, gridcolor="#E2E8F0"),
        yaxis=dict(tickfont=dict(size=12)),
        margin=dict(l=170, r=40, t=50, b=40),
        height=380
    )
    return fig


def plot_confusion_matrix_heatmap(df_cm: pd.DataFrame) -> go.Figure:
    """
    Renders normalized confusion matrix heatmap for final test evaluation.
    """
    fig = px.imshow(
        df_cm,
        text_auto=True,
        color_continuous_scale="Blues",
        labels=dict(x="Predicted Career Role", y="Actual True Career Role", color="Percentage (%)"),
        x=list(df_cm.columns),
        y=list(df_cm.index)
    )
    fig.update_layout(
        title=dict(text="<b>Normalized Confusion Matrix (Held-Out Test Set)</b>", font=dict(size=15)),
        xaxis=dict(tickangle=-35, tickfont=dict(size=10)),
        yaxis=dict(tickfont=dict(size=10)),
        margin=dict(l=140, r=40, t=60, b=100),
        height=520
    )
    return fig


def plot_feature_importances(df_feat_imp: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """
    Plots Permutation Feature Importance computed on held-out test split.
    """
    if df_feat_imp.empty:
        return go.Figure()
        
    df_top = df_feat_imp.head(top_n).sort_values(by="Mean_Importance", ascending=True)
    
    fig = go.Figure(go.Bar(
        x=df_top["Mean_Importance"],
        y=df_top["Feature"],
        orientation="h",
        marker=dict(
            color=df_top["Mean_Importance"],
            colorscale="Blues",
            line=dict(color="rgba(0,0,0,0.1)", width=1)
        ),
        text=[f"{val:.2f}%" for val in df_top["Mean_Importance"]],
        textposition="auto"
    ))
    fig.update_layout(
        title=dict(text=f"<b>Permutation Feature Importance (Top {top_n} Features on Test Split)</b>", font=dict(size=15)),
        xaxis=dict(title="Mean F1-Score Decrease When Feature Shuffled (%)", showgrid=True, gridcolor="#E2E8F0"),
        yaxis=dict(tickfont=dict(size=11)),
        margin=dict(l=160, r=40, t=50, b=40),
        height=450
    )
    return fig


def plot_skill_priority_breakdown(df_priorities: pd.DataFrame, top_n: int = 8) -> go.Figure:
    """
    Renders an interactive horizontal stacked bar chart showing the multi-factor priority score
    breakdown (Gap Severity 40%, Career Importance 30%, Dependency Impact 20%, Current Weakness 10%).
    """
    if df_priorities.empty:
        return go.Figure()
        
    df_top = df_priorities.head(top_n).sort_values(by="Priority_Score", ascending=True).copy()
    
    # Weighted component values
    gap_contrib = df_top["Gap_Severity"] * 0.40
    imp_contrib = df_top["Career_Importance"] * 0.30
    dep_contrib = df_top["Dependency_Impact"] * 0.20
    weak_contrib = df_top["Current_Weakness"] * 0.10
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=df_top["Display_Name"],
        x=gap_contrib,
        name="Gap Severity (40%)",
        orientation="h",
        marker=dict(color="#EF4444")
    ))
    
    fig.add_trace(go.Bar(
        y=df_top["Display_Name"],
        x=imp_contrib,
        name="Career Importance (30%)",
        orientation="h",
        marker=dict(color="#3B82F6")
    ))
    
    fig.add_trace(go.Bar(
        y=df_top["Display_Name"],
        x=dep_contrib,
        name="Dependency Impact (20%)",
        orientation="h",
        marker=dict(color="#8B5CF6")
    ))
    
    fig.add_trace(go.Bar(
        y=df_top["Display_Name"],
        x=weak_contrib,
        name="Current Weakness (10%)",
        orientation="h",
        marker=dict(color="#F59E0B")
    ))
    
    fig.update_layout(
        title=dict(text="<b>Multi-Factor Skill Improvement Priority Breakdown (Top Focus Skills)</b>", font=dict(size=15)),
        barmode="stack",
        xaxis=dict(title="Priority Score (0 – 100)", range=[0, 105], showgrid=True, gridcolor="#E2E8F0"),
        yaxis=dict(tickfont=dict(size=11)),
        legend=dict(orientation="h", yanchor="bottom", y=-0.22, xanchor="center", x=0.5, font=dict(size=10)),
        margin=dict(l=160, r=30, t=50, b=60),
        height=480
    )
    return fig


def generate_markdown_audit_report(
    student_name: str,
    degree: str,
    gpa: float,
    experience: int,
    projects: int,
    target_role: str,
    readiness_metrics: Dict[str, Any],
    recommendations: List[Dict[str, Any]],
    df_gaps: pd.DataFrame,
    roadmap: List[Dict[str, Any]],
    df_priorities: Optional[pd.DataFrame] = None,
    learning_sequence: Optional[List[Dict[str, Any]]] = None,
    course_recs: Optional[Dict[str, List[Dict[str, Any]]]] = None
) -> str:
    """
    Generates a full, professional Markdown Audit Report for download.
    """
    report = f"""# AI-Based Student Skill Gap Analysis & Employability Audit Report

**Generated on:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Student Name:** {student_name}  
**Academic Degree:** {degree} | **GPA:** {gpa:.2f}/10.0  
**Experience:** {experience} months | **Projects Completed:** {projects}  
**Evaluated Target Role:** {target_role}

---

## 1. Executive Summary & Readiness Score
* **Career Match Score:** `{readiness_metrics['career_match_percentage']}%`
* **Total Skill Gap Points:** `{readiness_metrics['total_gap_points']} pts`
* **Average Gap across Skills:** `{readiness_metrics['average_gap_points']} pts`
* **High Priority Skill Gaps:** `{readiness_metrics['high_priority_count']} skills`
* **Mastered / Sufficient Skills:** `{readiness_metrics['mastered_count']} skills`

---

## 2. Top Recommended Career Alignments
"""
    for i, r in enumerate(recommendations, 1):
        strengths = ", ".join(r["supporting_strengths"][:3]) if r["supporting_strengths"] else "Foundational technical skills"
        report += f"""
### {i}. {r['role']} — {r['hybrid_match_pct']}% Match ({r['fit_level']})
* **ML Model Confidence:** {r['ml_confidence_pct']}% | **Vector Cosine Similarity:** {r['vector_similarity_pct']}%
* **Key Supporting Strengths:** {strengths}
* **Analysis & Rationale:** {r['explanation']}
"""

    if df_priorities is not None and not df_priorities.empty:
        report += """
---

## 3. Intelligent Skill Improvement Priority Ranking

Priority Formula: `40% Gap Severity + 30% Career Importance + 20% Dependency Impact + 10% Current Weakness`

| Priority | Skill | Score (0-100) | Category | Current | Benchmark | Gap | Why Improve First? |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
"""
        for idx, row in df_priorities.head(8).iterrows():
            report += f"| **#{idx+1}** | **{row['Display_Name']}** | `{row['Priority_Score']}` | `{row['Priority_Category']}` | {row['Student_Score']} | {row['Required_Score']} | **{row['Gap']}** | {row['Reason']} |\n"

    if learning_sequence:
        report += """
---

## 4. Recommended Prerequisite-Aware Learning Sequence

Topological ordering ensuring foundation prerequisites are strengthened before advanced concepts:

"""
        for seq_idx, item in enumerate(learning_sequence[:8], 1):
            disp = item.get("Display_Name", item.get("Skill", ""))
            p_score = item.get("Priority_Score", 0)
            gap = item.get("Gap", 0)
            report += f"**Step {seq_idx}: {disp}** (Priority Score: `{p_score}` | Gap: `{gap}` pts)\n"

    if course_recs:
        report += """
---

## 5. Personalized Online Course Recommendations

Tailored recommendations matched by skill alignment, difficulty level, and prerequisite readiness:

"""
        for skill_name, courses in course_recs.items():
            disp_sk = SKILL_DISPLAY_NAMES.get(skill_name, skill_name)
            report += f"### 📚 Courses for {disp_sk}\n"
            for c_idx, c in enumerate(courses[:3], 1):
                missing_txt = f"⚠️ *Missing Prereqs: {', '.join(c['missing_prerequisites'])}*" if c['missing_prerequisites'] else "✅ *All Prerequisites Met*"
                report += f"""{c_idx}. **[{c['course_name']}]({c['url']})** ({c['provider']})
   * **Match Score:** `{c['match_score']}%` | **Level:** `{c['level']}` | **Duration:** `{c['duration']} hrs` | **Rating:** `⭐ {c['rating']}/5.0`
   * **Prerequisites Status:** {missing_txt}
   * **Why Recommended:** {c['why_recommended']}
"""

    report += """
---

## 6. Comprehensive Skill Gap Breakdown Table

| Skill Category | Current Score | Required Benchmark | Skill Gap (pts) | Severity Level |
| :--- | :---: | :---: | :---: | :---: |
"""
    for _, row in df_gaps.iterrows():
        report += f"| {row['Display_Name']} | {row['Student_Score']} | {row['Required_Score']} | **{row['Gap']}** | `{row['Severity']}` |\n"

    report += """
---

## 7. Actionable Learning Roadmap

"""
    for item in roadmap:
        gap = item.get("gap", 0)
        disp = item.get("display_name", "")
        p_lvl = item.get("priority_level", item.get("priority_label", "Medium"))
        est_h = item.get("estimated_hours", 20)
        topics = item.get("topics", item.get("phase_4_practice_topics", []))
        project = item.get("suggested_project", item.get("phase_5_portfolio_project", ""))
        
        if gap > 0:
            report += f"""
### 🔹 {disp} — Priority: `{p_lvl}` (Gap: {gap} pts | Est. Study: {est_h} hrs)
* **Key Topics to Learn:** {', '.join(topics) if isinstance(topics, list) else topics}
* **Suggested Portfolio Project:** *{project}*
"""

    report += """
---
*Report generated by Student Skill Gap Analyzer & Career Recommendation System.*  
*Methodology Note: Hybrid recommendation scores use configured heuristic weighting (40% ML Probability, 35% Vector Similarity, 25% Gap Readiness).*
"""
    return report
