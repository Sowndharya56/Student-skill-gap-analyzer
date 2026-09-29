"""
PDF Report Generation Module for Student Skill Gap Analyzer & Career Recommendation System.
Generates an official, publication-quality PDF Employability Audit Report using ReportLab Platypus.
"""

import sys
from pathlib import Path
import io
import datetime
from typing import Dict, List, Any, Optional, Tuple, Union

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np

import reportlab
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    KeepTogether,
    HRFlowable
)
from reportlab.pdfgen import canvas

from config.config import (
    CORE_SKILLS,
    SKILL_DISPLAY_NAMES,
    GAP_THRESHOLDS,
    PRIORITY_CATEGORIES
)


class NumberedCanvas(canvas.Canvas):
    """
    Two-pass canvas that adds running headers, footers, and accurate dynamic page counts:
    'Page X of Y' across all pages.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count: int):
        self.saveState()
        # Page dimensions for A4: 595.27 x 841.89 points
        # Margins: 36pt (0.5 inch) left/right, usable width = 523.27pt
        
        # Header (Top)
        self.setFont("Helvetica-Bold", 8)
        self.setFillColor(colors.HexColor("#475569"))
        self.drawString(36, 810, "STUDENT SKILL GAP ANALYZER — EMPLOYABILITY AUDIT REPORT")
        self.setFont("Helvetica", 8)
        self.drawRightString(559, 810, "OFFICIAL REPORT")
        
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.75)
        self.line(36, 802, 559, 802)
        
        # Footer (Bottom)
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.75)
        self.line(36, 42, 559, 42)
        
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))
        self.drawString(36, 28, "AI-Based Career Recommendation & Skill Gap Analysis Engine")
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(559, 28, page_str)
        
        self.restoreState()


def get_pdf_styles():
    """Constructs a clean, professional typography hierarchy for the audit report."""
    base_styles = getSampleStyleSheet()
    
    styles = {
        "DocTitle": ParagraphStyle(
            "DocTitle",
            parent=base_styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=22,
            textColor=colors.HexColor("#0F172A"),
            spaceAfter=3
        ),
        "DocSubtitle": ParagraphStyle(
            "DocSubtitle",
            parent=base_styles["Normal"],
            fontName="Helvetica",
            fontSize=10,
            leading=13,
            textColor=colors.HexColor("#64748B"),
            spaceAfter=12
        ),
        "SectionHeading": ParagraphStyle(
            "SectionHeading",
            parent=base_styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=16,
            textColor=colors.HexColor("#1E3A8A"),
            spaceBefore=12,
            spaceAfter=6,
            keepWithNext=True
        ),
        "SubSectionHeading": ParagraphStyle(
            "SubSectionHeading",
            parent=base_styles["Heading3"],
            fontName="Helvetica-Bold",
            fontSize=10.5,
            leading=14,
            textColor=colors.HexColor("#1E293B"),
            spaceBefore=8,
            spaceAfter=4,
            keepWithNext=True
        ),
        "Body": ParagraphStyle(
            "Body",
            parent=base_styles["BodyText"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=11.5,
            textColor=colors.HexColor("#334155"),
            spaceAfter=4
        ),
        "BodyBold": ParagraphStyle(
            "BodyBold",
            parent=base_styles["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=8.5,
            leading=11.5,
            textColor=colors.HexColor("#1E293B"),
            spaceAfter=4
        ),
        "BodySmall": ParagraphStyle(
            "BodySmall",
            parent=base_styles["Normal"],
            fontName="Helvetica",
            fontSize=7.5,
            leading=9.5,
            textColor=colors.HexColor("#475569")
        ),
        "TableHead": ParagraphStyle(
            "TableHead",
            fontName="Helvetica-Bold",
            fontSize=8,
            leading=10,
            textColor=colors.white,
            alignment=0
        ),
        "TableCell": ParagraphStyle(
            "TableCell",
            fontName="Helvetica",
            fontSize=7.5,
            leading=10,
            textColor=colors.HexColor("#1E293B")
        ),
        "TableCellBold": ParagraphStyle(
            "TableCellBold",
            fontName="Helvetica-Bold",
            fontSize=7.5,
            leading=10,
            textColor=colors.HexColor("#0F172A")
        ),
        "TableCellSmall": ParagraphStyle(
            "TableCellSmall",
            fontName="Helvetica",
            fontSize=7,
            leading=9,
            textColor=colors.HexColor("#475569")
        ),
        "BadgeText": ParagraphStyle(
            "BadgeText",
            fontName="Helvetica-Bold",
            fontSize=7.5,
            leading=9.5,
            alignment=1
        ),
        "CardText": ParagraphStyle(
            "CardText",
            fontName="Helvetica",
            fontSize=8,
            leading=11,
            textColor=colors.HexColor("#334155")
        ),
        "CalloutText": ParagraphStyle(
            "CalloutText",
            fontName="Helvetica-Oblique",
            fontSize=8,
            leading=11,
            textColor=colors.HexColor("#475569")
        )
    }
    return styles


import html


def _esc(val: Any) -> str:
    """Escapes XML/HTML special characters for ReportLab Paragraph compatibility."""
    if val is None:
        return ""
    return html.escape(str(val))


def generate_pdf_report(
    student_name: Union[str, Dict[str, Any]] = "Student",
    degree: str = "B.Tech Computer Science",
    gpa: float = 8.5,
    experience: int = 6,
    projects: int = 4,
    target_role: str = "Data Scientist",
    readiness_metrics: Optional[Dict[str, Any]] = None,
    recommendations: Optional[List[Dict[str, Any]]] = None,
    df_gaps: Optional[pd.DataFrame] = None,
    roadmap: Optional[List[Dict[str, Any]]] = None,
    df_priorities: Optional[pd.DataFrame] = None,
    learning_sequence: Optional[List[Dict[str, Any]]] = None,
    course_recs: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    **kwargs
) -> bytes:
    """
    Generates a publication-quality PDF audit report as bytes.
    
    Can be called either with individual keyword arguments or a single report_data dictionary.
    Returns:
        bytes: The compiled PDF binary content.
    """
    # Support report_data dictionary passed as first argument or keyword argument
    if isinstance(student_name, dict):
        data = student_name
        student_name = data.get("student_name", data.get("name", "Student"))
        degree = data.get("degree", "B.Tech Computer Science")
        gpa = float(data.get("gpa", 8.5))
        experience = int(data.get("experience", 6))
        projects = int(data.get("projects", 4))
        target_role = data.get("target_role", "Data Scientist")
        readiness_metrics = data.get("readiness_metrics", readiness_metrics)
        recommendations = data.get("recommendations", recommendations)
        df_gaps = data.get("df_gaps", df_gaps)
        roadmap = data.get("roadmap", roadmap)
        df_priorities = data.get("df_priorities", df_priorities)
        learning_sequence = data.get("learning_sequence", learning_sequence)
        course_recs = data.get("course_recs", course_recs)
    elif "report_data" in kwargs and isinstance(kwargs["report_data"], dict):
        data = kwargs["report_data"]
        student_name = data.get("student_name", data.get("name", student_name))
        degree = data.get("degree", degree)
        gpa = float(data.get("gpa", gpa))
        experience = int(data.get("experience", experience))
        projects = int(data.get("projects", projects))
        target_role = data.get("target_role", target_role)
        readiness_metrics = data.get("readiness_metrics", readiness_metrics)
        recommendations = data.get("recommendations", recommendations)
        df_gaps = data.get("df_gaps", df_gaps)
        roadmap = data.get("roadmap", roadmap)
        df_priorities = data.get("df_priorities", df_priorities)
        learning_sequence = data.get("learning_sequence", learning_sequence)
        course_recs = data.get("course_recs", course_recs)

    # Safe defaults
    readiness_m = readiness_metrics or {
        "career_match_percentage": 75.0,
        "total_gap_points": 50.0,
        "average_gap_points": 12.5,
        "high_priority_count": 2,
        "medium_priority_count": 2,
        "low_priority_count": 1,
        "mastered_count": 5
    }
    recs_list = recommendations or []
    gaps_table = df_gaps if df_gaps is not None and not df_gaps.empty else pd.DataFrame()
    roadmap_list = roadmap or []
    priorities_df = df_priorities if df_priorities is not None and not df_priorities.empty else pd.DataFrame()
    learning_seq = learning_sequence or []
    courses_dict = course_recs or {}

    buffer = io.BytesIO()
    
    # Target page width = 595.27pt, margins = 36pt -> Usable width = 523.27pt
    USABLE_WIDTH = 523.27
    
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=36,
        rightMargin=36,
        topMargin=48,
        bottomMargin=48
    )
    
    styles = get_pdf_styles()
    story = []
    
    # --------------------------------------------------------------------------
    # 1. HEADER & STUDENT PROFILE BLOCK
    # --------------------------------------------------------------------------
    now_str = datetime.datetime.now().strftime("%B %d, %Y at %I:%M %p")
    story.append(Paragraph("Student Skill Gap Analysis & Employability Audit", styles["DocTitle"]))
    story.append(Paragraph(f"<b>Generated:</b> {now_str} &nbsp;|&nbsp; <b>Evaluation Engine:</b> ML Classifier + Cosine Vector Space + Priority DAG", styles["DocSubtitle"]))
    
    # Profile Info Box Table
    prof_data = [
        [
            Paragraph(f"<b>Student Name:</b> {_esc(student_name)}", styles["TableCell"]),
            Paragraph(f"<b>Target Role:</b> {_esc(target_role)}", styles["TableCell"]),
            Paragraph(f"<b>GPA:</b> {gpa:.2f}/10.0", styles["TableCell"])
        ],
        [
            Paragraph(f"<b>Degree:</b> {_esc(degree)}", styles["TableCell"]),
            Paragraph(f"<b>Experience:</b> {experience} months", styles["TableCell"]),
            Paragraph(f"<b>Projects:</b> {projects} completed", styles["TableCell"])
        ]
    ]
    t_prof = Table(prof_data, colWidths=[USABLE_WIDTH * 0.38, USABLE_WIDTH * 0.38, USABLE_WIDTH * 0.24])
    t_prof.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#CBD5E1")),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(t_prof)
    story.append(Spacer(1, 10))
    
    # --------------------------------------------------------------------------
    # 2. EXECUTIVE SUMMARY & READINESS SCORE
    # --------------------------------------------------------------------------
    story.append(Paragraph("1. Executive Summary & Role Readiness", styles["SectionHeading"]))
    
    match_pct = float(readiness_m.get("career_match_percentage", 0.0))
    match_color = "#16A34A" if match_pct >= 75 else ("#D97706" if match_pct >= 55 else "#DC2626")
    
    summary_kpi_data = [
        [
            Paragraph("<b>Target Role Match Score</b>", styles["TableCellBold"]),
            Paragraph("<b>Total Deficit Points</b>", styles["TableCellBold"]),
            Paragraph("<b>Average Skill Deficit</b>", styles["TableCellBold"]),
            Paragraph("<b>Critical/High Gaps</b>", styles["TableCellBold"]),
            Paragraph("<b>Mastered Skills</b>", styles["TableCellBold"])
        ],
        [
            Paragraph(f"<font color='{match_color}' size=12><b>{match_pct:.1f}%</b></font>", styles["BadgeText"]),
            Paragraph(f"<font size=11><b>{float(readiness_m.get('total_gap_points', 0)):.0f} pts</b></font>", styles["BadgeText"]),
            Paragraph(f"<font size=11><b>{float(readiness_m.get('average_gap_points', 0)):.1f} pts</b></font>", styles["BadgeText"]),
            Paragraph(f"<font color='#DC2626' size=11><b>{int(readiness_m.get('high_priority_count', 0)) + int(readiness_m.get('medium_priority_count', 0))}</b></font>", styles["BadgeText"]),
            Paragraph(f"<font color='#16A34A' size=11><b>{int(readiness_m.get('mastered_count', 0))}</b></font>", styles["BadgeText"])
        ]
    ]
    t_kpi = Table(summary_kpi_data, colWidths=[USABLE_WIDTH * 0.22, USABLE_WIDTH * 0.19, USABLE_WIDTH * 0.20, USABLE_WIDTH * 0.19, USABLE_WIDTH * 0.20])
    t_kpi.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
        ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor("#FFFFFF")),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#CBD5E1")),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(t_kpi)
    story.append(Spacer(1, 10))

    # --------------------------------------------------------------------------
    # 3. TOP RECOMMENDED CAREER ALIGNMENTS
    # --------------------------------------------------------------------------
    if recs_list:
        story.append(Paragraph("2. Top Recommended Career Alignments", styles["SectionHeading"]))
        
        rec_table_data = [
            [
                Paragraph("<b>Rank & Role</b>", styles["TableHead"]),
                Paragraph("<b>Hybrid Match</b>", styles["TableHead"]),
                Paragraph("<b>ML Conf.</b>", styles["TableHead"]),
                Paragraph("<b>Vector Sim.</b>", styles["TableHead"]),
                Paragraph("<b>Fit Assessment & Strengths</b>", styles["TableHead"])
            ]
        ]
        for idx, r in enumerate(recs_list[:4], 1):
            fit_c = "#16A34A" if float(r.get("hybrid_match_pct", 0)) >= 80 else ("#2563EB" if float(r.get("hybrid_match_pct", 0)) >= 60 else "#D97706")
            strengths_txt = ", ".join(r.get("supporting_strengths", [])[:2]) if r.get("supporting_strengths") else "Core Technical Skills"
            
            fit_p = Paragraph(
                f"<b>{_esc(r.get('fit_level', 'Match'))}</b><br/>"
                f"<font color='#475569'>Key Strengths: {_esc(strengths_txt)}</font><br/>"
                f"<font color='#64748B'>{_esc(r.get('explanation', ''))}</font>",
                styles["TableCellSmall"]
            )
            
            rec_table_data.append([
                Paragraph(f"<b>#{idx}. {_esc(r.get('role', 'Career'))}</b>", styles["TableCellBold"]),
                Paragraph(f"<font color='{fit_c}'><b>{float(r.get('hybrid_match_pct', 0)):.1f}%</b></font>", styles["TableCell"]),
                Paragraph(f"{float(r.get('ml_confidence_pct', 0)):.1f}%", styles["TableCell"]),
                Paragraph(f"{float(r.get('vector_similarity_pct', 0)):.1f}%", styles["TableCell"]),
                fit_p
            ])
            
        t_recs = Table(rec_table_data, colWidths=[USABLE_WIDTH * 0.25, USABLE_WIDTH * 0.14, USABLE_WIDTH * 0.11, USABLE_WIDTH * 0.11, USABLE_WIDTH * 0.39], repeatRows=1)
        t_recs.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1E293B")),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#CBD5E1")),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(t_recs)
        story.append(Spacer(1, 10))

    # --------------------------------------------------------------------------
    # 4. INTELLIGENT SKILL IMPROVEMENT PRIORITY
    # --------------------------------------------------------------------------
    if not priorities_df.empty:
        story.append(Paragraph("3. Intelligent Skill Improvement Priorities", styles["SectionHeading"]))
        story.append(Paragraph(
            "<b>Multi-Factor Priority Formulation:</b> <code>40% Gap Severity + 30% Career Importance + 20% Dependency Impact + 10% Current Weakness</code>",
            styles["CalloutText"]
        ))
        story.append(Spacer(1, 4))
        
        prio_table_data = [
            [
                Paragraph("<b>Priority</b>", styles["TableHead"]),
                Paragraph("<b>Skill Name</b>", styles["TableHead"]),
                Paragraph("<b>Score</b>", styles["TableHead"]),
                Paragraph("<b>Tier</b>", styles["TableHead"]),
                Paragraph("<b>Current / Req.</b>", styles["TableHead"]),
                Paragraph("<b>Gap</b>", styles["TableHead"]),
                Paragraph("<b>Why Improve First?</b>", styles["TableHead"])
            ]
        ]
        for idx, (_, row) in enumerate(priorities_df.head(6).iterrows(), 1):
            tier = str(row.get("Priority_Category", "Medium"))
            tier_color = "#DC2626" if tier == "Critical" else ("#EA580C" if tier == "High" else ("#D97706" if tier == "Medium" else "#16A34A"))
            
            prio_table_data.append([
                Paragraph(f"<b>#{idx}</b>", styles["TableCellBold"]),
                Paragraph(f"<b>{_esc(row.get('Display_Name', row.get('Skill')))}</b>", styles["TableCellBold"]),
                Paragraph(f"<font color='{tier_color}'><b>{float(row.get('Priority_Score', 0)):.0f}/100</b></font>", styles["TableCell"]),
                Paragraph(f"<font color='{tier_color}'><b>{_esc(tier)}</b></font>", styles["TableCell"]),
                Paragraph(f"{float(row.get('Student_Score', 0)):.0f} / {float(row.get('Required_Score', 0)):.0f}", styles["TableCell"]),
                Paragraph(f"<b>{float(row.get('Gap', 0)):.0f} pts</b>", styles["TableCellBold"]),
                Paragraph(_esc(row.get("Reason", "")), styles["TableCellSmall"])
            ])
            
        t_prio = Table(prio_table_data, colWidths=[USABLE_WIDTH * 0.08, USABLE_WIDTH * 0.22, USABLE_WIDTH * 0.10, USABLE_WIDTH * 0.10, USABLE_WIDTH * 0.13, USABLE_WIDTH * 0.09, USABLE_WIDTH * 0.28], repeatRows=1)
        t_prio.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1E293B")),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#CBD5E1")),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('LEFTPADDING', (0, 0), (-1, -1), 5),
            ('RIGHTPADDING', (0, 0), (-1, -1), 5),
        ]))
        story.append(t_prio)
        story.append(Spacer(1, 10))

    # --------------------------------------------------------------------------
    # 5. RECOMMENDED PREREQUISITE-AWARE LEARNING SEQUENCE
    # --------------------------------------------------------------------------
    if learning_seq:
        story.append(Paragraph("4. Recommended Learning Sequence (Prerequisite-Aware)", styles["SectionHeading"]))
        story.append(Paragraph(
            "Skills ordered via topological dependency analysis to ensure prerequisite fundamentals are strengthened prior to downstream concepts:",
            styles["Body"]
        ))
        
        seq_data = [
            [
                Paragraph("<b>Step</b>", styles["TableHead"]),
                Paragraph("<b>Recommended Focus Skill</b>", styles["TableHead"]),
                Paragraph("<b>Priority Score</b>", styles["TableHead"]),
                Paragraph("<b>Current Gap</b>", styles["TableHead"]),
                Paragraph("<b>Foundational Role</b>", styles["TableHead"])
            ]
        ]
        for s_idx, item in enumerate(learning_seq[:6], 1):
            disp = item.get("Display_Name", item.get("Skill", ""))
            p_score = float(item.get("Priority_Score", 0))
            gap = float(item.get("Gap", 0))
            cat = str(item.get("Priority_Category", "Medium"))
            
            seq_data.append([
                Paragraph(f"<b>Step {s_idx}</b>", styles["TableCellBold"]),
                Paragraph(f"<b>{_esc(disp)}</b>", styles["TableCellBold"]),
                Paragraph(f"{p_score:.0f}/100", styles["TableCell"]),
                Paragraph(f"{gap:.0f} pts", styles["TableCell"]),
                Paragraph(f"Priority Tier: <b>{_esc(cat)}</b>", styles["TableCellSmall"])
            ])
            
        t_seq = Table(seq_data, colWidths=[USABLE_WIDTH * 0.12, USABLE_WIDTH * 0.32, USABLE_WIDTH * 0.16, USABLE_WIDTH * 0.15, USABLE_WIDTH * 0.25], repeatRows=1)
        t_seq.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1E3A8A")),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#CBD5E1")),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(t_seq)
        story.append(Spacer(1, 10))

    # --------------------------------------------------------------------------
    # 6. PERSONALIZED ONLINE COURSE RECOMMENDATIONS
    # --------------------------------------------------------------------------
    if courses_dict:
        story.append(Paragraph("5. Personalized Online Course Recommendations", styles["SectionHeading"]))
        story.append(Paragraph(
            "Matched via multi-criteria algorithm: <code>35% Skill + 25% Level + 20% Gap Relevance + 10% Prereq + 10% Quality</code>",
            styles["CalloutText"]
        ))
        story.append(Spacer(1, 4))
        
        course_table_data = [
            [
                Paragraph("<b>Skill / Course Name</b>", styles["TableHead"]),
                Paragraph("<b>Provider</b>", styles["TableHead"]),
                Paragraph("<b>Level & Match</b>", styles["TableHead"]),
                Paragraph("<b>Duration</b>", styles["TableHead"]),
                Paragraph("<b>Prerequisites & Topics</b>", styles["TableHead"])
            ]
        ]
        
        for sk_name, c_list in list(courses_dict.items())[:3]:
            sk_disp = SKILL_DISPLAY_NAMES.get(sk_name, sk_name)
            for c in c_list[:2]:
                missing = c.get("missing_prerequisites", [])
                missing_str = ", ".join([_esc(m) for m in missing])
                prereq_str = f"<font color='#DC2626'>⚠️ Missing: {missing_str}</font>" if missing else "<font color='#16A34A'>✅ Prereqs Met</font>"
                topics_str = ", ".join([_esc(t) for t in c.get("topics", [])[:3]])
                
                course_desc = (
                    f"<b>{_esc(c.get('course_name', ''))}</b><br/>"
                    f"<font color='#64748B'>Focus: {_esc(sk_disp)}</font><br/>"
                    f"<font color='#334155'><i>{_esc(c.get('why_recommended', ''))}</i></font>"
                )
                
                course_table_data.append([
                    Paragraph(course_desc, styles["TableCellSmall"]),
                    Paragraph(_esc(c.get('provider', '')), styles["TableCellSmall"]),
                    Paragraph(f"<b>{_esc(c.get('level', ''))}</b><br/><font color='#16A34A'><b>{float(c.get('match_score', 0)):.0f}% Match</b></font><br/>⭐ {float(c.get('rating', 4.7)):.1f}/5.0", styles["TableCellSmall"]),
                    Paragraph(f"{c.get('duration', 30)} hrs<br/>{_esc(c.get('cost_type', 'Free Audit'))}", styles["TableCellSmall"]),
                    Paragraph(f"{prereq_str}<br/><b>Topics:</b> {topics_str}", styles["TableCellSmall"])
                ])
                
        t_courses = Table(course_table_data, colWidths=[USABLE_WIDTH * 0.35, USABLE_WIDTH * 0.18, USABLE_WIDTH * 0.15, USABLE_WIDTH * 0.12, USABLE_WIDTH * 0.20], repeatRows=1)
        t_courses.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1E293B")),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#CBD5E1")),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ('LEFTPADDING', (0, 0), (-1, -1), 5),
            ('RIGHTPADDING', (0, 0), (-1, -1), 5),
        ]))
        story.append(t_courses)
        story.append(Spacer(1, 10))

    # --------------------------------------------------------------------------
    # 7. COMPREHENSIVE SKILL GAP BREAKDOWN TABLE
    # --------------------------------------------------------------------------
    if not gaps_table.empty:
        story.append(Paragraph("6. Comprehensive Skill Gap Breakdown Table", styles["SectionHeading"]))
        
        gaps_table_data = [
            [
                Paragraph("<b>Technical Skill</b>", styles["TableHead"]),
                Paragraph("<b>Current Level</b>", styles["TableHead"]),
                Paragraph("<b>Required Benchmark</b>", styles["TableHead"]),
                Paragraph("<b>Deficit Gap</b>", styles["TableHead"]),
                Paragraph("<b>Severity Status</b>", styles["TableHead"])
            ]
        ]
        
        for _, row in gaps_table.iterrows():
            gap = float(row.get("Gap", 0))
            sev = str(row.get("Severity", "Mastered"))
            sev_c = "#16A34A" if gap == 0 else ("#2563EB" if gap <= 15 else ("#D97706" if gap <= 35 else "#DC2626"))
            
            gaps_table_data.append([
                Paragraph(f"<b>{_esc(row.get('Display_Name', row.get('Skill')))}</b>", styles["TableCellBold"]),
                Paragraph(f"{float(row.get('Student_Score', 0)):.0f}", styles["TableCell"]),
                Paragraph(f"{float(row.get('Required_Score', 0)):.0f}", styles["TableCell"]),
                Paragraph(f"<b>{gap:.0f} pts</b>", styles["TableCellBold"]),
                Paragraph(f"<font color='{sev_c}'><b>{_esc(sev)}</b></font>", styles["TableCellSmall"])
            ])
            
        t_gaps = Table(gaps_table_data, colWidths=[USABLE_WIDTH * 0.34, USABLE_WIDTH * 0.16, USABLE_WIDTH * 0.18, USABLE_WIDTH * 0.14, USABLE_WIDTH * 0.18], repeatRows=1)
        t_gaps.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1E293B")),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#CBD5E1")),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
            ('LEFTPADDING', (0, 0), (-1, -1), 5),
            ('RIGHTPADDING', (0, 0), (-1, -1), 5),
        ]))
        story.append(t_gaps)
        story.append(Spacer(1, 10))

    # --------------------------------------------------------------------------
    # 8. ACTIONABLE 6-PHASE LEARNING ROADMAP
    # --------------------------------------------------------------------------
    if roadmap_list:
        story.append(Paragraph("7. Actionable 6-Phase Learning Roadmap", styles["SectionHeading"]))
        story.append(Paragraph(
            "Structured step-by-step roadmap ordered sequentially to bridge critical skill deficits efficiently:",
            styles["Body"]
        ))
        
        for item in roadmap_list[:4]:
            gap = float(item.get("gap", 0))
            if gap <= 0:
                continue
                
            step_num = item.get("step_number", 1)
            disp = item.get("display_name", "")
            p_label = str(item.get("priority_label", "High"))
            est_h = item.get("estimated_hours", 20)
            
            p_color = "#DC2626" if p_label == "Critical" else ("#EA580C" if p_label == "High" else "#2563EB")
            
            top_c = item.get("phase_3_recommended_course")
            course_name = top_c.get("course_name", "Online Course Specialization") if top_c else "Self-Paced Specialization"
            course_prov = top_c.get("provider", "Verified Online Provider") if top_c else "Verified Provider"
            
            topics = item.get("phase_4_practice_topics", item.get("topics", []))
            topics_txt = ", ".join([_esc(t) for t in topics]) if isinstance(topics, list) else _esc(topics)
            project_txt = item.get("phase_5_portfolio_project", item.get("suggested_project", "End-to-End Project"))
            
            prereq_list = item.get('phase_1_prerequisites', ['Foundational requirements satisfied'])
            prereq_joined = '; '.join([_esc(p) for p in prereq_list]) if isinstance(prereq_list, list) else _esc(prereq_list)

            step_card_content = [
                [
                    Paragraph(f"<b>STEP {step_num}: {_esc(disp)}</b> &nbsp;|&nbsp; <font color='{p_color}'><b>{_esc(p_label)} Priority</b></font> (Gap: {gap:.0f} pts | Est. Time: {est_h} hrs)", styles["TableCellBold"]),
                ],
                [
                    Paragraph(
                        f"• <b>Phase 1 (Prerequisites):</b> {prereq_joined}<br/>"
                        f"• <b>Phase 2 (Skill Focus):</b> Current Level: {float(item.get('student_score', 0)):.0f} → Required Benchmark: {float(item.get('required_score', 0)):.0f}<br/>"
                        f"• <b>Phase 3 (Recommended Course):</b> <b>{_esc(course_name)}</b> ({_esc(course_prov)})<br/>"
                        f"• <b>Phase 4 (Practice Topics):</b> {topics_txt}<br/>"
                        f"• <b>Phase 5 (Portfolio Project):</b> <i>{_esc(project_txt)}</i><br/>"
                        f"• <b>Phase 6 (Re-Assessment):</b> {_esc(item.get('phase_6_reassessment', 'Re-evaluate skill scores in Skill Assessment upon completion.'))}",
                        styles["TableCell"]
                    )
                ]
            ]
            t_step = Table(step_card_content, colWidths=[USABLE_WIDTH])
            t_step.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
                ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor("#FFFFFF")),
                ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#CBD5E1")),
                ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
                ('TOPPADDING', (0, 0), (-1, -1), 5),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
                ('LEFTPADDING', (0, 0), (-1, -1), 8),
                ('RIGHTPADDING', (0, 0), (-1, -1), 8),
            ]))
            
            story.append(KeepTogether([t_step, Spacer(1, 6)]))

    # --------------------------------------------------------------------------
    # 9. CLOSING AUDIT SIGN-OFF
    # --------------------------------------------------------------------------
    story.append(Spacer(1, 8))
    signoff_text = (
        "<b>Auditing Methodology Note:</b> This report combines a supervised multi-class Support Vector Machine (SVM) "
        "calibrated probability model with 20-dimensional cosine vector similarity and deterministic skill dependency DAG heuristics. "
        "Scores represent relative alignment with standardized industry job benchmarks and serve as structured guidance for academic upskilling."
    )
    story.append(Paragraph(signoff_text, styles["CalloutText"]))
    
    # Build document
    doc.build(story, canvasmaker=NumberedCanvas)
    
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes


if __name__ == "__main__":
    from src.data_loader import load_processed_data
    from src.skill_priority import calculate_skill_priorities, generate_learning_sequence
    from src.course_recommender import recommend_courses_for_profile
    from src.career_recommendation import recommend_careers
    from src.skill_gap import calculate_skill_gaps, compute_role_readiness_metrics, generate_enhanced_learning_roadmap
    
    test_skills = {"Python": 55, "Statistics_Math": 40, "Machine_Learning": 45, "SQL": 70}
    _, df_matrix = load_processed_data()
    role_row = df_matrix[df_matrix["Job_Role"] == "Data Scientist"].iloc[0]
    reqs = {s: float(role_row[s]) for s in CORE_SKILLS}
    df_gaps = calculate_skill_gaps(test_skills, reqs)
    readiness = compute_role_readiness_metrics(df_gaps)
    recs = recommend_careers(test_skills, top_n=3)
    
    df_p = calculate_skill_priorities(test_skills, "Data Scientist", df_matrix)
    _, l_seq, _, _ = generate_learning_sequence(df_p)
    c_recs = recommend_courses_for_profile(test_skills, df_p, top_skills_count=3)
    roadmap = generate_enhanced_learning_roadmap(l_seq, c_recs, test_skills)
    
    pdf = generate_pdf_report(
        student_name="Alex Chen",
        degree="B.Tech Computer Science",
        gpa=8.5,
        experience=6,
        projects=4,
        target_role="Data Scientist",
        readiness_metrics=readiness,
        recommendations=recs,
        df_gaps=df_gaps,
        roadmap=roadmap,
        df_priorities=df_p,
        learning_sequence=l_seq,
        course_recs=c_recs
    )
    print(f"Generated PDF successfully! Size: {len(pdf)} bytes, Header: {pdf[:4]}")
