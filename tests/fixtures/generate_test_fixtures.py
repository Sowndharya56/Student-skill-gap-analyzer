"""
========================================================================================
DEVELOPMENT & TEST BENCHMARK FIXTURE GENERATOR (TEST FIXTURES ONLY)
========================================================================================
WARNING / DISCLAIMER:
This script is intended STRICTLY for offline benchmark testing, continuous integration,
and generating test fixtures for unit tests in tests/.
It is NOT part of the active production Data Science pipeline.
It must NEVER be invoked during production runtime or called by app.py or model_training.py.
========================================================================================
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    CORE_SKILLS,
    TARGET_CAREERS,
    RANDOM_STATE
)

def generate_benchmark_test_fixtures(target_dir: Path = None):
    """
    Generates synthetic benchmark test fixtures for offline unit testing and test coverage.
    """
    if target_dir is None:
        target_dir = RAW_DATA_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    
    np.random.seed(RANDOM_STATE)
    
    # Benchmark job requirements
    role_benchmarks = {
        "Data Scientist": {
            "Python": 85, "SQL": 80, "R": 65, "Java": 40, "C++": 35, "JavaScript": 30,
            "Data_Analysis": 90, "Machine_Learning": 88, "Deep_Learning": 75, "NLP": 70,
            "Data_Visualization": 80, "Cloud_Computing": 65, "DevOps_CICD": 50, "Docker_Kubernetes": 55,
            "Database_Design": 65, "Big_Data": 70, "Cybersecurity_Basics": 35,
            "Statistics_Math": 88, "Problem_Solving": 85, "Communication": 80
        },
        "Data Analyst": {
            "Python": 70, "SQL": 88, "R": 50, "Java": 25, "C++": 20, "JavaScript": 35,
            "Data_Analysis": 92, "Machine_Learning": 50, "Deep_Learning": 30, "NLP": 35,
            "Data_Visualization": 92, "Cloud_Computing": 50, "DevOps_CICD": 35, "Docker_Kubernetes": 30,
            "Database_Design": 75, "Big_Data": 55, "Cybersecurity_Basics": 30,
            "Statistics_Math": 80, "Problem_Solving": 75, "Communication": 88
        },
        "Machine Learning Engineer": {
            "Python": 92, "SQL": 75, "R": 50, "Java": 60, "C++": 70, "JavaScript": 40,
            "Data_Analysis": 82, "Machine_Learning": 95, "Deep_Learning": 90, "NLP": 80,
            "Data_Visualization": 70, "Cloud_Computing": 80, "DevOps_CICD": 75, "Docker_Kubernetes": 82,
            "Database_Design": 70, "Big_Data": 78, "Cybersecurity_Basics": 45,
            "Statistics_Math": 90, "Problem_Solving": 90, "Communication": 75
        },
        "Full Stack Developer": {
            "Python": 75, "SQL": 78, "R": 20, "Java": 75, "C++": 50, "JavaScript": 92,
            "Data_Analysis": 50, "Machine_Learning": 35, "Deep_Learning": 25, "NLP": 25,
            "Data_Visualization": 60, "Cloud_Computing": 72, "DevOps_CICD": 70, "Docker_Kubernetes": 68,
            "Database_Design": 85, "Big_Data": 45, "Cybersecurity_Basics": 65,
            "Statistics_Math": 60, "Problem_Solving": 88, "Communication": 80
        },
        "Backend Developer": {
            "Python": 85, "SQL": 88, "R": 20, "Java": 90, "C++": 75, "JavaScript": 65,
            "Data_Analysis": 55, "Machine_Learning": 30, "Deep_Learning": 20, "NLP": 20,
            "Data_Visualization": 45, "Cloud_Computing": 75, "DevOps_CICD": 75, "Docker_Kubernetes": 75,
            "Database_Design": 92, "Big_Data": 60, "Cybersecurity_Basics": 70,
            "Statistics_Math": 65, "Problem_Solving": 90, "Communication": 75
        },
        "Frontend Developer": {
            "Python": 50, "SQL": 55, "R": 15, "Java": 40, "C++": 30, "JavaScript": 96,
            "Data_Analysis": 45, "Machine_Learning": 20, "Deep_Learning": 15, "NLP": 15,
            "Data_Visualization": 75, "Cloud_Computing": 55, "DevOps_CICD": 60, "Docker_Kubernetes": 50,
            "Database_Design": 60, "Big_Data": 30, "Cybersecurity_Basics": 55,
            "Statistics_Math": 50, "Problem_Solving": 80, "Communication": 82
        },
        "Cloud & DevOps Engineer": {
            "Python": 78, "SQL": 65, "R": 20, "Java": 60, "C++": 50, "JavaScript": 55,
            "Data_Analysis": 45, "Machine_Learning": 35, "Deep_Learning": 25, "NLP": 20,
            "Data_Visualization": 50, "Cloud_Computing": 95, "DevOps_CICD": 95, "Docker_Kubernetes": 94,
            "Database_Design": 70, "Big_Data": 65, "Cybersecurity_Basics": 80,
            "Statistics_Math": 60, "Problem_Solving": 85, "Communication": 78
        },
        "Cybersecurity Analyst": {
            "Python": 80, "SQL": 70, "R": 25, "Java": 60, "C++": 70, "JavaScript": 55,
            "Data_Analysis": 60, "Machine_Learning": 45, "Deep_Learning": 35, "NLP": 30,
            "Data_Visualization": 55, "Cloud_Computing": 80, "DevOps_CICD": 65, "Docker_Kubernetes": 65,
            "Database_Design": 70, "Big_Data": 50, "Cybersecurity_Basics": 96,
            "Statistics_Math": 68, "Problem_Solving": 88, "Communication": 80
        },
        "AI Specialist": {
            "Python": 94, "SQL": 72, "R": 60, "Java": 55, "C++": 78, "JavaScript": 45,
            "Data_Analysis": 85, "Machine_Learning": 95, "Deep_Learning": 96, "NLP": 92,
            "Data_Visualization": 72, "Cloud_Computing": 78, "DevOps_CICD": 68, "Docker_Kubernetes": 75,
            "Database_Design": 68, "Big_Data": 75, "Cybersecurity_Basics": 45,
            "Statistics_Math": 95, "Problem_Solving": 92, "Communication": 78
        },
        "Business Intelligence Developer": {
            "Python": 65, "SQL": 92, "R": 45, "Java": 30, "C++": 20, "JavaScript": 40,
            "Data_Analysis": 90, "Machine_Learning": 45, "Deep_Learning": 25, "NLP": 25,
            "Data_Visualization": 96, "Cloud_Computing": 60, "DevOps_CICD": 45, "Docker_Kubernetes": 40,
            "Database_Design": 88, "Big_Data": 70, "Cybersecurity_Basics": 40,
            "Statistics_Math": 78, "Problem_Solving": 75, "Communication": 85
        }
    }
    
    df_req = pd.DataFrame.from_dict(role_benchmarks, orient="index")
    df_req.index.name = "Job_Role"
    df_req.reset_index(inplace=True)
    df_req.to_csv(target_dir / "industry_job_requirements.csv", index=False)

    num_samples = 1500
    students_data = []
    degrees = ["B.Tech Computer Science", "B.Tech Information Technology", "B.Sc Data Science", "BCA", "MCA", "B.Tech AI & Data Science", "M.Tech CSE"]
    degree_probs = [0.35, 0.25, 0.15, 0.10, 0.05, 0.07, 0.03]
    first_names = ["Aarav", "Priya", "Rahul", "Ananya", "Rohan", "Sneha", "Aditya", "Neha", "Vikram", "Pooja", "Arjun", "Kavya", "Siddharth", "Ishita", "Varun", "Tanvi", "Manish", "Divya", "Karan", "Rhea"]
    last_names = ["Sharma", "Patel", "Verma", "Rao", "Nair", "Gupta", "Singh", "Reddy", "Mehta", "Iyer", "Kumar", "Das", "Joshi", "Bhat", "Chopra", "Deshmukh", "Kapoor", "Banerjee", "Menon", "Saxena"]
    roles_list = list(role_benchmarks.keys())
    
    for i in range(num_samples):
        true_role = np.random.choice(roles_list)
        benchmark = role_benchmarks[true_role]
        
        student_skills = {}
        for skill in CORE_SKILLS:
            mean_val = benchmark[skill]
            score = np.random.normal(loc=mean_val - 5, scale=12)
            score = int(np.clip(score, 10, 99))
            student_skills[skill] = score
        
        name = f"{np.random.choice(first_names)} {np.random.choice(last_names)}"
        gpa = round(np.clip(np.random.normal(7.8, 0.8), 5.5, 9.9), 2)
        degree = np.random.choice(degrees, p=degree_probs)
        experience_months = int(np.clip(np.random.exponential(scale=6), 0, 36))
        projects_completed = int(np.clip(np.random.poisson(lam=4), 1, 12))
        certifications_count = int(np.clip(np.random.poisson(lam=2), 0, 8))
        
        record = {
            "Student_ID": f"STU_{10001 + i}",
            "Student_Name": name,
            "Degree": degree,
            "GPA": gpa,
            "Experience_Months": experience_months,
            "Projects_Count": projects_completed,
            "Certifications_Count": certifications_count,
            **student_skills,
            "Target_Career": true_role
        }
        students_data.append(record)
        
    df_students = pd.DataFrame(students_data)
    
    raw_col_map = {
        "Python": "python_skill",
        "Machine_Learning": "ML_Proficiency",
        "Data_Visualization": "PowerBI_Tableau_Visuals",
        "Cloud_Computing": "cloud_aws_azure",
        "DevOps_CICD": "DevOps_Git_CICD"
    }
    df_students_raw = df_students.copy()
    df_students_raw.rename(columns=raw_col_map, inplace=True)
    
    duplicate_rows = df_students_raw.iloc[np.random.choice(len(df_students_raw), size=15, replace=False)]
    df_students_raw = pd.concat([df_students_raw, duplicate_rows], ignore_index=True)
    df_students_raw.loc[np.random.choice(len(df_students_raw), size=20, replace=False), "Certifications_Count"] = np.nan
    df_students_raw.loc[np.random.choice(len(df_students_raw), size=10, replace=False), "Experience_Months"] = np.nan
    
    df_students_raw.to_csv(target_dir / "student_career_profiles.csv", index=False)
    print("Generated unit test benchmark fixtures in:", target_dir)

if __name__ == "__main__":
    generate_benchmark_test_fixtures()
