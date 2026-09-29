"""
Configuration Module for AI-Based Student Skill Gap Analyzer & Career Recommendation System.
Contains global constants, file paths, taxonomy mappings, heuristic weights, and hyperparameters.
"""

from pathlib import Path

# Base Paths (Relative to project root for portability)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models" / "trained_models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Data File Paths
STUDENT_DATA_RAW = RAW_DATA_DIR / "student_career_profiles.csv"
JOB_REQUIREMENTS_RAW = RAW_DATA_DIR / "industry_job_requirements.csv"
JOB_POSTINGS_RAW = RAW_DATA_DIR / "tech_job_postings.csv"

STUDENT_DATA_CLEANED = PROCESSED_DATA_DIR / "cleaned_student_profiles.csv"
ROLE_SKILL_MATRIX = PROCESSED_DATA_DIR / "role_skill_matrix.csv"
ENGINEERED_FEATURES_DATA = PROCESSED_DATA_DIR / "engineered_features.csv"
COURSES_DATA = PROCESSED_DATA_DIR / "courses.csv"

# Model File Paths
BEST_MODEL_PATH = MODELS_DIR / "career_classifier.joblib"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.joblib"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"
SPLIT_INDICES_PATH = MODELS_DIR / "split_indices.json"

# Core Standardized Skills List (20 Core Tech & Analytical Skills used in ML vector space)
CORE_SKILLS = [
    "Python",
    "SQL",
    "R",
    "Java",
    "C++",
    "JavaScript",
    "Data_Analysis",
    "Machine_Learning",
    "Deep_Learning",
    "NLP",
    "Data_Visualization",
    "Cloud_Computing",
    "DevOps_CICD",
    "Docker_Kubernetes",
    "Database_Design",
    "Big_Data",
    "Cybersecurity_Basics",
    "Statistics_Math",
    "Problem_Solving",
    "Communication"
]

# Display Name Mapping for UI and Reports
SKILL_DISPLAY_NAMES = {
    "Python": "Python Programming",
    "SQL": "SQL & Querying",
    "R": "R Programming",
    "Java": "Java Development",
    "C++": "C++ Programming",
    "C#": "C# Programming",
    ".NET": ".NET Framework / Core",
    "Node.js": "Node.js Backend",
    "JavaScript": "JavaScript / Web Dev",
    "Data_Analysis": "Data Analysis & Wrangling",
    "Machine_Learning": "Machine Learning (Scikit-Learn)",
    "Deep_Learning": "Deep Learning (PyTorch/TF)",
    "NLP": "Natural Language Processing",
    "Data_Visualization": "Data Visualization (Power BI/Tableau)",
    "Cloud_Computing": "Cloud Architecture (AWS/Azure/GCP)",
    "DevOps_CICD": "DevOps & CI/CD Pipelines",
    "Docker_Kubernetes": "Containers (Docker / K8s)",
    "Database_Design": "Database Design & NoSQL",
    "Big_Data": "Big Data (Spark / Hadoop)",
    "Cybersecurity_Basics": "Cybersecurity & Network Security",
    "Statistics_Math": "Statistics & Applied Math",
    "Problem_Solving": "Algorithmic Problem Solving",
    "Communication": "Professional Communication"
}

# Skill Domain Categorization
SKILL_DOMAINS = {
    "Programming & Core CS": ["Python", "Java", "C++", "JavaScript", "R", "Problem_Solving"],
    "Data & Business Analytics": ["SQL", "Data_Analysis", "Data_Visualization", "Statistics_Math", "Database_Design", "Big_Data"],
    "AI & Intelligent Systems": ["Machine_Learning", "Deep_Learning", "NLP"],
    "Cloud, DevOps & Security": ["Cloud_Computing", "DevOps_CICD", "Docker_Kubernetes", "Cybersecurity_Basics"],
    "Professional & Soft Skills": ["Communication"]
}

# 10 High-Demand Target Career Roles
TARGET_CAREERS = [
    "Data Scientist",
    "Data Analyst",
    "Machine Learning Engineer",
    "Full Stack Developer",
    "Backend Developer",
    "Frontend Developer",
    "Cloud & DevOps Engineer",
    "Cybersecurity Analyst",
    "AI Specialist",
    "Business Intelligence Developer"
]

# Skill Name Variation Normalization Dictionary (Used during Data Cleaning & NLP)
# NOTE: C# is strictly mapped to C# (NOT C++), and .NET is strictly mapped to .NET (NOT Java)
SKILL_SYNONYMS = {
    # Python
    "python": "Python",
    "python programming": "Python",
    "python3": "Python",
    "py": "Python",
    # SQL
    "sql": "SQL",
    "structured query language": "SQL",
    "mysql": "SQL",
    "postgresql": "SQL",
    "postgres": "SQL",
    "plsql": "SQL",
    "tsql": "SQL",
    # R
    "r programming": "R",
    "r language": "R",
    # Java
    "java": "Java",
    "core java": "Java",
    "spring boot": "Java",
    "spring": "Java",
    # C++ (Strictly C++)
    "c++": "C++",
    "cpp": "C++",
    "c plus plus": "C++",
    # C# (Strictly distinct canonical skill C#)
    "c#": "C#",
    "csharp": "C#",
    "c sharp": "C#",
    # .NET (Strictly distinct canonical skill .NET)
    ".net": ".NET",
    "dotnet": ".NET",
    ".net core": ".NET",
    "asp.net": ".NET",
    # JavaScript & Web
    "javascript": "JavaScript",
    "js": "JavaScript",
    "typescript": "JavaScript",
    "ts": "JavaScript",
    "node.js": "Node.js",
    "nodejs": "Node.js",
    "react": "JavaScript",
    "react.js": "JavaScript",
    "vue": "JavaScript",
    "angular": "JavaScript",
    # Data Analysis
    "data analysis": "Data_Analysis",
    "data analytics": "Data_Analysis",
    "pandas": "Data_Analysis",
    "numpy": "Data_Analysis",
    "data wrangling": "Data_Analysis",
    "eda": "Data_Analysis",
    # Machine Learning
    "machine learning": "Machine_Learning",
    "machine-learning": "Machine_Learning",
    "ml": "Machine_Learning",
    "scikit-learn": "Machine_Learning",
    "sklearn": "Machine_Learning",
    "xgboost": "Machine_Learning",
    # Deep Learning
    "deep learning": "Deep_Learning",
    "deep-learning": "Deep_Learning",
    "neural networks": "Deep_Learning",
    "pytorch": "Deep_Learning",
    "tensorflow": "Deep_Learning",
    "keras": "Deep_Learning",
    # NLP
    "nlp": "NLP",
    "natural language processing": "NLP",
    "text analytics": "NLP",
    "spacy": "NLP",
    "nltk": "NLP",
    "llm": "NLP",
    "transformers": "NLP",
    # Data Visualization
    "data visualization": "Data_Visualization",
    "power bi": "Data_Visualization",
    "powerbi": "Data_Visualization",
    "tableau": "Data_Visualization",
    "matplotlib": "Data_Visualization",
    "seaborn": "Data_Visualization",
    "plotly": "Data_Visualization",
    # Cloud Computing
    "cloud computing": "Cloud_Computing",
    "cloud": "Cloud_Computing",
    "aws": "Cloud_Computing",
    "amazon web services": "Cloud_Computing",
    "azure": "Cloud_Computing",
    "microsoft azure": "Cloud_Computing",
    "gcp": "Cloud_Computing",
    "google cloud": "Cloud_Computing",
    # DevOps & CI/CD
    "devops": "DevOps_CICD",
    "ci/cd": "DevOps_CICD",
    "cicd": "DevOps_CICD",
    "continuous integration": "DevOps_CICD",
    "continuous deployment": "DevOps_CICD",
    "git": "DevOps_CICD",
    "github actions": "DevOps_CICD",
    "gitlab": "DevOps_CICD",
    "jenkins": "DevOps_CICD",
    "terraform": "DevOps_CICD",
    # Docker & Kubernetes
    "docker": "Docker_Kubernetes",
    "kubernetes": "Docker_Kubernetes",
    "k8s": "Docker_Kubernetes",
    "containers": "Docker_Kubernetes",
    # Database Design
    "database design": "Database_Design",
    "rdbms": "Database_Design",
    "nosql": "Database_Design",
    "mongodb": "Database_Design",
    "redis": "Database_Design",
    "database modeling": "Database_Design",
    # Big Data
    "big data": "Big_Data",
    "spark": "Big_Data",
    "pyspark": "Big_Data",
    "hadoop": "Big_Data",
    "kafka": "Big_Data",
    # Cybersecurity
    "cybersecurity": "Cybersecurity_Basics",
    "cyber security": "Cybersecurity_Basics",
    "network security": "Cybersecurity_Basics",
    "infosec": "Cybersecurity_Basics",
    "owasp": "Cybersecurity_Basics",
    # Statistics & Math
    "statistics": "Statistics_Math",
    "probability": "Statistics_Math",
    "applied math": "Statistics_Math",
    "linear algebra": "Statistics_Math",
    "hypothesis testing": "Statistics_Math",
    # Problem Solving & DSA
    "problem solving": "Problem_Solving",
    "data structures": "Problem_Solving",
    "algorithms": "Problem_Solving",
    "dsa": "Problem_Solving",
    "leetcode": "Problem_Solving",
    # Communication
    "communication": "Communication",
    "soft skills": "Communication",
    "presentation": "Communication",
    "teamwork": "Communication"
}

# Skill Gap Severity Thresholds (Heuristic Application Design Choices)
GAP_THRESHOLDS = {
    "NONE": (0, 0, "Mastered / Sufficient", "#10B981"),        # Green
    "LOW": (1, 15, "Low Priority Gap", "#3B82F6"),             # Blue
    "MEDIUM": (16, 35, "Medium Priority Gap", "#F59E0B"),       # Amber
    "HIGH": (36, 100, "High Priority Gap", "#EF4444")          # Red
}

# Hybrid Career Recommendation Weights
# NOTE: These weights are heuristic design parameters selected for the application architecture.
# They are not statistically learned or gradient-optimized parameters.
HYBRID_WEIGHTS = {
    "ml_probability": 0.40,
    "vector_similarity": 0.35,
    "readiness_match": 0.25
}

# Intelligent Skill Improvement Priority Weights
PRIORITY_WEIGHTS = {
    "gap_severity": 0.40,
    "career_importance": 0.30,
    "dependency_impact": 0.20,
    "current_weakness": 0.10
}

# Priority Category Thresholds
PRIORITY_CATEGORIES = {
    "CRITICAL": (80, 100, "Critical", "🔴", "#EF4444"),
    "HIGH": (60, 79, "High", "🟠", "#F97316"),
    "MEDIUM": (40, 59, "Medium", "🟡", "#F59E0B"),
    "LOW": (0, 39, "Low", "🟢", "#10B981")
}

# Personalized Course Recommendation Matching Weights
COURSE_MATCH_WEIGHTS = {
    "skill_match": 0.35,
    "level_match": 0.25,
    "gap_relevance": 0.20,
    "prerequisite_match": 0.10,
    "course_quality": 0.10
}

# Standardized Skill Dependency Directed Acyclic Graph (DAG) for 20 Core Skills
# Maps each skill to the list of downstream skills that depend directly on it
SKILL_DEPENDENCY_GRAPH = {
    "Python": ["Data_Analysis", "Machine_Learning", "Big_Data", "DevOps_CICD"],
    "Statistics_Math": ["Data_Analysis", "Machine_Learning"],
    "Data_Analysis": ["Machine_Learning", "Data_Visualization", "Big_Data"],
    "SQL": ["Database_Design", "Data_Analysis", "Big_Data"],
    "Machine_Learning": ["Deep_Learning", "NLP"],
    "Deep_Learning": ["NLP"],
    "Cloud_Computing": ["DevOps_CICD", "Docker_Kubernetes", "Big_Data"],
    "DevOps_CICD": ["Docker_Kubernetes"],
    "Database_Design": ["Big_Data"],
    "Problem_Solving": ["Python", "Java", "C++", "JavaScript", "Machine_Learning"],
    "Java": ["Big_Data"],
    "R": ["Data_Analysis"],
    "C++": [],
    "JavaScript": [],
    "NLP": [],
    "Data_Visualization": [],
    "Docker_Kubernetes": [],
    "Big_Data": [],
    "Cybersecurity_Basics": [],
    "Communication": []
}

# Honest Dataset Provenance Metadata
DATASET_PROVENANCE = {
    "student_profiles": {
        "name": "Student Career Profiles Dataset",
        "source": "Development Benchmark Fixture (Synthetic Profile Generator)",
        "kaggle_url": "Not verified (Project benchmark fixture)",
        "author": "Not verified / Academic Project Benchmark",
        "license": "Academic Demonstration / Open Educational Use",
        "verified_status": "Development Benchmark Fixture",
        "file_path": "data/raw/student_career_profiles.csv",
        "rows": 1500,
        "cols": 28,
        "purpose": "Provides structured baseline profiles with 20 technical skill metrics and academic indicators for algorithm prototyping and evaluation.",
        "limitations": "Generated based on role distributions for benchmark testing; not verified as an independent real-world field survey."
    },
    "job_requirements": {
        "name": "Industry Job Requirements Matrix",
        "source": "Curated Tech Job Role Baseline Matrix",
        "kaggle_url": "Not verified (Curated industry archetype matrix)",
        "author": "Academic Project Team",
        "license": "Open Educational Use",
        "verified_status": "Curated Archetype Benchmark",
        "file_path": "data/raw/industry_job_requirements.csv",
        "rows": 10,
        "cols": 21,
        "purpose": "Defines baseline proficiency requirements across 10 defined high-demand tech job roles.",
        "limitations": "Represents standardized generalized benchmarks; actual industry requirements vary significantly across companies and seniorities."
    },
    "job_postings": {
        "name": "Tech Job Postings & Requirements Corpus",
        "source": "Development Job Description Corpus",
        "kaggle_url": "Not verified",
        "author": "Academic Project Team",
        "license": "Open Educational Use",
        "verified_status": "Template-Generated Test Corpus",
        "file_path": "data/raw/tech_job_postings.csv",
        "rows": 350,
        "cols": 7,
        "purpose": "Corpus of tech job postings used for testing rule-based NLP entity extraction and TF-IDF keyword weighting.",
        "limitations": "Contains recurring template descriptions for NLP testing; does not reflect live unstructured labor market scraping."
    },
    "courses": {
        "name": "Personalized Course Recommendation Catalog",
        "source": "Curated Educational Course Catalog",
        "kaggle_url": "Not verified (Curated representative benchmark catalog)",
        "author": "Academic Project Team",
        "license": "Open Educational Use",
        "verified_status": "Curated Course Catalog",
        "file_path": "data/processed/courses.csv",
        "rows": 60,
        "cols": 12,
        "purpose": "Provides structured online courses with skill alignments, difficulty levels, duration, ratings, prerequisites, and learning topics for automated course recommendation.",
        "limitations": "Curated representative course catalog for demonstration and recommendation testing."
    }
}

# Curated Learning Resources and Projects per Skill
LEARNING_RESOURCES = {
    "Python": {
        "topics": ["OOP & Functional Paradigms", "Data Structures & Itertools", "AsyncIO & Fast Execution"],
        "recommended_courses": ["Coursera: Python for Everybody", "freeCodeCamp Python Masterclass"],
        "suggested_project": "Build an Automated ETL Pipeline & CLI Utility"
    },
    "SQL": {
        "topics": ["Window Functions (RANK, DENSE_RANK, LEAD, LAG)", "Complex CTEs & Query Optimization", "Indexing & Execution Plans"],
        "recommended_courses": ["Mode Analytics SQL Tutorial", "LeetCode Database SQL 50"],
        "suggested_project": "Design an E-Commerce Analytics Database Schema with Stored Procedures"
    },
    "R": {
        "topics": ["Tidyverse (dplyr, ggplot2)", "Statistical Hypothesis Testing", "R Shiny Dashboards"],
        "recommended_courses": ["HarvardX Data Science: R Basics", "Datacamp R Track"],
        "suggested_project": "Statistical Bio-Data Experiment Analysis with Shiny Web App"
    },
    "Java": {
        "topics": ["Spring Boot REST APIs", "JVM Memory Management & Concurrency", "Design Patterns & Microservices"],
        "recommended_courses": ["Baeldung Spring Boot Tutorials", "Java Brains Microservices Series"],
        "suggested_project": "Build a Resilient Banking Microservice with Spring Cloud"
    },
    "C++": {
        "topics": ["Modern C++ (C++17/20)", "Smart Pointers & Memory Management", "Multi-threading & STL Optimization"],
        "recommended_courses": ["learncpp.com Complete Guide", "Udacity C++ Nanodegree"],
        "suggested_project": "High-Performance In-Memory Key-Value Store"
    },
    "C#": {
        "topics": ["C# Language Fundamentals", "LINQ & Entity Framework Core", "ASP.NET Web APIs"],
        "recommended_courses": ["Microsoft Learn C# Track", "freeCodeCamp C# Full Course"],
        "suggested_project": "Enterprise Inventory Management REST API in ASP.NET Core"
    },
    ".NET": {
        "topics": [".NET Architecture & CLR", "Dependency Injection in .NET Core", "Microservices with .NET"],
        "recommended_courses": ["Microsoft .NET Documentation & Tutorials", "Pluralsight .NET Path"],
        "suggested_project": "Distributed Message Queue Worker Service using .NET Core"
    },
    "Node.js": {
        "topics": ["Node.js Event Loop & Streams", "Express.js REST Architecture", "JWT Authentication & Middleware"],
        "recommended_courses": ["Node.js Official Guide", "Coursera Server-side Development with NodeJS"],
        "suggested_project": "Real-Time Chat & Notification Server with WebSockets"
    },
    "JavaScript": {
        "topics": ["Asynchronous JS (Promises, Event Loop)", "Modern Frontend Frameworks (React/Vue/Next.js)", "DOM & Browser APIs"],
        "recommended_courses": ["JavaScript.info Master Guide", "Full Stack Open (University of Helsinki)"],
        "suggested_project": "Full-Stack Collaborative Workspace with WebSockets"
    },
    "Data_Analysis": {
        "topics": ["Pandas Advanced Aggregations", "Exploratory Data Analysis (EDA)", "Missing Value Strategies & Outlier Treatment"],
        "recommended_courses": ["Kaggle Data Analysis Course", "IBM Data Analyst Professional Certificate"],
        "suggested_project": "Comprehensive Sales & Churn Analytics Notebook"
    },
    "Machine_Learning": {
        "topics": ["Ensemble Methods (Random Forest, Gradient Boosting)", "Hyperparameter Tuning (Optuna/GridSearchCV)", "Model Evaluation Metrics & Cross-Validation"],
        "recommended_courses": ["Coursera: Machine Learning Specialization by Andrew Ng", "StatQuest ML Series"],
        "suggested_project": "End-to-End Customer Lifetime Value Prediction Pipeline"
    },
    "Deep_Learning": {
        "topics": ["Convolutional Neural Networks (CNNs)", "Transformers & Attention Mechanisms", "Transfer Learning in PyTorch"],
        "recommended_courses": ["DeepLearning.AI Deep Learning Specialization", "Fast.ai Practical Deep Learning for Coders"],
        "suggested_project": "Multi-Modal Image Classification Pipeline in PyTorch"
    },
    "NLP": {
        "topics": ["Tokenization, Lemmatization & POS Tagging", "TF-IDF & Word Embeddings", "LLM Fine-Tuning & Prompt Engineering"],
        "recommended_courses": ["Hugging Face NLP Course", "Stanford CS224N: NLP with Deep Learning"],
        "suggested_project": "Automated Resume Matcher & Skill Extractor"
    },
    "Data_Visualization": {
        "topics": ["DAX Formulas & Power BI Modeling", "Interactive Storytelling in Tableau", "Custom Plotly / Seaborn Statistical Visuals"],
        "recommended_courses": ["Microsoft Certified: Power BI Data Analyst Associate", "Tableau Desktop Specialist Guide"],
        "suggested_project": "Executive KPI Financial Dashboard with Live Drill-Downs"
    },
    "Cloud_Computing": {
        "topics": ["AWS EC2, S3, IAM, Lambda", "Serverless Architecture", "VPC & Cloud Networking (AWS/Azure/GCP)"],
        "recommended_courses": ["AWS Certified Solutions Architect Associate", "Google Cloud Skills Boost"],
        "suggested_project": "Serverless Document Processing Architecture on AWS Lambda & S3"
    },
    "DevOps_CICD": {
        "topics": ["GitHub Actions & GitLab CI", "Infrastructure as Code (Terraform)", "Monitoring & Observability (Prometheus/Grafana)"],
        "recommended_courses": ["DevOps Bootcamp by Nana", "Coursera Continuous Integration & Continuous Delivery"],
        "suggested_project": "Automated CI/CD Pipeline Deploying Containerized App with Smoke Tests"
    },
    "Docker_Kubernetes": {
        "topics": ["Dockerfile Multi-Stage Builds", "Docker Compose Multi-Container Setup", "Kubernetes Pods, Deployments & Services"],
        "recommended_courses": ["Docker & Kubernetes: The Complete Guide", "Mumshad Mannambeth CKA Course"],
        "suggested_project": "Deploy a Scalable Microservices Architecture to Local K8s Cluster"
    },
    "Database_Design": {
        "topics": ["Database Normalization (1NF to 3NF)", "Indexing & Query Optimization", "MongoDB NoSQL Schema Design"],
        "recommended_courses": ["Stanford Online: Databases", "MongoDB University: Data Modeling"],
        "suggested_project": "Hybrid Relational-Document Database Architecture for Healthcare Records"
    },
    "Big_Data": {
        "topics": ["Apache Spark DataFrames & RDDs", "Distributed Computing Concepts", "Kafka Event Streaming"],
        "recommended_courses": ["Coursera: Big Data Analysis with Scala and Spark", "Databricks Certified Associate"],
        "suggested_project": "Real-Time Clickstream Streaming Analytics with Kafka & PySpark"
    },
    "Cybersecurity_Basics": {
        "topics": ["OWASP Top 10 Web Vulnerabilities", "Cryptography & SSL/TLS", "Network Security & Penetration Testing"],
        "recommended_courses": ["CompTIA Security+ Certification Guide", "TryHackMe Pre-Security Path"],
        "suggested_project": "Web Application Vulnerability Scanner & Audit Tool"
    },
    "Statistics_Math": {
        "topics": ["Probability Distributions", "Hypothesis Testing (p-values, ANOVA, Chi-Square)", "Linear Algebra (Eigenvectors, SVD)"],
        "recommended_courses": ["Khan Academy Statistics & Probability", "Imperial College Mathematics for Machine Learning"],
        "suggested_project": "A/B Testing Simulator with Power Analysis and Confidence Intervals"
    },
    "Problem_Solving": {
        "topics": ["Arrays, Hash Tables & Two Pointers", "Trees, Graphs & Dynamic Programming", "Time & Space Complexity (Big-O)"],
        "recommended_courses": ["NeetCode 150 Algorithms Roadmap", "Grokking the Coding Interview"],
        "suggested_project": "Solve 100 Curated Algorithmic Problems and Build an Interactive DSA Visualizer"
    },
    "Communication": {
        "topics": ["Technical Writing & Documentation", "Executive Presentation Skills", "Cross-Functional Collaboration"],
        "recommended_courses": ["Google Technical Writing Course", "Toastmasters Public Speaking Essentials"],
        "suggested_project": "Author an In-Depth Technical Architecture Whitepaper & Video Walkthrough"
    }
}

# Random Seed for Reproducibility
RANDOM_STATE = 42
TEST_SIZE = 0.2
