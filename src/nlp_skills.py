"""
NLP Skills Extraction & Job Description Processing Module.
Methodology:
- Deterministic dictionary-based entity matching for recognized tech taxonomies
- Robust whitespace and delimiter padding handling punctuation-heavy terms:
  C++, C#, CI/CD, .NET, Node.js, Power BI, AWS, Azure, GCP
- Stopword filtering and token normalization
- TF-IDF keyword salience scoring across corpus
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import re
import string
import logging
from typing import List, Dict, Set, Tuple, Any
from collections import Counter
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

from config.config import (
    CORE_SKILLS,
    SKILL_SYNONYMS,
    SKILL_DISPLAY_NAMES
)

BASIC_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "can", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does",
    "doesn't", "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had",
    "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her",
    "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd",
    "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself",
    "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off",
    "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over",
    "own", "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so",
    "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've",
    "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't",
    "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when",
    "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's",
    "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your",
    "yours", "yourself", "yourselves", "role", "work", "experience", "candidate", "responsibilities",
    "requirements", "seeking", "looking", "team", "strong", "ability", "must", "years", "skills"
}


def normalize_text_for_entity_matching(text: str) -> str:
    """
    Standardizes text for robust dictionary entity extraction:
    - Lowercases text
    - Replaces punctuation delimiters (commas, semicolons, brackets, quotes, slashes) with spaces
      while preserving compound technical tokens like c++, c#, ci/cd, .net, node.js
    - Normalizes multiple spaces into a single space
    - Padds with leading and trailing spaces for clean boundary isolation
    """
    if not isinstance(text, str):
        return ""
        
    text = text.lower()
    # Normalize slashes when used as delimiters (e.g. "DevOps/CI/CD" -> "DevOps CI/CD")
    # But preserve "ci/cd"
    text = text.replace("ci/cd", "__cicd__")
    text = text.replace("node.js", "__nodejs__")
    text = text.replace(".net", "__dotnet__")
    text = text.replace("c++", "__cpp__")
    text = text.replace("c#", "__csharp__")
    
    # Replace standard punctuation with spaces
    text = re.sub(r"[,;:\(\)\[\]\{\}\"\'!/?\\|><~`*&]", " ", text)
    
    # Restore protected tokens with standard normalized representation
    text = text.replace("__cicd__", "ci/cd")
    text = text.replace("__nodejs__", "node.js")
    text = text.replace("__dotnet__", ".net")
    text = text.replace("__cpp__", "c++")
    text = text.replace("__csharp__", "c#")
    
    # Collapse multiple whitespaces and pad
    text = " " + " ".join(text.split()) + " "
    return text


def clean_text_for_tokens(text: str) -> str:
    """
    Standardizes text for general bag-of-words tokenization.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"[/\-_,;:\(\)\[\]\{\}]", " ", text)
    text = "".join([c for c in text if c.isalnum() or c.isspace()])
    return " ".join(text.split())


def tokenize_text(text: str) -> List[str]:
    """
    Tokenizes text into words, filtering out common stopwords.
    """
    cleaned = clean_text_for_tokens(text)
    tokens = [w for w in cleaned.split() if w not in BASIC_STOPWORDS and len(w) > 1]
    return tokens


def extract_skills_from_text(text: str) -> Dict[str, Any]:
    """
    Extracts recognized technical skills from job descriptions or resumes using
    dictionary-based entity matching with safe boundary checks.
    
    Returns:
        Dict containing:
        - 'extracted_skills': List of canonical skill names
        - 'skill_frequencies': Dict[skill_name, count]
        - 'skill_requirements_estimated': Dict[skill_name, estimated_score_0_100]
        - 'display_names': Dict[skill_name, display_name]
    """
    if not text or not isinstance(text, str):
        return {
            "extracted_skills": [],
            "skill_frequencies": {},
            "skill_requirements_estimated": {},
            "display_names": {}
        }
        
    working_text = normalize_text_for_entity_matching(text)
    matched_skills = Counter()
    
    # Sort synonyms by length descending to match longest multi-word phrases first
    sorted_synonyms = sorted(SKILL_SYNONYMS.keys(), key=lambda x: len(x), reverse=True)
    
    for phrase in sorted_synonyms:
        canonical_skill = SKILL_SYNONYMS[phrase]
        target_token = f" {phrase.lower()} "
        
        if target_token in working_text:
            count = working_text.count(target_token)
            matched_skills[canonical_skill] += count
            # Mask out matched phrase with spaces to prevent double counting
            working_text = working_text.replace(target_token, " ")
            
    # Estimate baseline proficiency requirements based on mention frequency
    skill_confidence = {}
    for skill, count in matched_skills.items():
        # Baseline score 70 for 1 mention, +10 for repeated mentions, capped at 95
        est_req = min(95, 70 + (count - 1) * 10)
        skill_confidence[skill] = est_req
        
    return {
        "extracted_skills": list(matched_skills.keys()),
        "skill_frequencies": dict(matched_skills),
        "skill_requirements_estimated": skill_confidence,
        "display_names": {k: SKILL_DISPLAY_NAMES.get(k, k) for k in matched_skills.keys()}
    }


def compute_tfidf_skill_importance(job_descriptions: List[str]) -> pd.DataFrame:
    """
    Computes TF-IDF weights across a collection of job postings to identify
    distinctive keywords.
    """
    if not job_descriptions:
        return pd.DataFrame(columns=["Keyword", "TFIDF_Importance"])
        
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=50)
    tfidf_matrix = vectorizer.fit_transform(job_descriptions)
    feature_names = vectorizer.get_feature_names_out()
    
    avg_scores = tfidf_matrix.mean(axis=0).A1
    df_tfidf = pd.DataFrame({
        "Keyword": feature_names,
        "TFIDF_Importance": avg_scores
    }).sort_values(by="TFIDF_Importance", ascending=False)
    
    return df_tfidf


if __name__ == "__main__":
    test_samples = [
        "Looking for a C++ developer with CI/CD, AWS, and Power BI skills.",
        "We need a C# and .NET developer with Node.js and Azure experience.",
        "Machine Learning Engineer proficient in Python, SQL, and GCP cloud."
    ]
    for sample in test_samples:
        res = extract_skills_from_text(sample)
        print(f"\nText: {sample}")
        print(f"Extracted: {res['extracted_skills']}")
