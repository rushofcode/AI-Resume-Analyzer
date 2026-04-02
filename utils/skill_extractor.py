# utils/skill_extractor.py
"""
Skill extraction and categorisation.

Strategy:
- Match against a curated taxonomy of ~300 tech + soft skills.
- Use regex with word-boundary matching so "Java" doesn't match "JavaScript".
- Also pull bigrams/trigrams that look like compound skills (e.g. "machine learning").
"""

import re
from typing import Dict, List, Set, Tuple

# ── Skill Taxonomy ────────────────────────────────────────────────────────────
# Organised by category for the radar chart breakdown.

SKILL_TAXONOMY: Dict[str, List[str]] = {
    "Programming Languages": [
        "python", "java", "javascript", "typescript", "c\\+\\+", "c#", "ruby",
        "go", "golang", "rust", "swift", "kotlin", "scala", "r", "matlab",
        "perl", "php", "bash", "shell", "powershell", "haskell", "elixir",
        "clojure", "dart", "lua", "groovy", "fortran", "cobol",
    ],
    "Web & Frontend": [
        "react", "angular", "vue", "next\\.js", "nuxt", "svelte", "html",
        "css", "sass", "tailwind", "bootstrap", "jquery", "webpack", "vite",
        "graphql", "rest", "restful", "api", "oauth", "websocket",
    ],
    "Backend & Frameworks": [
        "node\\.js", "express", "fastapi", "flask", "django", "spring",
        "spring boot", "rails", "laravel", "asp\\.net", "nest\\.js",
        "fastify", "fiber", "gin", "echo",
    ],
    "Data & ML": [
        "machine learning", "deep learning", "nlp", "natural language processing",
        "computer vision", "tensorflow", "pytorch", "keras", "scikit-learn",
        "pandas", "numpy", "scipy", "matplotlib", "seaborn", "plotly",
        "hugging face", "transformers", "bert", "gpt", "llm",
        "data analysis", "data science", "feature engineering",
        "regression", "classification", "clustering", "neural network",
        "reinforcement learning", "xgboost", "lightgbm",
    ],
    "Databases": [
        "sql", "mysql", "postgresql", "mongodb", "redis", "cassandra",
        "elasticsearch", "dynamodb", "sqlite", "oracle", "mssql",
        "neo4j", "firebase", "supabase", "pinecone", "vector database",
    ],
    "Cloud & DevOps": [
        "aws", "azure", "gcp", "google cloud", "docker", "kubernetes",
        "terraform", "ansible", "jenkins", "github actions", "ci/cd",
        "devops", "helm", "prometheus", "grafana", "datadog", "linux",
        "nginx", "apache", "serverless", "lambda", "cloudformation",
    ],
    "Tools & Practices": [
        "git", "github", "gitlab", "jira", "agile", "scrum", "kanban",
        "tdd", "bdd", "unit testing", "pytest", "jest", "postman",
        "swagger", "openapi", "microservices", "event-driven",
        "design patterns", "solid", "clean code", "code review",
    ],
    "Soft Skills": [
        "leadership", "communication", "collaboration", "problem solving",
        "critical thinking", "project management", "mentoring", "coaching",
        "presentation", "stakeholder management", "time management",
        "analytical", "strategic thinking", "creativity", "adaptability",
    ],
}

# Flatten for fast lookup
_ALL_SKILLS_FLAT: List[Tuple[str, str]] = []
for category, skills in SKILL_TAXONOMY.items():
    for skill in skills:
        _ALL_SKILLS_FLAT.append((category, skill))


def _compile_pattern(skill: str) -> re.Pattern:
    """Build a case-insensitive whole-word regex for a skill string."""
    return re.compile(rf"\b{skill}\b", re.IGNORECASE)


# Pre-compile all patterns once at module load
_COMPILED_PATTERNS: List[Tuple[str, str, re.Pattern]] = [
    (cat, skill, _compile_pattern(skill))
    for cat, skill in _ALL_SKILLS_FLAT
]


def extract_skills(text: str) -> Dict[str, Set[str]]:
    """
    Scan text and return matched skills grouped by category.

    Returns:
        { "Programming Languages": {"python", "java"}, "Data & ML": {...}, ... }
    """
    found: Dict[str, Set[str]] = {cat: set() for cat in SKILL_TAXONOMY}

    for category, skill, pattern in _COMPILED_PATTERNS:
        # Use the human-readable skill name (de-escaped from regex form)
        readable = re.sub(r"\\([+.*?()|])", r"\1", skill)
        if pattern.search(text):
            found[category].add(readable)

    # Remove empty categories
    return {k: v for k, v in found.items() if v}


def categorize_skills(
    resume_skills: Dict[str, Set[str]],
    jd_skills: Dict[str, Set[str]],
) -> Dict[str, Dict]:
    """
    Compare resume skills against job description skills.

    Returns a dict with:
    - present:  skills required by JD that the resume has
    - missing:  skills required by JD that the resume lacks
    - extra:    skills on resume not explicitly required by JD
    - by_category: per-category breakdown for radar chart
    """
    all_categories = set(resume_skills) | set(jd_skills)

    present: Set[str] = set()
    missing: Set[str] = set()
    extra: Set[str] = set()
    by_category: Dict[str, Dict] = {}

    for cat in all_categories:
        r = resume_skills.get(cat, set())
        j = jd_skills.get(cat, set())

        cat_present = r & j
        cat_missing = j - r
        cat_extra = r - j

        present |= cat_present
        missing |= cat_missing
        extra |= cat_extra

        if j:  # only track categories the JD cares about
            by_category[cat] = {
                "required": len(j),
                "matched": len(cat_present),
                "missing": sorted(cat_missing),
                "present": sorted(cat_present),
                "coverage": round(len(cat_present) / len(j) * 100) if j else 0,
            }

    return {
        "present": sorted(present),
        "missing": sorted(missing),
        "extra": sorted(extra),
        "by_category": by_category,
    }
