# services/analyzer.py
"""
ResumeAnalyzer — the central orchestration service.

Data flow:
  PDF bytes / raw text
       │
       ▼
  extract_text_from_pdf()
       │
       ├──► TFIDFSimilarityModel.compute_similarity()  ─┐
       │                                                 ├─► combined_score()
       ├──► BERTSimilarityModel.compute_similarity()   ─┘
       │
       ├──► extract_skills(resume) + extract_skills(jd)
       │         └──► categorize_skills()
       │
       └──► get_ai_feedback()  (LLM call)

All results are returned as a single AnalysisResult dataclass.
"""

from __future__ import annotations
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from models.tfidf_model import TFIDFSimilarityModel
from models.bert_model import BERTSimilarityModel
from utils.skill_extractor import extract_skills, categorize_skills
from utils.text_preprocessor import extract_keywords

logger = logging.getLogger(__name__)


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class AnalysisResult:
    # Raw inputs
    resume_text: str = ""
    jd_text: str = ""

    # Similarity scores (0–100)
    tfidf_score: float = 0.0
    bert_score: float = 0.0
    combined_score: float = 0.0

    # Score breakdown details
    tfidf_breakdown: Dict = field(default_factory=dict)
    bert_breakdown: Dict = field(default_factory=dict)

    # Skill analysis
    resume_skills: Dict = field(default_factory=dict)
    jd_skills: Dict = field(default_factory=dict)
    skill_categorization: Dict = field(default_factory=dict)

    # Keywords
    resume_keywords: List[str] = field(default_factory=list)
    jd_keywords: List[str] = field(default_factory=list)

    # LLM feedback
    ai_feedback: str = ""
    feedback_sections: Dict = field(default_factory=dict)

    # Meta
    processing_time_sec: float = 0.0
    bert_available: bool = False
    errors: List[str] = field(default_factory=list)


# ── Orchestrator ──────────────────────────────────────────────────────────────

class ResumeAnalyzer:
    """
    Orchestrates all analysis steps and returns a populated AnalysisResult.

    Weight split between TF-IDF and BERT:
    - When BERT is available: 40% TF-IDF + 60% BERT (semantic > lexical).
    - When BERT unavailable: 100% TF-IDF.
    """

    TFIDF_WEIGHT = 0.40
    BERT_WEIGHT = 0.60

    def __init__(self):
        self.tfidf_model = TFIDFSimilarityModel()
        self.bert_model = BERTSimilarityModel()

    def analyze(
        self,
        resume_text: str,
        jd_text: str,
        include_ai_feedback: bool = True,
        api_key: Optional[str] = None,
    ) -> AnalysisResult:
        """
        Run the full analysis pipeline.

        Args:
            resume_text:        Plain text extracted from the resume PDF.
            jd_text:            Job description pasted by the user.
            include_ai_feedback: Whether to call the LLM for feedback.
            api_key:            Anthropic API key (optional; falls back to env var).

        Returns:
            Fully populated AnalysisResult.
        """
        result = AnalysisResult(resume_text=resume_text, jd_text=jd_text)
        t0 = time.perf_counter()

        # ── 1. Similarity scores ──────────────────────────────────────────────
        logger.info("Computing TF-IDF similarity…")
        try:
            tfidf_bd = self.tfidf_model.get_score_breakdown(resume_text, jd_text)
            result.tfidf_score = tfidf_bd["score"]
            result.tfidf_breakdown = tfidf_bd
        except Exception as exc:
            result.errors.append(f"TF-IDF error: {exc}")
            logger.exception("TF-IDF failed.")

        logger.info("Computing BERT similarity…")
        result.bert_available = self.bert_model.is_available()
        try:
            bert_bd = self.bert_model.get_score_breakdown(resume_text, jd_text)
            result.bert_score = bert_bd["score"]
            result.bert_breakdown = bert_bd
        except Exception as exc:
            result.errors.append(f"BERT error: {exc}")
            logger.warning("BERT failed: %s", exc)

        # ── 2. Combined score ─────────────────────────────────────────────────
        if result.bert_available and result.bert_score > 0:
            result.combined_score = round(
                self.TFIDF_WEIGHT * result.tfidf_score
                + self.BERT_WEIGHT * result.bert_score,
                1,
            )
        else:
            result.combined_score = result.tfidf_score

        # ── 3. Skill extraction ───────────────────────────────────────────────
        logger.info("Extracting skills…")
        try:
            result.resume_skills = extract_skills(resume_text)
            result.jd_skills = extract_skills(jd_text)
            result.skill_categorization = categorize_skills(
                result.resume_skills, result.jd_skills
            )
        except Exception as exc:
            result.errors.append(f"Skill extraction error: {exc}")
            logger.exception("Skill extraction failed.")

        # ── 4. Keyword extraction ─────────────────────────────────────────────
        try:
            result.resume_keywords = extract_keywords(resume_text, top_n=20)
            result.jd_keywords = extract_keywords(jd_text, top_n=20)
        except Exception as exc:
            result.errors.append(f"Keyword extraction error: {exc}")

        # ── 5. AI feedback (LLM) ──────────────────────────────────────────────
        if include_ai_feedback:
            logger.info("Requesting AI feedback…")
            try:
                from services.llm_feedback import get_ai_feedback, parse_feedback_sections
                raw_feedback = get_ai_feedback(
                    resume_text=resume_text,
                    jd_text=jd_text,
                    match_score=result.combined_score,
                    missing_skills=result.skill_categorization.get("missing", []),
                    present_skills=result.skill_categorization.get("present", []),
                    api_key=api_key,
                )
                result.ai_feedback = raw_feedback
                result.feedback_sections = parse_feedback_sections(raw_feedback)
            except Exception as exc:
                result.errors.append(f"AI feedback error: {exc}")
                result.ai_feedback = "AI feedback unavailable. Please check your API key."
                logger.warning("LLM feedback failed: %s", exc)

        result.processing_time_sec = round(time.perf_counter() - t0, 2)
        logger.info("Analysis complete in %.2fs.", result.processing_time_sec)
        return result

    def get_score_label(self, score: float) -> tuple[str, str]:
        """Return (label, colour) pair for a 0-100 score."""
        if score >= 80:
            return "Excellent Match", "#00C851"
        elif score >= 65:
            return "Good Match", "#ffbb33"
        elif score >= 45:
            return "Moderate Match", "#ff8800"
        else:
            return "Low Match", "#ff4444"
