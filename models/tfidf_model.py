# models/tfidf_model.py
"""
TF-IDF + Cosine Similarity model for resume-JD matching.

Why TF-IDF?
- Fast and interpretable — no GPU required.
- Excellent for keyword-heavy domains like job matching.
- Provides a baseline that explains which exact terms drive the score.

Design:
- We fit the vectoriser on BOTH documents together so IDF weights reflect
  the combined vocabulary, which avoids the cold-start problem.
- sublinear_tf=True dampens the effect of high-frequency terms.
- ngram_range=(1,2) captures compound terms ("machine learning", "project management").
"""

from __future__ import annotations
import logging
from typing import Dict, List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils.text_preprocessor import preprocess_text

logger = logging.getLogger(__name__)


class TFIDFSimilarityModel:
    """Compute cosine similarity between two texts using TF-IDF vectors."""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            max_features=5000,
            min_df=1,
        )

    def compute_similarity(self, resume_text: str, jd_text: str) -> float:
        """
        Return similarity score in [0, 1].

        Steps:
        1. Preprocess both texts.
        2. Fit vectoriser on their union.
        3. Compute cosine similarity between their TF-IDF vectors.
        """
        resume_clean = preprocess_text(resume_text)
        jd_clean = preprocess_text(jd_text)

        if not resume_clean.strip() or not jd_clean.strip():
            logger.warning("One or both texts are empty after preprocessing.")
            return 0.0

        try:
            tfidf_matrix = self.vectorizer.fit_transform([resume_clean, jd_clean])
            score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(np.clip(score, 0.0, 1.0))
        except Exception as exc:
            logger.error("TF-IDF similarity failed: %s", exc)
            return 0.0

    def get_top_matching_terms(
        self, resume_text: str, jd_text: str, top_n: int = 15
    ) -> List[Tuple[str, float]]:
        """
        Return the top-N terms that contribute most to the similarity.

        Useful for explainability: "these terms are why your score is X."
        """
        resume_clean = preprocess_text(resume_text)
        jd_clean = preprocess_text(jd_text)

        try:
            tfidf_matrix = self.vectorizer.fit_transform([resume_clean, jd_clean])
            feature_names = self.vectorizer.get_feature_names_out()

            resume_vec = np.asarray(tfidf_matrix[0].todense()).flatten()
            jd_vec = np.asarray(tfidf_matrix[1].todense()).flatten()

            # Term contributes when it's present in BOTH documents
            overlap = np.minimum(resume_vec, jd_vec)
            top_indices = overlap.argsort()[::-1][:top_n]

            return [
                (feature_names[i], round(float(overlap[i]), 4))
                for i in top_indices
                if overlap[i] > 0
            ]
        except Exception as exc:
            logger.error("get_top_matching_terms failed: %s", exc)
            return []

    def get_score_breakdown(self, resume_text: str, jd_text: str) -> Dict:
        """Return a full breakdown dict for display in the UI."""
        score = self.compute_similarity(resume_text, jd_text)
        top_terms = self.get_top_matching_terms(resume_text, jd_text)
        return {
            "score": round(score * 100, 1),
            "top_matching_terms": top_terms,
            "method": "TF-IDF + Cosine Similarity",
        }
