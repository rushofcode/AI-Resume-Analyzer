# models/bert_model.py
"""
BERT-based semantic similarity model using Sentence Transformers.

Why BERT / Sentence Transformers?
- Captures semantic meaning, not just keyword overlap.
- "Software Engineer" ≈ "Developer" even without shared tokens.
- all-MiniLM-L6-v2 is a great balance of speed (6-layer) and quality.

Design decisions:
- We chunk long texts into 512-token segments, embed each chunk, then
  average the embeddings. This handles multi-page resumes gracefully.
- Cosine similarity on normalised embeddings is numerically stable.
- Model is cached at the module level to avoid reload overhead.
"""

from __future__ import annotations
import logging
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)

# Module-level cache — loaded once per Streamlit session
_MODEL_CACHE: dict = {}


def _get_model(model_name: str = "all-MiniLM-L6-v2"):
    """Load (or return cached) SentenceTransformer model."""
    if model_name not in _MODEL_CACHE:
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading SentenceTransformer model '%s'…", model_name)
            _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
            logger.info("Model loaded.")
        except Exception as exc:
            logger.error("Failed to load SentenceTransformer: %s", exc)
            _MODEL_CACHE[model_name] = None
    return _MODEL_CACHE[model_name]


def _chunk_text(text: str, max_chars: int = 2000) -> List[str]:
    """
    Split text into overlapping character chunks.
    SentenceTransformer has a 512-token limit; ~2000 chars ≈ 400 tokens (safe).
    """
    if len(text) <= max_chars:
        return [text]
    chunks = []
    step = max_chars - 200  # 200-char overlap for context continuity
    for i in range(0, len(text), step):
        chunk = text[i: i + max_chars]
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def _embed_text(model, text: str) -> np.ndarray:
    """Embed text, averaging over chunks for long documents."""
    chunks = _chunk_text(text)
    embeddings = model.encode(chunks, normalize_embeddings=True, show_progress_bar=False)
    return np.mean(embeddings, axis=0)  # average pooling over chunks


class BERTSimilarityModel:
    """
    Compute semantic similarity between resume and job description
    using a pre-trained Sentence Transformer.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name

    @property
    def model(self):
        return _get_model(self.model_name)

    def is_available(self) -> bool:
        """Returns False if the model couldn't be loaded (e.g. no internet on first run)."""
        return self.model is not None

    def compute_similarity(self, resume_text: str, jd_text: str) -> float:
        """
        Return cosine similarity in [0, 1].
        Falls back to 0.0 if model unavailable.
        """
        if not self.is_available():
            logger.warning("BERT model unavailable; returning 0.")
            return 0.0
        if not resume_text.strip() or not jd_text.strip():
            return 0.0
        try:
            resume_emb = _embed_text(self.model, resume_text)
            jd_emb = _embed_text(self.model, jd_text)
            # Both are already L2-normalised → dot product == cosine similarity
            score = float(np.dot(resume_emb, jd_emb))
            return float(np.clip(score, 0.0, 1.0))
        except Exception as exc:
            logger.error("BERT similarity computation failed: %s", exc)
            return 0.0

    def get_score_breakdown(self, resume_text: str, jd_text: str) -> Dict:
        """Return structured breakdown for UI display."""
        score = self.compute_similarity(resume_text, jd_text)
        return {
            "score": round(score * 100, 1),
            "model": self.model_name,
            "available": self.is_available(),
            "method": "BERT Semantic Similarity (Sentence Transformers)",
        }
