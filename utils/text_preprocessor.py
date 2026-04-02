# utils/text_preprocessor.py
"""
Text preprocessing utilities for NLP pipelines.
Handles tokenisation, stop-word removal, lemmatisation, and keyword extraction.
"""

import re
import string
import logging
from typing import List
from functools import lru_cache

import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

# ── NLTK one-time downloads ──────────────────────────────────────────────────
_NLTK_RESOURCES = ["punkt", "stopwords", "wordnet", "averaged_perceptron_tagger"]

def _ensure_nltk_resources():
    for resource in _NLTK_RESOURCES:
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            try:
                nltk.download(resource, quiet=True)
            except Exception as exc:
                logger.warning("Could not download NLTK resource '%s': %s", resource, exc)

_ensure_nltk_resources()

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

_STOP_WORDS = set(stopwords.words("english"))
_LEMMATIZER = WordNetLemmatizer()

# Extra noise words common in resumes but useless for matching
_EXTRA_STOP = {
    "experience", "work", "worked", "years", "year", "team", "company",
    "responsible", "including", "using", "used", "etc", "also", "well",
    "good", "strong", "ability", "knowledge", "skills", "skill",
    "seeking", "position", "opportunity", "highly", "results", "driven",
    "oriented", "proven", "track", "record", "excellent", "communication",
}
_STOP_WORDS |= _EXTRA_STOP


def clean_text(text: str) -> str:
    """
    Basic cleaning: lowercase, strip URLs/emails/special chars, collapse whitespace.
    """
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)           # remove URLs
    text = re.sub(r"\S+@\S+\.\S+", "", text)             # remove emails
    text = re.sub(r"\d{10,}", "", text)                   # remove phone numbers
    text = re.sub(r"[^\w\s\+#\.]", " ", text)            # keep + # . for tech terms
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_text(text: str, lemmatize: bool = True) -> str:
    """
    Full preprocessing pipeline:
    clean → tokenise → stop-word filter → optional lemmatisation → rejoin.

    Returns a single space-joined string suitable for TF-IDF vectorisation.
    """
    text = clean_text(text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in string.punctuation]
    tokens = [t for t in tokens if t not in _STOP_WORDS and len(t) > 1]
    if lemmatize:
        tokens = [_LEMMATIZER.lemmatize(t) for t in tokens]
    return " ".join(tokens)


def extract_keywords(text: str, top_n: int = 30) -> List[str]:
    """
    Extract the most significant keywords from text using TF-IDF on a
    single-document corpus (sublinear_tf ensures reasonable scoring).

    Returns a list of keywords sorted by TF-IDF score (descending).
    """
    processed = preprocess_text(text)
    if not processed.strip():
        return []

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),      # unigrams + bigrams to capture compound skills
        sublinear_tf=True,
        max_features=500,
    )
    try:
        tfidf_matrix = vectorizer.fit_transform([processed])
        scores = dict(zip(vectorizer.get_feature_names_out(),
                          tfidf_matrix.toarray()[0]))
        sorted_kw = sorted(scores, key=scores.get, reverse=True)
        return sorted_kw[:top_n]
    except ValueError:
        return []
