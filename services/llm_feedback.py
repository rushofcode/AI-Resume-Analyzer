# services/llm_feedback.py
"""
LLM-based feedback generation using the Anthropic Claude API.

The prompt is structured so the model returns sections delimited by
XML-style tags, making it easy to parse and display in the UI.
"""

from __future__ import annotations
import logging
import os
import re
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Prompt engineering ────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are ResumeCoach, an expert career advisor and hiring manager with 15+ years of experience across tech, finance, and consulting. You provide honest, structured, and actionable resume feedback.

Always respond in the exact XML-tagged format requested. Be specific — avoid generic advice. Reference actual content from the resume and job description."""

def _build_user_prompt(
    resume_text: str,
    jd_text: str,
    match_score: float,
    present_skills: List[str],
    missing_skills: List[str],
) -> str:
    # Truncate to avoid excessive token usage
    resume_excerpt = resume_text[:3000]
    jd_excerpt = jd_text[:2000]
    present_str = ", ".join(present_skills[:20]) if present_skills else "None detected"
    missing_str = ", ".join(missing_skills[:20]) if missing_skills else "None"

    return f"""Analyze this resume against the job description and provide structured feedback.

**MATCH SCORE**: {match_score:.1f}/100
**SKILLS PRESENT**: {present_str}
**SKILLS MISSING**: {missing_str}

---
**RESUME**:
{resume_excerpt}

---
**JOB DESCRIPTION**:
{jd_excerpt}

---
Please respond in this EXACT format with XML tags:

<overall_assessment>
2-3 sentences on the overall alignment and candidacy strength.
</overall_assessment>

<strengths>
• [Strength 1 with specific evidence from the resume]
• [Strength 2 with specific evidence]
• [Strength 3 with specific evidence]
</strengths>

<gaps>
• [Gap 1: what's missing and why it matters for this role]
• [Gap 2: what's missing and why it matters]
• [Gap 3: what's missing and why it matters]
</gaps>

<quick_wins>
• [Specific, actionable improvement #1 — something doable in under 1 hour]
• [Specific, actionable improvement #2]
• [Specific, actionable improvement #3]
</quick_wins>

<long_term_recommendations>
• [Strategic recommendation #1 — skill to develop, certification to get, etc.]
• [Strategic recommendation #2]
</long_term_recommendations>

<keywords_to_add>
List 5-8 exact keywords/phrases from the job description that should appear in the resume but currently don't.
</keywords_to_add>

<interview_talking_points>
• [A talking point that bridges resume experience to a key JD requirement]
• [Another talking point]
</interview_talking_points>"""


# ── Feedback retrieval ────────────────────────────────────────────────────────

def get_ai_feedback(
    resume_text: str,
    jd_text: str,
    match_score: float,
    present_skills: List[str],
    missing_skills: List[str],
    api_key: Optional[str] = None,
) -> str:
    """
    Call Claude API and return raw feedback string.
    Raises on API errors so the caller can handle gracefully.
    """
    import anthropic

    key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        raise ValueError("No Anthropic API key provided.")

    client = anthropic.Anthropic(api_key=api_key)
    user_prompt = _build_user_prompt(
        resume_text, jd_text, match_score, present_skills, missing_skills
    )

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1500,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return message.content[0].text


# ── Section parser ────────────────────────────────────────────────────────────

_TAG_PATTERN = re.compile(r"<(\w+)>(.*?)</\1>", re.DOTALL)

def parse_feedback_sections(raw_feedback: str) -> Dict[str, str]:
    """
    Parse XML-tagged sections from the LLM response.

    Returns a dict like:
    {
        "overall_assessment": "...",
        "strengths": "...",
        ...
    }
    """
    sections = {}
    for match in _TAG_PATTERN.finditer(raw_feedback):
        tag = match.group(1)
        content = match.group(2).strip()
        sections[tag] = content

    # Fallback: return the whole text under a single key
    if not sections:
        sections["full_feedback"] = raw_feedback

    return sections


def format_section_as_bullets(text: str) -> List[str]:
    """Convert bullet-point text block to a clean list."""
    lines = [line.strip().lstrip("•-*").strip() for line in text.splitlines()]
    return [l for l in lines if l]
