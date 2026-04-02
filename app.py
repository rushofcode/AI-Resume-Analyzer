# app.py
"""
AI Resume Analyzer — Streamlit Application
==========================================

Entry point for the application. Run with:
    streamlit run app.py

Architecture:
    app.py (UI layer)
      └── services/analyzer.py (orchestration)
            ├── models/tfidf_model.py  (TF-IDF similarity)
            ├── models/bert_model.py   (BERT semantic similarity)
            ├── utils/skill_extractor.py
            ├── utils/text_preprocessor.py
            ├── services/llm_feedback.py (Claude AI feedback)
            └── services/report_generator.py (PDF download)
"""

import io
import os
import sys
import logging

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ── Path setup ──────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

logging.basicConfig(level=logging.INFO)

# ── Streamlit page config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ─── Global ─────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

.main { background: #0a0f1e; }
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1200px; }

/* ─── Header ─────────────────────────────────────────── */
.hero-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle at 30% 30%, rgba(99,102,241,0.15) 0%, transparent 50%),
                radial-gradient(circle at 70% 70%, rgba(168,85,247,0.1) 0%, transparent 50%);
    pointer-events: none;
}
.hero-title {
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, #818cf8, #c4b5fd, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    line-height: 1.1;
}
.hero-subtitle {
    color: #94a3b8;
    font-size: 1.05rem;
    margin-top: 0.5rem;
    font-weight: 400;
}

/* ─── Score Cards ────────────────────────────────────── */
.score-card {
    background: #0f172a;
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    transition: border-color 0.2s;
}
.score-card:hover { border-color: rgba(99,102,241,0.6); }
.score-value {
    font-size: 3rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    line-height: 1;
}
.score-label {
    color: #94a3b8;
    font-size: 0.85rem;
    margin-top: 0.4rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* ─── Skill Tags ─────────────────────────────────────── */
.skill-tag-present {
    display: inline-block;
    background: rgba(34,197,94,0.12);
    border: 1px solid rgba(34,197,94,0.35);
    color: #4ade80;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.8rem;
    margin: 3px;
    font-family: 'JetBrains Mono', monospace;
}
.skill-tag-missing {
    display: inline-block;
    background: rgba(239,68,68,0.1);
    border: 1px solid rgba(239,68,68,0.35);
    color: #f87171;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.8rem;
    margin: 3px;
    font-family: 'JetBrains Mono', monospace;
}
.skill-tag-extra {
    display: inline-block;
    background: rgba(99,102,241,0.1);
    border: 1px solid rgba(99,102,241,0.35);
    color: #a5b4fc;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.8rem;
    margin: 3px;
    font-family: 'JetBrains Mono', monospace;
}

/* ─── Section Headers ────────────────────────────────── */
.section-header {
    color: #e2e8f0;
    font-size: 1.2rem;
    font-weight: 600;
    border-bottom: 1px solid rgba(99,102,241,0.3);
    padding-bottom: 0.5rem;
    margin-bottom: 1rem;
}

/* ─── Feedback Cards ─────────────────────────────────── */
.feedback-card {
    background: #0f172a;
    border-left: 3px solid #6366f1;
    border-radius: 0 12px 12px 0;
    padding: 1rem 1.5rem;
    margin-bottom: 1rem;
}
.feedback-card.strengths { border-left-color: #22c55e; }
.feedback-card.gaps { border-left-color: #f87171; }
.feedback-card.wins { border-left-color: #fbbf24; }
.feedback-card.strategy { border-left-color: #818cf8; }
.feedback-card-title {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 600;
    margin-bottom: 0.5rem;
    color: #94a3b8;
}
.feedback-card-content {
    color: #cbd5e1;
    font-size: 0.92rem;
    line-height: 1.7;
    white-space: pre-line;
}

/* ─── Match Badge ────────────────────────────────────── */
.match-badge {
    display: inline-block;
    padding: 6px 18px;
    border-radius: 999px;
    font-size: 0.85rem;
    font-weight: 600;
    margin-top: 0.5rem;
}

/* ─── Sidebar ────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #0a0f1e;
    border-right: 1px solid rgba(99,102,241,0.2);
}
[data-testid="stSidebar"] .stMarkdown { color: #94a3b8; }

/* ─── Buttons ────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.6rem 1.8rem;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    font-size: 0.95rem;
    transition: all 0.2s;
    width: 100%;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 8px 25px rgba(99,102,241,0.4);
}

/* ─── File uploader ──────────────────────────────────── */
[data-testid="stFileUploadDropzone"] {
    background: rgba(99,102,241,0.05);
    border: 2px dashed rgba(99,102,241,0.4);
    border-radius: 12px;
}

/* ─── Text areas ─────────────────────────────────────── */
textarea { font-family: 'Space Grotesk', sans-serif !important; }

/* ─── Divider ────────────────────────────────────────── */
hr { border-color: rgba(99,102,241,0.2); }

/* ─── Spinner ─────────────────────────────────────────── */
.stSpinner { color: #6366f1; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ──────────────────────────────────────────────────────────────────

def score_color(score: float) -> str:
    if score >= 80: return "#22c55e"
    elif score >= 65: return "#fbbf24"
    elif score >= 45: return "#f97316"
    else: return "#f87171"

def score_label(score: float) -> str:
    if score >= 80: return "Excellent Match 🚀"
    elif score >= 65: return "Good Match ✅"
    elif score >= 45: return "Moderate Match ⚠️"
    else: return "Low Match ❌"


@st.cache_resource(show_spinner=False)
def get_analyzer():
    """Cache the ResumeAnalyzer instance across Streamlit reruns."""
    from services.analyzer import ResumeAnalyzer
    return ResumeAnalyzer()


def render_skill_tags(skills: list, tag_class: str) -> str:
    return "".join(f'<span class="skill-tag-{tag_class}">{s}</span>' for s in skills)


def build_radar_chart(by_category: dict) -> go.Figure:
    cats = list(by_category.keys())
    coverage = [by_category[c]["coverage"] for c in cats]
    cats_closed = cats + [cats[0]]
    coverage_closed = coverage + [coverage[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=coverage_closed,
        theta=cats_closed,
        fill="toself",
        fillcolor="rgba(99,102,241,0.2)",
        line=dict(color="#6366f1", width=2),
        name="Coverage %",
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                color="#475569",
                gridcolor="rgba(99,102,241,0.2)",
                tickfont=dict(color="#94a3b8", size=10),
            ),
            angularaxis=dict(color="#cbd5e1", gridcolor="rgba(99,102,241,0.2)"),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0", family="Space Grotesk"),
        showlegend=False,
        margin=dict(l=40, r=40, t=40, b=40),
        height=380,
    )
    return fig


def build_bar_chart(tfidf: float, bert: float, combined: float, bert_available: bool) -> go.Figure:
    labels = ["TF-IDF", "Combined"]
    values = [tfidf, combined]
    colors = [score_color(tfidf), score_color(combined)]

    if bert_available:
        labels.insert(1, "BERT Semantic")
        values.insert(1, bert)
        colors.insert(1, score_color(bert))

    fig = go.Figure(go.Bar(
        x=labels,
        y=values,
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v:.1f}" for v in values],
        textposition="outside",
        textfont=dict(color="#e2e8f0", size=14, family="JetBrains Mono"),
        width=0.5,
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0", family="Space Grotesk"),
        yaxis=dict(range=[0, 115], gridcolor="rgba(99,102,241,0.2)", color="#475569"),
        xaxis=dict(color="#94a3b8"),
        margin=dict(l=20, r=20, t=20, b=20),
        height=300,
    )
    return fig


def build_skills_bar(by_category: dict) -> go.Figure:
    if not by_category:
        return go.Figure()
    cats = list(by_category.keys())
    matched = [by_category[c]["matched"] for c in cats]
    missing = [by_category[c]["required"] - by_category[c]["matched"] for c in cats]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Matched", x=cats, y=matched,
                         marker_color="#22c55e", width=0.5))
    fig.add_trace(go.Bar(name="Missing", x=cats, y=missing,
                         marker_color="#f87171", width=0.5))
    fig.update_layout(
        barmode="stack",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0", family="Space Grotesk"),
        yaxis=dict(gridcolor="rgba(99,102,241,0.2)", color="#475569"),
        xaxis=dict(color="#94a3b8"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    bgcolor="rgba(0,0,0,0)", font=dict(color="#e2e8f0")),
        margin=dict(l=20, r=20, t=40, b=80),
        height=320,
    )
    return fig


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0;">
        <div style="font-size:2.5rem">📄</div>
        <div style="font-size:1.1rem; font-weight:700; color:#818cf8;">Resume Analyzer</div>
        <div style="font-size:0.75rem; color:#475569; margin-top:0.2rem;">Powered by Claude AI</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### ⚙️ Configuration")

    api_key = st.text_input(
        "Anthropic API Key",
        type="password",
        placeholder="sk-ant-...",
        help="Required for AI feedback. Get yours at console.anthropic.com",
    )
    if not api_key:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")

    include_ai = st.checkbox("Generate AI Feedback", value=True,
                              help="Calls Claude API — requires API key.")
    use_bert = st.checkbox("Use BERT Similarity", value=True,
                           help="Downloads ~90MB model on first use.")

    st.markdown("---")
    st.markdown("""
    ### 📖 How It Works
    1. **Upload** your resume PDF
    2. **Paste** the job description
    3. **Analyze** — get instant scoring
    4. **Review** AI feedback & suggestions
    5. **Download** your report

    ### 🔬 Scoring Methods
    - **TF-IDF**: Keyword overlap analysis
    - **BERT**: Semantic meaning matching
    - **Combined**: Weighted 40/60 blend
    """)


# ── Hero Header ──────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero-header">
    <div class="hero-title">AI Resume Analyzer</div>
    <div class="hero-subtitle">
        Upload your resume · Paste a job description · Get instant AI-powered match scoring & feedback
    </div>
</div>
""", unsafe_allow_html=True)


# ── Main Input Area ──────────────────────────────────────────────────────────

col_upload, col_jd = st.columns(2, gap="large")

with col_upload:
    st.markdown('<div class="section-header">📄 Resume Upload</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Drop your PDF resume here",
        type=["pdf"],
        help="We accept PDF format only. Max 10MB.",
        label_visibility="collapsed",
    )
    if uploaded_file:
        st.success(f"✓ **{uploaded_file.name}** ({uploaded_file.size / 1024:.0f} KB)")

with col_jd:
    st.markdown('<div class="section-header">💼 Job Description</div>', unsafe_allow_html=True)
    jd_text = st.text_area(
        "Paste the full job description here",
        height=200,
        placeholder="Paste the complete job posting text here — including responsibilities, requirements, and preferred qualifications...",
        label_visibility="collapsed",
    )
    if jd_text:
        word_count = len(jd_text.split())
        st.caption(f"📝 {word_count} words · {len(jd_text)} characters")


# ── Analyze Button ───────────────────────────────────────────────────────────

st.markdown("<br>", unsafe_allow_html=True)
_, btn_col, _ = st.columns([2, 2, 2])

with btn_col:
    analyze_btn = st.button("🔍 Analyze Resume", type="primary")


# ── Analysis Pipeline ────────────────────────────────────────────────────────

if analyze_btn:
    if not uploaded_file:
        st.error("❌ Please upload a resume PDF before analyzing.")
        st.stop()
    if not jd_text.strip():
        st.error("❌ Please paste a job description before analyzing.")
        st.stop()

    # ── Extract PDF text ─────────────────────────────────────────────────────
    with st.spinner("📖 Extracting text from PDF…"):
        from utils.pdf_parser import extract_text_from_pdf, validate_pdf

        file_bytes = uploaded_file.read()
        is_valid, err_msg = validate_pdf(io.BytesIO(file_bytes))
        if not is_valid:
            st.error(f"❌ PDF error: {err_msg}")
            st.stop()

        resume_text = extract_text_from_pdf(io.BytesIO(file_bytes))
        if not resume_text.strip():
            st.error("❌ Could not extract text from PDF. Please try a text-based (not scanned) PDF.")
            st.stop()

    # ── Run analysis ─────────────────────────────────────────────────────────
    with st.spinner("🧠 Running AI analysis (this may take 15-30 seconds on first run)…"):
        analyzer = get_analyzer()

        # Temporarily disable BERT if user opted out
        if not use_bert:
            original_is_available = analyzer.bert_model.is_available
            analyzer.bert_model.is_available = lambda: False

        result = analyzer.analyze(
            resume_text=resume_text,
            jd_text=jd_text,
            include_ai_feedback=(include_ai and bool(api_key)),
            api_key=api_key or None,
        )

        if not use_bert:
            analyzer.bert_model.is_available = original_is_available

    # ── Store result in session state ─────────────────────────────────────────
    st.session_state["result"] = result
    st.session_state["resume_text"] = resume_text

    if result.errors:
        with st.expander("⚠️ Some non-critical warnings occurred"):
            for e in result.errors:
                st.warning(e)


# ── Results Display ──────────────────────────────────────────────────────────

if "result" in st.session_state:
    result = st.session_state["result"]
    resume_text = st.session_state.get("resume_text", "")

    st.markdown("---")

    # ── Score Cards ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📊 Match Score Overview</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3, gap="medium")

    with c1:
        color = score_color(result.combined_score)
        badge = score_label(result.combined_score)
        st.markdown(f"""
        <div class="score-card">
            <div class="score-value" style="color:{color};">{result.combined_score:.0f}%</div>
            <div class="score-label">Combined Score</div>
            <div class="match-badge" style="background:rgba(99,102,241,0.15);color:{color};border:1px solid {color}33;">
                {badge}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        color2 = score_color(result.tfidf_score)
        st.markdown(f"""
        <div class="score-card">
            <div class="score-value" style="color:{color2};">{result.tfidf_score:.0f}%</div>
            <div class="score-label">TF-IDF Score</div>
            <div style="color:#475569; font-size:0.75rem; margin-top:0.4rem;">Keyword Overlap</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        if result.bert_available:
            color3 = score_color(result.bert_score)
            st.markdown(f"""
            <div class="score-card">
                <div class="score-value" style="color:{color3};">{result.bert_score:.0f}%</div>
                <div class="score-label">BERT Score</div>
                <div style="color:#475569; font-size:0.75rem; margin-top:0.4rem;">Semantic Similarity</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="score-card">
                <div class="score-value" style="color:#475569;">—</div>
                <div class="score-label">BERT Score</div>
                <div style="color:#475569; font-size:0.75rem; margin-top:0.4rem;">Model not loaded</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Charts ───────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    chart_col1, chart_col2 = st.columns(2, gap="large")

    with chart_col1:
        st.markdown('<div class="section-header">📈 Score Comparison</div>', unsafe_allow_html=True)
        st.plotly_chart(
            build_bar_chart(result.tfidf_score, result.bert_score,
                            result.combined_score, result.bert_available),
            use_container_width=True, config={"displayModeBar": False}
        )

    with chart_col2:
        by_cat = result.skill_categorization.get("by_category", {})
        if by_cat:
            st.markdown('<div class="section-header">🕸️ Skill Coverage Radar</div>', unsafe_allow_html=True)
            st.plotly_chart(build_radar_chart(by_cat),
                            use_container_width=True, config={"displayModeBar": False})

    # Stacked bar for skills by category
    if by_cat:
        st.markdown('<div class="section-header">📊 Skills by Category</div>', unsafe_allow_html=True)
        st.plotly_chart(build_skills_bar(by_cat),
                        use_container_width=True, config={"displayModeBar": False})

    # ── Skill Analysis ───────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-header">🎯 Skill Analysis</div>', unsafe_allow_html=True)

    skill_col1, skill_col2, skill_col3 = st.columns(3, gap="medium")
    skill_cat = result.skill_categorization
    present = skill_cat.get("present", [])
    missing = skill_cat.get("missing", [])
    extra = skill_cat.get("extra", [])

    with skill_col1:
        st.markdown(f"**✅ Matched Skills ({len(present)})**")
        if present:
            st.markdown(render_skill_tags(present, "present"), unsafe_allow_html=True)
        else:
            st.caption("No matched skills detected.")

    with skill_col2:
        st.markdown(f"**❌ Missing Skills ({len(missing)})**")
        if missing:
            st.markdown(render_skill_tags(missing, "missing"), unsafe_allow_html=True)
        else:
            st.success("Great coverage — no key missing skills!")

    with skill_col3:
        st.markdown(f"**💡 Additional Skills ({len(extra)})**")
        if extra:
            st.markdown(render_skill_tags(extra[:20], "extra"), unsafe_allow_html=True)
        else:
            st.caption("No extra skills beyond JD requirements.")

    # ── Matching Terms ───────────────────────────────────────────────────────
    top_terms = result.tfidf_breakdown.get("top_matching_terms", [])
    if top_terms:
        with st.expander("🔍 Top Matching Keywords (TF-IDF analysis)"):
            term_cols = st.columns(3)
            for i, (term, score) in enumerate(top_terms[:15]):
                with term_cols[i % 3]:
                    bar_width = int(score / max(t[1] for t in top_terms) * 100)
                    st.markdown(
                        f"`{term}` — `{score:.4f}`",
                        help=f"TF-IDF overlap score: {score}"
                    )

    # ── AI Feedback ──────────────────────────────────────────────────────────
    if result.feedback_sections or result.ai_feedback:
        st.markdown("---")
        st.markdown('<div class="section-header">🤖 AI-Powered Feedback</div>', unsafe_allow_html=True)

        sections = result.feedback_sections
        processing_time = result.processing_time_sec

        if "overall_assessment" in sections:
            st.markdown(f"""
            <div class="feedback-card">
                <div class="feedback-card-title">📋 Overall Assessment</div>
                <div class="feedback-card-content">{sections['overall_assessment']}</div>
            </div>""", unsafe_allow_html=True)

        fb_col1, fb_col2 = st.columns(2, gap="large")
        with fb_col1:
            if "strengths" in sections:
                st.markdown(f"""
                <div class="feedback-card strengths">
                    <div class="feedback-card-title">💪 Strengths</div>
                    <div class="feedback-card-content">{sections['strengths']}</div>
                </div>""", unsafe_allow_html=True)
            if "quick_wins" in sections:
                st.markdown(f"""
                <div class="feedback-card wins">
                    <div class="feedback-card-title">⚡ Quick Wins</div>
                    <div class="feedback-card-content">{sections['quick_wins']}</div>
                </div>""", unsafe_allow_html=True)

        with fb_col2:
            if "gaps" in sections:
                st.markdown(f"""
                <div class="feedback-card gaps">
                    <div class="feedback-card-title">🔴 Gaps & Weaknesses</div>
                    <div class="feedback-card-content">{sections['gaps']}</div>
                </div>""", unsafe_allow_html=True)
            if "long_term_recommendations" in sections:
                st.markdown(f"""
                <div class="feedback-card strategy">
                    <div class="feedback-card-title">🎯 Long-Term Strategy</div>
                    <div class="feedback-card-content">{sections['long_term_recommendations']}</div>
                </div>""", unsafe_allow_html=True)

        if "keywords_to_add" in sections:
            with st.expander("🏷️ Keywords to Add to Your Resume"):
                st.markdown(f"""
                <div class="feedback-card">
                    <div class="feedback-card-content">{sections['keywords_to_add']}</div>
                </div>""", unsafe_allow_html=True)

        if "interview_talking_points" in sections:
            with st.expander("🎙️ Interview Talking Points"):
                st.markdown(f"""
                <div class="feedback-card strategy">
                    <div class="feedback-card-content">{sections['interview_talking_points']}</div>
                </div>""", unsafe_allow_html=True)

        if not sections and result.ai_feedback:
            st.text_area("Raw AI Feedback", result.ai_feedback, height=300)

        if not (sections or result.ai_feedback) and include_ai:
            st.info("💡 Add your Anthropic API key in the sidebar to unlock AI feedback.")

    # ── Meta info ─────────────────────────────────────────────────────────────
    st.markdown("---")
    meta_col1, meta_col2, meta_col3 = st.columns(3)
    with meta_col1:
        st.caption(f"⏱️ Analysis completed in **{result.processing_time_sec:.1f}s**")
    with meta_col2:
        st.caption(f"📄 Resume: **{len(resume_text):,} characters**")
    with meta_col3:
        bert_status = "✅ Active" if result.bert_available else "❌ Unavailable"
        st.caption(f"🧠 BERT: {bert_status}")

    # ── Download Report ───────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    _, dl_col, _ = st.columns([2, 2, 2])
    with dl_col:
        with st.spinner("Generating PDF report…"):
            from services.report_generator import generate_pdf_report
            pdf_bytes = generate_pdf_report(result)

        if pdf_bytes:
            st.download_button(
                label="📥 Download Full PDF Report",
                data=pdf_bytes,
                file_name="resume_analysis_report.pdf",
                mime="application/pdf",
            )
        else:
            st.warning("PDF generation failed. Check logs for details.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; color:#334155; font-size:0.78rem; margin-top:3rem; padding-top:1rem; border-top:1px solid rgba(99,102,241,0.15);">
    AI Resume Analyzer · Built with Streamlit + Claude · For guidance only, not a hiring guarantee
</div>
""", unsafe_allow_html=True)
