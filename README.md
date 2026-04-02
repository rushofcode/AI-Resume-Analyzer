# 📄 AI Resume Analyzer

A production-ready, full-stack AI application that analyzes resumes against job descriptions using **TF-IDF**, **BERT semantic similarity**, and **Claude LLM feedback**.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- pip
- (Optional) Anthropic API key for AI feedback

### Installation

```bash
# 1. Clone / download the project
cd resume_analyzer

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate      # macOS/Linux
# venv\Scripts\activate       # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download spaCy model (optional — for enhanced NER)
python -m spacy download en_core_web_sm

# 5. Set your API key (optional — for AI feedback)
export ANTHROPIC_API_KEY="sk-ant-..."   # macOS/Linux
# set ANTHROPIC_API_KEY=sk-ant-...      # Windows

# 6. Run the app
streamlit run app.py
```

The app opens at **http://localhost:8501**.

---

## 🏗️ Architecture

```
resume_analyzer/
├── app.py                     ← Streamlit UI (entry point)
├── requirements.txt
├── README.md
│
├── utils/                     ← Pure utility functions
│   ├── __init__.py
│   ├── pdf_parser.py          ← PDF text extraction (pdfplumber + PyPDF2)
│   ├── text_preprocessor.py   ← Tokenisation, stop-word removal, lemmatisation
│   └── skill_extractor.py     ← Skill taxonomy + regex matching
│
├── models/                    ← Similarity models
│   ├── __init__.py
│   ├── tfidf_model.py         ← TF-IDF + cosine similarity
│   └── bert_model.py          ← Sentence Transformers (all-MiniLM-L6-v2)
│
└── services/                  ← Business logic
    ├── __init__.py
    ├── analyzer.py            ← Orchestrates full pipeline → AnalysisResult
    ├── llm_feedback.py        ← Claude API calls + prompt engineering
    └── report_generator.py    ← PDF download report (fpdf2)
```

### Data Flow

```
PDF Upload  ──► extract_text_from_pdf()
                    │
Job Description ────┤
                    │
                    ├──► TFIDFSimilarityModel  ─┐
                    │                            ├─► combined_score (40/60 blend)
                    ├──► BERTSimilarityModel    ─┘
                    │
                    ├──► extract_skills(resume) + extract_skills(jd)
                    │          └──► categorize_skills()
                    │
                    └──► get_ai_feedback() → Claude API
                                    │
                                    ▼
                            AnalysisResult
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
                Streamlit UI   Plotly Charts   PDF Report
```

---

## 🎯 Features

| Feature | Implementation |
|---|---|
| PDF Parsing | pdfplumber (primary) + PyPDF2 (fallback) |
| Text Preprocessing | NLTK tokenisation + lemmatisation + custom stop-words |
| TF-IDF Similarity | scikit-learn TfidfVectorizer + cosine_similarity |
| BERT Similarity | sentence-transformers `all-MiniLM-L6-v2` |
| Combined Score | 40% TF-IDF + 60% BERT (100% TF-IDF when BERT unavailable) |
| Skill Extraction | Regex taxonomy of 300+ skills across 8 categories |
| AI Feedback | Claude claude-sonnet-4-20250514 with structured XML prompt |
| Visualisations | Plotly bar chart + radar chart + stacked skill chart |
| PDF Report | fpdf2 download with full breakdown |
| Caching | `@st.cache_resource` for model loading |

---

## 🔬 Scoring Explained

### TF-IDF Score
- Preprocesses both texts (lowercase, stop-word removal, lemmatisation)
- Fits a TF-IDF vectoriser on both documents combined
- Computes cosine similarity → 0–100 scale
- **Good for**: keyword-heavy matches, exact terminology overlap

### BERT Score
- Uses `all-MiniLM-L6-v2` (fast, 80MB, strong performance)
- Chunks long texts into 2000-char segments, averages embeddings
- Cosine similarity on L2-normalised vectors → 0–100 scale
- **Good for**: semantic matches ("developed software" ≈ "software engineering")

### Combined Score
```
Combined = 0.40 × TF-IDF + 0.60 × BERT
```
BERT gets higher weight because semantic understanding is more valuable than raw keyword overlap for job matching.

### Score Interpretation
| Score | Label |
|---|---|
| 80–100 | Excellent Match 🚀 |
| 65–79 | Good Match ✅ |
| 45–64 | Moderate Match ⚠️ |
| 0–44 | Low Match ❌ |

---

## 📋 Example Inputs & Expected Outputs

### Example Resume Excerpt
```
Senior Software Engineer with 5 years of experience building scalable 
Python microservices. Proficient in FastAPI, Docker, Kubernetes, AWS, 
and PostgreSQL. Led a team of 4 engineers delivering ML pipelines using 
PyTorch and scikit-learn. Strong Git, CI/CD (GitHub Actions), and Agile.
```

### Example Job Description
```
We are looking for a Backend Engineer with experience in Python, FastAPI 
or Django, Docker, Kubernetes, and cloud platforms (AWS preferred). 
Knowledge of ML systems is a plus. Agile team environment.
```

### Expected Output
- **TF-IDF Score**: ~72–78%
- **BERT Score**: ~80–86%
- **Combined Score**: ~77–83%
- **Label**: Good Match / Excellent Match
- **Matched Skills**: python, fastapi, docker, kubernetes, aws, pytorch, scikit-learn, git, agile
- **Missing Skills**: django (if only fastapi mentioned)

---

## 🔮 Future Improvements

1. **ATS Simulation** — Flag formatting issues that break applicant tracking systems
2. **Multi-format Support** — DOCX, TXT, LinkedIn PDF exports
3. **Job Role Classifier** — Auto-detect role category to adjust weights
4. **Historical Tracking** — Store and compare multiple analyses
5. **Cover Letter Generator** — Use LLM to draft a tailored cover letter
6. **Bulk Analysis** — Upload multiple JDs and rank-order matches
7. **Fine-tuned BERT** — Train on hire/no-hire datasets for domain-specific accuracy
8. **OCR Support** — Handle scanned resume images via Tesseract
9. **LinkedIn Import** — Scrape profile directly as resume source
10. **Streaming Feedback** — Stream Claude's response token-by-token for better UX

---

## ⚙️ Environment Variables

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | For AI feedback | Claude API key |

---

## 🐛 Troubleshooting

**BERT model not loading?**
- First run downloads ~80MB. Ensure internet access.
- Disable BERT in the sidebar if you want faster results without it.

**PDF text extraction empty?**
- Scanned/image PDFs won't work — text must be selectable.
- Try copy-pasting text from the PDF and using it directly.

**AI feedback not appearing?**
- Check your API key in the sidebar or `ANTHROPIC_API_KEY` env var.
- Ensure you have credits in your Anthropic account.
