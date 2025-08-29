# Student Manual Chatbot — RAG → AI only (no RAG display)
# - Single PDF: data/Student Manual 2025 Edition.pdf
# - Retrieval: TF-IDF (bigrams + policy-code tokenization) + phrase rerank
# - Answering: Gemini or OpenAI ONLY (no extractive chunks shown)
# - Small-talk bypass
# - Secrets loaded from .env (python-dotenv) or Streamlit secrets (if present)

import os
import re
import html
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load .env into environment (does nothing if .env missing)
load_dotenv(override=True)

# -------------------------------------------------------
# App config
# -------------------------------------------------------
st.set_page_config(page_title="Student Manual RAG → AI", page_icon="💬", layout="wide")

# -------------------------------------------------------
# Settings
# -------------------------------------------------------
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

FILENAME = "Student Manual 2025 Edition.pdf"
MANUAL_LABEL = "Student Manual 2025 Edition"

CHUNK_SIZE = 600
CHUNK_OVERLAP = 220
TOP_K = 6
MIN_SIM = 0.15  # raise to 0.20–0.25 later if retrieval looks strong

# -------------------------------------------------------
# Extraction / Chunking
# -------------------------------------------------------
def clean_text(t: str) -> str:
    t = t.replace("\u00ad", "")
    t = re.sub(r"[ \t]+\n", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

def read_pdf_pages(path: Path) -> List[Tuple[int, str]]:
    out = []
    with open(path, "rb") as f:
        reader = PdfReader(f)
        for i, page in enumerate(reader.pages):
            try:
                txt = page.extract_text() or ""
            except Exception:
                txt = ""
            out.append((i + 1, clean_text(txt)))
    return out

def chunk_text(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk = text[start:end]
        if end < len(text):
            last_space = chunk.rfind(" ")
            if last_space > chunk_size * 0.6:
                chunk = chunk[:last_space]
                end = start + last_space
        chunk = chunk.strip()
        if len(chunk) > 40:
            chunks.append(chunk)
        start = max(end - overlap, end)
    return chunks

def build_corpus() -> Tuple[List[Dict], Dict]:
    records = []
    diag = {"exists": False, "pages": 0, "chunks": 0, "first_extract_preview": ""}
    path = DATA_DIR / FILENAME
    diag["exists"] = path.exists()
    if not diag["exists"]:
        return [], diag
    pages = read_pdf_pages(path)
    diag["pages"] = len(pages)
    first_preview = None
    rec_id = 0
    for pnum, ptext in pages:
        if first_preview is None and ptext.strip():
            first_preview = ptext.strip()[:400]
        for chunk in chunk_text(ptext):
            records.append({"id": rec_id, "text": chunk, "file": MANUAL_LABEL, "page": pnum})
            rec_id += 1
            diag["chunks"] += 1
    diag["first_extract_preview"] = first_preview or "(no extractable text)"
    return records, diag

# -------------------------------------------------------
# Phrase handling
# -------------------------------------------------------
def build_phrase_regex(query: str):
    words = [w for w in re.split(r"\W+", query.strip()) if w]
    if len(words) < 2:
        return None
    pattern = r"\b" + r"[-\s]+".join(re.escape(w) for w in words) + r"\b"
    try:
        return re.compile(pattern, flags=re.IGNORECASE)
    except re.error:
        return None

def contains_phrase(text: str, phrase_re) -> bool:
    if not phrase_re:
        return False
    return bool(phrase_re.search(text))

# -------------------------------------------------------
# Retrieval
# -------------------------------------------------------
@st.cache_resource(show_spinner=False)
def build_vector_index(records: List[Dict]):
    texts = [r["text"] for r in records]
    tfidf = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b[\w\.\-]{2,}\b",  # keep digits/dots/hyphens
        min_df=1, max_df=0.98,
    )
    X = tfidf.fit_transform(texts)
    return tfidf, X

def retrieve(query: str, records: List[Dict], tfidf, X, top_k=TOP_K):
    if not query.strip():
        return []
    qv = tfidf.transform([query])
    sims = cosine_similarity(qv, X)[0]
    order = np.argsort(-sims)[: max(top_k * 2, 10)]
    hits = []
    for idx in order:
        r = records[int(idx)]
        hits.append({**r, "score": float(sims[idx])})
    return hits

def keyword_fallback(query: str, records: List[Dict], top_k=TOP_K):
    q = query.strip().lower()
    if not q:
        return []
    terms = [w for w in re.split(r"\W+", q) if len(w) >= 3]
    if not terms:
        return []
    scored = []
    for r in records:
        txt = r["text"].lower()
        score = sum(txt.count(t) for t in terms)
        if score > 0:
            scored.append((score, r))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [{**r, "score": float(s)} for s, r in scored[:top_k]]

def rerank_with_phrase(hits: List[Dict], phrase_re, top_k=TOP_K, boost=0.40, exact_filter=False):
    if not phrase_re:
        return hits[:top_k]
    if exact_filter:
        filtered = [h for h in hits if contains_phrase(h["text"], phrase_re)]
        return filtered[:top_k] if filtered else hits[:top_k]
    boosted = []
    for h in hits:
        s = h["score"] + (boost if contains_phrase(h["text"], phrase_re) else 0.0)
        boosted.append({**h, "score": s})
    boosted.sort(key=lambda x: x["score"], reverse=True)
    return boosted[:top_k]

# -------------------------------------------------------
# Small talk
# -------------------------------------------------------
def is_small_talk(q: str) -> bool:
    ql = q.strip().lower()
    if len(ql) <= 30:
        pats = [
            r"^(hi|hello|hey|yo)\b",
            r"^(good\s*(morning|afternoon|evening))\b",
            r"^(how are you|how's it going|sup)\b",
            r"^(thanks|thank you|ty)\b",
            r"^(ok|okay|noted|got it)$",
        ]
        return any(re.search(p, ql) for p in pats)
    return False

def small_talk_reply(q: str) -> str:
    ql = q.strip().lower()
    if re.search(r"thanks|thank you|ty", ql):
        return "You’re welcome! I can answer policy questions grounded in the Student Manual."
    if re.search(r"how are you|how's it going", ql):
        return "I’m good—ready to help with Student Manual topics. What would you like to check?"
    return "Hi! Ask about policies like “dress code”, “attendance”, or “grading”. I’ll answer from the Student Manual."

# -------------------------------------------------------
# AI Answerers (Gemini / OpenAI) — RAG context is passed, not shown
# -------------------------------------------------------
def build_llm_context(hits: List[Dict]) -> str:
    lines = []
    for h in hits:
        page = f", p. {h['page']}" if h.get("page") else ""
        lines.append(f"[{h['file']}{page}]")
        lines.append(h["text"])
    return "\n".join(lines[:TOP_K * 2])

def ai_prompt(query: str, context: str) -> str:
    return (
        "You are a STRICT RAG assistant for a university student manual.\n"
        "Answer using ONLY the context below. If the answer is not clearly supported, reply exactly:\n"
        "'Not covered in the manual excerpts provided.'\n\n"
        "Guidelines:\n"
        "- No new facts. Do not speculate.\n"
        "- Prefer short bullet points or numbered steps.\n"
        "- Include citations inline like [Student Manual 2025 Edition, p. X].\n"
        "- Be concise and student-friendly.\n\n"
        f"Question: {query}\n\n"
        f"Context:\n{context}\n"
    )

def get_secret(name: str) -> Optional[str]:
    v = os.environ.get(name)
    if v:
        return v
    try:
        return st.secrets.get(name)
    except Exception:
        return None

def try_gemini_answer(query: str, hits: List[Dict]) -> Optional[str]:
    key = get_secret("GEMINI_API_KEY")
    if not key:
        return None
    try:
        import google.generativeai as genai
        genai.configure(api_key=key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        context = build_llm_context(hits)
        prompt = ai_prompt(query, context)
        out = model.generate_content(prompt)
        txt = getattr(out, "text", "") or ""
        return txt.strip() or None
    except Exception as e:
        st.toast(f"Gemini unavailable: {e}", icon="⚠️")
        return None

def try_openai_answer(query: str, hits: List[Dict]) -> Optional[str]:
    key = get_secret("OPENAI_API_KEY")
    if not key:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=key)
        context = build_llm_context(hits)
        prompt = ai_prompt(query, context)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return (resp.choices[0].message.content or "").strip() or None
    except Exception:
        try:
            import openai
            openai.api_key = key
            context = build_llm_context(hits)
            prompt = ai_prompt(query, context)
            resp = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            return (resp.choices[0].message["content"] or "").strip() or None
        except Exception as e2:
            st.toast(f"OpenAI unavailable: {e2}", icon="⚠️")
            return None

# -------------------------------------------------------
# UI
# -------------------------------------------------------
st.title("💬 NEMSU Student Manual")
st.caption("AI answers grounded in: Student Manual 2025 Edition.pdf (sources not displayed)")

st.sidebar.header("Manual")
exists = (DATA_DIR / FILENAME).exists()
st.sidebar.write(f"• **{MANUAL_LABEL}** — {'✅ found' if exists else '❌ missing'}")
st.sidebar.code(f"data/{FILENAME}", language="bash")

up = st.sidebar.file_uploader("Upload Student Manual 2025 Edition.pdf", type=["pdf"])
if up:
    (DATA_DIR / FILENAME).write_bytes(up.read())
    st.sidebar.success("Saved Student Manual 2025 Edition.pdf")
    st.cache_resource.clear()

answer_mode = st.sidebar.radio("AI Provider", ["Gemini", "OpenAI"], index=0)
exact_phrase_mode = st.sidebar.toggle("Exact-phrase mode", value=True)

# Build corpus
records, diag = build_corpus()
st.write(f"**Pages:** {diag['pages']} | **Chunks:** {diag['chunks']}")

if diag["exists"] and diag["chunks"] > 0:
    tfidf, X = build_vector_index(records)

    if "chat" not in st.session_state:
        st.session_state.chat = []

    for role, content in st.session_state.chat:
        with st.chat_message(role):
            st.markdown(content, unsafe_allow_html=True)

    q = st.chat_input("Ask about the Student Manual...")
    if q:
        st.session_state.chat.append(("user", html.escape(q)))
        with st.chat_message("user"):
            st.markdown(html.escape(q))

        if is_small_talk(q):
            reply = small_talk_reply(q)
            with st.chat_message("assistant"):
                st.markdown(reply)
            st.session_state.chat.append(("assistant", reply))
            st.stop()

        phrase_re = build_phrase_regex(q)
        raw_hits = retrieve(q, records, tfidf, X, top_k=TOP_K)
        hits = rerank_with_phrase(
            raw_hits, phrase_re=phrase_re, top_k=TOP_K, boost=0.40,
            exact_filter=bool(phrase_re and exact_phrase_mode)
        )

        max_sim = max([h["score"] for h in hits], default=0.0)
        have_retrieval = bool(hits and max_sim >= MIN_SIM)
        if not have_retrieval:
            k_hits = keyword_fallback(q, records, top_k=TOP_K * 2)
            hits = rerank_with_phrase(
                k_hits, phrase_re=phrase_re, top_k=TOP_K, boost=0.40,
                exact_filter=bool(phrase_re and exact_phrase_mode)
            )

        ai_out = try_gemini_answer(q, hits) if answer_mode == "Gemini" else try_openai_answer(q, hits)

        with st.chat_message("assistant"):
            if ai_out:
                if not have_retrieval:
                    st.warning("Answer based on broadened matches.")
                st.markdown(ai_out)
                st.session_state.chat.append(("assistant", ai_out))
            else:
                msg = f"AI answer unavailable. Please set your {answer_mode} API key."
                st.markdown(msg)
                st.session_state.chat.append(("assistant", msg))
else:
    st.error("Manual not found or has no extractable text. Ensure the PDF is text-based (OCR if needed).")
