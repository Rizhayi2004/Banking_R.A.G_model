
# app.py  —  Bank RAG Streamlit Application (Ollama Edition)

import os
import json
import pickle
import numpy as np
import streamlit as st
import faiss
import ollama
from pathlib import Path
from sentence_transformers import SentenceTransformer

# ── Config ──────────────────────────────────────────────────
ARTIFACT_DIR = Path("data/artifacts")
CONFIG_FILE  = ARTIFACT_DIR / "config.json"
OLLAMA_MODEL = "llama3.2"

st.set_page_config(
    page_title="Bank AI Assistant",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a1628 0%, #0d2040 100%);
    border-right: 1px solid #1e3a5f;
}
[data-testid="stSidebar"] * { color: #c8d8e8 !important; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #e8f4ff !important; }

.main-header {
    font-family: 'DM Serif Display', serif;
    font-size: 2.4rem;
    color: #0a2540;
    letter-spacing: -0.5px;
    margin-bottom: 0;
}
.main-subtitle {
    color: #5a7fa0;
    font-size: 0.95rem;
    margin-top: 2px;
    margin-bottom: 1.5rem;
}
.user-bubble {
    background: #0a2540;
    color: #ffffff;
    padding: 14px 18px;
    border-radius: 18px 18px 4px 18px;
    margin: 8px 0 8px 15%;
    font-size: 0.95rem;
    line-height: 1.6;
    box-shadow: 0 2px 8px rgba(10,37,64,0.2);
}
.bot-bubble {
    background: #f0f6ff;
    color: #0a2540;
    padding: 14px 18px;
    border-radius: 18px 18px 18px 4px;
    margin: 8px 15% 8px 0;
    font-size: 0.95rem;
    line-height: 1.6;
    border: 1px solid #d0e4f7;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}
.source-tag {
    display: inline-block;
    background: #e3f0ff;
    color: #0a5fa8;
    font-size: 0.72rem;
    padding: 2px 8px;
    border-radius: 20px;
    margin: 2px 3px 0 0;
    font-weight: 500;
    border: 1px solid #b8d4f0;
}
.context-card {
    background: #f8fbff;
    border: 1px solid #d4e8fb;
    border-left: 3px solid #1a6fc4;
    border-radius: 8px;
    padding: 12px 14px;
    margin-bottom: 10px;
    font-size: 0.83rem;
    color: #2c4a6e;
    line-height: 1.5;
}
.score-badge {
    float: right;
    background: #1a6fc4;
    color: white;
    font-size: 0.7rem;
    padding: 1px 7px;
    border-radius: 10px;
    font-weight: 500;
}
.ollama-badge {
    display: inline-block;
    background: #0a2540;
    color: #7eb8f7;
    font-size: 0.75rem;
    padding: 3px 10px;
    border-radius: 20px;
    font-weight: 500;
    border: 1px solid #1e3a5f;
    margin-bottom: 8px;
}
.stTextInput > div > div > input {
    border-radius: 12px !important;
    border: 2px solid #d0e4f7 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    padding: 10px 16px !important;
}
.stTextInput > div > div > input:focus {
    border-color: #1a6fc4 !important;
    box-shadow: 0 0 0 3px rgba(26,111,196,0.15) !important;
}
.stButton > button {
    background: #0a2540 !important;
    color: white !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    border: none !important;
    padding: 10px 20px !important;
    transition: background 0.2s !important;
}
.stButton > button:hover {
    background: #1a4570 !important;
}
div[data-testid="stMetricValue"] {
    color: #0a2540;
    font-family: 'DM Serif Display', serif;
}
</style>
""", unsafe_allow_html=True)


# ── Check Ollama is running ───────────────────────────────────
@st.cache_resource(show_spinner=False)
def check_ollama():
    try:
        client = ollama.Client(host="http://127.0.0.1:11434")
        models = client.list()
        available = [m["model"] for m in models.get("models", [])]
        return True, available
    except Exception as e:
        return False, str(e)


# ── Load artefacts (cached) ───────────────────────────────────
@st.cache_resource(show_spinner="Loading knowledge base…")
def load_rag_components():
    if not CONFIG_FILE.exists():
        return None, None, None
    config   = json.loads(CONFIG_FILE.read_text())
    embedder = SentenceTransformer(config["embed_model"])
    index    = faiss.read_index(str(ARTIFACT_DIR / config["index_file"]))
    with open(ARTIFACT_DIR / config["metadata_file"], "rb") as f:
        chunks = pickle.load(f)
    return embedder, index, chunks


# ── Retrieval ─────────────────────────────────────────────────
def retrieve(query: str, embedder, index, chunks, top_k: int = 5):
    q_vec = embedder.encode([query], normalize_embeddings=True)
    scores, indices = index.search(np.array(q_vec, dtype="float32"), top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        item = chunks[idx].copy()
        item["score"] = float(score)
        results.append(item)
    return results


# ── Generation with Ollama ────────────────────────────────────
def generate_answer(query: str, hits: list[dict], model: str = OLLAMA_MODEL) -> str:
    context_parts = [
        f"[{i+1}] ({h['source']})\n{h['text']}"
        for i, h in enumerate(hits)
    ]
    context = "\n\n".join(context_parts)

    system_prompt = """You are an expert bank assistant with access to the bank's policy documents,
product catalogue, and customer activity data.

Rules:
- Answer ONLY from the provided context. Do not hallucinate.
- Be concise, professional, and friendly.
- If the answer isn't in the context, say: "I don't have enough information to answer that."
- When referencing policies or products, mention the source.
- Format numbers, rates, and fees clearly."""

    user_prompt = f"""Context:
{context}

Customer Question: {query}

Provide a clear, helpful answer based strictly on the context above."""

    client = ollama.Client(host="http://127.0.0.1:11434")  
    response = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt}
        ]
    )
    return response["message"]["content"]


# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏦 Bank RAG")
    st.markdown('<span class="ollama-badge">⚡ Powered by Ollama</span>',
                unsafe_allow_html=True)
    st.markdown("---")

    # Ollama status
    ollama_ok, ollama_info = check_ollama()
    if ollama_ok:
        st.success("✅ Ollama is running")
        if isinstance(ollama_info, list) and ollama_info:
            st.markdown("**Available models:**")
            for m in ollama_info:
                st.markdown(f"- `{m}`")
    else:
        st.error("❌ Ollama not detected")
        st.markdown("""
**To fix:**
1. Download from [ollama.com](https://ollama.com/download)
2. Install and restart PowerShell
3. Run: `ollama pull llama3.2`
4. Restart this app
        """)

    st.markdown("---")

    # Model selector
    st.markdown("### ⚙️ Settings")
    selected_model = st.selectbox(
        "Ollama Model",
        options=ollama_info if isinstance(ollama_info, list) and ollama_info else [OLLAMA_MODEL],
        index=0,
        help="Models available on your local Ollama install"
    )
    top_k = st.slider("Context chunks (top-k)", 2, 10, 5,
                       help="More chunks = more context but slower")

    st.markdown("---")
    st.markdown("### 📚 Knowledge Sources")
    st.markdown("""
- 📋 **Bank Policy**
- 🛍️ **Product Catalogue**
- 👤 **Customer Activity**
    """)

    st.markdown("---")
    st.markdown("### 💡 Sample Questions")
    sample_qs = [
        "What are the loan eligibility requirements?",
        "Which savings account has the best interest rate?",
        "What is the bank's policy on overdrafts?",
        "What are the fees for international transfers?",
        "How can a customer upgrade their account?"
    ]
    for q in sample_qs:
        if st.button(q, key=q):
            st.session_state["prefill"] = q

    st.markdown("---")
    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()


# ── Main area ─────────────────────────────────────────────────
st.markdown('<p class="main-header">Bank AI Assistant</p>', unsafe_allow_html=True)
st.markdown('<p class="main-subtitle">Powered by Retrieval-Augmented Generation · Policy · Products · Customer Data · 100% Local</p>',
            unsafe_allow_html=True)

# Block app if Ollama is not running
if not ollama_ok:
    st.error("⚠️ Ollama is not running. Please install and start Ollama, then refresh this page.")
    st.info("👈 See the sidebar for setup instructions.")
    st.stop()

# Load RAG components
embedder, faiss_index, chunks = load_rag_components()

if embedder is None:
    st.error("⚠️ Knowledge base not found. Run the notebook first to build `data/artifacts/`.")
    st.stop()

# Stats bar
col1, col2, col3 = st.columns(3)
col1.metric("📄 Total Chunks", f"{len(chunks):,}")
col2.metric("🔍 Retrieval",    f"Top {top_k}")
col3.metric("🧠 Model",        selected_model)

st.markdown("---")

# Chat state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render chat history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="user-bubble">🧑 {msg["content"]}</div>',
                    unsafe_allow_html=True)
    else:
        answer  = msg["content"]["answer"]
        sources = msg["content"].get("sources", [])
        source_tags = "".join(
            f'<span class="source-tag">📎 {s}</span>'
            for s in set(sources)
        )
        st.markdown(
            f'<div class="bot-bubble">🏦 {answer}<br/><br/>{source_tags}</div>',
            unsafe_allow_html=True
        )

# Input row
prefill = st.session_state.pop("prefill", "")
with st.container():
    col_input, col_btn = st.columns([5, 1])
    with col_input:
        user_input = st.text_input(
            "Ask anything about bank products, policies, or accounts…",
            value=prefill,
            key="chat_input",
            label_visibility="collapsed"
        )
    with col_btn:
        send = st.button("Send →", use_container_width=True)

# Handle submission
if (send or user_input) and user_input.strip():
    query = user_input.strip()
    st.session_state.messages.append({"role": "user", "content": query})
    st.markdown(f'<div class="user-bubble">🧑 {query}</div>', unsafe_allow_html=True)

    with st.spinner("Searching knowledge base…"):
        hits = retrieve(query, embedder, faiss_index, chunks, top_k)

    with st.spinner(f"Generating answer with {selected_model}…"):
        try:
            answer = generate_answer(query, hits, model=selected_model)
        except Exception as e:
            answer = f"⚠️ Error generating answer: {str(e)}. Make sure Ollama is running and `{selected_model}` is pulled."

    sources = [h["source"] for h in hits]
    st.session_state.messages.append({
        "role": "assistant",
        "content": {"answer": answer, "sources": sources}
    })

    source_tags = "".join(
        f'<span class="source-tag">📎 {s}</span>'
        for s in set(sources)
    )
    st.markdown(
        f'<div class="bot-bubble">🏦 {answer}<br/><br/>{source_tags}</div>',
        unsafe_allow_html=True
    )

    # Expandable retrieved context
    with st.expander("🔍 View retrieved context", expanded=False):
        for i, h in enumerate(hits, 1):
            st.markdown(
                f'<div class="context-card">'
                f'<span class="score-badge">{h["score"]:.3f}</span>'
                f'<strong>[{i}] {h["source"]}</strong><br/>{h["text"][:400]}…'
                f'</div>',
                unsafe_allow_html=True
            )