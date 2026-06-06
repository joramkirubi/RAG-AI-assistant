"""
app.py
Streamlit chatbot interface for the RAG Assistant.

Key improvement: embedding model and vector store are pre-loaded and cached
using @st.cache_resource when the app starts. This means the first question
is answered at the same speed as all subsequent ones — no cold-start delay.

Run with:
    python -m streamlit run app.py
"""

import os
from dotenv import load_dotenv
import streamlit as st
from rag_chain import ask
from vectordb import get_embeddings, get_vectorstore

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG AI Assistant",
    page_icon="🤖",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.stApp { background-color: #0f1117; }
.source-box {
    background-color: #111820;
    border: 1px solid #1e3a5f;
    border-left: 3px solid #185FA5;
    border-radius: 6px;
    padding: 8px 14px;
    margin-top: 8px;
    font-size: 0.82rem;
    color: #7aadcc;
}
.source-label {
    font-size: 0.78rem;
    font-weight: 600;
    color: #4a7a9b;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 4px;
}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("## 🤖 RAG AI Assistant")
st.markdown(
    "<span style='color:#4a7a9b; font-size:0.9rem;'>"
    "ChromaDB &nbsp;·&nbsp; HuggingFace Embeddings &nbsp;·&nbsp; Groq LLM &nbsp;·&nbsp; ReAct Reasoning"
    "</span>",
    unsafe_allow_html=True,
)
st.divider()

# ── Check setup before anything else ─────────────────────────────────────────
api_key   = os.environ.get("GROQ_API_KEY", "")
vs_exists = os.path.exists("vectorstore")

if not api_key:
    st.error("⚠️ **GROQ_API_KEY not found.** Copy `.env.example` to `.env` and add your key.")
    st.stop()

if not vs_exists:
    st.error("⚠️ **Vector store not found.** Run `python ingest.py` first.")
    st.stop()

# ── Pre-load and cache embedding model + vector store ─────────────────────────
# @st.cache_resource runs this function ONCE when the app starts and keeps
# the result in memory. Every query after the first uses the cached objects —
# no model re-loading, no cold-start delay.
@st.cache_resource(show_spinner="Loading knowledge base... (first time only)")
def load_rag_components():
    embeddings  = get_embeddings()
    vectorstore = get_vectorstore(embeddings)
    return embeddings, vectorstore

# Trigger the cache on startup so it's ready before the first question
load_rag_components()

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Render chat history ───────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            source_names = "  &nbsp;|&nbsp;  ".join(
                f"📄 {s['file']} <span style='color:#2a4a6a'>({s['type']})</span>"
                for s in msg["sources"]
            )
            st.markdown(
                f"<div class='source-box'>"
                f"<div class='source-label'>Referenced documents</div>"
                f"{source_names}"
                f"</div>",
                unsafe_allow_html=True,
            )

# ── Input ─────────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask a question about your documents..."):

    st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result  = ask(prompt)
                answer  = result["answer"]
                sources = result["sources"]
            except FileNotFoundError as e:
                answer, sources = f"❌ {e}", []
            except EnvironmentError as e:
                answer, sources = f"❌ {e}", []
            except Exception as e:
                answer, sources = f"❌ Unexpected error: {e}", []

        st.markdown(answer)

        if sources:
            source_names = "  &nbsp;|&nbsp;  ".join(
                f"📄 {s['file']} <span style='color:#2a4a6a'>({s['type']})</span>"
                for s in sources
            )
            st.markdown(
                f"<div class='source-box'>"
                f"<div class='source-label'>Referenced documents</div>"
                f"{source_names}"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.session_state.messages.append({
        "role":    "assistant",
        "content": answer,
        "sources": sources,
    })

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    if api_key:
        st.success("✅ GROQ_API_KEY loaded")
    else:
        st.error("❌ GROQ_API_KEY missing")

    st.divider()
    st.subheader("📂 Knowledge Base")
    data_dir = "data"
    if os.path.exists(data_dir):
        files = sorted(
            f for f in os.listdir(data_dir)
            if f.endswith(".txt") or f.endswith(".json")
        )
        if files:
            for f in files:
                icon = "📄" if f.endswith(".txt") else "📋"
                st.markdown(f"{icon} `{f}`")
        else:
            st.warning("No .txt or .json files found in data/")
    else:
        st.warning("data/ directory not found.")

    st.divider()
    st.subheader("🗄️ Vector Store")
    if vs_exists:
        st.success("✅ Ready")
    else:
        st.error("❌ Run python ingest.py")

    st.divider()
    st.subheader("🧠 Reasoning")
    st.info("Strategy: **ReAct**\nReason → Retrieve → Answer")

    st.divider()
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("RAG AI Assistant · Joram Kirubi")