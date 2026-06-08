"""
app.py
Medical AI Assistant — Streamlit Interface

Features:
- Conversation history passed to the LLM for follow-up question support
- Retrieval evaluation metrics displayed in the sidebar
- Source attribution on every answer
- Embedding model cached on startup to eliminate cold-start delay

Run with:
    python -m streamlit run app.py

If the above fails due to path conflicts:
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
    page_title="MedAssist — Medical AI Assistant",
    page_icon="🏥",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.stApp { background-color: #0f1117; }
.source-box {
    background-color: #0c1e2e;
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
.disclaimer {
    background-color: #1a1200;
    border: 1px solid #4a3800;
    border-left: 3px solid #c8960a;
    border-radius: 6px;
    padding: 8px 14px;
    margin-top: 6px;
    font-size: 0.78rem;
    color: #c8960a;
}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("## 🏥 MedAssist — Medical AI Assistant")
st.markdown(
    "<span style='color:#4a7a9b; font-size:0.9rem;'>"
    "Healthcare Knowledge Base &nbsp;·&nbsp; ChromaDB &nbsp;·&nbsp; "
    "Groq LLM &nbsp;·&nbsp; ReAct Reasoning"
    "</span>",
    unsafe_allow_html=True,
)
st.markdown(
    "<div class='disclaimer'>"
    "⚕️ <b>Medical Disclaimer:</b> This assistant provides general health information only. "
    "It is not a substitute for professional medical advice, diagnosis, or treatment. "
    "Always consult a qualified healthcare provider for personal medical decisions."
    "</div>",
    unsafe_allow_html=True,
)
st.divider()

# ── Setup checks ──────────────────────────────────────────────────────────────
api_key   = os.environ.get("GROQ_API_KEY", "")
vs_exists = os.path.exists("vectorstore")

if not api_key:
    st.error("⚠️ **GROQ_API_KEY not found.** Copy `.env.example` to `.env` and add your key.")
    st.stop()

if not vs_exists:
    st.error("⚠️ **Vector store not found.** Run `python ingest.py` first.")
    st.stop()

# ── Cache embedding model + vector store on startup ───────────────────────────
@st.cache_resource(show_spinner="Loading medical knowledge base... (first time only)")
def load_rag_components():
    embeddings  = get_embeddings()
    vectorstore = get_vectorstore(embeddings)
    return embeddings, vectorstore

load_rag_components()

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []      # full chat history {role, content, sources}

if "last_metrics" not in st.session_state:
    st.session_state.last_metrics = None

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
if prompt := st.chat_input("Ask a medical or health question..."):

    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Build history list for the LLM (exclude sources metadata)
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1]  # exclude current message
    ]

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result  = ask(prompt, history=history)
                answer  = result["answer"]
                sources = result["sources"]
                metrics = result["metrics"]
                st.session_state.last_metrics = metrics
            except FileNotFoundError as e:
                answer, sources, metrics = f"❌ {e}", [], None
            except EnvironmentError as e:
                answer, sources, metrics = f"❌ {e}", [], None
            except Exception as e:
                answer, sources, metrics = f"❌ Unexpected error: {e}", [], None

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
    st.header("⚙️ System Status")

    if api_key:
        st.success("✅ GROQ_API_KEY loaded")
    else:
        st.error("❌ GROQ_API_KEY missing")

    st.divider()

    # Knowledge base files
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
            st.warning("No documents found in data/")
    else:
        st.warning("data/ directory not found.")

    st.divider()

    # Vector store status
    st.subheader("🗄️ Vector Store")
    if vs_exists:
        st.success("✅ Ready")
    else:
        st.error("❌ Run python ingest.py")

    st.divider()

    # Reasoning strategy
    st.subheader("🧠 Reasoning")
    st.info("Strategy: **ReAct**\nReason → Retrieve → Answer")

    st.divider()

    # Retrieval metrics for last query
    st.subheader("📊 Last Query Metrics")
    m = st.session_state.last_metrics
    if m:
        col1, col2 = st.columns(2)
        col1.metric("Chunks", m["chunks_retrieved"])
        col2.metric("Sources", m["source_diversity"])
        col1.metric("Avg Length", f"{m['avg_chunk_length']}c")
        col2.metric("Coverage", m["coverage_score"])
    else:
        st.caption("Metrics appear after your first question.")

    st.divider()

    # Clear chat
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages      = []
        st.session_state.last_metrics  = None
        st.rerun()

    st.divider()
    st.caption("MedAssist · RAG AI Assistant · Joram Kirubi")