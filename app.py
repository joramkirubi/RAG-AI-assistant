"""
app.py
Streamlit chatbot interface for the RAG Assistant.
Shows the AI answer AND which documents were referenced.

Run with:
    streamlit run app.py
"""

import os
from dotenv import load_dotenv
import streamlit as st
from rag_chain import ask

# Load .env so the API key is available
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
    "ChromaDB &nbsp;·&nbsp; HuggingFace Embeddings &nbsp;·&nbsp; Groq LLM"
    "</span>",
    unsafe_allow_html=True,
)
st.divider()

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []  # {role, content, sources}

# ── Check setup on load ───────────────────────────────────────────────────────
api_key   = os.environ.get("GROQ_API_KEY", "")
vs_exists = os.path.exists("vectorstore")

if not api_key:
    st.error("⚠️ **GROQ_API_KEY not found.** Copy `.env.example` to `.env` and add your key.")
    st.stop()

if not vs_exists:
    st.error("⚠️ **Vector store not found.** Run `python ingest.py` first.")
    st.stop()

# ── Render existing chat history ──────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Show sources for assistant messages
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

    # User message
    st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant response
    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating answer..."):
            try:
                result  = ask(prompt)
                answer  = result["answer"]
                sources = result["sources"]
            except FileNotFoundError as e:
                answer  = f"❌ {e}"
                sources = []
            except EnvironmentError as e:
                answer  = f"❌ {e}"
                sources = []
            except Exception as e:
                answer  = f"❌ Unexpected error: {e}"
                sources = []

        st.markdown(answer)

        # Source citation block
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

    # API key status
    if api_key:
        st.success("✅ GROQ_API_KEY loaded from .env")
    else:
        st.error("❌ GROQ_API_KEY missing")

    st.divider()

    # Loaded documents
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

    # Vector store status
    st.subheader("🗄️ Vector Store")
    if vs_exists:
        st.success("✅ Vector store ready")
    else:
        st.error("❌ Not built — run python ingest.py")

    st.divider()

    # Clear chat
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("RAG AI Assistant · Joram Kirubi")
