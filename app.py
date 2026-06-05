"""
app.py
Streamlit chatbot interface for the RAG Assistant.

Run with:
    streamlit run app.py
"""

import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from rag_chain import build_rag_chain, ask

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG AI Assistant",
    page_icon="🤖",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .stApp { background-color: #0f1117; }
    .chat-title { font-size: 2rem; font-weight: 700; color: #f0f0f0; }
    .chat-sub { color: #888; font-size: 0.9rem; margin-bottom: 1.5rem; }
    .stChatMessage { border-radius: 12px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<p class="chat-title">🤖 RAG AI Assistant</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="chat-sub">Powered by ChromaDB · HuggingFace Embeddings · Groq LLM</p>',
    unsafe_allow_html=True,
)
st.divider()

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chain" not in st.session_state:
    with st.spinner("Loading RAG pipeline..."):
        try:
            st.session_state.chain = build_rag_chain()
            st.session_state.chain_error = None
        except FileNotFoundError as e:
            st.session_state.chain = None
            st.session_state.chain_error = str(e)
        except EnvironmentError as e:
            st.session_state.chain = None
            st.session_state.chain_error = str(e)

# ── Error banner ──────────────────────────────────────────────────────────────
if st.session_state.get("chain_error"):
    st.error(f"⚠️ Setup Error:\n\n{st.session_state.chain_error}")
    st.info("Fix the error above, then refresh this page.")
    st.stop()

# ── Chat history ──────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Input ─────────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask me anything about your documents..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer, sources = ask(prompt, st.session_state.chain)
            except Exception as e:
                answer = f"❌ Error generating response: {e}"
                sources = []

        st.markdown(answer)

        # ── Show Sources ───────────────────────────────────────────────
        if sources:
            with st.expander("📚 Sources"):
                seen = set()
                for doc in sources:
                    src = doc.metadata.get("source", "Unknown")
                    if src not in seen:
                        st.markdown(f"- `{src}`")
                        seen.add(src)

    st.session_state.messages.append({"role": "assistant", "content": answer})

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    groq_key = os.environ.get("GROQ_API_KEY", "")
    if groq_key:
        st.success("✅ GROQ_API_KEY is set")
    else:
        st.error("❌ GROQ_API_KEY not set")
        st.code("export GROQ_API_KEY='your_key'", language="bash")

    st.divider()
    st.subheader("📂 Knowledge Base")
    data_dir = "data"
    if os.path.exists(data_dir):
        files = [f for f in os.listdir(data_dir) if f.endswith((".txt", ".json"))]
        for f in files:
            st.markdown(f"📄 `{f}`")
    else:
        st.warning("No data/ directory found.")

    st.divider()
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("RAG Assistant · Capstone Project")