"""
rag_chain.py
Builds the RAG chain using Groq LLM.
Returns both the generated answer and the source documents used.
API key is loaded from .env via python-dotenv.
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough, RunnableParallel

from vectordb import get_retriever

# Load .env file
load_dotenv()

# ── Prompt ────────────────────────────────────────────────────────────────────

RAG_PROMPT = ChatPromptTemplate.from_template(
    """You are a knowledgeable and precise AI assistant. Your job is to answer
questions accurately using ONLY the context provided below.

Rules:
- Answer in clear, well-structured sentences.
- If the context contains a direct answer, provide it fully and completely.
- If the context is partially relevant, use what is available and be transparent about it.
- If the answer is genuinely not in the context, say exactly:
  "I don't have enough information in the provided documents to answer that."
- Never make up facts or use knowledge outside the provided context.

Context:
{context}

Question: {question}

Answer:"""
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def format_docs(docs) -> str:
    """Concatenate document chunks into a single context string."""
    return "\n\n".join(doc.page_content for doc in docs)


def get_sources(docs) -> list[dict]:
    """
    Extract unique source references from retrieved documents.
    Returns a list of dicts with source filename and file type.
    """
    seen = set()
    sources = []
    for doc in docs:
        name = doc.metadata.get("source", "Unknown")
        ftype = doc.metadata.get("file_type", "").upper()
        key = name
        if key not in seen:
            seen.add(key)
            sources.append({"file": name, "type": ftype})
    return sources


def get_llm(model: str = "llama-3.1-8b-instant", temperature: float = 0.2) -> ChatGroq:
    """Initialise the Groq LLM. Reads GROQ_API_KEY from .env."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY is not set.\n"
            "1. Copy .env.example to .env\n"
            "2. Add your key: GROQ_API_KEY=gsk_...\n"
            "3. Get a free key at https://console.groq.com"
        )
    return ChatGroq(
        groq_api_key=api_key,
        model_name=model,
        temperature=temperature,
    )


# ── RAG Chain ─────────────────────────────────────────────────────────────────

def ask(question: str, k: int = 4, model: str = "llama-3.1-8b-instant") -> dict:
    """
    Ask a question and return a dict with:
        {
            "answer":  str,          the generated answer
            "sources": list[dict],   source documents used
            "chunks":  list[str],    raw text of retrieved chunks
        }
    """
    retriever = get_retriever(k=k)
    llm       = get_llm(model=model)

    # Retrieve docs first so we can return them alongside the answer
    retrieved_docs = retriever.invoke(question)

    context = format_docs(retrieved_docs)
    sources = get_sources(retrieved_docs)
    chunks  = [doc.page_content for doc in retrieved_docs]

    # Build the prompt and call the LLM
    prompt_value = RAG_PROMPT.format_messages(
        context=context,
        question=question,
    )
    answer = llm.invoke(prompt_value).content

    return {
        "answer":  answer,
        "sources": sources,
        "chunks":  chunks,
    }
