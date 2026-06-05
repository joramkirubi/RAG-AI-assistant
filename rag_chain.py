"""
rag_chain.py
Builds the RAG chain: retriever + prompt + Groq LLM.
"""

import os
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from vectordb import get_retriever


# ── Prompt Template ──────────────────────────────────────────────────────────

RAG_PROMPT = ChatPromptTemplate.from_template(
    """You are a helpful AI assistant. Answer the user's question using ONLY
the context provided below. If the answer is not in the context, say:
"I don't have enough information in my documents to answer that."

Context:
{context}

Question: {question}

Answer:"""
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def format_docs(docs):
    """Concatenate retrieved document chunks into a single context string."""
    return "\n\n".join(doc.page_content for doc in docs)


def get_llm(model: str = "llama3-8b-8192", temperature: float = 0.2):
    """Initialise the Groq LLM."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY is not set.\n"
            "Export it first:  export GROQ_API_KEY='your_key_here'"
        )
    llm = ChatGroq(
        groq_api_key=api_key,
        model_name=model,
        temperature=temperature,
    )
    return llm


# ── RAG Chain ─────────────────────────────────────────────────────────────────

def build_rag_chain(model: str = "llama-3.1-8b-instant"):
    """
    Returns a runnable RAG chain.

    Usage:
        chain = build_rag_chain()
        answer = chain.invoke("What is RAG?")
    """
    retriever = get_retriever(k=4)
    llm = get_llm(model=model)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    return chain


def ask(question: str, chain=None) -> str:
    """Ask a question and return the answer string."""
    if chain is None:
        chain = build_rag_chain()
    return chain.invoke(question)
