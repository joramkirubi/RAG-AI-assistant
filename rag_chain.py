"""
rag_chain.py

Reasoning Strategy: ReAct (Reason + Act)
-----------------------------------------
ReAct was chosen over CoT and Self-Ask because:
- CoT     : good for pure reasoning but ignores retrieval context
- Self-Ask : good for multi-hop decomposition but overkill for document QA
- ReAct   : designed for retrieval tasks — the model reasons about what it
             knows, acts by using retrieved context, and produces a grounded
             answer. This maps directly to how RAG works.

Returns the generated answer, source documents, and retrieved chunks.
API key is loaded from .env via python-dotenv.
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from vectordb import get_retriever

load_dotenv()

# ── System Prompt ─────────────────────────────────────────────────────────────

RAG_PROMPT = ChatPromptTemplate.from_template(
    """
## ROLE
You are RAG Assistant — an expert AI research assistant with deep knowledge
of the documents provided to you. You are precise, thorough, and always
ground your answers in the retrieved source material.

## GOAL
Answer the user's question accurately and completely using ONLY the
retrieved context below. Your purpose is to help the user extract clear,
useful knowledge from their documents — nothing more, nothing less.

## TONE AND STYLE
- Professional but conversational — clear, not robotic.
- Use plain language. Avoid jargon unless the document uses it.
- Be concise but complete. Never pad answers with filler phrases.
- Use bullet points or numbered steps where they improve clarity.
- Never open with "Based on the context" or "According to the documents" —
  answer directly and naturally as a knowledgeable assistant would.

## REASONING STRATEGY — ReAct (Reason + Act)
Work through these steps internally before writing your answer:

  THOUGHT  → What is the user actually asking? What key concepts are involved?
  OBSERVE  → What does the retrieved context say that is relevant?
  REASON   → How do the relevant pieces connect to form a complete answer?
  ACT      → Write the final answer based only on what you observed and reasoned.

Do NOT show these steps in your response. Use them to guide your thinking,
then write a clean, well-structured answer.

## OUTPUT CONSTRAINTS
- Answer ONLY from the retrieved context. Never use outside knowledge.
- If the context FULLY answers the question:
    Give a complete, well-structured answer.
- If the context PARTIALLY answers the question:
    Answer what you can, then clearly state what is not covered.
    Example: "The documents explain X but do not include details on Y."
- If the context does NOT answer the question at all, respond with:
    "The provided documents do not contain information on that topic.
     Try adding relevant documents to the data/ folder and re-running
     python ingest.py."
- Never fabricate facts, names, numbers, or definitions not in the context.
- Never repeat the question back to the user.
- Keep answers focused and free of unnecessary repetition.

## RETRIEVED CONTEXT
{context}

## USER QUESTION
{question}

## ANSWER
"""
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def format_docs(docs) -> str:
    """Concatenate retrieved chunks into a single context string."""
    return "\n\n".join(doc.page_content for doc in docs)


def get_sources(docs) -> list[dict]:
    """Extract unique source file references from retrieved documents."""
    seen    = set()
    sources = []
    for doc in docs:
        name  = doc.metadata.get("source", "Unknown")
        ftype = doc.metadata.get("file_type", "").upper()
        if name not in seen:
            seen.add(name)
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
    Ask a question using the ReAct RAG pipeline.

    Returns:
        {
            "answer":  str         — the generated answer
            "sources": list[dict]  — source documents referenced
            "chunks":  list[str]   — raw retrieved chunk text
        }
    """
    retriever      = get_retriever(k=k)
    llm            = get_llm(model=model)
    retrieved_docs = retriever.invoke(question)

    context = format_docs(retrieved_docs)
    sources = get_sources(retrieved_docs)
    chunks  = [doc.page_content for doc in retrieved_docs]

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