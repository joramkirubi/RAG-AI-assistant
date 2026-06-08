"""
rag_chain.py

Domain: Healthcare / Medical AI Assistant
Reasoning Strategy: ReAct (Reason + Act)
-----------------------------------------
ReAct was chosen over CoT and Self-Ask because:
- CoT      : good for pure reasoning but ignores retrieval context
- Self-Ask : good for multi-hop decomposition, overkill for document QA
- ReAct    : designed for retrieval tasks — reason about the query, act
             by using retrieved context, produce a grounded answer.
             Maps directly to how RAG works.

Features:
- Healthcare-specific system prompt with medical accuracy constraints
- Retrieval evaluation metrics (relevance scoring, source diversity)
- Conversation history support for follow-up questions
- Source attribution on every answer
- API key loaded from .env via python-dotenv
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
You are MedAssist — a knowledgeable and precise AI healthcare assistant.
You support patients, caregivers, and healthcare students by answering
medical and health-related questions clearly and accurately using only
the provided medical documents.

You are NOT a replacement for a qualified medical professional. Always
encourage users to consult a licensed doctor for personal medical advice,
diagnoses, or treatment decisions.

## GOAL
Answer the user's health question accurately and completely using ONLY
the retrieved medical context below. Help the user understand their
condition, medication, or health topic as clearly as possible based on
the available documents.

## TONE AND STYLE
- Warm, clear, and professional — like a knowledgeable healthcare educator.
- Use plain language. Spell out medical terms when first used.
  Example: "hypertension (high blood pressure)"
- Be thorough but concise. Avoid unnecessary filler phrases.
- Use numbered steps for procedures and bullet points for lists of symptoms,
  medications, or risk factors — this improves readability for health content.
- Never open with "Based on the context" or "According to the documents."
  Answer naturally and directly.
- Always add a brief safety note when the topic involves emergency symptoms,
  medications, or dosage instructions.

## REASONING STRATEGY — ReAct (Reason + Act)
Work through these steps internally before writing your answer:

  THOUGHT  → What health topic is the user asking about? What do they need to know?
  OBSERVE  → What does the retrieved medical context say that is directly relevant?
  REASON   → How do the relevant pieces connect? Is the context complete or partial?
  ACT      → Write a clear, well-structured medical answer grounded in the context.

Do NOT show these steps in your response. Use them to guide your thinking,
then write a clean, helpful answer.

## OUTPUT CONSTRAINTS
- Answer ONLY from the retrieved context. Never use outside medical knowledge.
- If the context FULLY answers the question:
    Give a complete, well-structured answer with a safety note if relevant.
- If the context PARTIALLY answers the question:
    Answer what is covered and clearly state: "For more detail on [topic],
    please consult your healthcare provider or a current medical reference."
- If the context does NOT answer the question at all, respond with:
    "The medical documents provided do not cover that specific topic.
     Please consult a qualified healthcare provider for accurate guidance."
- Never fabricate drug names, dosages, symptoms, or clinical guidelines.
- Never provide a personal diagnosis or tell a user they have a specific condition.
- Always recommend consulting a healthcare professional for personal medical decisions.
- Keep answers focused and free of unnecessary repetition.

## CONVERSATION HISTORY
{history}

## RETRIEVED MEDICAL CONTEXT
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


def format_history(history: list[dict]) -> str:
    """Format conversation history for injection into the prompt."""
    if not history:
        return "No previous conversation."
    lines = []
    for msg in history[-6:]:  # keep last 3 exchanges (6 messages)
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


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


def evaluate_retrieval(question: str, docs) -> dict:
    """
    Retrieval Evaluation Metrics.

    Measures the quality of the retrieved chunks to help assess
    how well the vector store is serving each query.

    Metrics:
    - chunks_retrieved : number of chunks returned
    - avg_chunk_length : average character length of retrieved chunks
    - source_diversity : number of unique source documents used
    - coverage_score   : ratio of unique sources to chunks (0.0 - 1.0)
                         Higher = more diverse sourcing, less redundancy
    - query_length     : number of words in the user question
    """
    if not docs:
        return {
            "chunks_retrieved": 0,
            "avg_chunk_length": 0,
            "source_diversity": 0,
            "coverage_score":   0.0,
            "query_length":     len(question.split()),
        }

    unique_sources = len(set(d.metadata.get("source", "") for d in docs))
    avg_len        = sum(len(d.page_content) for d in docs) // len(docs)
    coverage       = round(unique_sources / len(docs), 2)

    return {
        "chunks_retrieved": len(docs),
        "avg_chunk_length": avg_len,
        "source_diversity": unique_sources,
        "coverage_score":   coverage,
        "query_length":     len(question.split()),
    }


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

def ask(
    question: str,
    history:  list[dict] = None,
    k:        int        = 5,
    model:    str        = "llama-3.1-8b-instant",
) -> dict:
    """
    Ask a medical question using the ReAct RAG pipeline.

    Args:
        question : the user's question
        history  : list of prior messages [{role, content}, ...]
        k        : number of chunks to retrieve (default 5 for medical depth)
        model    : Groq model name

    Returns:
        {
            "answer"   : str         — the generated answer
            "sources"  : list[dict]  — source documents referenced
            "chunks"   : list[str]   — raw retrieved chunk text
            "metrics"  : dict        — retrieval evaluation metrics
        }
    """
    if history is None:
        history = []

    retriever      = get_retriever(k=k)
    llm            = get_llm(model=model)
    retrieved_docs = retriever.invoke(question)

    context  = format_docs(retrieved_docs)
    sources  = get_sources(retrieved_docs)
    chunks   = [doc.page_content for doc in retrieved_docs]
    metrics  = evaluate_retrieval(question, retrieved_docs)
    hist_str = format_history(history)

    prompt_value = RAG_PROMPT.format_messages(
        context=context,
        question=question,
        history=hist_str,
    )
    answer = llm.invoke(prompt_value).content

    return {
        "answer":  answer,
        "sources": sources,
        "chunks":  chunks,
        "metrics": metrics,
    }