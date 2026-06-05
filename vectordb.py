"""
vectordb.py
Handles ChromaDB vector store setup and retriever creation.
"""

import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

VECTORSTORE_DIR = "vectorstore"
COLLECTION_NAME = "rag_documents"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_embeddings():
    """Load the HuggingFace embedding model safely."""
    print("[INFO] Loading embedding model...")

    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={
                "device": "cpu",
                "trust_remote_code": False
            },
            encode_kwargs={
                "normalize_embeddings": True
            },
        )
        return embeddings

    except Exception as e:
        raise RuntimeError(
            f"Failed to load embedding model: {e}\n"
            "Fix: ensure torch + sentence-transformers are correctly installed "
            "and you're using Python 3.11 (NOT 3.13)."
        )


def get_vectorstore(embeddings=None):
    """Load existing ChromaDB vector store from disk."""
    if embeddings is None:
        embeddings = get_embeddings()

    if not os.path.exists(VECTORSTORE_DIR):
        raise FileNotFoundError(
            f"Vector store not found at '{VECTORSTORE_DIR}'.\n"
            "Please run: python ingest.py"
        )

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=VECTORSTORE_DIR,
    )

    return vectorstore


def get_retriever(k=4):
    """Return a retriever that fetches top-k relevant chunks."""
    vectorstore = get_vectorstore()
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )