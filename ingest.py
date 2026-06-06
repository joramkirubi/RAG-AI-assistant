"""
ingest.py
Loads .txt and .json documents from data/, splits them into chunks,
embeds them with HuggingFace, and stores them in ChromaDB.

Run once before starting the chatbot, and re-run whenever you add new documents:
    python ingest.py
"""

import os
import json
import shutil
from pathlib import Path

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_community.vectorstores import Chroma

from vectordb import get_embeddings, VECTORSTORE_DIR, COLLECTION_NAME

DATA_DIR      = "data"
CHUNK_SIZE    = 600
CHUNK_OVERLAP = 80


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_txt_files(data_dir: str) -> list[Document]:
    """Load all .txt files, tagging each chunk with its source filename."""
    docs = []
    txt_files = list(Path(data_dir).glob("**/*.txt"))
    for path in txt_files:
        try:
            loader = TextLoader(str(path), encoding="utf-8")
            loaded = loader.load()
            for doc in loaded:
                doc.metadata["source"] = path.name
                doc.metadata["file_type"] = "txt"
            docs.extend(loaded)
            print(f"  [TXT] Loaded: {path.name}")
        except Exception as e:
            print(f"  [WARN] Could not load {path.name}: {e}")
    return docs


def load_json_files(data_dir: str) -> list[Document]:
    """
    Load all .json files.
    Handles both a single object and a list of objects.
    Each item is converted to a readable text block and tagged with its source.
    """
    docs = []
    json_files = list(Path(data_dir).glob("**/*.json"))
    for path in json_files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Normalise to a list of items
            items = data if isinstance(data, list) else [data]

            for i, item in enumerate(items):
                # Convert each JSON item to a readable key: value text block
                if isinstance(item, dict):
                    text = "\n".join(
                        f"{k}: {v}" for k, v in item.items()
                        if v is not None and str(v).strip() != ""
                    )
                else:
                    text = str(item)

                if text.strip():
                    doc = Document(
                        page_content=text,
                        metadata={
                            "source": path.name,
                            "file_type": "json",
                            "item_index": i,
                        },
                    )
                    docs.append(doc)

            print(f"  [JSON] Loaded: {path.name}  ({len(items)} item(s))")
        except Exception as e:
            print(f"  [WARN] Could not load {path.name}: {e}")
    return docs


def load_all_documents(data_dir: str) -> list[Document]:
    """Load all supported file types from data/."""
    print(f"\n[INFO] Scanning '{data_dir}/' for documents...")
    txt_docs  = load_txt_files(data_dir)
    json_docs = load_json_files(data_dir)
    all_docs  = txt_docs + json_docs
    print(f"[INFO] Total documents loaded: {len(all_docs)}")
    return all_docs


# ── Chunking ──────────────────────────────────────────────────────────────────

def split_documents(documents: list[Document]) -> list[Document]:
    """Split documents into overlapping chunks, preserving source metadata."""
    print(f"[INFO] Splitting into chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"[INFO] Total chunks created: {len(chunks)}")
    return chunks


# ── Embedding & Storage ───────────────────────────────────────────────────────

def build_vectorstore(chunks: list[Document], embeddings) -> None:
    """Embed all chunks and persist them to ChromaDB."""
    print(f"[INFO] Embedding and saving to '{VECTORSTORE_DIR}/'...")

    if os.path.exists(VECTORSTORE_DIR):
        shutil.rmtree(VECTORSTORE_DIR)
        print(f"[INFO] Cleared existing vector store.")

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=VECTORSTORE_DIR,
    )
    print(f"[INFO] Vector store saved successfully.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("   RAG Ingestion Pipeline")
    print("=" * 55)

    if not os.path.exists(DATA_DIR):
        print(f"[ERROR] '{DATA_DIR}/' directory not found. Create it and add documents.")
        return

    documents = load_all_documents(DATA_DIR)

    if not documents:
        print(f"[ERROR] No .txt or .json files found in '{DATA_DIR}/'. Add documents and retry.")
        return

    chunks    = split_documents(documents)
    embeddings = get_embeddings()
    build_vectorstore(chunks, embeddings)

    print("\n[SUCCESS] Ingestion complete!")
    print("   Run the chatbot with:  streamlit run app.py")
    print("=" * 55)


if __name__ == "__main__":
    main()
