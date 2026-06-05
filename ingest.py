"""
ingest.py
Loads documents from data/, splits them into chunks,
embeds them, and stores them in ChromaDB.

Run once before starting the chatbot:
    python ingest.py
"""

import os
import json
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from vectordb import get_embeddings, VECTORSTORE_DIR, COLLECTION_NAME


DATA_DIR = "data"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def load_documents(data_dir: str):
    print(f"[INFO] Loading documents from '{data_dir}/'...")

    documents = []

    for file in os.listdir(data_dir):
        file_path = os.path.join(data_dir, file)

        if file.endswith(".txt"):
            loader = TextLoader(file_path, encoding="utf-8")
            documents.extend(loader.load())

        elif file.endswith(".json"):
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # extract useful text
            if isinstance(data, dict):
                text = data.get("content") or data.get("text") or str(data)

                documents.append(
                    type("Doc", (), {"page_content": text, "metadata": {"source": file}})()
                )

            elif isinstance(data, list):
                for item in data:
                    text = item.get("content") or item.get("text") or str(item)

                    documents.append(
                        type("Doc", (), {"page_content": text, "metadata": {"source": file}})()
                    )

    print(f"[INFO] Loaded {len(documents)} document(s).")
    return documents


def split_documents(documents):
    """Split documents into smaller overlapping chunks."""
    print(f"[INFO] Splitting into chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"[INFO] Created {len(chunks)} chunks.")
    return chunks


def build_vectorstore(chunks, embeddings):
    """Embed chunks and persist them into ChromaDB."""
    print(f"[INFO] Embedding chunks and saving to '{VECTORSTORE_DIR}/'...")

    # Remove old store if it exists so we start fresh
    if os.path.exists(VECTORSTORE_DIR):
        import shutil
        shutil.rmtree(VECTORSTORE_DIR)
        print(f"[INFO] Cleared existing vector store.")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=VECTORSTORE_DIR,
    )
    print(f"[INFO] Vector store saved to '{VECTORSTORE_DIR}/'.")
    return vectorstore


def main():
    print("=" * 50)
    print("   RAG Ingestion Pipeline")
    print("=" * 50)

    # Step 1: Load
    documents = load_documents(DATA_DIR)
    if not documents:
        print("[ERROR] No documents found in data/. Add .txt or .json files and retry.")
        return

    # Step 2: Split
    chunks = split_documents(documents)

    # Step 3: Embed and store
    embeddings = get_embeddings()
    build_vectorstore(chunks, embeddings)

    print("\n[SUCCESS] Ingestion complete! You can now run the chatbot.")
    print("   Streamlit : streamlit run app.py")
    print("=" * 50)


if __name__ == "__main__":
    main()