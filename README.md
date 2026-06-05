# 🤖 RAG-Based AI Assistant Chatbot

A production-quality **Retrieval-Augmented Generation (RAG)** chatbot built with Python, LangChain, ChromaDB, HuggingFace Embeddings, and Groq LLM.

---

## 📁 Project Structure

```
rag-assistant/
├── app.py          ← Streamlit chatbot UI
├── ingest.py       ← Document loading, chunking & embedding pipeline
├── rag_chain.py    ← RAG chain (retriever + prompt + Groq LLM)
├── vectordb.py     ← ChromaDB setup and retriever
├── data/
│   └── docs.txt    ← Your knowledge base documents
├── vectorstore/    ← Auto-created by ingest.py (ChromaDB data)
├── requirements.txt
└── README.md
```

---

## ⚙️ How RAG Works

```
User Question
     ↓
Embed question with HuggingFace model
     ↓
Search ChromaDB for top-4 similar chunks
     ↓
Inject retrieved context into prompt
     ↓
Groq LLM (LLaMA 3) generates answer
     ↓
Answer shown in Streamlit chat UI
```

---

## 🚀 Setup & Run (Kali Linux Terminal)

### Step 1 — Clone / Create the project folder

```bash
mkdir ~/rag-assistant
cd ~/rag-assistant
```

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

> If you get permission errors, use:
> ```bash
> pip install -r requirements.txt --break-system-packages
> ```

### Step 3 — Set your Groq API Key

Get a free key at: https://console.groq.com

```bash
export GROQ_API_KEY="your_groq_api_key_here"
```

To make it permanent, add it to your shell profile:
```bash
echo 'export GROQ_API_KEY="your_key_here"' >> ~/.bashrc
source ~/.bashrc
```

### Step 4 — Add your documents

Place `.txt` files inside the `data/` folder.  
A sample `data/docs.txt` is already included.

### Step 5 — Run the ingestion pipeline

```bash
python ingest.py
```

This will:
- Load all `.txt` files from `data/`
- Split them into chunks
- Embed them using HuggingFace `all-MiniLM-L6-v2`
- Store embeddings in `vectorstore/` (ChromaDB)

### Step 6 — Launch the chatbot

```bash
streamlit run app.py
```

Open your browser to: **http://localhost:8501**

---

## 💬 Example Questions

After ingesting the sample `docs.txt`, try asking:

- *"What is RAG and how does it work?"*
- *"What is ChromaDB used for?"*
- *"Explain the difference between machine learning and deep learning."*
- *"What is LangChain?"*
- *"What is Groq?"*

---

## 🧩 Tech Stack

| Component | Library/Tool |
|---|---|
| Language | Python 3.10+ |
| LLM | Groq API (LLaMA 3 8B) |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Vector DB | ChromaDB |
| RAG Framework | LangChain |
| UI | Streamlit |

---

## 🛠️ Customisation

**Change the LLM model** — Edit `rag_chain.py`:
```python
chain = build_rag_chain(model="mixtral-8x7b-32768")
```

Available Groq models: `llama3-8b-8192`, `llama3-70b-8192`, `mixtral-8x7b-32768`, `gemma-7b-it`

**Add more documents** — Drop `.txt` files into `data/` and re-run:
```bash
python ingest.py
```

**Change chunk size** — Edit `ingest.py`:
```python
CHUNK_SIZE = 500    # characters per chunk
CHUNK_OVERLAP = 50  # overlap between chunks
```

---

## ✅ Checklist

- [ ] `pip install -r requirements.txt` completed
- [ ] `GROQ_API_KEY` exported in terminal
- [ ] `python ingest.py` ran successfully
- [ ] `streamlit run app.py` launched
- [ ] Questions answered correctly from documents

---

*Capstone Project — RAG AI Assistant*
