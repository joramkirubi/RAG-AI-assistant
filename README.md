RAG-Based AI Assistant Chatbot 

Overview

This project is a Retrieval-Augmented Generation (RAG) system that allows users to chat with their own documents. It combines document retrieval with a large language model to generate grounded, context-aware responses.

The system retrieves relevant document chunks from a vector database and uses them as context for an LLM to answer user questions.

Architecture

The system follows this flow:

User Query → Streamlit Interface → Retriever (ChromaDB) → Relevant Chunks → Prompt Builder → Groq LLM → Response + Sources

Features
Retrieval-Augmented Generation (RAG) pipeline
Document ingestion from .txt and .json files
Text chunking with overlap for better context retrieval
Embedding generation using HuggingFace sentence transformers
Vector storage using ChromaDB
LLM responses powered by Groq
Streamlit-based chat interface
Source tracking for retrieved answers
Environment variable support for API keys
Project Structure
rag-assistant/
│
├── app.py              # Streamlit user interface
├── rag_chain.py        # RAG pipeline (retriever + LLM)
├── ingest.py           # Document ingestion and embedding
├── vectordb.py         # Vector database setup and embeddings
├── data/               # Input documents (.txt, .json)
├── vectorstore/        # Persisted ChromaDB storage
├── requirements.txt
├── .env.example
└── README.md
How It Works
Document Ingestion

Run the ingestion script to process documents:

python ingest.py

This step:

Loads documents from the data/ folder
Splits them into chunks
Generates embeddings
Stores them in ChromaDB
Running the Application

Start the Streamlit app:

streamlit run app.py
Environment Variables

Create a .env file in the root directory:

GROQ_API_KEY=your_api_key_here

A .env.example file is included for reference.

Example Queries
What is the content of the documents?
Summarize the Ready Tensor publication
What are the project requirements?
Explain retrieval augmented generation
Tech Stack
Python
LangChain
ChromaDB
Groq LLM (LLaMA 3.1)
HuggingFace Sentence Transformers
Streamlit
Design Choices
Retrieval-Augmented Generation

RAG was chosen instead of fine-tuning because it:

Reduces computational cost
Works well with private documents
Improves explainability of responses
Vector Database

ChromaDB was used because:

It is lightweight and local
Easy integration with LangChain
Persistent storage support
LLM Provider

Groq was selected due to:

Fast inference speed
Low latency responses
Compatibility with LLaMA models
Evaluation Readiness

The system satisfies the core requirements:

RAG-based architecture implemented
Vector database integration
Document embedding pipeline
Working retrieval + generation system
Multi-format document ingestion
Reproducible setup
Source-grounded responses
Security Considerations
API keys are stored in environment variables
Sensitive files are excluded from version control
No hardcoded credentials in the codebase
Future Improvements
Add PDF document support
Improve retrieval with reranking models
Add conversation memory
Deploy to cloud platforms
Improve citation display in UI
Author

This project was built as part of an AI/RAG capstone submission to demonstrate practical implementation of retrieval-augmented generation systems.
