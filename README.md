# AI Agent Platform: Retrieval-Augmented Generation with Planner Routing

This project implements a **production-style Retrieval-Augmented Generation (RAG) backend** with safe routing, multi-document reasoning, and source-grounded answers.

The system answers questions **only using ingested knowledge**, avoiding hallucinations by combining:

* vector retrieval
* planner-based orchestration
* evidence-based prompting
* structured source attribution

---

# Architecture

The system is built around a **planner → retriever → generator** pipeline.

User Question
↓
Planner Agent
↓
Execution Router
↓
Vector Retrieval (Chroma)
↓
Reranking
↓
LLM Generation
↓
Answer + Sources

---

# Key Features

### Persistent Knowledge Base

Documents are embedded and stored in **ChromaDB** with metadata.

### Safe Planner Routing

A planner agent decides how to answer each question:

* **single** → retrieve from one document
* **multi** → compare multiple documents
* **unknown** → refuse unsafe questions

### Multi-Document Reasoning

Supports comparisons like:

```
Compare PostgreSQL MVCC with SQL Server snapshot isolation
```

### Evidence-Based Answers

All responses are grounded in retrieved context and return **source chunks**.

### Retrieval Reranking

Retrieved chunks are reranked to improve relevance before generation.

### Ingestion Safety

Documents are protected against duplication using SHA256 hashing.

### Evaluation Suite

Automated tests validate:

* planner routing
* API behavior
* answer correctness
* source attribution

---

# Project Structure

```
app/
│
├ api/            FastAPI endpoints
├ embeddings/     Embedding interface
├ ingestion/      Document chunking
├ llm/            LLM client
├ orchestration/  Planner + routing logic
├ rag/            Retrieval + generation pipeline
├ vectordb/       Chroma vector store
└ utils/          Shared utilities
```

Top-level files:

```
eval_cases.json   Evaluation test cases
run_eval.py       Automated evaluation script
requirements.txt  Dependencies
README.md
```

---

# API Endpoints

## Ingest document

```
POST /ingest
```

Example:

```
{
  "document_id": "db_postgres",
  "text": "PostgreSQL uses MVCC..."
}
```

---

## Ask a question

```
POST /ask
```

Single-document RAG query.

---

## Ask with routing

```
POST /ask_routed
```

Uses the planner to decide how to answer.

Example response:

```
{
  "answer": "...",
  "route": "single",
  "sources": [
    {
      "document_id": "db_postgres",
      "chunk_index": 1,
      "text": "MVCC works by keeping multiple versions..."
    }
  ]
}
```

---

# Running the system

## Install dependencies

```
pip install -r requirements.txt
```

---

## Start the API

```
uvicorn app.api.main:app --reload
```

Server runs at:

```
http://127.0.0.1:8000
```

Interactive docs:

```
http://127.0.0.1:8000/docs
```

---

# Evaluation

Run the automated evaluation suite:

```
python run_eval.py
```

The script validates:

* planner routing
* API route decisions
* answer behavior
* source attribution

Example output:

```
Planner: 4/4
API route: 4/4
Answer behavior: 4/4
Sources presence: 4/4
```

---

# Future Improvements

Possible next steps:

* hybrid search (BM25 + vectors)
* cross-encoder reranking
* query rewriting
* streaming responses
* UI for document ingestion

---

# License

MIT License
