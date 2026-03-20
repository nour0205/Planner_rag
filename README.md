# Planner RAG: Retrieval-Augmented Generation with Planner Routing

Planner RAG is a production-style AI backend that answers questions using only ingested knowledge. It combines Retrieval-Augmented Generation (RAG), planner-based routing, reranking, and source-grounded generation to reduce hallucinations and make answers traceable.

The system is designed around a **planner → retriever → generator** pipeline, with explicit routing for single-document questions, multi-document comparisons, and unsupported queries.

---

## What the project does

This project lets you:

- ingest documents into a persistent vector database
- prevent duplicate ingestion using SHA256 hashing
- route questions through a planner
- retrieve relevant chunks from ChromaDB
- rerank retrieved chunks for better relevance
- generate answers only from retrieved evidence
- return source chunks with every grounded answer
- explore the knowledge base from a simple Streamlit frontend

---

## Architecture

The system follows this flow:

```text
User Question
    ↓
Planner Agent
    ↓
Execution Router
    ↓
Vector Retrieval (ChromaDB)
    ↓
Reranking
    ↓
LLM Generation
    ↓
Answer + Sources
```

---

## Routing behavior

The planner chooses one of three routes:

- `single` → answer from one document
- `multi` → compare or combine evidence from multiple documents
- `unknown` → refuse unsupported or unsafe questions

This makes the system safer and easier to evaluate than sending every query through the same retrieval path.

---

## Key features

### Persistent knowledge base

Documents are chunked, embedded, and stored in ChromaDB with metadata such as document ID, chunk index, source, owner, and document hash.

### Safe planner routing

Questions are first classified into routing strategies before retrieval starts.

### Multi-document reasoning

The system can compare concepts across multiple documents, such as:

```text
Compare PostgreSQL MVCC with SQL Server snapshot isolation.
```

### Evidence-based answers

Answers are generated from retrieved context only, with source chunks returned in the response.

### Retrieval reranking

Initial retrieved chunks are reranked to improve answer quality.

### Ingestion safety

Duplicate documents are detected by content hash, and document ID collisions are also prevented.

### Evaluation suite

Automated tests check planner routing, route behavior, answer grounding, and source attribution.

### Streamlit frontend

A lightweight UI lets you:

- ingest documents
- ask routed questions
- inspect answers and sources
- browse the ingested knowledge base

---

## Project structure

```text
app/
│
├── api/            FastAPI endpoints
├── embeddings/     Embedding interface
├── ingestion/      Document chunking
├── llm/            LLM client
├── orchestration/  Planner + routing logic
├── rag/            Retrieval + generation pipeline
├── vectordb/       Chroma vector store
└── utils/          Shared utilities

frontend/
└── app.py          Streamlit frontend
```

### Top-level files

```text
eval_cases.json     Evaluation test cases
run_eval.py         Automated evaluation script
requirements.txt    Dependencies
README.md
```

---

## API endpoints

### `POST /ingest`

Ingest a document into the knowledge base.

#### Example request

```json
{
  "document_id": "db_postgres",
  "text": "PostgreSQL uses MVCC..."
}
```

#### Possible statuses

- `ingested`
- `duplicate`
- `conflict`
- `no content`

---

### `POST /ask`

Single-document RAG query.

---

### `POST /ask_routed`

Question answering with planner-based routing.

#### Example response

```json
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

### `GET /documents`

Returns a grouped summary of ingested documents in the knowledge base.

---

### `GET /documents/{document_id}`

Returns all stored chunks for a specific document.

---

## Running the project

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the FastAPI backend

```bash
python -m uvicorn app.api.main:app --reload
```

Backend URL:

```text
http://127.0.0.1:8000
```

Interactive docs:

```text
http://127.0.0.1:8000/docs
```

### 3. Start the Streamlit frontend

```bash
python -m streamlit run frontend/app.py
```

Frontend usually runs at:

```text
http://localhost:8501
```

---

## Frontend features

The Streamlit app includes:

- **Ask Questions** tab for routed question answering
- **Ingest Document** tab for adding new knowledge
- **Knowledge Base** tab for viewing ingested documents and chunk previews

This makes the system easier to demo, debug, and inspect.

---

## Evaluation

Run the automated evaluation suite with:

```bash
python run_eval.py
```

The evaluation checks:

- planner routing correctness
- API route decisions
- answer behavior
- source presence

### Example output

```text
Planner: 4/4
API route: 4/4
Answer behavior: 4/4
Sources presence: 4/4
```

---

## Example use cases

This project can be adapted into systems like:

- technical documentation assistant
- internal company knowledge assistant
- research comparison tool
- course policy Q&A system
- domain-specific document analysis assistant

---

## Current limitations

- document routing depends on a curated registry
- planner coverage is limited to supported domains
- retrieval quality depends on chunking and metadata quality
- reranking is basic and can be improved
- unsupported new domains require manual registry updates

---

## Future improvements

Possible next steps:

- dynamic document registry from ingestion metadata
- hybrid search (BM25 + vector retrieval)
- cross-encoder reranking
- query rewriting
- streaming responses
- document deletion and management tools
- richer frontend analytics and debugging views

---

## License

MIT License