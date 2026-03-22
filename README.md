# Planner RAG: Retrieval-Augmented Generation with Planner Routing

Planner RAG is a production-style AI backend for source-grounded question answering over ingested knowledge.

It combines **Retrieval-Augmented Generation (RAG)**, **planner-based routing**, **retrieval reranking**, and **evidence-constrained generation** to reduce hallucinations and make answers traceable to retrieved source chunks.

Unlike basic RAG systems that send every query through the same retrieval path, Planner RAG first decides **how the question should be handled**. It supports:

- **single-document retrieval** for focused questions
- **multi-document retrieval** for comparisons and synthesis
- **unsupported-query refusal** when the answer cannot be grounded safely

This makes the system more controllable, more explainable, and easier to evaluate.

## Overview

The project is built around a **planner → retriever → generator** pipeline:

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

The planner decides whether a question should be answered from one document, multiple documents, or refused if it falls outside the supported knowledge base.

## Key Features

### Planner-based routing

Questions are classified before retrieval begins. The planner chooses one of three routes:

- `single` — answer using one target document
- `multi` — compare or combine evidence from multiple documents
- `unknown` — refuse unsupported or unsafe questions

This routing layer makes the system safer and more structured than a one-size-fits-all RAG pipeline.

### Persistent vector knowledge base

Documents are chunked, embedded, and stored in **ChromaDB** with persistent metadata, including:

- document ID
- chunk index
- source
- owner
- document hash

### Ingestion safety

The ingestion pipeline prevents duplicate content using **SHA256 hashing** and also guards against document ID conflicts.

### Retrieval reranking

Retrieved chunks are reranked before generation to improve relevance and answer quality.

### Evidence-grounded generation

Answers are generated only from retrieved evidence. Each grounded response includes the supporting source chunks used to produce it.

### Multi-document reasoning

The system can handle comparative questions across documents, such as:

`Compare PostgreSQL MVCC with SQL Server snapshot isolation.`

### Evaluation suite

The project includes automated evaluation for:

- planner routing correctness
- API route behavior
- grounded answer behavior
- source attribution presence

### Streamlit frontend

A lightweight Streamlit interface makes it easy to:

- ingest documents
- ask routed questions
- inspect answers and supporting sources
- browse the knowledge base

## What the Project Does

Planner RAG allows you to:

- ingest documents into a persistent vector database
- prevent duplicate ingestion using content hashing
- classify questions through a planner layer
- retrieve relevant chunks from ChromaDB
- rerank retrieved chunks for stronger relevance
- generate answers constrained to retrieved evidence
- return supporting source chunks with each answer
- inspect the system through a simple frontend

## Architecture

The system follows a modular backend structure:

    app/
    ├── api/            FastAPI endpoints
    ├── embeddings/     Embedding interface
    ├── ingestion/      Document chunking and preprocessing
    ├── llm/            LLM client
    ├── orchestration/  Planner and routing logic
    ├── rag/            Retrieval and generation pipeline
    ├── vectordb/       ChromaDB integration
    └── utils/          Shared utilities

    frontend/
    └── app.py          Streamlit frontend

### Top-level files

    eval_cases.json     Evaluation test cases
    run_eval.py         Automated evaluation script
    requirements.txt    Project dependencies
    README.md


## Routing Behavior

The planner explicitly selects one of the following execution paths:

### `single`

Used when the question targets one document or one system.

Example: `How does MVCC work in PostgreSQL?`

### `multi`

Used when the question asks for comparison, contrast, or synthesis across multiple targets.

Example: `Compare PostgreSQL MVCC with SQL Server snapshot isolation.`

### `unknown`

Used when the question cannot be grounded in the ingested knowledge base or falls outside supported routing behavior.

Example: `What is the capital of Japan?`

This explicit routing design improves reliability and makes system behavior easier to test.

## API Endpoints

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

### `POST /ask`

Run a standard single-document RAG query.

### `POST /ask_routed`

Run planner-based question answering with routed retrieval.

#### Example response

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

### `GET /documents`

Return a grouped summary of ingested documents in the knowledge base.

### `GET /documents/{document_id}`

Return all stored chunks for a specific document.

## Running the Project

### 1. Install dependencies

    pip install -r requirements.txt

### 2. Start the FastAPI backend

    python -m uvicorn app.api.main:app --reload

Backend: `http://127.0.0.1:8000`  
Interactive API docs: `http://127.0.0.1:8000/docs`

### 3. Start the Streamlit frontend

    python -m streamlit run frontend/app.py

Frontend: `http://localhost:8501`

## Frontend

The Streamlit app includes three core views:

- **Ask Questions** — submit routed queries and inspect answers
- **Ingest Document** — add new knowledge to the vector database
- **Knowledge Base** — browse ingested documents and chunk previews

The frontend is designed for quick demos, debugging, and inspection of grounded system behavior.

## Evaluation

Run the evaluation suite with:

    python run_eval.py

The evaluation checks:

- planner routing correctness
- API route selection
- answer behavior
- source presence

### Example output

    Planner: 4/4
    API route: 4/4
    Answer behavior: 4/4
    Sources presence: 4/4

This makes it easier to validate that the system is not only functional, but also behaving as intended.

## Current Limitations

The current version has a few intentional limitations:

- document routing depends on a curated registry
- planner coverage is limited to supported domains
- retrieval quality depends on chunking and metadata quality
- reranking is currently simple and can be improved
- new unsupported domains may require manual routing updates

## Future Improvements

Planned or possible next steps include:

- dynamic document registry from ingestion metadata
- hybrid retrieval (`BM25 + vector search`)
- stronger reranking with cross-encoders
- query rewriting before retrieval
- streaming responses
- document deletion and management tools
- richer frontend debugging and analytics views
- more advanced evaluation coverage

## Why This Project Matters

Many RAG demos stop at basic retrieval and generation.

Planner RAG goes a step further by introducing **explicit routing**, **grounded answer generation**, **source attribution**, and **evaluation-aware design**. The result is a more controlled and inspectable backend for knowledge-grounded QA.

This project is meant to reflect a more production-oriented approach to building LLM systems.

## License

MIT License