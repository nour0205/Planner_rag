# RAGent: Retrieval-Augmented Generation with Planner Routing

RAGent is a production-style **Retrieval-Augmented Generation (RAG)** backend that answers questions using only ingested knowledge.

It combines:
- planner-based routing
- hybrid retrieval (vector + BM25)
- reranking
- evidence-grounded generation

to produce **traceable, source-backed answers**.

## Pipeline

User Question
      │
      ▼
Planner Agent (route: single / multi / unknown)
      │
      ▼
Execution Router
      │
      ▼
Hybrid Retrieval
  ├── Vector Search (ChromaDB)
  └── BM25 Search (Whoosh)
      │
      ▼
Reciprocal Rank Fusion (RRF)
      │
      ▼
Reranking
      │
      ▼
LLM Generation
      │
      ▼
Answer + Sources


## Key Features

### Planner-based routing

Questions are classified before retrieval begins. The planner chooses one of three routes:

- `single` — answer using one target document
- `multi` — compare or combine evidence from multiple documents
- `unknown` — refuse unsupported or unsafe questions



### Persistent vector knowledge base

Documents are chunked, embedded, and stored in **ChromaDB** with persistent metadata, including:

- document ID
- chunk index
- source
- owner
- document hash

### Ingestion safety

The ingestion pipeline prevents duplicate content using **SHA256 hashing** and also guards against document ID conflicts.

### Hybrid retrieval (BM25 + vector search)

The system combines:

- **Dense retrieval (ChromaDB)** for semantic similarity
- **Lexical retrieval (Whoosh BM25)** for exact keyword matching
- **Reciprocal Rank Fusion (RRF)** to merge both rankings




### Retrieval reranking

Retrieved chunks are reranked before generation to improve relevance and answer quality.



### Answer Generation

- Answers are generated only from retrieved evidence  
- Each response includes source attribution  
- Supports multi-document reasoning for comparisons and synthesis  

### Evaluation suite

The project includes automated evaluation for:

- planner routing correctness
- API route behavior
- grounded answer behavior
- source attribution presence



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



### Manual routing override

The API also allows explicit routing by providing document identifiers:

- `document_id` → forces single-document retrieval
- `document_ids` → forces multi-document retrieval

This bypasses the planner and allows precise control over which documents are used during retrieval.




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



## Current Limitations

The current version has a few intentional limitations:

- planner coverage is limited to supported question patterns
- retrieval quality depends on chunking and metadata quality
- reranking is currently heuristic-based (keyword overlap)
- no document deletion or update support yet
- frontend does not expose all retrieval diagnostics

## Future Improvements

Planned or possible next steps include:

- dynamic document registry from ingestion metadata
- stronger reranking with cross-encoders
- query rewriting before retrieval
- streaming responses
- document deletion and management tools
- richer frontend debugging and analytics views
- more advanced evaluation coverage



## License

MIT License
