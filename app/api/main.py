from fastapi import FastAPI
from collections import defaultdict
from pydantic import BaseModel
import uuid

# ---- Core RAG imports ----
from app.ingestion.chunker import base_chunks, apply_overlap
from app.embeddings.embedder import embed_texts
from app.vectordb.chroma_store import ChromaStore
from app.rag.pipeline import rag_answer_with_store, rag_answer_with_sources
from app.rag.hybrid_retriever import hybrid_retrieve
from app.utils.hash import hash_text
from app.llm.client import chat

# ---- Orchestration imports ----
from app.orchestration.registry import DOC_REGISTRY
from app.orchestration.retrieval import retrieve_for_document
from app.orchestration.prompts import build_compare_prompt
from app.orchestration.planner import plan_question


# -------------------------------------------------------------------
# App + Store
# -------------------------------------------------------------------
app = FastAPI(title="RAG API", version="1.0")
store = ChromaStore(collection_name="api-demo")
# -------------------------------------------------------------------
# ---- Whoosh imports ----
# -------------------------------------------------------------------
from app.vectordb.whoosh_index import add_chunks_to_whoosh

# -------------------------------------------------------------------
# Schemas
# -------------------------------------------------------------------
class QuestionRequest(BaseModel):
    question: str
    document_id: str | None = None
    document_ids: list[str] | None = None
    owner: str | None = None


class SourceItem(BaseModel):
    document_id: str
    chunk_index: int | None = None
    text: str
    retrieval_type: str | None = None
    hybrid_score: float | None = None


class AnswerResponse(BaseModel):
    answer: str
    sources: list[SourceItem] = []
    route: str | None = None


class IngestRequest(BaseModel):
    text: str
    document_id: str
    source: str | None = None
    owner: str | None = None


# -------------------------------------------------------------------
# ASK (simple RAG, no orchestration)
# -------------------------------------------------------------------
@app.post("/ask", response_model=AnswerResponse)
def ask(req: QuestionRequest):
    where = {}

    if req.document_id:
        where["document_id"] = req.document_id
    if req.owner:
        where["owner"] = req.owner

    if not where:
        where = None

    result = rag_answer_with_sources(
        question=req.question,
        store=store,
        k=5,
        where=where
    )

    return {
        "answer": result["answer"],
        "route": "single",
        "sources": result["sources"]
    }


# -------------------------------------------------------------------
# INGEST
# -------------------------------------------------------------------
@app.post("/ingest")
def ingest(req: IngestRequest):
    doc_hash = hash_text(req.text)

    # --- Dedup by content ---
    existing = store.collection.get(
        where={"doc_hash": doc_hash},
        limit=1
    )
    if existing["ids"]:
        return {"status": "duplicate", "reason": "document already ingested"}

    # --- Prevent document_id collision ---
    existing_id = store.collection.get(
        where={"document_id": req.document_id},
        limit=1
    )
    if existing_id["ids"]:
        return {"status": "conflict", "reason": "document_id already exists"}

    # 1) Chunk
    chunks = base_chunks(req.text)
    chunks = apply_overlap(chunks, overlap=1)
    if not chunks:
        return {"status": "no content"}

    # 2) Embed
    embeddings = embed_texts(chunks)

    # 3) Store
    ids = [str(uuid.uuid4()) for _ in chunks]
    metadatas = []

    for i in range(len(chunks)):
        meta = {
            "chunk_id": ids[i],
            "document_id": req.document_id,
            "doc_hash": doc_hash,
            "chunk_index": i,
        }
        if req.source:
            meta["source"] = req.source
        if req.owner:
            meta["owner"] = req.owner
        metadatas.append(meta)
    whoosh_chunks = []

    for i in range(len(chunks)):
        whoosh_chunks.append({
            "chunk_id": ids[i],
            "document_id": req.document_id,
            "chunk_index": i,
            "source": req.source or "",
            "owner": req.owner or "",
            "text": chunks[i],
        })
    store.add_texts(
        ids=ids,
        texts=chunks,
        embeddings=embeddings,
        metadatas=metadatas
    )
    add_chunks_to_whoosh(whoosh_chunks)
    

    return {"status": "ingested", "chunks_added": len(chunks)}


# -------------------------------------------------------------------
# DEBUG: inspect a document
# -------------------------------------------------------------------
@app.get("/debug/doc/{document_id}")
def debug_doc(document_id: str):
    result = store.collection.get(where={"document_id": document_id})
    return {
        "count": len(result["ids"]),
        "documents": result["documents"],
        "metadatas": result["metadatas"],
    }


# -------------------------------------------------------------------
# DEBUG: inspect retrieval
# -------------------------------------------------------------------
@app.post("/debug/retrieve")
def debug_retrieve(req: QuestionRequest):
    where = {}

    if req.document_id:
        where["document_id"] = req.document_id
    if req.owner:
        where["owner"] = req.owner

    if not where:
        where = None

    results = hybrid_retrieve(
        store=store,
        question=req.question,
        k=5,
        where=where
    )

    return {"question": req.question, "results": results}


# -------------------------------------------------------------------
# list ingested documents
# -------------------------------------------------------------------


@app.get("/documents")
def list_documents():
    """
    Return a grouped summary of ingested documents.
    Groups chunks by document_id.
    """
    data = store.collection.get(
        include=["metadatas", "documents"]
    )

    metadatas = data.get("metadatas", []) or []
    documents = data.get("documents", []) or []

    grouped = defaultdict(lambda: {
        "document_id": None,
        "chunks": 0,
        "source": None,
        "owner": None,
        "preview": None,
    })

    for meta, doc_text in zip(metadatas, documents):
        if not meta:
            continue

        document_id = meta.get("document_id", "unknown")
        entry = grouped[document_id]

        entry["document_id"] = document_id
        entry["chunks"] += 1
        entry["source"] = meta.get("source")
        entry["owner"] = meta.get("owner")

        if entry["preview"] is None and doc_text:
            entry["preview"] = doc_text[:200]

    return {
        "documents": list(grouped.values()),
        "count": len(grouped),
    }


# -------------------------------------------------------------------
# document preview 
# -------------------------------------------------------------------

@app.get("/documents/{document_id}")
def get_document_chunks(document_id: str):
    data = store.collection.get(
        where={"document_id": document_id},
        include=["metadatas", "documents"]
    )

    ids = data.get("ids", []) or []
    metadatas = data.get("metadatas", []) or []
    documents = data.get("documents", []) or []

    chunks = []
    for chunk_id, meta, text in zip(ids, metadatas, documents):
        chunks.append({
            "id": chunk_id,
            "document_id": meta.get("document_id"),
            "chunk_index": meta.get("chunk_index"),
            "source": meta.get("source"),
            "owner": meta.get("owner"),
            "text": text,
        })

    chunks.sort(key=lambda x: x.get("chunk_index", 0))

    return {
        "document_id": document_id,
        "chunks": chunks,
        "chunk_count": len(chunks),
    }
# -------------------------------------------------------------------
# ASK ROUTED (single-doc OR multi-doc)
# -------------------------------------------------------------------
@app.post("/ask_routed", response_model=AnswerResponse)
def ask_routed(req: QuestionRequest):
    # ---- Override planner if user provides document_ids ----
    if req.document_ids and len(req.document_ids) == 2:
        decision = {
            "route": "multi",
            "targets": req.document_ids
        }
    elif req.document_id:
        decision = {
            "route": "single",
            "targets": [req.document_id]
        }
    else:
        decision = plan_question(req.question)

    # ---------- UNKNOWN / UNSUPPORTED ----------
    if decision["route"] == "unknown":
        return {
            "answer": "I don't know.",
            "route": "unknown",
            "sources": []
        }

    # ---------- SINGLE DOCUMENT ----------
    if decision["route"] == "single":
        if req.document_id:
            where = {"document_id": req.document_id}
            if req.owner:
                where["owner"] = req.owner
        elif len(decision["targets"]) == 1:
            target = decision["targets"][0]
            where = {"document_id": target}
            if req.owner:
                where["owner"] = req.owner
        else:
            return {
                "answer": "I don't know.",
                "route": "single",
                "sources": []
            }

        result = rag_answer_with_sources(
            question=req.question,
            store=store,
            k=5,
            where=where
        )

        return {
            "answer": result["answer"],
            "route": "single",
            "sources": result["sources"]
        }

    # ---------- MULTI DOCUMENT ----------
    if decision["route"] == "multi":
        if len(decision["targets"]) != 2:
            return {
                "answer": "I don't know.",
                "route": "multi",
                "sources": []
            }

        doc1, doc2 = decision["targets"]

        ctx1 = retrieve_for_document(store, req.question, doc1)
        ctx2 = retrieve_for_document(store, req.question, doc2)

        if not ctx1 or not ctx2:
            return {
                "answer": "I don't know.",
                "route": "multi",
                "sources": []
            }

        contexts1 = [item.get("text") or item.get("document") or "" for item in ctx1]
        contexts2 = [item.get("text") or item.get("document") or "" for item in ctx2]   

        messages = build_compare_prompt(req.question, contexts1, contexts2)
        answer = chat(messages)

        sources = []
        for item in ctx1 + ctx2:
            meta = item["metadata"] or {}
            sources.append({
                "document_id": meta.get("document_id", "unknown"),
                "chunk_index": meta.get("chunk_index"),
                "text": item.get("text") or item.get("document") or "",
                "retrieval_type": item.get("retrieval_type", "unknown"),
                "hybrid_score": item.get("hybrid_score")
            })

        return {
            "answer": answer,
            "route": "multi",
            "sources": sources
        }

    return {
        "answer": "I don't know.",
        "route": "unknown",
        "sources": []
    }


# -------------------------------------------------------------------
# DEBUG: Planning
# -------------------------------------------------------------------
@app.post("/debug/plan")
def debug_plan(req: QuestionRequest):
    plan = plan_question(req.question)
    return {"question": req.question, "plan": plan}