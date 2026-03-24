from app.rag.hybrid_retriever import hybrid_retrieve
from app.rag.pipeline import rerank_items


def retrieve_for_document(
    store,
    question: str,
    document_id: str,
    k: int = 4,
    retrieve_k: int = 10
):
    where = {"document_id": document_id}

    candidates = hybrid_retrieve(
        store=store,
        question=question,
        k=retrieve_k,
        where=where
    )

    items = rerank_items(question, candidates, k=k)

    normalized = []
    for item in items:
        item_copy = dict(item)
        item_copy["text"] = item.get("text") or item.get("document") or ""
        normalized.append(item_copy)

    return normalized