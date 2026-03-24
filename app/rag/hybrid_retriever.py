from app.embeddings.embedder import embed_texts
from app.vectordb.whoosh_index import search_whoosh


def _get_chunk_key(item: dict) -> str:
    if "id" in item:
        return str(item["id"])

    meta = item.get("metadata", {})

    if "chunk_id" in meta:
        return str(meta["chunk_id"])

    document_id = meta.get("document_id", "unknown_doc")
    chunk_index = meta.get("chunk_index", "unknown_chunk")
    return f"{document_id}:{chunk_index}"

def reciprocal_rank_fusion(vector_results: list[dict], bm25_results: list[dict], k: int = 60):
    scores = {}
    merged = {}

    for item in vector_results:
        chunk_id = _get_chunk_key(item)
        rank = item["rank"]

        scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank)

        item_copy = dict(item)
        item_copy["id"] = chunk_id
        item_copy["retrieval_type"] = "vector"
        item_copy["found_by_vector"] = True
        item_copy["found_by_bm25"] = False

        merged[chunk_id] = item_copy

    for item in bm25_results:
        chunk_id = _get_chunk_key(item)
        rank = item["rank"]

        scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank)

        if chunk_id in merged:
            merged[chunk_id]["found_by_bm25"] = True
            merged[chunk_id]["retrieval_type"] = "hybrid"
            merged[chunk_id]["bm25_score"] = item.get("score")
        else:
            item_copy = dict(item)
            item_copy["id"] = chunk_id
            item_copy["retrieval_type"] = "bm25"
            item_copy["found_by_vector"] = False
            item_copy["found_by_bm25"] = True
            merged[chunk_id] = item_copy

    fused = []
    for chunk_id, item in merged.items():
        new_item = dict(item)
        new_item["hybrid_score"] = scores[chunk_id]
        fused.append(new_item)

    fused.sort(key=lambda x: x["hybrid_score"], reverse=True)
    return fused

def hybrid_retrieve(store, question: str, k: int = 5, where: dict | None = None):
    query_embedding = embed_texts([question])[0]

    vector_results = store.query_with_scores(
        query_embedding=query_embedding,
        k=k,
        where=where
    )

    for rank, item in enumerate(vector_results):
        item["rank"] = rank + 1
        item["retrieval_type"] = "vector"
        item["id"] = _get_chunk_key(item)

    bm25_results = search_whoosh(question, limit=k)

    if where:
        filtered_bm25 = []
        for item in bm25_results:
            meta = item.get("metadata", {})
            ok = True
            for key, value in where.items():
                if meta.get(key) != value:
                    ok = False
                    break
            if ok:
                filtered_bm25.append(item)
        bm25_results = filtered_bm25

    fused = reciprocal_rank_fusion(vector_results, bm25_results)

    return fused[:k]

