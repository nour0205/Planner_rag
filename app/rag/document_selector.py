import re
from app.catalog.document_catalog import list_document_entries


STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "why", "how", "what",
    "does", "do", "did", "and", "or", "of", "to", "in", "on", "for",
    "with", "by", "at", "from", "this", "that", "it", "as", "be",
    "compare", "difference", "between"
}


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"\b[a-zA-Z0-9_]+\b", (text or "").lower())

    # simple normalization: remove trailing 's'
    normalized = []
    for t in tokens:
        if t.endswith("s") and len(t) > 3:
            t = t[:-1]
        normalized.append(t)

    return normalized

def keyword_overlap_score(question: str, text: str) -> float:
    q_tokens = [t for t in tokenize(question) if t not in STOPWORDS]
    t_tokens = set(tokenize(text))

    if not q_tokens:
        return 0.0

    return float(sum(1 for token in q_tokens if token in t_tokens))


def select_documents(question: str, store=None, top_k: int = 2) -> list[dict]:
    docs = list_document_entries()

    scored = []
    for doc in docs:
        searchable_text = " ".join([
            doc.get("document_id", ""),
            doc.get("title", ""),
            doc.get("preview", ""),
            doc.get("source", "") or "",
            doc.get("owner", "") or "",
        ])

        lexical_score = keyword_overlap_score(question, searchable_text)

        # bonus for richer documents
        chunk_bonus = min(doc.get("chunk_count", 0), 5) * 0.2

        final_score = lexical_score + chunk_bonus

        scored.append((final_score, doc))

    scored.sort(key=lambda x: x[0], reverse=True)

    print("\n[DEBUG] document selection (CATALOG)")
    for score, doc in scored[:10]:
        print({
            "document_id": doc["document_id"],
            "score": score,
            "chunks": doc.get("chunk_count"),
            "preview": doc.get("preview", "")[:120]
        })

    selected = [doc for score, doc in scored if score > 0]
    return selected[:top_k]