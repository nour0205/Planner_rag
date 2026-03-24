from pathlib import Path
from whoosh import index
from whoosh.fields import Schema, ID, TEXT, NUMERIC
from whoosh.qparser import MultifieldParser

INDEX_DIR = Path("data/whoosh_index")


def get_schema():
    return Schema(
        chunk_id=ID(stored=True, unique=True),
        document_id=ID(stored=True),
        chunk_index=NUMERIC(stored=True),
        source=TEXT(stored=True),
        owner=TEXT(stored=True),
        content=TEXT(stored=True),
    )


def get_or_create_index():
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    if index.exists_in(INDEX_DIR):
        return index.open_dir(INDEX_DIR)

    return index.create_in(INDEX_DIR, get_schema())


def add_chunks_to_whoosh(chunks: list[dict]):
    ix = get_or_create_index()
    writer = ix.writer()

    try:
       
        for chunk in chunks:
            print({
                "chunk_id": chunk["chunk_id"],
                "document_id": chunk["document_id"],
                "chunk_index": chunk["chunk_index"],
                "text": chunk["text"]
            })

            writer.update_document(
                chunk_id=chunk["chunk_id"],
                document_id=chunk["document_id"],
                chunk_index=chunk["chunk_index"],
                source=chunk.get("source", ""),
                owner=chunk.get("owner", ""),
                content=chunk["text"],
            )
        writer.commit()
        
    except Exception:
        writer.cancel()
        raise


def search_whoosh(query_text: str, limit: int = 5):
    ix = get_or_create_index()

    with ix.searcher() as searcher:
        parser = MultifieldParser(["content", "document_id", "source"], schema=ix.schema)
        query = parser.parse(query_text)
        results = searcher.search(query, limit=limit)

        output = []
        for rank, hit in enumerate(results):
            output.append({
                "id": hit["chunk_id"],
                "text": hit["content"],
                "metadata": {
                    "document_id": hit["document_id"],
                    "chunk_index": hit["chunk_index"],
                    "source": hit.get("source"),
                    "owner": hit.get("owner"),
                },
                "score": float(hit.score),
                "rank": rank + 1,
                "retrieval_type": "bm25",
            })

        return output
    
