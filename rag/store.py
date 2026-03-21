from __future__ import annotations

import os
from typing import Any

from .chunker import Chunk, normalize_category
from .embedder import embed_texts

DEFAULT_COLLECTION_NAME = "knowledge_base"
_client = None
_collection = None


def get_chroma_path() -> str:
    base_dir = os.path.dirname(os.path.dirname(__file__))
    return os.getenv("CHROMA_DB_PATH") or os.path.join(base_dir, "chroma_db")


def get_client():
    global _client
    if _client is not None:
        return _client
    try:
        import chromadb
    except ImportError as exc:
        raise RuntimeError("ChromaDB dependencies are missing. Install chromadb before using RAG.") from exc
    _client = chromadb.PersistentClient(path=get_chroma_path())
    return _client


def get_collection():
    global _collection
    if _collection is not None:
        return _collection
    client = get_client()
    _collection = client.get_or_create_collection(name=DEFAULT_COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
    return _collection


def upsert_chunks(chunks: list[Chunk]) -> int:
    if not chunks:
        return 0

    collection = get_collection()
    documents = [chunk.text for chunk in chunks]
    embeddings = embed_texts(documents)
    if len(embeddings) != len(documents):
        raise RuntimeError("Embedding count mismatch while upserting chunks.")

    collection.upsert(
        ids=[chunk.id for chunk in chunks],
        documents=documents,
        embeddings=embeddings,
        metadatas=[chunk.to_metadata() for chunk in chunks],
    )
    return len(chunks)


def _build_where(category: str | None = None) -> dict[str, Any] | None:
    if not category:
        return None
    return {"category": normalize_category(category)}


def query_chunks(query: str, top_k: int = 5, category: str | None = None) -> list[dict]:
    query = str(query or "").strip()
    if not query:
        return []

    collection = get_collection()
    query_embedding = embed_texts([query])
    if not query_embedding:
        return []

    result = collection.query(
        query_embeddings=query_embedding,
        n_results=max(1, min(int(top_k or 5), 12)),
        where=_build_where(category),
        include=["documents", "metadatas", "distances"],
    )

    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]
    ids = (result.get("ids") or [[]])[0]

    rows: list[dict] = []
    for index, document in enumerate(documents):
        metadata = metadatas[index] if index < len(metadatas) else {}
        distance = distances[index] if index < len(distances) else None
        similarity = None if distance is None else max(0.0, min(1.0, 1.0 - float(distance)))
        rows.append(
            {
                "id": ids[index] if index < len(ids) else None,
                "text": document,
                "metadata": metadata or {},
                "distance": distance,
                "similarity": similarity,
            }
        )
    return rows


def delete_source(source_ref: str) -> int:
    cleaned = str(source_ref or "").strip()
    if not cleaned:
        return 0

    collection = get_collection()
    existing = collection.get(where={"source_key": cleaned}, include=[])
    ids = existing.get("ids") or []
    if not ids:
        existing = collection.get(where={"source_name": cleaned}, include=[])
        ids = existing.get("ids") or []
    if not ids:
        return 0
    collection.delete(ids=ids)
    return len(ids)
