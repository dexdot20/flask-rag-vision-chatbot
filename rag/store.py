from __future__ import annotations

import os
import time
from typing import Any

from config import RAG_SUPPORTED_CATEGORIES
from .chunker import Chunk, normalize_category
from .embedder import embed_texts

DEFAULT_COLLECTION_NAME = "knowledge_base"
CATEGORY_COLLECTION_PREFIX = f"{DEFAULT_COLLECTION_NAME}__"
_client = None
_collection_cache: dict[str, Any] = {}


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


def get_collection(name: str = DEFAULT_COLLECTION_NAME):
    normalized_name = str(name or DEFAULT_COLLECTION_NAME).strip() or DEFAULT_COLLECTION_NAME
    cached = _collection_cache.get(normalized_name)
    if cached is not None:
        return cached
    client = get_client()
    collection = client.get_or_create_collection(name=normalized_name, metadata={"hnsw:space": "cosine"})
    _collection_cache[normalized_name] = collection
    return collection


def get_category_collection_name(category: str | None) -> str:
    normalized_category = normalize_category(category)
    if normalized_category == normalize_category(DEFAULT_COLLECTION_NAME):
        return DEFAULT_COLLECTION_NAME
    return f"{CATEGORY_COLLECTION_PREFIX}{normalized_category}"


def _get_category_collection(category: str | None):
    return get_collection(get_category_collection_name(category))


def _iter_query_collections(category: str | None = None) -> list[tuple[Any, dict[str, Any] | None]]:
    collections: list[tuple[Any, dict[str, Any] | None]] = []
    seen_names: set[str] = set()

    def add_collection(name: str, where: dict[str, Any] | None = None) -> None:
        normalized_name = str(name or "").strip()
        if not normalized_name or normalized_name in seen_names:
            return
        seen_names.add(normalized_name)
        collections.append((get_collection(normalized_name), where))

    if category:
        normalized_category = normalize_category(category)
        add_collection(get_category_collection_name(normalized_category))
        add_collection(DEFAULT_COLLECTION_NAME, where={"category": normalized_category})
        return collections

    add_collection(DEFAULT_COLLECTION_NAME)
    for supported_category in sorted(RAG_SUPPORTED_CATEGORIES):
        add_collection(get_category_collection_name(supported_category))
    return collections


def _query_collection_rows(collection, query_embedding: list[list[float]], top_k: int, where: dict[str, Any] | None = None) -> list[dict]:
    result = collection.query(
        query_embeddings=query_embedding,
        n_results=max(1, min(int(top_k or 5), 12)),
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]
    ids = (result.get("ids") or [[]])[0]

    rows: list[dict] = []
    now_ts = int(time.time())
    for index, document in enumerate(documents):
        metadata = metadatas[index] if index < len(metadatas) else {}
        if _is_expired_metadata(metadata, now_ts=now_ts):
            continue
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


def _get_collection_rows(collection, where: dict[str, Any]) -> list[dict]:
    result = collection.get(where=where, include=["documents", "metadatas"])
    ids = result.get("ids") or []
    documents = result.get("documents") or []
    metadatas = result.get("metadatas") or []
    rows: list[dict] = []
    now_ts = int(time.time())
    for index, item_id in enumerate(ids):
        metadata = metadatas[index] if index < len(metadatas) else {}
        if _is_expired_metadata(metadata, now_ts=now_ts):
            continue
        rows.append(
            {
                "id": item_id,
                "text": documents[index] if index < len(documents) else "",
                "metadata": metadata or {},
            }
        )
    rows.sort(key=lambda item: int((item.get("metadata") or {}).get("chunk_index") or 0))
    return rows


def upsert_chunks(chunks: list[Chunk]) -> int:
    if not chunks:
        return 0

    documents = [chunk.text for chunk in chunks]
    embeddings = embed_texts(documents)
    if len(embeddings) != len(documents):
        raise RuntimeError("Embedding count mismatch while upserting chunks.")

    chunks_by_category: dict[str, list[tuple[Chunk, list[float]]]] = {}
    for chunk, embedding in zip(chunks, embeddings):
        normalized_category = normalize_category(chunk.category)
        chunks_by_category.setdefault(normalized_category, []).append((chunk, embedding))

    inserted = 0
    for category, grouped_items in chunks_by_category.items():
        collection = _get_category_collection(category)
        collection.upsert(
            ids=[chunk.id for chunk, _embedding in grouped_items],
            documents=[chunk.text for chunk, _embedding in grouped_items],
            embeddings=[embedding for _chunk, embedding in grouped_items],
            metadatas=[chunk.to_metadata() for chunk, _embedding in grouped_items],
        )
        inserted += len(grouped_items)
    return inserted


def _build_where(category: str | None = None) -> dict[str, Any] | None:
    if not category:
        return None
    return {"category": normalize_category(category)}


def _coerce_timestamp(value) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        normalized = int(value)
        return normalized if normalized > 0 else None
    text = str(value or "").strip()
    if not text:
        return None
    try:
        normalized = int(float(text))
    except (TypeError, ValueError):
        return None
    return normalized if normalized > 0 else None


def _is_expired_metadata(metadata: dict | None, now_ts: int | None = None) -> bool:
    source = metadata if isinstance(metadata, dict) else {}
    expires_at_ts = _coerce_timestamp(source.get("expires_at_ts"))
    if expires_at_ts is None:
        return False
    reference_now = int(now_ts if isinstance(now_ts, int) else time.time())
    return expires_at_ts <= reference_now


def query_chunks(query: str, top_k: int = 5, category: str | None = None) -> list[dict]:
    query = str(query or "").strip()
    if not query:
        return []

    query_embedding = embed_texts([query])
    if not query_embedding:
        return []

    rows: list[dict] = []
    seen_ids: set[str] = set()
    for collection, where in _iter_query_collections(category):
        for row in _query_collection_rows(collection, query_embedding, top_k=top_k, where=where):
            row_id = str(row.get("id") or "").strip()
            if row_id and row_id in seen_ids:
                continue
            if row_id:
                seen_ids.add(row_id)
            rows.append(row)
    rows.sort(key=lambda item: float(item.get("distance") if item.get("distance") is not None else 999999), reverse=False)
    return rows


def get_source_chunks(source_ref: str, category: str | None = None) -> list[dict]:
    cleaned = str(source_ref or "").strip()
    if not cleaned:
        return []

    rows: list[dict] = []
    seen_ids: set[str] = set()
    for collection, where in _iter_query_collections(category):
        conditions = [
            {"source_key": cleaned},
            {"source_name": cleaned},
        ]
        for condition in conditions:
            merged_where = dict(condition)
            if where:
                merged_where.update(where)
            for row in _get_collection_rows(collection, merged_where):
                row_id = str(row.get("id") or "").strip()
                if row_id and row_id in seen_ids:
                    continue
                if row_id:
                    seen_ids.add(row_id)
                rows.append(row)
    rows.sort(key=lambda item: int((item.get("metadata") or {}).get("chunk_index") or 0))
    return rows


def delete_source(source_ref: str) -> int:
    cleaned = str(source_ref or "").strip()
    if not cleaned:
        return 0
    deleted = 0
    deleted_ids: set[str] = set()
    for collection, where in _iter_query_collections(None):
        conditions = [
            {"source_key": cleaned},
            {"source_name": cleaned},
        ]
        for condition in conditions:
            merged_where = dict(condition)
            if where:
                merged_where.update(where)
            existing = collection.get(where=merged_where, include=[])
            ids = existing.get("ids") or []
            if not ids:
                continue
            ids_to_delete = [item_id for item_id in ids if str(item_id or "").strip() not in deleted_ids]
            if not ids_to_delete:
                continue
            collection.delete(ids=ids_to_delete)
            for item_id in ids_to_delete:
                deleted_ids.add(str(item_id or "").strip())
            deleted += len(ids_to_delete)
    return deleted
