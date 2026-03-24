from __future__ import annotations

import hashlib
import json
import logging
import re

from config import (
    RAG_DISABLED_FEATURE_ERROR,
    RAG_ENABLED,
    RAG_SEARCH_DEFAULT_TOP_K,
    RAG_SEARCH_MIN_SIMILARITY,
    RAG_SOURCE_CONVERSATION,
    RAG_SOURCE_TOOL_MEMORY,
    RAG_SOURCE_TOOL_RESULT,
    RAG_SUPPORTED_CATEGORIES,
    RAG_SUPPORTED_SOURCE_TYPES,
)
from db import extract_message_tool_results, get_db, parse_message_metadata
from messages import build_user_message_for_model
from rag import (
    chunks_from_records,
    normalize_category,
)
from rag import (
    delete_source as rag_delete_source,
)
from rag import (
    query_chunks as rag_query_chunks,
)
from rag import (
    upsert_chunks as rag_upsert_chunks,
)

_rag_sources_verified = False
CATEGORY_TOOL_MEMORY = RAG_SOURCE_TOOL_MEMORY
logger = logging.getLogger(__name__)


def _require_rag_enabled() -> None:
    if not RAG_ENABLED:
        raise RuntimeError(RAG_DISABLED_FEATURE_ERROR)


def _clean_rag_text_block(text: str, limit: int | None = None) -> str:
    cleaned = re.sub(r"\n{3,}", "\n\n", str(text or "").strip())
    if limit and len(cleaned) > limit:
        return cleaned[:limit].rstrip() + "…"
    return cleaned


def normalize_rag_category(category: str | None, default: str | None = RAG_SOURCE_CONVERSATION) -> str | None:
    cleaned = normalize_category(category)
    if cleaned in RAG_SUPPORTED_CATEGORIES:
        return cleaned
    if default is None:
        return None
    fallback = normalize_category(default)
    return fallback if fallback in RAG_SUPPORTED_CATEGORIES else None


def _is_supported_rag_source_type(source_type: str | None) -> bool:
    return normalize_category(source_type) in RAG_SUPPORTED_SOURCE_TYPES


def build_rag_source_key(source_type: str, source_name: str) -> str:
    normalized_type = str(source_type or "document").strip().lower() or "document"
    normalized_name = str(source_name or "untitled").strip() or "untitled"
    digest = hashlib.sha1(f"{normalized_type}|{normalized_name}".encode("utf-8")).hexdigest()
    return f"src-{digest}"


def _build_rag_sync_signature(payload) -> str:
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(serialized.encode("utf-8")).hexdigest()


def _conversation_rag_source_name(source_type: str, conversation_id: int, title: str) -> str:
    title = str(title or "Untitled")[:80]
    return f"{source_type}:{conversation_id}:{title}"


def conversation_rag_source_key(source_type: str, conversation_id: int) -> str:
    return build_rag_source_key(source_type, str(conversation_id))


def build_tool_memory_source_key(tool_name: str, args_hash: str) -> str:
    normalized_tool_name = normalize_category(tool_name) or "tool"
    normalized_args_hash = str(args_hash or "").strip() or "unknown"
    return build_rag_source_key(RAG_SOURCE_TOOL_MEMORY, f"{normalized_tool_name}:{normalized_args_hash}")


def build_tool_result_record_content(entry: dict, index: int) -> str:
    parts = [f"[{index}] tool:{entry['tool_name']}"]
    if entry.get("input_preview"):
        parts.append(f"Input: {entry['input_preview']}")
    if entry.get("summary"):
        parts.append(f"Summary: {entry['summary']}")
    if entry.get("content_mode"):
        parts.append(f"Content mode: {entry['content_mode']}")
    if entry.get("summary_notice"):
        parts.append(f"Note: {entry['summary_notice']}")
    parts.append(entry["content"])
    return "\n".join(parts)


def serialize_rag_metadata(metadata: dict | None) -> str | None:
    if not isinstance(metadata, dict) or not metadata:
        return None
    cleaned = {}
    for key, value in metadata.items():
        if value in (None, ""):
            continue
        if isinstance(value, (dict, list)):
            cleaned[str(key)] = value
        else:
            cleaned[str(key)] = str(value)
    if not cleaned:
        return None
    return json.dumps(cleaned, ensure_ascii=False)


def parse_rag_metadata(raw_metadata) -> dict:
    if isinstance(raw_metadata, dict):
        return raw_metadata
    if not raw_metadata:
        return {}
    try:
        parsed = json.loads(raw_metadata)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def upsert_rag_document_record(
    source_key: str, source_name: str, source_type: str, category: str, chunk_count: int, metadata: dict | None = None
):
    category = normalize_rag_category(category, default=RAG_SOURCE_CONVERSATION) or RAG_SOURCE_CONVERSATION
    source_type = normalize_category(source_type)
    with get_db() as conn:
        conn.execute(
            """INSERT INTO rag_documents
               (source_key, source_name, source_type, category, chunk_count, metadata, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'))
               ON CONFLICT(source_key) DO UPDATE SET
                   source_name = excluded.source_name,
                   source_type = excluded.source_type,
                   category = excluded.category,
                   chunk_count = excluded.chunk_count,
                   metadata = excluded.metadata,
                   updated_at = datetime('now')""",
            (source_key, source_name, source_type, category, int(chunk_count), serialize_rag_metadata(metadata)),
        )


def _fetch_rag_documents_db() -> list[dict]:
    with get_db() as conn:
        rows = conn.execute(
            """SELECT source_key, source_name, source_type, category, chunk_count, metadata, created_at, updated_at
               FROM rag_documents
               ORDER BY updated_at DESC, source_name ASC"""
        ).fetchall()
    return [
        {
            "source_key": row["source_key"],
            "source_name": row["source_name"],
            "source_type": row["source_type"],
            "category": row["category"],
            "chunk_count": row["chunk_count"],
            "metadata": parse_rag_metadata(row["metadata"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }
        for row in rows
    ]


def ensure_supported_rag_sources(force: bool = False) -> int:
    _require_rag_enabled()
    global _rag_sources_verified
    if _rag_sources_verified and not force:
        return 0

    with get_db() as conn:
        rows = conn.execute("SELECT source_key, source_type FROM rag_documents ORDER BY updated_at DESC").fetchall()

    removed = 0
    for row in rows:
        if _is_supported_rag_source_type(row["source_type"]):
            continue
        rag_delete_source(row["source_key"])
        delete_rag_document_record(row["source_key"])
        removed += 1

    _rag_sources_verified = True
    return removed


def list_rag_documents_db() -> list[dict]:
    _require_rag_enabled()
    ensure_supported_rag_sources()
    return [doc for doc in _fetch_rag_documents_db() if _is_supported_rag_source_type(doc["source_type"])]


def get_rag_document_record(source_key: str):
    with get_db() as conn:
        return conn.execute(
            "SELECT source_key, source_name, source_type, metadata, updated_at FROM rag_documents WHERE source_key = ?",
            (source_key,),
        ).fetchone()


def delete_rag_document_record(source_key: str):
    with get_db() as conn:
        conn.execute("DELETE FROM rag_documents WHERE source_key = ?", (source_key,))


def delete_rag_source_record(source_key: str) -> int:
    _require_rag_enabled()
    deleted_chunks = rag_delete_source(source_key)
    delete_rag_document_record(source_key)
    return deleted_chunks


def _delete_rag_source_if_present(source_key: str) -> int:
    existing = get_rag_document_record(source_key)
    if not existing:
        return 0
    return delete_rag_source_record(source_key)


def _build_conversation_sync_metadata(conversation: dict, source_key: str, sync_signature: str) -> dict:
    return {
        "source_key": source_key,
        "conversation_id": conversation["conversation_id"],
        "title": conversation["title"],
        "sync_signature": sync_signature,
    }


def _conversation_source_needs_sync(source_key: str, sync_signature: str, force: bool = False) -> bool:
    if force:
        return True
    row = get_rag_document_record(source_key)
    if not row:
        return True
    metadata = parse_rag_metadata(row["metadata"])
    return metadata.get("sync_signature") != sync_signature


def _clip_rag_excerpt(text: str, limit: int = 1200) -> str:
    text = str(text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "…"


def _normalize_rag_hits(query: str, hits: list[dict], threshold: float) -> list[dict]:
    matches = []
    for hit in hits:
        similarity = hit.get("similarity")
        if similarity is not None and similarity < threshold:
            continue
        metadata = hit.get("metadata") or {}
        source_type = normalize_category(metadata.get("source_type"))
        if source_type not in RAG_SUPPORTED_SOURCE_TYPES:
            continue
        matches.append(
            {
                "id": hit.get("id"),
                "source_key": metadata.get("source_key"),
                "source_name": metadata.get("source_name"),
                "source_type": source_type,
                "category": normalize_rag_category(metadata.get("category"), default=source_type),
                "chunk_index": metadata.get("chunk_index"),
                "similarity": round(float(similarity), 4) if similarity is not None else None,
                "text": _clip_rag_excerpt(hit.get("text", "")),
            }
        )
    return matches


def search_knowledge_base_tool(query: str, category: str | None = None, top_k: int = RAG_SEARCH_DEFAULT_TOP_K) -> dict:
    _require_rag_enabled()
    query = str(query or "").strip()
    if not query:
        return {"query": "", "matches": [], "count": 0}

    ensure_supported_rag_sources()
    normalized_category = normalize_rag_category(category, default=None) if category else None
    hits = rag_query_chunks(query, top_k=top_k, category=normalized_category)
    matches = _normalize_rag_hits(query, hits, RAG_SEARCH_MIN_SIMILARITY)
    return {
        "query": query,
        "category": normalized_category,
        "count": len(matches),
        "matches": matches,
    }


def upsert_tool_memory_result(tool_name: str, args_preview: str, result_content: str, summary: str = "") -> dict | None:
    _require_rag_enabled()
    cleaned_content = _clean_rag_text_block(result_content)
    if not cleaned_content:
        return None

    cleaned_tool_name = normalize_category(tool_name) or "tool"
    cleaned_args_preview = _clean_rag_text_block(args_preview, limit=300)
    cleaned_summary = _clean_rag_text_block(summary, limit=300)
    args_basis = cleaned_args_preview or cleaned_tool_name
    args_hash = hashlib.sha1(json.dumps(args_basis, ensure_ascii=False).encode("utf-8")).hexdigest()[:12]
    source_key = build_tool_memory_source_key(cleaned_tool_name, args_hash)
    source_name = cleaned_tool_name if not cleaned_args_preview else f"{cleaned_tool_name}: {cleaned_args_preview[:80]}"

    record_parts = [f"tool:{cleaned_tool_name}"]
    if cleaned_args_preview:
        record_parts.append(f"Input: {cleaned_args_preview}")
    if cleaned_summary:
        record_parts.append(f"Summary: {cleaned_summary}")
    record_parts.append(cleaned_content)
    records = [{"role": "tool_memory", "content": "\n".join(record_parts)}]
    metadata = {
        "source_key": source_key,
        "tool_name": cleaned_tool_name,
        "args_preview": cleaned_args_preview,
        "summary": cleaned_summary,
        "source_type": RAG_SOURCE_TOOL_MEMORY,
    }
    chunks = chunks_from_records(
        records,
        source_name=source_name,
        source_type=RAG_SOURCE_TOOL_MEMORY,
        category=CATEGORY_TOOL_MEMORY,
        metadata=metadata,
    )
    if not chunks:
        return None
    return ingest_rag_chunks(
        source_key=source_key,
        source_name=source_name,
        source_type=RAG_SOURCE_TOOL_MEMORY,
        category=CATEGORY_TOOL_MEMORY,
        chunks=chunks,
        metadata=metadata,
    )


def search_tool_memory(query: str, top_k: int = RAG_SEARCH_DEFAULT_TOP_K) -> dict:
    _require_rag_enabled()
    query = str(query or "").strip()
    if not query:
        return {"query": "", "count": 0, "matches": []}

    ensure_supported_rag_sources()
    hits = rag_query_chunks(query, top_k=top_k, category=CATEGORY_TOOL_MEMORY)
    matches = _normalize_rag_hits(query, hits, RAG_SEARCH_MIN_SIMILARITY)
    return {
        "query": query,
        "category": CATEGORY_TOOL_MEMORY,
        "count": len(matches),
        "matches": matches,
    }


def build_tool_memory_auto_context(query: str, top_k: int) -> str | None:
    query = str(query or "").strip()
    if not query:
        return None
    try:
        results = search_tool_memory(query, top_k=max(1, int(top_k)))
    except Exception:
        return None

    matches = results.get("matches") or []
    if not matches:
        return None

    sections = []
    for match in matches:
        source_name = str(match.get("source_name") or "Tool memory").strip()
        similarity = match.get("similarity")
        similarity_text = f"{float(similarity):.2f}" if isinstance(similarity, (int, float)) else "n/a"
        excerpt = _clip_rag_excerpt(match.get("text", ""), limit=1000)
        if excerpt:
            sections.append(f"Source: {source_name}\nSimilarity: {similarity_text}\n{excerpt}")
    if not sections:
        return None
    return "\n\n".join(sections)


def build_rag_auto_context(query: str, enabled: bool, threshold: float, top_k: int) -> dict | None:
    query = str(query or "").strip()
    if not RAG_ENABLED or not enabled or not query:
        return None
    try:
        ensure_supported_rag_sources()
        hits = rag_query_chunks(query, top_k=max(1, int(top_k)))
    except Exception:
        return None

    matches = _normalize_rag_hits(query, hits, max(0.0, min(1.0, float(threshold))))
    if not matches:
        return None

    return {
        "query": query,
        "count": len(matches),
        "matches": matches,
    }


def ingest_rag_chunks(
    source_key: str, source_name: str, source_type: str, category: str, chunks: list, metadata: dict | None = None
) -> dict:
    _require_rag_enabled()
    category = normalize_rag_category(category, default=source_type)
    source_type = normalize_category(source_type)
    if source_type not in RAG_SUPPORTED_SOURCE_TYPES or category not in RAG_SUPPORTED_CATEGORIES:
        raise ValueError("Unsupported RAG source type or category.")
    rag_delete_source(source_key)
    inserted = rag_upsert_chunks(chunks)
    upsert_rag_document_record(source_key, source_name, source_type, category, inserted, metadata=metadata)
    return {
        "source_key": source_key,
        "source_name": source_name,
        "source_type": source_type,
        "category": category,
        "chunk_count": inserted,
        "metadata": metadata or {},
    }


def get_conversation_records_for_rag(conversation_id: int | None = None) -> list[dict]:
    _require_rag_enabled()
    ensure_supported_rag_sources()
    with get_db() as conn:
        if conversation_id is None:
            rows = conn.execute("SELECT id, title FROM conversations ORDER BY updated_at DESC").fetchall()
        else:
            rows = conn.execute(
                "SELECT id, title FROM conversations WHERE id = ? ORDER BY updated_at DESC",
                (conversation_id,),
            ).fetchall()

        conversations = []
        for row in rows:
            messages = conn.execute(
                "SELECT role, content, metadata FROM messages WHERE conversation_id = ? ORDER BY position, id",
                (row["id"],),
            ).fetchall()
            conversation_messages = []
            tool_messages = []
            for msg in messages:
                role = str(msg["role"] or "").strip()
                metadata = parse_message_metadata(msg["metadata"])
                content = str(msg["content"] or "").strip()
                if role == "user":
                    content = build_user_message_for_model(content, metadata)
                if role == "summary" and content:
                    conversation_messages.append({"role": "assistant", "content": content})
                elif role in {"user", "assistant"} and content:
                    conversation_messages.append({"role": role, "content": content})

                for tool_index, tool_result in enumerate(extract_message_tool_results(metadata), start=1):
                    tool_messages.append(
                        {
                            "role": "tool",
                            "content": build_tool_result_record_content(tool_result, tool_index),
                        }
                    )

            conversations.append(
                {
                    "conversation_id": row["id"],
                    "title": row["title"],
                    "messages": conversation_messages,
                    "tool_results": tool_messages,
                }
            )
    return conversations


def sync_conversations_to_rag(conversation_id: int | None = None, force: bool = False) -> list[dict]:
    _require_rag_enabled()
    ensure_supported_rag_sources()
    synced = []
    conversations = get_conversation_records_for_rag(conversation_id)
    for conversation in conversations:
        conversation_key = conversation_rag_source_key(RAG_SOURCE_CONVERSATION, conversation["conversation_id"])
        conversation_name = _conversation_rag_source_name(
            RAG_SOURCE_CONVERSATION,
            conversation["conversation_id"],
            conversation["title"],
        )
        conversation_signature = _build_rag_sync_signature(
            {
                "title": conversation["title"],
                "messages": conversation["messages"],
            }
        )
        conversation_metadata = _build_conversation_sync_metadata(conversation, conversation_key, conversation_signature)
        if conversation["messages"]:
            if _conversation_source_needs_sync(conversation_key, conversation_signature, force=force):
                conversation_chunks = chunks_from_records(
                    conversation["messages"],
                    source_name=conversation_name,
                    source_type=RAG_SOURCE_CONVERSATION,
                    category=RAG_SOURCE_CONVERSATION,
                    metadata=conversation_metadata,
                )
                if conversation_chunks:
                    synced.append(
                        ingest_rag_chunks(
                            source_key=conversation_key,
                            source_name=conversation_name,
                            source_type=RAG_SOURCE_CONVERSATION,
                            category=RAG_SOURCE_CONVERSATION,
                            chunks=conversation_chunks,
                            metadata=conversation_metadata,
                        )
                    )
                else:
                    _delete_rag_source_if_present(conversation_key)
        else:
            _delete_rag_source_if_present(conversation_key)

        tool_key = conversation_rag_source_key(RAG_SOURCE_TOOL_RESULT, conversation["conversation_id"])
        tool_name = _conversation_rag_source_name(
            RAG_SOURCE_TOOL_RESULT,
            conversation["conversation_id"],
            conversation["title"],
        )
        tool_signature = _build_rag_sync_signature(
            {
                "title": conversation["title"],
                "tool_results": conversation["tool_results"],
            }
        )
        tool_metadata = _build_conversation_sync_metadata(conversation, tool_key, tool_signature)
        if conversation["tool_results"]:
            if _conversation_source_needs_sync(tool_key, tool_signature, force=force):
                tool_chunks = chunks_from_records(
                    conversation["tool_results"],
                    source_name=tool_name,
                    source_type=RAG_SOURCE_TOOL_RESULT,
                    category=RAG_SOURCE_TOOL_RESULT,
                    metadata=tool_metadata,
                )
                if tool_chunks:
                    synced.append(
                        ingest_rag_chunks(
                            source_key=tool_key,
                            source_name=tool_name,
                            source_type=RAG_SOURCE_TOOL_RESULT,
                            category=RAG_SOURCE_TOOL_RESULT,
                            chunks=tool_chunks,
                            metadata=tool_metadata,
                        )
                    )
                else:
                    _delete_rag_source_if_present(tool_key)
        else:
            _delete_rag_source_if_present(tool_key)
    return synced


def sync_conversations_to_rag_safe(conversation_id: int | None = None, force: bool = False) -> list[dict]:
    if not RAG_ENABLED:
        return []
    try:
        return sync_conversations_to_rag(conversation_id=conversation_id, force=force)
    except Exception:
        logger.exception("Automatic conversation sync failed", extra={"conversation_id": conversation_id, "force": force})
        return []
