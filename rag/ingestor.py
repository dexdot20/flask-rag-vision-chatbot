from __future__ import annotations

from typing import Iterable

from .chunker import Chunk, chunk_text_document, normalize_category


def chunks_from_text(
    text: str,
    source_name: str,
    source_type: str,
    category: str,
    metadata: dict | None = None,
) -> list[Chunk]:
    return chunk_text_document(
        text=text,
        source_name=source_name,
        source_type=source_type,
        category=normalize_category(category),
        metadata=metadata,
    )


def chunks_from_records(
    records: Iterable[dict],
    source_name: str,
    source_type: str,
    category: str,
    metadata: dict | None = None,
) -> list[Chunk]:
    parts: list[str] = []
    for index, record in enumerate(records, start=1):
        role = str(record.get("role") or "unknown").strip() or "unknown"
        content = str(record.get("content") or "").strip()
        if not content:
            continue
        parts.append(f"[{index}] {role}: {content}")
    if not parts:
        return []
    return chunks_from_text(
        text="\n\n".join(parts),
        source_name=source_name,
        source_type=source_type,
        category=category,
        metadata=metadata,
    )
