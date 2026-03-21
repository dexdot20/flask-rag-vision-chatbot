from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Iterable

DEFAULT_CHUNK_SIZE = 1800
DEFAULT_CHUNK_OVERLAP = 250
MAX_METADATA_VALUE_LENGTH = 500


@dataclass(slots=True)
class Chunk:
    id: str
    text: str
    source_name: str
    source_type: str
    category: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)

    def to_metadata(self) -> dict:
        metadata = {
            "source_name": self.source_name,
            "source_type": self.source_type,
            "category": normalize_category(self.category),
            "chunk_index": self.chunk_index,
        }
        for key, value in (self.metadata or {}).items():
            if value in (None, ""):
                continue
            metadata[str(key)] = _normalize_metadata_value(value)
        return metadata


def normalize_category(category: str | None) -> str:
    cleaned = re.sub(r"[^a-z0-9_-]+", "-", str(category or "general").strip().lower()).strip("-")
    return cleaned or "general"


def _normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _normalize_metadata_value(value):
    if isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(item).strip() for item in value if str(item).strip())[:MAX_METADATA_VALUE_LENGTH]
    return str(value).strip()[:MAX_METADATA_VALUE_LENGTH]


def _paragraphs_from_text(text: str) -> list[str]:
    normalized = _normalize_whitespace(text)
    if not normalized:
        return []
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", normalized) if part.strip()]
    if paragraphs:
        return paragraphs
    return [normalized]


def _slice_long_paragraph(paragraph: str, chunk_size: int, overlap: int) -> Iterable[str]:
    if len(paragraph) <= chunk_size:
        yield paragraph
        return

    start = 0
    step = max(1, chunk_size - max(0, overlap))
    while start < len(paragraph):
        end = min(len(paragraph), start + chunk_size)
        yield paragraph[start:end].strip()
        if end >= len(paragraph):
            break
        start += step


def split_text_into_chunks(
    text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP
) -> list[str]:
    chunk_size = max(300, int(chunk_size or DEFAULT_CHUNK_SIZE))
    overlap = max(0, min(int(overlap or DEFAULT_CHUNK_OVERLAP), chunk_size // 2))
    paragraphs = _paragraphs_from_text(text)
    if not paragraphs:
        return []

    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        pieces = list(_slice_long_paragraph(paragraph, chunk_size, overlap))
        for piece in pieces:
            if not current:
                current = piece
                continue
            candidate = f"{current}\n\n{piece}" if current else piece
            if len(candidate) <= chunk_size:
                current = candidate
                continue
            chunks.append(current.strip())
            carry = current[-overlap:].strip() if overlap and len(current) > overlap else ""
            current = f"{carry}\n\n{piece}".strip() if carry else piece
            if len(current) > chunk_size:
                overflow_parts = list(_slice_long_paragraph(current, chunk_size, overlap))
                chunks.extend(overflow_parts[:-1])
                current = overflow_parts[-1]

    if current.strip():
        chunks.append(current.strip())

    deduped: list[str] = []
    for chunk in chunks:
        cleaned = _normalize_whitespace(chunk)
        if cleaned and cleaned not in deduped:
            deduped.append(cleaned)
    return deduped


def _build_chunk_id(source_name: str, source_type: str, category: str, chunk_index: int, text: str) -> str:
    digest = hashlib.sha1(f"{source_name}|{source_type}|{category}|{chunk_index}|{text}".encode("utf-8")).hexdigest()
    return f"chunk-{digest}"


def chunk_text_document(
    text: str,
    source_name: str,
    source_type: str,
    category: str,
    metadata: dict | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Chunk]:
    normalized_category = normalize_category(category)
    normalized_text = _normalize_whitespace(text)
    if not normalized_text:
        return []

    chunks = split_text_into_chunks(normalized_text, chunk_size=chunk_size, overlap=overlap)
    base_metadata = dict(metadata or {})
    items: list[Chunk] = []
    for index, chunk_text in enumerate(chunks):
        chunk_id = _build_chunk_id(source_name, source_type, normalized_category, index, chunk_text)
        items.append(
            Chunk(
                id=chunk_id,
                text=chunk_text,
                source_name=source_name,
                source_type=source_type,
                category=normalized_category,
                chunk_index=index,
                metadata=base_metadata,
            )
        )
    return items
