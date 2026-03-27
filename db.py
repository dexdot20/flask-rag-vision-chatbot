from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from pathlib import Path
from typing import Iterable
from uuid import uuid4

from flask import current_app, has_app_context

from canvas_service import extract_canvas_active_document_id, extract_canvas_documents
from config import (
    CACHE_TTL_HOURS,
    CANVAS_EXPAND_DEFAULT_MAX_LINES,
    CANVAS_PROMPT_DEFAULT_MAX_LINES,
    CANVAS_SCROLL_WINDOW_LINES,
    CHAT_SUMMARY_ALLOWED_MODES,
    CHAT_SUMMARY_MODE,
    CHAT_SUMMARY_TRIGGER_TOKEN_COUNT,
    CONTENT_MAX_CHARS,
    DB_PATH,
    DEFAULT_ACTIVE_TOOL_NAMES,
    DEFAULT_SETTINGS,
    DOCUMENT_STORAGE_DIR,
    FETCH_RAW_TOOL_RESULT_MAX_TEXT_CHARS,
    FETCH_SUMMARY_TOKEN_THRESHOLD,
    IMAGE_STORAGE_DIR,
    PROMPT_MAX_INPUT_TOKENS,
    PROMPT_PREFLIGHT_SUMMARY_TOKEN_COUNT,
    PROMPT_RAG_MAX_TOKENS,
    PROMPT_RECENT_HISTORY_MAX_TOKENS,
    PROMPT_RESPONSE_TOKEN_RESERVE,
    PROMPT_SUMMARY_MAX_TOKENS,
    PROMPT_TOOL_MEMORY_MAX_TOKENS,
    RAG_CONTEXT_SIZE_PRESETS,
    RAG_DEFAULT_CONTEXT_SIZE_PRESET,
    RAG_DEFAULT_SENSITIVITY_PRESET,
    RAG_ENABLED,
    RAG_SENSITIVITY_PRESETS,
    RAG_TOOL_RESULT_MAX_TEXT_CHARS,
    RAG_TOOL_RESULT_SUMMARY_MAX_CHARS,
    SUMMARY_RETRY_MIN_SOURCE_TOKENS,
    SUMMARY_SOURCE_TARGET_TOKENS,
)
from token_utils import estimate_text_tokens

_db_path = DB_PATH
MESSAGE_USAGE_BREAKDOWN_KEYS = (
    "core_instructions",
    "tool_specs",
    "canvas",
    "scratchpad",
    "tool_trace",
    "tool_memory",
    "rag_context",
    "internal_state",
    "user_messages",
    "assistant_history",
    "assistant_tool_calls",
    "tool_results",
    "unknown_provider_overhead",
)
LEGACY_MESSAGE_USAGE_BREAKDOWN_KEYS = {
    "core_instructions": ("system_prompt", "final_instruction"),
}
MESSAGE_TOOL_TRACE_STATES = {"running", "done", "error"}
VISIBLE_CHAT_ROLES = {"user", "assistant", "summary"}
SUMMARY_TRIGGER_TOKEN_ROLES = {"user", "assistant", "tool"}


def configure_db_path(path: str | None = None) -> str:
    global _db_path
    _db_path = path or DB_PATH
    return _db_path


def get_configured_db_path() -> str:
    if has_app_context():
        return current_app.config.get("DATABASE_PATH") or _db_path
    return _db_path


def get_db():
    db_path = get_configured_db_path()
    db_dir = os.path.dirname(db_path)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 30000")
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    with get_db() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                title      TEXT    NOT NULL DEFAULT 'New Chat',
                model      TEXT    NOT NULL DEFAULT 'deepseek-chat',
                created_at TEXT    NOT NULL DEFAULT (datetime('now')),
                updated_at TEXT    NOT NULL DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS messages (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id   INTEGER NOT NULL,
                position          INTEGER,
                role              TEXT    NOT NULL,
                content           TEXT    NOT NULL,
                metadata          TEXT,
                tool_calls        TEXT,
                tool_call_id      TEXT,
                prompt_tokens     INTEGER,
                completion_tokens INTEGER,
                total_tokens      INTEGER,
                deleted_at        TEXT,
                created_at        TEXT    NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS app_settings (
                key        TEXT PRIMARY KEY,
                value      TEXT NOT NULL DEFAULT '',
                updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS user_profile (
                key        TEXT PRIMARY KEY,
                value      TEXT NOT NULL,
                confidence REAL NOT NULL DEFAULT 1.0,
                source     TEXT,
                updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS image_assets (
                image_id         TEXT PRIMARY KEY,
                conversation_id  INTEGER NOT NULL,
                message_id       INTEGER,
                filename         TEXT NOT NULL,
                mime_type        TEXT NOT NULL,
                storage_path     TEXT NOT NULL,
                initial_analysis TEXT,
                created_at       TEXT NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE,
                FOREIGN KEY (message_id) REFERENCES messages(id) ON DELETE SET NULL
            );
            CREATE TABLE IF NOT EXISTS web_cache (
                key       TEXT PRIMARY KEY,
                value     TEXT NOT NULL,
                cached_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS rag_documents (
                source_key TEXT PRIMARY KEY,
                source_name TEXT NOT NULL,
                source_type TEXT NOT NULL,
                category TEXT NOT NULL DEFAULT 'general',
                chunk_count INTEGER NOT NULL DEFAULT 0,
                metadata TEXT,
                expires_at TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            CREATE TRIGGER IF NOT EXISTS trg_messages_assign_position
            AFTER INSERT ON messages
            FOR EACH ROW
            WHEN NEW.position IS NULL
            BEGIN
                UPDATE messages
                   SET position = (
                       SELECT COALESCE(MAX(position), 0) + 1
                       FROM messages
                       WHERE conversation_id = NEW.conversation_id
                         AND id <> NEW.id
                   )
                 WHERE id = NEW.id;
            END;
            CREATE TABLE IF NOT EXISTS file_assets (
                file_id          TEXT PRIMARY KEY,
                conversation_id  INTEGER NOT NULL,
                message_id       INTEGER,
                filename         TEXT NOT NULL,
                mime_type        TEXT NOT NULL,
                storage_path     TEXT NOT NULL,
                extracted_text   TEXT,
                created_at       TEXT NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE,
                FOREIGN KEY (message_id) REFERENCES messages(id) ON DELETE SET NULL
            );
            CREATE INDEX IF NOT EXISTS idx_image_assets_conversation_created
            ON image_assets(conversation_id, created_at, image_id);
            CREATE INDEX IF NOT EXISTS idx_file_assets_conversation_created
            ON file_assets(conversation_id, created_at, file_id);
            """
        )


def ensure_messages_metadata_column() -> None:
    with get_db() as conn:
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(messages)").fetchall()}
        if "metadata" not in columns:
            conn.execute("ALTER TABLE messages ADD COLUMN metadata TEXT")


def ensure_messages_tool_history_columns() -> None:
    with get_db() as conn:
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(messages)").fetchall()}
        if "tool_calls" not in columns:
            conn.execute("ALTER TABLE messages ADD COLUMN tool_calls TEXT")
        if "tool_call_id" not in columns:
            conn.execute("ALTER TABLE messages ADD COLUMN tool_call_id TEXT")


def ensure_messages_position_column() -> None:
    with get_db() as conn:
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(messages)").fetchall()}
        if "position" not in columns:
            conn.execute("ALTER TABLE messages ADD COLUMN position INTEGER")

        rows = conn.execute(
            "SELECT id, conversation_id, position FROM messages ORDER BY conversation_id, id"
        ).fetchall()
        last_position_by_conversation = {}
        for row in rows:
            conversation_id = row["conversation_id"]
            current_position = row["position"]
            if isinstance(current_position, int) and current_position > 0:
                last_position_by_conversation[conversation_id] = current_position
                continue
            next_position = last_position_by_conversation.get(conversation_id, 0) + 1
            conn.execute("UPDATE messages SET position = ? WHERE id = ?", (next_position, row["id"]))
            last_position_by_conversation[conversation_id] = next_position
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_messages_conversation_position ON messages(conversation_id, position, id)"
        )


def ensure_messages_deleted_at_column() -> None:
    with get_db() as conn:
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(messages)").fetchall()}
        if "deleted_at" not in columns:
            conn.execute("ALTER TABLE messages ADD COLUMN deleted_at TEXT")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_messages_conversation_deleted_position ON messages(conversation_id, deleted_at, position, id)"
        )


def ensure_rag_documents_expires_at_column() -> None:
    with get_db() as conn:
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(rag_documents)").fetchall()}
        if "expires_at" not in columns:
            conn.execute("ALTER TABLE rag_documents ADD COLUMN expires_at TEXT")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_rag_documents_expires_at ON rag_documents(expires_at, category, updated_at)"
        )


def delete_rag_document_records(source_keys: Iterable[str]) -> None:
    normalized_keys = [str(source_key or "").strip() for source_key in source_keys if str(source_key or "").strip()]
    if not normalized_keys:
        return

    placeholders = ", ".join("?" for _ in normalized_keys)
    with get_db() as conn:
        conn.execute(f"DELETE FROM rag_documents WHERE source_key IN ({placeholders})", tuple(normalized_keys))


def get_expired_rag_document_source_keys(now_iso: str | None = None) -> list[str]:
    reference_time = str(now_iso or "").strip() or datetime_utc_now_iso()
    with get_db() as conn:
        rows = conn.execute(
            "SELECT source_key FROM rag_documents WHERE expires_at IS NOT NULL AND expires_at <= ? ORDER BY expires_at ASC",
            (reference_time,),
        ).fetchall()
    return [str(row["source_key"] or "").strip() for row in rows if str(row["source_key"] or "").strip()]


def datetime_utc_now_iso() -> str:
    with get_db() as conn:
        row = conn.execute("SELECT datetime('now') AS now_iso").fetchone()
    return str(row["now_iso"] or "").strip()


def initialize_database() -> None:
    init_db()
    ensure_messages_metadata_column()
    ensure_messages_tool_history_columns()
    ensure_messages_position_column()
    ensure_messages_deleted_at_column()
    ensure_rag_documents_expires_at_column()


def _normalize_user_profile_value(value, max_length: int = 500) -> str:
    return " ".join(str(value or "").strip().split())[:max_length]


def _build_user_profile_fact_key(value: str) -> str:
    normalized_value = _normalize_user_profile_value(value)
    digest = hashlib.sha1(normalized_value.encode("utf-8")).hexdigest()
    return f"fact:{digest}"


def _is_user_profile_fact_candidate(value: str) -> bool:
    normalized_value = _normalize_user_profile_value(value).casefold()
    if not normalized_value:
        return False
    keywords = (
        "the user",
        "user prefers",
        "user wants",
        "user is",
        "user uses",
        "user works",
        "prefers",
        "likes",
        "kullanıcı",
        "kullanici",
        "tercih",
        "istiyor",
        "kullanıyor",
        "çalışıyor",
        "works on",
        "working on",
        "name is",
        "adı",
    )
    return any(keyword in normalized_value for keyword in keywords)


def upsert_user_profile_entry(key: str, value: str, confidence: float = 1.0, source: str = "manual") -> dict | None:
    normalized_key = str(key or "").strip()[:120]
    normalized_value = _normalize_user_profile_value(value)
    normalized_source = str(source or "").strip()[:80] or "manual"
    try:
        normalized_confidence = float(confidence)
    except (TypeError, ValueError):
        normalized_confidence = 1.0
    normalized_confidence = max(0.0, min(1.0, normalized_confidence))

    if not normalized_key or not normalized_value:
        return None

    with get_db() as conn:
        conn.execute(
            """INSERT INTO user_profile (key, value, confidence, source, updated_at)
               VALUES (?, ?, ?, ?, datetime('now'))
               ON CONFLICT(key) DO UPDATE SET
                   value = excluded.value,
                   confidence = excluded.confidence,
                   source = excluded.source,
                   updated_at = datetime('now')""",
            (normalized_key, normalized_value, normalized_confidence, normalized_source),
        )
    return {
        "key": normalized_key,
        "value": normalized_value,
        "confidence": normalized_confidence,
        "source": normalized_source,
    }


def upsert_user_profile_facts(facts: list[str], confidence: float = 0.8, source: str = "summary_extraction") -> list[dict]:
    stored: list[dict] = []
    for raw_fact in facts or []:
        normalized_fact = _normalize_user_profile_value(raw_fact)
        if not normalized_fact or not _is_user_profile_fact_candidate(normalized_fact):
            continue
        entry = upsert_user_profile_entry(
            _build_user_profile_fact_key(normalized_fact),
            normalized_fact,
            confidence=confidence,
            source=source,
        )
        if entry is not None:
            stored.append(entry)
    return stored


def get_user_profile_entries(limit: int | None = None) -> list[dict]:
    query = "SELECT key, value, confidence, source, updated_at FROM user_profile ORDER BY confidence DESC, updated_at DESC, key ASC"
    params: tuple[object, ...] = ()
    if isinstance(limit, int) and limit > 0:
        query += " LIMIT ?"
        params = (limit,)
    with get_db() as conn:
        rows = conn.execute(query, params).fetchall()
    return [
        {
            "key": str(row["key"] or "").strip(),
            "value": str(row["value"] or "").strip(),
            "confidence": float(row["confidence"] or 0.0),
            "source": str(row["source"] or "").strip(),
            "updated_at": str(row["updated_at"] or "").strip(),
        }
        for row in rows
        if str(row["key"] or "").strip() and str(row["value"] or "").strip()
    ]


def build_user_profile_system_context(max_tokens: int = 500, limit: int = 12) -> str | None:
    if max_tokens <= 0:
        return None

    entries = get_user_profile_entries(limit=limit)
    if not entries:
        return None

    lines: list[str] = []
    total_tokens = 0
    for entry in entries:
        line = f"- {entry['value']}"
        line_tokens = estimate_text_tokens(line)
        if lines and total_tokens + line_tokens > max_tokens:
            break
        if not lines and line_tokens > max_tokens:
            break
        lines.append(line)
        total_tokens += line_tokens
    if not lines:
        return None
    return "\n".join(lines)


def _guess_extension_for_mime_type(mime_type: str) -> str:
    normalized = str(mime_type or "").strip().lower()
    return {
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/webp": ".webp",
    }.get(normalized, "")


def _normalize_initial_image_analysis(value) -> dict | None:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except Exception:
            return None
    if not isinstance(value, dict):
        return None

    cleaned = {}
    for key in ("ocr_text", "vision_summary", "assistant_guidance"):
        text = str(value.get(key) or "").strip()
        if text:
            cleaned[key] = text[:CONTENT_MAX_CHARS]

    key_points = value.get("key_points") if isinstance(value.get("key_points"), list) else []
    normalized_points = []
    for point in key_points[:8]:
        point_text = str(point or "").strip()
        if point_text and point_text not in normalized_points:
            normalized_points.append(point_text[:300])
    if normalized_points:
        cleaned["key_points"] = normalized_points

    return cleaned or None


def create_image_asset(conversation_id: int, filename: str, mime_type: str, image_bytes: bytes) -> dict:
    normalized_filename = os.path.basename(str(filename or "").strip())[:255]
    normalized_mime_type = str(mime_type or "").strip().lower()[:120]
    if not conversation_id:
        raise ValueError("conversation_id is required to persist an image.")
    if not normalized_filename:
        raise ValueError("filename is required.")
    if not image_bytes:
        raise ValueError("image_bytes is required.")

    image_id = uuid4().hex
    extension = _guess_extension_for_mime_type(normalized_mime_type)
    relative_path = os.path.join(image_id[:2], f"{image_id}{extension}")
    absolute_path = os.path.join(IMAGE_STORAGE_DIR, relative_path)
    os.makedirs(os.path.dirname(absolute_path), exist_ok=True)

    with open(absolute_path, "wb") as handle:
        handle.write(image_bytes)

    with get_db() as conn:
        conn.execute(
            """INSERT INTO image_assets (
                   image_id, conversation_id, filename, mime_type, storage_path
               ) VALUES (?, ?, ?, ?, ?)""",
            (image_id, conversation_id, normalized_filename, normalized_mime_type, absolute_path),
        )
        row = conn.execute(
            """SELECT image_id, conversation_id, message_id, filename, mime_type,
                      storage_path, initial_analysis, created_at
               FROM image_assets WHERE image_id = ?""",
            (image_id,),
        ).fetchone()
    return image_asset_row_to_dict(row)


def update_image_asset(image_id: str, *, message_id: int | None = None, initial_analysis: dict | None = None) -> dict | None:
    normalized_image_id = str(image_id or "").strip()
    if not normalized_image_id:
        return None

    assignments = []
    params = []
    if message_id is not None:
        assignments.append("message_id = ?")
        params.append(int(message_id))
    normalized_analysis = _normalize_initial_image_analysis(initial_analysis)
    if normalized_analysis is not None:
        assignments.append("initial_analysis = ?")
        params.append(json.dumps(normalized_analysis, ensure_ascii=False))

    if assignments:
        with get_db() as conn:
            conn.execute(
                f"UPDATE image_assets SET {', '.join(assignments)} WHERE image_id = ?",
                (*params, normalized_image_id),
            )

    return get_image_asset(normalized_image_id)


def image_asset_row_to_dict(row) -> dict | None:
    if not row:
        return None
    return {
        "image_id": row["image_id"],
        "conversation_id": row["conversation_id"],
        "message_id": row["message_id"],
        "filename": row["filename"],
        "mime_type": row["mime_type"],
        "storage_path": row["storage_path"],
        "initial_analysis": _normalize_initial_image_analysis(row["initial_analysis"]),
        "created_at": row["created_at"],
    }


def get_image_asset(image_id: str, conversation_id: int | None = None) -> dict | None:
    normalized_image_id = str(image_id or "").strip()
    if not normalized_image_id:
        return None
    query = (
        "SELECT image_id, conversation_id, message_id, filename, mime_type, storage_path, initial_analysis, created_at "
        "FROM image_assets WHERE image_id = ?"
    )
    params = [normalized_image_id]
    if conversation_id is not None:
        query += " AND conversation_id = ?"
        params.append(int(conversation_id))

    with get_db() as conn:
        row = conn.execute(query, tuple(params)).fetchone()
    return image_asset_row_to_dict(row)


def get_latest_conversation_image_asset(conversation_id: int) -> dict | None:
    with get_db() as conn:
        row = conn.execute(
            """SELECT image_id, conversation_id, message_id, filename, mime_type, storage_path, initial_analysis, created_at
               FROM image_assets
               WHERE conversation_id = ?
               ORDER BY created_at DESC, image_id DESC
               LIMIT 1""",
            (conversation_id,),
        ).fetchone()
    return image_asset_row_to_dict(row)


def read_image_asset_bytes(image_id: str, conversation_id: int | None = None) -> tuple[dict | None, bytes | None]:
    asset = get_image_asset(image_id, conversation_id=conversation_id)
    if not asset:
        return None, None
    storage_path = str(asset.get("storage_path") or "").strip()
    if not storage_path or not os.path.isfile(storage_path):
        return asset, None
    with open(storage_path, "rb") as handle:
        return asset, handle.read()


def delete_conversation_image_assets(conversation_id: int) -> list[str]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT storage_path FROM image_assets WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchall()
        conn.execute("DELETE FROM image_assets WHERE conversation_id = ?", (conversation_id,))

    deleted_paths = []
    for row in rows:
        storage_path = str(row["storage_path"] or "").strip()
        if not storage_path:
            continue
        try:
            os.remove(storage_path)
            deleted_paths.append(storage_path)
        except FileNotFoundError:
            continue
        except OSError:
            continue

        parent = Path(storage_path).parent
        root = Path(IMAGE_STORAGE_DIR)
        while parent != root and parent.exists():
            try:
                parent.rmdir()
            except OSError:
                break
            parent = parent.parent

    return deleted_paths


def delete_image_asset(image_id: str, conversation_id: int | None = None) -> bool:
    asset = get_image_asset(image_id, conversation_id=conversation_id)
    if not asset:
        return False

    with get_db() as conn:
        conn.execute("DELETE FROM image_assets WHERE image_id = ?", (asset["image_id"],))

    storage_path = str(asset.get("storage_path") or "").strip()
    if storage_path:
        try:
            os.remove(storage_path)
        except FileNotFoundError:
            pass
        except OSError:
            pass
    return True


# --- File asset CRUD ---------------------------------------------------

def _guess_extension_for_document_mime(mime_type: str) -> str:
    normalized = str(mime_type or "").strip().lower()
    return {
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
        "application/pdf": ".pdf",
        "text/plain": ".txt",
        "text/csv": ".csv",
        "text/markdown": ".md",
    }.get(normalized, "")


def create_file_asset(conversation_id: int, filename: str, mime_type: str, doc_bytes: bytes, extracted_text: str | None = None) -> dict:
    normalized_filename = os.path.basename(str(filename or "").strip())[:255]
    normalized_mime_type = str(mime_type or "").strip().lower()[:120]
    if not conversation_id:
        raise ValueError("conversation_id is required to persist a file.")
    if not normalized_filename:
        raise ValueError("filename is required.")
    if not doc_bytes:
        raise ValueError("doc_bytes is required.")

    file_id = uuid4().hex
    extension = _guess_extension_for_document_mime(normalized_mime_type)
    relative_path = os.path.join(file_id[:2], f"{file_id}{extension}")
    absolute_path = os.path.join(DOCUMENT_STORAGE_DIR, relative_path)
    os.makedirs(os.path.dirname(absolute_path), exist_ok=True)

    with open(absolute_path, "wb") as handle:
        handle.write(doc_bytes)

    with get_db() as conn:
        conn.execute(
            """INSERT INTO file_assets (
                   file_id, conversation_id, filename, mime_type, storage_path, extracted_text
               ) VALUES (?, ?, ?, ?, ?, ?)""",
            (file_id, conversation_id, normalized_filename, normalized_mime_type, absolute_path, extracted_text),
        )
        row = conn.execute(
            """SELECT file_id, conversation_id, message_id, filename, mime_type,
                      storage_path, extracted_text, created_at
               FROM file_assets WHERE file_id = ?""",
            (file_id,),
        ).fetchone()
    return _file_asset_row_to_dict(row)


def update_file_asset(file_id: str, *, message_id: int | None = None) -> dict | None:
    normalized_id = str(file_id or "").strip()
    if not normalized_id:
        return None
    assignments = []
    params = []
    if message_id is not None:
        assignments.append("message_id = ?")
        params.append(int(message_id))
    if assignments:
        with get_db() as conn:
            conn.execute(
                f"UPDATE file_assets SET {', '.join(assignments)} WHERE file_id = ?",
                (*params, normalized_id),
            )
    return get_file_asset(normalized_id)


def _file_asset_row_to_dict(row) -> dict | None:
    if not row:
        return None
    return {
        "file_id": row["file_id"],
        "conversation_id": row["conversation_id"],
        "message_id": row["message_id"],
        "filename": row["filename"],
        "mime_type": row["mime_type"],
        "storage_path": row["storage_path"],
        "extracted_text": row["extracted_text"],
        "created_at": row["created_at"],
    }


def get_file_asset(file_id: str, conversation_id: int | None = None) -> dict | None:
    normalized_id = str(file_id or "").strip()
    if not normalized_id:
        return None
    query = (
        "SELECT file_id, conversation_id, message_id, filename, mime_type, storage_path, extracted_text, created_at "
        "FROM file_assets WHERE file_id = ?"
    )
    params = [normalized_id]
    if conversation_id is not None:
        query += " AND conversation_id = ?"
        params.append(int(conversation_id))
    with get_db() as conn:
        row = conn.execute(query, tuple(params)).fetchone()
    return _file_asset_row_to_dict(row)


def delete_file_asset(file_id: str, conversation_id: int | None = None) -> bool:
    asset = get_file_asset(file_id, conversation_id=conversation_id)
    if not asset:
        return False
    with get_db() as conn:
        conn.execute("DELETE FROM file_assets WHERE file_id = ?", (asset["file_id"],))
    storage_path = str(asset.get("storage_path") or "").strip()
    if storage_path:
        try:
            os.remove(storage_path)
        except (FileNotFoundError, OSError):
            pass
    return True


def delete_conversation_file_assets(conversation_id: int) -> list[str]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT storage_path FROM file_assets WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchall()
        conn.execute("DELETE FROM file_assets WHERE conversation_id = ?", (conversation_id,))

    deleted_paths = []
    for row in rows:
        storage_path = str(row["storage_path"] or "").strip()
        if not storage_path:
            continue
        try:
            os.remove(storage_path)
            deleted_paths.append(storage_path)
        except (FileNotFoundError, OSError):
            continue
        parent = Path(storage_path).parent
        root = Path(DOCUMENT_STORAGE_DIR)
        while parent != root and parent.exists():
            try:
                parent.rmdir()
            except OSError:
                break
            parent = parent.parent
    return deleted_paths


def parse_message_metadata(raw_metadata) -> dict:
    if isinstance(raw_metadata, dict):
        return raw_metadata
    if not raw_metadata:
        return {}
    try:
        parsed = json.loads(raw_metadata)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _normalize_message_attachment(entry) -> dict | None:
    if not isinstance(entry, dict):
        return None

    kind = str(entry.get("kind") or "").strip().lower()
    if kind not in {"image", "document"}:
        return None

    cleaned = {"kind": kind}
    if kind == "image":
        image_id = str(entry.get("image_id") or "").strip()[:64]
        image_name = str(entry.get("image_name") or "").strip()[:255]
        image_mime_type = str(entry.get("image_mime_type") or "").strip()[:120]
        ocr_text = str(entry.get("ocr_text") or "").strip()[:CONTENT_MAX_CHARS]
        vision_summary = str(entry.get("vision_summary") or "").strip()[:CONTENT_MAX_CHARS]
        assistant_guidance = str(entry.get("assistant_guidance") or "").strip()[:CONTENT_MAX_CHARS]
        key_points = entry.get("key_points") if isinstance(entry.get("key_points"), list) else []

        if image_id:
            cleaned["image_id"] = image_id
        if image_name:
            cleaned["image_name"] = image_name
        if image_mime_type:
            cleaned["image_mime_type"] = image_mime_type
        if ocr_text:
            cleaned["ocr_text"] = ocr_text
        if vision_summary:
            cleaned["vision_summary"] = vision_summary
        if assistant_guidance:
            cleaned["assistant_guidance"] = assistant_guidance
        if key_points:
            normalized_points = []
            for point in key_points[:8]:
                point_text = str(point or "").strip()
                if point_text and point_text not in normalized_points:
                    normalized_points.append(point_text[:300])
            if normalized_points:
                cleaned["key_points"] = normalized_points

        if not cleaned.get("image_id") and not cleaned.get("image_name"):
            return None
        return cleaned

    file_id = str(entry.get("file_id") or "").strip()[:64]
    file_name = str(entry.get("file_name") or "").strip()[:255]
    file_mime_type = str(entry.get("file_mime_type") or "").strip()[:120]
    file_context_block = str(entry.get("file_context_block") or "").strip()[:CONTENT_MAX_CHARS]

    if file_id:
        cleaned["file_id"] = file_id
    if file_name:
        cleaned["file_name"] = file_name
    if file_mime_type:
        cleaned["file_mime_type"] = file_mime_type
    if entry.get("file_text_truncated") is True:
        cleaned["file_text_truncated"] = True
    if file_context_block:
        cleaned["file_context_block"] = file_context_block

    if not cleaned.get("file_id") and not cleaned.get("file_name"):
        return None
    return cleaned


def extract_message_attachments(metadata: dict | None) -> list[dict]:
    source = metadata if isinstance(metadata, dict) else {}
    normalized = []
    seen = set()

    def append_attachment(raw_attachment) -> None:
        cleaned = _normalize_message_attachment(raw_attachment)
        if not cleaned:
            return
        if cleaned["kind"] == "image":
            dedupe_key = (
                "image",
                cleaned.get("image_id") or "",
                cleaned.get("image_name") or "",
            )
        else:
            dedupe_key = (
                "document",
                cleaned.get("file_id") or "",
                cleaned.get("file_name") or "",
            )
        if dedupe_key in seen:
            return
        seen.add(dedupe_key)
        normalized.append(cleaned)

    raw_attachments = source.get("attachments") if isinstance(source.get("attachments"), list) else []
    for entry in raw_attachments[:24]:
        append_attachment(entry)

    legacy_image = {
        "kind": "image",
        "image_id": source.get("image_id"),
        "image_name": source.get("image_name"),
        "image_mime_type": source.get("image_mime_type"),
        "ocr_text": source.get("ocr_text"),
        "vision_summary": source.get("vision_summary"),
        "assistant_guidance": source.get("assistant_guidance"),
        "key_points": source.get("key_points"),
    }
    append_attachment(legacy_image)

    legacy_document = {
        "kind": "document",
        "file_id": source.get("file_id"),
        "file_name": source.get("file_name"),
        "file_mime_type": source.get("file_mime_type"),
        "file_text_truncated": source.get("file_text_truncated") is True,
        "file_context_block": source.get("file_context_block"),
    }
    append_attachment(legacy_document)

    return normalized


def _normalize_message_tool_calls(raw_tool_calls) -> list[dict]:
    if isinstance(raw_tool_calls, str):
        try:
            raw_tool_calls = json.loads(raw_tool_calls)
        except Exception:
            return []

    if not isinstance(raw_tool_calls, list):
        return []

    normalized = []
    for entry in raw_tool_calls[:32]:
        if not isinstance(entry, dict):
            continue
        tool_id = str(entry.get("id") or "").strip()[:120]
        tool_type = str(entry.get("type") or "function").strip()[:40] or "function"
        function = entry.get("function") if isinstance(entry.get("function"), dict) else {}
        function_name = str(function.get("name") or "").strip()[:80]
        raw_arguments = function.get("arguments")
        if isinstance(raw_arguments, (dict, list)):
            arguments = json.dumps(raw_arguments, ensure_ascii=False)
        else:
            arguments = str(raw_arguments or "").strip()
        if not function_name:
            continue
        normalized.append(
            {
                "id": tool_id,
                "type": tool_type,
                "function": {
                    "name": function_name,
                    "arguments": arguments,
                },
            }
        )
    return normalized


def parse_message_tool_calls(raw_tool_calls) -> list[dict]:
    return _normalize_message_tool_calls(raw_tool_calls)


def serialize_message_tool_calls(tool_calls) -> str | None:
    normalized = _normalize_message_tool_calls(tool_calls)
    if not normalized:
        return None
    return json.dumps(normalized, ensure_ascii=False)


def _coerce_non_negative_int(value) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        normalized = int(value)
    except (TypeError, ValueError):
        return None
    return max(0, normalized)


def _normalize_usage_breakdown(breakdown: dict | None, target_total: int | None = None) -> dict | None:
    if not isinstance(breakdown, dict):
        return None

    normalized_breakdown = {}
    for key in MESSAGE_USAGE_BREAKDOWN_KEYS:
        if key == "core_instructions":
            has_core_source = "core_instructions" in breakdown or any(
                legacy_key in breakdown for legacy_key in LEGACY_MESSAGE_USAGE_BREAKDOWN_KEYS.get(key, ())
            )
            if not has_core_source:
                normalized = None
            else:
                raw_total = breakdown.get("core_instructions")
                normalized = _coerce_non_negative_int(raw_total) or 0
                for legacy_key in LEGACY_MESSAGE_USAGE_BREAKDOWN_KEYS.get(key, ()):
                    normalized += _coerce_non_negative_int(breakdown.get(legacy_key)) or 0
        else:
            raw_value = breakdown.get(key)
            if raw_value is None:
                for legacy_key in LEGACY_MESSAGE_USAGE_BREAKDOWN_KEYS.get(key, ()): 
                    if legacy_key in breakdown:
                        raw_value = breakdown.get(legacy_key)
                        break
            normalized = _coerce_non_negative_int(raw_value)
        if normalized is not None:
            normalized_breakdown[key] = normalized

    if not normalized_breakdown:
        return None

    if target_total is None:
        return normalized_breakdown

    adjusted = {key: max(0, int(value)) for key, value in normalized_breakdown.items() if value and value > 0}
    current_total = sum(adjusted.values())
    if current_total < target_total:
        adjusted["unknown_provider_overhead"] = adjusted.get("unknown_provider_overhead", 0) + (target_total - current_total)
        return adjusted

    overflow = current_total - target_total
    if overflow <= 0:
        return adjusted

    reduction_order = (
        "tool_specs",
        "internal_state",
        "assistant_tool_calls",
        "tool_results",
        "canvas",
        "scratchpad",
        "tool_trace",
        "tool_memory",
        "rag_context",
        "assistant_history",
        "user_messages",
        "core_instructions",
    )
    for key in reduction_order:
        if overflow <= 0:
            break
        available = adjusted.get(key, 0)
        if available <= 0:
            continue
        reduction = min(available, overflow)
        adjusted[key] = available - reduction
        overflow -= reduction

    if overflow > 0:
        for key, available in sorted(adjusted.items(), key=lambda item: item[1], reverse=True):
            if overflow <= 0:
                break
            if available <= 0:
                continue
            reduction = min(available, overflow)
            adjusted[key] = available - reduction
            overflow -= reduction

    return {key: value for key, value in adjusted.items() if value > 0}


def _normalize_message_usage_call(value: dict | None) -> dict | None:
    if not isinstance(value, dict):
        return None

    cleaned = {}
    for key in (
        "index",
        "step",
        "message_count",
        "tool_schema_tokens",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "estimated_input_tokens",
    ):
        normalized = _coerce_non_negative_int(value.get(key))
        if normalized is not None:
            cleaned[key] = normalized

    call_type = str(value.get("call_type") or "").strip()[:40]
    if call_type:
        cleaned["call_type"] = call_type

    retry_reason = str(value.get("retry_reason") or "").strip()[:80]
    if retry_reason:
        cleaned["retry_reason"] = retry_reason

    if value.get("is_retry") is True:
        cleaned["is_retry"] = True
    if value.get("missing_provider_usage") is True:
        cleaned["missing_provider_usage"] = True

    target_total = cleaned.get("prompt_tokens")
    normalized_breakdown = _normalize_usage_breakdown(
        value.get("input_breakdown"),
        target_total=target_total or cleaned.get("estimated_input_tokens"),
    )
    if normalized_breakdown:
        cleaned["input_breakdown"] = normalized_breakdown

    if target_total is not None:
        cleaned["estimated_input_tokens"] = target_total
    elif normalized_breakdown:
        cleaned["estimated_input_tokens"] = sum(normalized_breakdown.values())

    return cleaned or None


def _normalize_message_usage(value: dict | None) -> dict | None:
    if not isinstance(value, dict):
        return None

    cleaned = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens", "estimated_input_tokens"):
        normalized = _coerce_non_negative_int(value.get(key))
        if normalized is not None:
            cleaned[key] = normalized

    normalized_breakdown = _normalize_usage_breakdown(
        value.get("input_breakdown"),
        target_total=cleaned.get("prompt_tokens") or cleaned.get("estimated_input_tokens"),
    )
    if normalized_breakdown:
        cleaned["input_breakdown"] = normalized_breakdown

    model_calls = []
    raw_model_calls = value.get("model_calls") if isinstance(value.get("model_calls"), list) else []
    for entry in raw_model_calls[:32]:
        normalized_call = _normalize_message_usage_call(entry)
        if normalized_call:
            model_calls.append(normalized_call)
    if model_calls:
        cleaned["model_calls"] = model_calls

    model_call_count = _coerce_non_negative_int(value.get("model_call_count"))
    if model_call_count is None and model_calls:
        model_call_count = len(model_calls)
    elif model_call_count is not None and model_calls:
        model_call_count = max(model_call_count, len(model_calls))
    if model_call_count is not None:
        cleaned["model_call_count"] = model_call_count

    cost = value.get("cost")
    if isinstance(cost, (int, float)) and not isinstance(cost, bool) and cost >= 0:
        cleaned["cost"] = round(float(cost), 6)

    currency = str(value.get("currency") or "").strip()[:16]
    if currency:
        cleaned["currency"] = currency

    model = str(value.get("model") or "").strip()[:80]
    if model:
        cleaned["model"] = model

    if cleaned.get("prompt_tokens") is not None:
        cleaned["estimated_input_tokens"] = cleaned["prompt_tokens"]
    elif normalized_breakdown:
        cleaned["estimated_input_tokens"] = sum(normalized_breakdown.values())

    return cleaned or None


def _normalize_message_tool_result(entry: dict) -> dict | None:
    if not isinstance(entry, dict):
        return None

    tool_name = str(entry.get("tool_name") or "").strip()[:80]
    content = str(entry.get("content") or "").strip()[:RAG_TOOL_RESULT_MAX_TEXT_CHARS]
    if not tool_name or not content:
        return None

    cleaned = {
        "tool_name": tool_name,
        "content": content,
    }
    summary = str(entry.get("summary") or "").strip()[:RAG_TOOL_RESULT_SUMMARY_MAX_CHARS]
    if summary:
        cleaned["summary"] = summary
    input_preview = str(entry.get("input_preview") or "").strip()[:300]
    if input_preview:
        cleaned["input_preview"] = input_preview

    raw_content = str(entry.get("raw_content") or "").strip()[:FETCH_RAW_TOOL_RESULT_MAX_TEXT_CHARS]
    if raw_content:
        cleaned["raw_content"] = raw_content

    content_mode = str(entry.get("content_mode") or "").strip()[:80]
    if content_mode:
        cleaned["content_mode"] = content_mode

    summary_notice = str(entry.get("summary_notice") or "").strip()[:300]
    if summary_notice:
        cleaned["summary_notice"] = summary_notice

    if isinstance(entry.get("cleanup_applied"), bool):
        cleaned["cleanup_applied"] = entry["cleanup_applied"]

    token_estimate = _coerce_non_negative_int(entry.get("content_token_estimate"))
    if token_estimate is not None:
        cleaned["content_token_estimate"] = token_estimate
    return cleaned


def _normalize_message_tool_trace_entry(entry: dict) -> dict | None:
    if not isinstance(entry, dict):
        return None

    tool_name = str(entry.get("tool_name") or entry.get("tool") or "").strip()[:80]
    if not tool_name:
        return None

    state = str(entry.get("state") or "").strip().lower()
    if state not in MESSAGE_TOOL_TRACE_STATES:
        state = "done"

    cleaned = {
        "tool_name": tool_name,
        "state": state,
    }

    step = _coerce_non_negative_int(entry.get("step"))
    if step is not None:
        cleaned["step"] = max(1, step)

    preview = str(entry.get("preview") or "").strip()[:300]
    if preview:
        cleaned["preview"] = preview

    summary = str(entry.get("summary") or "").strip()[:RAG_TOOL_RESULT_SUMMARY_MAX_CHARS]
    if summary:
        cleaned["summary"] = summary

    if isinstance(entry.get("cached"), bool):
        cleaned["cached"] = entry["cached"]

    return cleaned


def extract_message_tool_results(metadata: dict | None) -> list[dict]:
    source = metadata if isinstance(metadata, dict) else {}
    raw_results = source.get("tool_results")
    if not isinstance(raw_results, list):
        return []

    normalized = []
    for entry in raw_results:
        cleaned = _normalize_message_tool_result(entry)
        if cleaned and cleaned not in normalized:
            normalized.append(cleaned)
    return normalized


def extract_message_tool_trace(metadata: dict | None) -> list[dict]:
    source = metadata if isinstance(metadata, dict) else {}
    raw_trace = source.get("tool_trace")
    if not isinstance(raw_trace, list):
        return []

    normalized = []
    for entry in raw_trace[:64]:
        cleaned = _normalize_message_tool_trace_entry(entry)
        if cleaned:
            normalized.append(cleaned)
    return normalized


def extract_message_usage(
    metadata: dict | None,
    prompt_tokens=None,
    completion_tokens=None,
    total_tokens=None,
) -> dict | None:
    source = metadata if isinstance(metadata, dict) else {}
    usage = _normalize_message_usage(source.get("usage")) or {}

    fallback_prompt = _coerce_non_negative_int(prompt_tokens)
    fallback_completion = _coerce_non_negative_int(completion_tokens)
    fallback_total = _coerce_non_negative_int(total_tokens)

    if fallback_prompt is not None and "prompt_tokens" not in usage:
        usage["prompt_tokens"] = fallback_prompt
    if fallback_completion is not None and "completion_tokens" not in usage:
        usage["completion_tokens"] = fallback_completion
    if fallback_total is not None and "total_tokens" not in usage:
        usage["total_tokens"] = fallback_total

    target_total = usage.get("prompt_tokens")
    normalized_breakdown = _normalize_usage_breakdown(
        usage.get("input_breakdown"),
        target_total=target_total or usage.get("estimated_input_tokens"),
    )
    if normalized_breakdown:
        usage["input_breakdown"] = normalized_breakdown
    if target_total is not None:
        usage["estimated_input_tokens"] = target_total
    elif normalized_breakdown:
        usage["estimated_input_tokens"] = sum(normalized_breakdown.values())

    return usage or None


def _normalize_clarification_question_payload(value) -> dict | None:
    if not isinstance(value, dict):
        return None

    question_id = str(value.get("id") or "").strip()[:80]
    label = str(value.get("label") or "").strip()[:300]
    input_type = str(value.get("input_type") or "").strip()
    if not question_id or not label or input_type not in {"text", "single_select", "multi_select"}:
        return None

    cleaned = {
        "id": question_id,
        "label": label,
        "input_type": input_type,
    }
    if value.get("required") is False:
        cleaned["required"] = False

    placeholder = str(value.get("placeholder") or "").strip()[:200]
    if placeholder:
        cleaned["placeholder"] = placeholder

    if value.get("allow_free_text") is True:
        cleaned["allow_free_text"] = True

    raw_options = value.get("options") if isinstance(value.get("options"), list) else []
    normalized_options = []
    for option in raw_options[:12]:
        if not isinstance(option, dict):
            continue
        option_label = str(option.get("label") or "").strip()[:120]
        option_value = str(option.get("value") or "").strip()[:120]
        if not option_label or not option_value:
            continue
        normalized_option = {
            "label": option_label,
            "value": option_value,
        }
        description = str(option.get("description") or "").strip()[:200]
        if description:
            normalized_option["description"] = description
        normalized_options.append(normalized_option)
    if normalized_options:
        cleaned["options"] = normalized_options

    return cleaned


def extract_pending_clarification(metadata: dict | None) -> dict | None:
    source = metadata if isinstance(metadata, dict) else {}
    pending = source.get("pending_clarification")
    if not isinstance(pending, dict):
        return None

    questions = pending.get("questions") if isinstance(pending.get("questions"), list) else []
    normalized_questions = []
    for question in questions[:5]:
        normalized_question = _normalize_clarification_question_payload(question)
        if normalized_question is not None:
            normalized_questions.append(normalized_question)
    if not normalized_questions:
        return None

    cleaned = {"questions": normalized_questions}
    intro = str(pending.get("intro") or "").strip()[:300]
    if intro:
        cleaned["intro"] = intro
    submit_label = str(pending.get("submit_label") or "").strip()[:80]
    if submit_label:
        cleaned["submit_label"] = submit_label
    return cleaned


def extract_clarification_response(metadata: dict | None) -> dict | None:
    source = metadata if isinstance(metadata, dict) else {}
    response = source.get("clarification_response")
    if not isinstance(response, dict):
        return None

    cleaned = {}
    assistant_message_id = _coerce_non_negative_int(response.get("assistant_message_id"))
    if assistant_message_id is not None:
        cleaned["assistant_message_id"] = assistant_message_id

    answers = response.get("answers") if isinstance(response.get("answers"), dict) else {}
    normalized_answers = {}
    for key, value in list(answers.items())[:10]:
        key_text = str(key or "").strip()[:80]
        if not key_text or not isinstance(value, dict):
            continue
        display = str(value.get("display") or "").strip()[:500]
        if not display:
            continue
        normalized_answers[key_text] = {"display": display}
    if normalized_answers:
        cleaned["answers"] = normalized_answers

    return cleaned or None


def serialize_message_metadata(metadata: dict | None) -> str | None:
    metadata = metadata if isinstance(metadata, dict) else {}
    cleaned = {}

    attachments = extract_message_attachments(metadata)
    primary_image = next((entry for entry in attachments if entry.get("kind") == "image"), None)
    primary_document = next((entry for entry in attachments if entry.get("kind") == "document"), None)

    ocr_text = (metadata.get("ocr_text") or (primary_image or {}).get("ocr_text") or "").strip()
    vision_summary = (metadata.get("vision_summary") or (primary_image or {}).get("vision_summary") or "").strip()
    assistant_guidance = (
        metadata.get("assistant_guidance") or (primary_image or {}).get("assistant_guidance") or ""
    ).strip()
    image_name = (metadata.get("image_name") or (primary_image or {}).get("image_name") or "").strip()
    image_mime_type = (metadata.get("image_mime_type") or (primary_image or {}).get("image_mime_type") or "").strip()
    image_id = (metadata.get("image_id") or (primary_image or {}).get("image_id") or "").strip()
    key_points = metadata.get("key_points") if isinstance(metadata.get("key_points"), list) else (primary_image or {}).get("key_points")
    reasoning_content = (metadata.get("reasoning_content") or "").strip()
    summary_source = (metadata.get("summary_source") or "").strip()
    generated_at = (metadata.get("generated_at") or "").strip()

    if attachments:
        cleaned["attachments"] = attachments

    if ocr_text:
        cleaned["ocr_text"] = ocr_text
    if vision_summary:
        cleaned["vision_summary"] = vision_summary
    if assistant_guidance:
        cleaned["assistant_guidance"] = assistant_guidance
    if image_name:
        cleaned["image_name"] = image_name[:255]
    if image_mime_type:
        cleaned["image_mime_type"] = image_mime_type
    if image_id:
        cleaned["image_id"] = image_id[:64]
    if isinstance(key_points, list):
        normalized_points = []
        for point in key_points:
            point_text = str(point or "").strip()
            if point_text and point_text not in normalized_points:
                normalized_points.append(point_text[:300])
        if normalized_points:
            cleaned["key_points"] = normalized_points[:8]
    if reasoning_content:
        cleaned["reasoning_content"] = reasoning_content[:CONTENT_MAX_CHARS]
    if metadata.get("is_pruned") is True:
        cleaned["is_pruned"] = True
    pruned_original = str(metadata.get("pruned_original") or "").strip()
    if pruned_original:
        cleaned["pruned_original"] = pruned_original[:CONTENT_MAX_CHARS]
    if metadata.get("is_summary") is True:
        cleaned["is_summary"] = True
    if summary_source:
        cleaned["summary_source"] = summary_source[:120]
    if generated_at:
        cleaned["generated_at"] = generated_at[:80]

    for key in (
        "covers_from_position",
        "covers_to_position",
        "summary_position",
        "covered_message_count",
        "covered_tool_call_message_count",
        "covered_tool_message_count",
        "trigger_threshold",
        "trigger_token_count",
        "visible_token_count",
        "summary_source_token_target",
    ):
        normalized = _coerce_non_negative_int(metadata.get(key))
        if normalized is not None:
            cleaned[key] = normalized

    summary_mode = str(metadata.get("summary_mode") or "").strip().lower()
    if summary_mode in CHAT_SUMMARY_ALLOWED_MODES:
        cleaned["summary_mode"] = summary_mode

    summary_model = str(metadata.get("summary_model") or "").strip()
    if summary_model:
        cleaned["summary_model"] = summary_model[:120]

    summary_format = str(metadata.get("summary_format") or "").strip().lower()
    if summary_format in {"plain_text", "structured_json"}:
        cleaned["summary_format"] = summary_format

    summary_level = _coerce_non_negative_int(metadata.get("summary_level"))
    if summary_level is not None:
        cleaned["summary_level"] = max(1, summary_level)

    summary_data = metadata.get("summary_data") if isinstance(metadata.get("summary_data"), dict) else None
    if summary_data:
        normalized_summary_data = {}
        for key in ("facts", "decisions", "open_issues", "entities", "tool_outcomes"):
            raw_items = summary_data.get(key) if isinstance(summary_data.get(key), list) else []
            cleaned_items = []
            for raw_item in raw_items[:16]:
                item_text = str(raw_item or "").strip()
                if item_text and item_text not in cleaned_items:
                    cleaned_items.append(item_text[:500])
            if cleaned_items:
                normalized_summary_data[key] = cleaned_items
        if normalized_summary_data:
            cleaned["summary_data"] = normalized_summary_data

    summary_insert_strategy = str(metadata.get("summary_insert_strategy") or "").strip()
    if summary_insert_strategy in {
        "after_covered_block",
        "replace_first_covered_message",
        "replace_first_covered_message_preserve_positions",
    }:
        cleaned["summary_insert_strategy"] = summary_insert_strategy

    covered_message_ids = metadata.get("covered_message_ids")
    if isinstance(covered_message_ids, list):
        normalized_ids = []
        for raw_value in covered_message_ids[:64]:
            normalized = _coerce_non_negative_int(raw_value)
            if normalized is not None and normalized not in normalized_ids:
                normalized_ids.append(normalized)
        if normalized_ids:
            cleaned["covered_message_ids"] = normalized_ids

    for key in (
        "covered_visible_message_ids",
        "covered_tool_call_message_ids",
        "covered_tool_message_ids",
    ):
        raw_ids = metadata.get(key)
        if not isinstance(raw_ids, list):
            continue
        normalized_ids = []
        for raw_value in raw_ids[:64]:
            normalized = _coerce_non_negative_int(raw_value)
            if normalized is not None and normalized not in normalized_ids:
                normalized_ids.append(normalized)
        if normalized_ids:
            cleaned[key] = normalized_ids

    tool_results = extract_message_tool_results(metadata)
    if tool_results:
        cleaned["tool_results"] = tool_results

    tool_trace = extract_message_tool_trace(metadata)
    if tool_trace:
        cleaned["tool_trace"] = tool_trace

    usage = extract_message_usage(metadata)
    if usage:
        cleaned["usage"] = usage

    pending_clarification = extract_pending_clarification(metadata)
    if pending_clarification:
        cleaned["pending_clarification"] = pending_clarification

    clarification_response = extract_clarification_response(metadata)
    if clarification_response:
        cleaned["clarification_response"] = clarification_response

    canvas_documents = extract_canvas_documents(metadata)
    if canvas_documents or metadata.get("canvas_cleared") is True:
        cleaned["canvas_documents"] = canvas_documents
    active_document_id = extract_canvas_active_document_id(metadata, canvas_documents)
    if active_document_id:
        cleaned["active_document_id"] = active_document_id
    if metadata.get("canvas_cleared") is True:
        cleaned["canvas_cleared"] = True

    project_workflow = metadata.get("project_workflow") if isinstance(metadata.get("project_workflow"), dict) else None
    if project_workflow:
        cleaned_workflow = {}
        for key, max_length in (("project_name", 120), ("goal", 300), ("target_type", 40), ("stage", 40)):
            value = str(project_workflow.get(key) or "").strip()
            if value:
                cleaned_workflow[key] = value[:max_length]
        files = project_workflow.get("files") if isinstance(project_workflow.get("files"), list) else []
        cleaned_files = []
        for entry in files[:64]:
            if not isinstance(entry, dict):
                continue
            path = str(entry.get("path") or "").strip()[:240]
            if not path:
                continue
            cleaned_entry = {"path": path}
            for key, max_length in (("role", 24), ("purpose", 180), ("status", 40)):
                value = str(entry.get(key) or "").strip()
                if value:
                    cleaned_entry[key] = value[:max_length]
            cleaned_files.append(cleaned_entry)
        if cleaned_files:
            cleaned_workflow["files"] = cleaned_files
        for list_key in ("dependencies", "open_issues"):
            values = project_workflow.get(list_key) if isinstance(project_workflow.get(list_key), list) else []
            normalized_values = []
            for value in values[:24]:
                item = str(value or "").strip()
                if item and item not in normalized_values:
                    normalized_values.append(item[:200])
            if normalized_values:
                cleaned_workflow[list_key] = normalized_values
        validation = project_workflow.get("validation") if isinstance(project_workflow.get("validation"), dict) else None
        if validation:
            cleaned_validation = {}
            status = str(validation.get("status") or "").strip()[:40]
            if status:
                cleaned_validation["status"] = status
            for list_key in ("issues", "warnings"):
                values = validation.get(list_key) if isinstance(validation.get(list_key), list) else []
                normalized_values = []
                for value in values[:24]:
                    item = str(value or "").strip()
                    if item and item not in normalized_values:
                        normalized_values.append(item[:200])
                if normalized_values:
                    cleaned_validation[list_key] = normalized_values
            if cleaned_validation:
                cleaned_workflow["validation"] = cleaned_validation
        if cleaned_workflow:
            cleaned["project_workflow"] = cleaned_workflow

    file_id = (metadata.get("file_id") or (primary_document or {}).get("file_id") or "").strip()
    file_name = (metadata.get("file_name") or (primary_document or {}).get("file_name") or "").strip()
    file_mime_type = (metadata.get("file_mime_type") or (primary_document or {}).get("file_mime_type") or "").strip()
    file_context_block = (metadata.get("file_context_block") or "").strip()
    if not file_context_block and attachments:
        file_context_block = "\n\n".join(
            str(entry.get("file_context_block") or "").strip()
            for entry in attachments
            if entry.get("kind") == "document" and str(entry.get("file_context_block") or "").strip()
        ).strip()

    if file_id:
        cleaned["file_id"] = file_id[:64]
    if file_name:
        cleaned["file_name"] = file_name[:255]
    if file_mime_type:
        cleaned["file_mime_type"] = file_mime_type[:120]
    if metadata.get("file_text_truncated") is True or (primary_document or {}).get("file_text_truncated") is True:
        cleaned["file_text_truncated"] = True
    if file_context_block:
        cleaned["file_context_block"] = file_context_block[:CONTENT_MAX_CHARS]

    if not cleaned:
        return None
    return json.dumps(cleaned, ensure_ascii=False)


def message_row_to_dict(row) -> dict:
    row_keys = set(row.keys()) if hasattr(row, "keys") else set()
    metadata = parse_message_metadata(row["metadata"])
    usage = extract_message_usage(
        metadata,
        prompt_tokens=row["prompt_tokens"],
        completion_tokens=row["completion_tokens"],
        total_tokens=row["total_tokens"],
    )
    tool_calls = parse_message_tool_calls(row["tool_calls"]) if "tool_calls" in row_keys else []
    tool_call_id = str(row["tool_call_id"] or "").strip() if "tool_call_id" in row_keys else ""
    return {
        "id": row["id"],
        "position": row["position"] if "position" in row_keys else None,
        "role": row["role"],
        "content": row["content"],
        "metadata": metadata,
        "tool_calls": tool_calls,
        "tool_call_id": tool_call_id or None,
        "prompt_tokens": row["prompt_tokens"],
        "completion_tokens": row["completion_tokens"],
        "total_tokens": row["total_tokens"],
        "usage": usage,
        "deleted_at": row["deleted_at"] if "deleted_at" in row_keys else None,
    }


def get_app_settings() -> dict:
    settings = DEFAULT_SETTINGS.copy()
    with get_db() as conn:
        rows = conn.execute(
            "SELECT key, value FROM app_settings WHERE key IN ({})".format(", ".join("?" for _ in settings)),
            tuple(settings.keys()),
        ).fetchall()

    for row in rows:
        settings[row["key"]] = row["value"]

    return settings


def normalize_scratchpad_text(value) -> str:
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = []
    seen = set()
    for raw_line in text.split("\n"):
        line = " ".join(raw_line.strip().split())
        if not line or line in seen:
            continue
        seen.add(line)
        lines.append(line)

    normalized = "\n".join(lines)
    return normalized


def append_to_scratchpad(notes) -> tuple[dict, str]:
    """Append one or more notes. `notes` may be a string or a list of strings."""
    if isinstance(notes, str):
        note_list = [notes]
    else:
        note_list = list(notes or [])

    settings = get_app_settings()
    current = normalize_scratchpad_text(settings.get("scratchpad", ""))
    current_lines = current.splitlines() if current else []
    current_set = set(current_lines)

    appended = []
    skipped = []
    for raw in note_list:
        normalized_note = " ".join(str(raw or "").strip().split())
        if not normalized_note:
            continue
        if normalized_note in current_set:
            skipped.append(normalized_note)
        else:
            current_lines.append(normalized_note)
            current_set.add(normalized_note)
            appended.append(normalized_note)

    if not appended:
        if not skipped:
            return {"status": "rejected", "reason": "empty_notes"}, "Scratchpad notes are empty"
        return {
            "status": "skipped",
            "reason": "duplicate_notes",
            "notes": skipped,
            "scratchpad": current,
        }, "Scratchpad notes already exist"

    next_value = normalize_scratchpad_text("\n".join(current_lines))
    settings["scratchpad"] = next_value
    save_app_settings(settings)
    return {
        "status": "appended",
        "notes": appended,
        "skipped": skipped,
        "scratchpad": next_value,
    }, "Scratchpad updated"


def replace_scratchpad(new_content) -> tuple[dict, str]:
    normalized_content = normalize_scratchpad_text(new_content)
    
    settings = get_app_settings()
    settings["scratchpad"] = normalized_content
    save_app_settings(settings)
    return {
        "status": "replaced",
        "scratchpad": normalized_content,
    }, "Scratchpad content replaced successfully"


def save_app_settings(settings: dict) -> None:
    with get_db() as conn:
        for key, value in settings.items():
            if key == "scratchpad":
                value = normalize_scratchpad_text(value)
            conn.execute(
                """INSERT INTO app_settings (key, value, updated_at)
                   VALUES (?, ?, datetime('now'))
                   ON CONFLICT(key) DO UPDATE SET
                       value = excluded.value,
                       updated_at = datetime('now')""",
                (key, value),
            )


def cache_get(key: str):
    with get_db() as conn:
        row = conn.execute(
            "SELECT value FROM web_cache WHERE key = ? AND cached_at > datetime('now', ?)",
            (key, f"-{CACHE_TTL_HOURS} hours"),
        ).fetchone()
    if row:
        try:
            return json.loads(row["value"])
        except Exception:
            return None
    return None


def cache_set(key: str, value) -> None:
    with get_db() as conn:
        conn.execute(
            """INSERT INTO web_cache (key, value, cached_at)
               VALUES (?, ?, datetime('now'))
               ON CONFLICT(key) DO UPDATE SET
                   value     = excluded.value,
                   cached_at = excluded.cached_at""",
            (key, json.dumps(value, ensure_ascii=False)),
        )


def normalize_active_tool_names(raw_value) -> list[str]:
    if isinstance(raw_value, list):
        names = raw_value
    else:
        try:
            names = json.loads(raw_value or "[]")
        except Exception:
            names = []

    normalized = []
    allowed = set(DEFAULT_ACTIVE_TOOL_NAMES)
    for name in names:
        if isinstance(name, str) and name in allowed and name not in normalized:
            normalized.append(name)
    return normalized


def _ensure_tool(name: str, names: list[str]) -> list[str]:
    if name in names:
        return names
    return [*names, name]


def get_active_tool_names(settings: dict | None = None) -> list[str]:
    source = settings if settings is not None else get_app_settings()
    names = normalize_active_tool_names(source.get("active_tools"))
    if not RAG_ENABLED:
        names = [name for name in names if name != "search_knowledge_base"]
    if names:
        if "append_scratchpad" in names:
            names = _ensure_tool("replace_scratchpad", names)
        return names
    if source.get("active_tools") is None:
        names = normalize_active_tool_names(DEFAULT_SETTINGS["active_tools"])
        if not RAG_ENABLED:
            names = [name for name in names if name != "search_knowledge_base"]
        if "append_scratchpad" in names:
            names = _ensure_tool("replace_scratchpad", names)
        return names
    return []


def get_rag_auto_inject_enabled(settings: dict | None = None) -> bool:
    if not RAG_ENABLED:
        return False
    source = settings if settings is not None else get_app_settings()
    raw_value = source.get("rag_auto_inject", DEFAULT_SETTINGS["rag_auto_inject"])
    return str(raw_value).strip().lower() in {"1", "true", "yes", "on"}


def get_rag_sensitivity(settings: dict | None = None) -> str:
    source = settings if settings is not None else get_app_settings()
    raw_value = str(source.get("rag_sensitivity", DEFAULT_SETTINGS["rag_sensitivity"]) or "").strip().lower()
    if raw_value in RAG_SENSITIVITY_PRESETS:
        return raw_value
    return RAG_DEFAULT_SENSITIVITY_PRESET


def get_rag_context_size(settings: dict | None = None) -> str:
    source = settings if settings is not None else get_app_settings()
    raw_value = str(source.get("rag_context_size", DEFAULT_SETTINGS["rag_context_size"]) or "").strip().lower()
    if raw_value in RAG_CONTEXT_SIZE_PRESETS:
        return raw_value
    return RAG_DEFAULT_CONTEXT_SIZE_PRESET


def get_rag_auto_inject_top_k(settings: dict | None = None) -> int:
    return int(RAG_CONTEXT_SIZE_PRESETS[get_rag_context_size(settings)])


def get_tool_memory_auto_inject_enabled(settings: dict | None = None) -> bool:
    if not RAG_ENABLED:
        return False
    source = settings if settings is not None else get_app_settings()
    raw_value = source.get("tool_memory_auto_inject", DEFAULT_SETTINGS["tool_memory_auto_inject"])
    return str(raw_value).strip().lower() in {"1", "true", "yes", "on"}


def get_chat_summary_mode(settings: dict | None = None) -> str:
    source = settings if settings is not None else get_app_settings()
    raw_value = str(source.get("chat_summary_mode", DEFAULT_SETTINGS["chat_summary_mode"]) or "").strip().lower()
    if raw_value in CHAT_SUMMARY_ALLOWED_MODES:
        return raw_value
    fallback = CHAT_SUMMARY_MODE if CHAT_SUMMARY_MODE in CHAT_SUMMARY_ALLOWED_MODES else "auto"
    return fallback


def get_chat_summary_trigger_token_count(settings: dict | None = None) -> int:
    source = settings if settings is not None else get_app_settings()
    raw_value = source.get("chat_summary_trigger_token_count")
    if raw_value in (None, ""):
        raw_value = source.get(
            "chat_summary_trigger_message_count",
            DEFAULT_SETTINGS["chat_summary_trigger_token_count"],
        )
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        value = CHAT_SUMMARY_TRIGGER_TOKEN_COUNT
    return max(1_000, min(200_000, value))


def get_summary_skip_first(settings: dict | None = None) -> int:
    source = settings if settings is not None else get_app_settings()
    raw_value = source.get("summary_skip_first", DEFAULT_SETTINGS["summary_skip_first"])
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        value = 2
    return max(0, min(20, value))


def get_summary_skip_last(settings: dict | None = None) -> int:
    source = settings if settings is not None else get_app_settings()
    raw_value = source.get("summary_skip_last", DEFAULT_SETTINGS["summary_skip_last"])
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        value = 1
    return max(0, min(20, value))


def get_canvas_prompt_max_lines(settings: dict | None = None) -> int:
    source = settings if settings is not None else get_app_settings()
    raw_value = source.get("canvas_prompt_max_lines", DEFAULT_SETTINGS["canvas_prompt_max_lines"])
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        value = CANVAS_PROMPT_DEFAULT_MAX_LINES
    return max(100, min(3_000, value))


def get_canvas_expand_max_lines(settings: dict | None = None) -> int:
    source = settings if settings is not None else get_app_settings()
    raw_value = source.get("canvas_expand_max_lines", DEFAULT_SETTINGS["canvas_expand_max_lines"])
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        value = CANVAS_EXPAND_DEFAULT_MAX_LINES
    return max(100, min(4_000, value))


def get_canvas_scroll_window_lines(settings: dict | None = None) -> int:
    source = settings if settings is not None else get_app_settings()
    raw_value = source.get("canvas_scroll_window_lines", DEFAULT_SETTINGS["canvas_scroll_window_lines"])
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        value = CANVAS_SCROLL_WINDOW_LINES
    return max(50, min(800, value))


def get_prompt_max_input_tokens(settings: dict | None = None) -> int:
    del settings
    return PROMPT_MAX_INPUT_TOKENS


def get_prompt_response_token_reserve(settings: dict | None = None) -> int:
    del settings
    return PROMPT_RESPONSE_TOKEN_RESERVE


def get_prompt_recent_history_max_tokens(settings: dict | None = None) -> int:
    del settings
    return PROMPT_RECENT_HISTORY_MAX_TOKENS


def get_prompt_summary_max_tokens(settings: dict | None = None) -> int:
    del settings
    return PROMPT_SUMMARY_MAX_TOKENS


def get_prompt_rag_max_tokens(settings: dict | None = None) -> int:
    del settings
    return PROMPT_RAG_MAX_TOKENS


def get_prompt_tool_memory_max_tokens(settings: dict | None = None) -> int:
    del settings
    return PROMPT_TOOL_MEMORY_MAX_TOKENS


def get_prompt_preflight_summary_token_count(settings: dict | None = None) -> int:
    del settings
    return PROMPT_PREFLIGHT_SUMMARY_TOKEN_COUNT


def get_summary_source_target_tokens(settings: dict | None = None) -> int:
    del settings
    return SUMMARY_SOURCE_TARGET_TOKENS


def get_summary_retry_min_source_tokens(settings: dict | None = None) -> int:
    del settings
    return SUMMARY_RETRY_MIN_SOURCE_TOKENS


def get_fetch_url_token_threshold(settings: dict | None = None) -> int:
    source = settings if settings is not None else get_app_settings()
    raw_value = source.get("fetch_url_token_threshold", DEFAULT_SETTINGS["fetch_url_token_threshold"])
    try:
        threshold = int(raw_value)
    except (TypeError, ValueError):
        threshold = FETCH_SUMMARY_TOKEN_THRESHOLD
    return max(400, min(20_000, threshold))


def get_fetch_url_clip_aggressiveness(settings: dict | None = None) -> int:
    source = settings if settings is not None else get_app_settings()
    raw_value = source.get("fetch_url_clip_aggressiveness", DEFAULT_SETTINGS["fetch_url_clip_aggressiveness"])
    try:
        aggressiveness = int(raw_value)
    except (TypeError, ValueError):
        aggressiveness = 50
    return max(0, min(100, aggressiveness))


def get_pruning_enabled(settings: dict | None = None) -> bool:
    source = settings if settings is not None else get_app_settings()
    raw_value = source.get("pruning_enabled", DEFAULT_SETTINGS["pruning_enabled"])
    return str(raw_value).strip().lower() in {"1", "true", "yes", "on"}


def get_pruning_token_threshold(settings: dict | None = None) -> int:
    source = settings if settings is not None else get_app_settings()
    raw_value = source.get("pruning_token_threshold", DEFAULT_SETTINGS["pruning_token_threshold"])
    try:
        threshold = int(raw_value)
    except (TypeError, ValueError):
        threshold = CHAT_SUMMARY_TRIGGER_TOKEN_COUNT
    return max(1_000, min(200_000, threshold))


def get_pruning_batch_size(settings: dict | None = None) -> int:
    source = settings if settings is not None else get_app_settings()
    raw_value = source.get("pruning_batch_size", DEFAULT_SETTINGS["pruning_batch_size"])
    try:
        batch_size = int(raw_value)
    except (TypeError, ValueError):
        batch_size = 10
    return max(1, min(50, batch_size))


def get_next_message_position(conn: sqlite3.Connection, conversation_id: int) -> int:
    row = conn.execute(
        "SELECT COALESCE(MAX(position), 0) AS max_position FROM messages WHERE conversation_id = ?",
        (conversation_id,),
    ).fetchone()
    max_position = row["max_position"] if row else 0
    return int(max_position or 0) + 1


def insert_message(
    conn: sqlite3.Connection,
    conversation_id: int,
    role: str,
    content: str,
    metadata: str | None = None,
    tool_calls: str | None = None,
    tool_call_id: str | None = None,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    total_tokens: int | None = None,
    position: int | None = None,
) -> int:
    normalized_role = str(role or "").strip()
    normalized_content = content if isinstance(content, str) else str(content or "")
    cursor = conn.execute(
        """INSERT INTO messages (
               conversation_id, position, role, content, metadata, tool_calls, tool_call_id,
               prompt_tokens, completion_tokens, total_tokens
           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            conversation_id,
            position,
            normalized_role,
            normalized_content,
            metadata,
            tool_calls,
            tool_call_id,
            prompt_tokens,
            completion_tokens,
            total_tokens,
        ),
    )
    return int(cursor.lastrowid)


def get_conversation_message_rows(
    conn: sqlite3.Connection,
    conversation_id: int,
    include_deleted: bool = False,
) -> list[sqlite3.Row]:
    query = (
        """SELECT id, position, role, content, metadata, tool_calls, tool_call_id,
                  prompt_tokens, completion_tokens, total_tokens, deleted_at
           FROM messages
           WHERE conversation_id = ?"""
    )
    params: list[object] = [conversation_id]
    if not include_deleted:
        query += " AND deleted_at IS NULL"
    query += " ORDER BY position, id"
    return conn.execute(query, tuple(params)).fetchall()


def get_conversation_messages(conversation_id: int, include_deleted: bool = False) -> list[dict]:
    with get_db() as conn:
        rows = get_conversation_message_rows(conn, conversation_id, include_deleted=include_deleted)
    return [message_row_to_dict(row) for row in rows]


def soft_delete_messages(
    conn: sqlite3.Connection,
    conversation_id: int,
    message_ids: Iterable[int],
    deleted_at: str,
) -> None:
    normalized_ids = [int(message_id) for message_id in message_ids if int(message_id) > 0]
    if not normalized_ids:
        return
    placeholders = ", ".join("?" for _ in normalized_ids)
    conn.execute(
        f"UPDATE messages SET deleted_at = ? WHERE conversation_id = ? AND id IN ({placeholders}) AND deleted_at IS NULL",
        (deleted_at, conversation_id, *normalized_ids),
    )


def restore_soft_deleted_messages(
    conn: sqlite3.Connection,
    conversation_id: int,
    message_ids: Iterable[int],
) -> None:
    normalized_ids = [int(message_id) for message_id in message_ids if int(message_id) > 0]
    if not normalized_ids:
        return
    placeholders = ", ".join("?" for _ in normalized_ids)
    conn.execute(
        f"UPDATE messages SET deleted_at = NULL WHERE conversation_id = ? AND id IN ({placeholders}) AND deleted_at IS NOT NULL",
        (conversation_id, *normalized_ids),
    )


def shift_message_positions(
    conn: sqlite3.Connection,
    conversation_id: int,
    start_position: int,
    delta: int,
    exclude_message_ids: Iterable[int] | None = None,
) -> None:
    if delta == 0:
        return
    normalized_start = int(start_position or 0)
    excluded_ids = [int(message_id) for message_id in (exclude_message_ids or []) if int(message_id) > 0]
    query = "UPDATE messages SET position = position + ? WHERE conversation_id = ? AND position >= ?"
    params: list[object] = [delta, conversation_id, normalized_start]
    if excluded_ids:
        placeholders = ", ".join("?" for _ in excluded_ids)
        query += f" AND id NOT IN ({placeholders})"
        params.extend(excluded_ids)
    conn.execute(query, tuple(params))


def is_renderable_chat_message(message: dict) -> bool:
    if not isinstance(message, dict):
        return False
    role = str(message.get("role") or "").strip()
    if role == "assistant":
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls:
            return False
    return role in VISIBLE_CHAT_ROLES


def count_visible_message_tokens(messages: list[dict]) -> int:
    total = 0
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "").strip()
        if role == "assistant":
            tool_calls = message.get("tool_calls")
            if isinstance(tool_calls, list) and tool_calls:
                continue
        if role not in {"user", "assistant", "tool", "summary"}:
            continue
        total += estimate_text_tokens(str(message.get("content") or ""))
    return total


def get_unsummarized_visible_messages(
    messages: list[dict],
    limit: int | None = None,
    skip_first: int = 0,
    skip_last: int = 0,
) -> list[dict]:
    ordered_messages = sorted(
        (message for message in messages if isinstance(message, dict)),
        key=lambda message: (int(message.get("position") or 0), int(message.get("id") or 0)),
    )
    candidates = []
    for message in ordered_messages:
        if not is_renderable_chat_message(message):
            continue
        role = str(message.get("role") or "").strip()
        if role == "summary":
            continue
        candidates.append(message)

    skip_first = max(0, skip_first)
    skip_last = max(0, skip_last)
    if skip_first + skip_last >= len(candidates):
        return []
    eligible = candidates[skip_first:len(candidates) - skip_last] if skip_last > 0 else candidates[skip_first:]

    if limit is not None:
        eligible = eligible[:limit]
    return eligible


def find_summary_covering_message_id(conversation_id: int, message_id: int) -> dict | None:
    target_id = _coerce_non_negative_int(message_id)
    if target_id is None:
        return None
    for message in get_conversation_messages(conversation_id):
        metadata = message.get("metadata") if isinstance(message.get("metadata"), dict) else {}
        covered_ids = metadata.get("covered_message_ids") if isinstance(metadata.get("covered_message_ids"), list) else []
        if target_id in covered_ids:
            return message
    return None
