from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Iterable
from uuid import uuid4

from flask import current_app, has_app_context

from canvas_service import extract_canvas_documents
from config import (
    CACHE_TTL_HOURS,
    CHAT_SUMMARY_BATCH_SIZE,
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
    MAX_SCRATCHPAD_LENGTH,
    RAG_CONTEXT_SIZE_PRESETS,
    RAG_DEFAULT_CONTEXT_SIZE_PRESET,
    RAG_DEFAULT_SENSITIVITY_PRESET,
    RAG_ENABLED,
    RAG_SENSITIVITY_PRESETS,
    RAG_TOOL_RESULT_MAX_TEXT_CHARS,
    RAG_TOOL_RESULT_SUMMARY_MAX_CHARS,
)
from token_utils import estimate_text_tokens

_db_path = DB_PATH
MESSAGE_USAGE_BREAKDOWN_KEYS = (
    "system_prompt",
    "user_messages",
    "assistant_history",
    "tool_results",
    "rag_context",
    "final_instruction",
)
MESSAGE_TOOL_TRACE_STATES = {"running", "done", "error"}
VISIBLE_CHAT_ROLES = {"user", "assistant", "summary"}


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


def initialize_database() -> None:
    init_db()
    ensure_messages_metadata_column()
    ensure_messages_tool_history_columns()
    ensure_messages_position_column()
    ensure_messages_deleted_at_column()


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


def _normalize_message_usage(value: dict | None) -> dict | None:
    if not isinstance(value, dict):
        return None

    cleaned = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens", "estimated_input_tokens"):
        normalized = _coerce_non_negative_int(value.get(key))
        if normalized is not None:
            cleaned[key] = normalized

    breakdown = value.get("input_breakdown")
    if isinstance(breakdown, dict):
        normalized_breakdown = {}
        for key in MESSAGE_USAGE_BREAKDOWN_KEYS:
            normalized = _coerce_non_negative_int(breakdown.get(key))
            if normalized is not None:
                normalized_breakdown[key] = normalized
        if normalized_breakdown:
            cleaned["input_breakdown"] = normalized_breakdown

    cost = value.get("cost")
    if isinstance(cost, (int, float)) and not isinstance(cost, bool) and cost >= 0:
        cleaned["cost"] = round(float(cost), 6)

    currency = str(value.get("currency") or "").strip()[:16]
    if currency:
        cleaned["currency"] = currency

    model = str(value.get("model") or "").strip()[:80]
    if model:
        cleaned["model"] = model

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

    ocr_text = (metadata.get("ocr_text") or "").strip()
    vision_summary = (metadata.get("vision_summary") or "").strip()
    assistant_guidance = (metadata.get("assistant_guidance") or "").strip()
    image_name = (metadata.get("image_name") or "").strip()
    image_mime_type = (metadata.get("image_mime_type") or "").strip()
    image_id = (metadata.get("image_id") or "").strip()
    key_points = metadata.get("key_points")
    reasoning_content = (metadata.get("reasoning_content") or "").strip()
    summary_source = (metadata.get("summary_source") or "").strip()
    generated_at = (metadata.get("generated_at") or "").strip()

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
    if metadata.get("is_summary") is True:
        cleaned["is_summary"] = True
    if summary_source:
        cleaned["summary_source"] = summary_source[:120]
    if generated_at:
        cleaned["generated_at"] = generated_at[:80]

    for key in (
        "covers_from_position",
        "covers_to_position",
        "covered_message_count",
        "trigger_threshold",
        "trigger_token_count",
        "visible_token_count",
        "summary_batch_size",
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

    covered_message_ids = metadata.get("covered_message_ids")
    if isinstance(covered_message_ids, list):
        normalized_ids = []
        for raw_value in covered_message_ids[:64]:
            normalized = _coerce_non_negative_int(raw_value)
            if normalized is not None and normalized not in normalized_ids:
                normalized_ids.append(normalized)
        if normalized_ids:
            cleaned["covered_message_ids"] = normalized_ids

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
    if metadata.get("canvas_cleared") is True:
        cleaned["canvas_cleared"] = True

    file_id = (metadata.get("file_id") or "").strip()
    file_name = (metadata.get("file_name") or "").strip()
    file_mime_type = (metadata.get("file_mime_type") or "").strip()
    file_context_block = (metadata.get("file_context_block") or "").strip()

    if file_id:
        cleaned["file_id"] = file_id[:64]
    if file_name:
        cleaned["file_name"] = file_name[:255]
    if file_mime_type:
        cleaned["file_mime_type"] = file_mime_type[:120]
    if metadata.get("file_text_truncated") is True:
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
    if len(normalized) > MAX_SCRATCHPAD_LENGTH:
        raise ValueError(f"Scratchpad cannot exceed {MAX_SCRATCHPAD_LENGTH} characters.")
    return normalized


def append_to_scratchpad(note) -> tuple[dict, str]:
    normalized_note = " ".join(str(note or "").strip().split())
    if not normalized_note:
        return {"status": "rejected", "reason": "empty_note"}, "Scratchpad note is empty"

    settings = get_app_settings()
    current = normalize_scratchpad_text(settings.get("scratchpad", ""))
    current_lines = current.splitlines() if current else []
    if normalized_note in current_lines:
        return {
            "status": "skipped",
            "reason": "duplicate_note",
            "note": normalized_note,
            "scratchpad": current,
        }, "Scratchpad note already exists"

    next_value = normalize_scratchpad_text("\n".join([*current_lines, normalized_note]))
    settings["scratchpad"] = next_value
    save_app_settings(settings)
    return {
        "status": "appended",
        "note": normalized_note,
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


def get_chat_summary_batch_size(settings: dict | None = None) -> int:
    source = settings if settings is not None else get_app_settings()
    raw_value = source.get("chat_summary_batch_size", DEFAULT_SETTINGS["chat_summary_batch_size"])
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        value = CHAT_SUMMARY_BATCH_SIZE
    return max(5, min(100, value))


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


def count_visible_message_tokens(messages: list[dict]) -> int:
    total = 0
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "").strip()
        if role not in VISIBLE_CHAT_ROLES or role == "summary":
            continue
        total += estimate_text_tokens(str(message.get("content") or ""))
    return total


def get_unsummarized_visible_messages(messages: list[dict], limit: int | None = None) -> list[dict]:
    selected = []
    ordered_messages = sorted(
        (message for message in messages if isinstance(message, dict)),
        key=lambda message: (int(message.get("position") or 0), int(message.get("id") or 0)),
    )
    for message in ordered_messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "").strip()
        if role not in VISIBLE_CHAT_ROLES:
            continue
        if role == "summary":
            continue
        selected.append(message)
        if limit is not None and len(selected) >= limit:
            break
    return selected


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
