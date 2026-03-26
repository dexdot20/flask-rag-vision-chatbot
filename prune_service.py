from __future__ import annotations

from typing import Any

from config import client
from db import get_conversation_message_rows, get_db, message_row_to_dict, parse_message_metadata, serialize_message_metadata

PRUNABLE_ROLES = {"user", "assistant"}
PRUNING_MODEL = "deepseek-chat"
PRUNING_SYSTEM_PROMPT = (
    "You rewrite a single chat message. Preserve the original language, core meaning, intent, and tone. "
    "Remove repetition, filler, indirect phrasing, and unnecessary detail. Return only the refined message text."
)


def _extract_response_text(response: Any) -> str:
    choices = getattr(response, "choices", None) or []
    if not choices:
        return ""
    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text") or ""))
        return "".join(parts).strip()
    return str(content or "").strip()


def _build_pruning_messages(content: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": PRUNING_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Mesajin ana fikrini ve butunlugunu koruyarak; gereksiz detaylari, tekrarleri ve dolayli anlatimlari "
                "sadeleştir. Sonucta mesajin daha net, rafine ve duzenlenmis bir versiyonunu olustur.\n\n"
                f"Mesaj:\n{content}"
            ),
        },
    ]


def _is_prunable_message(message: dict) -> bool:
    role = str(message.get("role") or "").strip()
    if role not in PRUNABLE_ROLES:
        return False
    if role == "assistant" and message.get("tool_calls"):
        return False
    metadata = message.get("metadata") if isinstance(message.get("metadata"), dict) else {}
    if metadata.get("is_summary") is True or metadata.get("is_pruned") is True:
        return False
    return bool(str(message.get("content") or "").strip())


def _load_message_row(message_id: int):
    with get_db() as conn:
        return conn.execute(
        """SELECT id, conversation_id, position, role, content, metadata, tool_calls, tool_call_id,
                  prompt_tokens, completion_tokens, total_tokens, deleted_at
           FROM messages
           WHERE id = ? AND deleted_at IS NULL""",
            (message_id,),
        ).fetchone()


def _persist_pruned_message(message_id: int, pruned_content: str, metadata: dict) -> dict:
    serialized_metadata = serialize_message_metadata(metadata)
    with get_db() as conn:
        conn.execute(
            "UPDATE messages SET content = ?, metadata = ? WHERE id = ?",
            (pruned_content, serialized_metadata, message_id),
        )
        updated_row = conn.execute(
            """SELECT id, position, role, content, metadata, tool_calls, tool_call_id,
                      prompt_tokens, completion_tokens, total_tokens, deleted_at
               FROM messages
               WHERE id = ?""",
            (message_id,),
        ).fetchone()
    return message_row_to_dict(updated_row)


def prune_message(message_id: int) -> dict:
    row = _load_message_row(message_id)
    if not row:
        raise ValueError("Message not found.")

    message = message_row_to_dict(row)
    if not _is_prunable_message(message):
        raise ValueError("Only visible user or assistant messages can be pruned.")

    original_content = str(message.get("content") or "")
    metadata = parse_message_metadata(message.get("metadata"))

    response = client.chat.completions.create(
        model=PRUNING_MODEL,
        messages=_build_pruning_messages(original_content),
    )
    pruned_content = _extract_response_text(response)
    if not pruned_content:
        raise ValueError("Pruning model returned empty content.")

    metadata["pruned_original"] = original_content
    metadata["is_pruned"] = True
    pruned_message = _persist_pruned_message(message_id, pruned_content, metadata)
    pruned_message["conversation_id"] = int(row["conversation_id"] or 0)
    return pruned_message


def prune_conversation_batch(conversation_id: int, batch_size: int) -> int:
    normalized_conversation_id = int(conversation_id or 0)
    normalized_batch_size = max(1, min(50, int(batch_size or 1)))
    if normalized_conversation_id <= 0:
        return 0

    with get_db() as conn:
        rows = get_conversation_message_rows(conn, normalized_conversation_id)
        messages = [message_row_to_dict(row) for row in rows]
        candidate_ids = [message["id"] for message in messages if _is_prunable_message(message)][:normalized_batch_size]

    pruned_count = 0
    for message_id in candidate_ids:
        prune_message(message_id)
        pruned_count += 1

    if pruned_count:
        with get_db() as conn:
            conn.execute(
                "UPDATE conversations SET updated_at = datetime('now') WHERE id = ?",
                (normalized_conversation_id,),
            )
    return pruned_count