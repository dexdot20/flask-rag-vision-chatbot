from __future__ import annotations

import json
from datetime import datetime

from config import (
    MAX_SCRATCHPAD_LENGTH,
    MAX_USER_PREFERENCES_LENGTH,
    RAG_AUTO_INJECT_THRESHOLD,
    RAG_ENABLED,
    RAG_SEARCH_MIN_SIMILARITY,
)
from db import parse_message_metadata, parse_message_tool_calls
from tool_registry import get_prompt_tool_context


def _build_user_preferences_payload(preferences_text: str) -> str | None:
    return preferences_text or None


def _build_scratchpad_payload(scratchpad_text: str, active_tool_names: list[str]) -> dict | None:
    scratchpad_tools_enabled = any(name in {"append_scratchpad", "replace_scratchpad"} for name in active_tool_names)
    if not scratchpad_text and not scratchpad_tools_enabled:
        return None

    payload = {"content": scratchpad_text or None}
    if scratchpad_tools_enabled:
        payload["memory_write_policy"] = (
            "Save only durable user facts or preferences. Never save one-off tasks, secrets, large summaries, tool outputs, or speculation."
        )
    return payload


def _build_image_policy_payload(active_tool_names: list[str]) -> dict | None:
    if "image_explain" not in set(active_tool_names or []):
        return None
    return {
        "tool": "image_explain",
        "guidance": "Use for follow-up questions about a stored prior image. Send the question in English. Ask for clarification if multiple earlier images could match.",
    }


def _build_clarification_policy_payload(active_tool_names: list[str]) -> dict | None:
    if "ask_clarifying_question" not in set(active_tool_names or []):
        return None
    return {
        "tool": "ask_clarifying_question",
        "guidance": (
            "If a good answer depends on missing requirements, ask for clarification instead of guessing. "
            "If the user explicitly asks you to ask questions first, use this tool rather than asking inline. "
            "Ask only the minimum number of questions needed, use this tool alone, and wait for the user's reply before continuing."
        ),
    }


def _build_knowledge_base_payload(retrieved_context, active_tool_names: list[str]) -> dict | None:
    if not RAG_ENABLED:
        return None

    search_enabled = "search_knowledge_base" in set(active_tool_names or [])
    if not retrieved_context and not search_enabled:
        return None

    payload = {
        "auto_injected_context": retrieved_context or None,
    }
    if search_enabled:
        payload["guidance"] = "Use retrieved context directly when sufficient, and avoid redundant knowledge-base searches."
    return payload


def normalize_chat_messages(messages) -> list[dict]:
    normalized = []
    allowed_roles = {"user", "assistant", "system", "tool", "summary"}

    for message in messages or []:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "").strip()
        if role not in allowed_roles:
            continue
        content = message.get("content")
        if content is None:
            content = ""
        if not isinstance(content, str):
            content = str(content)
        normalized.append(
            {
                "role": role,
                "content": content,
                "metadata": parse_message_metadata(message.get("metadata")),
                "tool_calls": parse_message_tool_calls(message.get("tool_calls")),
                "tool_call_id": str(message.get("tool_call_id") or "").strip() or None,
            }
        )

    return normalized


def build_user_message_for_model(content: str, metadata: dict | None = None) -> str:
    content = (content or "").strip()
    metadata = metadata if isinstance(metadata, dict) else {}
    image_id = (metadata.get("image_id") or "").strip()
    image_name = (metadata.get("image_name") or "").strip()
    ocr_text = (metadata.get("ocr_text") or "").strip()
    vision_summary = (metadata.get("vision_summary") or "").strip()
    assistant_guidance = (metadata.get("assistant_guidance") or "").strip()
    key_points = metadata.get("key_points") if isinstance(metadata.get("key_points"), list) else []

    if not image_id and not ocr_text and not vision_summary and not assistant_guidance and not key_points:
        return content

    vision_parts = ["[Local Qwen2.5-VL-7B vision assistant analysis]"]
    if image_id:
        reference_label = f"Stored image reference: image_id={image_id}"
        if image_name:
            reference_label += f", file={image_name}"
        vision_parts.append(reference_label)
    if vision_summary:
        vision_parts.append(f"Visual summary: {vision_summary}")
    if key_points:
        vision_parts.append("Key observations:\n- " + "\n- ".join(str(point) for point in key_points))
    if ocr_text:
        vision_parts.append("OCR text:\n" + ocr_text)
    if assistant_guidance:
        vision_parts.append("Answering guidance: " + assistant_guidance)

    vision_block = "\n\n".join(vision_parts)
    if content:
        return f"{content}\n\n{vision_block}"
    return vision_block


def build_api_messages(messages: list[dict]) -> list[dict]:
    api_messages = []
    for message in messages:
        content = message["content"]
        role = message["role"]
        metadata = message.get("metadata") if isinstance(message.get("metadata"), dict) else {}
        if role == "user":
            content = build_user_message_for_model(content, metadata)
        elif role == "summary":
            role = "assistant"
            summary_prefix = "Conversation summary (generated from deleted messages):"
            if not content.strip().lower().startswith(summary_prefix.lower()):
                content = f"{summary_prefix}\n\n{content.strip()}" if content.strip() else summary_prefix

        api_message = {
            "role": role,
            "content": content,
        }

        if role == "assistant":
            tool_calls = parse_message_tool_calls(message.get("tool_calls"))
            if tool_calls:
                api_message["tool_calls"] = tool_calls
        elif role == "tool":
            tool_call_id = str(message.get("tool_call_id") or "").strip()
            if tool_call_id:
                api_message["tool_call_id"] = tool_call_id

        api_messages.append(api_message)
    return api_messages


def build_tool_call_contract(active_tool_names: list[str]) -> dict | None:
    tools = get_prompt_tool_context(active_tool_names)
    if not tools:
        return None
    return {
        "mode": "custom_json_tool_calls",
        "parse_channel": "assistant_content_only",
        "tool_json_shape": {
            "tool_calls": [
                {
                    "name": "tool_name",
                    "arguments": {"key": "value"},
                }
            ]
        },
        "rules": [
            "If you need tools, output only one JSON object with a top-level tool_calls array in assistant content.",
            "If you do not need tools, output the final answer normally.",
            "Do not mix final-answer text with tool JSON in the same assistant message.",
            "Use only the enabled tools listed above, and provide arguments as a JSON object.",
            "If the user explicitly asks you to ask questions first before answering, use ask_clarifying_question instead of asking inline.",
        ],
    }


def build_runtime_system_message(
    user_preferences="",
    active_tool_names=None,
    retrieved_context=None,
    now=None,
    scratchpad="",
):
    now = (now or datetime.now().astimezone()).astimezone()
    preferences_text = (user_preferences or "").strip()[:MAX_USER_PREFERENCES_LENGTH]
    scratchpad_text = (scratchpad or "").strip()[:MAX_SCRATCHPAD_LENGTH]
    active_tool_names = active_tool_names or []
    offset = now.strftime("%z")
    timezone_label = f"UTC{offset[:3]}:{offset[3:]}" if offset else (now.tzname() or "UTC")
    payload = {
        "context_type": "runtime_prompt_context",
        "current_datetime": {
            "iso": now.isoformat(timespec="seconds"),
            "date": now.date().isoformat(),
            "time": now.strftime("%H:%M:%S"),
            "weekday": now.strftime("%A"),
            "timezone": timezone_label,
        },
        "user_preferences": _build_user_preferences_payload(preferences_text),
        "scratchpad": _build_scratchpad_payload(scratchpad_text, active_tool_names),
        "clarification_policy": _build_clarification_policy_payload(active_tool_names),
        "image_follow_up_policy": _build_image_policy_payload(active_tool_names),
        "knowledge_base": _build_knowledge_base_payload(retrieved_context, active_tool_names),
        "available_tools": get_prompt_tool_context(active_tool_names) if active_tool_names else None,
        "tool_call_contract": build_tool_call_contract(active_tool_names) if active_tool_names else None,
    }
    return {
        "role": "system",
        "content": json.dumps(payload, ensure_ascii=False),
    }


def prepend_runtime_context(messages, user_preferences="", active_tool_names=None, retrieved_context=None, scratchpad=""):
    summary_count = sum(1 for message in messages if message.get("role") == "summary")
    runtime_message = build_runtime_system_message(
        user_preferences,
        active_tool_names or [],
        retrieved_context=retrieved_context,
        scratchpad=scratchpad,
    )
    runtime_payload = json.loads(runtime_message["content"])
    if summary_count:
        runtime_payload["conversation_summaries"] = {
            "count": summary_count,
            "guidance": "Summary-role messages compress earlier deleted conversation turns and should be treated as authoritative context.",
        }
    system_content = (
        "You are a helpful AI assistant. You must respect the rules and tools provided in the JSON configuration below.\n"
        "Pay special attention to the 'tool_call_contract', 'available_tools', and any policy guidance.\n\n"
        f"```json\n{json.dumps(runtime_payload, ensure_ascii=False, indent=2)}\n```"
    )

    return [
        {
            "role": "system",
            "content": system_content,
        },
        *messages,
    ]
