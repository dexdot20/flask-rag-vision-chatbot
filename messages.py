from __future__ import annotations

import json
from datetime import datetime

from config import (
    MAX_SCRATCHPAD_LENGTH,
    MAX_USER_PREFERENCES_LENGTH,
    RAG_ENABLED,
)
from db import parse_message_metadata, parse_message_tool_calls
from tool_registry import get_prompt_tool_context

SUMMARY_LABEL = "Conversation summary (generated from deleted messages):"


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

    payload = {}
    if retrieved_context:
        payload["auto_injected_context"] = retrieved_context
    if search_enabled:
        payload["guidance"] = "Use retrieved context directly when sufficient, and avoid redundant knowledge-base searches."
    return payload or None


def _build_tool_memory_payload(tool_memory_context, active_tool_names: list[str]) -> dict | None:
    search_enabled = "search_tool_memory" in set(active_tool_names or [])
    if not tool_memory_context and not search_enabled:
        return None

    payload = {}
    if tool_memory_context:
        payload["auto_injected_context"] = tool_memory_context
    if search_enabled:
        payload["guidance"] = (
            "Tool Memory stores results from previous web searches, news lookups, and URL fetches. "
            "BEFORE repeating any web request, check Tool Memory first by calling search_tool_memory with a relevant query. "
            "Use remembered results when they answer the question adequately. "
            "Only perform a new web request when no matching memory exists or the stored data is clearly outdated for the question at hand."
        )
    return payload or None


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

    # --- Document context block ---
    file_context_block = (metadata.get("file_context_block") or "").strip()

    # --- Vision/Image block ---
    image_id = (metadata.get("image_id") or "").strip()
    image_name = (metadata.get("image_name") or "").strip()
    ocr_text = (metadata.get("ocr_text") or "").strip()
    vision_summary = (metadata.get("vision_summary") or "").strip()
    assistant_guidance = (metadata.get("assistant_guidance") or "").strip()
    key_points = metadata.get("key_points") if isinstance(metadata.get("key_points"), list) else []

    has_vision = image_id or ocr_text or vision_summary or assistant_guidance or key_points
    if not has_vision and not file_context_block:
        return content

    parts = []
    if content:
        parts.append(content)

    if file_context_block:
        parts.append(file_context_block)

    if has_vision:
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
        parts.append("\n\n".join(vision_parts))

    return "\n\n".join(parts)


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
            if not content.strip().lower().startswith(SUMMARY_LABEL.lower()):
                content = f"{SUMMARY_LABEL}\n\n{content.strip()}" if content.strip() else SUMMARY_LABEL

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
        "rules": [
            "If you need a tool, call it via native function calling instead of writing tool JSON in assistant content.",
            "If you do not need a tool, answer normally in assistant content.",
            "Use only the enabled tools listed above, and provide arguments matching the documented types.",
        ],
    }


def build_runtime_system_message(
    user_preferences="",
    active_tool_names=None,
    retrieved_context=None,
    tool_memory_context=None,
    now=None,
    scratchpad="",
):
    now = (now or datetime.now().astimezone()).astimezone()
    preferences_text = (user_preferences or "").strip()[:MAX_USER_PREFERENCES_LENGTH]
    scratchpad_text = (scratchpad or "").strip()[:MAX_SCRATCHPAD_LENGTH]
    active_tool_names = active_tool_names or []
    offset = now.strftime("%z")
    timezone_label = f"UTC{offset[:3]}:{offset[3:]}" if offset else (now.tzname() or "UTC")
    
    parts = ["You are a helpful AI assistant. You must respect the rules and guidelines provided below.\n"]
    
    # Time context
    parts.append(f"## Current Date and Time\n- ISO: {now.isoformat(timespec='seconds')}\n- Date: {now.date().isoformat()}\n- Time: {now.strftime('%H:%M:%S')}\n- Weekday: {now.strftime('%A')}\n- Timezone: {timezone_label}\n")

    # User preferences
    if preferences_text:
        parts.append(f"## User Preferences\n{preferences_text}\n")

    # Scratchpad
    if scratchpad_text or any(name in {"append_scratchpad", "replace_scratchpad"} for name in active_tool_names):
        parts.append("## Scratchpad (AI Persistent Memory)")
        if scratchpad_text:
            parts.append(scratchpad_text)
        else:
            parts.append("(Empty)")
        if any(name in {"append_scratchpad", "replace_scratchpad"} for name in active_tool_names):
            parts.append(
                "\n### Memory Write Policy\n"
                "- **DO save**: Durable user facts, preferences, confirmed personal details, and cross-conversation insights.\n"
                "- **DO NOT save**: One-off tasks, secrets, passwords, large summaries, raw tool outputs, or speculation.\n"
                "- **Web findings**: After every valuable web search, news lookup, or URL fetch, append a concise dated note "
                "(in English) summarizing the key finding — but ONLY when the result is likely to matter in future conversations. "
                "Format: `[YYYY-MM-DD] <topic>: <one-line summary>`.\n"
                "- **Critical rule**: When a web tool call returns information the user explicitly asked for, "
                "always evaluate whether it deserves a scratchpad entry. Err on the side of saving if in doubt."
            )
        parts.append("")

    tool_memory_payload = _build_tool_memory_payload(tool_memory_context, active_tool_names)
    if tool_memory_payload:
        parts.append("## Tool Memory")
        if "guidance" in tool_memory_payload:
            parts.append(f"*{tool_memory_payload['guidance']}*\n")
        if tool_memory_payload.get("auto_injected_context"):
            if isinstance(tool_memory_payload["auto_injected_context"], str):
                parts.append(tool_memory_payload["auto_injected_context"])
            else:
                parts.append(json.dumps(tool_memory_payload["auto_injected_context"], ensure_ascii=False, indent=2))
        parts.append("")
        
    # Knowledge Base / RAG Context
    kb_payload = _build_knowledge_base_payload(retrieved_context, active_tool_names)
    if kb_payload:
        parts.append("## Knowledge Base")
        if "guidance" in kb_payload:
            parts.append(f"*{kb_payload['guidance']}*\n")
        if kb_payload.get("auto_injected_context"):
            if isinstance(kb_payload["auto_injected_context"], str):
                parts.append(kb_payload["auto_injected_context"])
            else:
                parts.append(json.dumps(kb_payload["auto_injected_context"], ensure_ascii=False, indent=2))
        parts.append("")

    # Policies
    policies = []
    clarification_policy = _build_clarification_policy_payload(active_tool_names)
    if clarification_policy:
        policies.append(f"**Clarification**: {clarification_policy['guidance']}")
    image_policy = _build_image_policy_payload(active_tool_names)
    if image_policy:
        policies.append(f"**Image Follow-up**: {image_policy['guidance']}")
    
    if policies:
        parts.append("## Important Policies\n" + "\n".join(f"- {p}" for p in policies) + "\n")

    # Tool capabilities
    tools = get_prompt_tool_context(active_tool_names) if active_tool_names else None
    if tools:
        parts.append("## Available Tools")
        parts.append("You have access to the following tools:\n```json\n" + json.dumps(tools, ensure_ascii=False, indent=2) + "\n```\n")
        
        contract = build_tool_call_contract(active_tool_names)
        if contract:
            parts.append("### How to Call Tools")
            for rule in contract["rules"]:
                parts.append(f"- {rule}")
            parts.append("")

    return {
        "role": "system",
        "content": "\n".join(parts).strip()
    }


def prepend_runtime_context(
    messages,
    user_preferences="",
    active_tool_names=None,
    retrieved_context=None,
    tool_memory_context=None,
    scratchpad="",
):
    summary_count = sum(1 for message in messages if message.get("role") == "summary")
    runtime_message = build_runtime_system_message(
        user_preferences,
        active_tool_names or [],
        retrieved_context=retrieved_context,
        tool_memory_context=tool_memory_context,
        scratchpad=scratchpad,
    )
    
    system_content = runtime_message["content"]
    
    if summary_count:
        system_content += f"\n\n## Conversation Summaries\nCount: {summary_count}\n*Guidance: Summary-role messages compress earlier deleted conversation turns and should be treated as authoritative context.*"

    return [
        {
            "role": "system",
            "content": system_content,
        },
        *messages,
    ]
