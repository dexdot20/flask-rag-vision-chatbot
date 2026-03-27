from __future__ import annotations

import json
from datetime import datetime

from canvas_service import build_canvas_project_manifest, extract_canvas_documents, scale_canvas_char_limit
from config import (
    MAX_USER_PREFERENCES_LENGTH,
    RAG_ENABLED,
)
from db import extract_message_attachments, parse_message_metadata, parse_message_tool_calls
from tool_registry import get_prompt_tool_context, resolve_runtime_tool_names

SUMMARY_LABEL = "Conversation summary (generated from deleted messages):"
CANVAS_PROMPT_MAX_CHARS = 12_000
CANVAS_PROMPT_MAX_LINES = 400


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

    attachments = extract_message_attachments(metadata)
    file_context_blocks = []
    vision_attachments = []
    for attachment in attachments:
        if attachment.get("kind") == "document":
            context_block = str(attachment.get("file_context_block") or "").strip()
            if context_block and context_block not in file_context_blocks:
                file_context_blocks.append(context_block)
            continue

        image_id = str(attachment.get("image_id") or "").strip()
        image_name = str(attachment.get("image_name") or "").strip()
        ocr_text = str(attachment.get("ocr_text") or "").strip()
        vision_summary = str(attachment.get("vision_summary") or "").strip()
        assistant_guidance = str(attachment.get("assistant_guidance") or "").strip()
        key_points = attachment.get("key_points") if isinstance(attachment.get("key_points"), list) else []
        has_vision = image_id or image_name or ocr_text or vision_summary or assistant_guidance or key_points
        if has_vision:
            vision_attachments.append(attachment)

    if not file_context_blocks and not vision_attachments:
        return content

    parts = []
    if content:
        parts.append(content)

    parts.extend(file_context_blocks)

    for index, attachment in enumerate(vision_attachments, start=1):
        image_id = str(attachment.get("image_id") or "").strip()
        image_name = str(attachment.get("image_name") or "").strip()
        ocr_text = str(attachment.get("ocr_text") or "").strip()
        vision_summary = str(attachment.get("vision_summary") or "").strip()
        assistant_guidance = str(attachment.get("assistant_guidance") or "").strip()
        key_points = attachment.get("key_points") if isinstance(attachment.get("key_points"), list) else []

        heading = "[Local Qwen2.5-VL-7B vision assistant analysis]"
        if len(vision_attachments) > 1:
            heading = f"{heading} Attachment {index}"
        vision_parts = [heading]
        if image_id:
            reference_label = f"Stored image reference: image_id={image_id}"
            if image_name:
                reference_label += f", file={image_name}"
            vision_parts.append(reference_label)
        elif image_name:
            vision_parts.append(f"Uploaded image: {image_name}")
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


def _build_canvas_prompt_payload(
    canvas_documents,
    active_document_id: str | None = None,
    *,
    max_lines: int = CANVAS_PROMPT_MAX_LINES,
    max_chars: int | None = None,
) -> dict | None:
    documents = extract_canvas_documents({"canvas_documents": canvas_documents or []})
    if not documents:
        return None

    if max_chars is None:
        max_chars = scale_canvas_char_limit(
            max_lines,
            default_lines=CANVAS_PROMPT_MAX_LINES,
            default_chars=CANVAS_PROMPT_MAX_CHARS,
        )

    manifest = build_canvas_project_manifest(documents, active_document_id=active_document_id)
    resolved_active_document_id = str((manifest or {}).get("active_document_id") or "").strip()
    active_document = documents[-1]
    if resolved_active_document_id:
        for document in documents:
            if str(document.get("id") or "") == resolved_active_document_id:
                active_document = document
                break

    content = str(active_document.get("content") or "")
    all_lines = content.split("\n") if content else []
    visible_lines = []
    visible_char_count = 0

    for index, line in enumerate(all_lines, start=1):
        numbered_line = f"{index}: {line}"
        extra_chars = len(numbered_line) + (1 if visible_lines else 0)
        if visible_lines and (len(visible_lines) >= max_lines or visible_char_count + extra_chars > max_chars):
            break
        if not visible_lines and extra_chars > max_chars:
            visible_lines.append(numbered_line[:max_chars])
            visible_char_count = len(visible_lines[0])
            break
        visible_lines.append(numbered_line)
        visible_char_count += extra_chars

    return {
        "mode": (manifest or {}).get("mode") or "document",
        "manifest": manifest,
        "relationship_map": (manifest or {}).get("relationship_map"),
        "document_count": len(documents),
        "active_document": active_document,
        "other_documents": [
            entry
            for entry in ((manifest or {}).get("file_list") or [])
            if entry.get("id") != active_document.get("id")
        ],
        "visible_lines": visible_lines,
        "is_truncated": len(visible_lines) < len(all_lines),
        "visible_line_end": len(visible_lines),
        "total_lines": int(active_document.get("line_count") or len(all_lines)),
    }


def build_tool_call_contract(active_tool_names: list[str], canvas_documents=None) -> dict | None:
    tools = get_prompt_tool_context(active_tool_names, canvas_documents=canvas_documents)
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
    user_profile_context=None,
    tool_trace_context=None,
    tool_memory_context=None,
    now=None,
    scratchpad="",
    canvas_documents=None,
    canvas_active_document_id: str | None = None,
    canvas_prompt_max_lines: int | None = None,
    workspace_root: str | None = None,
    project_workflow: dict | None = None,
):
    now = (now or datetime.now().astimezone()).astimezone()
    preferences_text = (user_preferences or "").strip()[:MAX_USER_PREFERENCES_LENGTH]
    scratchpad_text = (scratchpad or "").strip()
    active_tool_names = resolve_runtime_tool_names(active_tool_names or [], canvas_documents=canvas_documents)
    offset = now.strftime("%z")
    timezone_label = f"UTC{offset[:3]}:{offset[3:]}" if offset else (now.tzname() or "UTC")
    
    parts = ["You are a helpful AI assistant. You must respect the rules and guidelines provided below.\n"]
    
    # Time context
    parts.append(f"## Current Date and Time\n- ISO: {now.isoformat(timespec='seconds')}\n- Date: {now.date().isoformat()}\n- Time: {now.strftime('%H:%M:%S')}\n- Weekday: {now.strftime('%A')}\n- Timezone: {timezone_label}\n")

    # User preferences
    if preferences_text:
        parts.append(f"## User Preferences\n{preferences_text}\n")

    normalized_user_profile_context = str(user_profile_context or "").strip()
    if normalized_user_profile_context:
        parts.append("## User Profile")
        parts.append(
            "*Use this as durable cross-conversation memory about the user when it is relevant to the current request. Do not treat it as higher priority than the user's latest explicit instruction.*\n"
        )
        parts.append(normalized_user_profile_context)
        parts.append("")

    # Scratchpad
    if scratchpad_text or any(name in {"append_scratchpad", "replace_scratchpad"} for name in active_tool_names):
        parts.append("## Scratchpad (AI Persistent Memory)")
        parts.append("*This section is your complete scratchpad — you can read it directly here without calling any tool.*\n")
        if scratchpad_text:
            parts.append(scratchpad_text)
        else:
            parts.append("(Empty)")
        if any(name in {"append_scratchpad", "replace_scratchpad"} for name in active_tool_names):
            parts.append(
                "\n### Memory Write Policy\n"
                "- **DO save**: Only durable, high-signal facts that are likely to change future answers or actions. Examples: stable user preferences, long-lived constraints, confirmed identity details, and recurring requirements.\n"
                "- **DO NOT save**: One-off tasks, transient project state, raw tool outputs, web/search results, speculative inferences, broad summaries, or details already obvious from the current chat.\n"
                "- **Before saving**: Ask whether this information will still matter in a future conversation and whether it is specific enough to be useful as a single short note. If not, do not save it.\n"
                "- **Web findings**: Do not turn search/news/URL results into scratchpad entries unless the result is clearly durable and the user would reasonably expect it to be remembered later. Never save them just because they were requested.\n"
                "- **Style**: Each `notes` item must be one single short standalone fact. Never put multiple facts in one item. Call `append_scratchpad` once per batch of facts instead of once per fact."
            )
        parts.append("")

    normalized_tool_trace_context = str(tool_trace_context or "").strip()
    if normalized_tool_trace_context:
        parts.append("## Tool Execution History")
        parts.append(
            "*Use this as recent operational memory about which tools were already tried, what they returned, and which paths should not be repeated without a concrete reason.*\n"
        )
        parts.append(normalized_tool_trace_context)
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

    normalized_workspace_root = str(workspace_root or "").strip()
    if normalized_workspace_root:
        parts.append("## Workspace Sandbox")
        parts.append(f"- Root: {normalized_workspace_root}")
        parts.append("- Scope: All workspace file tools must stay inside this root.")
        parts.append("- Safety: If a batch write tool returns needs_confirmation, wait for explicit user approval before re-running with confirm=true.\n")
        parts.append("- Review: Prefer returned unified diffs or preview_workspace_changes before high-impact rewrites, and use workspace undo/redo tools for recovery.\n")

    if isinstance(project_workflow, dict) and project_workflow:
        parts.append("## Project Workflow")
        parts.append("```json\n" + json.dumps(project_workflow, ensure_ascii=False, indent=2) + "\n```\n")

    canvas_payload = _build_canvas_prompt_payload(
        canvas_documents,
        active_document_id=canvas_active_document_id,
        max_lines=canvas_prompt_max_lines or CANVAS_PROMPT_MAX_LINES,
    )
    if canvas_payload:
        manifest = canvas_payload.get("manifest")
        if manifest and canvas_payload.get("mode") == "project":
            parts.append("## Canvas Project Manifest")
            parts.append("```json\n" + json.dumps(manifest, ensure_ascii=False, indent=2) + "\n```\n")
        relationship_map = canvas_payload.get("relationship_map")
        if relationship_map and canvas_payload.get("mode") == "project":
            parts.append("## Canvas Relationship Map")
            parts.append("```json\n" + json.dumps(relationship_map, ensure_ascii=False, indent=2) + "\n```\n")
        active_document = canvas_payload["active_document"]
        parts.append("## Active Canvas Document")
        parts.append(f"- Working mode: {canvas_payload['mode']}")
        parts.append(f"- Document count: {canvas_payload['document_count']}")
        parts.append(f"- Active document id: {active_document['id']}")
        parts.append(f"- Title: {active_document['title']}")
        if active_document.get("path"):
            parts.append(f"- Path: {active_document['path']}")
        if active_document.get("role"):
            parts.append(f"- Role: {active_document['role']}")
        parts.append(f"- Format: {active_document['format']}")
        if active_document.get("language"):
            parts.append(f"- Language: {active_document['language']}")
        if active_document.get("summary"):
            parts.append(f"- Summary: {active_document['summary']}")
        parts.append(f"- Total lines: {canvas_payload['total_lines']}")
        parts.append(
            f"- Visible lines in prompt: 1-{canvas_payload['visible_line_end']}"
            + (" (truncated excerpt)" if canvas_payload["is_truncated"] else "")
        )
        parts.append(
            "- Guidance: Use visible line numbers for line-level canvas edits. "
            "If this excerpt is truncated, call expand_canvas_document for a larger view or scroll_canvas_document for a targeted range before editing. "
            "Never guess line numbers outside the visible excerpt."
        )
        if canvas_payload["visible_lines"]:
            parts.append("```json\n" + json.dumps(canvas_payload["visible_lines"], ensure_ascii=False, indent=2) + "\n```\n")
        else:
            parts.append("(The active canvas document is empty.)\n")
        other_documents = canvas_payload.get("other_documents") or []
        if other_documents:
            parts.append("## Other Canvas Documents")
            parts.append("```json\n" + json.dumps(other_documents, ensure_ascii=False, indent=2) + "\n```\n")

    # Tool capabilities
    tools = get_prompt_tool_context(active_tool_names, canvas_documents=canvas_documents) if active_tool_names else None
    if tools:
        parts.append("## Available Tools")
        parts.append("You have access to the following tools:\n```json\n" + json.dumps(tools, ensure_ascii=False, indent=2) + "\n```\n")
        
        contract = build_tool_call_contract(active_tool_names, canvas_documents=canvas_documents)
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
    user_profile_context=None,
    tool_trace_context=None,
    tool_memory_context=None,
    scratchpad="",
    canvas_documents=None,
    canvas_active_document_id: str | None = None,
    canvas_prompt_max_lines: int | None = None,
    workspace_root: str | None = None,
    project_workflow: dict | None = None,
):
    summary_count = sum(1 for message in messages if message.get("role") == "summary")
    runtime_message = build_runtime_system_message(
        user_preferences,
        active_tool_names or [],
        retrieved_context=retrieved_context,
        user_profile_context=user_profile_context,
        tool_trace_context=tool_trace_context,
        tool_memory_context=tool_memory_context,
        scratchpad=scratchpad,
        canvas_documents=canvas_documents,
        canvas_active_document_id=canvas_active_document_id,
        canvas_prompt_max_lines=canvas_prompt_max_lines,
        workspace_root=workspace_root,
        project_workflow=project_workflow,
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
