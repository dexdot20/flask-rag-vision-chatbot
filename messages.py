from __future__ import annotations

import json
import os
from datetime import datetime

from canvas_service import build_canvas_project_manifest, extract_canvas_documents, scale_canvas_char_limit
from config import (
    MAX_USER_PREFERENCES_LENGTH,
    RAG_ENABLED,
)
from db import extract_message_attachments, parse_message_metadata, parse_message_tool_calls
from tool_registry import build_canvas_decision_matrix, resolve_runtime_tool_names

SUMMARY_LABEL = "Conversation summary (generated from deleted messages):"
CANVAS_PROMPT_MAX_CHARS = 12_000
CANVAS_PROMPT_MAX_LINES = 400
PARALLEL_SAFE_READ_ONLY_TOOL_NAMES = (
    # Web / fetch
    "search_web",
    "fetch_url",
    "search_news_ddgs",
    "search_news_google",
    "image_explain",
    # RAG / memory reads
    "search_knowledge_base",
    "search_tool_memory",
    # Workspace reads
    "read_file",
    "list_dir",
    "search_files",
    "get_project_workflow_status",
    "validate_project_workspace",
    "get_workspace_file_history",
    "preview_workspace_changes",
    # Canvas inspection (non-mutating)
    "expand_canvas_document",
    "scroll_canvas_document",
)
# Tools whose results may still be inputs for other calls in the same batch;
# they are parallel-safe among themselves but must not be batched with any
# call that depends on their output.
DEPENDENT_TOOL_NAMES = (
    "search_knowledge_base",
    "search_tool_memory",
)


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


def format_knowledge_base_auto_context(retrieved_context) -> str:
    normalized = str(retrieved_context or "").strip()
    if isinstance(retrieved_context, str):
        return normalized
    if not isinstance(retrieved_context, dict):
        return normalized

    matches = retrieved_context.get("matches") if isinstance(retrieved_context.get("matches"), list) else []
    if not matches:
        return ""

    query = str(retrieved_context.get("query") or "").strip()
    sections: list[str] = []
    if query:
        sections.append(f"Auto-injected query: {query}")

    for index, match in enumerate(matches, start=1):
        if not isinstance(match, dict):
            continue
        source_name = str(match.get("source_name") or match.get("source") or f"Match {index}").strip() or f"Match {index}"
        similarity = match.get("similarity")
        heading = f"[{index}] Source: {source_name}"
        if isinstance(similarity, (int, float)):
            heading += f" | similarity {float(similarity):.2f}"
        excerpt = str(match.get("text") or match.get("excerpt") or "").strip()
        sections.append("\n".join(part for part in (heading, excerpt) if part))

    return "\n\n".join(section for section in sections if section).strip()


def _build_knowledge_base_payload(retrieved_context, active_tool_names: list[str]) -> dict | None:
    if not RAG_ENABLED:
        return None

    search_enabled = "search_knowledge_base" in set(active_tool_names or [])
    if not retrieved_context and not search_enabled:
        return None

    payload = {}
    if retrieved_context:
        formatted_context = format_knowledge_base_auto_context(retrieved_context)
        if formatted_context:
            payload["auto_injected_context"] = formatted_context
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


def _normalize_canvas_document_name(value: str | None) -> str:
    return os.path.basename(str(value or "").strip()).casefold()


def _extract_document_context_body(context_block: str | None) -> str:
    normalized = str(context_block or "").strip()
    if not normalized:
        return ""

    lines = normalized.splitlines()
    if lines and lines[0].startswith("[Uploaded document:"):
        lines = lines[1:]
        while lines and not lines[0].strip():
            lines = lines[1:]
    return "\n".join(lines).strip()


def _build_canvas_document_lookup(canvas_documents) -> dict[str, list[str]]:
    documents = extract_canvas_documents({"canvas_documents": canvas_documents or []})
    lookup: dict[str, list[str]] = {}
    for document in documents:
        normalized_content = str(document.get("content") or "").strip().casefold()
        if not normalized_content:
            continue
        for candidate in (
            _normalize_canvas_document_name(document.get("title")),
            _normalize_canvas_document_name(document.get("path")),
        ):
            if not candidate:
                continue
            lookup.setdefault(candidate, []).append(normalized_content)
    return lookup


def _document_attachment_is_represented_in_canvas(attachment: dict, canvas_document_lookup: dict[str, list[str]]) -> bool:
    if not canvas_document_lookup:
        return False

    normalized_name = _normalize_canvas_document_name(attachment.get("file_name"))
    if not normalized_name:
        return False

    candidate_contents = canvas_document_lookup.get(normalized_name) or []
    if not candidate_contents:
        return False

    body_excerpt = _extract_document_context_body(attachment.get("file_context_block"))
    if not body_excerpt:
        return True

    normalized_excerpt = body_excerpt[:500].casefold()
    return any(normalized_excerpt in content for content in candidate_contents)


def build_user_message_for_model(
    content: str,
    metadata: dict | None = None,
    *,
    canvas_documents: list[dict] | None = None,
) -> str:
    content = (content or "").strip()
    metadata = metadata if isinstance(metadata, dict) else {}

    attachments = extract_message_attachments(metadata)
    canvas_document_lookup = _build_canvas_document_lookup(canvas_documents)
    file_context_blocks = []
    vision_attachments = []
    for attachment in attachments:
        if attachment.get("kind") == "document":
            context_block = str(attachment.get("file_context_block") or "").strip()
            if _document_attachment_is_represented_in_canvas(attachment, canvas_document_lookup):
                continue
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

        heading = "[Local image analysis]"
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


def build_api_messages(messages: list[dict], *, canvas_documents: list[dict] | None = None) -> list[dict]:
    api_messages = []
    for message in messages:
        content = message["content"]
        role = message["role"]
        metadata = message.get("metadata") if isinstance(message.get("metadata"), dict) else {}
        if role == "user":
            content = build_user_message_for_model(content, metadata, canvas_documents=canvas_documents)
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


def _build_canvas_workspace_summary(canvas_payload: dict) -> list[str]:
    manifest = canvas_payload.get("manifest") if isinstance(canvas_payload.get("manifest"), dict) else {}
    active_document = canvas_payload.get("active_document") if isinstance(canvas_payload.get("active_document"), dict) else {}
    lines = ["## Canvas Workspace Summary"]
    lines.append(f"- Working mode: {canvas_payload.get('mode') or 'document'}")
    lines.append(f"- Document count: {canvas_payload.get('document_count') or 0}")

    project_name = str(manifest.get("project_name") or "").strip()
    if project_name:
        lines.append(f"- Project label: {project_name}")

    target_type = str(manifest.get("target_type") or "").strip()
    if target_type:
        lines.append(f"- Target type: {target_type}")

    active_label = str(active_document.get("path") or active_document.get("title") or active_document.get("id") or "Canvas").strip()
    lines.append(f"- Active file: {active_label}")

    validation_status = str(manifest.get("last_validation_status") or "ok").strip() or "ok"
    lines.append(f"- Validation status: {validation_status}")

    open_issues = [str(issue).strip() for issue in (manifest.get("open_issues") or []) if str(issue).strip()]
    if open_issues:
        lines.append(f"- Open issues: {'; '.join(open_issues[:4])}")
    else:
        lines.append("- Open issues: none")

    file_list = manifest.get("file_list") if isinstance(manifest.get("file_list"), list) else []
    if file_list:
        lines.append("- Files in scope:")
        for entry in file_list[:8]:
            label = str(entry.get("path") or entry.get("title") or entry.get("id") or "Canvas").strip() or "Canvas"
            summary_parts = []
            if entry.get("active"):
                summary_parts.append("active")
            if entry.get("role"):
                summary_parts.append(str(entry["role"]))
            if entry.get("language"):
                summary_parts.append(str(entry["language"]))
            summary_parts.append(f"{int(entry.get('line_count') or 0)} lines")
            summary = str(entry.get("summary") or "").strip()
            suffix = f" | {summary}" if summary else ""
            lines.append(f"  - {label} ({', '.join(summary_parts)}){suffix}")
        if len(file_list) > 8:
            lines.append(f"  - ... {len(file_list) - 8} more files omitted from this summary.")

    relationship_map = canvas_payload.get("relationship_map") if isinstance(canvas_payload.get("relationship_map"), dict) else {}
    for key, label in (("imports", "Shared imports"), ("exports", "Shared exports"), ("dependencies", "Shared dependencies")):
        values = relationship_map.get(key) if isinstance(relationship_map.get(key), list) else []
        compact_values = [str(value).strip() for value in values if str(value).strip()][:8]
        if compact_values:
            lines.append(f"- {label}: {', '.join(compact_values)}")

    lines.append("")
    return lines


def _build_canvas_decision_matrix_rows(active_tool_names: list[str], canvas_payload: dict | None = None) -> list[dict[str, str]]:
    active_set = set(active_tool_names or [])
    if not active_set.intersection(
        {
            "create_canvas_document",
            "rewrite_canvas_document",
            "replace_canvas_lines",
            "insert_canvas_lines",
            "delete_canvas_lines",
            "expand_canvas_document",
            "scroll_canvas_document",
        }
    ):
        return []

    return build_canvas_decision_matrix(
        active_tool_names,
        has_canvas_documents=bool(canvas_payload),
        canvas_mode=(canvas_payload or {}).get("mode"),
    )


def _build_canvas_editing_guidance(active_tool_names: list[str], canvas_payload: dict | None = None) -> list[str]:
    active_set = set(active_tool_names or [])
    if not active_set.intersection(
        {
            "create_canvas_document",
            "rewrite_canvas_document",
            "replace_canvas_lines",
            "insert_canvas_lines",
            "delete_canvas_lines",
            "expand_canvas_document",
            "scroll_canvas_document",
        }
    ):
        return []

    lines = [
        "## Canvas Editing Guidance",
        "- Prefer the smallest valid canvas change that satisfies the request.",
        "- Do not rewrite the whole document when only part needs to change; use replace_canvas_lines, insert_canvas_lines, or delete_canvas_lines for local edits when the exact visible lines are known.",
        "- If the target lines are not visible yet, inspect first with scroll_canvas_document for a focused range or expand_canvas_document for a wider view.",
        "- If you do not know the document_id, use the document_path from the workspace summary, active file label, or manifest; document_id is optional.",
        "- Use rewrite_canvas_document when most of the document should change or when you already know the complete intended replacement content.",
        "- Multiple canvas tool calls in one answer are fine when needed: inspect, then edit, then create or update other files.",
    ]
    if (canvas_payload or {}).get("mode") == "project":
        lines.append("- In project mode, prefer document_path for targeting; it is enough even when you do not know the document_id yet, and keep one file per canvas document.")
    lines.append("")
    return lines


def build_tool_call_contract(active_tool_names: list[str], canvas_documents=None) -> dict | None:
    runtime_tool_names = resolve_runtime_tool_names(active_tool_names or [], canvas_documents=canvas_documents)
    if not runtime_tool_names:
        return None
    rules = [
        "Call a tool only when it is required or when it will materially improve correctness, completeness, or safety. If you can answer reliably from the current context, do not call a tool.",
        "Unnecessary tool calls waste tokens and context, so do not use tools for convenience, repetition, or curiosity.",
        "If you do need a tool, call it via native function calling instead of writing tool JSON in assistant content.",
        "Use only the tools exposed by the API for this turn, and provide arguments matching the documented types.",
    ]

    parallel_safe_in_use = [name for name in PARALLEL_SAFE_READ_ONLY_TOOL_NAMES if name in runtime_tool_names]
    if parallel_safe_in_use:
        rules.append(
            "## Parallel and Batched Tool Calls\n"
            "Batching independent tool calls into one assistant turn reduces LLM round trips and saves tokens. "
            "Prefer this pattern aggressively.\n\n"
            "**GATHER → REASON → ACT** — the core pattern:\n"
            "  1. Issue ALL independent reads in one turn (gather phase).\n"
            "  2. Reason over all returned results together.\n"
            "  3. Issue writes / mutations based on what you learned.\n\n"
            "**Concurrently executed (I/O runs in parallel):** "
            + ", ".join(parallel_safe_in_use)
            + ".\n"
            "Emitting multiple calls to these tools in one turn causes them to run at the same time — "
            "use this whenever you need several independent lookups.\n\n"
            "**All other tools** run sequentially within a turn, but batching them still saves a full LLM "
            "round trip. Batch independent writes together when their inputs do not depend on each other "
            "(e.g. creating two unrelated files, updating two different canvas documents).\n\n"
            "**Examples of correct batching:**\n"
            "  - Read three files at once: emit read_file × 3 in one turn.\n"
            "  - Explore a directory and search for a keyword simultaneously: list_dir + search_files in one turn.\n"
            "  - Search the web and the knowledge base for the same topic simultaneously: search_web + search_knowledge_base in one turn.\n"
            "  - Write two independent new files in one turn: create_file × 2.\n\n"
            "**When NOT to batch (data dependency requires two turns):**\n"
            "  - fetch_url X → then parse X's result to decide the next tool call.\n"
            "  - search_knowledge_base → then use the returned chunk IDs to call another tool.\n"
            "  - Any case where tool B needs the output of tool A as its input."
        )

    if any(name in runtime_tool_names for name in DEPENDENT_TOOL_NAMES):
        rules.append(
            "search_knowledge_base and search_tool_memory may be batched freely with other independent reads "
            "(they run concurrently), but must not be batched with any tool call whose input depends on their output."
        )

    if "ask_clarifying_question" in runtime_tool_names:
        rules.append("ask_clarifying_question must be the only tool call in its assistant turn.")

    return {"rules": rules}


def _build_current_time_context(now: datetime) -> str:
    normalized_now = now.astimezone()
    offset = normalized_now.strftime("%z")
    timezone_label = f"UTC{offset[:3]}:{offset[3:]}" if offset else (normalized_now.tzname() or "UTC")
    return (
        f"## Current Date and Time\n- ISO: {normalized_now.isoformat(timespec='seconds')}\n"
        f"- Date: {normalized_now.date().isoformat()}\n- Time: {normalized_now.strftime('%H:%M:%S')}\n"
        f"- Weekday: {normalized_now.strftime('%A')}\n- Timezone: {timezone_label}\n"
    )


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
    include_time_context: bool = True,
):
    now = (now or datetime.now().astimezone()).astimezone()
    preferences_text = (user_preferences or "").strip()[:MAX_USER_PREFERENCES_LENGTH]
    scratchpad_text = (scratchpad or "").strip()
    active_tool_names = resolve_runtime_tool_names(active_tool_names or [], canvas_documents=canvas_documents)
    
    parts = ["You are a helpful AI assistant. You must respect the rules and guidelines provided below.\n"]
    volatile_parts: list[str] = []

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
        volatile_parts.append("## Tool Execution History")
        volatile_parts.append(
            "*Use this as recent operational memory about which tools were already tried, what they returned, and which paths should not be repeated without a concrete reason.*\n"
        )
        volatile_parts.append(normalized_tool_trace_context)
        volatile_parts.append("")

    tool_memory_payload = _build_tool_memory_payload(tool_memory_context, active_tool_names)
    if tool_memory_payload:
        volatile_parts.append("## Tool Memory")
        if "guidance" in tool_memory_payload:
            volatile_parts.append(f"*{tool_memory_payload['guidance']}*\n")
        if tool_memory_payload.get("auto_injected_context"):
            if isinstance(tool_memory_payload["auto_injected_context"], str):
                volatile_parts.append(tool_memory_payload["auto_injected_context"])
            else:
                volatile_parts.append(json.dumps(tool_memory_payload["auto_injected_context"], ensure_ascii=False, indent=2))
        volatile_parts.append("")
        
    # Knowledge Base / RAG Context
    kb_payload = _build_knowledge_base_payload(retrieved_context, active_tool_names)
    if kb_payload:
        volatile_parts.append("## Knowledge Base")
        if "guidance" in kb_payload:
            volatile_parts.append(f"*{kb_payload['guidance']}*\n")
        if kb_payload.get("auto_injected_context"):
            if isinstance(kb_payload["auto_injected_context"], str):
                volatile_parts.append(kb_payload["auto_injected_context"])
            else:
                volatile_parts.append(json.dumps(kb_payload["auto_injected_context"], ensure_ascii=False, indent=2))
        volatile_parts.append("")

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
        volatile_parts.append("## Project Workflow")
        volatile_parts.append("```json\n" + json.dumps(project_workflow, ensure_ascii=False, indent=2) + "\n```\n")

    canvas_payload = _build_canvas_prompt_payload(
        canvas_documents,
        active_document_id=canvas_active_document_id,
        max_lines=canvas_prompt_max_lines or CANVAS_PROMPT_MAX_LINES,
    )
    if canvas_payload:
        volatile_parts.extend(_build_canvas_workspace_summary(canvas_payload))
        active_document = canvas_payload["active_document"]
        volatile_parts.append("## Active Canvas Document")
        volatile_parts.append(f"- Working mode: {canvas_payload['mode']}")
        volatile_parts.append(f"- Document count: {canvas_payload['document_count']}")
        volatile_parts.append(f"- Active document id: {active_document['id']}")
        volatile_parts.append(f"- Title: {active_document['title']}")
        if active_document.get("path"):
            volatile_parts.append(f"- Path: {active_document['path']}")
        if active_document.get("role"):
            volatile_parts.append(f"- Role: {active_document['role']}")
        volatile_parts.append(f"- Format: {active_document['format']}")
        if active_document.get("language"):
            volatile_parts.append(f"- Language: {active_document['language']}")
        if active_document.get("summary"):
            volatile_parts.append(f"- Summary: {active_document['summary']}")
        volatile_parts.append(f"- Total lines: {canvas_payload['total_lines']}")
        volatile_parts.append(
            f"- Visible lines in prompt: 1-{canvas_payload['visible_line_end']}"
            + (" (truncated excerpt)" if canvas_payload["is_truncated"] else "")
        )
        volatile_parts.append(
            "- Guidance: Use visible line numbers for line-level canvas edits. "
            "If you do not know the document_id, target by document_path from the workspace summary or active file label instead. "
            "If this excerpt is truncated, call expand_canvas_document for a larger view or scroll_canvas_document for a targeted range before editing. "
            "Never guess line numbers outside the visible excerpt."
        )
        if canvas_payload["visible_lines"]:
            volatile_parts.append("```text\n" + "\n".join(canvas_payload["visible_lines"]) + "\n```\n")
        else:
            volatile_parts.append("(The active canvas document is empty.)\n")

    canvas_editing_guidance = _build_canvas_editing_guidance(active_tool_names, canvas_payload)
    if canvas_editing_guidance:
        parts.extend(canvas_editing_guidance)

    canvas_decision_matrix = _build_canvas_decision_matrix_rows(active_tool_names, canvas_payload)
    if canvas_decision_matrix:
        parts.append("## Canvas Decision Matrix")
        parts.append("| Situation | Preferred tool | Notes |")
        parts.append("| --- | --- | --- |")
        for row in canvas_decision_matrix:
            parts.append(f"| {row['situation']} | {row['tool']} | {row['notes']} |")
        parts.append("")

    contract = build_tool_call_contract(active_tool_names, canvas_documents=canvas_documents)
    if contract:
        parts.append("## Tool Calling")
        parts.append(
            "Native function calling is enabled for this turn. Do not restate tool schemas or invent unavailable tools. "
            "Only call tools when they are truly needed; unnecessary calls waste tokens and context.\n"
        )
        for rule in contract["rules"]:
            parts.append(f"- {rule}")
        parts.append("")

    parts.extend(volatile_parts)

    if include_time_context:
        parts.append(_build_current_time_context(now))

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
        include_time_context=False,
    )
    
    system_content = runtime_message["content"]
    
    if summary_count:
        system_content += f"\n\n## Conversation Summaries\nCount: {summary_count}\n*Guidance: Summary-role messages compress earlier deleted conversation turns and should be treated as authoritative context.*"

    system_content += "\n\n" + _build_current_time_context(datetime.now().astimezone()).strip()

    return [
        {
            "role": "system",
            "content": system_content,
        },
        *messages,
    ]
