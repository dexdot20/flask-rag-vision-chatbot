from __future__ import annotations

import ast
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import logging
import os
import re
import string
from logging.handlers import RotatingFileHandler
from uuid import uuid4

from canvas_service import (
    build_canvas_document_context_result,
    build_canvas_tool_result,
    clear_canvas,
    create_canvas_document,
    create_canvas_runtime_state,
    delete_canvas_document,
    delete_canvas_lines,
    get_canvas_runtime_active_document_id,
    get_canvas_runtime_documents,
    insert_canvas_lines,
    replace_canvas_lines,
    rewrite_canvas_document,
    scroll_canvas_document,
    scale_canvas_char_limit,
)
from project_workspace_service import (
    bulk_update_workspace_files,
    build_project_plan,
    create_directory as workspace_create_directory,
    create_file as workspace_create_file,
    create_project_workflow_runtime_state,
    create_project_scaffold,
    create_workspace_runtime_state,
    get_workspace_file_history,
    get_project_workflow,
    list_dir as workspace_list_dir,
    preview_workspace_changes,
    redo_workspace_file_change,
    update_project_workflow,
    read_file as workspace_read_file,
    search_files as workspace_search_files,
    undo_workspace_file_change,
    update_file as workspace_update_file,
    validate_project_workspace,
    write_project_tree,
)
from config import (
    AGENT_CONTEXT_COMPACTION_KEEP_RECENT_ROUNDS,
    AGENT_CONTEXT_COMPACTION_THRESHOLD,
    AGENT_TOOL_RESULT_TRANSCRIPT_MAX_CHARS,
    AGENT_TRACE_LOG_PATH,
    FETCH_RAW_TOOL_RESULT_MAX_TEXT_CHARS,
    FETCH_SUMMARY_MAX_CHARS,
    FETCH_SUMMARY_TOKEN_THRESHOLD,
    PROMPT_MAX_INPUT_TOKENS,
    RAG_SEARCH_DEFAULT_TOP_K,
    RAG_TOOL_RESULT_MAX_TEXT_CHARS,
    RAG_TOOL_RESULT_SUMMARY_MAX_CHARS,
    client,
)
from db import (
    MESSAGE_USAGE_BREAKDOWN_PROTECTED_KEYS,
    MESSAGE_USAGE_BREAKDOWN_REDUCTION_ORDER,
    append_to_scratchpad,
    get_rag_source_types,
    read_image_asset_bytes,
    replace_scratchpad,
)
from rag_service import get_exact_tool_memory_match, search_knowledge_base_tool, search_tool_memory, upsert_tool_memory_result
from tool_registry import TOOL_SPEC_BY_NAME, get_openai_tool_specs
from token_utils import estimate_text_tokens
from vision import answer_image_question
from web_tools import (
    fetch_url_tool,
    search_news_ddgs_tool,
    search_news_google_tool,
    search_web_tool,
)

FINAL_ANSWER_ERROR_TEXT = "The model returned an invalid tool instruction and no final answer could be produced."
FINAL_ANSWER_MISSING_TEXT = "The model did not produce a final answer in assistant content."
CONTEXT_OVERFLOW_RECOVERY_ERROR_TEXT = (
    "Context window is full and cannot be compacted further. "
    "Try starting a new conversation, disabling RAG or large canvas content, or reducing the request size."
)
MISSING_FINAL_ANSWER_MARKER = "[INSTRUCTION: MISSING FINAL ANSWER"
TOOL_EXECUTION_RESULTS_MARKER = "[TOOL EXECUTION RESULTS]"
REASONING_REPLAY_MARKER = "[AGENT REASONING CONTEXT]"
MAX_REASONING_REPLAY_ENTRIES = 2
MAX_REASONING_REPLAY_CHARS = 4_000
CANVAS_TOOL_NAMES = {
    "expand_canvas_document",
    "scroll_canvas_document",
    "create_canvas_document",
    "rewrite_canvas_document",
    "replace_canvas_lines",
    "insert_canvas_lines",
    "delete_canvas_lines",
    "delete_canvas_document",
    "clear_canvas",
}
CANVAS_MUTATION_TOOL_NAMES = {
    "create_canvas_document",
    "rewrite_canvas_document",
    "replace_canvas_lines",
    "insert_canvas_lines",
    "delete_canvas_lines",
    "delete_canvas_document",
    "clear_canvas",
}
CANVAS_STREAM_OPEN_TOOL_NAMES = {
    "create_canvas_document",
    "rewrite_canvas_document",
    "expand_canvas_document",
}
CANVAS_STREAM_CONTENT_TOOL_NAMES = {
    "create_canvas_document",
    "rewrite_canvas_document",
}
WEB_TOOL_NAMES = {
    "search_web",
    "fetch_url",
    "search_news_ddgs",
    "search_news_google",
}
PARALLEL_SAFE_TOOL_NAMES = WEB_TOOL_NAMES | {
    "image_explain",
}
INPUT_BREAKDOWN_KEYS = (
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
SYSTEM_BREAKDOWN_SECTION_KEY_BY_HEADING = {
    "## Scratchpad (AI Persistent Memory)": "scratchpad",
    "## Tool Execution History": "tool_trace",
    "## Tool Memory": "tool_memory",
    "## Knowledge Base": "rag_context",
    "## Canvas Project Manifest": "canvas",
    "## Canvas Relationship Map": "canvas",
    "## Active Canvas Document": "canvas",
    "## Other Canvas Documents": "canvas",
    "## Available Tools": "tool_specs",
}
SYSTEM_BREAKDOWN_REDUCTION_ORDER = MESSAGE_USAGE_BREAKDOWN_REDUCTION_ORDER
_AGENT_TRACE_LOGGER = None


def _get_agent_trace_logger():
    global _AGENT_TRACE_LOGGER
    if _AGENT_TRACE_LOGGER is not None:
        return _AGENT_TRACE_LOGGER

    logger = logging.getLogger("chatbot.agent.trace")
    if not logger.handlers:
        log_dir = os.path.dirname(AGENT_TRACE_LOG_PATH)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        handler = RotatingFileHandler(AGENT_TRACE_LOG_PATH, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    _AGENT_TRACE_LOGGER = logger
    return logger


def get_model_pricing(model_id: str) -> dict:
    pricing = {
        "deepseek-chat": {"input": 0.28, "output": 0.42},
        "deepseek-reasoner": {"input": 0.28, "output": 0.42},
    }
    return pricing.get(model_id, {"input": 0, "output": 0})


def _empty_input_breakdown() -> dict[str, int]:
    return {key: 0 for key in INPUT_BREAKDOWN_KEYS}


def _estimate_text_tokens(text: str) -> int:
    return estimate_text_tokens(text)


def _estimate_serialized_tokens(value) -> int:
    if value in (None, "", [], {}):
        return 0
    try:
        serialized = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        serialized = str(value)
    return _estimate_text_tokens(serialized)


def _estimate_message_wrapper_tokens(role: str, *, include_tool_calls: bool = False) -> int:
    payload = {
        "role": str(role or ""),
        "content": "",
    }
    if include_tool_calls:
        payload["tool_calls"] = []
    return _estimate_serialized_tokens(payload)


def _estimate_request_tools_tokens(request_tools: list[dict] | None) -> int:
    if not request_tools:
        return 0
    return _estimate_serialized_tokens({"tools": request_tools, "tool_choice": "auto"})


def _distribute_overhead_tokens(
    breakdown: dict[str, int],
    overhead_tokens: int,
    recipients: tuple[str, ...],
) -> dict[str, int]:
    remaining = max(0, int(overhead_tokens or 0))
    if remaining <= 0:
        return breakdown

    target_keys = [key for key in recipients if breakdown.get(key, 0) > 0]
    if not target_keys and recipients:
        target_keys = [recipients[0]]
    if not target_keys:
        target_keys = ["core_instructions"]

    weighted_total = sum(max(0, int(breakdown.get(key, 0))) for key in target_keys)
    if weighted_total <= 0:
        breakdown[target_keys[0]] = breakdown.get(target_keys[0], 0) + remaining
        return breakdown

    for index, key in enumerate(target_keys):
        if remaining <= 0:
            break
        if index == len(target_keys) - 1:
            share = remaining
        else:
            weight = max(0, int(breakdown.get(key, 0)))
            share = min(remaining, int((overhead_tokens * weight) / weighted_total))
        breakdown[key] = breakdown.get(key, 0) + share
        remaining -= share

    return breakdown


def _rebalance_breakdown_to_total(breakdown: dict[str, int], total_tokens: int) -> dict[str, int]:
    adjusted = {key: max(0, int(value)) for key, value in breakdown.items() if value and value > 0}
    current_total = sum(adjusted.values())
    if current_total < total_tokens:
        adjusted["core_instructions"] = adjusted.get("core_instructions", 0) + (total_tokens - current_total)
        return adjusted

    overflow = current_total - total_tokens
    if overflow <= 0:
        return adjusted

    for key in SYSTEM_BREAKDOWN_REDUCTION_ORDER:
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


def _align_breakdown_to_provider_total(breakdown: dict[str, int], total_tokens: int) -> dict[str, int]:
    adjusted = {key: max(0, int(value)) for key, value in breakdown.items() if key in INPUT_BREAKDOWN_KEYS and value and value > 0}
    target_total = max(0, int(total_tokens or 0))
    current_total = sum(adjusted.values())
    if current_total < target_total:
        adjusted["unknown_provider_overhead"] = adjusted.get("unknown_provider_overhead", 0) + (target_total - current_total)
        return adjusted

    overflow = current_total - target_total
    if overflow <= 0:
        return adjusted

    protected_floor_keys = set()
    if target_total > 0:
        protected_candidates = [key for key in MESSAGE_USAGE_BREAKDOWN_PROTECTED_KEYS if adjusted.get(key, 0) > 0]
        protected_floor_keys = set(protected_candidates[: min(len(protected_candidates), target_total)])

    for key in SYSTEM_BREAKDOWN_REDUCTION_ORDER:
        if overflow <= 0:
            break
        floor = 1 if key in protected_floor_keys else 0
        available = adjusted.get(key, 0) - floor
        if available <= 0:
            continue
        reduction = min(available, overflow)
        adjusted[key] = available - reduction + floor
        overflow -= reduction

    if overflow > 0:
        for key, available in sorted(adjusted.items(), key=lambda item: item[1], reverse=True):
            if overflow <= 0:
                break
            floor = 1 if key in protected_floor_keys else 0
            reducible = available - floor
            if reducible <= 0:
                continue
            reduction = min(reducible, overflow)
            adjusted[key] = available - reduction
            overflow -= reduction

    return {key: value for key, value in adjusted.items() if value > 0}


def _estimate_system_message_breakdown(content: str, total_tokens: int) -> dict[str, int]:
    section_matches = list(re.finditer(r"^## [^\n]+", content, flags=re.MULTILINE))
    if not section_matches:
        return {"core_instructions": total_tokens}

    breakdown: dict[str, int] = {}
    cursor = 0
    for index, match in enumerate(section_matches):
        start = match.start()
        end = section_matches[index + 1].start() if index + 1 < len(section_matches) else len(content)
        if start > cursor:
            prefix = content[cursor:start]
            prefix_tokens = _estimate_text_tokens(prefix)
            if prefix_tokens > 0:
                breakdown["core_instructions"] = breakdown.get("core_instructions", 0) + prefix_tokens

        section_text = content[start:end]
        section_tokens = _estimate_text_tokens(section_text)
        if section_tokens > 0:
            section_heading = match.group(0).strip()
            section_key = SYSTEM_BREAKDOWN_SECTION_KEY_BY_HEADING.get(section_heading, "core_instructions")
            breakdown[section_key] = breakdown.get(section_key, 0) + section_tokens
        cursor = end

    return _rebalance_breakdown_to_total(breakdown, total_tokens)


def _estimate_message_breakdown(message: dict) -> dict[str, int]:
    role = str(message.get("role") or "").strip()
    content = str(message.get("content") or "")
    total_tokens = _estimate_text_tokens(content)
    if total_tokens <= 0 and role != "assistant":
        return {}

    if role == "user":
        breakdown = {"user_messages": total_tokens}
        return _distribute_overhead_tokens(breakdown, _estimate_message_wrapper_tokens(role), ("user_messages",))
    if role == "assistant":
        breakdown = {}
        if total_tokens > 0:
            breakdown["assistant_history"] = total_tokens
        tool_calls = message.get("tool_calls") if isinstance(message.get("tool_calls"), list) else []
        tool_call_tokens = _estimate_serialized_tokens(tool_calls)
        if tool_call_tokens > 0:
            breakdown["assistant_tool_calls"] = tool_call_tokens
        return _distribute_overhead_tokens(
            breakdown,
            _estimate_message_wrapper_tokens(role, include_tool_calls=bool(tool_calls)),
            ("assistant_history", "assistant_tool_calls"),
        )
    if role == "tool":
        tool_call_id = str(message.get("tool_call_id") or "").strip()
        payload_tokens = total_tokens
        if tool_call_id:
            payload_tokens += _estimate_serialized_tokens({"tool_call_id": tool_call_id})
        if payload_tokens <= 0:
            return {}
        breakdown = {"tool_results": payload_tokens}
        return _distribute_overhead_tokens(breakdown, _estimate_message_wrapper_tokens(role), ("tool_results",))
    if role != "system":
        breakdown = {"core_instructions": total_tokens}
        return _distribute_overhead_tokens(breakdown, _estimate_message_wrapper_tokens(role), ("core_instructions",))

    # Classify system messages by their distinctive markers
    if content.startswith(TOOL_EXECUTION_RESULTS_MARKER):
        return {"tool_results": total_tokens}
    if content.startswith(REASONING_REPLAY_MARKER):
        return {"internal_state": total_tokens}
    if content.startswith("[AGENT WORKING MEMORY]"):
        return {"internal_state": total_tokens}
    if content.startswith("[INSTRUCTION: FINAL ANSWER REQUIRED]"):
        return {"core_instructions": total_tokens}
    if content.startswith("[INSTRUCTION: MISSING FINAL ANSWER"):
        return {"core_instructions": total_tokens}

    breakdown = _estimate_system_message_breakdown(content, total_tokens) or {"core_instructions": total_tokens}
    return _distribute_overhead_tokens(
        breakdown,
        _estimate_message_wrapper_tokens(role),
        tuple(key for key, value in breakdown.items() if value > 0) or ("core_instructions",),
    )


def _estimate_input_breakdown(
    messages_to_send: list[dict],
    *,
    provider_prompt_tokens: int | None = None,
    request_tools: list[dict] | None = None,
) -> tuple[dict[str, int], int, int]:
    breakdown = _empty_input_breakdown()
    for message in messages_to_send:
        for key, value in _estimate_message_breakdown(message).items():
            if key in breakdown and value > 0:
                breakdown[key] += value

    tool_schema_tokens = _estimate_request_tools_tokens(request_tools)
    if tool_schema_tokens > 0:
        breakdown["tool_specs"] += tool_schema_tokens

    measured_total = sum(breakdown.values())
    if provider_prompt_tokens is None:
        return breakdown, measured_total, tool_schema_tokens

    aligned_breakdown = _align_breakdown_to_provider_total(breakdown, provider_prompt_tokens)
    return aligned_breakdown, max(0, int(provider_prompt_tokens or 0)), tool_schema_tokens


def _estimate_messages_tokens(messages_to_send: list[dict]) -> int:
    return _estimate_input_breakdown(messages_to_send)[1]


def _get_model_call_input_tokens(call: dict) -> int:
    if not isinstance(call, dict):
        return 0

    prompt_tokens = call.get("prompt_tokens")
    if isinstance(prompt_tokens, (int, float)):
        return max(0, int(prompt_tokens))

    estimated_input_tokens = call.get("estimated_input_tokens")
    if isinstance(estimated_input_tokens, (int, float)):
        return max(0, int(estimated_input_tokens))

    return 0


def _summarize_model_call_usage(model_calls: list[dict], fallback_input_tokens: int = 0) -> dict[str, int]:
    max_input_tokens_per_call = 0
    for call in model_calls:
        max_input_tokens_per_call = max(max_input_tokens_per_call, _get_model_call_input_tokens(call))

    if max_input_tokens_per_call <= 0:
        max_input_tokens_per_call = max(0, int(fallback_input_tokens or 0))

    return {
        "max_input_tokens_per_call": max_input_tokens_per_call,
    }


def _is_context_overflow_error(error_str: str) -> bool:
    normalized = str(error_str or "").strip().lower()
    if not normalized:
        return False
    if "rate_limit" in normalized or re.search(r"\b429\b", normalized):
        return False

    known_phrases = (
        "context_length_exceeded",
        "maximum context length",
        "reduce the length",
        "request too large",
        "prompt is too long",
        "input is too long",
        "too many tokens",
        "context window",
        "context is full",
        "max_tokens",
    )
    if any(phrase in normalized for phrase in known_phrases):
        return True
    return "token" in normalized and ("exceed" in normalized or "too long" in normalized)


def _normalize_tool_args_for_cache(value):
    if isinstance(value, dict):
        return {str(key): _normalize_tool_args_for_cache(value[key]) for key in sorted(value.keys())}
    if isinstance(value, list):
        return [_normalize_tool_args_for_cache(item) for item in value]
    if isinstance(value, str):
        return value.strip()
    return value


def build_tool_cache_key(tool_name: str, tool_args: dict) -> str:
    normalized_args = _normalize_tool_args_for_cache(tool_args if isinstance(tool_args, dict) else {})
    payload = json.dumps(normalized_args, ensure_ascii=False, sort_keys=True)
    digest = hashlib.sha1(f"{tool_name}|{payload}".encode("utf-8")).hexdigest()
    return f"tool-cache:{digest}"


def _clean_tool_text(text: str, limit: int | None = None) -> str:
    cleaned = str(text or "").strip()
    if limit and len(cleaned) > limit:
        return cleaned[:limit].rstrip() + "…"
    return cleaned


def _has_missing_final_answer_instruction(messages: list[dict]) -> bool:
    return any(MISSING_FINAL_ANSWER_MARKER in str(message.get("content") or "") for message in messages)


def _is_tool_execution_result_message(message: dict) -> bool:
    return str(message.get("role") or "").strip() == "system" and str(message.get("content") or "").startswith(
        TOOL_EXECUTION_RESULTS_MARKER
    )


def _iter_agent_exchange_blocks(messages: list[dict]) -> list[dict]:
    blocks: list[dict] = []
    index = 0
    exchange_index = 0
    while index < len(messages):
        message = messages[index]
        role = str(message.get("role") or "").strip()
        tool_calls = message.get("tool_calls") if isinstance(message, dict) else None
        if role == "assistant" and isinstance(tool_calls, list) and tool_calls:
            exchange_index += 1
            block_messages = [message]
            index += 1
            while index < len(messages):
                candidate = messages[index]
                candidate_role = str(candidate.get("role") or "").strip()
                if candidate_role == "tool" or _is_tool_execution_result_message(candidate):
                    block_messages.append(candidate)
                    index += 1
                    continue
                break
            blocks.append({"type": "exchange", "step_index": exchange_index, "messages": block_messages})
            continue

        block_type = "system_prefix" if role == "system" and not blocks else "passthrough"
        blocks.append({"type": block_type, "messages": [message]})
        index += 1
    return blocks


def _flatten_agent_exchange_blocks(blocks: list[dict]) -> list[dict]:
    flattened: list[dict] = []
    for block in blocks:
        flattened.extend(block.get("messages") or [])
    return flattened


def _merge_adjacent_user_messages(messages: list[dict]) -> list[dict] | None:
    merged_messages: list[dict] = []
    buffered_user_contents: list[str] = []
    merged_any = False

    def flush_user_buffer():
        nonlocal merged_any
        if not buffered_user_contents:
            return
        merged_content = "\n\n".join(content for content in buffered_user_contents if content)
        if len(buffered_user_contents) > 1:
            merged_any = True
        merged_messages.append({"role": "user", "content": merged_content})
        buffered_user_contents.clear()

    for message in messages:
        if str(message.get("role") or "").strip() == "user":
            buffered_user_contents.append(str(message.get("content") or "").strip())
            continue
        flush_user_buffer()
        merged_messages.append(message)

    flush_user_buffer()
    return merged_messages if merged_any else None


def _extract_compaction_assistant_intent(message: dict) -> str:
    return _clean_tool_text(message.get("content") or "", limit=140)


def _extract_compaction_tool_call_preview(tool_call: dict) -> str:
    function = tool_call.get("function") or {}
    tool_name = str(function.get("name") or "").strip() or "tool"
    raw_arguments = function.get("arguments")
    parsed_arguments = _parse_json_like_value(raw_arguments)
    arguments = parsed_arguments if isinstance(parsed_arguments, dict) else {}

    if tool_name in {"search_web", "search_news_ddgs", "search_news_google"}:
        queries = arguments.get("queries")
        if isinstance(queries, list):
            preview = ", ".join(str(item).strip() for item in queries if str(item).strip())
            if preview:
                return f"{tool_name}: {_clean_tool_text(preview, limit=120)}"
    if tool_name == "fetch_url":
        url = str(arguments.get("url") or "").strip()
        if url:
            return f"{tool_name}: {_clean_tool_text(url, limit=140)}"
    if tool_name in {"search_knowledge_base", "search_tool_memory"}:
        query = str(arguments.get("query") or "").strip()
        if query:
            return f"{tool_name}: {_clean_tool_text(query, limit=120)}"

    scalar_parts: list[str] = []
    for key, value in list(arguments.items())[:3]:
        if isinstance(value, (str, int, float)):
            text = _clean_tool_text(value, limit=60)
            if text:
                scalar_parts.append(f"{key}={text}")
    if scalar_parts:
        return f"{tool_name}: " + ", ".join(scalar_parts)
    return tool_name


def _extract_compaction_tool_result_preview(message: dict) -> str:
    content = str(message.get("content") or "").strip()
    if not content:
        return ""

    parsed = _parse_json_like_value(content)
    if isinstance(parsed, dict):
        error = _clean_tool_text(parsed.get("error") or "", limit=120)
        if error:
            return f"error: {error}"
        summary = _clean_tool_text(parsed.get("summary") or parsed.get("title") or "", limit=120)
        if summary:
            return summary
        value = _clean_tool_text(parsed.get("content") or parsed.get("value") or "", limit=120)
        if value:
            return value

    normalized = content.replace(TOOL_EXECUTION_RESULTS_MARKER, "").strip()
    for line in normalized.splitlines():
        cleaned = _clean_tool_text(line, limit=120)
        if not cleaned:
            continue
        if cleaned.lower().startswith(("url:", "title:")):
            continue
        return cleaned
    return _clean_tool_text(normalized, limit=120)


def _count_exchange_blocks(messages: list[dict]) -> int:
    return sum(1 for block in _iter_agent_exchange_blocks(messages) if block.get("type") == "exchange")


def _compact_exchange_to_message(block: dict) -> dict:
    tool_previews: list[str] = []
    result_parts: list[str] = []
    assistant_intent = ""
    for message in block.get("messages") or []:
        role = str(message.get("role") or "").strip()
        if role == "assistant":
            assistant_intent = assistant_intent or _extract_compaction_assistant_intent(message)
            for tool_call in message.get("tool_calls") or []:
                preview = _extract_compaction_tool_call_preview(tool_call)
                if preview and preview not in tool_previews:
                    tool_previews.append(preview)
        elif role == "tool" or _is_tool_execution_result_message(message):
            content = _extract_compaction_tool_result_preview(message)
            if content:
                result_parts.append(content)

    parts = [f"[Context: compacted tool step {block.get('step_index') or '?'}]"]
    if assistant_intent:
        parts.append(f"Assistant intent: {assistant_intent}")
    if tool_previews:
        parts.append("Actions:\n- " + "\n- ".join(tool_previews[:4]))
    if result_parts:
        parts.append("Outcomes:\n- " + "\n- ".join(result_parts[:3]))
    return {"role": "user", "content": "\n".join(parts)}


def _try_compact_messages(messages: list[dict], budget: int, keep_recent: int = 2) -> list[dict] | None:
    if not isinstance(messages, list):
        return None

    blocks = _iter_agent_exchange_blocks(messages)
    exchange_positions = [index for index, block in enumerate(blocks) if block.get("type") == "exchange"]
    if not exchange_positions:
        return _merge_adjacent_user_messages(messages)

    keep_recent = max(0, int(keep_recent))
    compactable_positions = exchange_positions[:-keep_recent] if keep_recent else exchange_positions[:]
    if not compactable_positions:
        return None

    working_blocks = [{**block, "messages": list(block.get("messages") or [])} for block in blocks]
    best_messages: list[dict] | None = None
    for position in compactable_positions:
        block = working_blocks[position]
        block["messages"] = [_compact_exchange_to_message(block)]
        best_messages = _flatten_agent_exchange_blocks(working_blocks)
        merged_user_messages = _merge_adjacent_user_messages(best_messages)
        if merged_user_messages is not None:
            best_messages = merged_user_messages
        if _estimate_messages_tokens(best_messages) <= max(1, int(budget)):
            return best_messages
    return best_messages


def _serialize_for_log(value, depth: int = 0):
    if depth >= 2:
        if isinstance(value, str):
            return _clean_tool_text(value, limit=300)
        if isinstance(value, (int, float, bool)) or value is None:
            return value
        return _clean_tool_text(str(value), limit=300)

    if isinstance(value, str):
        return _clean_tool_text(value, limit=800)
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        items = list(value.items())[:20]
        return {str(key): _serialize_for_log(item, depth + 1) for key, item in items}
    if isinstance(value, (list, tuple)):
        return [_serialize_for_log(item, depth + 1) for item in list(value)[:20]]
    return _clean_tool_text(str(value), limit=800)


def _summarize_messages_for_log(messages_to_send: list[dict]) -> list[dict]:
    summary = []
    for message in messages_to_send[:20]:
        role = str(message.get("role") or "").strip()
        content = str(message.get("content") or "")
        context_type = ""
        if role == "system":
            try:
                payload = json.loads(content)
            except Exception:
                payload = None
            if isinstance(payload, dict):
                context_type = str(payload.get("context_type") or "").strip()
        summary.append(
            {
                "role": role,
                "context_type": context_type or None,
                "content_excerpt": _clean_tool_text(content, limit=240),
            }
        )
    return summary


def _trace_agent_event(event: str, **fields):
    payload = {"event": event}
    for key, value in fields.items():
        payload[key] = _serialize_for_log(value)
    try:
        _get_agent_trace_logger().info(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    except Exception:
        return


def _normalize_fetch_token_threshold(value) -> int:
    try:
        threshold = int(value)
    except (TypeError, ValueError):
        threshold = FETCH_SUMMARY_TOKEN_THRESHOLD
    return max(1, threshold)


def _normalize_fetch_clip_aggressiveness(value) -> int:
    try:
        aggressiveness = int(value)
    except (TypeError, ValueError):
        aggressiveness = 50
    return max(0, min(100, aggressiveness))


def _build_fetch_clipped_text(result: dict, token_threshold: int, clip_aggressiveness: int) -> tuple[str, int]:
    raw_content = _clean_tool_text(result.get("content") or "")
    token_estimate = _estimate_text_tokens(raw_content)
    if not raw_content:
        return "", token_estimate

    if token_estimate <= token_threshold:
        return raw_content, token_estimate

    clip_ratio = min(1.0, token_threshold / max(token_estimate, 1))
    preserve_multiplier = 1.8 - (_normalize_fetch_clip_aggressiveness(clip_aggressiveness) / 100) * 1.0
    target_chars = max(2000, min(FETCH_SUMMARY_MAX_CHARS, int(len(raw_content) * clip_ratio * preserve_multiplier)))
    clipped_content = _clean_tool_text(raw_content, limit=target_chars)
    return clipped_content or raw_content, token_estimate


def _build_fetch_diagnostic_fields(result: dict) -> dict:
    if not isinstance(result, dict):
        return {}

    content = _clean_tool_text(result.get("content") or "")
    warning = _clean_tool_text(result.get("fetch_warning") or "", limit=400)
    error = _clean_tool_text(result.get("error") or "", limit=400)
    status = result.get("status")
    status_label = f"HTTP {status}" if isinstance(status, int) and status > 0 else None

    if error:
        outcome = "error"
        detail = error
    elif not content:
        outcome = "empty_content"
        detail = warning or "The request completed but no extractable page content was returned."
    elif result.get("partial_content"):
        outcome = "partial_content"
        detail = warning or "Only partial page content could be recovered."
    elif warning:
        outcome = "limited_content"
        detail = warning
    else:
        outcome = "success"
        detail = "The page was fetched successfully and extractable content was returned."

    if status_label and detail:
        detail = f"{status_label}. {detail}"
    elif status_label:
        detail = status_label

    return {
        "fetch_attempted": True,
        "fetch_outcome": outcome,
        "content_char_count": len(content),
        "same_url_retry_recommended": False,
        "fetch_diagnostic": (
            f"fetch_url already attempted this URL. Outcome: {detail} "
            "Do not repeat the same fetch_url call for the same URL unless you have a concrete new reason, a different URL, or the user explicitly asks for a retry."
        ).strip(),
    }


def _summarize_fetch_result(result: dict, fallback_url: str = "") -> str:
    if not isinstance(result, dict):
        return fallback_url[:60]

    error = _clean_tool_text(result.get("error") or "", limit=180)
    warning = _clean_tool_text(result.get("fetch_warning") or "", limit=180)
    title = _clean_tool_text(result.get("title") or "", limit=120)
    url = _clean_tool_text(result.get("url") or fallback_url or "", limit=120)

    if error:
        return f"Fetch failed: {error}"
    if result.get("partial_content"):
        return f"Partial page content extracted: {title or url or 'page'}"
    if warning:
        return f"Limited page content extracted: {title or url or 'page'}"
    if result.get("content"):
        return f"Page content extracted: {title or url or 'page'}"
    return f"No extractable page content: {title or url or 'page'}"


def _prepare_fetch_result_for_model(
    result: dict,
    fetch_url_token_threshold: int | None = None,
    fetch_url_clip_aggressiveness: int | None = None,
) -> dict:
    if not isinstance(result, dict):
        return result

    content = _clean_tool_text(result.get("content") or "")
    prepared = dict(result)
    prepared["cleanup_applied"] = True
    prepared["content_token_estimate"] = _estimate_text_tokens(content)
    prepared.update(_build_fetch_diagnostic_fields(prepared))
    if not content or prepared.get("error"):
        return prepared

    prepared["content"] = content
    prepared["content_mode"] = "cleaned_full_text"

    token_threshold = _normalize_fetch_token_threshold(fetch_url_token_threshold)
    if prepared["content_token_estimate"] <= token_threshold:
        return prepared

    clip_aggressiveness = _normalize_fetch_clip_aggressiveness(fetch_url_clip_aggressiveness)
    clipped_text, token_estimate = _build_fetch_clipped_text(prepared, token_threshold, clip_aggressiveness)
    if not clipped_text or clipped_text == content:
        return prepared

    prepared["content"] = clipped_text
    prepared["content_mode"] = "clipped_text"
    prepared["summary_notice"] = (
        "Content was cleaned and clipped because the fetched page exceeded the token threshold. "
        "The leading portion is preserved to keep the original page wording intact."
    )
    prepared["content_token_estimate"] = token_estimate
    prepared["raw_content_available"] = True
    return prepared


def _prepare_tool_result_for_transcript(
    tool_name: str,
    result,
    fetch_url_token_threshold: int | None = None,
    fetch_url_clip_aggressiveness: int | None = None,
):
    if tool_name == "fetch_url" and isinstance(result, dict):
        return _prepare_fetch_result_for_model(
            result,
            fetch_url_token_threshold=fetch_url_token_threshold,
            fetch_url_clip_aggressiveness=fetch_url_clip_aggressiveness,
        )
    serialized = _serialize_tool_message_content(result)
    if len(serialized) > AGENT_TOOL_RESULT_TRANSCRIPT_MAX_CHARS:
        clipped = serialized[:AGENT_TOOL_RESULT_TRANSCRIPT_MAX_CHARS].rstrip() + "…"
        return f"{clipped} [CLIPPED: original {len(serialized)} chars]"
    return result


def _coerce_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, str):
                if item:
                    parts.append(item)
                continue
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
        return "".join(parts)
    return str(value)


def _extract_reasoning_and_content(message) -> tuple[str, str]:
    reasoning_text = _coerce_text(getattr(message, "reasoning_content", "")).strip()
    content_text = _coerce_text(getattr(message, "content", "")).strip()
    return reasoning_text, content_text


def _extract_stream_delta_texts(chunk) -> tuple[str, str]:
    if not getattr(chunk, "choices", None):
        return "", ""
    delta = getattr(chunk.choices[0], "delta", None)
    if delta is None:
        return "", ""
    reasoning_text = _coerce_text(getattr(delta, "reasoning_content", ""))
    content_text = _coerce_text(getattr(delta, "content", ""))
    return reasoning_text, content_text


def _read_api_field(value, key: str, default=None):
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _parse_tool_call_arguments(arguments_text: str, label: str) -> tuple[dict | None, str | None]:
    raw_arguments = str(arguments_text or "").strip()
    if not raw_arguments:
        return {}, None
    try:
        parsed_arguments = json.loads(raw_arguments)
    except json.JSONDecodeError as exc:
        return None, f"Invalid tool arguments JSON for {label}: {exc.msg}"
    if not isinstance(parsed_arguments, dict):
        return None, f"Tool arguments for {label} must be an object"
    return parsed_arguments, None


def _extract_native_tool_calls(message) -> tuple[list[dict] | None, str | None]:
    raw_tool_calls = _read_api_field(message, "tool_calls") or []
    if not raw_tool_calls:
        return None, None

    normalized_calls = []
    for index, raw_call in enumerate(raw_tool_calls, start=1):
        function = _read_api_field(raw_call, "function")
        tool_name = str(_read_api_field(function, "name") or "").strip()
        if not tool_name:
            return None, f"tool_calls[{index}] is missing a tool name"

        arguments_text = _coerce_text(_read_api_field(function, "arguments", ""))
        tool_args, parse_error = _parse_tool_call_arguments(arguments_text, tool_name)
        if parse_error:
            return None, parse_error

        normalized_calls.append(
            {
                "id": str(_read_api_field(raw_call, "id") or f"tool-call-{index}"),
                "name": tool_name,
                "arguments": tool_args or {},
            }
        )
    return normalized_calls, None


def _merge_stream_tool_call_delta(tool_call_parts: list[dict], delta) -> None:
    raw_tool_calls = _read_api_field(delta, "tool_calls") or []
    for fallback_index, raw_call in enumerate(raw_tool_calls):
        index_value = _read_api_field(raw_call, "index", fallback_index)
        try:
            index = max(0, int(index_value))
        except (TypeError, ValueError):
            index = fallback_index

        while len(tool_call_parts) <= index:
            tool_call_parts.append({"id": "", "name": "", "arguments_parts": []})

        entry = tool_call_parts[index]
        call_id = _read_api_field(raw_call, "id")
        if call_id:
            entry["id"] = str(call_id)

        function = _read_api_field(raw_call, "function")
        name_part = str(_read_api_field(function, "name") or "")
        if name_part:
            if not entry["name"]:
                entry["name"] = name_part
            elif not entry["name"].endswith(name_part):
                entry["name"] += name_part

        arguments_part = _coerce_text(_read_api_field(function, "arguments", ""))
        if arguments_part:
            entry["arguments_parts"].append(arguments_part)


def _extract_partial_json_string_value(arguments_text: str, field_name: str) -> str | None:
    raw_arguments = str(arguments_text or "")
    raw_field_name = str(field_name or "").strip()
    if not raw_arguments or not raw_field_name:
        return None

    depth = 0
    in_string = False
    escape_next = False
    string_chars: list[str] = []
    value_start = None

    for index, char in enumerate(raw_arguments):
        if in_string:
            if escape_next:
                string_chars.append(char)
                escape_next = False
                continue
            if char == "\\":
                escape_next = True
                string_chars.append(char)
                continue
            if char == '"':
                in_string = False
                if depth == 1:
                    candidate_key = "".join(string_chars)
                    look_ahead = index + 1
                    while look_ahead < len(raw_arguments) and raw_arguments[look_ahead].isspace():
                        look_ahead += 1
                    if candidate_key == raw_field_name and look_ahead < len(raw_arguments) and raw_arguments[look_ahead] == ":":
                        look_ahead += 1
                        while look_ahead < len(raw_arguments) and raw_arguments[look_ahead].isspace():
                            look_ahead += 1
                        if look_ahead < len(raw_arguments) and raw_arguments[look_ahead] == '"':
                            value_start = look_ahead + 1
                            break
                string_chars = []
                continue
            string_chars.append(char)
            continue

        if char == '"':
            in_string = True
            escape_next = False
            string_chars = []
            continue
        if char == "{":
            depth += 1
            continue
        if char == "}" and depth > 0:
            depth -= 1

    if value_start is None:
        return None

    value_chars = []
    index = value_start
    while index < len(raw_arguments):
        char = raw_arguments[index]
        if char == '"':
            return "".join(value_chars)
        if char != "\\":
            value_chars.append(char)
            index += 1
            continue

        index += 1
        if index >= len(raw_arguments):
            break

        escape_char = raw_arguments[index]
        if escape_char in {'"', "\\", "/"}:
            value_chars.append(escape_char)
            index += 1
            continue
        if escape_char == "b":
            value_chars.append("\b")
            index += 1
            continue
        if escape_char == "f":
            value_chars.append("\f")
            index += 1
            continue
        if escape_char == "n":
            value_chars.append("\n")
            index += 1
            continue
        if escape_char == "r":
            value_chars.append("\r")
            index += 1
            continue
        if escape_char == "t":
            value_chars.append("\t")
            index += 1
            continue
        if escape_char == "u":
            hex_value = raw_arguments[index + 1:index + 5]
            if len(hex_value) < 4 or any(char not in string.hexdigits for char in hex_value):
                break
            value_chars.append(chr(int(hex_value, 16)))
            index += 5
            continue

        value_chars.append(escape_char)
        index += 1

    return "".join(value_chars)


def _build_streaming_canvas_tool_preview(tool_call_parts: list[dict]) -> dict | None:
    for raw_call in tool_call_parts:
        tool_name = str(raw_call.get("name") or "").strip()
        if tool_name not in CANVAS_STREAM_OPEN_TOOL_NAMES:
            continue

        arguments_text = "".join(raw_call.get("arguments_parts") or [])
        snapshot = {}
        for field_name in ("title", "format", "language", "path", "role"):
            value = _extract_partial_json_string_value(arguments_text, field_name)
            if value is not None:
                snapshot[field_name] = value

        content = None
        if tool_name in CANVAS_STREAM_CONTENT_TOOL_NAMES:
            content = _extract_partial_json_string_value(arguments_text, "content")

        return {
            "tool": tool_name,
            "snapshot": snapshot,
            "content": content,
        }
    return None


def _finalize_stream_tool_calls(tool_call_parts: list[dict]) -> tuple[list[dict] | None, str | None]:
    if not tool_call_parts:
        return None, None

    normalized_calls = []
    for index, raw_call in enumerate(tool_call_parts, start=1):
        tool_name = str(raw_call.get("name") or "").strip()
        if not tool_name:
            return None, f"tool_calls[{index}] is missing a tool name"

        arguments_text = "".join(raw_call.get("arguments_parts") or [])
        tool_args, parse_error = _parse_tool_call_arguments(arguments_text, tool_name)
        if parse_error:
            return None, parse_error

        normalized_calls.append(
            {
                "id": str(raw_call.get("id") or f"tool-call-{index}"),
                "name": tool_name,
                "arguments": tool_args or {},
            }
        )
    return normalized_calls, None


def _build_assistant_tool_call_message(content_text: str, tool_calls: list[dict]) -> dict:
    serialized_tool_calls = []
    for tool_call in tool_calls:
        serialized_tool_calls.append(
            {
                "id": str(tool_call.get("id") or ""),
                "type": "function",
                "function": {
                    "name": str(tool_call.get("name") or "").strip(),
                    "arguments": json.dumps(tool_call.get("arguments") or {}, ensure_ascii=False),
                },
            }
        )
    return {
        "role": "assistant",
        "content": str(content_text or ""),
        "tool_calls": serialized_tool_calls,
    }


def _serialize_tool_message_content(payload) -> str:
    if isinstance(payload, str):
        return payload
    try:
        return json.dumps(payload, ensure_ascii=False)
    except TypeError:
        return json.dumps({"value": str(payload)}, ensure_ascii=False)


def _validate_scalar_type(value, expected_type: str) -> bool:
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "object":
        return isinstance(value, dict)
    if expected_type == "array":
        return isinstance(value, list)
    return True


def _parse_json_like_value(value):
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str):
        return None

    text = value.strip()
    if not text:
        return None

    try:
        return json.loads(text)
    except Exception:
        try:
            return ast.literal_eval(text)
        except Exception:
            return None


def _coerce_clarification_question_item(raw_question):
    if isinstance(raw_question, dict):
        return raw_question
    if not isinstance(raw_question, str):
        return None

    text = raw_question.strip()
    if not text:
        return None

    parsed = _parse_json_like_value(text)
    if isinstance(parsed, dict):
        return parsed

    return {
        "label": text,
        "input_type": "text",
    }


def _validate_tool_arguments(tool_name: str, tool_args: dict) -> str | None:
    spec = TOOL_SPEC_BY_NAME.get(tool_name)
    if not spec:
        return f"Unknown tool: {tool_name}"
    if not isinstance(tool_args, dict):
        return f"Tool arguments for {tool_name} must be a JSON object"

    if tool_name == "append_scratchpad" and "notes" not in tool_args and "note" in tool_args:
        legacy_note = tool_args.pop("note")
        tool_args["notes"] = [legacy_note]

    schema = spec.get("parameters") or {}
    properties = schema.get("properties") or {}
    required = schema.get("required") or []

    for field_name in required:
        if field_name not in tool_args:
            return f"Missing required argument '{field_name}' for {tool_name}"

    for key, value in tool_args.items():
        property_schema = properties.get(key)
        if not property_schema:
            return f"Unexpected argument '{key}' for {tool_name}"
        
        expected_type = property_schema.get("type")
        
        if expected_type == "array" and isinstance(value, str):
            parsed_value = _parse_json_like_value(value)
            if isinstance(parsed_value, list):
                value = parsed_value
            else:
                value = [value]
            tool_args[key] = value
        elif expected_type == "integer" and isinstance(value, str):
            try:
                coerced_value = int(value.strip())
            except (TypeError, ValueError):
                coerced_value = value
            else:
                value = coerced_value
                tool_args[key] = value
        elif expected_type == "object" and isinstance(value, str):
            parsed_value = _parse_json_like_value(value)
            if isinstance(parsed_value, dict):
                value = parsed_value
                tool_args[key] = value

        if expected_type and not _validate_scalar_type(value, expected_type):
            return f"Invalid type for '{key}' in {tool_name}: expected {expected_type}"
        if expected_type == "array":
            item_schema = property_schema.get("items") or {}
            item_type = item_schema.get("type")
            normalized_items = []
            for item in value:
                normalized_item = item
                if item_type == "object" and isinstance(item, str):
                    if tool_name == "ask_clarifying_question":
                        normalized_item = _coerce_clarification_question_item(item)
                    else:
                        parsed_item = _parse_json_like_value(item)
                        if isinstance(parsed_item, dict):
                            normalized_item = parsed_item
                normalized_items.append(normalized_item)
            if normalized_items != value:
                value = normalized_items
                tool_args[key] = value
            if item_type and any(not _validate_scalar_type(item, item_type) for item in value):
                return f"Invalid array item type for '{key}' in {tool_name}: expected {item_type}"
            min_items = property_schema.get("minItems")
            max_items = property_schema.get("maxItems")
            if isinstance(min_items, int) and len(value) < min_items:
                return f"Argument '{key}' in {tool_name} requires at least {min_items} items"
            if isinstance(max_items, int) and len(value) > max_items:
                return f"Argument '{key}' in {tool_name} allows at most {max_items} items"
        if expected_type in {"string", "integer", "number"}:
            minimum = property_schema.get("minimum")
            maximum = property_schema.get("maximum")
            if minimum is not None and value < minimum:
                return f"Argument '{key}' in {tool_name} must be >= {minimum}"
            if maximum is not None and value > maximum:
                return f"Argument '{key}' in {tool_name} must be <= {maximum}"
        enum_values = property_schema.get("enum")
        if enum_values and value not in enum_values:
            return f"Argument '{key}' in {tool_name} must be one of: {', '.join(str(item) for item in enum_values)}"
    return None


def _build_final_answer_instruction() -> dict:
    return {
        "role": "system",
        "content": (
            "[INSTRUCTION: FINAL ANSWER REQUIRED]\n\n"
            "Tool execution budget is exhausted. Do not call more tools.\n"
            "Respond with the best possible final answer using the available context.\n"
            "Place the final answer in assistant content, not reasoning_content."
        ),
    }


def _build_minimal_final_answer_instruction() -> dict:
    return {
        "role": "system",
        "content": "[FINAL ANSWER ONLY]\nNo tools. Answer in assistant content only.",
    }


def _build_missing_final_answer_instruction() -> dict:
    return {
        "role": "system",
        "content": (
            "[INSTRUCTION: MISSING FINAL ANSWER — RETRY]\n\n"
            "You have not returned any final answer in assistant content yet.\n"
            "Continue and respond now using assistant content only.\n"
            "If you need tools, place only the tool_calls JSON in assistant content.\n"
            "Do not place the final answer or tool JSON in reasoning_content."
        ),
    }


def _build_tool_execution_result_message(transcript_results: list[dict]) -> dict | None:
    if not transcript_results:
        return None

    includes_fetch_results = any(str(item.get("tool_name") or "") == "fetch_url" for item in transcript_results)
    if not includes_fetch_results:
        return None

    parts = [
        f"{TOOL_EXECUTION_RESULTS_MARKER}\n",
        "**Fetch Guidance**: Use the retrieved page content as the source of truth. "
        "Do not repeat the same fetch_url call unless the user explicitly asks to refresh.\n",
    ]
    for item in transcript_results:
        tool_name = str(item.get("tool_name") or "unknown")
        ok = item.get("ok", False)
        summary = str(item.get("summary") or "").strip()
        status = "OK" if ok else "FAILED"
        line = f"- **{tool_name}** [{status}]"
        if summary:
            line += f": {summary}"
        parts.append(line)

    return {"role": "system", "content": "\n".join(parts)}


def _normalize_clarification_question(raw_question: dict, index: int) -> dict | None:
    raw_question = _coerce_clarification_question_item(raw_question)
    if not isinstance(raw_question, dict):
        return None

    question_id = str(raw_question.get("id") or raw_question.get("key") or f"question_{index}").strip()[:80]
    label = str(raw_question.get("label") or raw_question.get("question") or raw_question.get("prompt") or "").strip()

    input_type_aliases = {
        "": "",
        "text": "text",
        "string": "text",
        "free_text": "text",
        "single": "single_select",
        "select": "single_select",
        "single_select": "single_select",
        "single-choice": "single_select",
        "single_choice": "single_select",
        "multiple": "multi_select",
        "multi": "multi_select",
        "multiselect": "multi_select",
        "multi_select": "multi_select",
        "multi-choice": "multi_select",
        "multi_choice": "multi_select",
    }
    raw_input_type = str(raw_question.get("input_type") or raw_question.get("type") or "").strip().lower()
    input_type = input_type_aliases.get(raw_input_type, raw_input_type)
    if not question_id or not label:
        return None

    normalized = {
        "id": question_id,
        "label": label,
        "required": raw_question.get("required") is not False,
    }

    placeholder = str(raw_question.get("placeholder") or "").strip()
    if placeholder:
        normalized["placeholder"] = placeholder[:200]

    allow_free_text = raw_question.get("allow_free_text") is True or raw_question.get("allowFreeText") is True
    if allow_free_text:
        normalized["allow_free_text"] = True

    raw_options = raw_question.get("options") if isinstance(raw_question.get("options"), list) else []
    normalized_options = []
    for option in raw_options[:10]:
        if isinstance(option, str):
            label_text = option.strip()
            value_text = label_text
            description = ""
        elif isinstance(option, dict):
            label_text = str(option.get("label") or option.get("value") or "").strip()
            value_text = str(option.get("value") or option.get("label") or "").strip()
            description = str(option.get("description") or "").strip()
        else:
            continue
        if not label_text or not value_text:
            continue
        normalized_option = {
            "label": label_text[:120],
            "value": value_text[:120],
        }
        if description:
            normalized_option["description"] = description[:200]
        normalized_options.append(normalized_option)

    if input_type not in {"text", "single_select", "multi_select"}:
        input_type = "single_select" if normalized_options else "text"
    if input_type in {"single_select", "multi_select"} and not normalized_options:
        input_type = "text"

    normalized["input_type"] = input_type
    if normalized_options:
        normalized["options"] = normalized_options

    return normalized


def _normalize_clarification_payload(tool_args: dict) -> dict:
    raw_questions = tool_args.get("questions") if isinstance(tool_args.get("questions"), list) else []
    questions = []
    for index, raw_question in enumerate(raw_questions[:5], start=1):
        normalized_question = _normalize_clarification_question(raw_question, index)
        if normalized_question is not None:
            questions.append(normalized_question)

    if not questions:
        raise ValueError("ask_clarifying_question requires at least one valid question.")

    payload = {"questions": questions}
    intro = str(tool_args.get("intro") or "").strip()
    if intro:
        payload["intro"] = intro[:300]
    submit_label = str(tool_args.get("submit_label") or "").strip()
    if submit_label:
        payload["submit_label"] = submit_label[:80]
    return payload


def _build_clarification_text(payload: dict) -> str:
    questions = payload.get("questions") if isinstance(payload.get("questions"), list) else []
    lines = []
    intro = str(payload.get("intro") or "").strip()
    if intro:
        lines.append(intro)
    else:
        lines.append("I need a few details before I can answer well:")

    for index, question in enumerate(questions, start=1):
        if not isinstance(question, dict):
            continue
        lines.append(f"{index}. {str(question.get('label') or '').strip()}")
        options = question.get("options") if isinstance(question.get("options"), list) else []
        if options:
            option_values = []
            for option in options:
                if not isinstance(option, dict):
                    continue
                label = str(option.get("label") or option.get("value") or "").strip()
                if label:
                    option_values.append(label)
            if option_values:
                lines.append(f"   Options: {', '.join(option_values)}")

    return "\n".join(line for line in lines if line).strip()


def _get_canvas_runtime_state(runtime_state: dict) -> dict:
    return runtime_state.setdefault("canvas", create_canvas_runtime_state())


def _run_append_scratchpad(tool_args: dict, runtime_state: dict):
    del runtime_state
    notes = tool_args.get("notes") or tool_args.get("note", "")
    return append_to_scratchpad(notes)


def _run_replace_scratchpad(tool_args: dict, runtime_state: dict):
    del runtime_state
    return replace_scratchpad(tool_args.get("new_content", ""))


def _run_ask_clarifying_question(tool_args: dict, runtime_state: dict):
    del runtime_state
    payload = _normalize_clarification_payload(tool_args)
    return {
        "status": "needs_user_input",
        "clarification": payload,
        "text": _build_clarification_text(payload),
    }, "Awaiting user clarification"


def _run_image_explain(tool_args: dict, runtime_state: dict):
    del runtime_state
    image_id = str(tool_args.get("image_id") or "").strip()
    conversation_id = tool_args.get("conversation_id")
    question = str(tool_args.get("question") or "").strip()
    try:
        normalized_conversation_id = int(conversation_id)
    except (TypeError, ValueError):
        return {
            "status": "error",
            "error": "conversation_id must be an integer.",
        }, "Invalid conversation id"

    asset, image_bytes = read_image_asset_bytes(image_id, conversation_id=normalized_conversation_id)
    if not asset or not image_bytes:
        return {
            "status": "missing_image",
            "error": "Stored image not found. Ask the user to re-upload the image.",
            "image_id": image_id,
            "conversation_id": normalized_conversation_id,
        }, "Stored image not found"

    answer = answer_image_question(
        image_bytes,
        asset.get("mime_type", ""),
        question,
        initial_analysis=asset.get("initial_analysis"),
    )
    return {
        "status": "ok",
        "image_id": image_id,
        "conversation_id": normalized_conversation_id,
        "answer": answer,
    }, "Image question answered"


def _run_search_knowledge_base(tool_args: dict, runtime_state: dict):
    del runtime_state
    result = search_knowledge_base_tool(
        tool_args.get("query", ""),
        category=tool_args.get("category"),
        top_k=tool_args.get("top_k", RAG_SEARCH_DEFAULT_TOP_K),
        allowed_source_types=get_rag_source_types(),
    )
    return result, f"{result.get('count', 0)} knowledge chunks found"


def _run_search_tool_memory(tool_args: dict, runtime_state: dict):
    del runtime_state
    result = search_tool_memory(
        tool_args.get("query", ""),
        top_k=tool_args.get("top_k", RAG_SEARCH_DEFAULT_TOP_K),
    )
    return result, f"{result.get('count', 0)} tool memory matches found"


def _run_search_web(tool_args: dict, runtime_state: dict):
    del runtime_state
    result = search_web_tool(tool_args.get("queries", []))
    ok_count = sum(1 for row in result if "error" not in row)
    return result, f"{ok_count} web results found"


def _run_search_news_ddgs(tool_args: dict, runtime_state: dict):
    del runtime_state
    result = search_news_ddgs_tool(
        tool_args.get("queries", []),
        lang=tool_args.get("lang", "tr"),
        when=tool_args.get("when"),
    )
    ok_count = sum(1 for row in result if "error" not in row)
    return result, f"{ok_count} news articles found"


def _run_search_news_google(tool_args: dict, runtime_state: dict):
    del runtime_state
    result = search_news_google_tool(
        tool_args.get("queries", []),
        lang=tool_args.get("lang", "tr"),
        when=tool_args.get("when"),
    )
    ok_count = sum(1 for row in result if "error" not in row)
    return result, f"{ok_count} news articles found"


def _run_fetch_url(tool_args: dict, runtime_state: dict):
    del runtime_state
    result = fetch_url_tool(tool_args.get("url", ""))
    return result, _summarize_fetch_result(result, tool_args.get("url", ""))


def _run_create_canvas_document(tool_args: dict, runtime_state: dict):
    canvas_state = _get_canvas_runtime_state(runtime_state)
    document = create_canvas_document(
        canvas_state,
        title=tool_args.get("title", "Canvas"),
        content=tool_args.get("content", ""),
        format_name=tool_args.get("format", "markdown"),
        language_name=tool_args.get("language"),
        path=tool_args.get("path"),
        role=tool_args.get("role"),
        summary=tool_args.get("summary"),
        imports=tool_args.get("imports"),
        exports=tool_args.get("exports"),
        symbols=tool_args.get("symbols"),
        dependencies=tool_args.get("dependencies"),
        project_id=tool_args.get("project_id"),
        workspace_id=tool_args.get("workspace_id"),
    )
    return build_canvas_tool_result(document, action="created"), f"Canvas created: {document['title']}"


def _run_expand_canvas_document(tool_args: dict, runtime_state: dict):
    canvas_state = _get_canvas_runtime_state(runtime_state)
    canvas_limits = runtime_state.get("canvas_limits") if isinstance(runtime_state.get("canvas_limits"), dict) else {}
    expand_max_lines = int(canvas_limits.get("expand_max_lines") or 0) or None
    result = build_canvas_document_context_result(
        canvas_state,
        document_id=tool_args.get("document_id"),
        document_path=tool_args.get("document_path"),
        max_lines=expand_max_lines,
        max_chars=scale_canvas_char_limit(expand_max_lines, default_lines=800, default_chars=20_000) if expand_max_lines else None,
    )
    target_label = str(result.get("document_path") or result.get("title") or "Canvas").strip()
    return result, f"Canvas expanded: {target_label}"


def _run_scroll_canvas_document(tool_args: dict, runtime_state: dict):
    canvas_state = _get_canvas_runtime_state(runtime_state)
    canvas_limits = runtime_state.get("canvas_limits") if isinstance(runtime_state.get("canvas_limits"), dict) else {}
    scroll_window_lines = int(canvas_limits.get("scroll_window_lines") or 0) or 200
    result = scroll_canvas_document(
        canvas_state,
        start_line=int(tool_args.get("start_line") or 0),
        end_line=int(tool_args.get("end_line") or 0),
        document_id=tool_args.get("document_id"),
        document_path=tool_args.get("document_path"),
        max_window_lines=scroll_window_lines,
    )
    target_label = str(result.get("document_path") or result.get("title") or "Canvas").strip()
    return result, f"Canvas scrolled: {target_label} {result.get('start_line')}-{result.get('end_line_actual')}"


def _get_workspace_runtime_state(runtime_state: dict) -> dict:
    return runtime_state.setdefault("workspace", create_workspace_runtime_state())


def _get_project_workflow_state(runtime_state: dict) -> dict:
    return runtime_state.setdefault("project_workflow", create_project_workflow_runtime_state())


def _run_plan_project_workspace(tool_args: dict, runtime_state: dict):
    workflow = build_project_plan(
        tool_args.get("goal", ""),
        tool_args.get("project_name", "Project"),
        target_type=tool_args.get("target_type", "python-project"),
        files=tool_args.get("files"),
        dependencies=tool_args.get("dependencies"),
    )
    stored = update_project_workflow(_get_project_workflow_state(runtime_state), **workflow)
    return {"status": "ok", "action": "planned", "project_workflow": stored}, f"Project planned: {stored.get('project_name', '')}"


def _run_get_project_workflow_status(tool_args: dict, runtime_state: dict):
    del tool_args
    workflow = get_project_workflow(_get_project_workflow_state(runtime_state))
    return {"status": "ok", "action": "workflow_status", "project_workflow": workflow}, f"Workflow stage: {str((workflow or {}).get('stage') or 'plan')}"


def _run_create_directory(tool_args: dict, runtime_state: dict):
    result = workspace_create_directory(_get_workspace_runtime_state(runtime_state), tool_args.get("path", ""))
    return result, f"Directory created: {result.get('path', '')}"


def _run_create_file(tool_args: dict, runtime_state: dict):
    result = workspace_create_file(
        _get_workspace_runtime_state(runtime_state),
        tool_args.get("path", ""),
        tool_args.get("content", ""),
    )
    return result, f"File created: {result.get('path', '')}"


def _run_update_file(tool_args: dict, runtime_state: dict):
    result = workspace_update_file(
        _get_workspace_runtime_state(runtime_state),
        tool_args.get("path", ""),
        tool_args.get("content", ""),
    )
    return result, f"File updated: {result.get('path', '')}"


def _run_read_file(tool_args: dict, runtime_state: dict):
    result = workspace_read_file(
        _get_workspace_runtime_state(runtime_state),
        tool_args.get("path", ""),
        start_line=tool_args.get("start_line", 1),
        end_line=tool_args.get("end_line"),
    )
    return result, f"File read: {result.get('path', '')}"


def _run_list_dir(tool_args: dict, runtime_state: dict):
    result = workspace_list_dir(_get_workspace_runtime_state(runtime_state), tool_args.get("path"))
    return result, f"Directory listed: {result.get('path', '')}"


def _run_search_files(tool_args: dict, runtime_state: dict):
    result = workspace_search_files(
        _get_workspace_runtime_state(runtime_state),
        tool_args.get("query", ""),
        path_prefix=tool_args.get("path_prefix"),
        search_content=tool_args.get("search_content") is True,
    )
    return result, f"{len(result.get('matches') or [])} workspace matches found"


def _run_preview_workspace_changes(tool_args: dict, runtime_state: dict):
    result = preview_workspace_changes(
        _get_workspace_runtime_state(runtime_state),
        tool_args.get("files") or [],
    )
    return result, f"Workspace diff preview: {len(result.get('diffs') or [])} files"


def _run_get_workspace_file_history(tool_args: dict, runtime_state: dict):
    result = get_workspace_file_history(
        _get_workspace_runtime_state(runtime_state),
        tool_args.get("path", ""),
    )
    return result, f"Workspace history: {result.get('path', '')}"


def _run_undo_workspace_file_change(tool_args: dict, runtime_state: dict):
    result = undo_workspace_file_change(
        _get_workspace_runtime_state(runtime_state),
        tool_args.get("path", ""),
    )
    return result, f"Workspace undo: {result.get('path', '')}"


def _run_redo_workspace_file_change(tool_args: dict, runtime_state: dict):
    result = redo_workspace_file_change(
        _get_workspace_runtime_state(runtime_state),
        tool_args.get("path", ""),
    )
    return result, f"Workspace redo: {result.get('path', '')}"


def _run_create_project_scaffold(tool_args: dict, runtime_state: dict):
    result = create_project_scaffold(
        _get_workspace_runtime_state(runtime_state),
        tool_args.get("project_name", "Project"),
        target_type=tool_args.get("target_type", "python-project"),
        confirm=tool_args.get("confirm") is True,
    )
    if result.get("status") == "ok":
        update_project_workflow(
            _get_project_workflow_state(runtime_state),
            project_name=tool_args.get("project_name", "Project"),
            target_type=tool_args.get("target_type", "python-project"),
            stage="skeleton",
        )
    return result, f"Project scaffold: {result.get('project_root', '')}"


def _run_write_project_tree(tool_args: dict, runtime_state: dict):
    result = write_project_tree(
        _get_workspace_runtime_state(runtime_state),
        directories=tool_args.get("directories") or [],
        files=tool_args.get("files") or [],
        confirm=tool_args.get("confirm") is True,
    )
    if result.get("status") == "ok":
        current = get_project_workflow(_get_project_workflow_state(runtime_state)) or {}
        file_updates = list(current.get("files") or [])
        known_by_path = {str(entry.get("path") or ""): dict(entry) for entry in file_updates}
        for path in result.get("files") or []:
            entry = known_by_path.get(path, {"path": path})
            entry["status"] = "written"
            known_by_path[path] = entry
        update_project_workflow(_get_project_workflow_state(runtime_state), files=list(known_by_path.values()), stage="content")
    return result, f"Project tree write: {len(result.get('files') or [])} files"


def _run_bulk_update_workspace_files(tool_args: dict, runtime_state: dict):
    result = bulk_update_workspace_files(
        _get_workspace_runtime_state(runtime_state),
        tool_args.get("files") or [],
        confirm=tool_args.get("confirm") is True,
    )
    if result.get("status") == "ok":
        current = get_project_workflow(_get_project_workflow_state(runtime_state)) or {}
        file_updates = list(current.get("files") or [])
        known_by_path = {str(entry.get("path") or ""): dict(entry) for entry in file_updates}
        for path in result.get("files") or []:
            entry = known_by_path.get(path, {"path": path})
            entry["status"] = "written"
            known_by_path[path] = entry
        update_project_workflow(_get_project_workflow_state(runtime_state), files=list(known_by_path.values()), stage="content")
    return result, f"Bulk workspace update: {len(result.get('files') or [])} files"


def _run_validate_project_workspace(tool_args: dict, runtime_state: dict):
    result = validate_project_workspace(
        _get_workspace_runtime_state(runtime_state),
        path=tool_args.get("path"),
    )
    workflow_state = _get_project_workflow_state(runtime_state)
    next_stage = "validated" if result.get("status") == "ok" else "fix"
    update_project_workflow(
        workflow_state,
        stage=next_stage,
        validation={
            "status": result.get("status", "ok"),
            "issues": result.get("issues") or [],
            "warnings": result.get("warnings") or [],
        },
        open_issues=result.get("issues") or [],
    )
    return result, f"Workspace validation: {result.get('status', 'ok')}"


def _run_rewrite_canvas_document(tool_args: dict, runtime_state: dict):
    canvas_state = _get_canvas_runtime_state(runtime_state)
    document = rewrite_canvas_document(
        canvas_state,
        content=tool_args.get("content", ""),
        document_id=tool_args.get("document_id"),
        document_path=tool_args.get("document_path"),
        title=tool_args.get("title"),
        format_name=tool_args.get("format"),
        language_name=tool_args.get("language"),
        path=tool_args.get("path"),
        role=tool_args.get("role"),
        summary=tool_args.get("summary"),
        imports=tool_args.get("imports"),
        exports=tool_args.get("exports"),
        symbols=tool_args.get("symbols"),
        dependencies=tool_args.get("dependencies"),
        project_id=tool_args.get("project_id"),
        workspace_id=tool_args.get("workspace_id"),
    )
    return build_canvas_tool_result(document, action="rewritten"), f"Canvas updated: {document['title']}"


def _run_replace_canvas_lines(tool_args: dict, runtime_state: dict):
    canvas_state = _get_canvas_runtime_state(runtime_state)
    document = replace_canvas_lines(
        canvas_state,
        start_line=tool_args.get("start_line", 0),
        end_line=tool_args.get("end_line", 0),
        lines=tool_args.get("lines", []),
        document_id=tool_args.get("document_id"),
        document_path=tool_args.get("document_path"),
    )
    return build_canvas_tool_result(document, action="lines_replaced"), f"Canvas lines replaced in {document['title']}"


def _run_insert_canvas_lines(tool_args: dict, runtime_state: dict):
    canvas_state = _get_canvas_runtime_state(runtime_state)
    document = insert_canvas_lines(
        canvas_state,
        after_line=tool_args.get("after_line", 0),
        lines=tool_args.get("lines", []),
        document_id=tool_args.get("document_id"),
        document_path=tool_args.get("document_path"),
    )
    return build_canvas_tool_result(document, action="lines_inserted"), f"Canvas lines inserted in {document['title']}"


def _run_delete_canvas_lines(tool_args: dict, runtime_state: dict):
    canvas_state = _get_canvas_runtime_state(runtime_state)
    document = delete_canvas_lines(
        canvas_state,
        start_line=tool_args.get("start_line", 0),
        end_line=tool_args.get("end_line", 0),
        document_id=tool_args.get("document_id"),
        document_path=tool_args.get("document_path"),
    )
    return build_canvas_tool_result(document, action="lines_deleted"), f"Canvas lines deleted in {document['title']}"


def _run_delete_canvas_document(tool_args: dict, runtime_state: dict):
    canvas_state = _get_canvas_runtime_state(runtime_state)
    result = delete_canvas_document(
        canvas_state,
        document_id=tool_args.get("document_id"),
        document_path=tool_args.get("document_path"),
    )
    deleted_title = str(result.get("deleted_title") or "Canvas")
    return result, f"Canvas deleted: {deleted_title}"


def _run_clear_canvas(tool_args: dict, runtime_state: dict):
    del tool_args
    canvas_state = _get_canvas_runtime_state(runtime_state)
    result = clear_canvas(canvas_state)
    return result, f"Canvas cleared ({result.get('cleared_count', 0)} documents removed)"


_TOOL_EXECUTORS = {
    "append_scratchpad": _run_append_scratchpad,
    "replace_scratchpad": _run_replace_scratchpad,
    "ask_clarifying_question": _run_ask_clarifying_question,
    "image_explain": _run_image_explain,
    "search_knowledge_base": _run_search_knowledge_base,
    "search_tool_memory": _run_search_tool_memory,
    "search_web": _run_search_web,
    "search_news_ddgs": _run_search_news_ddgs,
    "search_news_google": _run_search_news_google,
    "fetch_url": _run_fetch_url,
    "expand_canvas_document": _run_expand_canvas_document,
    "scroll_canvas_document": _run_scroll_canvas_document,
    "plan_project_workspace": _run_plan_project_workspace,
    "get_project_workflow_status": _run_get_project_workflow_status,
    "create_directory": _run_create_directory,
    "create_file": _run_create_file,
    "update_file": _run_update_file,
    "read_file": _run_read_file,
    "list_dir": _run_list_dir,
    "search_files": _run_search_files,
    "preview_workspace_changes": _run_preview_workspace_changes,
    "get_workspace_file_history": _run_get_workspace_file_history,
    "undo_workspace_file_change": _run_undo_workspace_file_change,
    "redo_workspace_file_change": _run_redo_workspace_file_change,
    "create_project_scaffold": _run_create_project_scaffold,
    "write_project_tree": _run_write_project_tree,
    "bulk_update_workspace_files": _run_bulk_update_workspace_files,
    "validate_project_workspace": _run_validate_project_workspace,
    "create_canvas_document": _run_create_canvas_document,
    "rewrite_canvas_document": _run_rewrite_canvas_document,
    "replace_canvas_lines": _run_replace_canvas_lines,
    "insert_canvas_lines": _run_insert_canvas_lines,
    "delete_canvas_lines": _run_delete_canvas_lines,
    "delete_canvas_document": _run_delete_canvas_document,
    "clear_canvas": _run_clear_canvas,
}


def _execute_tool(tool_name: str, tool_args: dict, runtime_state: dict | None = None):
    runtime_state = runtime_state if isinstance(runtime_state, dict) else {}
    handler = _TOOL_EXECUTORS.get(tool_name)
    if handler is not None:
        return handler(tool_args if isinstance(tool_args, dict) else {}, runtime_state)
    return {"error": f"Unknown tool: {tool_name}"}, f"Unknown tool: {tool_name}"


def collect_agent_response(
    api_messages: list,
    model: str,
    max_steps: int,
    enabled_tool_names: list[str],
    fetch_url_token_threshold: int | None = None,
    fetch_url_clip_aggressiveness: int | None = None,
) -> dict:
    full_response = ""
    full_reasoning = ""
    usage_data = None
    tool_results = []
    errors = []

    for event in run_agent_stream(
        api_messages,
        model,
        max_steps,
        enabled_tool_names,
        fetch_url_token_threshold=fetch_url_token_threshold,
        fetch_url_clip_aggressiveness=fetch_url_clip_aggressiveness,
    ):
        if event["type"] == "answer_delta":
            full_response += event.get("text", "")
        elif event["type"] == "reasoning_delta":
            full_reasoning += event.get("text", "")
        elif event["type"] == "usage":
            usage_data = event
        elif event["type"] == "tool_capture":
            tool_results = event.get("tool_results") or []
        elif event["type"] == "tool_error":
            errors.append(event.get("error") or "Unknown tool error")

    return {
        "content": full_response,
        "reasoning_content": full_reasoning,
        "usage": usage_data,
        "tool_results": tool_results,
        "errors": errors,
    }


def _tool_input_preview(tool_name: str, tool_args: dict) -> str:
    tool_args = tool_args if isinstance(tool_args, dict) else {}
    if tool_name in {"search_web", "search_news_ddgs", "search_news_google"}:
        values = tool_args.get("queries")
        if isinstance(values, list):
            return ", ".join(str(value).strip() for value in values if str(value).strip())[:300]
    if tool_name == "fetch_url":
        return str(tool_args.get("url") or "").strip()[:300]
    return ""


def _build_compact_tool_message_content(
    tool_name: str,
    tool_args: dict,
    result,
    summary: str,
    transcript_result=None,
    storage_entry: dict | None = None,
) -> str:
    del result
    if tool_name == "fetch_url" and isinstance(transcript_result, dict):
        parts = []
        title = _clean_tool_text(transcript_result.get("title") or "", limit=160)
        url = _clean_tool_text(transcript_result.get("url") or tool_args.get("url") or "", limit=200)
        notice = _clean_tool_text(transcript_result.get("summary_notice") or "", limit=240)
        diagnostic = _clean_tool_text(transcript_result.get("fetch_diagnostic") or "", limit=280)
        body = _clean_tool_text(transcript_result.get("content") or "", limit=4_000)
        if title:
            parts.append(f"Title: {title}")
        if url:
            parts.append(f"URL: {url}")
        if summary:
            parts.append(f"Summary: {_clean_tool_text(summary, limit=300)}")
        if notice:
            parts.append(f"Note: {notice}")
        if diagnostic:
            parts.append(f"Fetch status: {diagnostic}")
        if body:
            parts.append(body)
        return "\n\n".join(parts).strip()

    if isinstance(transcript_result, str):
        if len(transcript_result) <= RAG_TOOL_RESULT_MAX_TEXT_CHARS:
            return transcript_result
        clip_marker = " [CLIPPED: original "
        marker_index = transcript_result.find(clip_marker)
        if marker_index > 0:
            marker = transcript_result[marker_index:]
            prefix_limit = max(0, RAG_TOOL_RESULT_MAX_TEXT_CHARS - len(marker) - 1)
            prefix = transcript_result[:prefix_limit].rstrip()
            return f"{prefix}…{marker}"
        return _clean_tool_text(transcript_result, limit=RAG_TOOL_RESULT_MAX_TEXT_CHARS)

    preferred_entry = storage_entry if isinstance(storage_entry, dict) else None
    if preferred_entry:
        content = _clean_tool_text(preferred_entry.get("content") or "", limit=RAG_TOOL_RESULT_MAX_TEXT_CHARS)
        if content:
            return content

    if tool_name == "fetch_url" and isinstance(transcript_result, dict):
        parts = []
        title = _clean_tool_text(transcript_result.get("title") or "", limit=160)
        url = _clean_tool_text(transcript_result.get("url") or tool_args.get("url") or "", limit=200)
        notice = _clean_tool_text(transcript_result.get("summary_notice") or "", limit=240)
        diagnostic = _clean_tool_text(transcript_result.get("fetch_diagnostic") or "", limit=280)
        body = _clean_tool_text(transcript_result.get("content") or "", limit=4_000)
        if title:
            parts.append(f"Title: {title}")
        if url:
            parts.append(f"URL: {url}")
        if summary:
            parts.append(f"Summary: {_clean_tool_text(summary, limit=300)}")
        if notice:
            parts.append(f"Note: {notice}")
        if diagnostic:
            parts.append(f"Fetch status: {diagnostic}")
        if body:
            parts.append(body)
        return "\n\n".join(parts).strip()

    if isinstance(transcript_result, str):
        return _clean_tool_text(transcript_result, limit=RAG_TOOL_RESULT_MAX_TEXT_CHARS)

    try:
        return _serialize_tool_message_content(transcript_result)
    except Exception:
        return _serialize_tool_message_content({"tool_name": tool_name, "summary": _clean_tool_text(summary, limit=300)})


def _format_list_tool_result(items: list[dict], title: str, link_key: str, extra_keys: tuple[str, ...] = ()) -> str:
    lines = [title]
    added = 0
    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict) or item.get("error"):
            continue
        entry_lines = [f"{index}. {str(item.get('title') or 'Untitled').strip()}"]
        link = str(item.get(link_key) or "").strip()
        if link:
            entry_lines.append(f"URL: {link}")
        snippet = str(item.get("snippet") or item.get("body") or "").strip()
        if snippet:
            entry_lines.append(f"Snippet: {snippet}")
        for extra_key in extra_keys:
            value = str(item.get(extra_key) or "").strip()
            if value:
                entry_lines.append(f"{extra_key.title()}: {value}")
        lines.append("\n".join(entry_lines))
        added += 1
    if added == 0:
        return ""
    return _clean_tool_text("\n\n".join(lines), limit=RAG_TOOL_RESULT_MAX_TEXT_CHARS)


def _build_tool_result_storage_entry(tool_name: str, tool_args: dict, result, summary: str, transcript_result=None) -> dict | None:
    if tool_name in {"search_knowledge_base", "search_tool_memory"}:
        return None

    text = ""
    if tool_name == "fetch_url":
        if isinstance(result, dict):
            display_result = transcript_result if isinstance(transcript_result, dict) else result
            display_content = _clean_tool_text(display_result.get("content") or "", limit=RAG_TOOL_RESULT_MAX_TEXT_CHARS)
            raw_content = _clean_tool_text(result.get("content") or "", limit=FETCH_RAW_TOOL_RESULT_MAX_TEXT_CHARS)
            parts = []
            title = str(result.get("title") or "").strip()
            url = str(result.get("url") or tool_args.get("url") or "").strip()
            summary_notice = str(display_result.get("summary_notice") or "").strip()
            fetch_diagnostic = str(display_result.get("fetch_diagnostic") or "").strip()
            if title:
                parts.append(f"Title: {title}")
            if url:
                parts.append(f"URL: {url}")
            if summary_notice:
                parts.append(f"Note: {summary_notice}")
            if fetch_diagnostic:
                parts.append(f"Fetch status: {fetch_diagnostic}")
            if display_content:
                parts.append(display_content)
            text = "\n\n".join(parts)
    elif tool_name == "search_web" and isinstance(result, list):
        text = _format_list_tool_result(result, "Web results", link_key="url")
    elif tool_name in {"search_news_ddgs", "search_news_google"} and isinstance(result, list):
        text = _format_list_tool_result(result, "News results", link_key="link", extra_keys=("time", "source"))

    text = _clean_tool_text(text, limit=RAG_TOOL_RESULT_MAX_TEXT_CHARS)
    if not text:
        return None

    entry = {
        "tool_name": tool_name,
        "content": text,
    }
    cleaned_summary = _clean_tool_text(summary, limit=RAG_TOOL_RESULT_SUMMARY_MAX_CHARS)
    if cleaned_summary:
        entry["summary"] = cleaned_summary
    input_preview = _tool_input_preview(tool_name, tool_args)
    if input_preview:
        entry["input_preview"] = input_preview
    if tool_name == "fetch_url" and isinstance(result, dict):
        display_result = transcript_result if isinstance(transcript_result, dict) else result
        raw_content = _clean_tool_text(result.get("content") or "", limit=FETCH_RAW_TOOL_RESULT_MAX_TEXT_CHARS)
        display_content = _clean_tool_text(display_result.get("content") or "", limit=FETCH_RAW_TOOL_RESULT_MAX_TEXT_CHARS)
        content_mode = str(display_result.get("content_mode") or "").strip()
        summary_notice = _clean_tool_text(display_result.get("summary_notice") or "", limit=300)
        token_estimate = display_result.get("content_token_estimate")
        fetch_outcome = _clean_tool_text(display_result.get("fetch_outcome") or "", limit=80)
        fetch_diagnostic = _clean_tool_text(display_result.get("fetch_diagnostic") or "", limit=500)
        content_char_count = display_result.get("content_char_count")
        if raw_content and raw_content != display_content:
            entry["raw_content"] = raw_content
        if content_mode:
            entry["content_mode"] = content_mode
        if summary_notice:
            entry["summary_notice"] = summary_notice
        if fetch_outcome:
            entry["fetch_outcome"] = fetch_outcome
        if fetch_diagnostic:
            entry["fetch_diagnostic"] = fetch_diagnostic
        if display_result.get("cleanup_applied"):
            entry["cleanup_applied"] = True
        if isinstance(token_estimate, int) and token_estimate >= 0:
            entry["content_token_estimate"] = token_estimate
        if isinstance(content_char_count, int) and content_char_count >= 0:
            entry["content_char_count"] = content_char_count
    return entry


def _lookup_cross_turn_tool_memory(tool_name: str, tool_args: dict) -> tuple[object, str] | None:
    if tool_name == "fetch_url":
        url = _tool_input_preview(tool_name, tool_args)
        if not url:
            return None
        try:
            exact_match = get_exact_tool_memory_match(tool_name, url)
        except Exception:
            return None
        if not exact_match:
            return None
        excerpt = _clean_tool_text(exact_match.get("content") or "", limit=RAG_TOOL_RESULT_MAX_TEXT_CHARS)
        if not excerpt:
            return None
        summary = _clean_tool_text(exact_match.get("summary") or "", limit=RAG_TOOL_RESULT_SUMMARY_MAX_CHARS)
        if not summary:
            summary = f"Reused cached fetch_url result for {url}"
        return excerpt, summary

    if tool_name not in {"search_web", "search_news_ddgs", "search_news_google"}:
        return None

    query = _tool_input_preview(tool_name, tool_args)
    if not query:
        return None

    try:
        matches = (search_tool_memory(query, top_k=1).get("matches") or [])[:1]
    except Exception:
        return None
    if not matches:
        return None

    best_match = matches[0]
    similarity = best_match.get("similarity")
    if not isinstance(similarity, (int, float)) or similarity < 0.85:
        return None

    excerpt = _clean_tool_text(best_match.get("text") or "", limit=RAG_TOOL_RESULT_MAX_TEXT_CHARS)
    if not excerpt:
        return None

    source_name = _clean_tool_text(best_match.get("source_name") or "Tool memory", limit=120)
    summary = f"Reused tool memory from {source_name}"
    return excerpt, summary


def _extract_clarification_event(result: dict) -> dict | None:
    if not isinstance(result, dict):
        return None
    if str(result.get("status") or "").strip() != "needs_user_input":
        return None
    payload = result.get("clarification") if isinstance(result.get("clarification"), dict) else None
    if not payload:
        return None
    text = str(result.get("text") or "").strip() or _build_clarification_text(payload)
    return {
        "type": "clarification_request",
        "clarification": payload,
        "text": text,
    }


def _extract_initial_goal(messages: list[dict]) -> str:
    for message in messages:
        if str(message.get("role") or "").strip() != "user":
            continue
        content = _clean_tool_text(message.get("content") or "", limit=180)
        if content:
            return content
    return ""


def _append_working_state_attempt(working_state: dict, tool_name: str, preview: str) -> None:
    attempts = working_state.setdefault("steps_tried", [])
    entry = {
        "tool_name": str(tool_name or "").strip() or "tool",
        "preview": _clean_tool_text(preview or "", limit=140),
    }
    if attempts and attempts[-1] == entry:
        return
    attempts.append(entry)
    if len(attempts) > 8:
        del attempts[:-8]


def _append_working_state_blocker(working_state: dict, tool_name: str, error: str) -> None:
    blockers = working_state.setdefault("blockers", [])
    entry = {
        "tool_name": str(tool_name or "").strip() or "tool",
        "error": _clean_tool_text(error or "", limit=220),
    }
    if blockers and blockers[-1] == entry:
        return
    blockers.append(entry)
    if len(blockers) > 6:
        del blockers[:-6]


def _append_reasoning_replay_entry(reasoning_state: dict, step: int, reasoning_text: str, tool_calls: list[dict] | None) -> None:
    if not isinstance(reasoning_state, dict):
        return

    cleaned_reasoning = _clean_tool_text(reasoning_text or "", limit=MAX_REASONING_REPLAY_CHARS)
    if not cleaned_reasoning:
        return

    entries = reasoning_state.setdefault("entries", [])
    tool_names = [
        str(tool_call.get("name") or "").strip()
        for tool_call in (tool_calls or [])
        if str(tool_call.get("name") or "").strip()
    ]
    entry = {
        "step": max(1, int(step or 0)),
        "reasoning": cleaned_reasoning,
        "tool_names": tool_names,
    }
    if entries and entries[-1] == entry:
        return
    entries.append(entry)
    if len(entries) > MAX_REASONING_REPLAY_ENTRIES:
        del entries[:-MAX_REASONING_REPLAY_ENTRIES]


def _build_reasoning_replay_instruction(reasoning_state: dict, current_goal: str = "") -> dict | None:
    if not isinstance(reasoning_state, dict):
        return None

    entries = reasoning_state.get("entries") if isinstance(reasoning_state.get("entries"), list) else []
    if not entries:
        return None

    parts = [REASONING_REPLAY_MARKER]
    parts.append(
        "Use this as your prior reasoning from earlier steps in the current run. Continue from it when it still fits, but correct it if tool results changed the picture."
    )

    normalized_goal = _clean_tool_text(current_goal or "", limit=180)
    if normalized_goal:
        parts.append(f"Current goal: {normalized_goal}")

    for entry in entries[-MAX_REASONING_REPLAY_ENTRIES:]:
        step_number = max(1, int(entry.get("step") or 0))
        tool_names = [
            str(tool_name or "").strip()
            for tool_name in (entry.get("tool_names") or [])
            if str(tool_name or "").strip()
        ]
        header = f"Step {step_number} reasoning"
        if tool_names:
            header += ": planned tools = " + ", ".join(tool_names)
        parts.append(header + "\n" + str(entry.get("reasoning") or ""))

    return {"role": "system", "content": "\n\n".join(parts)}


def _build_working_state_instruction(working_state: dict) -> dict | None:
    if not isinstance(working_state, dict):
        return None

    current_goal = _clean_tool_text(working_state.get("current_goal") or "", limit=180)
    attempts = working_state.get("steps_tried") if isinstance(working_state.get("steps_tried"), list) else []
    blockers = working_state.get("blockers") if isinstance(working_state.get("blockers"), list) else []
    if not blockers:
        return None

    parts = ["[AGENT WORKING MEMORY]"]
    if current_goal:
        parts.append(f"Current goal: {current_goal}")
    if attempts:
        lines = []
        for entry in attempts[-5:]:
            tool_name = _clean_tool_text(entry.get("tool_name") or "tool", limit=80)
            preview = _clean_tool_text(entry.get("preview") or "", limit=120)
            line = f"- {tool_name}"
            if preview:
                line += f": {preview}"
            lines.append(line)
        if lines:
            parts.append("Tried in this run:\n" + "\n".join(lines))
    if blockers:
        lines = []
        for entry in blockers[-4:]:
            tool_name = _clean_tool_text(entry.get("tool_name") or "tool", limit=80)
            error = _clean_tool_text(entry.get("error") or "", limit=180)
            line = f"- {tool_name}"
            if error:
                line += f": {error}"
            lines.append(line)
        if lines:
            parts.append("Failed paths to avoid repeating without a concrete reason:\n" + "\n".join(lines))
    parts.append("Prefer a different tool or produce the best available answer if these blockers make repetition low-value.")
    return {"role": "system", "content": "\n\n".join(parts)}


def _get_tool_step_limit(tool_name: str, max_steps: int = 5) -> int:
    del tool_name
    try:
        limit = int(max_steps)
    except (TypeError, ValueError):
        limit = max_steps
    return max(1, limit)


def run_agent_stream(
    api_messages: list,
    model: str,
    max_steps: int,
    enabled_tool_names: list[str],
    fetch_url_token_threshold: int | None = None,
    fetch_url_clip_aggressiveness: int | None = None,
    initial_canvas_documents: list[dict] | None = None,
    initial_canvas_active_document_id: str | None = None,
    canvas_expand_max_lines: int | None = None,
    canvas_scroll_window_lines: int | None = None,
    workspace_runtime_state: dict | None = None,
    initial_project_workflow: dict | None = None,
):
    messages = list(api_messages)
    step = 0
    tool_result_cache = {}
    persisted_tool_results = []
    persisted_tool_cache_keys = set()
    reasoning_started = False
    answer_started = False
    pending_answer_separator = False
    fatal_api_error = None
    trace_id = uuid4().hex[:12]
    total_clean_content = ""
    fetch_attempt_counts: dict[str, int] = {}
    tool_call_counts: dict[str, int] = defaultdict(int)
    canvas_modified = False
    usage_totals = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "estimated_input_tokens": 0,
        "input_breakdown": _empty_input_breakdown(),
        "model_call_count": 0,
        "model_calls": [],
    }
    runtime_state = {
        "canvas": create_canvas_runtime_state(
            initial_canvas_documents,
            active_document_id=initial_canvas_active_document_id,
        ),
        "canvas_limits": {
            "expand_max_lines": int(canvas_expand_max_lines or 800),
            "scroll_window_lines": int(canvas_scroll_window_lines or 200),
        },
        "workspace": workspace_runtime_state if isinstance(workspace_runtime_state, dict) else create_workspace_runtime_state(),
        "project_workflow": create_project_workflow_runtime_state(initial_project_workflow),
    }
    working_state = {
        "current_goal": _extract_initial_goal(messages),
        "steps_tried": [],
        "blockers": [],
    }
    reasoning_state = {
        "entries": [],
    }

    def build_tool_capture_event() -> dict:
        current_canvas_documents = get_canvas_runtime_documents(runtime_state.get("canvas"))
        active_canvas_document_id = get_canvas_runtime_active_document_id(runtime_state.get("canvas"))
        project_workflow = get_project_workflow(runtime_state.get("project_workflow"))
        return {
            "type": "tool_capture",
            "tool_results": persisted_tool_results,
            "canvas_documents": current_canvas_documents,
            "active_document_id": active_canvas_document_id,
            "canvas_cleared": canvas_modified and not current_canvas_documents,
            "project_workflow": project_workflow,
        }

    pricing = get_model_pricing(model)
    _trace_agent_event(
        "agent_run_started",
        trace_id=trace_id,
        model=model,
        max_steps=max_steps,
        enabled_tool_names=enabled_tool_names,
        api_messages=_summarize_messages_for_log(messages),
        log_path=AGENT_TRACE_LOG_PATH,
    )

    def add_usage(usage):
        if not usage:
            return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "received": False}

        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        total_tokens = int(getattr(usage, "total_tokens", 0) or 0)

        usage_totals["prompt_tokens"] += prompt_tokens
        usage_totals["completion_tokens"] += completion_tokens
        usage_totals["total_tokens"] += total_tokens
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "received": any(value > 0 for value in (prompt_tokens, completion_tokens, total_tokens)),
        }

    def calculate_cost(prompt_tokens, completion_tokens):
        input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
        output_cost = (completion_tokens / 1_000_000) * pricing["output"]
        return round(input_cost + output_cost, 6)

    def apply_context_compaction(extra_messages: list[dict] | None = None, reason: str = "", force: bool = False):
        nonlocal messages
        extra_messages = list(extra_messages or [])
        turn_messages = [*messages, *extra_messages]
        threshold = max(1, int(PROMPT_MAX_INPUT_TOKENS * AGENT_CONTEXT_COMPACTION_THRESHOLD))
        before_tokens = _estimate_messages_tokens(turn_messages)
        before_message_count = len(turn_messages)
        before_exchange_count = _count_exchange_blocks(messages)
        if not force and before_tokens <= threshold:
            return turn_messages, False

        compacted_messages = _try_compact_messages(
            messages,
            max(1, int(PROMPT_MAX_INPUT_TOKENS * 0.75)),
            keep_recent=0 if force else AGENT_CONTEXT_COMPACTION_KEEP_RECENT_ROUNDS,
        )
        if compacted_messages is None:
            return turn_messages, False

        messages = compacted_messages
        compacted_turn_messages = [*messages, *extra_messages]
        after_tokens = _estimate_messages_tokens(compacted_turn_messages)
        after_message_count = len(compacted_turn_messages)
        after_exchange_count = _count_exchange_blocks(messages)
        _trace_agent_event(
            "context_compacted",
            trace_id=trace_id,
            step=step,
            reason=reason,
            before_tokens=before_tokens,
            after_tokens=after_tokens,
            threshold=threshold,
            force=force,
            before_message_count=before_message_count,
            after_message_count=after_message_count,
            compacted_exchange_count=max(0, before_exchange_count - after_exchange_count),
            merged_message_delta=max(0, before_message_count - after_message_count),
            keep_recent=0 if force else AGENT_CONTEXT_COMPACTION_KEEP_RECENT_ROUNDS,
        )
        return compacted_turn_messages, True

    def usage_event():
        call_usage_summary = _summarize_model_call_usage(
            usage_totals["model_calls"],
            fallback_input_tokens=usage_totals["prompt_tokens"],
        )
        total_cost = calculate_cost(usage_totals["prompt_tokens"], usage_totals["completion_tokens"])
        return {
            "type": "usage",
            "prompt_tokens": usage_totals["prompt_tokens"],
            "completion_tokens": usage_totals["completion_tokens"],
            "total_tokens": usage_totals["total_tokens"],
            "estimated_input_tokens": usage_totals["estimated_input_tokens"],
            "input_breakdown": dict(usage_totals["input_breakdown"]),
            "model_call_count": usage_totals["model_call_count"],
            "model_calls": list(usage_totals["model_calls"]),
            "max_input_tokens_per_call": call_usage_summary["max_input_tokens_per_call"],
            "configured_prompt_max_input_tokens": PROMPT_MAX_INPUT_TOKENS,
            "cost": total_cost,
            "currency": "USD",
            "model": model,
        }

    def remember_tool_result(tool_name: str, tool_args: dict, result, summary: str, cache_key: str, transcript_result=None):
        if cache_key in persisted_tool_cache_keys:
            return
        entry = _build_tool_result_storage_entry(tool_name, tool_args, result, summary, transcript_result=transcript_result)
        if not entry:
            return
        persisted_tool_cache_keys.add(cache_key)
        persisted_tool_results.append(entry)

    def emit_reasoning(reasoning_text: str):
        nonlocal reasoning_started
        if not reasoning_text:
            return
        if not reasoning_started:
            yield {"type": "reasoning_start"}
            reasoning_started = True
        yield {"type": "reasoning_delta", "text": reasoning_text}

    def emit_reasoning_separator():
        if not reasoning_started:
            return
        yield {"type": "reasoning_delta", "text": "\n\n"}

    def emit_answer(answer_text: str):
        nonlocal answer_started, pending_answer_separator
        if pending_answer_separator and str(answer_text or "").strip():
            yield {"type": "answer_delta", "text": "\n\n"}
            pending_answer_separator = False
        if not answer_started:
            yield {"type": "answer_start"}
            answer_started = True
        yield {"type": "answer_delta", "text": answer_text}

    def stream_model_turn(
        messages_to_send: list[dict],
        allow_tools: bool = True,
        *,
        call_type: str = "agent_step",
        retry_reason: str | None = None,
    ) -> dict:
        turn_reasoning_emitted = False
        turn_tools = []
        provider_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "received": False}
        _trace_agent_event(
            "model_turn_started",
            trace_id=trace_id,
            step=step,
            message_count=len(messages_to_send),
            messages=_summarize_messages_for_log(messages_to_send),
        )

        def emit_turn_reasoning(reasoning_text: str):
            nonlocal turn_reasoning_emitted
            if not reasoning_text:
                return
            if not turn_reasoning_emitted and reasoning_started:
                for event in emit_reasoning_separator():
                    yield event
            turn_reasoning_emitted = True
            for event in emit_reasoning(reasoning_text):
                yield event

        request_kwargs = {
            "model": model,
            "messages": messages_to_send,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if allow_tools:
            current_canvas_documents = get_canvas_runtime_documents(runtime_state.get("canvas"))
            turn_tools = get_openai_tool_specs(enabled_tool_names, canvas_documents=current_canvas_documents)
            if turn_tools:
                request_kwargs["tools"] = turn_tools
                request_kwargs["tool_choice"] = "auto"
        response = client.chat.completions.create(**request_kwargs)

        def finalize_call_usage() -> tuple[dict[str, int], int, int]:
            nonlocal provider_usage
            estimated_breakdown, estimated_input_tokens, tool_schema_tokens = _estimate_input_breakdown(
                messages_to_send,
                provider_prompt_tokens=provider_usage["prompt_tokens"] if provider_usage["received"] else None,
                request_tools=turn_tools,
            )
            usage_totals["model_call_count"] += 1
            usage_totals["model_calls"].append(
                {
                    "index": usage_totals["model_call_count"],
                    "call_type": call_type,
                    "step": step,
                    "is_retry": bool(retry_reason),
                    "retry_reason": str(retry_reason or "").strip() or None,
                    "message_count": len(messages_to_send),
                    "tool_schema_tokens": tool_schema_tokens,
                    "prompt_tokens": provider_usage["prompt_tokens"] if provider_usage["received"] else None,
                    "completion_tokens": provider_usage["completion_tokens"] if provider_usage["received"] else None,
                    "total_tokens": provider_usage["total_tokens"] if provider_usage["received"] else None,
                    "estimated_input_tokens": estimated_input_tokens,
                    "input_breakdown": dict(estimated_breakdown),
                    "missing_provider_usage": not provider_usage["received"],
                }
            )
            if provider_usage["received"]:
                usage_totals["estimated_input_tokens"] += estimated_input_tokens
                for key, value in estimated_breakdown.items():
                    usage_totals["input_breakdown"][key] += value
            return estimated_breakdown, estimated_input_tokens, tool_schema_tokens

        if getattr(response, "choices", None):
            provider_usage = add_usage(getattr(response, "usage", None))
            finalize_call_usage()
            message = response.choices[0].message
            reasoning_text, content_text = _extract_reasoning_and_content(message)
            tool_calls, tool_call_error = _extract_native_tool_calls(message)
            _trace_agent_event(
                "model_turn_completed",
                trace_id=trace_id,
                step=step,
                reasoning_excerpt=reasoning_text,
                content_excerpt=content_text,
                tool_calls=tool_calls or [],
            )
            for event in emit_turn_reasoning(reasoning_text):
                yield event
            return {
                "reasoning_text": reasoning_text,
                "content_text": content_text,
                "tool_calls": tool_calls,
                "tool_call_error": tool_call_error,
                "stream_error": None,
            }

        reasoning_parts = []
        content_parts = []
        buffered_content_deltas = []
        tool_call_parts = []
        content_streaming_live = False
        stream_error = None
        announced_canvas_tool = None
        streamed_canvas_content_length = 0

        try:
            for chunk in response:
                reasoning_delta, content_delta = _extract_stream_delta_texts(chunk)
                if reasoning_delta:
                    reasoning_parts.append(reasoning_delta)
                    for event in emit_turn_reasoning(reasoning_delta):
                        yield event
                if getattr(chunk, "choices", None):
                    delta = getattr(chunk.choices[0], "delta", None)
                    if delta is not None:
                        _merge_stream_tool_call_delta(tool_call_parts, delta)
                        canvas_preview = _build_streaming_canvas_tool_preview(tool_call_parts)
                        if canvas_preview is not None:
                            preview_tool_name = canvas_preview["tool"]
                            if announced_canvas_tool != preview_tool_name:
                                announced_canvas_tool = preview_tool_name
                                streamed_canvas_content_length = 0
                                yield {
                                    "type": "canvas_tool_starting",
                                    "tool": preview_tool_name,
                                    "snapshot": canvas_preview["snapshot"],
                                }
                            preview_content = canvas_preview.get("content")
                            if preview_content is not None and len(preview_content) > streamed_canvas_content_length:
                                next_content_delta = preview_content[streamed_canvas_content_length:]
                                streamed_canvas_content_length = len(preview_content)
                                if next_content_delta:
                                    yield {
                                        "type": "canvas_content_delta",
                                        "tool": preview_tool_name,
                                        "delta": next_content_delta,
                                        "snapshot": canvas_preview["snapshot"],
                                    }
                if content_delta:
                    content_parts.append(content_delta)
                    if not turn_tools:
                        for event in emit_answer(content_delta):
                            yield event
                    elif content_streaming_live:
                        for event in emit_answer(content_delta):
                            yield event
                    elif tool_call_parts:
                        buffered_content_deltas.append(content_delta)
                    else:
                        content_streaming_live = True
                        for event in emit_answer(content_delta):
                            yield event
                if getattr(chunk, "usage", None):
                    usage_snapshot = add_usage(chunk.usage)
                    provider_usage["prompt_tokens"] += usage_snapshot["prompt_tokens"]
                    provider_usage["completion_tokens"] += usage_snapshot["completion_tokens"]
                    provider_usage["total_tokens"] += usage_snapshot["total_tokens"]
                    provider_usage["received"] = provider_usage["received"] or usage_snapshot["received"]
        except Exception as exc:
            stream_error = str(exc)
            _trace_agent_event(
                "model_stream_interrupted",
                trace_id=trace_id,
                step=step,
                error=stream_error,
                partial_content_excerpt="".join(content_parts),
            )

        final_reasoning = "".join(reasoning_parts).strip()
        final_content = "".join(content_parts).strip()
        tool_calls, tool_call_error = _finalize_stream_tool_calls(tool_call_parts)
        finalize_call_usage()
        if buffered_content_deltas and not tool_calls and not tool_call_error:
            for pending_delta in buffered_content_deltas:
                for event in emit_answer(pending_delta):
                    yield event

        _trace_agent_event(
            "model_turn_completed",
            trace_id=trace_id,
            step=step,
            reasoning_excerpt=final_reasoning,
            content_excerpt=final_content,
            tool_calls=tool_calls or [],
            stream_error=stream_error,
        )
        return {
            "reasoning_text": final_reasoning,
            "content_text": final_content,
            "tool_calls": tool_calls,
            "tool_call_error": tool_call_error,
            "stream_error": stream_error,
        }

    pending_step_retry_reason: str | None = None
    while step < max_steps:
        step += 1
        yield {"type": "step_started", "step": step, "max_steps": max_steps}
        _trace_agent_event("agent_step_started", trace_id=trace_id, step=step, max_steps=max_steps)
        context_compacted_this_step = False
        needs_separator_for_sync = pending_answer_separator
        step_retry_reason = pending_step_retry_reason
        pending_step_retry_reason = None
        reasoning_replay_instruction = _build_reasoning_replay_instruction(
            reasoning_state,
            current_goal=working_state.get("current_goal") or "",
        )
        working_memory_instruction = _build_working_state_instruction(working_state)
        extra_messages = []
        if reasoning_replay_instruction:
            extra_messages.append(reasoning_replay_instruction)
            _trace_agent_event(
                "reasoning_replay_injected",
                trace_id=trace_id,
                step=step,
                entry_count=len(reasoning_state.get("entries") or []),
            )
        if working_memory_instruction:
            extra_messages.append(working_memory_instruction)
        turn_messages, _ = apply_context_compaction(extra_messages, reason="pre_model_turn")

        try:
            turn_result = yield from stream_model_turn(
                turn_messages,
                call_type="agent_step",
                retry_reason=step_retry_reason,
            )
        except Exception as exc:
            fatal_api_error = str(exc)
            if _is_context_overflow_error(fatal_api_error) and not context_compacted_this_step:
                _, compacted = apply_context_compaction(extra_messages, reason="reactive_model_turn", force=True)
                if compacted:
                    context_compacted_this_step = True
                    _trace_agent_event(
                        "context_overflow_recovered",
                        trace_id=trace_id,
                        step=step,
                        phase="main_loop",
                        source="model_turn_exception",
                    )
                    pending_step_retry_reason = "context_overflow_recovery"
                    step -= 1
                    continue
                _trace_agent_event(
                    "context_overflow_unrecoverable",
                    trace_id=trace_id,
                    step=step,
                    phase="main_loop",
                    source="model_turn_exception",
                    error=fatal_api_error,
                    message_count=len(turn_messages),
                )
                fatal_api_error = CONTEXT_OVERFLOW_RECOVERY_ERROR_TEXT
            _trace_agent_event("agent_api_error", trace_id=trace_id, step=step, error=fatal_api_error)
            yield {"type": "tool_error", "step": step, "tool": "api", "error": fatal_api_error}
            break

        reasoning_text = turn_result.get("reasoning_text") or ""
        content_text = turn_result.get("content_text") or ""
        tool_calls = turn_result.get("tool_calls")
        tool_call_error = turn_result.get("tool_call_error")
        stream_error = turn_result.get("stream_error")

        if tool_call_error:
            _trace_agent_event(
                "tool_parse_error",
                trace_id=trace_id,
                step=step,
                parse_error=tool_call_error,
                content_excerpt=content_text,
            )
            yield {"type": "tool_error", "step": step, "tool": "parser", "error": tool_call_error}
            break

        if content_text and not tool_calls:
            if needs_separator_for_sync and content_text.strip():
                total_clean_content += "\n\n"
            total_clean_content += content_text

        _trace_agent_event(
            "tool_parse_result",
            trace_id=trace_id,
            step=step,
            tool_calls=tool_calls or [],
            content_excerpt=content_text,
        )

        if tool_calls and any(call["name"] == "ask_clarifying_question" for call in tool_calls) and len(tool_calls) > 1:
            yield {
                "type": "tool_error",
                "step": step,
                "tool": "ask_clarifying_question",
                "error": "ask_clarifying_question must be the only tool call in a single assistant turn.",
            }
            break

        if not tool_calls:
            if content_text:
                _trace_agent_event("final_answer_received", trace_id=trace_id, step=step, content_excerpt=content_text)
                if not answer_started:
                    for event in emit_answer(content_text):
                        yield event
                if stream_error:
                    yield {"type": "tool_error", "step": step, "tool": "api", "error": stream_error}
                if usage_totals["total_tokens"]:
                    yield usage_event()
                yield build_tool_capture_event()
                yield {"type": "done"}
                return

            if stream_error:
                if _is_context_overflow_error(stream_error) and not context_compacted_this_step:
                    _, compacted = apply_context_compaction(extra_messages, reason="reactive_stream_error", force=True)
                    if compacted:
                        context_compacted_this_step = True
                        _trace_agent_event(
                            "context_overflow_recovered",
                            trace_id=trace_id,
                            step=step,
                            phase="main_loop",
                            source="stream_error",
                        )
                        pending_step_retry_reason = "context_overflow_recovery"
                        step -= 1
                        continue
                    _trace_agent_event(
                        "context_overflow_unrecoverable",
                        trace_id=trace_id,
                        step=step,
                        phase="main_loop",
                        source="stream_error",
                        error=stream_error,
                        message_count=len(turn_messages),
                    )
                    fatal_api_error = CONTEXT_OVERFLOW_RECOVERY_ERROR_TEXT
                else:
                    fatal_api_error = stream_error
                _trace_agent_event("agent_api_error", trace_id=trace_id, step=step, error=fatal_api_error)
                yield {"type": "tool_error", "step": step, "tool": "api", "error": fatal_api_error}
                break

            _trace_agent_event("missing_final_answer", trace_id=trace_id, step=step)
            yield {
                "type": "tool_error",
                "step": step,
                "tool": "agent",
                "error": "The model returned no final answer content. Retrying and waiting for a final answer.",
            }
            if not _has_missing_final_answer_instruction(messages):
                messages.append(_build_missing_final_answer_instruction())
            pending_step_retry_reason = "missing_final_answer"
            continue

        _append_reasoning_replay_entry(reasoning_state, step, reasoning_text, tool_calls)
        if reasoning_text:
            _trace_agent_event(
                "reasoning_replay_updated",
                trace_id=trace_id,
                step=step,
                chars=len(reasoning_text),
                tool_names=[
                    str(tool_call.get("name") or "").strip()
                    for tool_call in (tool_calls or [])
                    if str(tool_call.get("name") or "").strip()
                ],
            )
        assistant_tool_call_message = _build_assistant_tool_call_message(content_text, tool_calls)
        messages.append(assistant_tool_call_message)
        if content_text.strip() and answer_started:
            pending_answer_separator = True
        transcript_results = []
        tool_messages = []

        # ---- Phase 1: validate, pre-check, build execution slots (sequential) ----
        slots = []
        for call_index, tool_call in enumerate(tool_calls, start=1):
            tool_name = tool_call["name"]
            tool_args = tool_call["arguments"]
            call_id = str(tool_call.get("id") or f"step-{step}-call-{call_index}-{tool_name}")
            slot = {
                "call_index": call_index,
                "tool_name": tool_name,
                "tool_args": tool_args,
                "call_id": call_id,
                "preview": "",
                "cache_key": "",
                "has_step_update": False,
            }

            if tool_name not in enabled_tool_names:
                slot["kind"] = "error"
                slot["error"] = f"Tool disabled: {tool_name}"
                slots.append(slot)
                continue

            validation_error = _validate_tool_arguments(tool_name, tool_args)
            if validation_error:
                slot["kind"] = "error"
                slot["error"] = validation_error
                slots.append(slot)
                continue

            cache_key = build_tool_cache_key(tool_name, tool_args)
            slot["cache_key"] = cache_key
            if tool_name == "search_knowledge_base":
                preview = tool_args.get("query", "")[:80]
            elif tool_name in ("search_web", "search_news_ddgs", "search_news_google"):
                preview = ", ".join(str(query) for query in tool_args.get("queries", []))[:80]
            elif tool_name == "fetch_url":
                preview = tool_args.get("url", "")[:80]
            else:
                preview = ""
            slot["preview"] = preview

            if tool_name == "fetch_url":
                fetch_url_val = str(tool_args.get("url") or "").strip()
                fetch_attempt_counts[fetch_url_val] = fetch_attempt_counts.get(fetch_url_val, 0) + 1
                _trace_agent_event(
                    "fetch_url_requested",
                    trace_id=trace_id,
                    step=step,
                    url=fetch_url_val,
                    attempt_count=fetch_attempt_counts[fetch_url_val],
                    repeated=fetch_attempt_counts[fetch_url_val] > 1,
                    call_id=call_id,
                )
                if fetch_attempt_counts[fetch_url_val] > 1:
                    _trace_agent_event(
                        "duplicate_fetch_attempt",
                        trace_id=trace_id,
                        step=step,
                        url=fetch_url_val,
                        attempt_count=fetch_attempt_counts[fetch_url_val],
                        call_id=call_id,
                    )

            _trace_agent_event(
                "tool_call_started",
                trace_id=trace_id,
                step=step,
                tool_name=tool_name,
                tool_args=tool_args,
                preview=preview,
                cache_key=cache_key,
            )
            _append_working_state_attempt(working_state, tool_name, preview)
            slot["has_step_update"] = True

            tool_limit = _get_tool_step_limit(tool_name, max_steps)
            if tool_call_counts[tool_name] >= tool_limit:
                error = f"Per-tool step limit reached for {tool_name}. Try a different tool or produce the best available answer."
                _append_working_state_blocker(working_state, tool_name, error)
                slot["kind"] = "error"
                slot["error"] = error
                slots.append(slot)
                continue
            tool_call_counts[tool_name] += 1

            if tool_name not in CANVAS_TOOL_NAMES and cache_key in tool_result_cache:
                cached_result, cached_summary = tool_result_cache[cache_key]
                transcript_result = _prepare_tool_result_for_transcript(
                    tool_name,
                    cached_result,
                    fetch_url_token_threshold=fetch_url_token_threshold,
                    fetch_url_clip_aggressiveness=fetch_url_clip_aggressiveness,
                )
                remember_tool_result(
                    tool_name,
                    tool_args,
                    cached_result,
                    cached_summary,
                    cache_key,
                    transcript_result=transcript_result,
                )
                storage_entry = _build_tool_result_storage_entry(
                    tool_name,
                    tool_args,
                    cached_result,
                    cached_summary,
                    transcript_result=transcript_result,
                )
                _trace_agent_event(
                    "tool_cache_hit",
                    trace_id=trace_id,
                    step=step,
                    tool_name=tool_name,
                    tool_args=tool_args,
                    summary=cached_summary,
                    transcript_result=transcript_result,
                )
                slot["kind"] = "session_cache_hit"
                slot["result"] = cached_result
                slot["summary"] = cached_summary
                slot["transcript_result"] = transcript_result
                slot["storage_entry"] = storage_entry
                slots.append(slot)
                continue

            cross_turn_cache_hit = _lookup_cross_turn_tool_memory(tool_name, tool_args)
            if cross_turn_cache_hit is not None:
                cached_excerpt, cached_summary = cross_turn_cache_hit
                _trace_agent_event(
                    "tool_memory_cache_hit",
                    trace_id=trace_id,
                    step=step,
                    tool_name=tool_name,
                    tool_args=tool_args,
                    summary=cached_summary,
                )
                slot["kind"] = "memory_cache_hit"
                slot["result"] = cached_excerpt
                slot["summary"] = cached_summary
                slot["transcript_result"] = cached_excerpt
                slots.append(slot)
                continue

            slot["kind"] = "execute"
            slot["is_canvas"] = tool_name in CANVAS_TOOL_NAMES
            slots.append(slot)

        # ---- Phase 1b: yield step_update events for all non-error, non-disabled calls ----
        for slot in slots:
            if slot.get("has_step_update"):
                yield {
                    "type": "step_update",
                    "step": step,
                    "tool": slot["tool_name"],
                    "preview": slot["preview"],
                    "call_id": slot["call_id"],
                }

        # ---- Phase 2: execute pending slots (parallel for safe read-only tools, sequential for mutators) ----
        pending_slots = [s for s in slots if s["kind"] == "execute"]
        if pending_slots:
            parallel_slots = [s for s in pending_slots if s["tool_name"] in PARALLEL_SAFE_TOOL_NAMES]
            sequential_slots = [s for s in pending_slots if s["tool_name"] not in PARALLEL_SAFE_TOOL_NAMES]

            if len(parallel_slots) > 1:
                def _run_slot(s):
                    try:
                        res, summ = _execute_tool(s["tool_name"], s["tool_args"], runtime_state=runtime_state)
                        return {"ok": True, "result": res, "summary": summ}
                    except Exception as exc:
                        return {"ok": False, "error": str(exc)}

                with ThreadPoolExecutor(max_workers=len(parallel_slots)) as executor:
                    futures_list = [(executor.submit(_run_slot, s), s) for s in parallel_slots]
                for future, s in futures_list:
                    s["exec_result"] = future.result()
            else:
                for s in parallel_slots:
                    try:
                        res, summ = _execute_tool(s["tool_name"], s["tool_args"], runtime_state=runtime_state)
                        s["exec_result"] = {"ok": True, "result": res, "summary": summ}
                    except Exception as exc:
                        s["exec_result"] = {"ok": False, "error": str(exc)}

            for s in sequential_slots:
                try:
                    res, summ = _execute_tool(s["tool_name"], s["tool_args"], runtime_state=runtime_state)
                    s["exec_result"] = {"ok": True, "result": res, "summary": summ}
                except Exception as exc:
                    s["exec_result"] = {"ok": False, "error": str(exc)}

        # ---- Phase 3: post-process all slots in original order ----
        for slot in slots:
            kind = slot["kind"]
            tool_name = slot["tool_name"]
            tool_args = slot["tool_args"]
            call_id = slot["call_id"]
            preview = slot["preview"]
            cache_key = slot["cache_key"]

            if kind == "error":
                error = slot["error"]
                yield {"type": "tool_error", "step": step, "tool": tool_name, "error": error, "call_id": call_id}
                tool_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": _serialize_tool_message_content({"ok": False, "error": error}),
                    }
                )
                transcript_results.append({"tool_name": tool_name, "arguments": tool_args, "ok": False, "error": error})

            elif kind == "session_cache_hit":
                result = slot["result"]
                summary = slot["summary"]
                transcript_result = slot["transcript_result"]
                storage_entry = slot["storage_entry"]
                yield {
                    "type": "tool_result",
                    "step": step,
                    "tool": tool_name,
                    "summary": f"{summary} (cached)",
                    "call_id": call_id,
                    "cached": True,
                }
                tool_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": _build_compact_tool_message_content(
                            tool_name,
                            tool_args,
                            result,
                            f"{summary} (cached)",
                            transcript_result=transcript_result,
                            storage_entry=storage_entry,
                        ),
                    }
                )
                transcript_results.append(
                    {
                        "tool_name": tool_name,
                        "arguments": tool_args,
                        "ok": not (tool_name == "fetch_url" and isinstance(result, dict) and result.get("error")),
                        "summary": f"{summary} (cached)",
                        "result": transcript_result,
                        "cached": True,
                    }
                )

            elif kind == "memory_cache_hit":
                result = slot["result"]
                summary = slot["summary"]
                transcript_result = slot["transcript_result"]
                yield {
                    "type": "tool_result",
                    "step": step,
                    "tool": tool_name,
                    "summary": f"{summary} (cached)",
                    "call_id": call_id,
                    "cached": True,
                }
                tool_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": _build_compact_tool_message_content(
                            tool_name,
                            tool_args,
                            result,
                            f"{summary} (cached)",
                            transcript_result=transcript_result,
                        ),
                    }
                )
                transcript_results.append(
                    {
                        "tool_name": tool_name,
                        "arguments": tool_args,
                        "ok": True,
                        "summary": f"{summary} (cached)",
                        "result": transcript_result,
                        "cached": True,
                    }
                )

            elif kind == "execute":
                exec_result = slot["exec_result"]
                if exec_result["ok"]:
                    result = exec_result["result"]
                    summary = exec_result["summary"]
                    transcript_result = _prepare_tool_result_for_transcript(
                        tool_name,
                        result,
                        fetch_url_token_threshold=fetch_url_token_threshold,
                        fetch_url_clip_aggressiveness=fetch_url_clip_aggressiveness,
                    )
                    if tool_name not in CANVAS_TOOL_NAMES:
                        tool_result_cache[cache_key] = (result, summary)
                    storage_entry = _build_tool_result_storage_entry(
                        tool_name,
                        tool_args,
                        result,
                        summary,
                        transcript_result=transcript_result,
                    )
                    if storage_entry and cache_key not in persisted_tool_cache_keys:
                        persisted_tool_cache_keys.add(cache_key)
                        persisted_tool_results.append(storage_entry)
                    if tool_name in WEB_TOOL_NAMES and storage_entry:
                        try:
                            upsert_tool_memory_result(
                                tool_name,
                                storage_entry.get("input_preview", ""),
                                storage_entry.get("content", ""),
                                storage_entry.get("summary", ""),
                            )
                        except Exception as exc:
                            _trace_agent_event(
                                "tool_memory_upsert_failed",
                                trace_id=trace_id,
                                step=step,
                                tool_name=tool_name,
                                tool_args=tool_args,
                                error=str(exc),
                            )
                    _trace_agent_event(
                        "tool_call_completed",
                        trace_id=trace_id,
                        step=step,
                        tool_name=tool_name,
                        tool_args=tool_args,
                        summary=summary,
                        result=result,
                        transcript_result=transcript_result,
                    )
                    yield {
                        "type": "tool_result",
                        "step": step,
                        "tool": tool_name,
                        "summary": summary,
                        "call_id": call_id,
                        "cached": False,
                    }
                    tool_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": _build_compact_tool_message_content(
                                tool_name,
                                tool_args,
                                result,
                                summary,
                                transcript_result=transcript_result,
                                storage_entry=storage_entry,
                            ),
                        }
                    )
                    transcript_results.append(
                        {
                            "tool_name": tool_name,
                            "arguments": tool_args,
                            "ok": not (tool_name == "fetch_url" and isinstance(result, dict) and result.get("error")),
                            "summary": summary,
                            "result": transcript_result,
                        }
                    )
                    if tool_name in CANVAS_MUTATION_TOOL_NAMES:
                        canvas_modified = True
                    clarification_event = _extract_clarification_event(result)
                    if clarification_event is not None:
                        _trace_agent_event(
                            "clarification_requested",
                            trace_id=trace_id,
                            step=step,
                            clarification=clarification_event.get("clarification"),
                        )
                        yield {
                            "type": "tool_history",
                            "step": step,
                            "messages": [assistant_tool_call_message, *tool_messages],
                        }
                        yield clarification_event
                        if usage_totals["total_tokens"]:
                            yield usage_event()
                        yield build_tool_capture_event()
                        yield {"type": "done"}
                        return
                else:
                    error = exec_result["error"]
                    _append_working_state_blocker(working_state, tool_name, error)
                    _trace_agent_event(
                        "tool_call_failed",
                        trace_id=trace_id,
                        step=step,
                        tool_name=tool_name,
                        tool_args=tool_args,
                        error=error,
                    )
                    yield {"type": "tool_error", "step": step, "tool": tool_name, "error": error, "call_id": call_id}
                    tool_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": _serialize_tool_message_content({"ok": False, "error": error}),
                        }
                    )
                    transcript_results.append(
                        {
                            "tool_name": tool_name,
                            "arguments": tool_args,
                            "ok": False,
                            "error": error,
                        }
                    )

        _trace_agent_event(
            "tool_transcript_appended",
            trace_id=trace_id,
            step=step,
            transcript_results=transcript_results,
        )
        yield {
            "type": "tool_history",
            "step": step,
            "messages": [assistant_tool_call_message, *tool_messages],
        }
        messages.extend(tool_messages)
        tool_execution_result_message = _build_tool_execution_result_message(transcript_results)
        if tool_execution_result_message is not None:
            messages.append(tool_execution_result_message)

    if fatal_api_error is not None:
        if not answer_started:
            for event in emit_answer(FINAL_ANSWER_ERROR_TEXT):
                yield event
        if usage_totals["total_tokens"]:
            yield usage_event()
        yield build_tool_capture_event()
        yield {"type": "done"}
        return

    final_phase_compaction_used = False
    final_instruction_builder = _build_final_answer_instruction
    pending_final_retry_reason: str | None = None
    while True:
        final_extra_messages = []
        try:
            _trace_agent_event("final_answer_phase_started", trace_id=trace_id, step=step)
            final_retry_reason = pending_final_retry_reason
            pending_final_retry_reason = None
            working_memory_instruction = _build_working_state_instruction(working_state)
            final_extra_messages = [working_memory_instruction] if working_memory_instruction is not None else []
            final_messages, _ = apply_context_compaction(final_extra_messages, reason="pre_final_answer")
            final_messages = [*final_messages, final_instruction_builder()]
            turn_result = yield from stream_model_turn(
                final_messages,
                allow_tools=False,
                call_type="final_answer",
                retry_reason=final_retry_reason,
            )
            content_text = turn_result.get("content_text") or ""
            tool_calls = turn_result.get("tool_calls")
            stream_error = turn_result.get("stream_error")
            if stream_error and _is_context_overflow_error(stream_error) and not final_phase_compaction_used:
                _, compacted = apply_context_compaction(final_extra_messages, reason="reactive_final_stream", force=True)
                if compacted:
                    final_phase_compaction_used = True
                    _trace_agent_event(
                        "context_overflow_recovered",
                        trace_id=trace_id,
                        step=step,
                        phase="final_answer",
                        source="stream_error",
                    )
                    pending_final_retry_reason = "context_overflow_recovery"
                    continue
                if final_instruction_builder is _build_final_answer_instruction:
                    _trace_agent_event(
                        "context_overflow_minimal_final_instruction",
                        trace_id=trace_id,
                        step=step,
                        phase="final_answer",
                        source="stream_error",
                    )
                    final_phase_compaction_used = True
                    final_instruction_builder = _build_minimal_final_answer_instruction
                    pending_final_retry_reason = "minimal_final_instruction"
                    continue
                _trace_agent_event(
                    "context_overflow_unrecoverable",
                    trace_id=trace_id,
                    step=step,
                    phase="final_answer",
                    source="stream_error",
                    error=stream_error,
                    message_count=len(final_messages),
                )
                stream_error = CONTEXT_OVERFLOW_RECOVERY_ERROR_TEXT
            if tool_calls:
                yield {
                    "type": "tool_error",
                    "step": step,
                    "tool": "agent",
                    "error": "Tool limit reached before the model produced a final answer.",
                }
                final_text = FINAL_ANSWER_ERROR_TEXT
            elif not content_text:
                yield {
                    "type": "tool_error",
                    "step": step,
                    "tool": "agent",
                    "error": "The model still did not provide a final answer in assistant content.",
                }
                final_text = FINAL_ANSWER_MISSING_TEXT
            else:
                final_text = content_text
            if stream_error:
                yield {"type": "tool_error", "step": step, "tool": "final_answer", "error": stream_error}
            for event in emit_answer(final_text):
                yield event
            break
        except Exception as exc:
            error = str(exc)
            if _is_context_overflow_error(error) and not final_phase_compaction_used:
                _, compacted = apply_context_compaction(final_extra_messages, reason="reactive_final_answer", force=True)
                if compacted:
                    final_phase_compaction_used = True
                    _trace_agent_event(
                        "context_overflow_recovered",
                        trace_id=trace_id,
                        step=step,
                        phase="final_answer",
                        source="exception",
                    )
                    pending_final_retry_reason = "context_overflow_recovery"
                    continue
                if final_instruction_builder is _build_final_answer_instruction:
                    _trace_agent_event(
                        "context_overflow_minimal_final_instruction",
                        trace_id=trace_id,
                        step=step,
                        phase="final_answer",
                        source="exception",
                    )
                    final_phase_compaction_used = True
                    final_instruction_builder = _build_minimal_final_answer_instruction
                    pending_final_retry_reason = "minimal_final_instruction"
                    continue
                _trace_agent_event(
                    "context_overflow_unrecoverable",
                    trace_id=trace_id,
                    step=step,
                    phase="final_answer",
                    source="exception",
                    error=error,
                    message_count=len([*messages, *final_extra_messages]),
                )
                error = CONTEXT_OVERFLOW_RECOVERY_ERROR_TEXT
            yield {"type": "tool_error", "step": step, "tool": "final_answer", "error": error}
            for event in emit_answer(FINAL_ANSWER_ERROR_TEXT):
                yield event
            break

    if usage_totals["total_tokens"]:
        yield usage_event()
    yield build_tool_capture_event()
    yield {"type": "done"}
