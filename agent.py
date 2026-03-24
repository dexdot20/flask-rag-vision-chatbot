from __future__ import annotations

import ast
import hashlib
import json
import logging
import os
import re
from logging.handlers import RotatingFileHandler
from uuid import uuid4

from canvas_service import (
    build_canvas_tool_result,
    clear_canvas,
    create_canvas_document,
    create_canvas_runtime_state,
    delete_canvas_document,
    delete_canvas_lines,
    get_canvas_runtime_documents,
    insert_canvas_lines,
    replace_canvas_lines,
    rewrite_canvas_document,
)
from config import (
    AGENT_TRACE_LOG_PATH,
    FETCH_RAW_TOOL_RESULT_MAX_TEXT_CHARS,
    FETCH_SUMMARY_MAX_CHARS,
    FETCH_SUMMARY_TOKEN_THRESHOLD,
    RAG_SEARCH_DEFAULT_TOP_K,
    RAG_TOOL_RESULT_MAX_TEXT_CHARS,
    RAG_TOOL_RESULT_SUMMARY_MAX_CHARS,
    client,
)
from db import append_to_scratchpad, read_image_asset_bytes, replace_scratchpad
from rag_service import search_knowledge_base_tool, search_tool_memory, upsert_tool_memory_result
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
CANVAS_TOOL_NAMES = {
    "create_canvas_document",
    "rewrite_canvas_document",
    "replace_canvas_lines",
    "insert_canvas_lines",
    "delete_canvas_lines",
    "delete_canvas_document",
    "clear_canvas",
}
WEB_TOOL_NAMES = {
    "search_web",
    "fetch_url",
    "search_news_ddgs",
    "search_news_google",
}
INPUT_BREAKDOWN_KEYS = (
    "system_prompt",
    "user_messages",
    "assistant_history",
    "tool_results",
    "rag_context",
    "final_instruction",
)
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


def _estimate_message_breakdown(message: dict) -> dict[str, int]:
    role = str(message.get("role") or "").strip()
    content = str(message.get("content") or "")
    total_tokens = _estimate_text_tokens(content)
    if total_tokens <= 0:
        return {}

    if role == "user":
        return {"user_messages": total_tokens}
    if role == "assistant":
        return {"assistant_history": total_tokens}
    if role == "tool":
        return {"tool_results": total_tokens}
    if role != "system":
        return {"system_prompt": total_tokens}

    # Classify system messages by their distinctive markers
    if content.startswith("[TOOL EXECUTION RESULTS]"):
        return {"tool_results": total_tokens}
    if content.startswith("[INSTRUCTION: FINAL ANSWER REQUIRED]"):
        return {"final_instruction": total_tokens}
    if content.startswith("[INSTRUCTION: MISSING FINAL ANSWER"):
        return {"final_instruction": total_tokens}

    # Extract RAG context token count by identifying the knowledge base section in markdown
    rag_tokens = 0
    if "## Knowledge Base" in content:
        match = re.search(r"## Knowledge Base(.*?)(?:\n## |\Z)", content, flags=re.DOTALL)
        if match:
            rag_context_text = match.group(1)
            rag_tokens = min(total_tokens, _estimate_text_tokens(rag_context_text))

    system_tokens = max(total_tokens - rag_tokens, 0)
    if rag_tokens > 0 and system_tokens == 0 and total_tokens > 0:
        system_tokens = 1
        rag_tokens = max(total_tokens - 1, 0)

    breakdown = {}
    if system_tokens:
        breakdown["system_prompt"] = system_tokens
    if rag_tokens:
        breakdown["rag_context"] = rag_tokens
    return breakdown or {"system_prompt": total_tokens}


def _estimate_input_breakdown(messages_to_send: list[dict]) -> tuple[dict[str, int], int]:
    breakdown = _empty_input_breakdown()
    for message in messages_to_send:
        for key, value in _estimate_message_breakdown(message).items():
            if key in breakdown and value > 0:
                breakdown[key] += value
    return breakdown, sum(breakdown.values())


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
        "[TOOL EXECUTION RESULTS]\n",
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
    return append_to_scratchpad(tool_args.get("note", ""))


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
    )
    return build_canvas_tool_result(document, action="created"), f"Canvas created: {document['title']}"


def _run_rewrite_canvas_document(tool_args: dict, runtime_state: dict):
    canvas_state = _get_canvas_runtime_state(runtime_state)
    document = rewrite_canvas_document(
        canvas_state,
        content=tool_args.get("content", ""),
        document_id=tool_args.get("document_id"),
        title=tool_args.get("title"),
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
    )
    return build_canvas_tool_result(document, action="lines_replaced"), f"Canvas lines replaced in {document['title']}"


def _run_insert_canvas_lines(tool_args: dict, runtime_state: dict):
    canvas_state = _get_canvas_runtime_state(runtime_state)
    document = insert_canvas_lines(
        canvas_state,
        after_line=tool_args.get("after_line", 0),
        lines=tool_args.get("lines", []),
        document_id=tool_args.get("document_id"),
    )
    return build_canvas_tool_result(document, action="lines_inserted"), f"Canvas lines inserted in {document['title']}"


def _run_delete_canvas_lines(tool_args: dict, runtime_state: dict):
    canvas_state = _get_canvas_runtime_state(runtime_state)
    document = delete_canvas_lines(
        canvas_state,
        start_line=tool_args.get("start_line", 0),
        end_line=tool_args.get("end_line", 0),
        document_id=tool_args.get("document_id"),
    )
    return build_canvas_tool_result(document, action="lines_deleted"), f"Canvas lines deleted in {document['title']}"


def _run_delete_canvas_document(tool_args: dict, runtime_state: dict):
    canvas_state = _get_canvas_runtime_state(runtime_state)
    result = delete_canvas_document(
        canvas_state,
        document_id=tool_args.get("document_id"),
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


def run_agent_stream(
    api_messages: list,
    model: str,
    max_steps: int,
    enabled_tool_names: list[str],
    fetch_url_token_threshold: int | None = None,
    fetch_url_clip_aggressiveness: int | None = None,
    initial_canvas_documents: list[dict] | None = None,
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
    canvas_modified = False
    usage_totals = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "estimated_input_tokens": 0,
        "input_breakdown": _empty_input_breakdown(),
    }
    runtime_state = {
        "canvas": create_canvas_runtime_state(initial_canvas_documents),
    }

    def build_tool_capture_event() -> dict:
        current_canvas_documents = get_canvas_runtime_documents(runtime_state.get("canvas"))
        return {
            "type": "tool_capture",
            "tool_results": persisted_tool_results,
            "canvas_documents": current_canvas_documents,
            "canvas_cleared": canvas_modified and not current_canvas_documents,
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
            return
        usage_totals["prompt_tokens"] += usage.prompt_tokens or 0
        usage_totals["completion_tokens"] += usage.completion_tokens or 0
        usage_totals["total_tokens"] += usage.total_tokens or 0

    def calculate_cost(prompt_tokens, completion_tokens):
        input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
        output_cost = (completion_tokens / 1_000_000) * pricing["output"]
        return round(input_cost + output_cost, 6)

    def usage_event():
        total_cost = calculate_cost(usage_totals["prompt_tokens"], usage_totals["completion_tokens"])
        return {
            "type": "usage",
            "prompt_tokens": usage_totals["prompt_tokens"],
            "completion_tokens": usage_totals["completion_tokens"],
            "total_tokens": usage_totals["total_tokens"],
            "estimated_input_tokens": usage_totals["estimated_input_tokens"],
            "input_breakdown": dict(usage_totals["input_breakdown"]),
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

    def stream_model_turn(messages_to_send: list[dict], allow_tools: bool = True) -> dict:
        turn_reasoning_emitted = False
        turn_tools = []
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
            turn_tools = get_openai_tool_specs(enabled_tool_names)
            if turn_tools:
                request_kwargs["tools"] = turn_tools
                request_kwargs["tool_choice"] = "auto"
        response = client.chat.completions.create(**request_kwargs)
        estimated_breakdown, estimated_input_tokens = _estimate_input_breakdown(messages_to_send)
        usage_totals["estimated_input_tokens"] += estimated_input_tokens
        for key, value in estimated_breakdown.items():
            usage_totals["input_breakdown"][key] += value

        if getattr(response, "choices", None):
            add_usage(getattr(response, "usage", None))
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
        _CONTENT_FLUSH_CHAR_THRESHOLD = 80

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
                        buffered_content_deltas.append(content_delta)
                        buffered_chars = sum(len(d) for d in buffered_content_deltas)
                        if buffered_chars >= _CONTENT_FLUSH_CHAR_THRESHOLD:
                            content_streaming_live = True
                            for pending in buffered_content_deltas:
                                for event in emit_answer(pending):
                                    yield event
                            buffered_content_deltas = []
                if getattr(chunk, "usage", None):
                    add_usage(chunk.usage)
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

    while step < max_steps:
        step += 1
        yield {"type": "step_started", "step": step, "max_steps": max_steps}
        _trace_agent_event("agent_step_started", trace_id=trace_id, step=step, max_steps=max_steps)

        try:
            needs_separator_for_sync = pending_answer_separator
            turn_result = yield from stream_model_turn(messages)
        except Exception as exc:
            fatal_api_error = str(exc)
            _trace_agent_event("agent_api_error", trace_id=trace_id, step=step, error=fatal_api_error)
            yield {"type": "tool_error", "step": step, "tool": "api", "error": fatal_api_error}
            break

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
            messages.append(_build_missing_final_answer_instruction())
            continue

        assistant_tool_call_message = _build_assistant_tool_call_message(content_text, tool_calls)
        messages.append(assistant_tool_call_message)
        if content_text.strip() and answer_started:
            pending_answer_separator = True
        transcript_results = []
        tool_messages = []

        for call_index, tool_call in enumerate(tool_calls, start=1):
            tool_name = tool_call["name"]
            tool_args = tool_call["arguments"]
            call_id = str(tool_call.get("id") or f"step-{step}-call-{call_index}-{tool_name}")
            if tool_name not in enabled_tool_names:
                error = f"Tool disabled: {tool_name}"
                yield {"type": "tool_error", "step": step, "tool": tool_name, "error": error, "call_id": call_id}
                tool_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": _serialize_tool_message_content({"ok": False, "error": error}),
                    }
                )
                transcript_results.append({"tool_name": tool_name, "arguments": tool_args, "ok": False, "error": error})
                continue

            validation_error = _validate_tool_arguments(tool_name, tool_args)
            if validation_error:
                yield {
                    "type": "tool_error",
                    "step": step,
                    "tool": tool_name,
                    "error": validation_error,
                    "call_id": call_id,
                }
                tool_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": _serialize_tool_message_content({"ok": False, "error": validation_error}),
                    }
                )
                transcript_results.append(
                    {"tool_name": tool_name, "arguments": tool_args, "ok": False, "error": validation_error}
                )
                continue

            cache_key = build_tool_cache_key(tool_name, tool_args)
            if tool_name == "search_knowledge_base":
                preview = tool_args.get("query", "")[:80]
            elif tool_name in ("search_web", "search_news_ddgs", "search_news_google"):
                preview = ", ".join(str(query) for query in tool_args.get("queries", []))[:80]
            elif tool_name == "fetch_url":
                preview = tool_args.get("url", "")[:80]
            else:
                preview = ""

            if tool_name == "fetch_url":
                fetch_url = str(tool_args.get("url") or "").strip()
                fetch_attempt_counts[fetch_url] = fetch_attempt_counts.get(fetch_url, 0) + 1
                _trace_agent_event(
                    "fetch_url_requested",
                    trace_id=trace_id,
                    step=step,
                    url=fetch_url,
                    attempt_count=fetch_attempt_counts[fetch_url],
                    repeated=fetch_attempt_counts[fetch_url] > 1,
                    call_id=call_id,
                )
                if fetch_attempt_counts[fetch_url] > 1:
                    _trace_agent_event(
                        "duplicate_fetch_attempt",
                        trace_id=trace_id,
                        step=step,
                        url=fetch_url,
                        attempt_count=fetch_attempt_counts[fetch_url],
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

            yield {"type": "step_update", "step": step, "tool": tool_name, "preview": preview, "call_id": call_id}

            if tool_name not in CANVAS_TOOL_NAMES and cache_key in tool_result_cache:
                cached_result, cached_summary = tool_result_cache[cache_key]
                transcript_result = _prepare_tool_result_for_transcript(
                    tool_name,
                    cached_result,
                    fetch_url_token_threshold=fetch_url_token_threshold,
                    fetch_url_clip_aggressiveness=fetch_url_clip_aggressiveness,
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

                remember_tool_result(
                    tool_name,
                    tool_args,
                    cached_result,
                    cached_summary,
                    cache_key,
                    transcript_result=transcript_result,
                )
                yield {
                    "type": "tool_result",
                    "step": step,
                    "tool": tool_name,
                    "summary": f"{cached_summary} (cached)",
                    "call_id": call_id,
                }
                tool_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": _serialize_tool_message_content(transcript_result),
                    }
                )
                transcript_results.append(
                    {
                        "tool_name": tool_name,
                        "arguments": tool_args,
                        "ok": not (tool_name == "fetch_url" and isinstance(cached_result, dict) and cached_result.get("error")),
                        "summary": f"{cached_summary} (cached)",
                        "result": transcript_result,
                        "cached": True,
                    }
                )
                continue

            try:
                result, summary = _execute_tool(tool_name, tool_args, runtime_state=runtime_state)
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

                yield {"type": "tool_result", "step": step, "tool": tool_name, "summary": summary, "call_id": call_id}
                tool_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": _serialize_tool_message_content(transcript_result),
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
                if tool_name in CANVAS_TOOL_NAMES:
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
            except Exception as exc:
                error = str(exc)
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

    try:
        _trace_agent_event("final_answer_phase_started", trace_id=trace_id, step=step)
        turn_result = yield from stream_model_turn([*messages, _build_final_answer_instruction()], allow_tools=False)
        content_text = turn_result.get("content_text") or ""
        tool_calls = turn_result.get("tool_calls")
        stream_error = turn_result.get("stream_error")
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
    except Exception as exc:
        yield {"type": "tool_error", "step": step, "tool": "final_answer", "error": str(exc)}
        for event in emit_answer(FINAL_ANSWER_ERROR_TEXT):
            yield event

    if usage_totals["total_tokens"]:
        yield usage_event()
    yield build_tool_capture_event()
    yield {"type": "done"}
