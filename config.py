from __future__ import annotations

import ipaddress
import json
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

BASE_DIR = os.path.dirname(__file__)
DB_PATH = os.path.join(BASE_DIR, "chatbot.db")
IMAGE_STORAGE_DIR = (os.getenv("IMAGE_STORAGE_DIR") or os.path.join(BASE_DIR, "data", "images")).strip()
PROXIES_PATH = os.path.join(BASE_DIR, "proxies.txt")
AGENT_TRACE_LOG_PATH = (os.getenv("AGENT_TRACE_LOG_PATH") or os.path.join(BASE_DIR, "logs", "agent-trace.log")).strip()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

AVAILABLE_MODELS = [
    {"id": "deepseek-chat", "name": "DeepSeek Chat"},
    {"id": "deepseek-reasoner", "name": "DeepSeek Reasoner"},
]
AVAILABLE_MODEL_IDS = {model["id"] for model in AVAILABLE_MODELS}


def _parse_int_env(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value in (None, ""):
        return default
    try:
        return int(raw_value)
    except (TypeError, ValueError):
        return default


def _parse_bool_env(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value in (None, ""):
        return default
    return str(raw_value).strip().lower() in {"1", "true", "yes", "on"}


def _parse_float_env(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value in (None, ""):
        return default
    try:
        return float(raw_value)
    except (TypeError, ValueError):
        return default


def _get_torch_dtype_name() -> str:
    raw_value = (os.getenv("QWEN_VL_TORCH_DTYPE") or "float16").strip().lower()
    allowed = {"float16", "bfloat16", "float32"}
    return raw_value if raw_value in allowed else "float16"


OCR_MODEL_PATH = (os.getenv("QWEN_VL_MODEL_PATH") or "").strip()
OCR_MAX_IMAGE_BYTES = 10 * 1024 * 1024
OCR_ALLOWED_IMAGE_TYPES = {
    "image/png",
    "image/jpeg",
    "image/webp",
}

DOCUMENT_ALLOWED_MIME_TYPES = {
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/pdf",
    "text/plain",
    "text/csv",
    "text/markdown",
}
DOCUMENT_MAX_BYTES = 20 * 1024 * 1024
DOCUMENT_MAX_TEXT_CHARS = 50_000
DOCUMENT_STORAGE_DIR = (os.getenv("DOCUMENT_STORAGE_DIR") or os.path.join(BASE_DIR, "data", "documents")).strip()
OCR_ATTENTION_IMPL = (os.getenv("QWEN_VL_ATTENTION") or "").strip() or None
OCR_MAX_NEW_TOKENS = max(128, min(4096, _parse_int_env("QWEN_VL_MAX_NEW_TOKENS", 768)))
OCR_MIN_PIXELS = max(28 * 28, _parse_int_env("QWEN_VL_MIN_PIXELS", 256 * 28 * 28))
OCR_MAX_PIXELS = max(OCR_MIN_PIXELS, _parse_int_env("QWEN_VL_MAX_PIXELS", 896 * 28 * 28))
OCR_MAX_IMAGE_SIDE = max(560, _parse_int_env("QWEN_VL_MAX_IMAGE_SIDE", 1280))
OCR_LOAD_IN_4BIT = _parse_bool_env("QWEN_VL_LOAD_IN_4BIT", True)
OCR_TORCH_DTYPE_NAME = _get_torch_dtype_name()
OCR_PRELOAD_ON_STARTUP = _parse_bool_env("QWEN_VL_PRELOAD", True)
VISION_ENABLED = _parse_bool_env("VISION_ENABLED", True)

FETCH_TIMEOUT = 20
FETCH_MAX_SIZE = 5 * 1024 * 1024
FETCH_MAX_REDIRECTS = 5
CACHE_TTL_HOURS = 24
SEARCH_MAX_RESULTS = 5
CONTENT_MAX_CHARS = 100_000
FETCH_SUMMARY_TOKEN_THRESHOLD = max(400, _parse_int_env("FETCH_SUMMARY_TOKEN_THRESHOLD", 3500))
FETCH_SUMMARY_MAX_CHARS = max(2000, min(CONTENT_MAX_CHARS, _parse_int_env("FETCH_SUMMARY_MAX_CHARS", 8000)))
FETCH_SUMMARY_GENERAL_TOP_K = max(1, min(6, _parse_int_env("FETCH_SUMMARY_GENERAL_TOP_K", 3)))
FETCH_SUMMARY_QUERY_TOP_K = max(1, min(8, _parse_int_env("FETCH_SUMMARY_QUERY_TOP_K", 4)))
FETCH_SUMMARY_EXCERPT_MAX_CHARS = max(200, min(1200, _parse_int_env("FETCH_SUMMARY_EXCERPT_MAX_CHARS", 500)))
CHAT_SUMMARY_TRIGGER_TOKEN_COUNT = max(1_000, min(200_000, _parse_int_env("CHAT_SUMMARY_TRIGGER_TOKEN_COUNT", 80_000)))
CHAT_SUMMARY_MODE = (os.getenv("CHAT_SUMMARY_MODE") or "auto").strip().lower()
CHAT_SUMMARY_MODEL = (os.getenv("CHAT_SUMMARY_MODEL") or "deepseek-chat").strip() or "deepseek-chat"
CHAT_SUMMARY_ALLOWED_MODES = {"auto", "never", "aggressive"}
PROMPT_MAX_INPUT_TOKENS = max(8_000, min(120_000, _parse_int_env("PROMPT_MAX_INPUT_TOKENS", 100_000)))
PROMPT_RESPONSE_TOKEN_RESERVE = max(1_000, min(32_000, _parse_int_env("PROMPT_RESPONSE_TOKEN_RESERVE", 8_000)))
PROMPT_RECENT_HISTORY_MAX_TOKENS = max(1_000, min(PROMPT_MAX_INPUT_TOKENS, _parse_int_env("PROMPT_RECENT_HISTORY_MAX_TOKENS", 70_000)))
PROMPT_SUMMARY_MAX_TOKENS = max(500, min(PROMPT_MAX_INPUT_TOKENS, _parse_int_env("PROMPT_SUMMARY_MAX_TOKENS", 15_000)))
PROMPT_RAG_MAX_TOKENS = max(0, min(PROMPT_MAX_INPUT_TOKENS, _parse_int_env("PROMPT_RAG_MAX_TOKENS", 6_000)))
PROMPT_TOOL_MEMORY_MAX_TOKENS = max(0, min(PROMPT_MAX_INPUT_TOKENS, _parse_int_env("PROMPT_TOOL_MEMORY_MAX_TOKENS", 4_000)))
PROMPT_PREFLIGHT_SUMMARY_TOKEN_COUNT = max(2_000, min(200_000, _parse_int_env("PROMPT_PREFLIGHT_SUMMARY_TOKEN_COUNT", 90_000)))
SUMMARY_SOURCE_TARGET_TOKENS = max(1_000, min(40_000, _parse_int_env("SUMMARY_SOURCE_TARGET_TOKENS", 6_000)))
SUMMARY_RETRY_MIN_SOURCE_TOKENS = max(500, min(SUMMARY_SOURCE_TARGET_TOKENS, _parse_int_env("SUMMARY_RETRY_MIN_SOURCE_TOKENS", 1_500)))

DEFAULT_ACTIVE_TOOL_NAMES = [
    "append_scratchpad",
    "replace_scratchpad",
    "ask_clarifying_question",
    "image_explain",
    "search_knowledge_base",
    "search_tool_memory",
    "search_web",
    "fetch_url",
    "search_news_ddgs",
    "search_news_google",
    "create_canvas_document",
    "rewrite_canvas_document",
    "replace_canvas_lines",
    "insert_canvas_lines",
    "delete_canvas_lines",
]

PRIVATE_NETWORKS = [
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("0.0.0.0/8"),
]

MAX_USER_PREFERENCES_LENGTH = 2000
MAX_SCRATCHPAD_LENGTH = 4000
SCRATCHPAD_ADMIN_EDITING_ENABLED = _parse_bool_env("SCRATCHPAD_ADMIN_EDITING_ENABLED", False)
RAG_ENABLED = _parse_bool_env("RAG_ENABLED", True)
RAG_AUTO_INJECT_TOP_K = max(1, min(8, _parse_int_env("RAG_AUTO_INJECT_TOP_K", 5)))
RAG_SEARCH_DEFAULT_TOP_K = max(1, min(12, _parse_int_env("RAG_SEARCH_DEFAULT_TOP_K", 5)))
RAG_AUTO_INJECT_THRESHOLD = max(0.0, min(1.0, _parse_float_env("RAG_AUTO_INJECT_THRESHOLD", 0.35)))
RAG_SEARCH_MIN_SIMILARITY = max(0.0, min(1.0, _parse_float_env("RAG_SEARCH_MIN_SIMILARITY", 0.2)))
RAG_SENSITIVITY_PRESETS = {
    "flexible": 0.20,
    "normal": 0.35,
    "strict": 0.55,
}
RAG_CONTEXT_SIZE_PRESETS = {
    "small": 2,
    "medium": 5,
    "large": 8,
}
RAG_SOURCE_CONVERSATION = "conversation"
RAG_SOURCE_TOOL_RESULT = "tool_result"
RAG_SOURCE_TOOL_MEMORY = "tool_memory"
RAG_SUPPORTED_SOURCE_TYPES = {RAG_SOURCE_CONVERSATION, RAG_SOURCE_TOOL_RESULT, RAG_SOURCE_TOOL_MEMORY}
RAG_SUPPORTED_CATEGORIES = {RAG_SOURCE_CONVERSATION, RAG_SOURCE_TOOL_RESULT, RAG_SOURCE_TOOL_MEMORY}
RAG_TOOL_RESULT_MAX_TEXT_CHARS = 12_000
RAG_TOOL_RESULT_SUMMARY_MAX_CHARS = 300
FETCH_RAW_TOOL_RESULT_MAX_TEXT_CHARS = max(
    RAG_TOOL_RESULT_MAX_TEXT_CHARS,
    min(CONTENT_MAX_CHARS, _parse_int_env("FETCH_RAW_TOOL_RESULT_MAX_TEXT_CHARS", 24_000)),
)
RAG_DISABLED_INGEST_ERROR = (
    "Manual RAG ingestion is disabled. RAG now only indexes conversation history and successful text-like tool results."
)
RAG_DISABLED_FEATURE_ERROR = "RAG is disabled in configuration. Set RAG_ENABLED=true to use it."
VISION_DISABLED_FEATURE_ERROR = "Vision is disabled in configuration. Set VISION_ENABLED=true to use it."


def _nearest_preset_name(value: float, presets: dict[str, float | int], fallback: str) -> str:
    if fallback not in presets:
        raise ValueError("Fallback preset must exist.")
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return fallback
    return min(presets, key=lambda name: abs(float(presets[name]) - numeric_value))


RAG_DEFAULT_SENSITIVITY_PRESET = _nearest_preset_name(
    RAG_AUTO_INJECT_THRESHOLD,
    RAG_SENSITIVITY_PRESETS,
    "normal",
)
RAG_DEFAULT_CONTEXT_SIZE_PRESET = _nearest_preset_name(
    RAG_AUTO_INJECT_TOP_K,
    RAG_CONTEXT_SIZE_PRESETS,
    "medium",
)

DEFAULT_SETTINGS = {
    "user_preferences": "",
    "scratchpad": "",
    "max_steps": "5",
    "active_tools": json.dumps(DEFAULT_ACTIVE_TOOL_NAMES, ensure_ascii=False),
    "rag_auto_inject": "true",
    "rag_sensitivity": RAG_DEFAULT_SENSITIVITY_PRESET,
    "rag_context_size": RAG_DEFAULT_CONTEXT_SIZE_PRESET,
    "tool_memory_auto_inject": "true",
    "fetch_url_token_threshold": str(FETCH_SUMMARY_TOKEN_THRESHOLD),
    "fetch_url_clip_aggressiveness": "50",
    "chat_summary_trigger_token_count": str(CHAT_SUMMARY_TRIGGER_TOKEN_COUNT),
    "chat_summary_mode": CHAT_SUMMARY_MODE if CHAT_SUMMARY_MODE in CHAT_SUMMARY_ALLOWED_MODES else "auto",
    "summary_skip_first": "2",
    "summary_skip_last": "1",
}


def get_feature_flags() -> dict:
    return {
        "rag_enabled": RAG_ENABLED,
        "vision_enabled": VISION_ENABLED,
        "scratchpad_admin_editing": SCRATCHPAD_ADMIN_EDITING_ENABLED,
    }
