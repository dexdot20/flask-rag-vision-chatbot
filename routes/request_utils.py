from __future__ import annotations

import json

from config import AVAILABLE_MODEL_IDS


def parse_optional_int(value) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_messages_payload(raw_value) -> list:
    if isinstance(raw_value, list):
        return raw_value
    if raw_value in (None, ""):
        return []
    try:
        parsed = json.loads(raw_value)
    except Exception:
        return []
    return parsed if isinstance(parsed, list) else []


def normalize_model_id(value, default: str = "deepseek-chat") -> str:
    return str(value or default).strip() or default


def is_valid_model_id(model_id: str) -> bool:
    return model_id in AVAILABLE_MODEL_IDS