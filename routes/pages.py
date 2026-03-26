from __future__ import annotations

import json

from flask import jsonify, render_template, request

from config import (
    AVAILABLE_MODELS,
    CHAT_SUMMARY_ALLOWED_MODES,
    DEFAULT_SETTINGS,
    MAX_USER_PREFERENCES_LENGTH,
    RAG_CONTEXT_SIZE_PRESETS,
    RAG_ENABLED,
    RAG_SENSITIVITY_PRESETS,
    get_feature_flags,
)
from db import (
    get_active_tool_names,
    get_app_settings,
    get_chat_summary_mode,
    get_chat_summary_trigger_token_count,
    get_fetch_url_clip_aggressiveness,
    get_fetch_url_token_threshold,
    get_pruning_batch_size,
    get_pruning_enabled,
    get_pruning_token_threshold,
    get_rag_auto_inject_enabled,
    get_rag_context_size,
    get_rag_sensitivity,
    get_summary_skip_first,
    get_summary_skip_last,
    get_tool_memory_auto_inject_enabled,
    normalize_active_tool_names,
    normalize_scratchpad_text,
    save_app_settings,
)


def build_settings_payload() -> dict:
    raw = get_app_settings()
    return {
        "user_preferences": raw["user_preferences"],
        "scratchpad": raw.get("scratchpad", ""),
        "max_steps": int(raw.get("max_steps", DEFAULT_SETTINGS["max_steps"])),
        "active_tools": get_active_tool_names(raw),
        "rag_auto_inject": get_rag_auto_inject_enabled(raw),
        "rag_sensitivity": get_rag_sensitivity(raw),
        "rag_context_size": get_rag_context_size(raw),
        "tool_memory_auto_inject": get_tool_memory_auto_inject_enabled(raw),
        "chat_summary_mode": get_chat_summary_mode(raw),
        "chat_summary_trigger_token_count": get_chat_summary_trigger_token_count(raw),
        "summary_skip_first": get_summary_skip_first(raw),
        "summary_skip_last": get_summary_skip_last(raw),
        "pruning_enabled": get_pruning_enabled(raw),
        "pruning_token_threshold": get_pruning_token_threshold(raw),
        "pruning_batch_size": get_pruning_batch_size(raw),
        "fetch_url_token_threshold": get_fetch_url_token_threshold(raw),
        "fetch_url_clip_aggressiveness": get_fetch_url_clip_aggressiveness(raw),
        "features": get_feature_flags(),
    }


def register_page_routes(app) -> None:
    @app.route("/")
    def index():
        settings = build_settings_payload()
        return render_template(
            "index.html",
            models=AVAILABLE_MODELS,
            settings=settings,
        )

    @app.route("/settings")
    def settings_page():
        settings = build_settings_payload()
        return render_template("settings.html", settings=settings)

    @app.route("/api/settings", methods=["GET"])
    def get_settings():
        return jsonify(build_settings_payload())

    @app.route("/api/settings", methods=["PATCH"])
    def update_settings():
        data = request.get_json(silent=True) or {}
        user_preferences = data.get("user_preferences")
        max_steps_raw = data.get("max_steps")
        active_tools_raw = data.get("active_tools")
        rag_auto_inject = data.get("rag_auto_inject")
        rag_sensitivity = data.get("rag_sensitivity")
        rag_context_size = data.get("rag_context_size")
        tool_memory_auto_inject = data.get("tool_memory_auto_inject")
        chat_summary_mode_raw = data.get("chat_summary_mode")
        chat_summary_trigger_raw = data.get("chat_summary_trigger_token_count")
        summary_skip_first_raw = data.get("summary_skip_first")
        summary_skip_last_raw = data.get("summary_skip_last")
        pruning_enabled_raw = data.get("pruning_enabled")
        pruning_token_threshold_raw = data.get("pruning_token_threshold")
        pruning_batch_size_raw = data.get("pruning_batch_size")
        fetch_url_token_threshold_raw = data.get("fetch_url_token_threshold")
        fetch_url_clip_aggressiveness_raw = data.get("fetch_url_clip_aggressiveness")
        scratchpad = data.get("scratchpad")

        if (
            user_preferences is None
            and scratchpad is None
            and max_steps_raw is None
            and active_tools_raw is None
            and rag_auto_inject is None
            and rag_sensitivity is None
            and rag_context_size is None
            and tool_memory_auto_inject is None
            and chat_summary_mode_raw is None
            and chat_summary_trigger_raw is None
            and summary_skip_first_raw is None
            and summary_skip_last_raw is None
            and pruning_enabled_raw is None
            and pruning_token_threshold_raw is None
            and pruning_batch_size_raw is None
            and fetch_url_token_threshold_raw is None
            and fetch_url_clip_aggressiveness_raw is None
        ):
            return jsonify({"error": "No settings provided."}), 400

        settings = get_app_settings()

        if user_preferences is not None:
            if not isinstance(user_preferences, str):
                return jsonify({"error": "Invalid user preferences."}), 400
            settings["user_preferences"] = user_preferences.strip()[:MAX_USER_PREFERENCES_LENGTH]

        if scratchpad is not None:
            if not isinstance(scratchpad, str):
                return jsonify({"error": "Invalid scratchpad."}), 400
            try:
                settings["scratchpad"] = normalize_scratchpad_text(scratchpad)
            except ValueError as exc:
                return jsonify({"error": str(exc)}), 400

        if max_steps_raw is not None:
            try:
                max_steps = int(max_steps_raw)
            except (TypeError, ValueError):
                return jsonify({"error": "max_steps must be an integer."}), 400
            if not (1 <= max_steps <= 10):
                return jsonify({"error": "max_steps must be between 1 and 10."}), 400
            settings["max_steps"] = str(max_steps)

        if active_tools_raw is not None:
            if not isinstance(active_tools_raw, list):
                return jsonify({"error": "Invalid active tools."}), 400
            settings["active_tools"] = json.dumps(normalize_active_tool_names(active_tools_raw), ensure_ascii=False)

        if rag_auto_inject is not None and RAG_ENABLED:
            if isinstance(rag_auto_inject, bool):
                settings["rag_auto_inject"] = "true" if rag_auto_inject else "false"
            else:
                settings["rag_auto_inject"] = (
                    "true" if str(rag_auto_inject).strip().lower() in {"1", "true", "yes", "on"} else "false"
                )
        elif not RAG_ENABLED:
            settings["rag_auto_inject"] = "false"

        if rag_sensitivity is not None:
            normalized_rag_sensitivity = str(rag_sensitivity or "").strip().lower()
            if normalized_rag_sensitivity not in RAG_SENSITIVITY_PRESETS:
                return jsonify({"error": "rag_sensitivity must be one of flexible, normal, or strict."}), 400
            settings["rag_sensitivity"] = normalized_rag_sensitivity

        if rag_context_size is not None:
            normalized_rag_context_size = str(rag_context_size or "").strip().lower()
            if normalized_rag_context_size not in RAG_CONTEXT_SIZE_PRESETS:
                return jsonify({"error": "rag_context_size must be one of small, medium, or large."}), 400
            settings["rag_context_size"] = normalized_rag_context_size

        if tool_memory_auto_inject is not None and RAG_ENABLED:
            if isinstance(tool_memory_auto_inject, bool):
                settings["tool_memory_auto_inject"] = "true" if tool_memory_auto_inject else "false"
            else:
                settings["tool_memory_auto_inject"] = (
                    "true" if str(tool_memory_auto_inject).strip().lower() in {"1", "true", "yes", "on"} else "false"
                )
        elif not RAG_ENABLED:
            settings["tool_memory_auto_inject"] = "false"

        if chat_summary_mode_raw is not None:
            normalized_summary_mode = str(chat_summary_mode_raw or "").strip().lower()
            if normalized_summary_mode not in CHAT_SUMMARY_ALLOWED_MODES:
                return jsonify({"error": "chat_summary_mode must be one of auto, never, or aggressive."}), 400
            settings["chat_summary_mode"] = normalized_summary_mode

        if chat_summary_trigger_raw is not None:
            try:
                chat_summary_trigger = int(chat_summary_trigger_raw)
            except (TypeError, ValueError):
                return jsonify({"error": "chat_summary_trigger_token_count must be an integer."}), 400
            if not (1_000 <= chat_summary_trigger <= 200_000):
                return jsonify({"error": "chat_summary_trigger_token_count must be between 1000 and 200000."}), 400
            settings["chat_summary_trigger_token_count"] = str(chat_summary_trigger)

        if summary_skip_first_raw is not None:
            try:
                summary_skip_first = int(summary_skip_first_raw)
            except (TypeError, ValueError):
                return jsonify({"error": "summary_skip_first must be an integer."}), 400
            if not (0 <= summary_skip_first <= 20):
                return jsonify({"error": "summary_skip_first must be between 0 and 20."}), 400
            settings["summary_skip_first"] = str(summary_skip_first)

        if summary_skip_last_raw is not None:
            try:
                summary_skip_last = int(summary_skip_last_raw)
            except (TypeError, ValueError):
                return jsonify({"error": "summary_skip_last must be an integer."}), 400
            if not (0 <= summary_skip_last <= 20):
                return jsonify({"error": "summary_skip_last must be between 0 and 20."}), 400
            settings["summary_skip_last"] = str(summary_skip_last)

        if pruning_enabled_raw is not None:
            if isinstance(pruning_enabled_raw, bool):
                settings["pruning_enabled"] = "true" if pruning_enabled_raw else "false"
            else:
                settings["pruning_enabled"] = (
                    "true" if str(pruning_enabled_raw).strip().lower() in {"1", "true", "yes", "on"} else "false"
                )

        if pruning_token_threshold_raw is not None:
            try:
                pruning_token_threshold = int(pruning_token_threshold_raw)
            except (TypeError, ValueError):
                return jsonify({"error": "pruning_token_threshold must be an integer."}), 400
            if not (1_000 <= pruning_token_threshold <= 200_000):
                return jsonify({"error": "pruning_token_threshold must be between 1000 and 200000."}), 400
            settings["pruning_token_threshold"] = str(pruning_token_threshold)

        if pruning_batch_size_raw is not None:
            try:
                pruning_batch_size = int(pruning_batch_size_raw)
            except (TypeError, ValueError):
                return jsonify({"error": "pruning_batch_size must be an integer."}), 400
            if not (1 <= pruning_batch_size <= 50):
                return jsonify({"error": "pruning_batch_size must be between 1 and 50."}), 400
            settings["pruning_batch_size"] = str(pruning_batch_size)

        if fetch_url_token_threshold_raw is not None:
            try:
                fetch_url_token_threshold = int(fetch_url_token_threshold_raw)
            except (TypeError, ValueError):
                return jsonify({"error": "fetch_url_token_threshold must be an integer."}), 400
            if not (400 <= fetch_url_token_threshold <= 20_000):
                return jsonify({"error": "fetch_url_token_threshold must be between 400 and 20000."}), 400
            settings["fetch_url_token_threshold"] = str(fetch_url_token_threshold)

        if fetch_url_clip_aggressiveness_raw is not None:
            try:
                fetch_url_clip_aggressiveness = int(fetch_url_clip_aggressiveness_raw)
            except (TypeError, ValueError):
                return jsonify({"error": "fetch_url_clip_aggressiveness must be an integer."}), 400
            if not (0 <= fetch_url_clip_aggressiveness <= 100):
                return jsonify({"error": "fetch_url_clip_aggressiveness must be between 0 and 100."}), 400
            settings["fetch_url_clip_aggressiveness"] = str(fetch_url_clip_aggressiveness)

        save_app_settings(settings)
        return jsonify(
            {
                "user_preferences": settings["user_preferences"],
                "scratchpad": settings.get("scratchpad", ""),
                "max_steps": int(settings["max_steps"]),
                "active_tools": get_active_tool_names(settings),
                "rag_auto_inject": get_rag_auto_inject_enabled(settings),
                "rag_sensitivity": get_rag_sensitivity(settings),
                "rag_context_size": get_rag_context_size(settings),
                "tool_memory_auto_inject": get_tool_memory_auto_inject_enabled(settings),
                "chat_summary_mode": get_chat_summary_mode(settings),
                "chat_summary_trigger_token_count": get_chat_summary_trigger_token_count(settings),
                "summary_skip_first": get_summary_skip_first(settings),
                "summary_skip_last": get_summary_skip_last(settings),
                "pruning_enabled": get_pruning_enabled(settings),
                "pruning_token_threshold": get_pruning_token_threshold(settings),
                "pruning_batch_size": get_pruning_batch_size(settings),
                "fetch_url_token_threshold": get_fetch_url_token_threshold(settings),
                "fetch_url_clip_aggressiveness": get_fetch_url_clip_aggressiveness(settings),
                "features": get_feature_flags(),
            }
        )
