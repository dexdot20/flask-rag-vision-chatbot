from __future__ import annotations

import json

from flask import jsonify, render_template, request

from config import (
    AVAILABLE_MODELS,
    DEFAULT_SETTINGS,
    MAX_USER_PREFERENCES_LENGTH,
    RAG_ENABLED,
    get_feature_flags,
)
from db import (
    get_chat_summary_batch_size,
    get_chat_summary_trigger_message_count,
    get_fetch_url_clip_aggressiveness,
    get_fetch_url_token_threshold,
    get_active_tool_names,
    get_app_settings,
    get_rag_auto_inject_enabled,
    normalize_scratchpad_text,
    normalize_active_tool_names,
    save_app_settings,
)


def register_page_routes(app) -> None:
    @app.route("/")
    def index():
        raw = get_app_settings()
        settings = {
            "user_preferences": raw["user_preferences"],
            "scratchpad": raw.get("scratchpad", ""),
            "max_steps": int(raw.get("max_steps", DEFAULT_SETTINGS["max_steps"])),
            "active_tools": get_active_tool_names(raw),
            "rag_auto_inject": get_rag_auto_inject_enabled(raw),
            "chat_summary_trigger_message_count": get_chat_summary_trigger_message_count(raw),
            "chat_summary_batch_size": get_chat_summary_batch_size(raw),
            "fetch_url_token_threshold": get_fetch_url_token_threshold(raw),
            "fetch_url_clip_aggressiveness": get_fetch_url_clip_aggressiveness(raw),
            "features": get_feature_flags(),
        }
        return render_template(
            "index.html",
            models=AVAILABLE_MODELS,
            settings=settings,
        )

    @app.route("/api/settings", methods=["GET"])
    def get_settings():
        raw = get_app_settings()
        return jsonify(
            {
                "user_preferences": raw["user_preferences"],
                "scratchpad": raw.get("scratchpad", ""),
                "max_steps": int(raw.get("max_steps", DEFAULT_SETTINGS["max_steps"])),
                "active_tools": get_active_tool_names(raw),
                "rag_auto_inject": get_rag_auto_inject_enabled(raw),
                "chat_summary_trigger_message_count": get_chat_summary_trigger_message_count(raw),
                "chat_summary_batch_size": get_chat_summary_batch_size(raw),
                "fetch_url_token_threshold": get_fetch_url_token_threshold(raw),
                "fetch_url_clip_aggressiveness": get_fetch_url_clip_aggressiveness(raw),
                "features": get_feature_flags(),
            }
        )

    @app.route("/api/settings", methods=["PATCH"])
    def update_settings():
        data = request.get_json(silent=True) or {}
        user_preferences = data.get("user_preferences")
        max_steps_raw = data.get("max_steps")
        active_tools_raw = data.get("active_tools")
        rag_auto_inject = data.get("rag_auto_inject")
        chat_summary_trigger_raw = data.get("chat_summary_trigger_message_count")
        chat_summary_batch_raw = data.get("chat_summary_batch_size")
        fetch_url_token_threshold_raw = data.get("fetch_url_token_threshold")
        fetch_url_clip_aggressiveness_raw = data.get("fetch_url_clip_aggressiveness")
        scratchpad = data.get("scratchpad")

        if (
            user_preferences is None
            and scratchpad is None
            and max_steps_raw is None
            and active_tools_raw is None
            and rag_auto_inject is None
            and chat_summary_trigger_raw is None
            and chat_summary_batch_raw is None
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
            if not get_feature_flags().get("scratchpad_admin_editing"):
                return jsonify({"error": "scratchpad is read-only and can only be updated by the AI tool."}), 400
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

        if chat_summary_trigger_raw is not None:
            try:
                chat_summary_trigger = int(chat_summary_trigger_raw)
            except (TypeError, ValueError):
                return jsonify({"error": "chat_summary_trigger_message_count must be an integer."}), 400
            if not (10 <= chat_summary_trigger <= 500):
                return jsonify({"error": "chat_summary_trigger_message_count must be between 10 and 500."}), 400
            settings["chat_summary_trigger_message_count"] = str(chat_summary_trigger)

        if chat_summary_batch_raw is not None:
            try:
                chat_summary_batch = int(chat_summary_batch_raw)
            except (TypeError, ValueError):
                return jsonify({"error": "chat_summary_batch_size must be an integer."}), 400
            if not (5 <= chat_summary_batch <= 100):
                return jsonify({"error": "chat_summary_batch_size must be between 5 and 100."}), 400
            settings["chat_summary_batch_size"] = str(chat_summary_batch)

        if get_chat_summary_batch_size(settings) > get_chat_summary_trigger_message_count(settings):
            return jsonify({"error": "chat_summary_batch_size must be less than or equal to chat_summary_trigger_message_count."}), 400

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
                "chat_summary_trigger_message_count": get_chat_summary_trigger_message_count(settings),
                "chat_summary_batch_size": get_chat_summary_batch_size(settings),
                "fetch_url_token_threshold": get_fetch_url_token_threshold(settings),
                "fetch_url_clip_aggressiveness": get_fetch_url_clip_aggressiveness(settings),
                "features": get_feature_flags(),
            }
        )
