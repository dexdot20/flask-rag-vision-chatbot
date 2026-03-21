from __future__ import annotations

import json
from datetime import datetime

from flask import Response, jsonify, request, stream_with_context

from agent import collect_agent_response, run_agent_stream
from config import RAG_ENABLED, VISION_DISABLED_FEATURE_ERROR, VISION_ENABLED
from db import (
    count_unsummarized_visible_messages,
    create_image_asset,
    delete_image_asset,
    extract_message_tool_results,
    find_summary_covering_message_id,
    get_chat_summary_batch_size,
    get_chat_summary_trigger_message_count,
    get_conversation_messages,
    get_fetch_url_clip_aggressiveness,
    get_fetch_url_token_threshold,
    get_active_tool_names,
    get_app_settings,
    get_db,
    get_rag_auto_inject_enabled,
    get_unsummarized_visible_messages,
    insert_message,
    parse_message_metadata,
    serialize_message_metadata,
    serialize_message_tool_calls,
    update_image_asset,
)
from messages import build_api_messages, build_user_message_for_model, normalize_chat_messages, prepend_runtime_context
from rag import preload_embedder
from rag_service import build_rag_auto_context
from routes.request_utils import is_valid_model_id, normalize_model_id, parse_messages_payload, parse_optional_int
from vision import preload_local_ocr_engine, read_uploaded_image, run_image_vision_analysis


def parse_chat_request_payload():
    if request.mimetype and request.mimetype.startswith("multipart/form-data"):
        return {
            "messages": parse_messages_payload(request.form.get("messages", "[]")),
            "model": normalize_model_id(request.form.get("model")),
            "conversation_id": parse_optional_int(request.form.get("conversation_id")),
            "edited_message_id": parse_optional_int(request.form.get("edited_message_id")),
            "user_content": request.form.get("user_content", ""),
            "image": request.files.get("image"),
        }

    data = request.get_json(silent=True) or {}

    return {
        "messages": parse_messages_payload(data.get("messages", [])),
        "model": normalize_model_id(data.get("model")),
        "conversation_id": parse_optional_int(data.get("conversation_id")),
        "edited_message_id": parse_optional_int(data.get("edited_message_id")),
        "user_content": data.get("user_content", ""),
        "image": None,
    }


def register_chat_routes(app) -> None:
    summary_label = "Conversation summary (generated from deleted messages):"

    def build_summary_content(summary_text: str) -> str:
        text = str(summary_text or "").strip()
        if not text:
            return summary_label
        if text.lower().startswith(summary_label.lower()):
            return text
        return f"{summary_label}\n\n{text}"

    def build_summary_prompt_messages(source_messages: list[dict], user_preferences: str) -> list[dict]:
        prompt_messages = [
            {
                "role": "system",
                "content": (
                    "You are compressing earlier conversation history for later reuse. "
                    "Write in the dominant language of the conversation. "
                    "Preserve user goals, constraints, important facts, decisions, promises, unresolved questions, names, numbers, and preferences that later turns may rely on. "
                    "Prefer short paragraphs or a few compact bullets. If helpful, start with a one-line overview, then include only the most useful follow-up details. "
                    "Do not mention tool internals unless they materially affect future replies. "
                    "Do not use markdown headings, tables, or code fences. Return only the summary text."
                ),
            },
            *build_api_messages(source_messages),
        ]
        return prepend_runtime_context(prompt_messages, user_preferences, [], retrieved_context=None)

    def maybe_create_conversation_summary(
        conversation_id: int,
        model: str,
        settings: dict,
        fetch_url_token_threshold: int,
        fetch_url_clip_aggressiveness: int,
    ) -> dict:
        canonical_messages = get_conversation_messages(conversation_id)
        trigger_count = get_chat_summary_trigger_message_count(settings)
        batch_size = get_chat_summary_batch_size(settings)
        unsummarized_visible_count = count_unsummarized_visible_messages(canonical_messages)
        if unsummarized_visible_count < trigger_count:
            return {"applied": False, "messages": canonical_messages}

        source_messages = get_unsummarized_visible_messages(canonical_messages, limit=batch_size)
        if len(source_messages) < batch_size:
            return {"applied": False, "messages": canonical_messages}

        prompt_messages = build_summary_prompt_messages(source_messages, settings.get("user_preferences", ""))
        result = collect_agent_response(
            prompt_messages,
            model,
            1,
            [],
            fetch_url_token_threshold=fetch_url_token_threshold,
            fetch_url_clip_aggressiveness=fetch_url_clip_aggressiveness,
        )
        summary_text = (result.get("content") or "").strip()
        if not summary_text:
            return {"applied": False, "messages": canonical_messages, "error": "summary_generation_failed"}

        selected_message_ids = {
            int(message["id"])
            for message in source_messages
            if int(message.get("id") or 0) > 0
        }
        start_position = min(int(message.get("position") or 0) for message in source_messages)
        end_position = max(int(message.get("position") or 0) for message in source_messages)
        covered_message_ids = [int(message["id"]) for message in source_messages if int(message.get("id") or 0) > 0]
        summary_metadata = serialize_message_metadata(
            {
                "is_summary": True,
                "summary_source": "conversation_history",
                "covers_from_position": start_position,
                "covers_to_position": end_position,
                "covered_message_count": len(source_messages),
                "covered_message_ids": covered_message_ids,
                "trigger_threshold": trigger_count,
                "summary_batch_size": batch_size,
                "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            }
        )

        rebuilt_order = []
        summary_inserted = False
        for message in canonical_messages:
            message_id = int(message.get("id") or 0)
            if message_id in selected_message_ids:
                if not summary_inserted:
                    rebuilt_order.append({"kind": "summary"})
                    summary_inserted = True
                continue
            rebuilt_order.append({"kind": "existing", "id": message_id})

        with get_db() as conn:
            conn.execute(
                "DELETE FROM messages WHERE conversation_id = ? AND id IN ({})".format(
                    ", ".join("?" for _ in covered_message_ids)
                ),
                (conversation_id, *covered_message_ids),
            )
            summary_message_id = insert_message(
                conn,
                conversation_id,
                "summary",
                build_summary_content(summary_text),
                metadata=summary_metadata,
                position=start_position,
            )
            for index, entry in enumerate(rebuilt_order, start=1):
                target_id = summary_message_id if entry["kind"] == "summary" else entry["id"]
                conn.execute("UPDATE messages SET position = ? WHERE id = ?", (index, target_id))
            conn.execute(
                "UPDATE conversations SET updated_at = datetime('now') WHERE id = ?",
                (conversation_id,),
            )

        return {
            "applied": True,
            "summary_message_id": summary_message_id,
            "messages": get_conversation_messages(conversation_id),
        }

    def upsert_tool_trace_entry(entries: list[dict], call_map: dict[str, int], event: dict) -> None:
        tool_name = str(event.get("tool") or "").strip()
        if not tool_name:
            return

        call_id = str(event.get("call_id") or f"step-{event.get('step') or 1}-{tool_name}").strip()
        step_value = event.get("step")
        try:
            normalized_step = max(1, int(step_value))
        except (TypeError, ValueError):
            normalized_step = 1

        entry = {
            "tool_name": tool_name,
            "step": normalized_step,
        }

        preview = str(event.get("preview") or "").strip()
        if preview:
            entry["preview"] = preview

        event_type = str(event.get("type") or "").strip()
        if event_type == "step_update":
            entry["state"] = "running"
        elif event_type == "tool_error":
            entry["state"] = "error"
            summary = str(event.get("error") or "").strip()
            if summary:
                entry["summary"] = summary
        elif event_type == "tool_result":
            entry["state"] = "done"
            summary = str(event.get("summary") or "").strip()
            if summary:
                entry["summary"] = summary
            if "(cached)" in summary.lower():
                entry["cached"] = True
        else:
            return

        existing_index = call_map.get(call_id)
        if existing_index is None:
            call_map[call_id] = len(entries)
            entries.append(entry)
            return

        current = entries[existing_index]
        current.update(entry)

    def build_tool_results_ui_payload(tool_results: list[dict]) -> list[dict]:
        payload = []
        for entry in tool_results:
            if not isinstance(entry, dict):
                continue
            tool_name = str(entry.get("tool_name") or "").strip()
            if not tool_name:
                continue

            item = {"tool_name": tool_name}
            content_mode = str(entry.get("content_mode") or "").strip()
            summary_notice = str(entry.get("summary_notice") or "").strip()
            if content_mode:
                item["content_mode"] = content_mode
            if summary_notice:
                item["summary_notice"] = summary_notice
            if entry.get("cleanup_applied") is True:
                item["cleanup_applied"] = True

            payload.append(item)
        return payload

    def persist_tool_history_rows(conversation_id: int, tool_history_messages: list[dict]) -> None:
        if not conversation_id or not isinstance(tool_history_messages, list):
            return

        with get_db() as conn:
            for message in tool_history_messages:
                if not isinstance(message, dict):
                    continue

                role = str(message.get("role") or "").strip()
                content = message.get("content")
                if content is None:
                    content = ""
                if not isinstance(content, str):
                    content = str(content)

                if role == "assistant":
                    insert_message(
                        conn,
                        conversation_id,
                        "assistant",
                        content,
                        tool_calls=serialize_message_tool_calls(message.get("tool_calls")),
                    )
                elif role == "tool":
                    insert_message(
                        conn,
                        conversation_id,
                        "tool",
                        content,
                        tool_call_id=str(message.get("tool_call_id") or "").strip() or None,
                    )

            conn.execute(
                "UPDATE conversations SET updated_at = datetime('now') WHERE id = ?",
                (conversation_id,),
            )

    @app.route("/api/fix-text", methods=["POST"])
    def fix_text():
        data = request.get_json(silent=True) or {}
        text = (data.get("text") or "").strip()
        model = data.get("model", "deepseek-chat")

        if not text:
            return jsonify({"error": "No text provided."}), 400

        if not is_valid_model_id(model):
            return jsonify({"error": "Invalid model."}), 400

        settings = get_app_settings()
        active_tool_names = get_active_tool_names(settings)
        fetch_url_clip_aggressiveness = get_fetch_url_clip_aggressiveness(settings)
        fetch_url_token_threshold = get_fetch_url_token_threshold(settings)
        messages = prepend_runtime_context(
            [
                {
                    "role": "system",
                    "content": (
                        "You improve user-written text without changing its core meaning. "
                        "Correct spelling, grammar, punctuation, and clarity. Preserve the original language, tone, and intent. "
                        "Do not add commentary, explanations, quotes, labels, or markdown fences. "
                        "Return only the improved text."
                    ),
                },
                {
                    "role": "user",
                    "content": text,
                },
            ],
            settings["user_preferences"],
            active_tool_names,
        )

        max_steps = max(1, min(10, int(settings.get("max_steps", 5))))
        result = collect_agent_response(
            messages,
            model,
            max_steps,
            active_tool_names,
            fetch_url_token_threshold=fetch_url_token_threshold,
            fetch_url_clip_aggressiveness=fetch_url_clip_aggressiveness,
        )
        fixed_text = (result.get("content") or "").strip()
        if not fixed_text:
            errors = result.get("errors") or []
            return jsonify({"error": errors[-1] if errors else "No text returned."}), 502
        return jsonify({"text": fixed_text})

    @app.route("/chat", methods=["POST"])
    def chat():
        payload = parse_chat_request_payload()
        messages = normalize_chat_messages(payload["messages"])
        model = payload["model"]
        conv_id = payload["conversation_id"]
        edited_message_id = payload["edited_message_id"]
        user_content = payload["user_content"]
        uploaded_file = payload["image"]

        if not messages:
            return jsonify({"error": "No messages provided."}), 400

        if not is_valid_model_id(model):
            return jsonify({"error": "Invalid model."}), 400

        vision_event = None
        latest_user_message = messages[-1] if messages and messages[-1]["role"] == "user" else None

        if uploaded_file is not None:
            if latest_user_message is None:
                return jsonify({"error": "Image uploads require a user message."}), 400
            if not VISION_ENABLED:
                return jsonify({"error": VISION_DISABLED_FEATURE_ERROR}), 410
            if conv_id is None:
                return jsonify({"error": "Image uploads require an existing saved conversation."}), 400
            created_image_asset = None
            try:
                image_name, image_mime_type, image_bytes = read_uploaded_image(uploaded_file)
                created_image_asset = create_image_asset(conv_id, image_name, image_mime_type, image_bytes)
                vision_analysis = run_image_vision_analysis(
                    image_bytes,
                    image_mime_type,
                    user_text=latest_user_message["content"],
                )
            except ValueError as exc:
                if created_image_asset is not None:
                    delete_image_asset(created_image_asset["image_id"], conversation_id=conv_id)
                return jsonify({"error": str(exc)}), 400
            except Exception as exc:
                if created_image_asset is not None:
                    delete_image_asset(created_image_asset["image_id"], conversation_id=conv_id)
                return jsonify({"error": f"Local image analysis failed: {exc}"}), 502

            latest_user_message["metadata"] = {
                **latest_user_message.get("metadata", {}),
                "image_id": created_image_asset["image_id"],
                "ocr_text": vision_analysis.get("ocr_text", ""),
                "vision_summary": vision_analysis.get("vision_summary", ""),
                "assistant_guidance": vision_analysis.get("assistant_guidance", ""),
                "key_points": vision_analysis.get("key_points", []),
                "image_name": image_name,
                "image_mime_type": image_mime_type,
            }
            vision_event = {
                "type": "vision_complete",
                "image_id": created_image_asset["image_id"],
                "image_name": image_name,
                "ocr_text": vision_analysis.get("ocr_text", ""),
                "vision_summary": vision_analysis.get("vision_summary", ""),
                "assistant_guidance": vision_analysis.get("assistant_guidance", ""),
                "key_points": vision_analysis.get("key_points", []),
            }

        settings = get_app_settings()
        max_steps = max(1, min(10, int(settings.get("max_steps", 5))))
        active_tool_names = get_active_tool_names(settings)
        fetch_url_clip_aggressiveness = get_fetch_url_clip_aggressiveness(settings)
        fetch_url_token_threshold = get_fetch_url_token_threshold(settings)
        rag_query_text = ""
        if latest_user_message is not None:
            rag_query_text = build_user_message_for_model(
                latest_user_message["content"],
                latest_user_message.get("metadata"),
            )
        persisted_user_message_id = None
        canonical_messages = messages

        if latest_user_message is not None and conv_id:
            user_message_metadata = serialize_message_metadata(latest_user_message.get("metadata"))
            persisted_user_content = latest_user_message["content"]
            if user_content is not None:
                persisted_user_content = str(user_content)

            if edited_message_id is not None:
                with get_db() as conn:
                    existing_message = conn.execute(
                        "SELECT id, role, position FROM messages WHERE id = ? AND conversation_id = ?",
                        (edited_message_id, conv_id),
                    ).fetchone()
                    if not existing_message:
                        summary_message = find_summary_covering_message_id(conv_id, edited_message_id)
                        if summary_message is not None:
                            return jsonify({"error": "This message can no longer be edited because it was summarized."}), 400
                        return jsonify({"error": "Edited message not found."}), 404
                    if existing_message["role"] != "user":
                        return jsonify({"error": "Only user messages can be edited."}), 400

                    conn.execute(
                        "UPDATE messages SET content = ?, metadata = ? WHERE id = ?",
                        (persisted_user_content, user_message_metadata, edited_message_id),
                    )
                    conn.execute(
                        "DELETE FROM messages WHERE conversation_id = ? AND position > ?",
                        (conv_id, existing_message["position"]),
                    )
                    conn.execute(
                        "UPDATE conversations SET updated_at = datetime('now') WHERE id = ?",
                        (conv_id,),
                    )
                persisted_user_message_id = edited_message_id
            elif persisted_user_content or user_message_metadata:
                with get_db() as conn:
                    persisted_user_message_id = insert_message(
                        conn,
                        conv_id,
                        "user",
                        persisted_user_content,
                        metadata=user_message_metadata,
                    )
                    conn.execute(
                        "UPDATE conversations SET updated_at = datetime('now') WHERE id = ?",
                        (conv_id,),
                    )

            image_id = ((latest_user_message.get("metadata") or {}).get("image_id") or "").strip()
            if image_id and persisted_user_message_id is not None:
                update_image_asset(
                    image_id,
                    message_id=persisted_user_message_id,
                    initial_analysis=latest_user_message.get("metadata"),
                )

            summary_outcome = maybe_create_conversation_summary(
                conv_id,
                model,
                settings,
                fetch_url_token_threshold,
                fetch_url_clip_aggressiveness,
            )
            canonical_messages = summary_outcome.get("messages") or get_conversation_messages(conv_id)
        elif conv_id:
            canonical_messages = get_conversation_messages(conv_id)

        retrieved_context = build_rag_auto_context(rag_query_text, get_rag_auto_inject_enabled(settings))
        api_messages = prepend_runtime_context(
            build_api_messages(canonical_messages),
            settings["user_preferences"],
            active_tool_names,
            retrieved_context=retrieved_context,
            scratchpad=settings.get("scratchpad", ""),
        )

        def generate():
            full_response = ""
            full_reasoning = ""
            usage_data = None
            stored_tool_results = []
            pending_clarification = None
            persisted_tool_history = []
            tool_trace_entries = []
            tool_trace_by_call_id = {}
            persisted_assistant_message_id = None

            if vision_event:
                yield json.dumps(vision_event, ensure_ascii=False) + "\n"

            for event in run_agent_stream(
                api_messages,
                model,
                max_steps,
                active_tool_names,
                fetch_url_token_threshold=fetch_url_token_threshold,
                fetch_url_clip_aggressiveness=fetch_url_clip_aggressiveness,
            ):
                if event["type"] == "answer_delta":
                    full_response += event["text"]
                elif event["type"] == "answer_sync":
                    full_response = event["text"]
                elif event["type"] == "clarification_request":
                    full_response = str(event.get("text") or "").strip()
                    pending_clarification = event.get("clarification") if isinstance(event.get("clarification"), dict) else None
                elif event["type"] == "reasoning_delta":
                    full_reasoning += event["text"]
                elif event["type"] == "usage":
                    usage_data = event
                elif event["type"] in {"step_update", "tool_result", "tool_error"}:
                    upsert_tool_trace_entry(tool_trace_entries, tool_trace_by_call_id, event)
                elif event["type"] == "tool_history":
                    history_messages = normalize_chat_messages(event.get("messages") or [])
                    if history_messages:
                        persisted_tool_history.extend(history_messages)
                        yield json.dumps(
                            {
                                "type": "assistant_tool_history",
                                "messages": history_messages,
                            },
                            ensure_ascii=False,
                        ) + "\n"
                    continue
                elif event["type"] == "tool_capture":
                    stored_tool_results = extract_message_tool_results({"tool_results": event.get("tool_results")})
                    ui_tool_results = build_tool_results_ui_payload(stored_tool_results)
                    if ui_tool_results:
                        yield json.dumps(
                            {
                                "type": "assistant_tool_results",
                                "tool_results": ui_tool_results,
                            },
                            ensure_ascii=False,
                        ) + "\n"
                    continue
                yield json.dumps(event, ensure_ascii=False) + "\n"

            if conv_id and persisted_tool_history:
                persist_tool_history_rows(conv_id, persisted_tool_history)

            if conv_id and (full_response or full_reasoning or pending_clarification):
                prompt_tokens = usage_data.get("prompt_tokens") if usage_data else None
                completion_tokens = usage_data.get("completion_tokens") if usage_data else None
                total_tokens = usage_data.get("total_tokens") if usage_data else None
                assistant_message_metadata = serialize_message_metadata(
                    {
                        "tool_results": stored_tool_results,
                        "tool_trace": tool_trace_entries,
                        "reasoning_content": full_reasoning,
                        "pending_clarification": pending_clarification,
                        "usage": usage_data,
                    }
                )
                with get_db() as conn:
                    persisted_assistant_message_id = insert_message(
                        conn,
                        conv_id,
                        "assistant",
                        full_response,
                        metadata=assistant_message_metadata,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=total_tokens,
                    )
                    conn.execute(
                        "UPDATE conversations SET updated_at = datetime('now') WHERE id = ?",
                        (conv_id,),
                    )

            if persisted_user_message_id is not None or persisted_assistant_message_id is not None:
                yield json.dumps(
                    {
                        "type": "message_ids",
                        "user_message_id": persisted_user_message_id,
                        "assistant_message_id": persisted_assistant_message_id,
                    },
                    ensure_ascii=False,
                ) + "\n"

            if conv_id and (persisted_user_message_id is not None or persisted_assistant_message_id is not None):
                yield json.dumps(
                    {
                        "type": "history_sync",
                        "messages": get_conversation_messages(conv_id),
                    },
                    ensure_ascii=False,
                ) + "\n"

        return Response(
            stream_with_context(generate()),
            content_type="application/x-ndjson; charset=utf-8",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )


    @app.route("/api/conversations/<int:conv_id>/generate-title", methods=["POST"])
    def generate_title(conv_id):
        with get_db() as conn:
            conversation = conn.execute(
                "SELECT title FROM conversations WHERE id = ?",
                (conv_id,),
            ).fetchone()
            if not conversation:
                return jsonify({"error": "Not found."}), 404
            messages = conn.execute(
                     """SELECT role, content, metadata FROM messages
                         WHERE conversation_id = ?
                                     AND (role = 'user' OR role = 'summary' OR (role = 'assistant' AND COALESCE(tool_calls, '') = ''))
                                 ORDER BY position, id LIMIT 2""",
                (conv_id,),
            ).fetchall()

        if len(messages) < 2:
            return jsonify({"title": conversation["title"]})

        settings = get_app_settings()
        active_tool_names = get_active_tool_names(settings)
        fetch_url_clip_aggressiveness = get_fetch_url_clip_aggressiveness(settings)
        fetch_url_token_threshold = get_fetch_url_token_threshold(settings)
        max_steps = max(1, min(10, int(settings.get("max_steps", 5))))

        prompt = prepend_runtime_context(
            [
                {
                    "role": "system",
                    "content": (
                        "Generate a short, descriptive title (max 6 words) for this conversation. "
                        "Return only the title — no quotes, no punctuation at the end."
                    ),
                },
                {
                    "role": messages[0]["role"],
                    "content": build_user_message_for_model(
                        messages[0]["content"],
                        parse_message_metadata(messages[0]["metadata"]),
                    ),
                },
                {"role": messages[1]["role"], "content": messages[1]["content"]},
            ],
            settings["user_preferences"],
            active_tool_names,
        )

        result = collect_agent_response(
            prompt,
            "deepseek-chat",
            max_steps,
            active_tool_names,
            fetch_url_token_threshold=fetch_url_token_threshold,
            fetch_url_clip_aggressiveness=fetch_url_clip_aggressiveness,
        )
        title = (result.get("content") or "").strip().strip('"').strip("'")[:120]
        if not title:
            errors = result.get("errors") or []
            return jsonify({"error": errors[-1] if errors else "No title returned."}), 502

        with get_db() as conn:
            conn.execute(
                "UPDATE conversations SET title = ?, updated_at = datetime('now') WHERE id = ?",
                (title, conv_id),
            )

        return jsonify({"title": title})


def preload_dependencies(app) -> None:
    if VISION_ENABLED:
        preload_local_ocr_engine(app)
    if RAG_ENABLED:
        preload_embedder()
