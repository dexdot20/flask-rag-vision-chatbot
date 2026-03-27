from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import requests as http_requests
from werkzeug.datastructures import MultiDict

import web_tools
from agent import (
    CONTEXT_OVERFLOW_RECOVERY_ERROR_TEXT,
    FINAL_ANSWER_ERROR_TEXT,
    FINAL_ANSWER_MISSING_TEXT,
    _build_compact_tool_message_content,
    _build_streaming_canvas_tool_preview,
    _estimate_input_breakdown,
    _execute_tool,
    _estimate_message_breakdown,
    _extract_partial_json_string_value,
    _is_context_overflow_error,
    _iter_agent_exchange_blocks,
    _prepare_tool_result_for_transcript,
    _try_compact_messages,
    collect_agent_response,
    run_agent_stream,
)
from app import create_app
from canvas_service import (
    build_canvas_project_manifest,
    create_canvas_runtime_state,
    find_latest_canvas_documents,
    find_latest_canvas_state,
    get_canvas_runtime_active_document_id,
    normalize_canvas_document,
    replace_canvas_lines,
    scroll_canvas_document,
)
from doc_service import build_canvas_markdown, infer_canvas_format, infer_canvas_language
from db import (
    build_user_profile_system_context,
    append_to_scratchpad,
    count_visible_message_tokens,
    create_image_asset,
    extract_message_usage,
    get_file_asset,
    get_canvas_expand_max_lines,
    get_canvas_prompt_max_lines,
    get_canvas_scroll_window_lines,
    get_active_tool_names,
    get_app_settings,
    get_db,
    get_image_asset,
    get_user_profile_entries,
    insert_message,
    normalize_active_tool_names,
    parse_message_metadata,
    save_app_settings,
    serialize_message_metadata,
    upsert_user_profile_entry,
)
from prune_service import _build_pruning_messages
from project_workspace_service import create_workspace_runtime_state
from messages import (
    SUMMARY_LABEL,
    _build_canvas_prompt_payload,
    build_api_messages,
    build_runtime_system_message,
    build_user_message_for_model,
    normalize_chat_messages,
    prepend_runtime_context,
)
from rag import Chunk
from rag.store import query_chunks, upsert_chunks
from rag_service import build_rag_auto_context, get_conversation_records_for_rag, get_exact_tool_memory_match, search_knowledge_base_tool, upsert_tool_memory_result
from routes.auth import AUTH_LAST_SEEN_KEY, AUTH_REMEMBER_KEY, AUTH_SESSION_KEY
from routes.chat import (
    OMITTED_TOOL_OUTPUT_TEXT,
    _count_prunable_message_tokens,
    _estimate_prompt_tokens,
    _is_failed_tool_summary,
    _select_recent_prompt_window,
    _select_summary_source_messages_by_token_budget,
    build_summary_prompt_messages,
    maybe_create_conversation_summary,
)
from tool_registry import TOOL_SPEC_BY_NAME, get_openai_tool_specs
from web_tools import (
    _extract_html,
    fetch_url_tool,
    load_proxies,
    search_news_ddgs_tool,
    search_news_google_tool,
    search_web_tool,
)
from token_utils import estimate_text_tokens


class AppRoutesTestCase(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = f"{self.temp_dir.name}/test.db"
        self.image_storage_dir = f"{self.temp_dir.name}/image-store"
        self.login_pin_patcher = patch("config.LOGIN_PIN", "")
        self.login_pin_patcher.start()
        self.app = create_app(database_path=self.db_path)
        self.app.config.update(TESTING=True)
        self.client = self.app.test_client()

    def tearDown(self):
        self.login_pin_patcher.stop()
        self.temp_dir.cleanup()

    def _create_conversation(self, title: str = "Test Chat") -> int:
        response = self.client.post(
            "/api/conversations",
            json={"title": title, "model": "deepseek-chat"},
        )
        self.assertEqual(response.status_code, 201)
        return response.get_json()["id"]

    @staticmethod
    def _stream_chunk(reasoning: str = "", content: str = "", tool_calls=None, usage=None):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        reasoning_content=reasoning,
                        content=content,
                        tool_calls=tool_calls or [],
                    )
                )
            ]
            if (reasoning or content or tool_calls)
            else [],
            usage=usage,
        )

    @staticmethod
    def _tool_call_chunk(name: str, arguments: dict, call_id: str = "tool-call-1", index: int = 0):
        return AppRoutesTestCase._stream_chunk(
            tool_calls=[
                {
                    "index": index,
                    "id": call_id,
                    "function": {
                        "name": name,
                        "arguments": json.dumps(arguments, ensure_ascii=False),
                    },
                }
            ]
        )

    def test_settings_roundtrip(self):
        response = self.client.get("/api/settings")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["scratchpad"], "")
        self.assertEqual(payload["max_steps"], 5)
        self.assertEqual(payload["canvas_prompt_max_lines"], 800)
        self.assertEqual(payload["canvas_expand_max_lines"], 1600)
        self.assertEqual(payload["canvas_scroll_window_lines"], 200)
        self.assertTrue(payload["rag_auto_inject"])
        self.assertEqual(payload["chat_summary_mode"], "auto")
        self.assertEqual(payload["chat_summary_trigger_token_count"], 80000)
        self.assertFalse(payload["pruning_enabled"])
        self.assertEqual(payload["pruning_token_threshold"], 80000)
        self.assertEqual(payload["pruning_batch_size"], 10)
        self.assertEqual(payload["fetch_url_token_threshold"], 3500)
        self.assertEqual(payload["fetch_url_clip_aggressiveness"], 50)
        self.assertEqual(payload["rag_sensitivity"], "strict")
        self.assertEqual(payload["rag_context_size"], "small")
        self.assertTrue(payload["tool_memory_auto_inject"])
        self.assertIn("features", payload)
        self.assertTrue(payload["features"]["rag_enabled"])
        self.assertTrue(payload["features"]["vision_enabled"])

        response = self.client.patch(
            "/api/settings",
            json={
                "user_preferences": "Keep answers short.",
                "max_steps": 3,
                "chat_summary_mode": "aggressive",
                "chat_summary_trigger_token_count": 9000,
                "pruning_enabled": True,
                "pruning_token_threshold": 12000,
                "pruning_batch_size": 4,
                "fetch_url_token_threshold": 4200,
                "fetch_url_clip_aggressiveness": 70,
                "canvas_prompt_max_lines": 1200,
                "canvas_expand_max_lines": 2200,
                "canvas_scroll_window_lines": 150,
                "active_tools": ["fetch_url", "search_web"],
                "rag_auto_inject": False,
                "rag_sensitivity": "strict",
                "rag_context_size": "large",
                "tool_memory_auto_inject": False,
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["user_preferences"], "Keep answers short.")
        self.assertEqual(payload["scratchpad"], "")
        self.assertEqual(payload["max_steps"], 3)
        self.assertEqual(payload["chat_summary_mode"], "aggressive")
        self.assertEqual(payload["chat_summary_trigger_token_count"], 9000)
        self.assertTrue(payload["pruning_enabled"])
        self.assertEqual(payload["pruning_token_threshold"], 12000)
        self.assertEqual(payload["pruning_batch_size"], 4)
        self.assertEqual(payload["fetch_url_token_threshold"], 4200)
        self.assertEqual(payload["fetch_url_clip_aggressiveness"], 70)
        self.assertEqual(payload["canvas_prompt_max_lines"], 1200)
        self.assertEqual(payload["canvas_expand_max_lines"], 2200)
        self.assertEqual(payload["canvas_scroll_window_lines"], 150)
        self.assertEqual(payload["active_tools"], ["fetch_url", "search_web"])
        self.assertFalse(payload["rag_auto_inject"])
        self.assertEqual(payload["rag_sensitivity"], "strict")
        self.assertEqual(payload["rag_context_size"], "large")
        self.assertFalse(payload["tool_memory_auto_inject"])

    def test_settings_patch_rejects_invalid_rag_presets(self):
        response = self.client.patch(
            "/api/settings",
            json={"rag_sensitivity": "aggressive"},
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("rag_sensitivity", response.get_json()["error"])

        response = self.client.patch(
            "/api/settings",
            json={"rag_context_size": "huge"},
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("rag_context_size", response.get_json()["error"])

    def test_settings_patch_rejects_invalid_pruning_values(self):
        response = self.client.patch(
            "/api/settings",
            json={"pruning_token_threshold": 999},
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("pruning_token_threshold", response.get_json()["error"])

        response = self.client.patch(
            "/api/settings",
            json={"pruning_batch_size": 0},
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("pruning_batch_size", response.get_json()["error"])

    def test_settings_patch_rejects_invalid_canvas_values(self):
        response = self.client.patch(
            "/api/settings",
            json={"canvas_prompt_max_lines": 99},
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("canvas_prompt_max_lines", response.get_json()["error"])

        response = self.client.patch(
            "/api/settings",
            json={"canvas_expand_max_lines": 5000},
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("canvas_expand_max_lines", response.get_json()["error"])

        response = self.client.patch(
            "/api/settings",
            json={"canvas_scroll_window_lines": 49},
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("canvas_scroll_window_lines", response.get_json()["error"])

    def test_login_pin_protects_page_and_api_routes(self):
        with patch("config.LOGIN_PIN", "2468"):
            response = self.client.get("/login")
            self.assertEqual(response.status_code, 200)
            self.assertIn("Enter PIN", response.get_data(as_text=True))

            response = self.client.get("/")
            self.assertEqual(response.status_code, 302)
            self.assertIn("/login", response.headers["Location"])

            response = self.client.get("/api/settings")
            self.assertEqual(response.status_code, 401)
            self.assertEqual(response.get_json()["error"], "Login PIN required.")

            response = self.client.post("/login", data={"pin": "0000"})
            self.assertEqual(response.status_code, 401)
            self.assertIn("Invalid PIN.", response.get_data(as_text=True))

            response = self.client.post("/login", data={"pin": "2468"})
            self.assertEqual(response.status_code, 302)
            self.assertEqual(response.headers["Location"], "/")

            response = self.client.get("/")
            self.assertEqual(response.status_code, 200)

    def test_login_pin_times_out_without_remember(self):
        with patch("config.LOGIN_PIN", "2468"), patch("config.LOGIN_SESSION_TIMEOUT_MINUTES", 1):
            response = self.client.post("/login", data={"pin": "2468"})
            self.assertEqual(response.status_code, 302)

            with self.client.session_transaction() as session_data:
                session_data[AUTH_SESSION_KEY] = True
                session_data[AUTH_REMEMBER_KEY] = False
                session_data[AUTH_LAST_SEEN_KEY] = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()

            response = self.client.get("/settings")
            self.assertEqual(response.status_code, 302)
            self.assertIn("/login", response.headers["Location"])

    def test_login_pin_remember_me_skips_timeout(self):
        with patch("config.LOGIN_PIN", "2468"), patch("config.LOGIN_SESSION_TIMEOUT_MINUTES", 1):
            response = self.client.post("/login", data={"pin": "2468", "remember": "on"})
            self.assertEqual(response.status_code, 302)

            with self.client.session_transaction() as session_data:
                session_data[AUTH_SESSION_KEY] = True
                session_data[AUTH_REMEMBER_KEY] = True
                session_data[AUTH_LAST_SEEN_KEY] = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()

            response = self.client.get("/settings")
            self.assertEqual(response.status_code, 200)

    def test_login_pin_locks_out_after_failed_attempts(self):
        with patch("config.LOGIN_PIN", "2468"), patch("config.LOGIN_MAX_FAILED_ATTEMPTS", 2), patch(
            "config.LOGIN_LOCKOUT_SECONDS", 60
        ):
            response = self.client.post("/login", data={"pin": "0000"})
            self.assertEqual(response.status_code, 401)

            response = self.client.post("/login", data={"pin": "1111"})
            self.assertEqual(response.status_code, 429)
            self.assertIn("Too many failed attempts.", response.get_data(as_text=True))

            response = self.client.post("/login", data={"pin": "2468"})
            self.assertEqual(response.status_code, 429)

    def test_canvas_limit_getters_clamp_values(self):
        settings = get_app_settings()
        settings["canvas_prompt_max_lines"] = "50000"
        settings["canvas_expand_max_lines"] = "-1"
        settings["canvas_scroll_window_lines"] = "nope"

        self.assertEqual(get_canvas_prompt_max_lines(settings), 3000)
        self.assertEqual(get_canvas_expand_max_lines(settings), 100)
        self.assertEqual(get_canvas_scroll_window_lines(settings), 200)

    def test_build_canvas_prompt_payload_respects_max_lines(self):
        content = "\n".join(f"line {index}" for index in range(1, 51))
        document = normalize_canvas_document(
            {
                "id": "doc-1",
                "title": "Large file",
                "format": "code",
                "language": "python",
                "content": content,
            }
        )

        payload = _build_canvas_prompt_payload([document], max_lines=10)

        self.assertIsNotNone(payload)
        self.assertEqual(len(payload["visible_lines"]), 10)
        self.assertEqual(payload["visible_line_end"], 10)
        self.assertTrue(payload["is_truncated"])

    def test_scroll_canvas_document_returns_window_flags(self):
        content = "\n".join(f"line {index}" for index in range(1, 101))
        runtime_state = create_canvas_runtime_state(
            [
                {
                    "id": "doc-1",
                    "title": "Large file",
                    "format": "code",
                    "language": "python",
                    "content": content,
                }
            ]
        )

        result = scroll_canvas_document(runtime_state, 20, 60, max_window_lines=15)

        self.assertEqual(result["start_line"], 20)
        self.assertEqual(result["end_line_actual"], 34)
        self.assertEqual(len(result["visible_lines"]), 15)
        self.assertTrue(result["has_more_above"])
        self.assertTrue(result["has_more_below"])

    def test_execute_tool_scroll_canvas_document_uses_runtime_window_limit(self):
        content = "\n".join(f"line {index}" for index in range(1, 101))
        runtime_state = {
            "canvas": create_canvas_runtime_state(
                [
                    {
                        "id": "doc-1",
                        "title": "Large file",
                        "format": "code",
                        "language": "python",
                        "content": content,
                    }
                ]
            ),
            "canvas_limits": {"scroll_window_lines": 12},
        }

        result, summary = _execute_tool(
            "scroll_canvas_document",
            {"start_line": 5, "end_line": 99},
            runtime_state=runtime_state,
        )

        self.assertEqual(result["action"], "scrolled")
        self.assertEqual(result["start_line"], 5)
        self.assertEqual(result["end_line_actual"], 16)
        self.assertIn("Canvas scrolled", summary)

    def test_runtime_system_message_mentions_canvas_scroll_for_truncated_excerpt(self):
        content = "\n".join(f"line {index}" for index in range(1, 51))
        document = normalize_canvas_document(
            {
                "id": "doc-1",
                "title": "Large file",
                "format": "code",
                "language": "python",
                "content": content,
            }
        )

        message = build_runtime_system_message(
            canvas_documents=[document],
            canvas_prompt_max_lines=10,
        )

        self.assertIn("scroll_canvas_document", message["content"])
        self.assertIn("expand_canvas_document", message["content"])

    def test_extract_partial_json_string_value_handles_partial_escapes(self):
        arguments_text = "{\"title\":\"Plan\",\"content\":\"Line 1\\nLine 2\\u00e7 ve \\\"quote\\\""

        extracted = _extract_partial_json_string_value(arguments_text, "content")

        self.assertEqual(extracted, 'Line 1\nLine 2ç ve "quote"')

    def test_build_streaming_canvas_tool_preview_reads_partial_canvas_args(self):
        tool_call_parts = [
            {
                "name": "create_canvas_document",
                "arguments_parts": [
                    '{"title":"Spec","format":"code","language":"python","content":"print(1)\\nprint(2)"',
                ],
            }
        ]

        preview = _build_streaming_canvas_tool_preview(tool_call_parts)

        self.assertEqual(preview["tool"], "create_canvas_document")
        self.assertEqual(preview["snapshot"]["title"], "Spec")
        self.assertEqual(preview["snapshot"]["format"], "code")
        self.assertEqual(preview["snapshot"]["language"], "python")
        self.assertEqual(preview["content"], "print(1)\nprint(2)")

    def test_create_conversation_rejects_invalid_model(self):
        response = self.client.post(
            "/api/conversations",
            json={"title": "Test Chat", "model": "invalid-model"},
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["error"], "Invalid model.")

    def test_create_app_runs_bootstrap_rag_sync(self):
        with patch("app.sync_conversations_to_rag_safe") as mocked_sync:
            create_app(database_path=f"{self.temp_dir.name}/bootstrap.db")

        mocked_sync.assert_called_once_with()

    def test_database_initialization_adds_rag_document_expiration_column(self):
        with get_db() as conn:
            columns = {row["name"] for row in conn.execute("PRAGMA table_info(rag_documents)").fetchall()}

        self.assertIn("expires_at", columns)

    def test_create_conversation_triggers_auto_rag_sync(self):
        with patch("routes.conversations.sync_conversations_to_rag_safe") as mocked_sync:
            response = self.client.post(
                "/api/conversations",
                json={"title": "Test Chat", "model": "deepseek-chat"},
            )

        self.assertEqual(response.status_code, 201)
        mocked_sync.assert_called_once_with(conversation_id=response.get_json()["id"])

    def test_update_conversation_title_triggers_auto_rag_sync(self):
        with patch("routes.conversations.sync_conversations_to_rag_safe") as mocked_sync:
            conversation_id = self._create_conversation()
            mocked_sync.reset_mock()

            response = self.client.patch(
                f"/api/conversations/{conversation_id}",
                json={"title": "Renamed Chat"},
            )

        self.assertEqual(response.status_code, 200)
        mocked_sync.assert_called_once_with(conversation_id=conversation_id)

    def test_chat_route_triggers_auto_rag_sync_after_persist(self):
        fake_events = iter(
            [
                {"type": "answer_start"},
                {"type": "answer_delta", "text": "Done."},
                {"type": "tool_capture", "tool_results": [], "canvas_documents": []},
                {"type": "done"},
            ]
        )

        with patch("routes.conversations.sync_conversations_to_rag_safe") as conversation_sync:
            conversation_id = self._create_conversation()
            conversation_sync.reset_mock()

        with patch("routes.chat.run_agent_stream", return_value=fake_events), patch(
            "routes.chat.sync_conversations_to_rag_safe"
        ) as chat_sync:
            response = self.client.post(
                "/chat",
                json={
                    "conversation_id": conversation_id,
                    "model": "deepseek-chat",
                    "user_content": "Hello",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )
            response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        chat_sync.assert_called_once_with(conversation_id=conversation_id)

    def test_get_conversation_records_for_rag_excludes_soft_deleted_messages(self):
        conversation_id = self._create_conversation()
        assistant_metadata = serialize_message_metadata(
            {
                "tool_results": [
                    {
                        "tool_name": "fetch_url",
                        "content": "Stale tool result",
                    }
                ]
            }
        )

        with get_db() as conn:
            insert_message(conn, conversation_id, "user", "Original prompt")
            deleted_assistant_id = insert_message(
                conn,
                conversation_id,
                "assistant",
                "Outdated answer",
                metadata=assistant_metadata,
            )
            insert_message(conn, conversation_id, "assistant", "Current answer")
            conn.execute(
                "UPDATE messages SET deleted_at = datetime('now') WHERE id = ?",
                (deleted_assistant_id,),
            )

        records = get_conversation_records_for_rag(conversation_id)

        self.assertEqual(len(records), 1)
        self.assertEqual(
            records[0]["messages"],
            [
                {"role": "user", "content": "Original prompt"},
                {"role": "assistant", "content": "Current answer"},
            ],
        )
        self.assertEqual(records[0]["tool_results"], [])

    def test_chat_edit_resyncs_rag_before_retrieval(self):
        fake_events = iter(
            [
                {"type": "answer_start"},
                {"type": "answer_delta", "text": "Updated answer."},
                {"type": "tool_capture", "tool_results": [], "canvas_documents": []},
                {"type": "done"},
            ]
        )

        with patch("routes.conversations.sync_conversations_to_rag_safe"):
            conversation_id = self._create_conversation()

        with get_db() as conn:
            edited_message_id = insert_message(conn, conversation_id, "user", "Original prompt")
            insert_message(conn, conversation_id, "assistant", "Original answer")

        def check_rag_context(*args, **kwargs):
            self.assertTrue(chat_sync.called)
            self.assertEqual(chat_sync.call_args.kwargs, {"conversation_id": conversation_id})
            return None

        with patch("routes.chat.run_agent_stream", return_value=fake_events), patch(
            "routes.chat.sync_conversations_to_rag_safe"
        ) as chat_sync, patch("routes.chat.build_rag_auto_context", side_effect=check_rag_context):
            response = self.client.post(
                "/chat",
                json={
                    "conversation_id": conversation_id,
                    "model": "deepseek-chat",
                    "edited_message_id": edited_message_id,
                    "user_content": "Edited prompt",
                    "messages": [{"role": "user", "content": "Edited prompt"}],
                },
            )
            response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)

    def test_chat_edit_restores_canvas_state_from_that_point(self):
        captured = {}
        conversation_id = self._create_conversation()

        with get_db() as conn:
            insert_message(conn, conversation_id, "user", "First prompt")
            insert_message(
                conn,
                conversation_id,
                "assistant",
                "Initial canvas",
                metadata=serialize_message_metadata(
                    {
                        "canvas_documents": [
                            {
                                "id": "canvas-a",
                                "title": "draft-a.md",
                                "format": "markdown",
                                "content": "# Draft A",
                            }
                        ],
                        "active_document_id": "canvas-a",
                    }
                ),
            )
            edited_message_id = insert_message(conn, conversation_id, "user", "Second prompt")
            insert_message(
                conn,
                conversation_id,
                "assistant",
                "Updated canvas",
                metadata=serialize_message_metadata(
                    {
                        "canvas_documents": [
                            {
                                "id": "canvas-b",
                                "title": "draft-b.md",
                                "format": "markdown",
                                "content": "# Draft B",
                            }
                        ],
                        "active_document_id": "canvas-b",
                    }
                ),
            )

        def fake_run_agent_stream(*args, **kwargs):
            captured["initial_canvas_documents"] = kwargs.get("initial_canvas_documents") or []
            captured["initial_canvas_active_document_id"] = kwargs.get("initial_canvas_active_document_id")
            return iter([{"type": "done"}])

        with patch("routes.chat.run_agent_stream", side_effect=fake_run_agent_stream), patch(
            "routes.chat.sync_conversations_to_rag_safe"
        ):
            response = self.client.post(
                "/chat",
                json={
                    "conversation_id": conversation_id,
                    "model": "deepseek-chat",
                    "edited_message_id": edited_message_id,
                    "user_content": "Second prompt revised",
                    "messages": [{"role": "user", "content": "Second prompt revised"}],
                },
            )
            response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertEqual([doc["title"] for doc in captured["initial_canvas_documents"]], ["draft-a.md"])
        self.assertEqual(captured["initial_canvas_active_document_id"], "canvas-a")

        conversation_response = self.client.get(f"/api/conversations/{conversation_id}")
        self.assertEqual(conversation_response.status_code, 200)
        messages = conversation_response.get_json()["messages"]
        runtime_state = find_latest_canvas_state(messages)
        self.assertEqual([doc["title"] for doc in runtime_state["documents"]], ["draft-a.md"])

    def test_chat_route_defers_postprocess_outside_testing(self):
        fake_events = iter(
            [
                {"type": "answer_start"},
                {"type": "answer_delta", "text": "Done."},
                {"type": "tool_capture", "tool_results": [], "canvas_documents": []},
                {"type": "done"},
            ]
        )

        conversation_id = self._create_conversation()
        previous_testing = self.app.config.get("TESTING", False)
        self.app.config["TESTING"] = False

        try:
            with patch("routes.chat.run_agent_stream", return_value=fake_events), patch(
                "routes.chat.POST_RESPONSE_EXECUTOR.submit"
            ) as mocked_submit, patch("routes.chat.SUMMARY_EXECUTOR.submit") as mocked_summary_submit:
                response = self.client.post(
                    "/chat",
                    json={
                        "conversation_id": conversation_id,
                        "model": "deepseek-chat",
                        "user_content": "Hello",
                        "messages": [{"role": "user", "content": "Hello"}],
                    },
                )
                response.get_data(as_text=True)
        finally:
            self.app.config["TESTING"] = previous_testing

        self.assertEqual(response.status_code, 200)
        mocked_submit.assert_called_once()
        mocked_summary_submit.assert_not_called()

    def test_generate_title_triggers_auto_rag_sync(self):
        with patch("routes.conversations.sync_conversations_to_rag_safe"):
            conversation_id = self._create_conversation()

        with get_db() as conn:
            insert_message(conn, conversation_id, "user", "Need a title")
            insert_message(conn, conversation_id, "assistant", "Sure, here is the answer")
            conn.execute(
                "UPDATE conversations SET updated_at = datetime('now') WHERE id = ?",
                (conversation_id,),
            )

        with patch(
            "routes.chat.collect_agent_response",
            return_value={"content": "Better Title", "errors": []},
        ), patch("routes.chat.sync_conversations_to_rag_safe") as chat_sync:
            response = self.client.post(f"/api/conversations/{conversation_id}/generate-title")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["title"], "Better Title")
        chat_sync.assert_called_once_with(conversation_id=conversation_id)

    def test_manual_prune_endpoint_updates_message_and_preserves_original(self):
        conversation_id = self._create_conversation()
        with get_db() as conn:
            message_id = insert_message(conn, conversation_id, "assistant", "Bu mesaj gereksiz ayrıntılar içeriyor ve budanmalı.")

        mock_response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Bu mesaj budanmış halidir."))]
        )
        with patch("prune_service.client.chat.completions.create", return_value=mock_response), patch(
            "routes.conversations.sync_conversations_to_rag_safe"
        ) as mocked_sync:
            response = self.client.post(
                f"/api/messages/{message_id}/prune",
                json={"conversation_id": conversation_id},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["pruned"])
        self.assertEqual(payload["message"]["content"], "Bu mesaj budanmış halidir.")
        self.assertTrue(payload["message"]["metadata"]["is_pruned"])
        self.assertEqual(
            payload["message"]["metadata"]["pruned_original"],
            "Bu mesaj gereksiz ayrıntılar içeriyor ve budanmalı.",
        )
        mocked_sync.assert_called_once_with(conversation_id=conversation_id)

    def test_pruning_prompt_requires_preserving_critical_and_code_content(self):
        prompt_messages = _build_pruning_messages(
            """Bunu kısalt ama şu kodu bozma:\n\n```python\ndef add(a, b):\n    return a + b\n```\n\nAPI key: sk-test-12345\nURL: https://example.com/docs\nSayi: 4096"""
        )

        system_prompt = prompt_messages[0]["content"]
        user_prompt = prompt_messages[1]["content"]

        self.assertIn("all critical facts", system_prompt)
        self.assertIn("code blocks", system_prompt)
        self.assertIn("keep those sections verbatim", system_prompt)
        self.assertIn("Preserve the message's core idea", user_prompt)
        self.assertIn("Code blocks", user_prompt)
        self.assertIn("JSON", user_prompt)
        self.assertIn("URLs", user_prompt)
        self.assertIn("must be kept verbatim", user_prompt)

    def test_background_post_response_pruning_runs_when_threshold_exceeded(self):
        fake_events = iter(
            [
                {"type": "answer_start"},
                {"type": "answer_delta", "text": "Kısa cevap."},
                {"type": "tool_capture", "tool_results": [], "canvas_documents": []},
                {"type": "done"},
            ]
        )

        conversation_id = self._create_conversation()

        with patch("routes.chat.run_agent_stream", return_value=fake_events), patch(
            "routes.chat.maybe_create_conversation_summary",
            return_value={"applied": False},
        ), patch("routes.chat._count_prunable_message_tokens", return_value=90_000), patch(
            "routes.chat.prune_conversation_batch"
        ) as mocked_prune, patch("routes.chat.sync_conversations_to_rag_safe"):
            response = self.client.patch(
                "/api/settings",
                json={
                    "pruning_enabled": True,
                    "pruning_token_threshold": 80000,
                    "pruning_batch_size": 3,
                },
            )
            self.assertEqual(response.status_code, 200)

            response = self.client.post(
                "/chat",
                json={
                    "conversation_id": conversation_id,
                    "model": "deepseek-chat",
                    "user_content": "Merhaba",
                    "messages": [{"role": "user", "content": "Merhaba"}],
                },
            )
            response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        mocked_prune.assert_called_once_with(conversation_id, 3)

    def test_count_prunable_message_tokens_ignores_tool_and_summary_messages(self):
        messages = [
            {"role": "user", "content": "Visible user text"},
            {"role": "assistant", "content": "Visible assistant text"},
            {
                "role": "assistant",
                "content": "Tool call envelope",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "search_web", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "content": "Very large tool result" * 1000},
            {"role": "summary", "content": "Conversation summary"},
            {"role": "assistant", "content": "", "metadata": {"is_pruned": True}},
        ]

        expected = _count_prunable_message_tokens(
            [
                {"role": "user", "content": "Visible user text"},
                {"role": "assistant", "content": "Visible assistant text"},
            ]
        )

        self.assertEqual(_count_prunable_message_tokens(messages), expected)

    def test_active_tools_include_replace_scratchpad_for_existing_scratchpad_mode(self):
        settings = {"active_tools": json.dumps(["append_scratchpad", "search_web"]) }
        self.assertIn("replace_scratchpad", get_active_tool_names(settings))

    def test_db_connections_enable_busy_timeout_and_wal_mode(self):
        with get_db() as conn:
            busy_timeout = conn.execute("PRAGMA busy_timeout").fetchone()[0]
            journal_mode = conn.execute("PRAGMA journal_mode").fetchone()[0]

        self.assertEqual(busy_timeout, 30000)
        self.assertEqual(str(journal_mode).lower(), "wal")

    def test_disabled_features_reflect_in_settings_and_routes(self):
        with patch("config.RAG_ENABLED", False), patch("db.RAG_ENABLED", False), patch("routes.pages.RAG_ENABLED", False), patch("routes.conversations.RAG_ENABLED", False):
            response = self.client.get("/api/settings")
            self.assertEqual(response.status_code, 200)
            payload = response.get_json()
            self.assertFalse(payload["rag_auto_inject"])
            self.assertFalse(payload["features"]["rag_enabled"])

            response = self.client.patch(
                "/api/settings",
                json={"rag_auto_inject": True},
            )
            self.assertEqual(response.status_code, 200)
            self.assertFalse(response.get_json()["rag_auto_inject"])

            response = self.client.get("/api/rag/documents")
            self.assertEqual(response.status_code, 410)

            conversation_id = self._create_conversation()
            response = self.client.delete(f"/api/conversations/{conversation_id}")
            self.assertEqual(response.status_code, 204)

        with patch("routes.chat.VISION_ENABLED", False):
            response = self.client.post(
                "/chat",
                data={
                    "messages": json.dumps([{"role": "user", "content": "Test"}]),
                    "model": "deepseek-chat",
                    "conversation_id": "",
                    "user_content": "Test",
                    "image": (io.BytesIO(b"fake image bytes"), "test.png"),
                },
            )
            self.assertEqual(response.status_code, 410)

    def test_runtime_system_message_includes_explicit_current_date_and_time(self):
        now = datetime(2026, 3, 15, 21, 42, 5, tzinfo=timezone(timedelta(hours=3)))

        message = build_runtime_system_message(
            user_preferences="Keep answers short.",
            scratchpad="The user is 22 years old.",
            active_tool_names=[
                "append_scratchpad",
                "ask_clarifying_question",
                "image_explain",
                "search_knowledge_base",
                "search_tool_memory",
            ],
            retrieved_context="Context block",
            tool_memory_context="Remembered web result",
            now=now,
        )

        self.assertEqual(message["role"], "system")
        content = message["content"]
        self.assertIn("## Current Date and Time", content)
        self.assertIn("2026-03-15T21:42:05+03:00", content)
        self.assertIn("User Preferences\nKeep answers short.", content)
        self.assertIn("Scratchpad (AI Persistent Memory)", content)
        self.assertIn("The user is 22 years old.", content)
        self.assertIn("Only durable, high-signal facts", content)
        self.assertIn("Web findings", content)
        self.assertIn("Ask whether this information will still matter in a future conversation", content)
        self.assertIn("Never save them just because they were requested.", content)
        self.assertNotIn("Err on the side of saving if in doubt.", content)
        self.assertIn("Clarification**: If a good answer depends", content)
        self.assertIn("Image Follow-up**: Use for follow-up questions", content)
        self.assertIn("Tool Memory", content)
        self.assertIn("Remembered web result", content)
        self.assertIn("Knowledge Base", content)
        self.assertIn("Context block", content)

    def test_runtime_system_message_includes_user_profile_context(self):
        upsert_user_profile_entry("pref:concise", "The user prefers concise answers.", confidence=0.95, source="manual")

        message = build_runtime_system_message(
            user_profile_context=build_user_profile_system_context(),
            active_tool_names=[],
        )

        content = message["content"]
        self.assertIn("## User Profile", content)
        self.assertIn("The user prefers concise answers.", content)

    def test_structured_summary_persists_user_profile_facts(self):
        conversation_id = self._create_conversation()
        with get_db() as conn:
            insert_message(conn, conversation_id, "user", "Please keep answers short and concise in future replies.")
            insert_message(conn, conversation_id, "assistant", "Understood. I will keep future answers concise.")

        summary_payload = {
            "content": json.dumps(
                {
                    "facts": [
                        "The user prefers concise answers in future replies.",
                        "The user is working on an os-chatbot codebase and wants continuity.",
                    ],
                    "decisions": ["Future replies should stay concise."],
                    "open_issues": [],
                    "entities": ["os-chatbot"],
                    "tool_outcomes": [],
                },
                ensure_ascii=False,
            ),
            "errors": [],
        }

        settings = {**get_app_settings(), "summary_skip_first": "0", "summary_skip_last": "0"}
        with patch("routes.chat.collect_agent_response", return_value=summary_payload):
            outcome = maybe_create_conversation_summary(
                conversation_id,
                "deepseek-chat",
                settings,
                3500,
                50,
                force=True,
            )

        self.assertTrue(outcome["applied"])
        entries = get_user_profile_entries()
        self.assertIn("The user prefers concise answers in future replies.", [entry["value"] for entry in entries])
        self.assertGreaterEqual(outcome.get("stored_profile_fact_count", 0), 1)

    def test_query_chunks_skips_expired_metadata(self):
        now_ts = int(datetime.now(timezone.utc).timestamp())
        future_ts = now_ts + 3600
        past_ts = now_ts - 3600

        fake_collection = Mock()
        fake_collection.query.return_value = {
            "documents": [["expired result", "fresh result"]],
            "metadatas": [[
                {"source_key": "expired", "source_type": "tool_memory", "category": "tool_memory", "expires_at_ts": past_ts},
                {"source_key": "fresh", "source_type": "tool_memory", "category": "tool_memory", "expires_at_ts": future_ts},
            ]],
            "distances": [[0.1, 0.05]],
            "ids": [["old-id", "fresh-id"]],
        }

        with patch("rag.store._iter_query_collections", return_value=[(fake_collection, {"category": "tool_memory"})]), patch("rag.store.embed_texts", return_value=[[0.1, 0.2]]):
            rows = query_chunks("latest result", top_k=5, category="tool_memory")

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["id"], "fresh-id")
        self.assertEqual(rows[0]["metadata"]["source_key"], "fresh")

    def test_upsert_chunks_writes_to_category_collections(self):
        collection_conversation = Mock()
        collection_tool_memory = Mock()

        chunks = [
            Chunk(
                id="chunk-1",
                text="conversation text",
                source_name="conversation-doc",
                source_type="conversation",
                category="conversation",
                chunk_index=0,
                metadata={"source_key": "src-1"},
            ),
            Chunk(
                id="chunk-2",
                text="tool memory text",
                source_name="tool-memory-doc",
                source_type="tool_memory",
                category="tool_memory",
                chunk_index=0,
                metadata={"source_key": "src-2"},
            ),
        ]

        def fake_get_collection(name="knowledge_base"):
            if name == "knowledge_base__conversation":
                return collection_conversation
            if name == "knowledge_base__tool_memory":
                return collection_tool_memory
            return Mock()

        with patch("rag.store.get_collection", side_effect=fake_get_collection), patch("rag.store.embed_texts", return_value=[[0.1, 0.2], [0.3, 0.4]]):
            inserted = upsert_chunks(chunks)

        self.assertEqual(inserted, 2)
        collection_conversation.upsert.assert_called_once()
        collection_tool_memory.upsert.assert_called_once()

    def test_get_exact_tool_memory_match_returns_joined_chunk_content(self):
        with patch("rag_service.ensure_supported_rag_sources"), patch(
            "rag_service.get_rag_document_record",
            return_value={
                "source_key": "src-1",
                "source_name": "fetch_url: https://example.com/page",
                "metadata": json.dumps({"summary": "Page content extracted: Example"}),
                "expires_at": None,
            },
        ), patch(
            "rag_service.rag_get_source_chunks",
            return_value=[
                {"id": "chunk-1", "text": "Title: Example", "metadata": {"chunk_index": 0}},
                {"id": "chunk-2", "text": "Body: Retrieved content", "metadata": {"chunk_index": 1}},
            ],
        ):
            match = get_exact_tool_memory_match("fetch_url", "https://example.com/page")

        self.assertIsNotNone(match)
        self.assertEqual(match["summary"], "Page content extracted: Example")
        self.assertIn("Title: Example", match["content"])
        self.assertIn("Body: Retrieved content", match["content"])

    def test_upsert_tool_memory_result_assigns_ttl_metadata(self):
        with patch("rag_service.time.time", return_value=1_000), patch("rag_service.ingest_rag_chunks") as mocked_ingest:
            mocked_ingest.return_value = {"ok": True}
            upsert_tool_memory_result(
                "search_news_ddgs",
                "economy headlines",
                "headline body",
                "brief summary",
            )

        _, kwargs = mocked_ingest.call_args
        self.assertEqual(kwargs["expires_at"], "1970-01-01 02:16:40")
        self.assertEqual(kwargs["metadata"]["expires_at_ts"], 8_200)

    def test_search_knowledge_base_tool_uses_query_expansion_and_dedupes_hits(self):
        original_query = "python liste sıralama nasıl yapılır"

        def fake_query(query, top_k=5, category=None):
            if query == original_query:
                return [
                    {
                        "id": "chunk-1",
                        "text": "sort docs",
                        "metadata": {"source_key": "src-1", "source_name": "doc-1", "source_type": "conversation", "category": "conversation", "chunk_index": 0, "indexed_at_ts": 2_000},
                        "similarity": 0.40,
                    }
                ]
            return [
                {
                    "id": "chunk-1",
                    "text": "sort docs",
                    "metadata": {"source_key": "src-1", "source_name": "doc-1", "source_type": "conversation", "category": "conversation", "chunk_index": 0, "indexed_at_ts": 2_000},
                    "similarity": 0.52,
                },
                {
                    "id": "chunk-2",
                    "text": "list order",
                    "metadata": {"source_key": "src-2", "source_name": "doc-2", "source_type": "conversation", "category": "conversation", "chunk_index": 1, "indexed_at_ts": 2_000},
                    "similarity": 0.45,
                },
            ]

        with patch("rag_service.ensure_supported_rag_sources"), patch("rag_service.rag_query_chunks", side_effect=fake_query) as mocked_query, patch("rag_service.time.time", return_value=2_000):
            result = search_knowledge_base_tool(original_query, top_k=5)

        self.assertGreaterEqual(mocked_query.call_count, 2)
        self.assertEqual(result["count"], 2)
        self.assertEqual([match["id"] for match in result["matches"]], ["chunk-1", "chunk-2"])

    def test_build_rag_auto_context_prefers_recent_hits_with_temporal_decay(self):
        old_timestamp = int((datetime.now(timezone.utc) - timedelta(days=60)).timestamp())
        new_timestamp = int(datetime.now(timezone.utc).timestamp())
        fake_hits = [
            {
                "id": "old-hit",
                "text": "older memory",
                "metadata": {"source_key": "old", "source_name": "Old", "source_type": "tool_memory", "category": "tool_memory", "chunk_index": 0, "indexed_at_ts": old_timestamp},
                "similarity": 0.50,
            },
            {
                "id": "new-hit",
                "text": "newer memory",
                "metadata": {"source_key": "new", "source_name": "New", "source_type": "tool_memory", "category": "tool_memory", "chunk_index": 0, "indexed_at_ts": new_timestamp},
                "similarity": 0.50,
            },
        ]

        with patch("rag_service.ensure_supported_rag_sources"), patch("rag_service.rag_query_chunks", return_value=fake_hits), patch("rag_service.time.time", return_value=new_timestamp):
            result = build_rag_auto_context("recent memory", True, threshold=0.1, top_k=5)

        self.assertIsNotNone(result)
        self.assertEqual(result["matches"][0]["id"], "new-hit")
        self.assertGreater(result["matches"][0]["similarity"], result["matches"][1]["similarity"])

    def test_build_rag_auto_context_excludes_current_conversation_sources(self):
        fake_hits = [
            {
                "id": "conversation-hit",
                "text": "conversation memory",
                "metadata": {"source_key": "conversation-1", "source_name": "Conversation", "source_type": "conversation", "category": "conversation", "chunk_index": 0},
                "similarity": 0.98,
            },
            {
                "id": "tool-hit",
                "text": "tool memory",
                "metadata": {"source_key": "tool-1", "source_name": "Tool result", "source_type": "tool_result", "category": "tool_result", "chunk_index": 0},
                "similarity": 0.95,
            },
            {
                "id": "other-hit",
                "text": "other memory",
                "metadata": {"source_key": "other-1", "source_name": "Other", "source_type": "tool_memory", "category": "tool_memory", "chunk_index": 0},
                "similarity": 0.90,
            },
        ]

        with patch("rag_service.ensure_supported_rag_sources"), patch("rag_service.rag_query_chunks", return_value=fake_hits):
            result = build_rag_auto_context(
                "recent memory",
                True,
                threshold=0.1,
                top_k=5,
                exclude_source_keys={"conversation-1", "tool-1"},
            )

        self.assertIsNotNone(result)
        self.assertEqual([match["id"] for match in result["matches"]], ["other-hit"])

    def test_runtime_system_message_hides_canvas_edit_tools_without_canvas_document(self):
        message = build_runtime_system_message(
            active_tool_names=[
                "expand_canvas_document",
                "create_canvas_document",
                "rewrite_canvas_document",
                "replace_canvas_lines",
                "insert_canvas_lines",
                "delete_canvas_lines",
                "delete_canvas_document",
                "clear_canvas",
            ],
        )

        content = message["content"]
        self.assertIn("## Tool Calling", content)
        self.assertIn("Native function calling is enabled for this turn.", content)
        self.assertNotIn("## Active Canvas Document", content)
        self.assertNotIn("Available Tools", content)

    def test_runtime_system_message_includes_active_canvas_document_context(self):
        message = build_runtime_system_message(
            active_tool_names=[
                "create_canvas_document",
                "rewrite_canvas_document",
                "replace_canvas_lines",
            ],
            canvas_documents=[
                {
                    "id": "canvas-1",
                    "title": "main.py",
                    "format": "markdown",
                    "language": "python",
                    "content": "print('hello')\nprint('world')",
                }
            ],
        )

        content = message["content"]
        self.assertIn("## Active Canvas Document", content)
        self.assertIn("- Language: python", content)
        self.assertIn("1: print('hello')", content)
        self.assertIn("2: print('world')", content)
        self.assertIn("## Tool Calling", content)
        self.assertIn("Use only the tools exposed by the API for this turn", content)

    def test_runtime_system_message_includes_canvas_project_manifest(self):
        message = build_runtime_system_message(
            active_tool_names=[
                "create_canvas_document",
                "rewrite_canvas_document",
                "replace_canvas_lines",
            ],
            canvas_documents=[
                {
                    "id": "canvas-1",
                    "title": "app.py",
                    "path": "src/app.py",
                    "role": "source",
                    "project_id": "demo-app",
                    "workspace_id": "demo-workspace",
                    "format": "code",
                    "language": "python",
                    "content": "from config import settings\n\nprint(settings)",
                    "imports": ["config"],
                    "symbols": ["main"],
                },
                {
                    "id": "canvas-2",
                    "title": "config.py",
                    "path": "src/config.py",
                    "role": "config",
                    "project_id": "demo-app",
                    "workspace_id": "demo-workspace",
                    "format": "code",
                    "language": "python",
                    "content": "settings = {'debug': True}",
                    "exports": ["settings"],
                },
            ],
            canvas_active_document_id="canvas-1",
        )

        content = message["content"]
        self.assertIn("## Canvas Project Manifest", content)
        self.assertIn("## Canvas Relationship Map", content)
        self.assertIn('"project_name": "demo-app"', content)
        self.assertIn('"active_document_id": "canvas-1"', content)
        self.assertIn('"last_validation_status": "ok"', content)
        self.assertIn('"imports": [', content)
        self.assertIn("- Working mode: project", content)
        self.assertIn("- Path: src/app.py", content)
        self.assertIn("- Role: source", content)
        self.assertIn("## Other Canvas Documents", content)
        self.assertIn('"path": "src/config.py"', content)

    def test_openai_tool_specs_include_expand_canvas_document_with_canvas_documents(self):
        tools = get_openai_tool_specs(
            [
                "expand_canvas_document",
                "create_canvas_document",
                "rewrite_canvas_document",
            ],
            canvas_documents=[
                {
                    "id": "canvas-1",
                    "title": "app.py",
                    "path": "src/app.py",
                    "format": "code",
                    "content": "print('hello')",
                }
            ],
        )

        tool_names = [entry["function"]["name"] for entry in tools]
        self.assertEqual(tool_names, ["expand_canvas_document", "create_canvas_document", "rewrite_canvas_document"])

    def test_openai_tool_specs_hide_canvas_edit_tools_without_canvas_document(self):
        tools = get_openai_tool_specs(
            [
                "create_canvas_document",
                "rewrite_canvas_document",
                "replace_canvas_lines",
                "clear_canvas",
            ]
        )

        tool_names = [entry["function"]["name"] for entry in tools]
        self.assertEqual(tool_names, ["create_canvas_document"])

    def test_prepend_runtime_context_places_datetime_system_message_first(self):
        messages = prepend_runtime_context(
            [{"role": "user", "content": "Hello"}],
            user_preferences="",
            active_tool_names=[],
            scratchpad="Persistent note",
        )

        self.assertEqual(messages[0]["role"], "system")
        
        content = messages[0]["content"]
        self.assertIn("Current Date and Time", content)
        self.assertIn("Persistent note", content)
        self.assertNotIn("User Preferences", content)
        self.assertIn("Date: ", content)
        self.assertIn("Time: ", content)
        self.assertEqual(messages[1]["role"], "user")

    def test_runtime_system_message_includes_workspace_sandbox(self):
        message = build_runtime_system_message(
            active_tool_names=["create_file", "list_dir"],
            workspace_root="/tmp/workspace-root",
        )

        content = message["content"]
        self.assertIn("## Workspace Sandbox", content)
        self.assertIn("- Root: /tmp/workspace-root", content)
        self.assertIn("needs_confirmation", content)

    def test_runtime_system_message_includes_project_workflow(self):
        message = build_runtime_system_message(
            active_tool_names=["plan_project_workspace", "get_project_workflow_status"],
            project_workflow={
                "stage": "content",
                "project_name": "Demo App",
                "goal": "Build a small Python service",
                "target_type": "python-project",
                "files": [{"path": "demo-app/app.py", "role": "source", "status": "written"}],
            },
        )

        content = message["content"]
        self.assertIn("## Project Workflow", content)
        self.assertIn('"stage": "content"', content)
        self.assertIn('"project_name": "Demo App"', content)

    def test_settings_patch_allows_manual_scratchpad_updates(self):
        response = self.client.patch(
            "/api/settings",
            json={"scratchpad": "The user likes concise answers.\nThe user likes concise answers.\n"},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["scratchpad"], "The user likes concise answers.")
        self.assertFalse(payload["features"]["scratchpad_admin_editing"])

    def test_settings_get_reports_scratchpad_admin_feature_flag(self):
        with patch("config.SCRATCHPAD_ADMIN_EDITING_ENABLED", True):
            response = self.client.get("/api/settings")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.get_json()["features"]["scratchpad_admin_editing"])

    def test_append_to_scratchpad_deduplicates_and_persists(self):
        result, summary = append_to_scratchpad("The user is 22 years old.")
        self.assertEqual(summary, "Scratchpad updated")
        self.assertEqual(result["status"], "appended")

        duplicate_result, duplicate_summary = append_to_scratchpad("The   user is 22 years old.   ")
        self.assertEqual(duplicate_summary, "Scratchpad notes already exist")
        self.assertEqual(duplicate_result["status"], "skipped")

        settings = get_app_settings()
        self.assertEqual(settings["scratchpad"], "The user is 22 years old.")

    def test_chat_runtime_context_injects_saved_scratchpad(self):
        conversation_id = self._create_conversation()
        save_app_settings(
            {
                "user_preferences": "",
                "scratchpad": "The user prefers concise answers.",
                "max_steps": "2",
                "active_tools": "[]",
                "rag_auto_inject": "false",
            }
        )

        fake_events = iter(
            [
                {"type": "answer_start"},
                {"type": "answer_delta", "text": "Done."},
                {"type": "tool_capture", "tool_results": []},
                {"type": "done"},
            ]
        )

        with patch("routes.chat.run_agent_stream", return_value=fake_events) as mocked_stream:
            response = self.client.post(
                "/chat",
                json={
                    "conversation_id": conversation_id,
                    "model": "deepseek-chat",
                    "user_content": "Hello",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )

        self.assertEqual(response.status_code, 200)
        first_call_messages = mocked_stream.call_args.args[0]
        full_content = first_call_messages[0]["content"]
        self.assertIn("The user prefers concise answers.", full_content)

    def test_chat_uses_saved_rag_sensitivity_and_context_size(self):
        conversation_id = self._create_conversation()
        save_app_settings(
            {
                "user_preferences": "",
                "scratchpad": "",
                "max_steps": "2",
                "active_tools": "[]",
                "rag_auto_inject": "true",
                "rag_sensitivity": "strict",
                "rag_context_size": "large",
            }
        )

        fake_events = iter(
            [
                {"type": "answer_start"},
                {"type": "answer_delta", "text": "Done."},
                {"type": "tool_capture", "tool_results": []},
                {"type": "done"},
            ]
        )

        with patch("routes.chat.build_rag_auto_context", return_value=None) as mocked_rag, patch(
            "routes.chat.run_agent_stream", return_value=fake_events
        ):
            response = self.client.post(
                "/chat",
                json={
                    "conversation_id": conversation_id,
                    "model": "deepseek-chat",
                    "user_content": "Hello",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(mocked_rag.call_args.args[0], "Hello")
        self.assertTrue(mocked_rag.call_args.args[1])
        self.assertEqual(mocked_rag.call_args.kwargs["threshold"], 0.55)
        self.assertEqual(mocked_rag.call_args.kwargs["top_k"], 8)

    def test_index_uses_external_app_script(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn("marked/marked.min.js", html)
        self.assertIn("dompurify/dist/purify.min.js", html)
        self.assertIn('id="app-bootstrap"', html)
        self.assertIn('href="/settings"', html)
        self.assertNotIn('id="scratchpad-list"', html)
        self.assertNotIn('id="settings-panel"', html)

    def test_settings_page_renders_dedicated_layout(self):
        response = self.client.get("/settings")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn("Separate page, grouped controls, less noise", html)
        self.assertIn('data-settings-tab="assistant"', html)
        self.assertIn('data-settings-tab="memory"', html)
        self.assertIn('data-settings-tab="tools"', html)
        self.assertIn('data-settings-tab="knowledge"', html)
        self.assertIn('src="/static/settings.js"', html)
        self.assertIn('id="kb-sync-btn"', html)

    def test_build_user_message_for_model_includes_stored_image_reference(self):
        content = build_user_message_for_model(
            "What does this mean?",
            {
                "image_id": "img_123",
                "image_name": "screen.png",
                "vision_summary": "A pricing table is visible.",
            },
        )

        self.assertIn("Stored image reference: image_id=img_123, file=screen.png", content)
        self.assertIn("Visual summary: A pricing table is visible.", content)

    def test_build_user_message_for_model_includes_multiple_attachments(self):
        content = build_user_message_for_model(
            "Compare these assets.",
            {
                "attachments": [
                    {
                        "kind": "image",
                        "image_id": "img_123",
                        "image_name": "screen-a.png",
                        "vision_summary": "A dashboard is visible.",
                    },
                    {
                        "kind": "image",
                        "image_id": "img_456",
                        "image_name": "screen-b.png",
                        "vision_summary": "A settings page is visible.",
                    },
                    {
                        "kind": "document",
                        "file_id": "file_123",
                        "file_name": "notes.txt",
                        "file_context_block": "[Uploaded document: notes.txt]\n\nProject notes",
                    },
                ]
            },
        )

        self.assertIn("Stored image reference: image_id=img_123, file=screen-a.png", content)
        self.assertIn("Stored image reference: image_id=img_456, file=screen-b.png", content)
        self.assertIn("[Uploaded document: notes.txt]", content)
        self.assertIn("Visual summary: A dashboard is visible.", content)
        self.assertIn("Visual summary: A settings page is visible.", content)

    def test_build_user_message_for_model_omits_document_context_already_in_canvas(self):
        content = build_user_message_for_model(
            "Use the document in canvas.",
            {
                "attachments": [
                    {
                        "kind": "document",
                        "file_id": "file_123",
                        "file_name": "notes.txt",
                        "file_context_block": "[Uploaded document: notes.txt]\n\nProject notes\nLine two",
                    }
                ]
            },
            canvas_documents=[
                {
                    "id": "doc-1",
                    "title": "notes.txt",
                    "content": "# notes.txt\n\nProject notes\nLine two",
                    "format": "markdown",
                    "language": "markdown",
                }
            ],
        )

        self.assertIn("Use the document in canvas.", content)
        self.assertNotIn("[Uploaded document: notes.txt]", content)
        self.assertNotIn("Project notes", content)

    def test_image_explain_tool_spec_requires_image_and_conversation_ids(self):
        spec = TOOL_SPEC_BY_NAME["image_explain"]

        self.assertEqual(spec["parameters"]["required"], ["image_id", "conversation_id", "question"])
        self.assertIn("Write this question in English", spec["parameters"]["properties"]["question"]["description"])
        self.assertIn("Always send the question in English", spec["prompt"]["guidance"])

    def test_clarification_tool_spec_supports_structured_questions(self):
        spec = TOOL_SPEC_BY_NAME["ask_clarifying_question"]

        self.assertEqual(spec["parameters"]["required"], ["questions"])
        self.assertEqual(spec["parameters"]["properties"]["questions"]["minItems"], 1)
        self.assertIn("explicitly asks you to ask questions first", spec["description"])
        self.assertIn("only tool call", spec["prompt"]["guidance"])

    def test_clarification_tool_validator_accepts_stringified_question_objects(self):
        from agent import _validate_tool_arguments

        tool_args = {
            "questions": [
                '{"id":"scope","label":"Which scope?","input_type":"single_select","options":[{"label":"Only this repo","value":"repo"}]}',
                '{"id":"notes","label":"Anything else?","input_type":"text"}',
            ]
        }

        self.assertIsNone(_validate_tool_arguments("ask_clarifying_question", tool_args))
        self.assertIsInstance(tool_args["questions"][0], dict)
        self.assertEqual(tool_args["questions"][0]["id"], "scope")

    def test_clarification_tool_validator_accepts_plain_string_questions(self):
        from agent import _validate_tool_arguments

        tool_args = {
            "questions": [
                "Which operating system should I target?",
                "Any hard constraints?",
            ]
        }

        self.assertIsNone(_validate_tool_arguments("ask_clarifying_question", tool_args))
        self.assertEqual(tool_args["questions"][0]["label"], "Which operating system should I target?")
        self.assertEqual(tool_args["questions"][0]["input_type"], "text")

    def test_execute_clarification_tool_normalizes_question_aliases(self):
        result, summary = _execute_tool(
            "ask_clarifying_question",
            {
                "questions": [
                    {
                        "key": "software",
                        "question": "Hangi sanallaştırma yazılımını kullanacaksın?",
                        "type": "single",
                    }
                ]
            },
        )

        self.assertEqual(summary, "Awaiting user clarification")
        self.assertEqual(result["status"], "needs_user_input")
        self.assertEqual(result["clarification"]["questions"][0]["id"], "software")
        self.assertEqual(result["clarification"]["questions"][0]["label"], "Hangi sanallaştırma yazılımını kullanacaksın?")
        self.assertEqual(result["clarification"]["questions"][0]["input_type"], "text")

    def test_load_proxies_uses_cached_file_contents(self):
        proxies_path = Path(self.temp_dir.name) / "proxies.txt"
        proxies_path.write_text("http://127.0.0.1:8080\n", encoding="utf-8")
        web_tools._proxy_cache = None
        web_tools._proxy_cache_mtime = None

        with patch("web_tools.PROXIES_PATH", str(proxies_path)):
            first = load_proxies()
            self.assertEqual(first, ["http://127.0.0.1:8080"])

            mocked_open = Mock(side_effect=AssertionError("Proxy file should not be reopened when unchanged"))
            with patch("builtins.open", mocked_open):
                second = load_proxies()

        self.assertEqual(second, first)

    def test_image_explain_returns_reupload_instruction_when_asset_is_missing(self):
        result, summary = _execute_tool(
            "image_explain",
            {
                "image_id": "missing-image",
                "conversation_id": 99,
                "question": "What does the chart show?",
            },
        )

        self.assertEqual(summary, "Stored image not found")
        self.assertEqual(result["status"], "missing_image")
        self.assertIn("re-upload", result["error"])

    def test_canvas_tools_create_and_edit_document_in_runtime_state(self):
        runtime_state = {}

        created, create_summary = _execute_tool(
            "create_canvas_document",
            {
                "title": "Release Plan",
                "content": "# Release Plan\n\n- Draft\n- Review",
            },
            runtime_state=runtime_state,
        )
        self.assertEqual(create_summary, "Canvas created: Release Plan")
        self.assertEqual(created["action"], "created")
        document_id = created["document_id"]

        inserted, insert_summary = _execute_tool(
            "insert_canvas_lines",
            {
                "document_id": document_id,
                "after_line": 2,
                "lines": ["", "## Notes", "- Ship after QA"],
            },
            runtime_state=runtime_state,
        )
        self.assertEqual(insert_summary, "Canvas lines inserted in Release Plan")
        self.assertIn("## Notes", inserted["content"])

        deleted, delete_summary = _execute_tool(
            "delete_canvas_lines",
            {
                "document_id": document_id,
                "start_line": 2,
                "end_line": 2,
            },
            runtime_state=runtime_state,
        )
        self.assertEqual(delete_summary, "Canvas lines deleted in Release Plan")
        self.assertNotIn("", deleted["content"].splitlines()[:1])

    def test_canvas_tools_support_code_format(self):
        runtime_state = {}

        created, _ = _execute_tool(
            "create_canvas_document",
            {
                "title": "main.py",
                "content": "print('hello')",
                "format": "code",
                "language": "python",
            },
            runtime_state=runtime_state,
        )

        self.assertEqual(created["format"], "code")
        self.assertEqual(created["language"], "python")

        updated, _ = _execute_tool(
            "rewrite_canvas_document",
            {
                "document_id": created["document_id"],
                "content": "console.log('hello');",
                "format": "code",
                "language": "javascript",
            },
            runtime_state=runtime_state,
        )

        self.assertEqual(updated["format"], "code")
        self.assertEqual(updated["language"], "javascript")

    def test_canvas_tools_delete_and_clear_documents_in_runtime_state(self):
        runtime_state = {}

        created_first, _ = _execute_tool(
            "create_canvas_document",
            {
                "title": "Draft One",
                "content": "# Draft One",
            },
            runtime_state=runtime_state,
        )
        created_second, _ = _execute_tool(
            "create_canvas_document",
            {
                "title": "Draft Two",
                "content": "# Draft Two",
            },
            runtime_state=runtime_state,
        )

        deleted, delete_summary = _execute_tool(
            "delete_canvas_document",
            {"document_id": created_first["document_id"]},
            runtime_state=runtime_state,
        )

        self.assertEqual(delete_summary, "Canvas deleted: Draft One")
        self.assertEqual(deleted["action"], "deleted")
        self.assertEqual(deleted["remaining_count"], 1)

        cleared, clear_summary = _execute_tool(
            "clear_canvas",
            {},
            runtime_state=runtime_state,
        )

        self.assertEqual(clear_summary, "Canvas cleared (1 documents removed)")
        self.assertEqual(cleared["action"], "cleared")
        self.assertEqual(cleared["cleared_count"], 1)
        self.assertEqual(runtime_state["canvas"]["documents"], [])
        self.assertIsNone(runtime_state["canvas"]["active_document_id"])
        self.assertEqual(created_second["title"], "Draft Two")

    def test_workspace_tools_create_read_list_search_update_and_validate(self):
        workspace_root = os.path.join(self.temp_dir.name, "workspace-tools")
        runtime_state = {"workspace": create_workspace_runtime_state(root_path=workspace_root)}

        created_dir, _ = _execute_tool("create_directory", {"path": "demo/tests"}, runtime_state=runtime_state)
        self.assertEqual(created_dir["path"], "demo/tests")

        created_file, _ = _execute_tool(
            "create_file",
            {"path": "demo/app.py", "content": "def main():\n    return 'ok'\n"},
            runtime_state=runtime_state,
        )
        self.assertEqual(created_file["action"], "file_created")

        updated_file, _ = _execute_tool(
            "update_file",
            {"path": "demo/app.py", "content": "def main():\n    return 'updated'\n"},
            runtime_state=runtime_state,
        )
        self.assertEqual(updated_file["action"], "file_updated")

        read_result, _ = _execute_tool(
            "read_file",
            {"path": "demo/app.py", "start_line": 1, "end_line": 2},
            runtime_state=runtime_state,
        )
        self.assertIn("1: def main():", read_result["content"])
        self.assertIn("2:     return 'updated'", read_result["content"])

        list_result, _ = _execute_tool("list_dir", {"path": "demo"}, runtime_state=runtime_state)
        self.assertEqual([entry["path"] for entry in list_result["entries"]], ["demo/tests", "demo/app.py"])

        search_result, _ = _execute_tool(
            "search_files",
            {"query": "updated", "path_prefix": "demo", "search_content": True},
            runtime_state=runtime_state,
        )
        self.assertEqual(search_result["matches"][0]["path"], "demo/app.py")

        history_result, _ = _execute_tool(
            "get_workspace_file_history",
            {"path": "demo/app.py"},
            runtime_state=runtime_state,
        )
        self.assertEqual(history_result["undo_count"], 2)
        self.assertEqual(history_result["redo_count"], 0)

        validate_result, _ = _execute_tool(
            "validate_project_workspace",
            {"path": "demo"},
            runtime_state=runtime_state,
        )
        self.assertEqual(validate_result["status"], "ok")

    def test_project_workflow_tools_track_stage_transitions(self):
        workspace_root = os.path.join(self.temp_dir.name, "workflow-tools")
        runtime_state = {
            "workspace": create_workspace_runtime_state(root_path=workspace_root),
        }

        planned, summary = _execute_tool(
            "plan_project_workspace",
            {
                "goal": "Build a small Python demo",
                "project_name": "Demo App",
                "target_type": "python-project",
            },
            runtime_state=runtime_state,
        )
        self.assertEqual(summary, "Project planned: Demo App")
        self.assertEqual(planned["project_workflow"]["stage"], "plan")
        self.assertEqual(planned["project_workflow"]["target_type"], "python-project")

        scaffolded, _ = _execute_tool(
            "create_project_scaffold",
            {"project_name": "Demo App", "target_type": "python-project"},
            runtime_state=runtime_state,
        )
        self.assertEqual(scaffolded["status"], "ok")

        status_after_scaffold, _ = _execute_tool(
            "get_project_workflow_status",
            {},
            runtime_state=runtime_state,
        )
        self.assertEqual(status_after_scaffold["project_workflow"]["stage"], "skeleton")

        updated, _ = _execute_tool(
            "bulk_update_workspace_files",
            {
                "files": [
                    {"path": "demo-app/app.py", "content": "def main():\n    return 'ok'\n"},
                ],
                "confirm": True,
            },
            runtime_state=runtime_state,
        )
        self.assertEqual(updated["status"], "ok")

        status_after_write, _ = _execute_tool(
            "get_project_workflow_status",
            {},
            runtime_state=runtime_state,
        )
        self.assertEqual(status_after_write["project_workflow"]["stage"], "content")

        validated, _ = _execute_tool(
            "validate_project_workspace",
            {"path": "demo-app"},
            runtime_state=runtime_state,
        )
        self.assertEqual(validated["status"], "ok")

        final_status, _ = _execute_tool(
            "get_project_workflow_status",
            {},
            runtime_state=runtime_state,
        )
        self.assertEqual(final_status["project_workflow"]["stage"], "validated")
        self.assertEqual(final_status["project_workflow"]["validation"]["status"], "ok")

    def test_validate_project_workspace_reports_python_project_rule_warnings(self):
        workspace_root = os.path.join(self.temp_dir.name, "validation-rules")
        runtime_state = {"workspace": create_workspace_runtime_state(root_path=workspace_root)}

        written, _ = _execute_tool(
            "write_project_tree",
            {
                "directories": ["demo/src/demo_pkg", "demo/tests"],
                "files": [
                    {"path": "demo/README.md", "content": "# Demo\n"},
                    {"path": "demo/requirements.txt", "content": "requests\n"},
                    {"path": "demo/pyproject.toml", "content": "[project]\nname = 'demo'\nversion = '0.1.0'\n"},
                    {"path": "demo/config.py", "content": "\n"},
                    {"path": "demo/src/demo_pkg/__init__.py", "content": ""},
                    {"path": "demo/src/demo_pkg/main.py", "content": "from .missing import run\n"},
                ],
            },
            runtime_state=runtime_state,
        )
        self.assertEqual(written["status"], "ok")

        validation, _ = _execute_tool(
            "validate_project_workspace",
            {"path": "demo"},
            runtime_state=runtime_state,
        )
        self.assertEqual(validation["status"], "ok")
        self.assertTrue(validation["summary"]["looks_like_python_project"])
        self.assertGreaterEqual(validation["summary"]["warning_count"], 4)
        self.assertIn("Missing expected file: app.py", validation["warnings"])
        self.assertIn("Missing tests directory or test files.", validation["warnings"])
        self.assertIn("config.py is empty.", validation["warnings"])
        self.assertIn("No obvious Python entry point found. Add app.py, main.py, __main__.py, or declare scripts in pyproject.toml.", validation["warnings"])
        self.assertIn("Relative import target is missing in src/demo_pkg/main.py: .missing", validation["warnings"])

    def test_project_scaffold_and_batch_tools_require_confirmation_for_overwrites(self):
        workspace_root = os.path.join(self.temp_dir.name, "workspace-project")
        runtime_state = {"workspace": create_workspace_runtime_state(root_path=workspace_root)}

        preview, _ = _execute_tool(
            "create_project_scaffold",
            {"project_name": "Demo App"},
            runtime_state=runtime_state,
        )
        self.assertEqual(preview["action"], "project_scaffold_created")
        self.assertEqual(preview["project_root"], "demo-app")

        overwrite_preview, _ = _execute_tool(
            "write_project_tree",
            {
                "files": [
                    {"path": "demo-app/app.py", "content": "print('rewrite')\n"},
                    {"path": "demo-app/config.py", "content": "SETTINGS = {'debug': True}\n"},
                ]
            },
            runtime_state=runtime_state,
        )
        self.assertEqual(overwrite_preview["status"], "needs_confirmation")
        self.assertIn("demo-app/app.py", overwrite_preview["overwrites"])
        self.assertIn("--- a/demo-app/app.py", overwrite_preview["diffs"][0]["diff"])

        applied, _ = _execute_tool(
            "write_project_tree",
            {
                "files": [
                    {"path": "demo-app/app.py", "content": "print('rewrite')\n"},
                    {"path": "demo-app/config.py", "content": "SETTINGS = {'debug': True}\n"},
                ],
                "confirm": True,
            },
            runtime_state=runtime_state,
        )
        self.assertEqual(applied["status"], "ok")
        self.assertTrue(applied["revision_ids"])

        bulk_preview, _ = _execute_tool(
            "bulk_update_workspace_files",
            {
                "files": [
                    {"path": "demo-app/README.md", "content": "# Demo App\n"},
                    {"path": "demo-app/tests/test_app.py", "content": "def test_placeholder():\n    assert True\n"},
                ]
            },
            runtime_state=runtime_state,
        )
        self.assertEqual(bulk_preview["status"], "needs_confirmation")
        self.assertEqual(len(bulk_preview["diffs"]), 2)

        bulk_applied, _ = _execute_tool(
            "bulk_update_workspace_files",
            {
                "files": [
                    {"path": "demo-app/README.md", "content": "# Demo App\n"},
                    {"path": "demo-app/tests/test_app.py", "content": "def test_placeholder():\n    assert True\n"},
                ],
                "confirm": True,
            },
            runtime_state=runtime_state,
        )
        self.assertEqual(bulk_applied["status"], "ok")

    def test_preview_and_undo_redo_workspace_file_changes(self):
        workspace_root = os.path.join(self.temp_dir.name, "workspace-history")
        runtime_state = {"workspace": create_workspace_runtime_state(root_path=workspace_root)}

        created, _ = _execute_tool(
            "create_file",
            {"path": "demo/app.py", "content": "print('v1')\n"},
            runtime_state=runtime_state,
        )
        self.assertEqual(created["action"], "file_created")

        preview, summary = _execute_tool(
            "preview_workspace_changes",
            {"files": [{"path": "demo/app.py", "content": "print('v2')\n"}]},
            runtime_state=runtime_state,
        )
        self.assertEqual(summary, "Workspace diff preview: 1 files")
        self.assertEqual(preview["diffs"][0]["change_type"], "modified")
        self.assertIn("-print('v1')", preview["diffs"][0]["diff"])
        self.assertIn("+print('v2')", preview["diffs"][0]["diff"])

        updated, _ = _execute_tool(
            "update_file",
            {"path": "demo/app.py", "content": "print('v2')\n"},
            runtime_state=runtime_state,
        )
        self.assertEqual(updated["action"], "file_updated")

        undo_result, undo_summary = _execute_tool(
            "undo_workspace_file_change",
            {"path": "demo/app.py"},
            runtime_state=runtime_state,
        )
        self.assertEqual(undo_summary, "Workspace undo: demo/app.py")
        self.assertEqual(undo_result["action"], "workspace_file_undone")
        self.assertEqual(undo_result["content"], "print('v1')\n")
        self.assertEqual(undo_result["redo_count"], 1)

        redo_result, redo_summary = _execute_tool(
            "redo_workspace_file_change",
            {"path": "demo/app.py"},
            runtime_state=runtime_state,
        )
        self.assertEqual(redo_summary, "Workspace redo: demo/app.py")
        self.assertEqual(redo_result["action"], "workspace_file_redone")
        self.assertEqual(redo_result["content"], "print('v2')\n")

        history_result, _ = _execute_tool(
            "get_workspace_file_history",
            {"path": "demo/app.py"},
            runtime_state=runtime_state,
        )
        self.assertEqual(history_result["undo_count"], 2)
        self.assertEqual(history_result["redo_count"], 0)

    def test_hidden_workspace_history_is_not_listed_or_searched(self):
        workspace_root = os.path.join(self.temp_dir.name, "workspace-hidden-history")
        runtime_state = {"workspace": create_workspace_runtime_state(root_path=workspace_root)}

        _execute_tool(
            "create_file",
            {"path": "demo/app.py", "content": "print('v1')\n"},
            runtime_state=runtime_state,
        )
        _execute_tool(
            "update_file",
            {"path": "demo/app.py", "content": "print('v2')\n"},
            runtime_state=runtime_state,
        )

        listed, _ = _execute_tool("list_dir", {}, runtime_state=runtime_state)
        self.assertEqual([entry["path"] for entry in listed["entries"]], ["demo"])

        searched, _ = _execute_tool(
            "search_files",
            {"query": ".workspace-history", "search_content": True},
            runtime_state=runtime_state,
        )
        self.assertEqual(searched["matches"], [])

    def test_expand_canvas_document_tool_returns_line_numbered_context(self):
        runtime_state = {"canvas": create_canvas_runtime_state([
            {
                "id": "canvas-1",
                "title": "app.py",
                "path": "src/app.py",
                "role": "source",
                "format": "code",
                "language": "python",
                "content": "import os\n\nprint('hello')",
                "imports": ["os"],
                "symbols": ["main"],
            },
            {
                "id": "canvas-2",
                "title": "config.py",
                "path": "src/config.py",
                "role": "config",
                "format": "code",
                "language": "python",
                "content": "DEBUG = True",
                "exports": ["DEBUG"],
            },
        ], active_document_id="canvas-2")}

        expanded, summary = _execute_tool(
            "expand_canvas_document",
            {"document_path": "src/app.py"},
            runtime_state=runtime_state,
        )

        self.assertEqual(summary, "Canvas expanded: src/app.py")
        self.assertEqual(expanded["action"], "expanded")
        self.assertEqual(expanded["document_path"], "src/app.py")
        self.assertEqual(expanded["visible_lines"][0], "1: import os")
        self.assertEqual(expanded["visible_lines"][2], "3: print('hello')")
        self.assertEqual(expanded["manifest_excerpt"]["active_file"], "src/config.py")
        self.assertIn("os", expanded["relationship_map"]["imports"])

    def test_chat_image_upload_persists_image_asset_and_metadata(self):
        conversation_id = self._create_conversation()
        fake_events = iter(
            [
                {"type": "answer_start"},
                {"type": "answer_delta", "text": "Done."},
                {"type": "tool_capture", "tool_results": []},
                {"type": "done"},
            ]
        )

        with patch("db.IMAGE_STORAGE_DIR", self.image_storage_dir), patch(
            "routes.chat.run_image_vision_analysis",
            return_value={
                "ocr_text": "hello",
                "vision_summary": "A login screen is shown.",
                "assistant_guidance": "Use the labels and values when answering.",
                "key_points": ["Email field", "Password field"],
            },
        ), patch("routes.chat.run_agent_stream", return_value=fake_events):
            response = self.client.post(
                "/chat",
                data={
                    "messages": json.dumps([{"role": "user", "content": "Bu görsel ne anlatıyor?"}]),
                    "model": "deepseek-chat",
                    "conversation_id": str(conversation_id),
                    "user_content": "Bu görsel ne anlatıyor?",
                    "image": (io.BytesIO(b"fake image bytes"), "screen.png", "image/png"),
                },
            )

        self.assertEqual(response.status_code, 200)
        response.get_data(as_text=True)

        conversation_response = self.client.get(f"/api/conversations/{conversation_id}")
        self.assertEqual(conversation_response.status_code, 200)
        messages = conversation_response.get_json()["messages"]
        user_messages = [message for message in messages if message["role"] == "user"]
        self.assertEqual(len(user_messages), 1)

        metadata = user_messages[0]["metadata"]
        self.assertIn("image_id", metadata)
        self.assertEqual(metadata["image_name"], "screen.png")
        self.assertEqual(metadata["vision_summary"], "A login screen is shown.")

        asset = get_image_asset(metadata["image_id"], conversation_id=conversation_id)
        self.assertIsNotNone(asset)
        self.assertEqual(asset["filename"], "screen.png")
        self.assertEqual(asset["message_id"], user_messages[0]["id"])
        self.assertTrue(Path(asset["storage_path"]).is_file())

    def test_chat_mixed_multi_attachment_upload_persists_assets_and_canvas_documents(self):
        conversation_id = self._create_conversation()
        captured = {}

        def fake_run_agent_stream(*args, **kwargs):
            captured["initial_canvas_documents"] = kwargs.get("initial_canvas_documents") or []
            return iter([{"type": "done"}])

        with patch("db.IMAGE_STORAGE_DIR", self.image_storage_dir), patch(
            "routes.chat.run_image_vision_analysis",
            side_effect=[
                {
                    "ocr_text": "alpha",
                    "vision_summary": "First screen.",
                    "assistant_guidance": "Use alpha.",
                    "key_points": ["A"],
                },
                {
                    "ocr_text": "beta",
                    "vision_summary": "Second screen.",
                    "assistant_guidance": "Use beta.",
                    "key_points": ["B"],
                },
            ],
        ), patch("routes.chat.run_agent_stream", side_effect=fake_run_agent_stream):
            response = self.client.post(
                "/chat",
                data=MultiDict(
                    [
                        ("messages", json.dumps([{"role": "user", "content": "Review everything"}])),
                        ("model", "deepseek-chat"),
                        ("conversation_id", str(conversation_id)),
                        ("user_content", "Review everything"),
                        ("image", (io.BytesIO(b"image-a"), "screen-a.png", "image/png")),
                        ("image", (io.BytesIO(b"image-b"), "screen-b.png", "image/png")),
                        ("document", (io.BytesIO(b"Doc A"), "notes-a.txt", "text/plain")),
                        ("document", (io.BytesIO(b"Doc B"), "notes-b.txt", "text/plain")),
                    ]
                ),
                content_type="multipart/form-data",
            )

        self.assertEqual(response.status_code, 200)
        events = [json.loads(line) for line in response.get_data(as_text=True).strip().splitlines()]
        self.assertEqual(len([event for event in events if event["type"] == "vision_complete"]), 2)
        self.assertEqual(len([event for event in events if event["type"] == "document_processed"]), 2)
        canvas_event = next((event for event in events if event["type"] == "canvas_sync"), None)
        self.assertIsNotNone(canvas_event)
        self.assertEqual(len(canvas_event["documents"]), 2)
        self.assertEqual([doc["title"] for doc in canvas_event["documents"]], ["notes-a.txt", "notes-b.txt"])

        conversation_response = self.client.get(f"/api/conversations/{conversation_id}")
        self.assertEqual(conversation_response.status_code, 200)
        messages = conversation_response.get_json()["messages"]
        user_messages = [message for message in messages if message["role"] == "user"]
        self.assertEqual(len(user_messages), 1)

        metadata = user_messages[0]["metadata"]
        attachments = metadata.get("attachments") or []
        self.assertEqual(len(attachments), 4)
        self.assertEqual(len([entry for entry in attachments if entry["kind"] == "image"]), 2)
        self.assertEqual(len([entry for entry in attachments if entry["kind"] == "document"]), 2)

        for attachment in attachments:
            if attachment["kind"] == "image":
                asset = get_image_asset(attachment["image_id"], conversation_id=conversation_id)
                self.assertIsNotNone(asset)
                self.assertEqual(asset["message_id"], user_messages[0]["id"])
                continue
            asset = get_file_asset(attachment["file_id"], conversation_id=conversation_id)
            self.assertIsNotNone(asset)
            self.assertEqual(asset["message_id"], user_messages[0]["id"])

    def test_delete_conversation_removes_persisted_image_files(self):
        conversation_id = self._create_conversation()

        with patch("db.IMAGE_STORAGE_DIR", self.image_storage_dir):
            asset = create_image_asset(conversation_id, "screen.png", "image/png", b"raw bytes")
            self.assertTrue(Path(asset["storage_path"]).is_file())

            response = self.client.delete(f"/api/conversations/{conversation_id}")

        self.assertEqual(response.status_code, 204)
        self.assertFalse(Path(asset["storage_path"]).exists())

    def test_external_app_script_exists_and_contains_bootstrap_reader(self):
        script_path = Path(__file__).resolve().parent.parent / "static" / "app.js"
        self.assertTrue(script_path.exists())
        script_text = script_path.read_text(encoding="utf-8")
        self.assertIn('document.getElementById("app-bootstrap")', script_text)
        self.assertIn("globalThis.marked", script_text)
        self.assertIn("function renderMarkdown", script_text)
        self.assertIn("INPUT_BREAKDOWN_ORDER", script_text)
        self.assertIn("loadSidebar()", script_text)
        self.assertIn('const editBanner = document.getElementById("edit-banner")', script_text)
        self.assertIn('const editedMessageId = isEditing ? Number(editingEntry.id) : null;', script_text)
        self.assertIn('edited_message_id: editedMessageId', script_text)
        self.assertIn('formData.append("edited_message_id", String(editedMessageId));', script_text)
        self.assertIn("clearEditTarget();", script_text)
        self.assertIn("const fragment = document.createDocumentFragment();", script_text)
        self.assertIn("messagesEl.replaceChildren(fragment);", script_text)

    def test_external_settings_script_exists_and_contains_tabbed_settings_logic(self):
        script_path = Path(__file__).resolve().parent.parent / "static" / "settings.js"
        self.assertTrue(script_path.exists())
        script_text = script_path.read_text(encoding="utf-8")
        self.assertIn('document.getElementById("app-bootstrap")', script_text)
        self.assertIn("function activateTab", script_text)
        self.assertIn("void loadKnowledgeBaseDocuments();", script_text)
        self.assertIn("void refreshSettings();", script_text)
        self.assertIn("window.addEventListener(\"beforeunload\"", script_text)

    def test_reasoning_panel_uses_markdown_rendering(self):
        script_path = Path(__file__).resolve().parent.parent / "static" / "app.js"
        script_text = script_path.read_text(encoding="utf-8")
        self.assertIn('body.innerHTML = renderMarkdown(text);', script_text)

    def test_reasoning_css_includes_markdown_styles(self):
        style_path = Path(__file__).resolve().parent.parent / "static" / "style.css"
        style_text = style_path.read_text(encoding="utf-8")
        self.assertIn(".reasoning-body code", style_text)
        self.assertIn(".reasoning-body ul", style_text)

    def test_estimate_message_breakdown_splits_runtime_system_sections(self):
        message = build_runtime_system_message(
            active_tool_names=["search_web", "search_tool_memory", "append_scratchpad"],
            retrieved_context="Alpha knowledge block",
            tool_trace_context="search_web -> returned 3 results",
            tool_memory_context="Stored page snapshot",
            scratchpad="Remember the preferred deployment region.",
            canvas_documents=[
                {
                    "id": "doc-1",
                    "title": "spec.md",
                    "content": "line one\nline two",
                    "format": "markdown",
                    "language": "markdown",
                }
            ],
        )

        breakdown = _estimate_message_breakdown(message)

        self.assertGreater(breakdown["core_instructions"], 0)
        self.assertGreater(breakdown["canvas"], 0)
        self.assertGreater(breakdown["scratchpad"], 0)
        self.assertGreater(breakdown["tool_trace"], 0)
        self.assertGreater(breakdown["tool_memory"], 0)
        self.assertGreater(breakdown["rag_context"], 0)
        self.assertNotIn("tool_specs", breakdown)
        self.assertNotIn("Available Tools", message["content"])
        self.assertEqual(sum(breakdown.values()), estimate_text_tokens(message["content"]))

    def test_estimate_input_breakdown_counts_native_tool_schemas(self):
        message = build_runtime_system_message(active_tool_names=["search_web"])
        request_tools = get_openai_tool_specs(["search_web"])

        breakdown, _total_tokens, tool_schema_tokens = _estimate_input_breakdown(
            [message, {"role": "user", "content": "Find the release notes."}],
            request_tools=request_tools,
        )

        self.assertNotIn("Available Tools", message["content"])
        self.assertGreater(tool_schema_tokens, 0)
        self.assertEqual(breakdown["tool_specs"], tool_schema_tokens)

    def test_extract_message_usage_maps_legacy_system_prompt_breakdown(self):
        usage = extract_message_usage(
            {
                "usage": {
                    "estimated_input_tokens": 4,
                    "input_breakdown": {
                        "system_prompt": 4,
                    },
                }
            }
        )

        self.assertEqual(usage["input_breakdown"]["core_instructions"], 4)
        self.assertNotIn("system_prompt", usage["input_breakdown"])

    def test_extract_message_usage_adds_unknown_overhead_to_match_prompt_tokens(self):
        usage = extract_message_usage(
            {
                "usage": {
                    "prompt_tokens": 12,
                    "estimated_input_tokens": 6,
                    "input_breakdown": {
                        "system_prompt": 4,
                        "user_messages": 2,
                    },
                }
            }
        )

        self.assertEqual(usage["estimated_input_tokens"], 12)
        self.assertEqual(usage["input_breakdown"]["core_instructions"], 4)
        self.assertEqual(usage["input_breakdown"]["user_messages"], 2)
        self.assertEqual(usage["input_breakdown"]["unknown_provider_overhead"], 6)

    def test_frontend_restores_persistent_tool_trace_panel(self):
        script_path = Path(__file__).resolve().parent.parent / "static" / "app.js"
        script_text = script_path.read_text(encoding="utf-8")
        self.assertIn("function updateAssistantToolTrace(group, metadata)", script_text)
        self.assertIn("tool_trace: assistantToolTrace", script_text)

    def test_frontend_includes_clarification_ui_hooks(self):
        script_path = Path(__file__).resolve().parent.parent / "static" / "app.js"
        script_text = script_path.read_text(encoding="utf-8")
        self.assertIn("function appendClarificationPanel(group, metadata, options = {})", script_text)
        self.assertIn("pending_clarification: pendingClarification", script_text)
        self.assertIn('clarification_response', script_text)
        self.assertIn("function renderBubbleWithCursor(bubbleEl, text)", script_text)
        self.assertIn('bubbleEl.classList.add("streaming-text")', script_text)
        self.assertIn("function renderBubbleMarkdown(bubbleEl, text)", script_text)

        style_path = Path(__file__).resolve().parent.parent / "static" / "style.css"
        style_text = style_path.read_text(encoding="utf-8")
        self.assertIn(".clarification-card", style_text)
        self.assertIn(".clarification-form", style_text)
        self.assertIn(".bubble.streaming-text", style_text)

    def test_frontend_streaming_render_uses_typed_markdown_queue(self):
        script_path = Path(__file__).resolve().parent.parent / "static" / "app.js"
        script_text = script_path.read_text(encoding="utf-8")
        style_path = Path(__file__).resolve().parent.parent / "static" / "style.css"
        style_text = style_path.read_text(encoding="utf-8")
        html_path = Path(__file__).resolve().parent.parent / "templates" / "index.html"
        html_text = html_path.read_text(encoding="utf-8")

        self.assertIn("const STREAM_TYPING_INTERVAL_MS = 24;", script_text)
        self.assertIn("let visibleAnswer = \"\";", script_text)
        self.assertIn("visibleAnswer = fullAnswer.slice(0, visibleAnswer.length + stepSize);", script_text)
        self.assertIn("function confirmCanvasOpenForDocument(", script_text)
        self.assertIn("function openCanvasConfirmModal(options = {})", script_text)
        self.assertIn("function getConversationSignature(entries = history)", script_text)
        self.assertIn("function scheduleConversationRefreshAfterStream()", script_text)
        self.assertIn("async function refreshConversationFromServer()", script_text)
        self.assertIn(".stream-cursor", style_text)
        self.assertIn("@keyframes streamCursorBlink", style_text)
        self.assertIn('id="canvas-confirm-modal"', html_text)
        self.assertIn('id="canvas-confirm-open"', html_text)
        self.assertIn('id="canvas-delete-btn"', html_text)
        self.assertIn('id="canvas-clear-btn"', html_text)
        self.assertIn('id="canvas-edit-btn"', html_text)
        self.assertIn('id="canvas-save-btn"', html_text)
        self.assertIn('id="canvas-format-select"', html_text)
        self.assertIn('id="canvas-diff"', html_text)
        self.assertIn('id="canvas-role-filter"', html_text)
        self.assertIn('id="canvas-path-filter"', html_text)
        self.assertIn('id="canvas-tree"', html_text)
        self.assertIn('id="canvas-resize-handle"', html_text)
        self.assertIn('const canvasDeleteBtn = document.getElementById("canvas-delete-btn")', script_text)
        self.assertIn('const canvasClearBtn = document.getElementById("canvas-clear-btn")', script_text)
        self.assertIn('const canvasEditBtn = document.getElementById("canvas-edit-btn")', script_text)
        self.assertIn('const canvasSaveBtn = document.getElementById("canvas-save-btn")', script_text)
        self.assertIn('const canvasRoleFilter = document.getElementById("canvas-role-filter")', script_text)
        self.assertIn('const canvasPathFilter = document.getElementById("canvas-path-filter")', script_text)
        self.assertIn('function getCanvasVisibleDocuments(documents)', script_text)
        self.assertIn('function buildCanvasTreeNodes(documents)', script_text)
        self.assertIn('function renderCanvasTree(documents, activeDocument)', script_text)
        self.assertIn('function renderHighlightedCodeBlock(codeText, rawLang = null)', script_text)
        self.assertIn('async function saveCanvasEdits()', script_text)
        self.assertIn("async function deleteCanvasDocuments(", script_text)
        self.assertIn('.canvas-workspace-shell', style_text)
        self.assertIn('.canvas-tree-panel', style_text)
        self.assertIn('.canvas-tree-file.active', style_text)

    def test_settings_ui_exposes_fetch_threshold_input(self):
        html = self.client.get("/settings").get_data(as_text=True)
        self.assertIn("Tool step budget", html)
        self.assertIn("Tool step limit (1-10)", html)
        self.assertIn('value="append_scratchpad"', html)
        self.assertIn('value="ask_clarifying_question"', html)
        self.assertIn('id="scratchpad-list"', html)
        self.assertIn('id="scratchpad-add-btn"', html)
        self.assertIn('id="summary-mode-select"', html)
        self.assertIn('id="summary-trigger-input"', html)
        self.assertIn('id="fetch-threshold-input"', html)
        self.assertIn('id="fetch-aggressiveness-input"', html)
        self.assertIn('id="rag-sensitivity-select"', html)
        self.assertIn('id="rag-context-size-select"', html)
        self.assertIn('id="rag-sensitivity-hint"', html)

        script_path = Path(__file__).resolve().parent.parent / "static" / "app.js"
        script_text = script_path.read_text(encoding="utf-8")
        self.assertIn("const RAG_SENSITIVITY_HINTS = {", script_text)

    def test_run_agent_stream_executes_append_scratchpad_tool(self):
        responses = [
            iter(
                [
                    self._tool_call_chunk("append_scratchpad", {"notes": ["The user is 22 years old."]}),
                    self._stream_chunk(usage=SimpleNamespace(prompt_tokens=3, completion_tokens=3, total_tokens=6)),
                ]
            ),
            iter(
                [
                    self._stream_chunk(content="Saved."),
                    self._stream_chunk(usage=SimpleNamespace(prompt_tokens=2, completion_tokens=2, total_tokens=4)),
                ]
            ),
        ]

        with patch("agent.client.chat.completions.create", side_effect=responses), patch(
            "agent.append_to_scratchpad",
            return_value=({"status": "appended", "scratchpad": "The user is 22 years old."}, "Scratchpad updated"),
        ) as mocked_append:
            events = list(run_agent_stream([{"role": "user", "content": "Remember this"}], "deepseek-chat", 2, ["append_scratchpad"]))

        self.assertTrue(mocked_append.called)
        tool_result_event = next(event for event in events if event["type"] == "tool_result")
        self.assertEqual(tool_result_event["tool"], "append_scratchpad")
        self.assertEqual(tool_result_event["summary"], "Scratchpad updated")

    def test_run_agent_stream_emits_clarification_request_and_stops(self):
        responses = [
            iter(
                [
                    self._tool_call_chunk(
                        "ask_clarifying_question",
                        {
                            "intro": "Before I answer, I need two details.",
                            "questions": [
                                {
                                    "id": "scope",
                                    "label": "Which scope?",
                                    "input_type": "single_select",
                                    "options": [
                                        {"label": "Only this repo", "value": "repo"},
                                        {"label": "General tool", "value": "general"},
                                    ],
                                },
                                {"id": "notes", "label": "Anything else?", "input_type": "text", "required": False},
                            ],
                        },
                    ),
                    self._stream_chunk(usage=SimpleNamespace(prompt_tokens=3, completion_tokens=4, total_tokens=7)),
                ]
            ),
        ]

        with patch("agent.client.chat.completions.create", side_effect=responses):
            events = list(
                run_agent_stream(
                    [{"role": "user", "content": "Build this for me"}],
                    "deepseek-chat",
                    2,
                    ["ask_clarifying_question"],
                )
            )

        clarification_event = next(event for event in events if event["type"] == "clarification_request")
        self.assertIn("Before I answer", clarification_event["text"])
        self.assertEqual(clarification_event["clarification"]["questions"][0]["id"], "scope")
        self.assertFalse(any(event["type"] == "answer_delta" for event in events))
        self.assertEqual(events[-1]["type"], "done")

    def test_active_tool_normalization_filters_invalid_entries(self):
        normalized = normalize_active_tool_names(
            [
                "fetch_url",
                "search_web",
                "fetch_url",
                "invalid_tool",
                123,
            ]
        )
        self.assertEqual(normalized, ["fetch_url", "search_web"])

    def test_multiple_app_instances_use_separate_databases(self):
        second_dir = tempfile.TemporaryDirectory()
        self.addCleanup(second_dir.cleanup)
        second_db_path = f"{second_dir.name}/second.db"
        second_app = create_app(database_path=second_db_path)
        second_app.config.update(TESTING=True)
        second_client = second_app.test_client()

        first_response = self.client.post(
            "/api/conversations",
            json={"title": "First App", "model": "deepseek-chat"},
        )
        second_response = second_client.post(
            "/api/conversations",
            json={"title": "Second App", "model": "deepseek-chat"},
        )

        self.assertEqual(first_response.status_code, 201)
        self.assertEqual(second_response.status_code, 201)

        first_id = first_response.get_json()["id"]
        second_id = second_response.get_json()["id"]

        first_get = self.client.get(f"/api/conversations/{first_id}")
        second_get = second_client.get(f"/api/conversations/{second_id}")
        self.assertEqual(first_get.status_code, 200)
        self.assertEqual(second_get.status_code, 200)
        self.assertEqual(first_get.get_json()["conversation"]["title"], "First App")
        self.assertEqual(second_get.get_json()["conversation"]["title"], "Second App")

        with self.app.app_context():
            first_count = get_db().execute("SELECT COUNT(*) AS count FROM conversations").fetchone()["count"]
        with second_app.app_context():
            second_count = get_db().execute("SELECT COUNT(*) AS count FROM conversations").fetchone()["count"]

        self.assertEqual(first_count, 1)
        self.assertEqual(second_count, 1)

    def test_conversation_crud_flow(self):
        conversation_id = self._create_conversation()

        response = self.client.get("/api/conversations")
        self.assertEqual(response.status_code, 200)
        rows = response.get_json()
        self.assertTrue(any(row["id"] == conversation_id for row in rows))

        response = self.client.patch(
            f"/api/conversations/{conversation_id}",
            json={"title": "Updated Title"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["title"], "Updated Title")

        response = self.client.get(f"/api/conversations/{conversation_id}")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["conversation"]["id"], conversation_id)
        self.assertEqual(payload["messages"], [])

        response = self.client.delete(f"/api/conversations/{conversation_id}")
        self.assertEqual(response.status_code, 204)

        response = self.client.get(f"/api/conversations/{conversation_id}")
        self.assertEqual(response.status_code, 404)

    def test_update_conversation_title_rejects_blank_values(self):
        conversation_id = self._create_conversation()

        response = self.client.patch(
            f"/api/conversations/{conversation_id}",
            json={"title": "   "},
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["error"], "Title required.")

    def test_rag_endpoints_safe_defaults(self):
        response = self.client.get("/api/rag/documents")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), [])

        response = self.client.post("/api/rag/ingest", json={"text": "ignored"})
        self.assertEqual(response.status_code, 410)
        self.assertIn("disabled", response.get_json()["error"].lower())

    def test_fix_text_endpoint(self):
        fake_result = {
            "content": "Improved text",
            "reasoning_content": "",
            "usage": None,
            "tool_results": [],
            "errors": [],
        }
        with patch("routes.chat.collect_agent_response", return_value=fake_result) as mocked_collect:
            response = self.client.post(
                "/api/fix-text",
                json={"text": "improved text", "model": "deepseek-chat"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["text"], "Improved text")
        self.assertTrue(mocked_collect.called)

    def test_collect_agent_response_keeps_reasoning_separate_from_content(self):
        fake_events = iter(
            [
                {"type": "reasoning_start"},
                {"type": "reasoning_delta", "text": "Internal chain"},
                {"type": "done"},
            ]
        )

        with patch("agent.run_agent_stream", return_value=fake_events):
            result = collect_agent_response([{"role": "user", "content": "Test"}], "deepseek-reasoner", 1, [])

        self.assertEqual(result["content"], "")
        self.assertEqual(result["reasoning_content"], "Internal chain")

    def test_fix_text_reasoner_requires_final_content(self):
        fake_result = {
            "content": "",
            "reasoning_content": "Improved via reasoning",
            "usage": None,
            "tool_results": [],
            "errors": [],
        }
        with patch("routes.chat.collect_agent_response", return_value=fake_result):
            response = self.client.post(
                "/api/fix-text",
                json={"text": "improved text", "model": "deepseek-reasoner"},
            )

        self.assertEqual(response.status_code, 502)
        self.assertEqual(response.get_json()["error"], "No text returned.")

    def test_chat_stream_persists_messages(self):
        conversation_id = self._create_conversation()
        save_app_settings(
            {
                "user_preferences": "",
                "max_steps": "2",
                "active_tools": "[]",
                "rag_auto_inject": "false",
            }
        )

        fake_events = iter(
            [
                {"type": "step_started", "step": 1, "max_steps": 2},
                {"type": "step_update", "step": 1, "tool": "search_web", "preview": "hello", "call_id": "c1"},
                {"type": "tool_result", "step": 1, "tool": "search_web", "summary": "1 web result found", "call_id": "c1"},
                {"type": "reasoning_start"},
                {"type": "reasoning_delta", "text": "Analyzing request"},
                {"type": "answer_start"},
                {"type": "answer_delta", "text": "Hello "},
                {"type": "answer_delta", "text": "world"},
                {
                    "type": "usage",
                    "prompt_tokens": 11,
                    "completion_tokens": 7,
                    "total_tokens": 18,
                    "estimated_input_tokens": 14,
                    "input_breakdown": {
                        "core_instructions": 4,
                        "user_messages": 6,
                        "assistant_history": 0,
                        "tool_results": 0,
                        "rag_context": 3,
                        "final_instruction": 1,
                    },
                    "cost": 0.0,
                    "currency": "USD",
                    "model": "deepseek-chat",
                },
                {"type": "tool_capture", "tool_results": []},
                {"type": "done"},
            ]
        )

        with patch("routes.chat.run_agent_stream", return_value=fake_events):
            response = self.client.post(
                "/chat",
                json={
                    "conversation_id": conversation_id,
                    "model": "deepseek-chat",
                    "user_content": "Hello",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )

        self.assertEqual(response.status_code, 200)
        body = response.get_data(as_text=True).strip().splitlines()
        events = [json.loads(line) for line in body]
        event_types = [event["type"] for event in events]
        self.assertIn("answer_start", event_types)
        self.assertIn("answer_delta", event_types)
        self.assertIn("usage", event_types)
        self.assertIn("done", event_types)
        self.assertLess(event_types.index("done"), event_types.index("message_ids"))

        with get_db() as conn:
            rows = conn.execute(
                "SELECT role, content, metadata, prompt_tokens, completion_tokens, total_tokens FROM messages WHERE conversation_id = ? ORDER BY id",
                (conversation_id,),
            ).fetchall()

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["role"], "user")
        self.assertEqual(rows[0]["content"], "Hello")
        self.assertEqual(rows[1]["role"], "assistant")
        self.assertEqual(rows[1]["content"], "Hello world")
        assistant_metadata = json.loads(rows[1]["metadata"])
        self.assertEqual(assistant_metadata["reasoning_content"], "Analyzing request")
        self.assertEqual(assistant_metadata["tool_trace"][0]["tool_name"], "search_web")
        self.assertEqual(assistant_metadata["tool_trace"][0]["state"], "done")
        self.assertEqual(assistant_metadata["usage"]["estimated_input_tokens"], 11)
        self.assertEqual(assistant_metadata["usage"]["input_breakdown"]["user_messages"], 6)
        self.assertEqual(assistant_metadata["usage"]["input_breakdown"]["core_instructions"], 5)
        self.assertEqual(rows[1]["prompt_tokens"], 11)
        self.assertEqual(rows[1]["completion_tokens"], 7)
        self.assertEqual(rows[1]["total_tokens"], 18)

        conversation_response = self.client.get(f"/api/conversations/{conversation_id}")
        self.assertEqual(conversation_response.status_code, 200)
        assistant_message = conversation_response.get_json()["messages"][1]
        self.assertEqual(assistant_message["usage"]["estimated_input_tokens"], 11)
        self.assertEqual(assistant_message["usage"]["input_breakdown"]["core_instructions"], 5)
        self.assertEqual(assistant_message["metadata"]["tool_trace"][0]["tool_name"], "search_web")

    def test_failed_tool_summary_detection_marks_fetch_failures(self):
        self.assertTrue(_is_failed_tool_summary("Fetch failed: HTTP 403"))
        self.assertTrue(_is_failed_tool_summary("failed: timeout while reading response"))
        self.assertFalse(_is_failed_tool_summary("Page content extracted: Example"))

    def test_chat_stream_persists_pending_clarification(self):
        conversation_id = self._create_conversation()
        save_app_settings(
            {
                "user_preferences": "",
                "max_steps": "2",
                "active_tools": '["ask_clarifying_question"]',
                "rag_auto_inject": "false",
            }
        )

        fake_events = iter(
            [
                {
                    "type": "clarification_request",
                    "text": "Before I answer, I need two details.\n1. Which scope?\n2. Anything else?",
                    "clarification": {
                        "intro": "Before I answer, I need two details.",
                        "submit_label": "Continue",
                        "questions": [
                            {
                                "id": "scope",
                                "label": "Which scope?",
                                "input_type": "single_select",
                                "options": [
                                    {"label": "Only this repo", "value": "repo"},
                                    {"label": "General tool", "value": "general"},
                                ],
                            },
                            {
                                "id": "notes",
                                "label": "Anything else?",
                                "input_type": "text",
                                "required": False,
                            },
                        ],
                    },
                },
                {"type": "tool_capture", "tool_results": []},
                {"type": "done"},
            ]
        )

        with patch("routes.chat.run_agent_stream", return_value=fake_events):
            response = self.client.post(
                "/chat",
                json={
                    "conversation_id": conversation_id,
                    "model": "deepseek-chat",
                    "user_content": "Build this for me",
                    "messages": [{"role": "user", "content": "Build this for me"}],
                },
            )

        self.assertEqual(response.status_code, 200)
        streamed_events = [json.loads(line) for line in response.get_data(as_text=True).strip().splitlines() if line.strip()]
        clarification_event = next((event for event in streamed_events if event["type"] == "clarification_request"), None)
        self.assertIsNotNone(clarification_event)
        self.assertEqual(clarification_event["clarification"]["submit_label"], "Continue")

        with get_db() as conn:
            rows = conn.execute(
                "SELECT role, content, metadata FROM messages WHERE conversation_id = ? ORDER BY id",
                (conversation_id,),
            ).fetchall()

        self.assertEqual([row["role"] for row in rows], ["user", "assistant"])
        assistant_metadata = json.loads(rows[1]["metadata"])
        self.assertEqual(rows[1]["content"], clarification_event["text"])
        self.assertEqual(assistant_metadata["pending_clarification"]["questions"][0]["id"], "scope")

    def test_chat_stream_persists_tool_history_rows(self):
        conversation_id = self._create_conversation()
        save_app_settings(
            {
                "user_preferences": "",
                "max_steps": "2",
                "active_tools": '["search_web"]',
                "rag_auto_inject": "false",
            }
        )

        fake_events = iter(
            [
                {"type": "step_started", "step": 1, "max_steps": 2},
                {
                    "type": "tool_history",
                    "step": 1,
                    "messages": [
                        {
                            "role": "assistant",
                            "content": "I will search first.",
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "type": "function",
                                    "function": {
                                        "name": "search_web",
                                        "arguments": '{"queries":["istanbul"]}',
                                    },
                                }
                            ],
                        },
                        {
                            "role": "tool",
                            "tool_call_id": "call-1",
                            "content": '{"ok":true,"results":[{"title":"Istanbul"}]}',
                        },
                    ],
                },
                {"type": "answer_start"},
                {"type": "answer_delta", "text": "Results are ready."},
                {"type": "tool_capture", "tool_results": []},
                {"type": "done"},
            ]
        )

        with patch("routes.chat.run_agent_stream", return_value=fake_events):
            response = self.client.post(
                "/chat",
                json={
                    "conversation_id": conversation_id,
                    "model": "deepseek-chat",
                    "user_content": "What is Istanbul?",
                    "messages": [{"role": "user", "content": "What is Istanbul?"}],
                },
            )

        self.assertEqual(response.status_code, 200)
        events = [json.loads(line) for line in response.get_data(as_text=True).strip().splitlines()]
        tool_history_event = next((event for event in events if event["type"] == "assistant_tool_history"), None)
        self.assertIsNotNone(tool_history_event)
        self.assertEqual(len(tool_history_event["messages"]), 2)

        with get_db() as conn:
            rows = conn.execute(
                "SELECT role, content, tool_calls, tool_call_id FROM messages WHERE conversation_id = ? ORDER BY id",
                (conversation_id,),
            ).fetchall()

        self.assertEqual([row["role"] for row in rows], ["user", "assistant", "tool", "assistant"])
        self.assertIn("search_web", rows[1]["tool_calls"])
        self.assertEqual(rows[2]["tool_call_id"], "call-1")
        self.assertEqual(rows[3]["content"], "Results are ready.")

        conversation_response = self.client.get(f"/api/conversations/{conversation_id}")
        self.assertEqual(conversation_response.status_code, 200)
        messages = conversation_response.get_json()["messages"]
        self.assertEqual([message["role"] for message in messages], ["user", "assistant", "tool", "assistant"])
        self.assertEqual(messages[1]["tool_calls"][0]["function"]["name"], "search_web")
        self.assertEqual(messages[2]["tool_call_id"], "call-1")

    def test_serialize_message_metadata_keeps_tool_trace(self):
        payload = serialize_message_metadata(
            {
                "tool_trace": [
                    {
                        "tool_name": "search_web",
                        "step": 1,
                        "preview": "query",
                        "summary": "2 web results found",
                        "state": "done",
                        "cached": True,
                    }
                ]
            }
        )

        metadata = parse_message_metadata(payload)
        self.assertEqual(metadata["tool_trace"][0]["tool_name"], "search_web")
        self.assertEqual(metadata["tool_trace"][0]["state"], "done")
        self.assertTrue(metadata["tool_trace"][0]["cached"])

    def test_serialize_message_metadata_keeps_canvas_documents(self):
        payload = serialize_message_metadata(
            {
                "canvas_documents": [
                    {
                        "id": "canvas-1",
                        "title": "Draft",
                        "format": "markdown",
                        "content": "# Draft\n\nHello",
                    },
                    {
                        "id": "canvas-2",
                        "title": "Notes",
                        "format": "markdown",
                        "content": "# Notes\n\nExtra",
                    }
                ]
            }
        )

        metadata = parse_message_metadata(payload)
        self.assertEqual(metadata["canvas_documents"][0]["id"], "canvas-1")
        self.assertEqual(metadata["canvas_documents"][0]["title"], "Draft")
        self.assertEqual(metadata["canvas_documents"][1]["id"], "canvas-2")

    def test_serialize_message_metadata_keeps_active_canvas_document_id(self):
        payload = serialize_message_metadata(
            {
                "canvas_documents": [
                    {
                        "id": "canvas-1",
                        "title": "Draft",
                        "format": "markdown",
                        "content": "# Draft",
                    },
                    {
                        "id": "canvas-2",
                        "title": "Notes",
                        "format": "markdown",
                        "content": "# Notes",
                    },
                ],
                "active_document_id": "canvas-1",
            }
        )

        metadata = parse_message_metadata(payload)
        self.assertEqual(metadata["active_document_id"], "canvas-1")

    def test_serialize_message_metadata_keeps_project_workflow(self):
        payload = serialize_message_metadata(
            {
                "project_workflow": {
                    "stage": "validate",
                    "project_name": "Demo App",
                    "goal": "Build a service",
                    "target_type": "python-project",
                    "files": [
                        {"path": "demo-app/app.py", "role": "source", "purpose": "Entry point", "status": "written"},
                    ],
                    "validation": {
                        "status": "needs_attention",
                        "issues": ["Example issue"],
                        "warnings": ["Example warning"],
                    },
                }
            }
        )

        metadata = parse_message_metadata(payload)
        self.assertEqual(metadata["project_workflow"]["stage"], "validate")
        self.assertEqual(metadata["project_workflow"]["files"][0]["path"], "demo-app/app.py")
        self.assertEqual(metadata["project_workflow"]["validation"]["status"], "needs_attention")

    def test_serialize_message_metadata_keeps_canvas_cleared_flag(self):
        payload = serialize_message_metadata(
            {
                "canvas_documents": [],
                "canvas_cleared": True,
            }
        )

        metadata = parse_message_metadata(payload)
        self.assertEqual(metadata["canvas_documents"], [])
        self.assertTrue(metadata["canvas_cleared"])

    def test_find_latest_canvas_documents_stops_at_cleared_marker(self):
        messages = [
            {
                "id": 1,
                "metadata": {
                    "canvas_documents": [
                        {
                            "id": "canvas-1",
                            "title": "Draft",
                            "format": "markdown",
                            "content": "# Draft",
                        }
                    ]
                },
            },
            {
                "id": 2,
                "metadata": {
                    "canvas_documents": [],
                    "canvas_cleared": True,
                },
            },
        ]

        self.assertEqual(find_latest_canvas_documents(messages), [])

    def test_find_latest_canvas_state_restores_active_document_id(self):
        messages = [
            {
                "id": 1,
                "metadata": {
                    "canvas_documents": [
                        {
                            "id": "canvas-1",
                            "title": "app.py",
                            "path": "src/app.py",
                            "format": "code",
                            "content": "print('hello')",
                        },
                        {
                            "id": "canvas-2",
                            "title": "README.md",
                            "path": "README.md",
                            "format": "markdown",
                            "content": "# Demo",
                        },
                    ],
                    "active_document_id": "canvas-1",
                },
            }
        ]

        runtime_state = find_latest_canvas_state(messages)
        self.assertEqual(get_canvas_runtime_active_document_id(runtime_state), "canvas-1")
        self.assertEqual(runtime_state["mode"], "project")

    def test_canvas_project_manifest_prioritizes_source_and_config_files(self):
        manifest = build_canvas_project_manifest(
            [
                {
                    "id": "canvas-1",
                    "title": "README.md",
                    "path": "README.md",
                    "role": "docs",
                    "format": "markdown",
                    "content": "# Demo",
                },
                {
                    "id": "canvas-2",
                    "title": "config.py",
                    "path": "src/config.py",
                    "role": "config",
                    "format": "code",
                    "content": "settings = {}",
                },
                {
                    "id": "canvas-3",
                    "title": "app.py",
                    "path": "src/app.py",
                    "role": "source",
                    "format": "code",
                    "content": "print('hello')",
                },
            ],
            active_document_id="canvas-3",
        )

        self.assertEqual(manifest["active_file"], "src/app.py")
        self.assertEqual(manifest["last_validation_status"], "ok")
        self.assertEqual([entry["path"] for entry in manifest["file_list"]], ["src/app.py", "src/config.py", "README.md"])
        self.assertEqual([entry["priority"] for entry in manifest["file_list"]], [10, 20, 60])

    def test_canvas_project_manifest_flags_duplicate_paths(self):
        manifest = build_canvas_project_manifest(
            [
                {
                    "id": "canvas-1",
                    "title": "app.py",
                    "path": "src/app.py",
                    "role": "source",
                    "format": "code",
                    "content": "print('one')",
                },
                {
                    "id": "canvas-2",
                    "title": "app copy.py",
                    "path": "src/app.py",
                    "role": "source",
                    "format": "code",
                    "content": "print('two')",
                },
            ],
            active_document_id="canvas-1",
        )

        self.assertEqual(manifest["last_validation_status"], "needs_attention")
        self.assertIn("Duplicate project paths detected.", manifest["open_issues"])

    def test_normalize_canvas_document_accepts_code_format(self):
        document = normalize_canvas_document(
            {
                "id": "canvas-code",
                "title": "Script",
                "format": "code",
                "language": "python",
                "content": "print('ok')",
            }
        )

        self.assertEqual(document["format"], "code")
        self.assertEqual(document["language"], "python")

    def test_create_canvas_runtime_state_preserves_multiple_documents(self):
        runtime_state = create_canvas_runtime_state(
            [
                {"id": "canvas-1", "title": "Draft", "format": "markdown", "content": "one"},
                {"id": "canvas-2", "title": "Notes", "format": "markdown", "content": "two"},
            ]
        )

        self.assertEqual(len(runtime_state["documents"]), 2)
        self.assertEqual(runtime_state["active_document_id"], "canvas-2")

    def test_chat_persists_canvas_documents_from_tool_capture(self):
        conversation_id = self._create_conversation()
        fake_events = iter(
            [
                {"type": "answer_start"},
                {"type": "answer_delta", "text": "I prepared a draft."},
                {
                    "type": "tool_capture",
                    "tool_results": [],
                    "canvas_documents": [
                        {
                            "id": "canvas-1",
                            "title": "Draft",
                            "format": "markdown",
                            "content": "# Draft\n\nInitial version",
                        }
                    ],
                },
                {"type": "done"},
            ]
        )

        with patch("routes.chat.run_agent_stream", return_value=fake_events):
            response = self.client.post(
                "/chat",
                json={
                    "conversation_id": conversation_id,
                    "model": "deepseek-chat",
                    "user_content": "Create a draft",
                    "messages": [{"role": "user", "content": "Create a draft"}],
                },
            )

        self.assertEqual(response.status_code, 200)
        events = [json.loads(line) for line in response.get_data(as_text=True).strip().splitlines()]
        canvas_event = next((event for event in events if event["type"] == "canvas_sync"), None)
        self.assertIsNotNone(canvas_event)
        self.assertEqual(canvas_event["documents"][0]["title"], "Draft")

        conversation_response = self.client.get(f"/api/conversations/{conversation_id}")
        messages = conversation_response.get_json()["messages"]
        assistant_messages = [message for message in messages if message["role"] == "assistant"]
        self.assertEqual(assistant_messages[-1]["metadata"]["canvas_documents"][0]["id"], "canvas-1")

    def test_chat_persists_canvas_documents_without_text_response(self):
        conversation_id = self._create_conversation()
        fake_events = iter(
            [
                {
                    "type": "tool_capture",
                    "tool_results": [],
                    "canvas_documents": [
                        {
                            "id": "canvas-empty-answer",
                            "title": "Draft",
                            "format": "markdown",
                            "content": "# Draft\n\nInitial version",
                        }
                    ],
                },
                {"type": "done"},
            ]
        )

        with patch("routes.chat.run_agent_stream", return_value=fake_events):
            response = self.client.post(
                "/chat",
                json={
                    "conversation_id": conversation_id,
                    "model": "deepseek-chat",
                    "user_content": "Create a draft",
                    "messages": [{"role": "user", "content": "Create a draft"}],
                },
            )

        self.assertEqual(response.status_code, 200)
        events = [json.loads(line) for line in response.get_data(as_text=True).strip().splitlines()]
        canvas_event = next((event for event in events if event["type"] == "canvas_sync"), None)
        self.assertIsNotNone(canvas_event)

        conversation_response = self.client.get(f"/api/conversations/{conversation_id}")
        messages = conversation_response.get_json()["messages"]
        assistant_messages = [message for message in messages if message["role"] == "assistant"]
        self.assertEqual(len(assistant_messages), 1)
        self.assertEqual(assistant_messages[0]["content"], "")
        self.assertEqual(
            assistant_messages[0]["metadata"]["canvas_documents"][0]["id"],
            "canvas-empty-answer",
        )

    def test_chat_persists_cleared_canvas_without_documents(self):
        conversation_id = self._create_conversation()
        fake_events = iter(
            [
                {
                    "type": "tool_capture",
                    "tool_results": [],
                    "canvas_documents": [],
                    "canvas_cleared": True,
                },
                {"type": "done"},
            ]
        )

        with patch("routes.chat.run_agent_stream", return_value=fake_events):
            response = self.client.post(
                "/chat",
                json={
                    "conversation_id": conversation_id,
                    "model": "deepseek-chat",
                    "user_content": "Clear the canvas",
                    "messages": [{"role": "user", "content": "Clear the canvas"}],
                },
            )

        self.assertEqual(response.status_code, 200)
        events = [json.loads(line) for line in response.get_data(as_text=True).strip().splitlines()]
        canvas_event = next((event for event in events if event["type"] == "canvas_sync"), None)
        self.assertIsNotNone(canvas_event)
        self.assertEqual(canvas_event["documents"], [])
        self.assertTrue(canvas_event["cleared"])

        conversation_response = self.client.get(f"/api/conversations/{conversation_id}")
        messages = conversation_response.get_json()["messages"]
        assistant_messages = [message for message in messages if message["role"] == "assistant"]
        self.assertEqual(len(assistant_messages), 1)
        self.assertEqual(assistant_messages[0]["metadata"]["canvas_documents"], [])
        self.assertTrue(assistant_messages[0]["metadata"]["canvas_cleared"])

    def test_uploaded_document_prompts_before_opening_canvas(self):
        conversation_id = self._create_conversation()

        fake_events = iter([
            {"type": "done"},
        ])

        with patch("routes.chat.run_agent_stream", return_value=fake_events):
            response = self.client.post(
                "/chat",
                data={
                    "conversation_id": str(conversation_id),
                    "model": "deepseek-chat",
                    "user_content": "Please review this file",
                    "messages": json.dumps([
                        {"role": "user", "content": "Please review this file"},
                    ]),
                    "document": (io.BytesIO(b"Project notes\n\nDetails"), "notes.txt", "text/plain"),
                },
                content_type="multipart/form-data",
            )

        self.assertEqual(response.status_code, 200)
        events = [json.loads(line) for line in response.get_data(as_text=True).strip().splitlines()]
        document_event = next((event for event in events if event["type"] == "document_processed"), None)
        self.assertIsNotNone(document_event)
        self.assertIn("canvas_document", document_event)

        canvas_event = next((event for event in events if event["type"] == "canvas_sync"), None)
        self.assertIsNotNone(canvas_event)
        self.assertFalse(canvas_event.get("auto_open"))

    def test_canvas_export_endpoint_returns_markdown_and_pdf(self):
        conversation_id = self._create_conversation()
        metadata = serialize_message_metadata(
            {
                "canvas_documents": [
                    {
                        "id": "canvas-export",
                        "title": "Draft Export",
                        "format": "markdown",
                        "content": "# Export\n\n- One\n- Two",
                    }
                ]
            }
        )

        with get_db() as conn:
            insert_message(conn, conversation_id, "assistant", "Here is the draft.", metadata=metadata)

        markdown_response = self.client.get(
            f"/api/conversations/{conversation_id}/canvas/export?format=md&document_id=canvas-export"
        )
        self.assertEqual(markdown_response.status_code, 200)
        self.assertEqual(markdown_response.mimetype, "text/markdown")
        self.assertIn("attachment; filename=\"Draft-Export.md\"", markdown_response.headers["Content-Disposition"])
        self.assertIn("# Export", markdown_response.get_data(as_text=True))

        pdf_response = self.client.get(
            f"/api/conversations/{conversation_id}/canvas/export?format=pdf&document_id=canvas-export"
        )
        self.assertEqual(pdf_response.status_code, 200)
        self.assertEqual(pdf_response.mimetype, "application/pdf")
        self.assertTrue(pdf_response.data.startswith(b"%PDF"))

        html_response = self.client.get(
            f"/api/conversations/{conversation_id}/canvas/export?format=html&document_id=canvas-export"
        )
        self.assertEqual(html_response.status_code, 200)
        self.assertEqual(html_response.mimetype, "text/html")
        self.assertIn("attachment; filename=\"Draft-Export.html\"", html_response.headers["Content-Disposition"])
        self.assertIn("<ul>", html_response.get_data(as_text=True))

    def test_canvas_export_endpoint_renders_code_format(self):
        conversation_id = self._create_conversation()
        metadata = serialize_message_metadata(
            {
                "canvas_documents": [
                    {
                        "id": "canvas-code-export",
                        "title": "main.py",
                        "format": "code",
                        "language": "python",
                        "content": "print('hello')",
                    }
                ]
            }
        )

        with get_db() as conn:
            insert_message(conn, conversation_id, "assistant", "Here is the code.", metadata=metadata)

        markdown_response = self.client.get(
            f"/api/conversations/{conversation_id}/canvas/export?format=md&document_id=canvas-code-export"
        )
        self.assertEqual(markdown_response.status_code, 200)
        self.assertIn("```python", markdown_response.get_data(as_text=True))

        html_response = self.client.get(
            f"/api/conversations/{conversation_id}/canvas/export?format=html&document_id=canvas-code-export"
        )
        self.assertEqual(html_response.status_code, 200)
        self.assertIn("language-python", html_response.get_data(as_text=True))

    def test_canvas_delete_endpoint_updates_canvas_state(self):
        conversation_id = self._create_conversation()
        metadata = serialize_message_metadata(
            {
                "canvas_documents": [
                    {
                        "id": "canvas-one",
                        "title": "Draft One",
                        "format": "markdown",
                        "content": "# One",
                    },
                    {
                        "id": "canvas-two",
                        "title": "Draft Two",
                        "format": "markdown",
                        "content": "# Two",
                    },
                ]
            }
        )

        with get_db() as conn:
            insert_message(conn, conversation_id, "assistant", "Canvas ready.", metadata=metadata)

        delete_response = self.client.delete(
            f"/api/conversations/{conversation_id}/canvas?document_id=canvas-two"
        )
        self.assertEqual(delete_response.status_code, 200)
        delete_payload = delete_response.get_json()
        self.assertFalse(delete_payload["cleared"])
        self.assertEqual(delete_payload["remaining_count"], 1)
        self.assertEqual(delete_payload["deleted_document_id"], "canvas-two")
        self.assertEqual(delete_payload["documents"][0]["id"], "canvas-one")

        clear_response = self.client.delete(
            f"/api/conversations/{conversation_id}/canvas?clear_all=true"
        )
        self.assertEqual(clear_response.status_code, 200)
        clear_payload = clear_response.get_json()
        self.assertTrue(clear_payload["cleared"])
        self.assertEqual(clear_payload["remaining_count"], 0)
        self.assertEqual(clear_payload["documents"], [])

        conversation_response = self.client.get(f"/api/conversations/{conversation_id}")
        self.assertEqual(conversation_response.status_code, 200)
        messages = conversation_response.get_json()["messages"]
        self.assertEqual(messages[-1]["role"], "tool")
        self.assertEqual(messages[-1]["metadata"]["canvas_documents"], [])
        self.assertTrue(messages[-1]["metadata"]["canvas_cleared"])

    def test_canvas_delete_endpoint_accepts_document_path(self):
        conversation_id = self._create_conversation()
        metadata = serialize_message_metadata(
            {
                "canvas_documents": [
                    {
                        "id": "canvas-one",
                        "title": "app.py",
                        "path": "src/app.py",
                        "format": "code",
                        "content": "print('one')",
                    },
                    {
                        "id": "canvas-two",
                        "title": "config.py",
                        "path": "src/config.py",
                        "format": "code",
                        "content": "settings = {}",
                    },
                ],
                "active_document_id": "canvas-two",
            }
        )

        with get_db() as conn:
            insert_message(conn, conversation_id, "assistant", "Canvas ready.", metadata=metadata)

        delete_response = self.client.delete(
            f"/api/conversations/{conversation_id}/canvas?document_path=src/config.py"
        )
        self.assertEqual(delete_response.status_code, 200)
        delete_payload = delete_response.get_json()
        self.assertEqual(delete_payload["remaining_count"], 1)
        self.assertEqual(delete_payload["documents"][0]["path"], "src/app.py")

    def test_canvas_patch_endpoint_updates_document_content_and_format(self):
        conversation_id = self._create_conversation()
        metadata = serialize_message_metadata(
            {
                "canvas_documents": [
                    {
                        "id": "canvas-edit",
                        "title": "Draft",
                        "format": "markdown",
                        "content": "# Draft\n\nInitial",
                    }
                ]
            }
        )

        with get_db() as conn:
            insert_message(conn, conversation_id, "assistant", "Here is the draft.", metadata=metadata)

        response = self.client.patch(
            f"/api/conversations/{conversation_id}/canvas",
            json={
                "document_id": "canvas-edit",
                "content": "print('saved')",
                "format": "code",
                "language": "python",
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["document"]["format"], "code")
        self.assertEqual(payload["document"]["language"], "python")
        self.assertEqual(payload["document"]["content"], "print('saved')")

        conversation_response = self.client.get(f"/api/conversations/{conversation_id}")
        messages = conversation_response.get_json()["messages"]
        latest_canvas = find_latest_canvas_documents(messages)
        self.assertEqual(latest_canvas[0]["format"], "code")
        self.assertEqual(latest_canvas[0]["content"], "print('saved')")

    def test_canvas_patch_endpoint_accepts_document_path(self):
        conversation_id = self._create_conversation()
        metadata = serialize_message_metadata(
            {
                "canvas_documents": [
                    {
                        "id": "canvas-edit",
                        "title": "app.py",
                        "path": "src/app.py",
                        "format": "code",
                        "content": "print('old')",
                    }
                ]
            }
        )

        with get_db() as conn:
            insert_message(conn, conversation_id, "assistant", "Here is the draft.", metadata=metadata)

        response = self.client.patch(
            f"/api/conversations/{conversation_id}/canvas",
            json={
                "document_path": "src/app.py",
                "content": "print('new')",
                "format": "code",
                "language": "python",
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["document"]["path"], "src/app.py")
        self.assertEqual(payload["document"]["content"], "print('new')")

    def test_document_canvas_inference_for_code_files(self):
        self.assertEqual(infer_canvas_format("main.py"), "code")
        self.assertEqual(infer_canvas_language("main.py"), "python")
        self.assertEqual(build_canvas_markdown("main.py", "print('hello')"), "print('hello')")
        self.assertEqual(infer_canvas_format("notes.md"), "markdown")

    def test_conversation_export_endpoint_returns_markdown_docx_and_pdf(self):
        conversation_id = self._create_conversation("Exportable Chat")

        with get_db() as conn:
            insert_message(conn, conversation_id, "user", "Hello")
            insert_message(
                conn,
                conversation_id,
                "assistant",
                "Here is the answer.",
                metadata=serialize_message_metadata(
                    {
                        "reasoning_content": "Reasoned through the request.",
                        "tool_trace": [
                            {
                                "tool_name": "search_web",
                                "step": 1,
                                "preview": "hello",
                                "summary": "1 web result found",
                                "state": "done",
                            }
                        ],
                    }
                ),
            )

        markdown_response = self.client.get(f"/api/conversations/{conversation_id}/export?format=md")
        self.assertEqual(markdown_response.status_code, 200)
        self.assertEqual(markdown_response.mimetype, "text/markdown")
        self.assertIn("attachment; filename=\"Exportable-Chat.md\"", markdown_response.headers["Content-Disposition"])
        self.assertIn("## 1. User", markdown_response.get_data(as_text=True))
        self.assertIn("### Tool Trace", markdown_response.get_data(as_text=True))

        docx_response = self.client.get(f"/api/conversations/{conversation_id}/export?format=docx")
        self.assertEqual(docx_response.status_code, 200)
        self.assertEqual(
            docx_response.mimetype,
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
        self.assertTrue(docx_response.data.startswith(b"PK"))

        pdf_response = self.client.get(f"/api/conversations/{conversation_id}/export?format=pdf")
        self.assertEqual(pdf_response.status_code, 200)
        self.assertEqual(pdf_response.mimetype, "application/pdf")
        self.assertTrue(pdf_response.data.startswith(b"%PDF"))

    def test_canvas_line_replace_rejects_out_of_bounds_start_line(self):
        runtime_state = create_canvas_runtime_state(
            [
                {
                    "id": "canvas-1",
                    "title": "Draft",
                    "content": "line 1\nline 2",
                    "format": "markdown",
                }
            ]
        )

        with self.assertRaisesRegex(ValueError, "Line range exceeds"):
            replace_canvas_lines(runtime_state, 3, 3, ["replacement"], document_id="canvas-1")

    def test_normalize_canvas_document_falls_back_to_default_title(self):
        normalized = normalize_canvas_document(
            {
                "id": "canvas-blank-title",
                "title": "   ",
                "format": "markdown",
                "content": "content",
            }
        )

        self.assertEqual(normalized["title"], "Canvas")

    def test_normalize_canvas_document_preserves_language_metadata(self):
        normalized = normalize_canvas_document(
            {
                "id": "canvas-python",
                "title": "script.py",
                "format": "markdown",
                "language": " Python ",
                "content": "print('hello')",
            }
        )

        self.assertEqual(normalized["language"], "python")

    def test_normalize_canvas_document_preserves_project_metadata(self):
        normalized = normalize_canvas_document(
            {
                "id": "canvas-project",
                "title": "app.py",
                "path": "./src/app.py",
                "role": "source",
                "summary": " Main application entry point ",
                "imports": ["config", "config", "os"],
                "exports": ["create_app"],
                "symbols": ["create_app", "main"],
                "dependencies": ["flask", "python-dotenv"],
                "project_id": "Demo App",
                "workspace_id": "Workspace-1",
                "format": "code",
                "language": "python",
                "content": "print('hello')",
            }
        )

        self.assertEqual(normalized["path"], "src/app.py")
        self.assertEqual(normalized["role"], "source")
        self.assertEqual(normalized["summary"], "Main application entry point")
        self.assertEqual(normalized["imports"], ["config", "os"])
        self.assertEqual(normalized["dependencies"], ["flask", "python-dotenv"])
        self.assertEqual(normalized["project_id"], "demoapp")
        self.assertEqual(normalized["workspace_id"], "workspace-1")

    def test_canvas_line_tools_accept_document_path(self):
        runtime_state = create_canvas_runtime_state(
            [
                {
                    "id": "canvas-1",
                    "title": "app.py",
                    "path": "src/app.py",
                    "format": "code",
                    "content": "line 1\nline 2",
                }
            ]
        )

        updated = replace_canvas_lines(runtime_state, 2, 2, ["line changed"], document_path="src/app.py")
        self.assertEqual(updated["path"], "src/app.py")
        self.assertEqual(updated["content"], "line 1\nline changed")

    def test_run_agent_stream_executes_native_tool_calls(self):
        responses = [
            iter(
                [
                    self._stream_chunk(reasoning="Need current info. "),
                    self._stream_chunk(
                        tool_calls=[
                            {
                                "index": 0,
                                "id": "tool-call-1",
                                "function": {
                                    "name": "search_web",
                                    "arguments": '{"queries":["test query"]}',
                                },
                            }
                        ]
                    ),
                    self._stream_chunk(usage=SimpleNamespace(prompt_tokens=4, completion_tokens=6, total_tokens=10)),
                ]
            ),
            iter(
                [
                    self._stream_chunk(reasoning="Using fetched context. "),
                    self._stream_chunk(content="Final answer."),
                    self._stream_chunk(usage=SimpleNamespace(prompt_tokens=3, completion_tokens=5, total_tokens=8)),
                ]
            ),
        ]

        with patch("agent.client.chat.completions.create", side_effect=responses) as mocked_create, patch(
            "agent.search_web_tool",
            return_value=[{"title": "Test", "url": "https://example.com", "snippet": "Snippet"}],
        ):
            events = list(run_agent_stream([{"role": "user", "content": "Test"}], "deepseek-chat", 3, ["search_web"]))

        event_types = [event["type"] for event in events]
        self.assertIn("step_started", event_types)
        self.assertIn("reasoning_start", event_types)
        self.assertIn({"type": "reasoning_delta", "text": "Need current info. "}, events)
        self.assertIn("step_update", event_types)
        self.assertIn("tool_result", event_types)
        self.assertIn("answer_start", event_types)
        self.assertIn({"type": "answer_delta", "text": "Final answer."}, events)

        for call in mocked_create.call_args_list:
            _, kwargs = call
            self.assertTrue(kwargs.get("stream"))

        first_call_kwargs = mocked_create.call_args_list[0].kwargs
        self.assertIn("tools", first_call_kwargs)
        self.assertEqual(first_call_kwargs["tool_choice"], "auto")

        second_call_kwargs = mocked_create.call_args_list[1].kwargs
        self.assertIn("tools", second_call_kwargs)
        self.assertEqual(second_call_kwargs["tool_choice"], "auto")

        usage_events = [event for event in events if event["type"] == "usage"]
        self.assertEqual(len(usage_events), 1)
        usage_event = usage_events[0]
        self.assertEqual(usage_event["prompt_tokens"], 7)
        self.assertEqual(usage_event["completion_tokens"], 11)
        self.assertEqual(usage_event["total_tokens"], 18)
        self.assertGreater(usage_event["estimated_input_tokens"], 0)
        self.assertGreater(usage_event["input_breakdown"]["user_messages"], 0)
        self.assertGreater(usage_event["input_breakdown"]["tool_results"], 0)
        self.assertEqual(usage_event["input_breakdown"]["assistant_history"], 0)
        self.assertEqual(usage_event["model_call_count"], 2)
        self.assertEqual(len(usage_event["model_calls"]), 2)
        self.assertEqual(usage_event["model_calls"][0]["call_type"], "agent_step")
        self.assertEqual(usage_event["model_calls"][0]["step"], 1)
        self.assertFalse(usage_event["model_calls"][0]["missing_provider_usage"])
        self.assertEqual(usage_event["model_calls"][1]["call_type"], "agent_step")
        self.assertEqual(usage_event["model_calls"][1]["prompt_tokens"], 3)
        self.assertEqual(
            usage_event["model_calls"][1]["estimated_input_tokens"],
            usage_event["model_calls"][1]["prompt_tokens"],
        )

        second_call_messages = mocked_create.call_args_list[1].kwargs["messages"]
        self.assertEqual([message["role"] for message in second_call_messages], ["user", "assistant", "tool", "system"])
        self.assertEqual(second_call_messages[1]["tool_calls"][0]["function"]["name"], "search_web")
        self.assertEqual(second_call_messages[2]["tool_call_id"], "tool-call-1")
        self.assertIn("[AGENT REASONING CONTEXT]", second_call_messages[3]["content"])

        tool_history_event = next((event for event in events if event["type"] == "tool_history"), None)
        self.assertIsNotNone(tool_history_event)
        self.assertEqual(tool_history_event["messages"][0]["role"], "assistant")
        self.assertEqual(tool_history_event["messages"][1]["role"], "tool")

    def test_build_api_messages_preserves_tool_history_fields(self):
        normalized = normalize_chat_messages(
            [
                {"role": "user", "content": "Hello"},
                {
                    "role": "assistant",
                    "content": "I am searching.",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "search_web",
                                "arguments": '{"queries":["hello"]}',
                            },
                        }
                    ],
                },
                {"role": "tool", "content": "{}", "tool_call_id": "call-1"},
            ]
        )

        api_messages = build_api_messages(normalized)

        self.assertEqual(api_messages[1]["tool_calls"][0]["function"]["name"], "search_web")
        self.assertEqual(api_messages[2]["tool_call_id"], "call-1")

    def test_build_api_messages_maps_summary_role_to_assistant_context(self):
        normalized = normalize_chat_messages(
            [
                {"role": "summary", "content": "The user asked for a short answer."},
            ]
        )

        api_messages = build_api_messages(normalized)

        self.assertEqual(api_messages[0]["role"], "assistant")
        self.assertIn("Conversation summary", api_messages[0]["content"])

    def test_run_agent_stream_retries_until_content_final_answer_arrives(self):
        responses = [
            iter(
                [
                    self._stream_chunk(reasoning="Thinking step by step."),
                    self._stream_chunk(usage=SimpleNamespace(prompt_tokens=2, completion_tokens=3, total_tokens=5)),
                ]
            ),
            iter(
                [
                    self._stream_chunk(content="Final answer."),
                    self._stream_chunk(usage=SimpleNamespace(prompt_tokens=2, completion_tokens=4, total_tokens=6)),
                ]
            ),
        ]

        with patch("agent.client.chat.completions.create", side_effect=responses) as mocked_create:
            events = list(run_agent_stream([{"role": "user", "content": "Test"}], "deepseek-reasoner", 2, []))

        self.assertIn({"type": "reasoning_delta", "text": "Thinking step by step."}, events)
        self.assertIn(
            {
                "type": "tool_error",
                "step": 1,
                "tool": "agent",
                "error": "The model returned no final answer content. Retrying and waiting for a final answer.",
            },
            events,
        )
        self.assertIn({"type": "answer_delta", "text": "Final answer."}, events)
        leaked_reasoning = [event for event in events if event["type"] == "answer_delta" and "Thinking" in event["text"]]
        self.assertEqual(leaked_reasoning, [])

        usage_event = next(event for event in events if event["type"] == "usage")
        self.assertEqual(usage_event["model_call_count"], 2)
        self.assertTrue(usage_event["model_calls"][1]["is_retry"])
        self.assertEqual(usage_event["model_calls"][1]["retry_reason"], "missing_final_answer")

        second_call_messages = mocked_create.call_args_list[1].kwargs["messages"]
        retry_content = second_call_messages[-1]["content"]
        self.assertIn("MISSING FINAL ANSWER", retry_content)
        self.assertIn("assistant content only", retry_content)

    def test_context_overflow_error_detection(self):
        self.assertTrue(_is_context_overflow_error("context_length_exceeded: requested 200000 tokens"))
        self.assertTrue(_is_context_overflow_error("This model's maximum context length is 128000 tokens."))
        self.assertFalse(_is_context_overflow_error("rate_limit_exceeded"))
        self.assertFalse(_is_context_overflow_error("429 Too Many Requests"))

    def test_iter_agent_exchange_blocks_keeps_assistant_and_tool_together(self):
        blocks = _iter_agent_exchange_blocks(
            [
                {"role": "system", "content": "sys"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "search_web", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call-1", "content": "tool result"},
                {"role": "user", "content": "next"},
            ]
        )

        self.assertEqual(blocks[1]["type"], "exchange")
        self.assertEqual([message["role"] for message in blocks[1]["messages"]], ["assistant", "tool"])

    def test_try_compact_messages_preserves_recent_exchanges(self):
        messages = [
            {"role": "system", "content": "sys"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "search_web", "arguments": "{}"}}],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "A" * 1200},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "call-2", "type": "function", "function": {"name": "fetch_url", "arguments": "{}"}}],
            },
            {"role": "tool", "tool_call_id": "call-2", "content": "B" * 1200},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "call-3", "type": "function", "function": {"name": "search_news_ddgs", "arguments": "{}"}}],
            },
            {"role": "tool", "tool_call_id": "call-3", "content": "C" * 1200},
        ]

        compacted = _try_compact_messages(messages, budget=200, keep_recent=2)

        self.assertIsNotNone(compacted)
        self.assertEqual(compacted[0]["role"], "system")
        self.assertEqual(compacted[1]["role"], "user")
        self.assertIn("compacted tool step 1", compacted[1]["content"])
        self.assertEqual(compacted[-2]["role"], "assistant")
        self.assertEqual(compacted[-1]["role"], "tool")

    def test_try_compact_messages_builds_semantic_summary(self):
        messages = [
            {"role": "system", "content": "sys"},
            {
                "role": "assistant",
                "content": "I should search for current release notes before answering.",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "search_web", "arguments": '{"queries": ["python release notes"]}'},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "content": "Web results\n\n1. Python 3.13 notes\nURL: https://example.com\nSnippet: Latest changes overview",
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "call-2", "type": "function", "function": {"name": "fetch_url", "arguments": '{"url": "https://example.com"}'}}],
            },
            {"role": "tool", "tool_call_id": "call-2", "content": "Title: Python 3.13\n\nKey highlights and changes."},
        ]

        compacted = _try_compact_messages(messages, budget=120, keep_recent=1)

        self.assertIsNotNone(compacted)
        compacted_summary = compacted[1]["content"]
        self.assertIn("Assistant intent:", compacted_summary)
        self.assertIn("search_web: python release notes", compacted_summary)
        self.assertIn("Outcomes:", compacted_summary)
        self.assertIn("Web results", compacted_summary)

    def test_prepare_tool_result_for_transcript_clips_large_non_fetch_payloads(self):
        result = _prepare_tool_result_for_transcript("search_web", {"items": ["x" * 30000]})
        rendered = _build_compact_tool_message_content("search_web", {}, {"ignored": True}, "summary", transcript_result=result)

        self.assertIsInstance(result, str)
        self.assertIn("[CLIPPED: original", result)
        self.assertIn("[CLIPPED: original", rendered)

    def test_run_agent_stream_deduplicates_missing_final_answer_instruction(self):
        responses = [
            iter([self._stream_chunk(reasoning="One."), self._stream_chunk(usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2))]),
            iter([self._stream_chunk(reasoning="Two."), self._stream_chunk(usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2))]),
            iter([self._stream_chunk(content="Final."), self._stream_chunk(usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2))]),
        ]

        with patch("agent.client.chat.completions.create", side_effect=responses) as mocked_create:
            events = list(run_agent_stream([{"role": "user", "content": "Test"}], "deepseek-reasoner", 3, []))

        self.assertIn({"type": "answer_delta", "text": "Final."}, events)
        third_call_messages = mocked_create.call_args_list[2].kwargs["messages"]
        retry_markers = [message for message in third_call_messages if "MISSING FINAL ANSWER" in str(message.get("content") or "")]
        self.assertEqual(len(retry_markers), 1)

    def test_run_agent_stream_recovers_from_context_overflow_before_model_turn(self):
        api_messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "Need answer"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "search_web", "arguments": "{}"}}],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "A" * 2500},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "call-2", "type": "function", "function": {"name": "fetch_url", "arguments": "{}"}}],
            },
            {"role": "tool", "tool_call_id": "call-2", "content": "B" * 2500},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "call-3", "type": "function", "function": {"name": "search_news_ddgs", "arguments": "{}"}}],
            },
            {"role": "tool", "tool_call_id": "call-3", "content": "C" * 2500},
        ]
        responses = [
            Exception("context_length_exceeded"),
            iter([self._stream_chunk(content="Recovered."), self._stream_chunk(usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2))]),
        ]

        with patch("agent.PROMPT_MAX_INPUT_TOKENS", 300), patch(
            "agent.client.chat.completions.create",
            side_effect=responses,
        ) as mocked_create:
            events = list(run_agent_stream(api_messages, "deepseek-chat", 2, []))

        self.assertIn({"type": "answer_delta", "text": "Recovered."}, events)
        self.assertEqual(mocked_create.call_count, 2)
        retried_messages = mocked_create.call_args_list[1].kwargs["messages"]
        self.assertTrue(any(message["role"] == "user" and "compacted tool step" in message["content"] for message in retried_messages))

    def test_run_agent_stream_traces_overflow_recovery_telemetry(self):
        api_messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "Need answer"},
            {
                "role": "assistant",
                "content": "I should inspect older search results.",
                "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "search_web", "arguments": "{}"}}],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "A" * 2500},
            {
                "role": "assistant",
                "content": "Now inspect fetched details.",
                "tool_calls": [{"id": "call-2", "type": "function", "function": {"name": "fetch_url", "arguments": "{}"}}],
            },
            {"role": "tool", "tool_call_id": "call-2", "content": "B" * 2500},
        ]
        responses = [
            Exception("context_length_exceeded"),
            iter([self._stream_chunk(content="Recovered."), self._stream_chunk(usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2))]),
        ]

        with patch("agent.PROMPT_MAX_INPUT_TOKENS", 250), patch(
            "agent.client.chat.completions.create",
            side_effect=responses,
        ), patch("agent._trace_agent_event") as mocked_trace:
            events = list(run_agent_stream(api_messages, "deepseek-chat", 2, []))

        self.assertIn({"type": "answer_delta", "text": "Recovered."}, events)
        compacted_calls = [call for call in mocked_trace.call_args_list if call.args and call.args[0] == "context_compacted"]
        self.assertTrue(compacted_calls)
        self.assertTrue(any(call.kwargs.get("force") is True for call in compacted_calls))
        self.assertTrue(any((call.kwargs.get("compacted_exchange_count") or 0) >= 1 for call in compacted_calls))
        recovered_calls = [call for call in mocked_trace.call_args_list if call.args and call.args[0] == "context_overflow_recovered"]
        self.assertTrue(recovered_calls)
        self.assertEqual(recovered_calls[-1].kwargs.get("phase"), "main_loop")

    def test_run_agent_stream_reports_unrecoverable_context_overflow(self):
        with patch("agent.client.chat.completions.create", side_effect=Exception("maximum context length exceeded")):
            events = list(run_agent_stream([{"role": "user", "content": "Test"}], "deepseek-chat", 1, []))

        self.assertIn(
            {
                "type": "tool_error",
                "step": 1,
                "tool": "api",
                "error": CONTEXT_OVERFLOW_RECOVERY_ERROR_TEXT,
            },
            events,
        )
        self.assertIn({"type": "answer_delta", "text": FINAL_ANSWER_ERROR_TEXT}, events)

    def test_final_answer_phase_recovers_from_context_overflow(self):
        responses = [
            iter(
                [
                    self._tool_call_chunk("search_web", {"queries": ["x"]}),
                    self._stream_chunk(usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2)),
                ]
            ),
            Exception("context_length_exceeded"),
            iter([self._stream_chunk(content="Final after compaction."), self._stream_chunk(usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2))]),
        ]

        with patch("agent.PROMPT_MAX_INPUT_TOKENS", 200), patch(
            "agent.AGENT_CONTEXT_COMPACTION_KEEP_RECENT_ROUNDS",
            0,
        ), patch("agent.client.chat.completions.create", side_effect=responses), patch(
            "agent.search_web_tool",
            return_value=[{"title": "Test", "url": "https://example.com", "snippet": "Snippet" * 200}],
        ):
            events = list(run_agent_stream([{"role": "user", "content": "Test"}], "deepseek-chat", 1, ["search_web"]))

        self.assertIn({"type": "answer_delta", "text": "Final after compaction."}, events)

    def test_run_agent_stream_separates_reasoning_turns_with_blank_line(self):
        responses = [
            iter(
                [
                    self._stream_chunk(reasoning="First reasoning block."),
                    self._tool_call_chunk("search_web", {"queries": ["x"]}),
                    self._stream_chunk(usage=SimpleNamespace(prompt_tokens=2, completion_tokens=3, total_tokens=5)),
                ]
            ),
            iter(
                [
                    self._stream_chunk(reasoning="Second reasoning block."),
                    self._stream_chunk(content="Final answer."),
                    self._stream_chunk(usage=SimpleNamespace(prompt_tokens=2, completion_tokens=4, total_tokens=6)),
                ]
            ),
        ]

        with patch("agent.client.chat.completions.create", side_effect=responses), patch(
            "agent.search_web_tool",
            return_value=[{"title": "Test", "url": "https://example.com", "snippet": "Snippet"}],
        ):
            events = list(run_agent_stream([{"role": "user", "content": "Test"}], "deepseek-reasoner", 2, ["search_web"]))

        reasoning_deltas = [event["text"] for event in events if event["type"] == "reasoning_delta"]
        self.assertEqual(reasoning_deltas, ["First reasoning block.", "\n\n", "Second reasoning block."])

    def test_run_agent_stream_replays_reasoning_into_next_tool_step(self):
        responses = [
            iter(
                [
                    self._stream_chunk(reasoning="Need current info before I answer."),
                    self._tool_call_chunk("search_web", {"queries": ["latest update"]}),
                    self._stream_chunk(usage=SimpleNamespace(prompt_tokens=2, completion_tokens=3, total_tokens=5)),
                ]
            ),
            iter(
                [
                    self._stream_chunk(content="Final answer."),
                    self._stream_chunk(usage=SimpleNamespace(prompt_tokens=2, completion_tokens=4, total_tokens=6)),
                ]
            ),
        ]

        with patch("agent.client.chat.completions.create", side_effect=responses) as mocked_create, patch(
            "agent.search_web_tool",
            return_value=[{"title": "Test", "url": "https://example.com", "snippet": "Snippet"}],
        ):
            events = list(run_agent_stream([{"role": "user", "content": "Test"}], "deepseek-reasoner", 2, ["search_web"]))

        self.assertIn({"type": "answer_delta", "text": "Final answer."}, events)
        second_call_messages = mocked_create.call_args_list[1].kwargs["messages"]
        replay_message = next(
            (
                message
                for message in second_call_messages
                if message.get("role") == "system" and "[AGENT REASONING CONTEXT]" in message.get("content", "")
            ),
            None,
        )
        self.assertIsNotNone(replay_message)
        self.assertIn("Need current info before I answer.", replay_message["content"])
        self.assertIn("planned tools = search_web", replay_message["content"])

    def test_run_agent_stream_skips_reasoning_replay_when_reasoning_empty(self):
        responses = [
            iter(
                [
                    self._tool_call_chunk("search_web", {"queries": ["latest update"]}),
                    self._stream_chunk(usage=SimpleNamespace(prompt_tokens=2, completion_tokens=3, total_tokens=5)),
                ]
            ),
            iter(
                [
                    self._stream_chunk(content="Final answer."),
                    self._stream_chunk(usage=SimpleNamespace(prompt_tokens=2, completion_tokens=4, total_tokens=6)),
                ]
            ),
        ]

        with patch("agent.client.chat.completions.create", side_effect=responses) as mocked_create, patch(
            "agent.search_web_tool",
            return_value=[{"title": "Test", "url": "https://example.com", "snippet": "Snippet"}],
        ):
            events = list(run_agent_stream([{"role": "user", "content": "Test"}], "deepseek-chat", 2, ["search_web"]))

        self.assertIn({"type": "answer_delta", "text": "Final answer."}, events)
        second_call_messages = mocked_create.call_args_list[1].kwargs["messages"]
        replay_messages = [
            message
            for message in second_call_messages
            if message.get("role") == "system" and "[AGENT REASONING CONTEXT]" in message.get("content", "")
        ]
        self.assertEqual(replay_messages, [])

    def test_run_agent_stream_replays_reasoning_alongside_blocker_memory(self):
        responses = [
            iter(
                [
                    self._stream_chunk(reasoning="I should search first, then summarize whatever I find."),
                    self._tool_call_chunk("search_web", {"queries": ["latest update"]}),
                    self._stream_chunk(usage=SimpleNamespace(prompt_tokens=2, completion_tokens=3, total_tokens=5)),
                ]
            ),
            iter(
                [
                    self._stream_chunk(content="Fallback answer."),
                    self._stream_chunk(usage=SimpleNamespace(prompt_tokens=2, completion_tokens=4, total_tokens=6)),
                ]
            ),
        ]

        with patch("agent.client.chat.completions.create", side_effect=responses) as mocked_create, patch(
            "agent.search_web_tool",
            side_effect=RuntimeError("search backend unavailable"),
        ):
            events = list(run_agent_stream([{"role": "user", "content": "Find current info"}], "deepseek-reasoner", 2, ["search_web"]))

        self.assertIn({"type": "answer_delta", "text": "Fallback answer."}, events)
        second_call_messages = mocked_create.call_args_list[1].kwargs["messages"]
        replay_message = next(
            (
                message
                for message in second_call_messages
                if message.get("role") == "system" and "[AGENT REASONING CONTEXT]" in message.get("content", "")
            ),
            None,
        )
        blocker_message = next(
            (
                message
                for message in second_call_messages
                if message.get("role") == "system" and "AGENT WORKING MEMORY" in message.get("content", "")
            ),
            None,
        )
        self.assertIsNotNone(replay_message)
        self.assertIsNotNone(blocker_message)
        self.assertIn("I should search first", replay_message["content"])
        self.assertIn("search backend unavailable", blocker_message["content"])

    def test_estimate_message_breakdown_counts_reasoning_replay_as_internal_state(self):
        content = "[AGENT REASONING CONTEXT]\n\nPrior reasoning"
        breakdown = _estimate_message_breakdown({"role": "system", "content": content})

        self.assertEqual(breakdown, {"internal_state": estimate_text_tokens(content)})

    def test_run_agent_stream_reports_missing_final_content_after_retry_budget(self):
        responses = [
            iter(
                [
                    self._stream_chunk(reasoning="First reasoning pass."),
                    self._stream_chunk(usage=SimpleNamespace(prompt_tokens=2, completion_tokens=3, total_tokens=5)),
                ]
            ),
            iter(
                [
                    self._stream_chunk(reasoning="Second reasoning pass."),
                    self._stream_chunk(usage=SimpleNamespace(prompt_tokens=2, completion_tokens=3, total_tokens=5)),
                ]
            ),
        ]

        with patch("agent.client.chat.completions.create", side_effect=responses):
            events = list(run_agent_stream([{"role": "user", "content": "Test"}], "deepseek-reasoner", 1, []))

        self.assertIn({"type": "reasoning_delta", "text": "First reasoning pass."}, events)
        self.assertIn({"type": "reasoning_delta", "text": "Second reasoning pass."}, events)
        self.assertIn(
            {
                "type": "tool_error",
                "step": 1,
                "tool": "agent",
                "error": "The model still did not provide a final answer in assistant content.",
            },
            events,
        )
        self.assertIn({"type": "answer_delta", "text": FINAL_ANSWER_MISSING_TEXT}, events)
        leaked_reasoning = [event for event in events if event["type"] == "answer_delta" and "reasoning pass" in event["text"]]
        self.assertEqual(leaked_reasoning, [])

    def test_run_agent_stream_injects_blocker_memory_after_tool_failure(self):
        responses = [
            iter(
                [
                    self._tool_call_chunk("search_web", {"queries": ["bad query"]}),
                    self._stream_chunk(usage=SimpleNamespace(prompt_tokens=2, completion_tokens=3, total_tokens=5)),
                ]
            ),
            iter(
                [
                    self._stream_chunk(content="Fallback answer."),
                    self._stream_chunk(usage=SimpleNamespace(prompt_tokens=2, completion_tokens=4, total_tokens=6)),
                ]
            ),
        ]

        with patch("agent.client.chat.completions.create", side_effect=responses) as mocked_create, patch(
            "agent.search_web_tool",
            side_effect=RuntimeError("search backend unavailable"),
        ):
            events = list(run_agent_stream([{"role": "user", "content": "Find current info"}], "deepseek-chat", 2, ["search_web"]))

        self.assertIn({"type": "answer_delta", "text": "Fallback answer."}, events)
        second_call_messages = mocked_create.call_args_list[1].kwargs["messages"]
        blocker_message = next((message for message in second_call_messages if message["role"] == "system" and "AGENT WORKING MEMORY" in message["content"]), None)
        self.assertIsNotNone(blocker_message)
        self.assertIn("search backend unavailable", blocker_message["content"])
        self.assertIn("Failed paths to avoid repeating", blocker_message["content"])

    def test_run_agent_stream_enforces_per_tool_step_limit(self):
        responses = [
            iter(
                [
                    self._tool_call_chunk("fetch_url", {"url": "https://example.com/1"}, call_id="tool-call-1", index=0),
                    self._tool_call_chunk("fetch_url", {"url": "https://example.com/2"}, call_id="tool-call-2", index=1),
                    self._tool_call_chunk("fetch_url", {"url": "https://example.com/3"}, call_id="tool-call-3", index=2),
                    self._stream_chunk(usage=SimpleNamespace(prompt_tokens=2, completion_tokens=3, total_tokens=5)),
                ]
            ),
            iter(
                [
                    self._stream_chunk(content="Final answer."),
                    self._stream_chunk(usage=SimpleNamespace(prompt_tokens=2, completion_tokens=4, total_tokens=6)),
                ]
            ),
        ]

        with patch("agent.client.chat.completions.create", side_effect=responses), patch(
            "agent.fetch_url_tool",
            return_value={"url": "https://example.com", "title": "Example", "content": "Body"},
        ) as mocked_fetch:
            events = list(run_agent_stream([{"role": "user", "content": "Fetch several pages"}], "deepseek-chat", 2, ["fetch_url"]))

        self.assertEqual(mocked_fetch.call_count, 2)
        per_tool_errors = [
            event for event in events
            if event["type"] == "tool_error" and event["tool"] == "fetch_url" and "Per-tool step limit reached" in event["error"]
        ]
        self.assertEqual(len(per_tool_errors), 1)
        self.assertIn({"type": "answer_delta", "text": "Final answer."}, events)

    def test_run_agent_stream_uses_max_steps_as_default_per_tool_budget(self):
        responses = [
            iter(
                [
                    self._tool_call_chunk("append_scratchpad", {"notes": ["note 1"]}, call_id="tool-call-1"),
                    self._stream_chunk(usage=SimpleNamespace(prompt_tokens=2, completion_tokens=3, total_tokens=5)),
                ]
            ),
            iter(
                [
                    self._tool_call_chunk("append_scratchpad", {"notes": ["note 2"]}, call_id="tool-call-2"),
                    self._stream_chunk(usage=SimpleNamespace(prompt_tokens=2, completion_tokens=3, total_tokens=5)),
                ]
            ),
            iter(
                [
                    self._tool_call_chunk("append_scratchpad", {"notes": ["note 3"]}, call_id="tool-call-3"),
                    self._stream_chunk(usage=SimpleNamespace(prompt_tokens=2, completion_tokens=3, total_tokens=5)),
                ]
            ),
            iter(
                [
                    self._tool_call_chunk("append_scratchpad", {"notes": ["note 4"]}, call_id="tool-call-4"),
                    self._stream_chunk(usage=SimpleNamespace(prompt_tokens=2, completion_tokens=3, total_tokens=5)),
                ]
            ),
            iter(
                [
                    self._tool_call_chunk("append_scratchpad", {"notes": ["note 5"]}, call_id="tool-call-5"),
                    self._stream_chunk(usage=SimpleNamespace(prompt_tokens=2, completion_tokens=3, total_tokens=5)),
                ]
            ),
            iter(
                [
                    self._tool_call_chunk("append_scratchpad", {"notes": ["note 6"]}, call_id="tool-call-6"),
                    self._stream_chunk(usage=SimpleNamespace(prompt_tokens=2, completion_tokens=3, total_tokens=5)),
                ]
            ),
            iter(
                [
                    self._stream_chunk(content="Final answer."),
                    self._stream_chunk(usage=SimpleNamespace(prompt_tokens=2, completion_tokens=4, total_tokens=6)),
                ]
            ),
        ]

        with patch("agent.client.chat.completions.create", side_effect=responses), patch(
            "agent.append_to_scratchpad",
            side_effect=[
                ({"status": "appended", "scratchpad": "note 1"}, "Scratchpad updated"),
                ({"status": "appended", "scratchpad": "note 1\nnote 2"}, "Scratchpad updated"),
                ({"status": "appended", "scratchpad": "note 1\nnote 2\nnote 3"}, "Scratchpad updated"),
                ({"status": "appended", "scratchpad": "note 1\nnote 2\nnote 3\nnote 4"}, "Scratchpad updated"),
                ({"status": "appended", "scratchpad": "note 1\nnote 2\nnote 3\nnote 4\nnote 5"}, "Scratchpad updated"),
                ({"status": "appended", "scratchpad": "note 1\nnote 2\nnote 3\nnote 4\nnote 5\nnote 6"}, "Scratchpad updated"),
            ],
        ) as mocked_append:
            events = list(
                run_agent_stream(
                    [{"role": "user", "content": "Remember these notes"}],
                    "deepseek-chat",
                    6,
                    ["append_scratchpad"],
                )
            )

        self.assertEqual(mocked_append.call_count, 6)
        per_tool_errors = [
            event
            for event in events
            if event["type"] == "tool_error" and event["tool"] == "append_scratchpad" and "Per-tool step limit reached" in event["error"]
        ]
        self.assertEqual(per_tool_errors, [])
        self.assertIn({"type": "answer_delta", "text": "Final answer."}, events)

    def test_run_agent_stream_stops_after_api_error_without_duplicate_retry(self):
        responses = [RuntimeError("Error code: 400 - {'error': {'message': 'Invalid consecutive assistant message at message index 2', 'type': 'invalid_request_error'}}")]

        with patch("agent.client.chat.completions.create", side_effect=responses):
            events = list(run_agent_stream([{"role": "user", "content": "Test"}], "deepseek-reasoner", 2, []))

        api_errors = [event for event in events if event["type"] == "tool_error" and event["tool"] == "api"]
        self.assertEqual(len(api_errors), 1)
        self.assertIn({"type": "answer_delta", "text": FINAL_ANSWER_ERROR_TEXT}, events)

    def test_extract_html_cleans_noise_and_whitespace(self):
        html = """
        <html>
          <head>
            <title>Example Title</title>
            <style>.hidden { display: none; }</style>
            <script>console.log('ignore')</script>
          </head>
          <body>
            <nav>Navigation</nav>
            <main>
              <h1>  Heading\u200b </h1>
              <h1>  Heading\u200b </h1>
              <p>First line&nbsp;&nbsp; here.</p>
              <div>\n\nSecond   line\t\tcontinues.\n</div>
              <div>-----</div>
            </main>
          </body>
        </html>
        """

        result = _extract_html(html, "https://example.com/page")

        self.assertEqual(result["title"], "Example Title")
        self.assertEqual(result["content_format"], "html")
        self.assertTrue(result["content"].startswith("Heading"))
        self.assertTrue(result["content"].startswith("Heading"))
        self.assertIn("First line here.", result["content"])
        self.assertIn("Second line continues.", result["content"])
        self.assertNotIn("Navigation", result["content"])
        self.assertNotIn("console.log", result["content"])
        self.assertNotIn("-----", result["content"])
        self.assertNotIn("\u200b", result["content"])

        def test_extract_html_falls_back_to_meta_noscript_and_json_ld_when_body_is_thin(self):
                html = """
                <html>
                    <head>
                        <title>Rates Page</title>
                        <title>Rates Page</title>
                        <meta name="description" content="Current market summary for dollars and euros.">
                        <script type="application/ld+json">
                            {
                                "headline": "Live USD/TRY and EUR/TRY data",
                                "description": "Current exchange-rate information on the open market."
                            }
                        </script>
                    </head>
                    <body>
                        <main><div></div></main>
                        <noscript>Fallback rate summary shown without JavaScript.</noscript>
                    </body>
                </html>
                """

                result = _extract_html(html, "https://example.com/rates")

                self.assertEqual(result["title"], "Rates Page")
                self.assertEqual(result["title"], "Rates Page")
                self.assertIn("Current market summary for dollars and euros.", result["content"])
                self.assertIn("Fallback rate summary shown without JavaScript.", result["content"])
                self.assertIn("Live USD/TRY and EUR/TRY data", result["content"])
                self.assertIn("Live USD/TRY and EUR/TRY data", result["content"])

    def test_fetch_url_tool_recovers_partial_chunked_content(self):
        class FakeResponse:
            def __init__(self):
                self.headers = {"Content-Type": "text/html; charset=utf-8"}
                self.url = "https://example.com/page"
                self.encoding = "utf-8"
                self.status_code = 200

            def iter_content(self, chunk_size=8192):
                yield b"<html><head><title>Example</title></head><body><main>"
                yield b"Recovered partial content"
                raise http_requests.exceptions.ChunkedEncodingError("incomplete chunked read")

        class FakeSession:
            def __init__(self):
                self.max_redirects = 0
                self.trust_env = False
                self.proxies = {}

            def get(self, *args, **kwargs):
                return FakeResponse()

            def close(self):
                return None

        with patch("web_tools._is_safe_url", return_value=(True, "")), patch(
            "web_tools.cache_get",
            return_value=None,
        ), patch("web_tools.cache_set") as mocked_cache_set, patch(
            "web_tools.get_proxy_candidates",
            return_value=[None],
        ), patch("web_tools.http_requests.Session", return_value=FakeSession()):
            result = fetch_url_tool("https://example.com/page")

        self.assertEqual(result["title"], "Example")
        self.assertIn("Recovered partial content", result["content"])
        self.assertTrue(result["partial_content"])
        self.assertIn("partial page content was recovered", result["fetch_warning"])
        self.assertTrue(mocked_cache_set.called)

    def test_fetch_url_tool_uses_proxies_before_direct_fallback(self):
        attempts = []

        class FakeResponse:
            def __init__(self):
                self.headers = {"Content-Type": "text/plain; charset=utf-8"}
                self.url = "https://example.com/page"
                self.encoding = "utf-8"
                self.status_code = 200

            def iter_content(self, chunk_size=8192):
                yield b"Recovered without proxy"

        class FakeSession:
            def __init__(self):
                self.max_redirects = 0
                self.trust_env = False
                self.proxies = {}

            def get(self, *args, **kwargs):
                proxy = self.proxies.get("https") if self.proxies else None
                attempts.append(proxy)
                if proxy:
                    raise http_requests.exceptions.Timeout("proxy failed")
                return FakeResponse()

            def close(self):
                return None

        with patch("web_tools._is_safe_url", return_value=(True, "")), patch(
            "web_tools.cache_get",
            return_value=None,
        ), patch("web_tools.cache_set"), patch(
            "web_tools.get_proxy_candidates",
            return_value=["http://proxy.example:8080", None],
        ) as mocked_candidates, patch("web_tools.http_requests.Session", side_effect=FakeSession):
            result = fetch_url_tool("https://example.com/page")

        mocked_candidates.assert_called_once_with(include_direct_fallback=True)
        self.assertEqual(attempts, ["http://proxy.example:8080", None])
        self.assertEqual(result["content"], "Recovered without proxy")

    def test_fetch_url_tool_retries_with_alternate_headers_when_first_response_is_thin(self):
        header_attempts = []

        class FakeResponse:
            def __init__(self, html):
                self.headers = {"Content-Type": "text/html; charset=utf-8"}
                self.url = "https://example.com/page"
                self.encoding = "utf-8"
                self.status_code = 200
                self._html = html

            def iter_content(self, chunk_size=8192):
                yield self._html.encode("utf-8")

        class FakeSession:
            def __init__(self):
                self.max_redirects = 0
                self.trust_env = False
                self.proxies = {}

            def get(self, *args, **kwargs):
                headers = kwargs.get("headers") or {}
                header_attempts.append(headers.get("Cache-Control"))
                if len(header_attempts) == 1:
                    return FakeResponse("<html><body><main>ok</main></body></html>")
                return FakeResponse(
                    """
                    <html><head><title>Rates</title><meta name=\"description\" content=\"Current USD and EUR rates are listed here.\"></head>
                    <body><main><div>Sufficient fallback content and current market summary are included here.</div></main></body></html>
                    """
                )

            def close(self):
                return None

        with patch("web_tools._is_safe_url", return_value=(True, "")), patch(
            "web_tools.cache_get",
            return_value=None,
        ), patch("web_tools.cache_set") as mocked_cache_set, patch(
            "web_tools.get_proxy_candidates",
            return_value=[None],
        ), patch("web_tools.http_requests.Session", side_effect=FakeSession):
            result = fetch_url_tool("https://example.com/page")

        self.assertEqual(header_attempts[:2], ["max-age=0", "no-cache"])
        self.assertIn("Current USD and EUR rates are listed here.", result["content"])
        self.assertTrue(mocked_cache_set.called)

    def test_fetch_url_tool_retries_without_ssl_verification_on_cert_failure(self):
        verify_values = []

        class FakeResponse:
            def __init__(self):
                self.headers = {"Content-Type": "text/html; charset=utf-8"}
                self.url = "https://example.com/page"
                self.encoding = "utf-8"
                self.status_code = 200

            def iter_content(self, chunk_size=8192):
                yield b"<html><head><title>Example</title></head><body><main>Trusted content</main></body></html>"

        class FakeSession:
            def __init__(self):
                self.max_redirects = 0
                self.trust_env = False
                self.proxies = {}

            def get(self, *args, **kwargs):
                verify_values.append(kwargs.get("verify", True))
                if kwargs.get("verify", True):
                    raise http_requests.exceptions.SSLError(
                        "[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed"
                    )
                return FakeResponse()

            def close(self):
                return None

        with patch("web_tools._is_safe_url", return_value=(True, "")), patch(
            "web_tools.cache_get",
            return_value=None,
        ), patch("web_tools.cache_set") as mocked_cache_set, patch(
            "web_tools.get_proxy_candidates",
            return_value=[None],
        ), patch("web_tools.http_requests.Session", side_effect=FakeSession):
            result = fetch_url_tool("https://example.com/page")

        self.assertGreaterEqual(len(verify_values), 2)
        self.assertEqual(verify_values[:2], [True, False])
        self.assertEqual(verify_values[::2], [True] * (len(verify_values) // 2))
        self.assertEqual(verify_values[1::2], [False] * (len(verify_values) // 2))
        self.assertEqual(result["title"], "Example")
        self.assertTrue(result["ssl_verification_bypassed"])
        self.assertIn("without certificate verification", result["fetch_warning"])
        self.assertTrue(mocked_cache_set.called)

    def test_fetch_url_tool_does_not_cache_empty_blocked_page(self):
        class FakeResponse:
            def __init__(self):
                self.headers = {"Content-Type": "text/html; charset=utf-8"}
                self.url = "https://example.com/blocked"
                self.encoding = "utf-8"
                self.status_code = 403

            def iter_content(self, chunk_size=8192):
                yield b"<html><head><title>Forbidden</title></head><body><main></main></body></html>"

        class FakeSession:
            def __init__(self):
                self.max_redirects = 0
                self.trust_env = False
                self.proxies = {}

            def get(self, *args, **kwargs):
                return FakeResponse()

            def close(self):
                return None

        with patch("web_tools._is_safe_url", return_value=(True, "")), patch(
            "web_tools.cache_get",
            return_value=None,
        ), patch("web_tools.cache_set") as mocked_cache_set, patch(
            "web_tools.get_proxy_candidates",
            return_value=[None],
        ), patch("web_tools.http_requests.Session", side_effect=FakeSession):
            result = fetch_url_tool("https://example.com/blocked")

        self.assertFalse(mocked_cache_set.called)
        self.assertEqual(result["error"], "HTTP 403")
        self.assertEqual(result["content"], "")

    def test_search_web_uses_proxies_before_direct_fallback(self):
        attempts = []

        class FakeDDGS:
            def __init__(self, proxy=None):
                self.proxy = proxy

            def __enter__(self):
                attempts.append(self.proxy)
                if self.proxy:
                    raise RuntimeError("proxy failed")
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def text(self, query, max_results=5):
                return [{"title": "Result", "href": "https://example.com", "body": "Snippet"}]

        with patch("web_tools.cache_get", return_value=None), patch("web_tools.cache_set"), patch(
            "web_tools.get_proxy_candidates",
            return_value=["http://proxy.example:8080", None],
        ) as mocked_candidates, patch("web_tools.DDGS", FakeDDGS):
            result = search_web_tool(["example"])

        mocked_candidates.assert_called_once_with(include_direct_fallback=True)
        self.assertEqual(attempts, ["http://proxy.example:8080", None])
        self.assertEqual(result[0]["url"], "https://example.com")

    def test_search_news_ddgs_uses_proxies_before_direct_fallback(self):
        attempts = []

        class FakeDDGS:
            def __init__(self, proxy=None):
                self.proxy = proxy

            def __enter__(self):
                attempts.append(self.proxy)
                if self.proxy:
                    raise RuntimeError("proxy failed")
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def news(self, query, region=None, safesearch=None, timelimit=None, max_results=5):
                return [{"title": "News", "url": "https://example.com/news", "date": "today", "source": "Example"}]

        with patch("web_tools.cache_get", return_value=None), patch("web_tools.cache_set"), patch(
            "web_tools.get_proxy_candidates",
            return_value=["http://proxy.example:8080", None],
        ) as mocked_candidates, patch("web_tools.DDGS", FakeDDGS):
            result = search_news_ddgs_tool(["example"])

        mocked_candidates.assert_called_once_with(include_direct_fallback=True)
        self.assertEqual(attempts, ["http://proxy.example:8080", None])
        self.assertEqual(result[0]["link"], "https://example.com/news")

    def test_search_news_google_uses_proxies_before_direct_fallback(self):
        attempts = []

        class FakeResponse:
            content = b"""<?xml version=\"1.0\"?><rss><channel><item><title>News - Example</title><link>https://example.com/news</link><pubDate>today</pubDate><source>Example</source></item></channel></rss>"""

            def raise_for_status(self):
                return None

        def fake_get(url, headers=None, timeout=None, proxies=None):
            proxy = (proxies or {}).get("https") if proxies else None
            attempts.append(proxy)
            if proxy:
                raise RuntimeError("proxy failed")
            return FakeResponse()

        with patch("web_tools.cache_get", return_value=None), patch("web_tools.cache_set"), patch(
            "web_tools.get_proxy_candidates",
            return_value=["http://proxy.example:8080", None],
        ) as mocked_candidates, patch("web_tools.http_requests.get", side_effect=fake_get):
            result = search_news_google_tool(["example"])

        mocked_candidates.assert_called_once_with(include_direct_fallback=True)
        self.assertEqual(attempts, ["http://proxy.example:8080", None])
        self.assertEqual(result[0]["link"], "https://example.com/news")

    def test_run_agent_stream_clips_long_fetch_results_before_transcript(self):
        responses = [
            iter(
                [
                    self._tool_call_chunk("fetch_url", {"url": "https://example.com/long"}),
                    self._stream_chunk(usage=SimpleNamespace(prompt_tokens=3, completion_tokens=5, total_tokens=8)),
                ]
            ),
            iter(
                [
                    self._stream_chunk(content="Final answer."),
                    self._stream_chunk(usage=SimpleNamespace(prompt_tokens=4, completion_tokens=6, total_tokens=10)),
                ]
            ),
        ]
        long_content = "\n\n".join(
            [
                "Overview block with broad context and repeated details. " * 12,
                "Focus block about integration strategy and cleanup pipeline. " * 12,
                "Another block covering token threshold handling and retrieval summaries. " * 12,
                "Final block with implementation notes and metadata persistence. " * 12,
            ]
        )

        with patch("agent.client.chat.completions.create", side_effect=responses) as mocked_create, patch(
            "agent.fetch_url_tool",
            return_value={
                "url": "https://example.com/long",
                "title": "Long Example",
                "content": long_content,
                "status": 200,
                "content_format": "html",
                "cleanup_applied": True,
            },
        ), patch("agent.FETCH_SUMMARY_MAX_CHARS", 5000):
            events = list(
                run_agent_stream(
                    [{"role": "user", "content": "Focus on cleanup and token handling"}],
                    "deepseek-chat",
                    2,
                    ["fetch_url"],
                    fetch_url_token_threshold=50,
                )
            )

        tool_capture_event = next(event for event in events if event["type"] == "tool_capture")
        stored_result = tool_capture_event["tool_results"][0]
        self.assertEqual(stored_result["tool_name"], "fetch_url")
        self.assertEqual(stored_result["content_mode"], "clipped_text")
        self.assertIn("cleaned and clipped", stored_result["summary_notice"])
        self.assertIn("raw_content", stored_result)
        self.assertEqual(stored_result["raw_content"], long_content.strip())

        second_call_messages = mocked_create.call_args_list[1].kwargs["messages"]
        transcript_content = second_call_messages[-1]["content"]
        self.assertIn("TOOL EXECUTION RESULTS", transcript_content)
        self.assertIn("fetch_url", transcript_content)
        self.assertIn("OK", transcript_content)

    def test_run_agent_stream_marks_fetch_failures_clearly_in_transcript(self):
        responses = [
            iter(
                [
                    self._tool_call_chunk("fetch_url", {"url": "https://example.com/blocked"}),
                    self._stream_chunk(usage=SimpleNamespace(prompt_tokens=3, completion_tokens=5, total_tokens=8)),
                ]
            ),
            iter(
                [
                    self._stream_chunk(content="Blocked answer."),
                    self._stream_chunk(usage=SimpleNamespace(prompt_tokens=4, completion_tokens=6, total_tokens=10)),
                ]
            ),
        ]

        with patch("agent.client.chat.completions.create", side_effect=responses) as mocked_create, patch(
            "agent.fetch_url_tool",
            return_value={
                "url": "https://example.com/blocked",
                "content": "",
                "error": "HTTP 403",
                "status": 403,
            },
        ):
            events = list(run_agent_stream([{"role": "user", "content": "Fetch it"}], "deepseek-chat", 2, ["fetch_url"]))

        tool_result_event = next(event for event in events if event["type"] == "tool_result")
        self.assertIn("Fetch failed", tool_result_event["summary"])

        tool_capture_event = next(event for event in events if event["type"] == "tool_capture")
        stored_result = tool_capture_event["tool_results"][0]
        self.assertEqual(stored_result["fetch_outcome"], "error")
        self.assertIn("fetch_url already attempted this URL", stored_result["fetch_diagnostic"])

        second_call_messages = mocked_create.call_args_list[1].kwargs["messages"]
        transcript_content = second_call_messages[-1]["content"]
        self.assertIn("TOOL EXECUTION RESULTS", transcript_content)
        self.assertIn("source of truth", transcript_content)
        self.assertIn("fetch_url", transcript_content)
        self.assertIn("FAILED", transcript_content)

    def test_run_agent_stream_marks_partial_fetch_as_already_attempted(self):
        responses = [
            iter(
                [
                    self._tool_call_chunk("fetch_url", {"url": "https://example.com/page"}),
                    self._stream_chunk(usage=SimpleNamespace(prompt_tokens=3, completion_tokens=5, total_tokens=8)),
                ]
            ),
            iter(
                [
                    self._stream_chunk(content="Partial answer."),
                    self._stream_chunk(usage=SimpleNamespace(prompt_tokens=4, completion_tokens=6, total_tokens=10)),
                ]
            ),
        ]

        with patch("agent.client.chat.completions.create", side_effect=responses) as mocked_create, patch(
            "agent.fetch_url_tool",
            return_value={
                "url": "https://example.com/page",
                "title": "Example",
                "content": "Recovered partial content from the page body for analysis.",
                "status": 200,
                "partial_content": True,
                "fetch_warning": "Connection ended early; partial page content was recovered",
                "content_format": "html",
            },
        ):
            events = list(run_agent_stream([{"role": "user", "content": "Fetch it"}], "deepseek-chat", 2, ["fetch_url"]))

        tool_result_event = next(event for event in events if event["type"] == "tool_result")
        self.assertIn("Partial page content extracted", tool_result_event["summary"])

        tool_capture_event = next(event for event in events if event["type"] == "tool_capture")
        stored_result = tool_capture_event["tool_results"][0]
        self.assertEqual(stored_result["fetch_outcome"], "partial_content")
        self.assertIn("Do not repeat the same fetch_url call", stored_result["fetch_diagnostic"])

        second_call_messages = mocked_create.call_args_list[1].kwargs["messages"]
        transcript_content = second_call_messages[-1]["content"]
        self.assertIn("TOOL EXECUTION RESULTS", transcript_content)
        self.assertIn("fetch_url", transcript_content)
        self.assertIn("OK", transcript_content)

    def test_run_agent_stream_logs_duplicate_fetch_url_calls(self):
        responses = [
            iter(
                [
                    self._tool_call_chunk("fetch_url", {"url": "https://example.com/page"}),
                    self._stream_chunk(usage=SimpleNamespace(prompt_tokens=3, completion_tokens=5, total_tokens=8)),
                ]
            ),
            iter(
                [
                    self._tool_call_chunk("fetch_url", {"url": "https://example.com/page"}),
                    self._stream_chunk(usage=SimpleNamespace(prompt_tokens=3, completion_tokens=5, total_tokens=8)),
                ]
            ),
            iter(
                [
                    self._stream_chunk(content="Final answer."),
                    self._stream_chunk(usage=SimpleNamespace(prompt_tokens=4, completion_tokens=6, total_tokens=10)),
                ]
            ),
        ]

        with patch("agent.client.chat.completions.create", side_effect=responses) as mocked_create, patch(
            "agent.fetch_url_tool",
            return_value={
                "url": "https://example.com/page",
                "title": "Example",
                "content": "Fetched page body with enough detail to reuse without fetching again.",
                "status": 200,
                "content_format": "html",
            },
        ) as mocked_fetch, patch("agent._trace_agent_event") as mocked_trace:
            events = list(run_agent_stream([{"role": "user", "content": "Fetch it"}], "deepseek-chat", 3, ["fetch_url"]))

        mocked_fetch.assert_called_with("https://example.com/page")
        self.assertEqual(mocked_fetch.call_count, 1)
        tool_result_summaries = [event["summary"] for event in events if event["type"] == "tool_result"]
        self.assertIn("Page content extracted", tool_result_summaries[0])
        self.assertIn("Page content extracted", tool_result_summaries[1])

        third_call_messages = mocked_create.call_args_list[2].kwargs["messages"]
        transcript_content = third_call_messages[-1]["content"]
        self.assertIn("TOOL EXECUTION RESULTS", transcript_content)
        self.assertIn("fetch_url", transcript_content)
        duplicate_logs = [call for call in mocked_trace.call_args_list if call.args and call.args[0] == "duplicate_fetch_attempt"]
        self.assertEqual(len(duplicate_logs), 1)
        self.assertEqual(duplicate_logs[0].kwargs["url"], "https://example.com/page")

    def test_higher_clip_aggressiveness_keeps_less_fetch_content(self):
        result = {
            "url": "https://example.com/long",
            "title": "Long Example",
            "content": "A" * 12000,
            "status": 200,
            "content_format": "html",
            "cleanup_applied": True,
        }

        from agent import _prepare_fetch_result_for_model

        less_aggressive = _prepare_fetch_result_for_model(
            result,
            fetch_url_token_threshold=1000,
            fetch_url_clip_aggressiveness=10,
        )
        more_aggressive = _prepare_fetch_result_for_model(
            result,
            fetch_url_token_threshold=1000,
            fetch_url_clip_aggressiveness=90,
        )

        self.assertGreater(len(less_aggressive["content"]), len(more_aggressive["content"]))

    def test_run_agent_stream_executes_native_tool_call_without_content_fallback(self):
        responses = [
            iter(
                [
                    self._tool_call_chunk("search_web", {"queries": ["x"]}),
                    self._stream_chunk(usage=SimpleNamespace(prompt_tokens=2, completion_tokens=3, total_tokens=5)),
                ]
            ),
            iter(
                [
                    self._stream_chunk(content="Recovered answer"),
                    self._stream_chunk(usage=SimpleNamespace(prompt_tokens=2, completion_tokens=4, total_tokens=6)),
                ]
            ),
        ]

        with patch("agent.client.chat.completions.create", side_effect=responses), patch(
            "agent.search_web_tool",
            return_value=[{"title": "Test", "url": "https://example.com", "snippet": "Snippet"}],
        ):
            events = list(run_agent_stream([{"role": "user", "content": "Test"}], "deepseek-chat", 2, ["search_web"]))

        self.assertIn("step_update", [event["type"] for event in events])
        self.assertIn("tool_result", [event["type"] for event in events])
        self.assertIn({"type": "answer_delta", "text": "Recovered answer"}, events)

    def test_run_agent_stream_streams_native_pre_tool_text_live_and_preserves_history(self):
        captured_calls = []

        def fake_create(model, messages, stream, stream_options=None, tools=None, tool_choice=None):
            captured_calls.append(list(messages))
            call_index = len(captured_calls)
            if call_index == 1:
                return iter(
                    [
                        self._stream_chunk(content="Okay, I will check now."),
                        self._tool_call_chunk("search_web", {"queries": ["test"]}),
                        self._stream_chunk(
                            usage=SimpleNamespace(prompt_tokens=5, completion_tokens=20, total_tokens=25)
                        ),
                    ]
                )
            else:
                return iter(
                    [
                        self._stream_chunk(content="Here is the result."),
                        self._stream_chunk(
                            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15)
                        ),
                    ]
                )

        with patch("agent.client.chat.completions.create", side_effect=fake_create), patch(
            "agent.search_web_tool",
            return_value=[{"title": "R", "url": "https://example.com", "snippet": "S"}],
        ):
            events = list(run_agent_stream([{"role": "user", "content": "Test"}], "deepseek-chat", 2, ["search_web"]))

        answer_texts = [event["text"] for event in events if event["type"] == "answer_delta"]
        combined_answer = "".join(answer_texts)
        self.assertIn("Okay, I will check now.", combined_answer)
        self.assertIn("Here is the result.", combined_answer)

        self.assertGreaterEqual(len(captured_calls), 2, "Expected at least 2 model calls")
        second_call_messages = captured_calls[1]
        assistant_messages = [m for m in second_call_messages if m.get("role") == "assistant"]
        pre_tool_contents = [m["content"] for m in assistant_messages if "Okay, I will check now" in m.get("content", "")]
        self.assertTrue(
            len(pre_tool_contents) > 0,
            "Pre-tool narrative text must be injected as an assistant message for the next turn",
        )

    def test_run_agent_stream_emits_short_pre_tool_text_without_waiting_for_threshold(self):
        responses = [
            iter(
                [
                    self._stream_chunk(content="Hi"),
                    self._tool_call_chunk("search_web", {"queries": ["test"]}),
                    self._stream_chunk(
                        usage=SimpleNamespace(prompt_tokens=5, completion_tokens=20, total_tokens=25)
                    ),
                ]
            ),
            iter(
                [
                    self._stream_chunk(content="Final answer."),
                    self._stream_chunk(
                        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15)
                    ),
                ]
            ),
        ]

        with patch("agent.client.chat.completions.create", side_effect=responses), patch(
            "agent.search_web_tool",
            return_value=[{"title": "R", "url": "https://example.com", "snippet": "S"}],
        ):
            events = list(run_agent_stream([{"role": "user", "content": "Test"}], "deepseek-chat", 2, ["search_web"]))

        answer_deltas = [event["text"] for event in events if event["type"] == "answer_delta"]
        self.assertEqual(answer_deltas[0], "Hi")
        self.assertIn("Final answer.", answer_deltas)

    def test_run_agent_stream_streams_plain_answer_chunks_live(self):
        responses = [
            iter(
                [
                    self._stream_chunk(content="Hello "),
                    self._stream_chunk(content="world"),
                    self._stream_chunk(usage=SimpleNamespace(prompt_tokens=2, completion_tokens=3, total_tokens=5)),
                ]
            )
        ]

        with patch("agent.client.chat.completions.create", side_effect=responses):
            events = list(run_agent_stream([{"role": "user", "content": "Selam"}], "deepseek-chat", 1, []))

        answer_deltas = [event["text"] for event in events if event["type"] == "answer_delta"]
        self.assertEqual(answer_deltas, ["Hello ", "world"])

        answer_start_index = next(index for index, event in enumerate(events) if event["type"] == "answer_start")
        first_delta_index = next(
            index for index, event in enumerate(events) if event["type"] == "answer_delta" and event["text"] == "Hello "
        )
        second_delta_index = next(
            index for index, event in enumerate(events) if event["type"] == "answer_delta" and event["text"] == "world"
        )
        usage_index = next(index for index, event in enumerate(events) if event["type"] == "usage")

        self.assertLess(answer_start_index, first_delta_index)
        self.assertLess(first_delta_index, second_delta_index)
        self.assertLess(second_delta_index, usage_index)

    def test_chat_stream_response_disables_buffering(self):
        conversation_id = self._create_conversation()
        save_app_settings(
            {
                "user_preferences": "",
                "max_steps": "1",
                "active_tools": "[]",
                "rag_auto_inject": "false",
            }
        )

        fake_events = iter(
            [
                {"type": "answer_start"},
                {"type": "answer_delta", "text": "Live output"},
                {"type": "tool_capture", "tool_results": []},
                {"type": "done"},
            ]
        )

        with patch("routes.chat.run_agent_stream", return_value=fake_events):
            response = self.client.post(
                "/chat",
                json={
                    "conversation_id": conversation_id,
                    "model": "deepseek-chat",
                    "user_content": "Hello",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get("Cache-Control"), "no-cache")
        self.assertEqual(response.headers.get("X-Accel-Buffering"), "no")

    def test_chat_edit_resend_replaces_future_messages(self):
        conversation_id = self._create_conversation()
        with get_db() as conn:
            user_one_id = conn.execute(
                "INSERT INTO messages (conversation_id, role, content, metadata) VALUES (?, 'user', ?, ?)",
                (conversation_id, "First question", None),
            ).lastrowid
            first_assistant_id = conn.execute(
                "INSERT INTO messages (conversation_id, role, content, metadata) VALUES (?, 'assistant', ?, ?)",
                (conversation_id, "First answer", None),
            ).lastrowid
            edited_user_id = conn.execute(
                "INSERT INTO messages (conversation_id, role, content, metadata) VALUES (?, 'user', ?, ?)",
                (conversation_id, "Old message", None),
            ).lastrowid
            stale_assistant_id = conn.execute(
                "INSERT INTO messages (conversation_id, role, content, metadata) VALUES (?, 'assistant', ?, ?)",
                (conversation_id, "Old answer", None),
            ).lastrowid

        save_app_settings(
            {
                "user_preferences": "",
                "max_steps": "1",
                "active_tools": "[]",
                "rag_auto_inject": "false",
            }
        )

        fake_events = iter(
            [
                {"type": "answer_start"},
                {"type": "answer_delta", "text": "New answer"},
                {"type": "tool_capture", "tool_results": []},
                {"type": "done"},
            ]
        )

        with patch("routes.chat.run_agent_stream", return_value=fake_events):
            response = self.client.post(
                "/chat",
                json={
                    "conversation_id": conversation_id,
                    "edited_message_id": edited_user_id,
                    "model": "deepseek-chat",
                    "user_content": "New message",
                    "messages": [
                        {"role": "user", "content": "First question"},
                        {"role": "assistant", "content": "First answer"},
                        {"role": "user", "content": "New message"},
                    ],
                },
            )

        self.assertEqual(response.status_code, 200)
        events = [json.loads(line) for line in response.get_data(as_text=True).strip().splitlines()]
        message_ids_event = next((event for event in events if event["type"] == "message_ids"), None)
        self.assertIsNotNone(message_ids_event)
        self.assertEqual(message_ids_event["user_message_id"], edited_user_id)
        self.assertIsInstance(message_ids_event["assistant_message_id"], int)

        with get_db() as conn:
            rows = conn.execute(
                "SELECT id, role, content FROM messages WHERE conversation_id = ? AND deleted_at IS NULL ORDER BY id",
                (conversation_id,),
            ).fetchall()
            deleted_row = conn.execute(
                "SELECT deleted_at FROM messages WHERE id = ?",
                (stale_assistant_id,),
            ).fetchone()

        self.assertEqual(
            [(row["id"], row["role"], row["content"]) for row in rows],
            [
                (user_one_id, "user", "First question"),
                (first_assistant_id, "assistant", "First answer"),
                (edited_user_id, "user", "New message"),
                (message_ids_event["assistant_message_id"], "assistant", "New answer"),
            ],
        )
        self.assertIsNotNone(deleted_row)
        self.assertIsNotNone(deleted_row["deleted_at"])

    def test_chat_stream_emits_history_sync_with_canonical_messages(self):
        conversation_id = self._create_conversation()
        save_app_settings(
            {
                "user_preferences": "",
                "max_steps": "1",
                "active_tools": "[]",
                "rag_auto_inject": "false",
            }
        )

        fake_events = iter(
            [
                {"type": "answer_start"},
                {"type": "answer_delta", "text": "Senkron cevap"},
                {"type": "tool_capture", "tool_results": []},
                {"type": "done"},
            ]
        )

        with patch("routes.chat.run_agent_stream", return_value=fake_events):
            response = self.client.post(
                "/chat",
                json={
                    "conversation_id": conversation_id,
                    "model": "deepseek-chat",
                    "user_content": "Hello",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )

        self.assertEqual(response.status_code, 200)
        events = [json.loads(line) for line in response.get_data(as_text=True).strip().splitlines()]
        history_sync_event = next((event for event in events if event["type"] == "history_sync"), None)
        self.assertIsNotNone(history_sync_event)
        self.assertEqual([message["role"] for message in history_sync_event["messages"]], ["user", "assistant"])

    def test_chat_summarizes_oldest_unsummarized_visible_messages(self):
        conversation_id = self._create_conversation()
        dense_message = " ".join(["context"] * 120)
        save_app_settings(
            {
                "user_preferences": "",
                "max_steps": "1",
                "active_tools": "[]",
                "rag_auto_inject": "false",
                "chat_summary_mode": "auto",
                "chat_summary_trigger_token_count": "1000",
                "summary_skip_first": "0",
                "summary_skip_last": "0",
            }
        )

        with get_db() as conn:
            for index in range(39):
                conn.execute(
                    "INSERT INTO messages (conversation_id, role, content, metadata) VALUES (?, 'user', ?, ?)",
                    (conversation_id, f"Message {index + 1} {dense_message}", None),
                )

        fake_summary = {
            "content": "Summary of the first 20 messages with enough retained detail to meet the minimum length validation threshold for summaries.",
            "reasoning_content": "",
            "usage": None,
            "tool_results": [],
            "errors": [],
        }
        fake_events = iter(
            [
                {"type": "answer_start"},
                {"type": "answer_delta", "text": "Live answer"},
                {"type": "tool_capture", "tool_results": []},
                {"type": "done"},
            ]
        )

        with patch("routes.chat.get_prompt_preflight_summary_token_count", return_value=1000), patch(
            "routes.chat.collect_agent_response", return_value=fake_summary
        ), patch("routes.chat.run_agent_stream", return_value=fake_events), patch("routes.chat.sync_conversations_to_rag_safe"):
            response = self.client.post(
                "/chat",
                json={
                    "conversation_id": conversation_id,
                    "model": "deepseek-chat",
                    "user_content": f"Message 40 {dense_message}",
                    "messages": [{"role": "user", "content": f"Message 40 {dense_message}"}],
                },
            )
            streamed_events = [json.loads(line) for line in response.get_data(as_text=True).strip().splitlines()]

        self.assertEqual(response.status_code, 200)
        conversation_response = self.client.get(f"/api/conversations/{conversation_id}")
        self.assertEqual(conversation_response.status_code, 200)
        conversation_messages = conversation_response.get_json()["messages"]
        self.assertEqual(conversation_messages[0]["role"], "summary")
        self.assertTrue(conversation_messages[0]["metadata"]["is_summary"])
        self.assertEqual(conversation_messages[0]["metadata"]["covered_message_count"], 39)
        self.assertEqual(conversation_messages[0]["metadata"]["summary_mode"], "auto")
        self.assertEqual(conversation_messages[0]["metadata"]["trigger_token_count"], 1000)
        self.assertEqual(
            conversation_messages[0]["content"],
            "Conversation summary (generated from deleted messages):\n\nSummary of the first 20 messages with enough retained detail to meet the minimum length validation threshold for summaries.",
        )
        self.assertEqual(len(conversation_messages), 3)
        self.assertEqual(conversation_messages[-1]["role"], "assistant")

        summary_event = next((event for event in streamed_events if event["type"] == "conversation_summary_applied"), None)
        self.assertIsNotNone(summary_event)
        self.assertEqual(summary_event["covered_message_count"], 39)
        self.assertEqual(summary_event["mode"], "auto")

        history_sync_events = [event for event in streamed_events if event["type"] == "history_sync"]
        self.assertEqual(len(history_sync_events), 2)
        self.assertEqual(history_sync_events[-1]["messages"][0]["role"], "summary")

        with get_db() as conn:
            deleted_count = conn.execute(
                "SELECT COUNT(*) AS count FROM messages WHERE conversation_id = ? AND deleted_at IS NOT NULL",
                (conversation_id,),
            ).fetchone()["count"]
        self.assertEqual(deleted_count, 39)

    def test_chat_rejects_edit_for_message_removed_by_summary(self):
        conversation_id = self._create_conversation()
        summary_metadata = serialize_message_metadata(
            {
                "is_summary": True,
                "summary_source": "conversation_history",
                "covered_message_ids": [999],
                "covered_message_count": 20,
            }
        )

        with get_db() as conn:
            conn.execute(
                "INSERT INTO messages (conversation_id, role, content, metadata) VALUES (?, 'summary', ?, ?)",
                (conversation_id, "Conversation summary (generated from deleted messages):\n\nSummary", summary_metadata),
            )

        response = self.client.post(
            "/chat",
            json={
                "conversation_id": conversation_id,
                "edited_message_id": 999,
                "model": "deepseek-chat",
                "user_content": "New content",
                "messages": [{"role": "user", "content": "New content"}],
            },
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["error"], "This message can no longer be edited because it was summarized.")

    def test_chat_summary_covers_interleaved_tool_messages(self):
        conversation_id = self._create_conversation()
        dense_message = " ".join(["history"] * 120)
        save_app_settings(
            {
                "user_preferences": "",
                "max_steps": "1",
                "active_tools": "[]",
                "rag_auto_inject": "false",
                "chat_summary_mode": "auto",
                "chat_summary_trigger_token_count": "1000",
                "summary_skip_first": "0",
                "summary_skip_last": "0",
            }
        )

        with get_db() as conn:
            conn.execute(
                "INSERT INTO messages (conversation_id, role, content, metadata) VALUES (?, 'user', ?, ?)",
                (conversation_id, f"First user message {dense_message}", None),
            )
            conn.execute(
                "INSERT INTO messages (conversation_id, role, content, metadata) VALUES (?, 'assistant', ?, ?)",
                (conversation_id, f"First answer {dense_message}", None),
            )
            conn.execute(
                "INSERT INTO messages (conversation_id, role, content, tool_call_id) VALUES (?, 'tool', ?, ?)",
                (conversation_id, '{"ok":true}', "call-1"),
            )
            for index in range(7):
                conn.execute(
                    "INSERT INTO messages (conversation_id, role, content, metadata) VALUES (?, 'user', ?, ?)",
                    (conversation_id, f"Follow-up user message {index + 1} {dense_message}", None),
                )

        fake_summary = {
            "content": "Summary of the first five visible messages with enough retained detail to meet the minimum length validation threshold for summaries.",
            "reasoning_content": "",
            "usage": None,
            "tool_results": [],
            "errors": [],
        }
        fake_events = iter(
            [
                {"type": "answer_start"},
                {"type": "answer_delta", "text": "New live answer"},
                {"type": "tool_capture", "tool_results": []},
                {"type": "done"},
            ]
        )

        with patch("routes.chat.get_prompt_preflight_summary_token_count", return_value=1000), patch(
            "routes.chat.collect_agent_response", return_value=fake_summary
        ), patch("routes.chat.run_agent_stream", return_value=fake_events), patch("routes.chat.sync_conversations_to_rag_safe"):
            response = self.client.post(
                "/chat",
                json={
                    "conversation_id": conversation_id,
                    "model": "deepseek-chat",
                    "user_content": f"Third user message {dense_message}",
                    "messages": [{"role": "user", "content": f"Third user message {dense_message}"}],
                },
            )
            response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)

        with get_db() as conn:
            visible_rows = conn.execute(
                "SELECT role, content, tool_call_id FROM messages WHERE conversation_id = ? AND deleted_at IS NULL ORDER BY position, id",
                (conversation_id,),
            ).fetchall()
            deleted_count = conn.execute(
                "SELECT COUNT(*) AS count FROM messages WHERE conversation_id = ? AND deleted_at IS NOT NULL",
                (conversation_id,),
            ).fetchone()["count"]

        self.assertEqual(
            [(row["role"], row["tool_call_id"]) for row in visible_rows],
            [("summary", None), ("user", None), ("assistant", None)],
        )
        self.assertGreaterEqual(deleted_count, 6)

    def test_chat_summary_can_trigger_from_large_tool_history(self):
        conversation_id = self._create_conversation()
        dense_tool_payload = json.dumps(
            {
                "results": [
                    {
                        "title": f"Result {index}",
                        "content": " ".join(["tool-context"] * 120),
                    }
                    for index in range(12)
                ]
            },
            ensure_ascii=False,
        )
        save_app_settings(
            {
                "user_preferences": "",
                "max_steps": "1",
                "active_tools": "[]",
                "rag_auto_inject": "false",
                "chat_summary_mode": "auto",
                "chat_summary_trigger_token_count": "1000",
                "summary_skip_first": "0",
                "summary_skip_last": "0",
            }
        )

        with get_db() as conn:
            conn.execute(
                "INSERT INTO messages (conversation_id, role, content, metadata) VALUES (?, 'user', ?, ?)",
                (conversation_id, "Short user message", None),
            )
            conn.execute(
                "INSERT INTO messages (conversation_id, role, content, metadata) VALUES (?, 'assistant', ?, ?)",
                (conversation_id, "Short assistant reply", None),
            )
            conn.execute(
                "INSERT INTO messages (conversation_id, role, content, tool_call_id) VALUES (?, 'tool', ?, ?)",
                (conversation_id, dense_tool_payload, "call-1"),
            )

        fake_summary = {
            "content": "Summary of the early visible conversation with enough retained detail to meet the minimum length validation threshold for summaries.",
            "reasoning_content": "",
            "usage": None,
            "tool_results": [],
            "errors": [],
        }
        fake_events = iter(
            [
                {"type": "answer_start"},
                {"type": "answer_delta", "text": "Fresh answer"},
                {"type": "tool_capture", "tool_results": []},
                {"type": "done"},
            ]
        )

        with patch("routes.chat.get_prompt_preflight_summary_token_count", return_value=1000), patch(
            "routes.chat.collect_agent_response", return_value=fake_summary
        ), patch("routes.chat.run_agent_stream", return_value=fake_events), patch("routes.chat.sync_conversations_to_rag_safe"):
            response = self.client.post(
                "/chat",
                json={
                    "conversation_id": conversation_id,
                    "model": "deepseek-chat",
                    "user_content": "Latest question",
                    "messages": [{"role": "user", "content": "Latest question"}],
                },
            )
            streamed_events = [json.loads(line) for line in response.get_data(as_text=True).strip().splitlines()]

        self.assertEqual(response.status_code, 200)
        summary_event = next((event for event in streamed_events if event["type"] == "conversation_summary_applied"), None)
        self.assertIsNotNone(summary_event)
        self.assertEqual(summary_event["covered_message_count"], 2)
        self.assertEqual(summary_event["covered_tool_message_count"], 1)

        with get_db() as conn:
            visible_rows = conn.execute(
                "SELECT role, tool_call_id FROM messages WHERE conversation_id = ? AND deleted_at IS NULL ORDER BY position, id",
                (conversation_id,),
            ).fetchall()

        self.assertEqual(
            [(row["role"], row["tool_call_id"]) for row in visible_rows],
            [
                ("summary", None),
                ("user", None),
                ("assistant", None),
            ],
        )

    def test_build_summary_prompt_messages_filters_empty_and_merges_assistant_history(self):
        prompt_messages = build_summary_prompt_messages(
            [
                {"role": "user", "content": "First user request"},
                {"role": "assistant", "content": "   "},
                {"role": "assistant", "content": "First assistant note"},
                {"role": "assistant", "content": "Second assistant note"},
                {"role": "assistant", "content": FINAL_ANSWER_ERROR_TEXT},
                {"role": "user", "content": "   "},
                {"role": "user", "content": "Second user request"},
            ],
            "",
        )

        self.assertEqual([message["role"] for message in prompt_messages], ["system", "user"])
        self.assertIn("USER:\nFirst user request", prompt_messages[1]["content"])
        self.assertIn("ASSISTANT:\nFirst assistant note\n\nSecond assistant note", prompt_messages[1]["content"])
        self.assertIn("USER:\nSecond user request", prompt_messages[1]["content"])
        self.assertNotIn(FINAL_ANSWER_ERROR_TEXT, prompt_messages[1]["content"])

    def test_build_summary_prompt_messages_include_tool_findings_from_assistant_metadata(self):
        prompt_messages = build_summary_prompt_messages(
            [
                {"role": "user", "content": "Research the market"},
                {
                    "role": "assistant",
                    "content": "I reviewed the sources.",
                    "metadata": {
                        "tool_results": [
                            {
                                "tool_name": "fetch_url",
                                "summary": "Revenue grew 18 percent year over year.",
                                "content": "Very long tool payload that should not be needed when a summary is available.",
                            },
                            {
                                "tool_name": "search_web",
                                "content": "Analysts expect demand to remain strong through Q4 based on recent filings and channel checks.",
                            },
                        ]
                    },
                },
            ],
            "",
        )

        transcript = prompt_messages[1]["content"]
        self.assertIn("Tool findings:", transcript)
        self.assertIn("fetch_url: Revenue grew 18 percent year over year.", transcript)
        self.assertIn("search_web: Analysts expect demand to remain strong through Q4", transcript)

    def test_build_summary_prompt_messages_include_tool_role_messages(self):
        prompt_messages = build_summary_prompt_messages(
            [
                {"role": "user", "content": "Check the source"},
                {"role": "assistant", "content": "I will inspect the page."},
                {"role": "tool", "content": '{"ok": true, "headline": "Market expands"}', "tool_call_id": "call-9"},
            ],
            "",
        )

        transcript = prompt_messages[1]["content"]
        self.assertIn("TOOL RESULT:\ncall call-9: {\"ok\": true, \"headline\": \"Market expands\"}", transcript)

    def test_estimate_prompt_tokens_counts_tool_call_payload(self):
        plain_messages = [
            {
                "role": "assistant",
                "content": "Done.",
            }
        ]
        tool_call_messages = [
            {
                "role": "assistant",
                "content": "Done.",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "search_web",
                            "arguments": '{"queries":["token budgets"]}',
                        },
                    }
                ],
            }
        ]

        self.assertGreater(_estimate_prompt_tokens(tool_call_messages), _estimate_prompt_tokens(plain_messages))

    def test_select_recent_prompt_window_respects_token_budget(self):
        messages = [
            {"role": "user", "content": "alpha " * 120},
            {"role": "assistant", "content": "beta " * 120},
            {"role": "user", "content": "gamma " * 120},
        ]

        selected = _select_recent_prompt_window(messages, max_tokens=120, min_user_messages=2)

        self.assertLessEqual(_estimate_prompt_tokens(build_api_messages(selected)), 120)
        self.assertLessEqual(sum(1 for message in selected if message["role"] == "user"), 2)

    def test_select_recent_prompt_window_keeps_complete_tool_call_blocks(self):
        messages = [
            {"role": "user", "content": "older context " * 80, "position": 1, "id": 1},
            {
                "role": "assistant",
                "content": "Calling a tool now.",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "search_web", "arguments": "{}"},
                    }
                ],
                "position": 2,
                "id": 2,
            },
            {"role": "tool", "content": '{"ok": true}', "tool_call_id": "call-1", "position": 3, "id": 3},
            {"role": "user", "content": "latest question", "position": 4, "id": 4},
        ]

        selected = _select_recent_prompt_window(messages, max_tokens=200)

        self.assertEqual([message["role"] for message in selected], ["assistant", "tool", "user"])
        self.assertEqual(selected[0]["tool_calls"][0]["id"], "call-1")
        self.assertEqual(selected[1]["tool_call_id"], "call-1")

    def test_select_recent_prompt_window_skips_incomplete_tool_call_blocks(self):
        messages = [
            {
                "role": "assistant",
                "content": "Calling a tool now.",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "search_web", "arguments": "{}"},
                    }
                ],
                "position": 1,
                "id": 1,
            },
            {"role": "user", "content": "latest question", "position": 2, "id": 2},
        ]

        selected = _select_recent_prompt_window(messages, max_tokens=300)

        self.assertEqual([message["role"] for message in selected], ["user"])

    def test_select_recent_prompt_window_redacts_old_tool_output_but_keeps_current_turn_tool_output(self):
        messages = [
            {"role": "user", "content": "Older question", "position": 1, "id": 1},
            {
                "role": "assistant",
                "content": "Calling old tool.",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "search_web", "arguments": "{}"},
                    }
                ],
                "position": 2,
                "id": 2,
            },
            {"role": "tool", "content": "A" * 2500, "tool_call_id": "call-1", "position": 3, "id": 3},
            {"role": "user", "content": "Latest question", "position": 4, "id": 4},
            {
                "role": "assistant",
                "content": "Calling current tool.",
                "tool_calls": [
                    {
                        "id": "call-2",
                        "type": "function",
                        "function": {"name": "fetch_url", "arguments": "{}"},
                    }
                ],
                "position": 5,
                "id": 5,
            },
            {"role": "tool", "content": "Current tool result", "tool_call_id": "call-2", "position": 6, "id": 6},
        ]

        selected = _select_recent_prompt_window(messages, max_tokens=500)

        redacted_old_tool = next(message for message in selected if message.get("tool_call_id") == "call-1")
        current_turn_tool = next(message for message in selected if message.get("tool_call_id") == "call-2")

        self.assertEqual(redacted_old_tool["role"], "tool")
        self.assertEqual(redacted_old_tool["content"], OMITTED_TOOL_OUTPUT_TEXT)
        self.assertEqual(current_turn_tool["role"], "tool")
        self.assertEqual(current_turn_tool["content"], "Current tool result")
        self.assertLessEqual(_estimate_prompt_tokens(build_api_messages(selected)), 500)

    def test_summary_source_selection_uses_expanded_prompt_budget(self):
        canonical_messages = [
            {"id": 1, "position": 1, "role": "user", "content": "Kickoff"},
            {
                "id": 2,
                "position": 2,
                "role": "assistant",
                "content": "Short reply",
                "metadata": {
                    "tool_results": [
                        {
                            "tool_name": "fetch_url",
                            "summary": "Important result " * 60,
                            "content": "Raw payload " * 300,
                        }
                    ]
                },
            },
        ]

        selected = _select_summary_source_messages_by_token_budget(
            canonical_messages,
            canonical_messages,
            target_tokens=240,
            user_preferences="",
        )

        self.assertEqual([message["role"] for message in selected], ["user"])
        prompt_messages = build_summary_prompt_messages(selected, "")
        self.assertLessEqual(_estimate_prompt_tokens(prompt_messages), 240)

    def test_chat_summary_status_reports_detailed_failure_stage(self):
        conversation_id = self._create_conversation()
        dense_message = " ".join(["context"] * 120)
        save_app_settings(
            {
                "user_preferences": "",
                "max_steps": "1",
                "active_tools": "[]",
                "rag_auto_inject": "false",
                "chat_summary_mode": "auto",
                "chat_summary_trigger_token_count": "1000",
                "summary_skip_first": "0",
                "summary_skip_last": "0",
            }
        )

        with get_db() as conn:
            for index in range(12):
                conn.execute(
                    "INSERT INTO messages (conversation_id, role, content, metadata) VALUES (?, 'user', ?, ?)",
                    (conversation_id, f"Seed {index + 1} {dense_message}", None),
                )

        failing_summary = {
            "content": "",
            "reasoning_content": "",
            "usage": None,
            "tool_results": [],
            "errors": ["This model's maximum context length is 131072 tokens. However, you requested 132465 tokens."],
        }
        fake_events = iter(
            [
                {"type": "answer_start"},
                {"type": "answer_delta", "text": "Live answer"},
                {"type": "tool_capture", "tool_results": []},
                {"type": "done"},
            ]
        )

        with patch("routes.chat.collect_agent_response", return_value=failing_summary), patch(
            "routes.chat.run_agent_stream", return_value=fake_events
        ), patch("routes.chat.sync_conversations_to_rag_safe"):
            response = self.client.post(
                "/chat",
                json={
                    "conversation_id": conversation_id,
                    "model": "deepseek-chat",
                    "user_content": f"Latest {dense_message}",
                    "messages": [{"role": "user", "content": f"Latest {dense_message}"}],
                },
            )
            events = [json.loads(line) for line in response.get_data(as_text=True).strip().splitlines()]

        self.assertEqual(response.status_code, 200)
        status_event = next((event for event in events if event["type"] == "conversation_summary_status"), None)
        self.assertIsNotNone(status_event)
        self.assertEqual(status_event["reason"], "summary_generation_failed")
        self.assertEqual(status_event["failure_stage"], "context_too_large")
        self.assertIn("maximum context length", status_event["failure_detail"])
        self.assertEqual(status_event["returned_text_length"], 0)
        self.assertEqual(status_event["summary_error_count"], 1)
        self.assertGreaterEqual(status_event["candidate_message_count"], 1)

    def test_chat_summary_rejects_error_summary_without_deleting_messages(self):
        conversation_id = self._create_conversation()
        dense_message = " ".join(["history"] * 120)
        save_app_settings(
            {
                "user_preferences": "",
                "max_steps": "1",
                "active_tools": "[]",
                "rag_auto_inject": "false",
                "chat_summary_mode": "auto",
                "chat_summary_trigger_token_count": "1000",
                "summary_skip_first": "0",
                "summary_skip_last": "0",
            }
        )

        with get_db() as conn:
            for index in range(12):
                conn.execute(
                    "INSERT INTO messages (conversation_id, role, content, metadata) VALUES (?, 'user', ?, ?)",
                    (conversation_id, f"Seed {index + 1} {dense_message}", None),
                )

        failing_summary = {
            "content": FINAL_ANSWER_ERROR_TEXT,
            "reasoning_content": "",
            "usage": None,
            "tool_results": [],
            "errors": [FINAL_ANSWER_ERROR_TEXT],
        }
        fake_events = iter(
            [
                {"type": "answer_start"},
                {"type": "answer_delta", "text": "Live answer"},
                {"type": "tool_capture", "tool_results": []},
                {"type": "done"},
            ]
        )

        with patch("routes.chat.collect_agent_response", return_value=failing_summary), patch(
            "routes.chat.run_agent_stream", return_value=fake_events
        ), patch("routes.chat.sync_conversations_to_rag_safe"):
            response = self.client.post(
                "/chat",
                json={
                    "conversation_id": conversation_id,
                    "model": "deepseek-chat",
                    "user_content": f"Latest {dense_message}",
                    "messages": [{"role": "user", "content": f"Latest {dense_message}"}],
                },
            )
            events = [json.loads(line) for line in response.get_data(as_text=True).strip().splitlines()]

        self.assertEqual(response.status_code, 200)
        self.assertFalse(any(event["type"] == "conversation_summary_applied" for event in events))

        with get_db() as conn:
            visible_rows = conn.execute(
                "SELECT role FROM messages WHERE conversation_id = ? AND deleted_at IS NULL ORDER BY position, id",
                (conversation_id,),
            ).fetchall()
            deleted_count = conn.execute(
                "SELECT COUNT(*) AS count FROM messages WHERE conversation_id = ? AND deleted_at IS NOT NULL",
                (conversation_id,),
            ).fetchone()["count"]

        self.assertEqual(deleted_count, 0)
        self.assertEqual([row["role"] for row in visible_rows][-2:], ["user", "assistant"])

    def test_chat_can_create_multiple_summary_passes_without_resummarizing_old_ones(self):
        conversation_id = self._create_conversation()
        dense_message = " ".join(["seed"] * 120)
        save_app_settings(
            {
                "user_preferences": "",
                "max_steps": "1",
                "active_tools": "[]",
                "rag_auto_inject": "false",
                "chat_summary_mode": "auto",
                "chat_summary_trigger_token_count": "1000",
                "summary_skip_first": "0",
                "summary_skip_last": "0",
            }
        )

        with get_db() as conn:
            for index in range(13):
                conn.execute(
                    "INSERT INTO messages (conversation_id, role, content, metadata) VALUES (?, 'user', ?, ?)",
                    (conversation_id, f"Seed {index + 1} {dense_message}", None),
                )

        first_summary = {
            "content": "First summary block with enough retained detail to satisfy the minimum length validation threshold for summary generation.",
            "reasoning_content": "",
            "usage": None,
            "tool_results": [],
            "errors": [],
        }
        second_summary = {
            "content": "Second summary block with enough retained detail to satisfy the minimum length validation threshold for summary generation.",
            "reasoning_content": "",
            "usage": None,
            "tool_results": [],
            "errors": [],
        }

        for user_text, summary_payload, answer_text in [
            ("Turn 14", first_summary, "Answer 1"),
            ("Turn 15", second_summary, "Answer 2"),
        ]:
            fake_events = iter(
                [
                    {"type": "answer_start"},
                    {"type": "answer_delta", "text": answer_text},
                    {"type": "tool_capture", "tool_results": []},
                    {"type": "done"},
                ]
            )
            with patch("routes.chat.collect_agent_response", return_value=summary_payload), patch(
                "routes.chat.run_agent_stream", return_value=fake_events
            ), patch("routes.chat.sync_conversations_to_rag_safe"):
                response = self.client.post(
                    "/chat",
                    json={
                        "conversation_id": conversation_id,
                        "model": "deepseek-chat",
                        "user_content": f"{user_text} {dense_message}",
                        "messages": [{"role": "user", "content": f"{user_text} {dense_message}"}],
                    },
                )
                response.get_data(as_text=True)
            self.assertEqual(response.status_code, 200)

        conversation_response = self.client.get(f"/api/conversations/{conversation_id}")
        self.assertEqual(conversation_response.status_code, 200)
        messages = conversation_response.get_json()["messages"]
        summary_messages = [message for message in messages if message["role"] == "summary"]

        self.assertEqual(len(summary_messages), 1)
        self.assertEqual(
            summary_messages[0]["content"],
            "Conversation summary (generated from deleted messages):\n\nFirst summary block with enough retained detail to satisfy the minimum length validation threshold for summary generation.",
        )
        self.assertEqual(summary_messages[0]["metadata"]["covered_message_count"], 13)
        self.assertEqual(
            [message["role"] for message in messages],
            ["summary", "user", "assistant", "user", "assistant"],
        )

    def test_chat_summary_covers_tool_call_assistant_messages(self):
        conversation_id = self._create_conversation()
        dense_message = " ".join(["history"] * 120)
        save_app_settings(
            {
                "user_preferences": "",
                "max_steps": "1",
                "active_tools": "[]",
                "rag_auto_inject": "false",
                "chat_summary_mode": "never",
                "chat_summary_trigger_token_count": "1000",
                "summary_skip_first": "0",
                "summary_skip_last": "0",
            }
        )

        with get_db() as conn:
            conn.execute(
                "INSERT INTO messages (conversation_id, role, content, metadata, position) VALUES (?, 'user', ?, ?, ?)",
                (conversation_id, f"First user {dense_message}", None, 1),
            )
            assistant_tool_id = conn.execute(
                "INSERT INTO messages (conversation_id, role, content, tool_calls, position) VALUES (?, 'assistant', ?, ?, ?)",
                (
                    conversation_id,
                    "I will use a tool.",
                    json.dumps(
                        [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {"name": "search_web", "arguments": "{}"},
                            }
                        ]
                    ),
                    2,
                ),
            ).lastrowid
            tool_id = conn.execute(
                "INSERT INTO messages (conversation_id, role, content, tool_call_id, position) VALUES (?, 'tool', ?, ?, ?)",
                (conversation_id, '{"ok": true, "headline": "Important finding"}', "call-1", 3),
            ).lastrowid
            conn.execute(
                "INSERT INTO messages (conversation_id, role, content, metadata, position) VALUES (?, 'user', ?, ?, ?)",
                (conversation_id, f"Second user {dense_message}", None, 4),
            )
            conn.execute(
                "INSERT INTO messages (conversation_id, role, content, metadata, position) VALUES (?, 'assistant', ?, ?, ?)",
                (conversation_id, f"Plain answer {dense_message}", None, 5),
            )

        fake_summary = {
            "content": "Summary block with enough retained detail to satisfy the minimum length validation threshold and preserve tool-chain coverage in metadata.",
            "reasoning_content": "",
            "usage": None,
            "tool_results": [],
            "errors": [],
        }

        with patch("routes.chat.collect_agent_response", return_value=fake_summary), patch(
            "routes.chat.sync_conversations_to_rag_safe"
        ), patch("routes.chat.get_prompt_summary_max_tokens", return_value=300):
            response = self.client.post(
                f"/api/conversations/{conversation_id}/summarize",
                json={"force": True},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["applied"])

        summary_message = next(message for message in payload["messages"] if message["role"] == "summary")
        self.assertEqual(summary_message["metadata"]["covered_tool_call_message_count"], 1)
        self.assertEqual(summary_message["metadata"]["covered_tool_message_count"], 1)
        self.assertIn(assistant_tool_id, summary_message["metadata"]["covered_message_ids"])
        self.assertIn(tool_id, summary_message["metadata"]["covered_message_ids"])

        with get_db() as conn:
            visible_rows = conn.execute(
                "SELECT id, role FROM messages WHERE conversation_id = ? AND deleted_at IS NULL ORDER BY position, id",
                (conversation_id,),
            ).fetchall()

        visible_pairs = [(row["id"], row["role"]) for row in visible_rows]
        self.assertNotIn((assistant_tool_id, "assistant"), visible_pairs)
        self.assertNotIn((tool_id, "tool"), visible_pairs)

    def test_manual_summarize_can_create_hierarchical_summary_from_existing_summaries(self):
        conversation_id = self._create_conversation()
        save_app_settings(
            {
                "user_preferences": "",
                "max_steps": "1",
                "active_tools": "[]",
                "rag_auto_inject": "false",
                "chat_summary_mode": "never",
                "chat_summary_trigger_token_count": "1000",
                "summary_skip_first": "0",
                "summary_skip_last": "0",
            }
        )

        long_summary = " ".join(["summary-detail"] * 250)
        with get_db() as conn:
            conn.execute(
                "INSERT INTO messages (conversation_id, role, content, metadata, position) VALUES (?, 'summary', ?, ?, ?)",
                (
                    conversation_id,
                    f"Conversation summary (generated from deleted messages):\n\n{long_summary}",
                    serialize_message_metadata({"is_summary": True, "summary_source": "conversation_history", "summary_level": 1}),
                    1,
                ),
            )
            conn.execute(
                "INSERT INTO messages (conversation_id, role, content, metadata, position) VALUES (?, 'summary', ?, ?, ?)",
                (
                    conversation_id,
                    f"Conversation summary (generated from deleted messages):\n\n{long_summary}",
                    serialize_message_metadata({"is_summary": True, "summary_source": "conversation_history", "summary_level": 1}),
                    2,
                ),
            )
            conn.execute(
                "INSERT INTO messages (conversation_id, role, content, metadata, position) VALUES (?, 'summary', ?, ?, ?)",
                (
                    conversation_id,
                    "Conversation summary (generated from deleted messages):\n\nRecent compact summary.",
                    serialize_message_metadata({"is_summary": True, "summary_source": "conversation_history", "summary_level": 1}),
                    3,
                ),
            )

        fake_summary = {
            "content": json.dumps(
                {
                    "facts": ["Older summary facts were merged."],
                    "decisions": ["A hierarchical summary replaced older summaries."],
                    "open_issues": [],
                    "entities": [],
                    "tool_outcomes": [],
                },
                ensure_ascii=False,
            ),
            "reasoning_content": "",
            "usage": None,
            "tool_results": [],
            "errors": [],
        }

        with patch("routes.chat.collect_agent_response", return_value=fake_summary), patch(
            "routes.chat.sync_conversations_to_rag_safe"
        ), patch("routes.chat.get_prompt_summary_max_tokens", return_value=300):
            response = self.client.post(
                f"/api/conversations/{conversation_id}/summarize",
                json={"force": True},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["applied"])
        summary_messages = [message for message in payload["messages"] if message["role"] == "summary"]
        self.assertEqual(len(summary_messages), 2)
        hierarchical_summary = next(message for message in summary_messages if message["metadata"]["summary_level"] == 2)
        self.assertEqual(hierarchical_summary["metadata"]["summary_source"], "summary_history")

    def test_run_agent_stream_reuses_cross_turn_fetch_url_memory(self):
        responses = [
            iter(
                [
                    self._tool_call_chunk("fetch_url", {"url": "https://example.com/page"}),
                    self._stream_chunk(usage=SimpleNamespace(prompt_tokens=3, completion_tokens=5, total_tokens=8)),
                ]
            ),
            iter(
                [
                    self._stream_chunk(content="Final answer from cached fetch."),
                    self._stream_chunk(usage=SimpleNamespace(prompt_tokens=4, completion_tokens=6, total_tokens=10)),
                ]
            ),
        ]

        with patch("agent.client.chat.completions.create", side_effect=responses), patch(
            "agent.get_exact_tool_memory_match",
            return_value={
                "content": "URL: https://example.com/page\n\nTitle: Example\n\nReused page content.",
                "summary": "Page content extracted: Example",
            },
        ), patch("agent.fetch_url_tool") as mocked_fetch:
            events = list(run_agent_stream([{"role": "user", "content": "Fetch it again"}], "deepseek-chat", 2, ["fetch_url"]))

        mocked_fetch.assert_not_called()
        tool_result_event = next(event for event in events if event["type"] == "tool_result")
        self.assertTrue(tool_result_event["cached"])
        self.assertIn("Page content extracted", tool_result_event["summary"])

    def test_chat_summary_mode_never_skips_summary_even_above_token_threshold(self):
        conversation_id = self._create_conversation()
        dense_message = " ".join(["never"] * 120)
        save_app_settings(
            {
                "user_preferences": "",
                "max_steps": "1",
                "active_tools": "[]",
                "rag_auto_inject": "false",
                "chat_summary_mode": "never",
                "chat_summary_trigger_token_count": "1000",
                "summary_skip_first": "0",
                "summary_skip_last": "0",
            }
        )

        with get_db() as conn:
            for index in range(12):
                conn.execute(
                    "INSERT INTO messages (conversation_id, role, content, metadata) VALUES (?, 'user', ?, ?)",
                    (conversation_id, f"Seed {index + 1} {dense_message}", None),
                )

        fake_events = iter(
            [
                {"type": "answer_start"},
                {"type": "answer_delta", "text": "Answer without summary"},
                {"type": "tool_capture", "tool_results": []},
                {"type": "done"},
            ]
        )

        with patch("routes.chat.collect_agent_response") as mocked_summary, patch(
            "routes.chat.run_agent_stream", return_value=fake_events
        ), patch("routes.chat.sync_conversations_to_rag_safe"):
            response = self.client.post(
                "/chat",
                json={
                    "conversation_id": conversation_id,
                    "model": "deepseek-chat",
                    "user_content": f"Latest {dense_message}",
                    "messages": [{"role": "user", "content": f"Latest {dense_message}"}],
                },
            )
            events = [json.loads(line) for line in response.get_data(as_text=True).strip().splitlines()]

        self.assertEqual(response.status_code, 200)
        self.assertFalse(any(event["type"] == "conversation_summary_applied" for event in events))
        mocked_summary.assert_not_called()

        conversation_response = self.client.get(f"/api/conversations/{conversation_id}")
        self.assertEqual(conversation_response.status_code, 200)
        visible_messages = conversation_response.get_json()["messages"]
        self.assertFalse(any(message["role"] == "summary" for message in visible_messages))

    def test_run_agent_stream_blocks_tool_json_after_max_steps(self):
        responses = [
            iter(
                [
                    self._stream_chunk(reasoning="Need web data."),
                    self._tool_call_chunk("search_web", {"queries": ["x"]}),
                    self._stream_chunk(usage=SimpleNamespace(prompt_tokens=2, completion_tokens=3, total_tokens=5)),
                ]
            ),
            iter(
                [
                    self._stream_chunk(reasoning="Still trying to call a tool."),
                    self._tool_call_chunk("search_web", {"queries": ["y"]}),
                    self._stream_chunk(usage=SimpleNamespace(prompt_tokens=2, completion_tokens=3, total_tokens=5)),
                ]
            ),
        ]

        with patch("agent.client.chat.completions.create", side_effect=responses), patch(
            "agent.search_web_tool",
            return_value=[{"title": "Test", "url": "https://example.com", "snippet": "Snippet"}],
        ):
            events = list(run_agent_stream([{"role": "user", "content": "Test"}], "deepseek-chat", 1, ["search_web"]))

        self.assertIn(
            {
                "type": "tool_error",
                "step": 1,
                "tool": "agent",
                "error": "Tool limit reached before the model produced a final answer.",
            },
            events,
        )
        self.assertIn({"type": "answer_delta", "text": FINAL_ANSWER_ERROR_TEXT}, events)
        leaked_json = [event for event in events if event["type"] == "answer_delta" and "tool_calls" in event["text"]]
        self.assertEqual(leaked_json, [])

    def test_generate_title_updates_conversation(self):
        conversation_id = self._create_conversation(title="Untitled")
        with get_db() as conn:
            conn.execute(
                "INSERT INTO messages (conversation_id, role, content, metadata) VALUES (?, 'user', ?, ?)",
                (conversation_id, "Generate a title", None),
            )
            conn.execute(
                "INSERT INTO messages (conversation_id, role, content, metadata) VALUES (?, 'assistant', ?, ?)",
                (conversation_id, "Certainly", None),
            )

        fake_result = {
            "content": "Updated Title",
            "reasoning_content": "",
            "usage": None,
            "tool_results": [],
            "errors": [],
        }
        with patch("routes.chat.collect_agent_response", return_value=fake_result):
            response = self.client.post(f"/api/conversations/{conversation_id}/generate-title")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["title"], "Updated Title")

        with get_db() as conn:
            row = conn.execute("SELECT title FROM conversations WHERE id = ?", (conversation_id,)).fetchone()
        self.assertEqual(row["title"], "Updated Title")

    def test_generate_title_falls_back_to_new_chat_for_noisy_output(self):
        conversation_id = self._create_conversation(title="Untitled")
        with get_db() as conn:
            conn.execute(
                "INSERT INTO messages (conversation_id, role, content, metadata) VALUES (?, 'user', ?, ?)",
                (conversation_id, "Need a short title", None),
            )
            conn.execute(
                "INSERT INTO messages (conversation_id, role, content, metadata) VALUES (?, 'assistant', ?, ?)",
                (conversation_id, "Sure, here is the answer", None),
            )

        fake_result = {
            "content": "**Tamamlandı!** 🚀 Canvas'a bakıp detayları ekleyebilirim.",
            "reasoning_content": "",
            "usage": None,
            "tool_results": [],
            "errors": [],
        }
        with patch("routes.chat.collect_agent_response", return_value=fake_result):
            response = self.client.post(f"/api/conversations/{conversation_id}/generate-title")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["title"], "New Chat")

        with get_db() as conn:
            row = conn.execute("SELECT title FROM conversations WHERE id = ?", (conversation_id,)).fetchone()
        self.assertEqual(row["title"], "New Chat")

    def test_generate_title_uses_source_fallback_when_model_errors(self):
        conversation_id = self._create_conversation(title="Untitled")
        with get_db() as conn:
            conn.execute(
                "INSERT INTO messages (conversation_id, role, content, metadata) VALUES (?, 'user', ?, ?)",
                (conversation_id, "python list sorting", None),
            )

        with patch("routes.chat.collect_agent_response", side_effect=RuntimeError("model unavailable")):
            response = self.client.post(f"/api/conversations/{conversation_id}/generate-title")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["title"], "Python List Sorting")

        with get_db() as conn:
            row = conn.execute("SELECT title FROM conversations WHERE id = ?", (conversation_id,)).fetchone()
        self.assertEqual(row["title"], "Python List Sorting")

    def test_generate_title_uses_minimal_prompt_without_runtime_context_or_tools(self):
        """Root-cause regression test: generate_title must NOT inject runtime context or tools."""
        conversation_id = self._create_conversation(title="Untitled")
        with get_db() as conn:
            conn.execute(
                "INSERT INTO messages (conversation_id, role, content, metadata) VALUES (?, 'user', ?, ?)",
                (conversation_id, "How do I sort a list?", None),
            )

        fake_result = {
            "content": "Sorting a List",
            "reasoning_content": "",
            "usage": None,
            "tool_results": [],
            "errors": [],
        }
        with patch("routes.chat.collect_agent_response", return_value=fake_result) as mocked_collect:
            self.client.post(f"/api/conversations/{conversation_id}/generate-title")

        mocked_collect.assert_called_once()
        args, _kwargs = mocked_collect.call_args
        prompt_messages, _model, max_steps, enabled_tool_names = args

        # Exactly two messages: system directive + user content — no runtime system context
        self.assertEqual(len(prompt_messages), 2)
        self.assertEqual(prompt_messages[0]["role"], "system")
        self.assertEqual(prompt_messages[1]["role"], "user")

        system_content = prompt_messages[0]["content"]
        self.assertNotIn("helpful AI assistant", system_content)
        self.assertNotIn("Available Tools", system_content)
        self.assertIn("title generator", system_content)

        # Must use exactly 1 step and zero tools — prevents multi-turn tool calls
        self.assertEqual(max_steps, 1)
        self.assertEqual(enabled_tool_names, [])

    def test_get_unsummarized_visible_messages_skip_first_and_last(self):
        from db import get_unsummarized_visible_messages

        messages = [
            {"id": 1, "position": 1, "role": "user", "content": "First"},
            {"id": 2, "position": 2, "role": "assistant", "content": "Second"},
            {"id": 3, "position": 3, "role": "user", "content": "Third"},
            {"id": 4, "position": 4, "role": "assistant", "content": "Fourth"},
            {"id": 5, "position": 5, "role": "user", "content": "Fifth"},
        ]
        result = get_unsummarized_visible_messages(messages, skip_first=1, skip_last=1)
        self.assertEqual([m["id"] for m in result], [2, 3, 4])

        result_all = get_unsummarized_visible_messages(messages, skip_first=0, skip_last=0)
        self.assertEqual([m["id"] for m in result_all], [1, 2, 3, 4, 5])

        result_over = get_unsummarized_visible_messages(messages, skip_first=3, skip_last=3)
        self.assertEqual(result_over, [])

        result_with_limit = get_unsummarized_visible_messages(messages, skip_first=1, skip_last=1, limit=2)
        self.assertEqual([m["id"] for m in result_with_limit], [2, 3])

    def test_get_unsummarized_visible_messages_ignores_hidden_tool_call_assistant_entries(self):
        from db import get_unsummarized_visible_messages

        messages = [
            {"id": 1, "position": 1, "role": "user", "content": "First", "tool_calls": []},
            {
                "id": 2,
                "position": 2,
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "call-1", "function": {"name": "search_web", "arguments": "{}"}}],
            },
            {"id": 3, "position": 3, "role": "assistant", "content": "Visible assistant", "tool_calls": []},
            {"id": 4, "position": 4, "role": "user", "content": "Fourth", "tool_calls": []},
            {"id": 5, "position": 5, "role": "assistant", "content": "Fifth", "tool_calls": []},
        ]

        result = get_unsummarized_visible_messages(messages, skip_first=1, skip_last=1)
        self.assertEqual([message["id"] for message in result], [3, 4])

    def test_chat_preflight_summary_respects_mode_never(self):
        conversation_id = self._create_conversation()
        dense_message = " ".join(["preflight"] * 900)
        save_app_settings(
            {
                "user_preferences": "",
                "max_steps": "1",
                "active_tools": "[]",
                "rag_auto_inject": "false",
                "chat_summary_mode": "never",
                "chat_summary_trigger_token_count": "1000",
                "summary_skip_first": "0",
                "summary_skip_last": "0",
            }
        )

        with get_db() as conn:
            for index in range(8):
                conn.execute(
                    "INSERT INTO messages (conversation_id, role, content, metadata) VALUES (?, 'user', ?, ?)",
                    (conversation_id, f"Seed {index + 1} {dense_message}", None),
                )

        fake_events = iter(
            [
                {"type": "answer_start"},
                {"type": "answer_delta", "text": "Main answer without preflight summary."},
                {"type": "tool_capture", "tool_results": []},
                {"type": "done"},
            ]
        )

        with patch("routes.chat.collect_agent_response") as mocked_collect, patch(
            "routes.chat.run_agent_stream", return_value=fake_events
        ), patch("routes.chat.sync_conversations_to_rag_safe"):
            response = self.client.post(
                "/chat",
                json={
                    "conversation_id": conversation_id,
                    "model": "deepseek-chat",
                    "user_content": f"Latest {dense_message}",
                    "messages": [{"role": "user", "content": f"Latest {dense_message}"}],
                },
            )

        self.assertEqual(response.status_code, 200)
        events = [json.loads(line) for line in response.get_data(as_text=True).strip().splitlines()]
        summary_status = next((event for event in events if event["type"] == "conversation_summary_status"), None)
        self.assertIsNotNone(summary_status)
        self.assertEqual(summary_status["reason"], "mode_never")
        mocked_collect.assert_not_called()

    def test_manual_summarize_endpoint_returns_404_for_missing_conversation(self):
        response = self.client.post(
            "/api/conversations/999999/summarize",
            json={"force": True},
        )
        self.assertEqual(response.status_code, 404)

    def test_manual_summarize_endpoint_can_force_summarize(self):
        conversation_id = self._create_conversation()
        dense_message = " ".join(["manual"] * 120)
        save_app_settings(
            {
                "user_preferences": "",
                "max_steps": "1",
                "active_tools": "[]",
                "rag_auto_inject": "false",
                "chat_summary_mode": "never",
                "chat_summary_trigger_token_count": "1000",
                "summary_skip_first": "0",
                "summary_skip_last": "0",
            }
        )

        with get_db() as conn:
            for index in range(12):
                conn.execute(
                    "INSERT INTO messages (conversation_id, role, content, metadata) VALUES (?, 'user', ?, ?)",
                    (conversation_id, f"Seed {index + 1} {dense_message}", None),
                )

        fake_summary = {
            "content": "Manual summary with enough retained detail to satisfy the minimum length validation threshold for summary generation results.",
            "reasoning_content": "",
            "usage": None,
            "tool_results": [],
            "errors": [],
        }
        with patch("routes.chat.collect_agent_response", return_value=fake_summary), patch(
            "routes.chat.sync_conversations_to_rag_safe"
        ):
            response = self.client.post(
                f"/api/conversations/{conversation_id}/summarize",
                json={"force": True},
            )

        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data["applied"])
        self.assertGreater(data["covered_message_count"], 0)

    def test_manual_summarize_endpoint_honors_false_force_strings(self):
        conversation_id = self._create_conversation()
        save_app_settings(
            {
                "user_preferences": "",
                "max_steps": "1",
                "active_tools": "[]",
                "rag_auto_inject": "false",
                "chat_summary_mode": "never",
                "chat_summary_trigger_token_count": "1000",
                "summary_skip_first": "0",
                "summary_skip_last": "0",
            }
        )

        with get_db() as conn:
            conn.execute(

                "INSERT INTO messages (conversation_id, role, content, metadata) VALUES (?, 'user', ?, ?)",
                (conversation_id, "Seed message", None),
            )

        with patch("routes.chat.collect_agent_response") as mocked_collect:
            response = self.client.post(
                f"/api/conversations/{conversation_id}/summarize",
                json={"force": "false"},
            )

        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertFalse(data["applied"])
        self.assertEqual(data["reason"], "mode_never")
        mocked_collect.assert_not_called()

    def test_manual_summarize_preserves_edges_and_inserts_summary_at_first_covered_slot(self):
        conversation_id = self._create_conversation()
        save_app_settings(
            {
                "user_preferences": "",
                "max_steps": "1",
                "active_tools": "[]",
                "rag_auto_inject": "false",
                "chat_summary_mode": "never",
                "chat_summary_trigger_token_count": "1000",
                "summary_skip_first": "2",
                "summary_skip_last": "2",
            }
        )

        with get_db() as conn:
            for index in range(8):
                conn.execute(
                    "INSERT INTO messages (conversation_id, role, content, metadata) VALUES (?, 'user', ?, ?)",
                    (conversation_id, f"Message {index + 1}", None),
                )

        fake_summary = {
            "content": "Summary block with enough retained detail to satisfy the minimum length validation threshold and preserve chronology across the covered messages.",
            "reasoning_content": "",
            "usage": None,
            "tool_results": [],
            "errors": [],
        }

        with patch("routes.chat.collect_agent_response", return_value=fake_summary), patch(
            "routes.chat.sync_conversations_to_rag_safe"
        ):
            response = self.client.post(
                f"/api/conversations/{conversation_id}/summarize",
                json={"force": True},
            )

        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data["applied"])
        self.assertEqual(data["covered_message_count"], 4)

        messages = data["messages"]
        self.assertEqual([message["content"] for message in messages if message["role"] == "user"], ["Message 1", "Message 2", "Message 7", "Message 8"])
        self.assertEqual([message["role"] for message in messages], ["user", "user", "summary", "user", "user"])

        summary_message = next(message for message in messages if message["role"] == "summary")
        self.assertEqual(summary_message["metadata"]["covered_message_ids"], [3, 4, 5, 6])
        self.assertEqual(summary_message["metadata"]["summary_insert_strategy"], "replace_first_covered_message_preserve_positions")
        self.assertEqual(summary_message["position"], 3)

    def test_visible_token_count_ignores_hidden_tool_messages_and_tool_call_assistants(self):
        visible_messages = [
            {"role": "user", "content": "Visible user text"},
            {"role": "assistant", "content": "Visible assistant text"},
            {"role": "tool", "content": "Hidden tool result"},
            {"role": "summary", "content": "Visible summary text"},
        ]
        mixed_messages = [
            {"role": "user", "content": "Visible user text"},
            {"role": "assistant", "content": "Visible assistant text"},
            {
                "role": "assistant",
                "content": "Hidden assistant tool call",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "search_web", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "content": "Hidden tool result"},
            {"role": "summary", "content": "Visible summary text"},
        ]

        self.assertEqual(count_visible_message_tokens(mixed_messages), count_visible_message_tokens(visible_messages))

    def test_undo_summary_restores_messages_in_original_order(self):
        conversation_id = self._create_conversation()
        save_app_settings(
            {
                "user_preferences": "",
                "max_steps": "1",
                "active_tools": "[]",
                "rag_auto_inject": "false",
                "chat_summary_mode": "never",
                "chat_summary_trigger_token_count": "1000",
                "summary_skip_first": "2",
                "summary_skip_last": "2",
            }
        )

        with get_db() as conn:
            for index in range(8):
                conn.execute(
                    "INSERT INTO messages (conversation_id, role, content, metadata) VALUES (?, 'user', ?, ?)",
                    (conversation_id, f"Message {index + 1}", None),
                )

        fake_summary = {
            "content": "Summary block with enough retained detail to satisfy the minimum length validation threshold and allow undo restoration.",
            "reasoning_content": "",
            "usage": None,
            "tool_results": [],
            "errors": [],
        }

        with patch("routes.chat.collect_agent_response", return_value=fake_summary), patch(
            "routes.chat.sync_conversations_to_rag_safe"
        ):
            summarize_response = self.client.post(
                f"/api/conversations/{conversation_id}/summarize",
                json={"force": True},
            )

        summary_message_id = summarize_response.get_json()["summary_message_id"]
        with patch("routes.chat.sync_conversations_to_rag_safe"):
            undo_response = self.client.post(
                f"/api/conversations/{conversation_id}/summaries/{summary_message_id}/undo"
            )

        self.assertEqual(undo_response.status_code, 200)
        data = undo_response.get_json()
        self.assertTrue(data["reverted"])
        self.assertEqual(data["restored_message_count"], 4)
        self.assertEqual([message["role"] for message in data["messages"]], ["user"] * 8)
        self.assertEqual([message["content"] for message in data["messages"]], [f"Message {index}" for index in range(1, 9)])
        self.assertEqual([message["position"] for message in data["messages"]], list(range(1, 9)))

    def test_undo_summary_restores_legacy_summary_layout(self):
        conversation_id = self._create_conversation()

        with get_db() as conn:
            inserted_ids = []
            for index in range(8):
                inserted_ids.append(
                    conn.execute(
                        "INSERT INTO messages (conversation_id, role, content, metadata, position) VALUES (?, 'user', ?, ?, ?)",
                        (conversation_id, f"Message {index + 1}", None, index + 1),
                    ).lastrowid
                )

            deleted_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
            conn.execute(
                "UPDATE messages SET deleted_at = ? WHERE conversation_id = ? AND id IN (?, ?, ?, ?)",
                (deleted_at, conversation_id, inserted_ids[2], inserted_ids[3], inserted_ids[4], inserted_ids[5]),
            )
            conn.execute(
                "UPDATE messages SET position = 4 WHERE id = ?",
                (inserted_ids[6],),
            )
            conn.execute(
                "UPDATE messages SET position = 5 WHERE id = ?",
                (inserted_ids[7],),
            )
            summary_id = conn.execute(
                "INSERT INTO messages (conversation_id, role, content, metadata, position) VALUES (?, 'summary', ?, ?, 3)",
                (
                    conversation_id,
                    "Conversation summary (generated from deleted messages):\n\nLegacy summary block.",
                    serialize_message_metadata(
                        {
                            "is_summary": True,
                            "summary_source": "conversation_history",
                            "covers_from_position": 3,
                            "covers_to_position": 6,
                            "covered_message_count": 4,
                            "covered_message_ids": [inserted_ids[2], inserted_ids[3], inserted_ids[4], inserted_ids[5]],
                            "generated_at": deleted_at,
                        }
                    ),
                ),
            ).lastrowid

        with patch("routes.chat.sync_conversations_to_rag_safe"):
            response = self.client.post(
                f"/api/conversations/{conversation_id}/summaries/{summary_id}/undo"
            )

        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data["reverted"])
        self.assertEqual([message["content"] for message in data["messages"]], [f"Message {index}" for index in range(1, 9)])
        self.assertEqual([message["position"] for message in data["messages"]], list(range(1, 9)))

    def test_undo_summary_restores_tool_chain_missing_from_metadata(self):
        conversation_id = self._create_conversation()

        with get_db() as conn:
            user_one_id = conn.execute(
                "INSERT INTO messages (conversation_id, role, content, metadata, position) VALUES (?, 'user', ?, ?, ?)",
                (conversation_id, "Message 1", None, 1),
            ).lastrowid
            assistant_tool_id = conn.execute(
                "INSERT INTO messages (conversation_id, role, content, tool_calls, position) VALUES (?, 'assistant', ?, ?, ?)",
                (
                    conversation_id,
                    "Working on it.",
                    json.dumps(
                        [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {"name": "search_web", "arguments": "{}"},
                            }
                        ]
                    ),
                    2,
                ),
            ).lastrowid
            tool_id = conn.execute(
                "INSERT INTO messages (conversation_id, role, content, tool_call_id, position) VALUES (?, 'tool', ?, ?, ?)",
                (conversation_id, '{"ok": true}', "call-1", 3),
            ).lastrowid
            user_two_id = conn.execute(
                "INSERT INTO messages (conversation_id, role, content, metadata, position) VALUES (?, 'user', ?, ?, ?)",
                (conversation_id, "Message 4", None, 4),
            ).lastrowid
            user_three_id = conn.execute(
                "INSERT INTO messages (conversation_id, role, content, metadata, position) VALUES (?, 'user', ?, ?, ?)",
                (conversation_id, "Message 5", None, 5),
            ).lastrowid

            deleted_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
            conn.execute(
                "UPDATE messages SET deleted_at = ? WHERE conversation_id = ? AND id IN (?, ?, ?)",
                (deleted_at, conversation_id, assistant_tool_id, tool_id, user_two_id),
            )
            summary_id = conn.execute(
                "INSERT INTO messages (conversation_id, role, content, metadata, position) VALUES (?, 'summary', ?, ?, ?)",
                (
                    conversation_id,
                    "Conversation summary (generated from deleted messages):\n\nSummary block.",
                    serialize_message_metadata(
                        {
                            "is_summary": True,
                            "summary_source": "conversation_history",
                            "covers_from_position": 2,
                            "covers_to_position": 4,
                            "summary_insert_strategy": "replace_first_covered_message_preserve_positions",
                            "covered_message_count": 1,
                            "covered_message_ids": [user_two_id],
                            "generated_at": deleted_at,
                        }
                    ),
                    2,
                ),
            ).lastrowid

        with patch("routes.chat.sync_conversations_to_rag_safe"):
            response = self.client.post(
                f"/api/conversations/{conversation_id}/summaries/{summary_id}/undo"
            )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["reverted"])
        self.assertEqual(payload["restored_message_count"], 3)
        self.assertEqual(
            [(message["role"], message.get("tool_call_id")) for message in payload["messages"]],
            [
                ("user", None),
                ("assistant", None),
                ("tool", "call-1"),
                ("user", None),
                ("user", None),
            ],
        )
        self.assertEqual([message["position"] for message in payload["messages"]], [1, 2, 3, 4, 5])

        with get_db() as conn:
            visible_ids = {
                row["id"]
                for row in conn.execute(
                    "SELECT id FROM messages WHERE conversation_id = ? AND deleted_at IS NULL",
                    (conversation_id,),
                ).fetchall()
            }

        self.assertTrue({user_one_id, assistant_tool_id, tool_id, user_two_id, user_three_id}.issubset(visible_ids))

    def test_settings_include_new_summary_params(self):
        save_app_settings(
            {
                "user_preferences": "",
                "max_steps": "5",
                "active_tools": "[]",
                "summary_skip_first": "3",
                "summary_skip_last": "2",
            }
        )
        response = self.client.get("/api/settings")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["summary_skip_first"], 3)
        self.assertEqual(data["summary_skip_last"], 2)

    def test_settings_patch_validates_new_summary_params(self):
        response = self.client.patch(
            "/api/settings",
            json={"summary_skip_first": 5, "summary_skip_last": 3},
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["summary_skip_first"], 5)
        self.assertEqual(data["summary_skip_last"], 3)

    def test_settings_patch_rejects_invalid_skip_values(self):
        response = self.client.patch(
            "/api/settings",
            json={"summary_skip_first": 25},
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)

    def test_structured_summary_prompt_contains_sections(self):
        prompt_messages = build_summary_prompt_messages(
            [
                {"role": "user", "content": "Tell me about Python"},
                {"role": "assistant", "content": "Python is a programming language."},
            ],
            "",
        )
        system_content = prompt_messages[0]["content"]
        self.assertIn("User Goals & Intentions", system_content)
        self.assertIn("Key Facts & Information", system_content)
        self.assertIn("Decisions & Agreements", system_content)
        self.assertIn("Unresolved Questions", system_content)
        self.assertIn("Important Context", system_content)
        self.assertIn("sufficient detail", system_content)

    def test_chat_applies_preflight_summary_before_main_agent_run(self):
        conversation_id = self._create_conversation()
        dense_message = " ".join(["preflight"] * 900)
        save_app_settings(
            {
                "user_preferences": "",
                "max_steps": "1",
                "active_tools": "[]",
                "rag_auto_inject": "false",
                "chat_summary_mode": "auto",
                "chat_summary_trigger_token_count": "80000",
                "summary_skip_first": "0",
                "summary_skip_last": "0",
            }
        )

        with get_db() as conn:
            for index in range(20):
                conn.execute(
                    "INSERT INTO messages (conversation_id, role, content, metadata) VALUES (?, 'user', ?, ?)",
                    (conversation_id, f"Seed {index + 1} {dense_message}", None),
                )

        fake_summary = {
            "content": "Preflight summary block with enough retained detail to satisfy the minimum length validation threshold and reduce prompt size before the main answer.",
            "reasoning_content": "",
            "usage": None,
            "tool_results": [],
            "errors": [],
        }
        fake_events = iter(
            [
                {"type": "answer_start"},
                {"type": "answer_delta", "text": "Main answer after preflight summary."},
                {"type": "tool_capture", "tool_results": []},
                {"type": "done"},
            ]
        )

        with patch("routes.chat.get_prompt_preflight_summary_token_count", return_value=1000), patch(
            "routes.chat.collect_agent_response", return_value=fake_summary
        ), patch("routes.chat.run_agent_stream", return_value=fake_events), patch("routes.chat.sync_conversations_to_rag_safe"):
            response = self.client.post(
                "/chat",
                json={
                    "conversation_id": conversation_id,
                    "model": "deepseek-chat",
                    "user_content": f"Latest {dense_message}",
                    "messages": [{"role": "user", "content": f"Latest {dense_message}"}],
                },
            )

        self.assertEqual(response.status_code, 200)
        events = [json.loads(line) for line in response.get_data(as_text=True).strip().splitlines()]
        summary_event = next((event for event in events if event["type"] == "conversation_summary_applied"), None)
        self.assertIsNotNone(summary_event)
        self.assertTrue(summary_event["preflight"])
        history_sync_event = next((event for event in events if event["type"] == "history_sync"), None)
        self.assertIsNotNone(history_sync_event)
        self.assertEqual(history_sync_event["messages"][0]["role"], "summary")
        self.assertIn(SUMMARY_LABEL, history_sync_event["messages"][0]["content"])

    def test_chat_preflight_summary_skips_second_summary_pass_in_same_request(self):
        conversation_id = self._create_conversation()
        dense_message = " ".join(["preflight"] * 900)
        save_app_settings(
            {
                "user_preferences": "",
                "max_steps": "1",
                "active_tools": "[]",
                "rag_auto_inject": "false",
                "chat_summary_mode": "auto",
                "chat_summary_trigger_token_count": "80000",
                "summary_skip_first": "0",
                "summary_skip_last": "0",
            }
        )

        with get_db() as conn:
            for index in range(20):
                conn.execute(
                    "INSERT INTO messages (conversation_id, role, content, metadata) VALUES (?, 'user', ?, ?)",
                    (conversation_id, f"Seed {index + 1} {dense_message}", None),
                )

        fake_summary = {
            "content": "Preflight summary block with enough retained detail to satisfy the minimum length validation threshold and reduce prompt size before the main answer.",
            "reasoning_content": "",
            "usage": None,
            "tool_results": [],
            "errors": [],
        }
        fake_events = iter(
            [
                {"type": "answer_start"},
                {"type": "answer_delta", "text": "Main answer after preflight summary."},
                {"type": "tool_capture", "tool_results": []},
                {"type": "done"},
            ]
        )

        with patch("routes.chat.get_prompt_preflight_summary_token_count", return_value=1000), patch(
            "routes.chat.collect_agent_response", return_value=fake_summary
        ), patch("routes.chat.run_agent_stream", return_value=fake_events), patch(
            "routes.chat.SUMMARY_EXECUTOR.submit"
        ) as mocked_summary_submit, patch("routes.chat.sync_conversations_to_rag_safe"):
            response = self.client.post(
                "/chat",
                json={
                    "conversation_id": conversation_id,
                    "model": "deepseek-chat",
                    "user_content": f"Latest {dense_message}",
                    "messages": [{"role": "user", "content": f"Latest {dense_message}"}],
                },
            )
            response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        mocked_summary_submit.assert_not_called()

    def test_run_agent_stream_uses_compact_tool_message_content_for_followup_prompt(self):
        responses = [
            iter(
                [
                    self._tool_call_chunk("fetch_url", {"url": "https://example.com/compact"}),
                    self._stream_chunk(usage=SimpleNamespace(prompt_tokens=3, completion_tokens=5, total_tokens=8)),
                ]
            ),
            iter(
                [
                    self._stream_chunk(content="Compact answer."),
                    self._stream_chunk(usage=SimpleNamespace(prompt_tokens=4, completion_tokens=6, total_tokens=10)),
                ]
            ),
        ]

        long_content = "\n\n".join(
            [
                "Overview block with broad context and repeated details. " * 12,
                "Evidence block with implementation details and caveats. " * 12,
                "Closing block with more raw detail that should stay out of the prompt-facing tool message. " * 12,
            ]
        )

        with patch("agent.client.chat.completions.create", side_effect=responses) as mocked_create, patch(
            "agent.fetch_url_tool",
            return_value={
                "url": "https://example.com/compact",
                "title": "Compact Example",
                "content": long_content,
                "status": 200,
                "content_format": "html",
                "cleanup_applied": True,
            },
        ):
            list(
                run_agent_stream(
                    [{"role": "user", "content": "Fetch and summarize compactly"}],
                    "deepseek-chat",
                    2,
                    ["fetch_url"],
                    fetch_url_token_threshold=50,
                )
            )

        second_call_messages = mocked_create.call_args_list[1].kwargs["messages"]
        tool_message = next(message for message in second_call_messages if message["role"] == "tool")
        self.assertIn("Title: Compact Example", tool_message["content"])
        self.assertIn("URL: https://example.com/compact", tool_message["content"])
        self.assertNotIn("raw_content", tool_message["content"])


if __name__ == "__main__":
    unittest.main()
