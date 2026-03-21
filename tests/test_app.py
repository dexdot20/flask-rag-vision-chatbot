from __future__ import annotations

import io
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import requests as http_requests
import web_tools

from agent import FINAL_ANSWER_ERROR_TEXT, FINAL_ANSWER_MISSING_TEXT, _execute_tool, collect_agent_response, run_agent_stream
from app import create_app
from db import (
    append_to_scratchpad,
    create_image_asset,
    get_app_settings,
    get_db,
    get_active_tool_names,
    get_image_asset,
    normalize_active_tool_names,
    parse_message_metadata,
    save_app_settings,
    serialize_message_metadata,
)
from messages import build_api_messages, build_runtime_system_message, build_user_message_for_model, normalize_chat_messages, prepend_runtime_context
from tool_registry import TOOL_SPEC_BY_NAME
from web_tools import _extract_html, fetch_url_tool, load_proxies, search_news_ddgs_tool, search_news_google_tool, search_web_tool


class AppRoutesTestCase(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = f"{self.temp_dir.name}/test.db"
        self.image_storage_dir = f"{self.temp_dir.name}/image-store"
        self.app = create_app(database_path=self.db_path)
        self.app.config.update(TESTING=True)
        self.client = self.app.test_client()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_conversation(self, title: str = "Test Chat") -> int:
        response = self.client.post(
            "/api/conversations",
            json={"title": title, "model": "deepseek-chat"},
        )
        self.assertEqual(response.status_code, 201)
        return response.get_json()["id"]

    @staticmethod
    def _stream_chunk(reasoning: str = "", content: str = "", usage=None):
        return SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(reasoning_content=reasoning, content=content))] if (reasoning or content) else [],
            usage=usage,
        )

    def test_settings_roundtrip(self):
        response = self.client.get("/api/settings")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["scratchpad"], "")
        self.assertEqual(payload["max_steps"], 5)
        self.assertTrue(payload["rag_auto_inject"])
        self.assertEqual(payload["chat_summary_trigger_message_count"], 40)
        self.assertEqual(payload["chat_summary_batch_size"], 20)
        self.assertEqual(payload["fetch_url_token_threshold"], 3500)
        self.assertEqual(payload["fetch_url_clip_aggressiveness"], 50)
        self.assertIn("features", payload)
        self.assertTrue(payload["features"]["rag_enabled"])
        self.assertTrue(payload["features"]["vision_enabled"])

        response = self.client.patch(
            "/api/settings",
            json={
                "user_preferences": "Keep answers short.",
                "max_steps": 3,
                "chat_summary_trigger_message_count": 60,
                "chat_summary_batch_size": 15,
                "fetch_url_token_threshold": 4200,
                "fetch_url_clip_aggressiveness": 70,
                "active_tools": ["fetch_url", "search_web"],
                "rag_auto_inject": False,
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["user_preferences"], "Keep answers short.")
        self.assertEqual(payload["scratchpad"], "")
        self.assertEqual(payload["max_steps"], 3)
        self.assertEqual(payload["chat_summary_trigger_message_count"], 60)
        self.assertEqual(payload["chat_summary_batch_size"], 15)
        self.assertEqual(payload["fetch_url_token_threshold"], 4200)
        self.assertEqual(payload["fetch_url_clip_aggressiveness"], 70)
        self.assertEqual(payload["active_tools"], ["fetch_url", "search_web"])
        self.assertFalse(payload["rag_auto_inject"])

    def test_create_conversation_rejects_invalid_model(self):
        response = self.client.post(
            "/api/conversations",
            json={"title": "Test Chat", "model": "invalid-model"},
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["error"], "Invalid model.")

    def test_active_tools_include_replace_scratchpad_for_existing_scratchpad_mode(self):
        settings = {"active_tools": json.dumps(["append_scratchpad", "search_web"]) }
        self.assertIn("replace_scratchpad", get_active_tool_names(settings))

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
            active_tool_names=["append_scratchpad", "ask_clarifying_question", "image_explain", "search_knowledge_base"],
            retrieved_context="Context block",
            now=now,
        )

        self.assertEqual(message["role"], "system")
        payload = json.loads(message["content"])
        self.assertEqual(payload["context_type"], "runtime_prompt_context")
        self.assertEqual(
            payload["current_datetime"],
            {
                "iso": "2026-03-15T21:42:05+03:00",
                "date": "2026-03-15",
                "time": "21:42:05",
                "weekday": "Sunday",
                "timezone": "UTC+03:00",
            },
        )
        self.assertEqual(payload["user_preferences"], "Keep answers short.")
        self.assertEqual(payload["scratchpad"]["content"], "The user is 22 years old.")
        self.assertIn("durable user facts", payload["scratchpad"]["memory_write_policy"])
        self.assertEqual(payload["clarification_policy"]["tool"], "ask_clarifying_question")
        self.assertIn("missing requirements", payload["clarification_policy"]["guidance"])
        self.assertEqual(payload["image_follow_up_policy"]["tool"], "image_explain")
        self.assertIn("stored prior image", payload["image_follow_up_policy"]["guidance"])
        self.assertIsNotNone(payload["knowledge_base"])
        self.assertEqual(payload["knowledge_base"]["auto_injected_context"], "Context block")

    def test_prepend_runtime_context_places_datetime_system_message_first(self):
        messages = prepend_runtime_context(
            [{"role": "user", "content": "Hello"}],
            user_preferences="",
            active_tool_names=[],
            scratchpad="Persistent note",
        )

        self.assertEqual(messages[0]["role"], "system")
        
        # Extract the JSON part from inside the ```json block
        content = messages[0]["content"]
        json_str = content.split("```json")[-1].split("```")[0].strip()
        payload = json.loads(json_str)
        
        self.assertIn("current_datetime", payload)
        self.assertEqual(payload["scratchpad"]["content"], "Persistent note")
        self.assertIsNone(payload["user_preferences"])
        self.assertIn("date", payload["current_datetime"])
        self.assertIn("time", payload["current_datetime"])
        self.assertEqual(messages[1]["role"], "user")

    def test_settings_patch_rejects_manual_scratchpad_updates(self):
        response = self.client.patch(
            "/api/settings",
            json={"scratchpad": "manual override"},
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("read-only", response.get_json()["error"])

    def test_settings_patch_allows_manual_scratchpad_updates_in_admin_mode(self):
        with patch("config.SCRATCHPAD_ADMIN_EDITING_ENABLED", True):
            response = self.client.patch(
                "/api/settings",
                json={"scratchpad": "The user likes concise answers.\nThe user likes concise answers.\n"},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["scratchpad"], "The user likes concise answers.")
        self.assertTrue(payload["features"]["scratchpad_admin_editing"])

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
        self.assertEqual(duplicate_summary, "Scratchpad note already exists")
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
        json_str = full_content.split("```json")[-1].split("```")[0].strip()
        runtime_payload = json.loads(json_str)
        self.assertEqual(runtime_payload["scratchpad"]["content"], "The user prefers concise answers.")

    def test_index_uses_external_app_script(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn("marked/marked.min.js", html)
        self.assertIn("dompurify/dist/purify.min.js", html)
        self.assertIn('id="app-bootstrap"', html)
        self.assertIn('id="scratchpad-output"', html)
        self.assertIn('id="scratchpad-admin-block"', html)

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
        self.assertIn('document.getElementById("scratchpad-output")', script_text)
        self.assertIn('document.getElementById("scratchpad-input")', script_text)
        self.assertIn("function renderScratchpad", script_text)
        self.assertIn('fetch("/api/settings")', script_text)
        self.assertIn("scratchpad_admin_editing", script_text)
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

    def test_reasoning_panel_uses_markdown_rendering(self):
        script_path = Path(__file__).resolve().parent.parent / "static" / "app.js"
        script_text = script_path.read_text(encoding="utf-8")
        self.assertIn('body.innerHTML = renderMarkdown(text);', script_text)

    def test_reasoning_css_includes_markdown_styles(self):
        style_path = Path(__file__).resolve().parent.parent / "static" / "style.css"
        style_text = style_path.read_text(encoding="utf-8")
        self.assertIn(".reasoning-body code", style_text)
        self.assertIn(".reasoning-body ul", style_text)

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
        self.assertIn("NodeFilter.SHOW_TEXT", script_text)
        self.assertIn("insertBefore(cursor", script_text)

        style_path = Path(__file__).resolve().parent.parent / "static" / "style.css"
        style_text = style_path.read_text(encoding="utf-8")
        self.assertIn(".clarification-card", style_text)
        self.assertIn(".clarification-form", style_text)

    def test_settings_ui_exposes_fetch_threshold_input(self):
        html = self.client.get("/").get_data(as_text=True)
        self.assertIn('value="append_scratchpad"', html)
        self.assertIn('value="ask_clarifying_question"', html)
        self.assertIn('id="scratchpad-output"', html)
        self.assertIn('id="scratchpad-input"', html)
        self.assertIn('id="summary-trigger-input"', html)
        self.assertIn('id="summary-batch-input"', html)
        self.assertIn('id="fetch-threshold-input"', html)
        self.assertIn('id="fetch-aggressiveness-input"', html)

    def test_run_agent_stream_executes_append_scratchpad_tool(self):
        responses = [
            iter(
                [
                    self._stream_chunk(content='{"tool_calls":[{"name":"append_scratchpad","arguments":{"note":"The user is 22 years old."}}]}'),
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
                    self._stream_chunk(
                        content=(
                            '{"tool_calls":[{"name":"ask_clarifying_question","arguments":'
                            '{"intro":"Before I answer, I need two details.","questions":['
                            '{"id":"scope","label":"Which scope?","input_type":"single_select","options":['
                            '{"label":"Only this repo","value":"repo"},'
                            '{"label":"General tool","value":"general"}]},'
                            '{"id":"notes","label":"Anything else?","input_type":"text","required":false}]}}]}'
                        )
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
                        "system_prompt": 4,
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
        self.assertEqual(assistant_metadata["usage"]["estimated_input_tokens"], 14)
        self.assertEqual(assistant_metadata["usage"]["input_breakdown"]["user_messages"], 6)
        self.assertEqual(rows[1]["prompt_tokens"], 11)
        self.assertEqual(rows[1]["completion_tokens"], 7)
        self.assertEqual(rows[1]["total_tokens"], 18)

        conversation_response = self.client.get(f"/api/conversations/{conversation_id}")
        self.assertEqual(conversation_response.status_code, 200)
        assistant_message = conversation_response.get_json()["messages"][1]
        self.assertEqual(assistant_message["usage"]["estimated_input_tokens"], 14)
        self.assertEqual(assistant_message["usage"]["input_breakdown"]["system_prompt"], 4)
        self.assertEqual(assistant_message["metadata"]["tool_trace"][0]["tool_name"], "search_web")

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

    def test_run_agent_stream_executes_custom_json_tool_calls(self):
        responses = [
            iter(
                [
                    self._stream_chunk(reasoning="Need current info. "),
                    self._stream_chunk(content='{"tool_calls":[{"name":"search_web","arguments":{"queries":["test query"]}}]}'),
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
            self.assertNotIn("tools", kwargs)
            self.assertNotIn("tool_choice", kwargs)
            self.assertTrue(kwargs.get("stream"))

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

        second_call_messages = mocked_create.call_args_list[1].kwargs["messages"]
        self.assertEqual([message["role"] for message in second_call_messages], ["user", "assistant", "tool"])
        self.assertEqual(second_call_messages[1]["tool_calls"][0]["function"]["name"], "search_web")
        self.assertEqual(second_call_messages[2]["tool_call_id"], "tool-call-1")

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

        second_call_messages = mocked_create.call_args_list[1].kwargs["messages"]
        retry_instruction = json.loads(second_call_messages[-1]["content"])
        self.assertEqual(retry_instruction["context_type"], "final_answer_retry")

    def test_run_agent_stream_separates_reasoning_turns_with_blank_line(self):
        responses = [
            iter(
                [
                    self._stream_chunk(reasoning="First reasoning block."),
                    self._stream_chunk(content='{"tool_calls":[{"name":"search_web","arguments":{"queries":["x"]}}]}'),
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
                    self._stream_chunk(
                        content='{"tool_calls":[{"name":"fetch_url","arguments":{"url":"https://example.com/long"}}]}'
                    ),
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
        transcript_payload = json.loads(second_call_messages[-1]["content"])
        transcript_result = transcript_payload["tool_results"][0]["result"]
        self.assertEqual(transcript_result["content_mode"], "clipped_text")
        self.assertIn("cleaned and clipped", transcript_result["summary_notice"])
        self.assertTrue(long_content.startswith(transcript_result["content"].rstrip("…")))
        self.assertNotEqual(transcript_result["content"], long_content)

    def test_run_agent_stream_marks_fetch_failures_clearly_in_transcript(self):
        responses = [
            iter(
                [
                    self._stream_chunk(
                        content='{"tool_calls":[{"name":"fetch_url","arguments":{"url":"https://example.com/blocked"}}]}'
                    ),
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
        transcript_payload = json.loads(second_call_messages[-1]["content"])
        self.assertIn("For fetch_url results", transcript_payload["fetch_guidance"])
        transcript_entry = transcript_payload["tool_results"][0]
        self.assertFalse(transcript_entry["ok"])
        self.assertEqual(transcript_entry["result"]["fetch_outcome"], "error")
        self.assertIn("HTTP 403", transcript_entry["result"]["fetch_diagnostic"])

    def test_run_agent_stream_marks_partial_fetch_as_already_attempted(self):
        responses = [
            iter(
                [
                    self._stream_chunk(
                        content='{"tool_calls":[{"name":"fetch_url","arguments":{"url":"https://example.com/page"}}]}'
                    ),
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
        transcript_payload = json.loads(second_call_messages[-1]["content"])
        transcript_entry = transcript_payload["tool_results"][0]
        self.assertTrue(transcript_entry["ok"])
        self.assertEqual(transcript_entry["result"]["fetch_outcome"], "partial_content")
        self.assertIn("partial page content", transcript_entry["result"]["fetch_diagnostic"].lower())

    def test_run_agent_stream_logs_duplicate_fetch_url_calls(self):
        responses = [
            iter(
                [
                    self._stream_chunk(
                        content='{"tool_calls":[{"name":"fetch_url","arguments":{"url":"https://example.com/page"}}]}'
                    ),
                    self._stream_chunk(usage=SimpleNamespace(prompt_tokens=3, completion_tokens=5, total_tokens=8)),
                ]
            ),
            iter(
                [
                    self._stream_chunk(
                        content='{"tool_calls":[{"name":"fetch_url","arguments":{"url":"https://example.com/page"}}]}'
                    ),
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
        transcript_payload = json.loads(third_call_messages[-1]["content"])
        transcript_entry = transcript_payload["tool_results"][0]
        self.assertTrue(transcript_entry["ok"])
        self.assertTrue(transcript_entry["cached"])
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

    def test_run_agent_stream_recovers_tool_json_from_mixed_text(self):
        responses = [
            iter(
                [
                    self._stream_chunk(content='Before\n```json\n{"tool_calls":[{"name":"search_web","arguments":{"queries":["x"]}}]}\n```'),
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

        parser_errors = [event for event in events if event["type"] == "tool_error" and event["tool"] == "parser"]
        self.assertEqual(parser_errors, [])
        self.assertIn("step_update", [event["type"] for event in events])
        self.assertIn("tool_result", [event["type"] for event in events])
        self.assertIn({"type": "answer_delta", "text": "Recovered answer"}, events)

    def test_run_agent_stream_suppresses_pre_tool_text_from_answer_and_injects_as_assistant_message(self):
        """When model outputs text before a ```json tool_call block:
        - The text must NOT leak into answer_delta events (no raw JSON shown to user)
        - The text must be injected as an assistant message so the model sees its own commitment
        """
        captured_calls = []

        def fake_create(model, messages, stream, stream_options=None):
            captured_calls.append(list(messages))
            call_index = len(captured_calls)
            if call_index == 1:
                # Step 1: model outputs narrative text + tool_call JSON block
                return iter(
                    [
                        self._stream_chunk(
                            content='Okay, I will check now.\n\n```json\n{"tool_calls":[{"name":"search_web","arguments":{"queries":["test"]}}]}\n```'
                        ),
                        self._stream_chunk(
                            usage=SimpleNamespace(prompt_tokens=5, completion_tokens=20, total_tokens=25)
                        ),
                    ]
                )
            else:
                # Step 2: model returns final answer
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

        # Raw JSON block must NOT appear in any answer_delta
        answer_texts = [event["text"] for event in events if event["type"] == "answer_delta"]
        combined_answer = "".join(answer_texts)
        self.assertNotIn("tool_calls", combined_answer, "Raw JSON tool_call must not leak into answer stream")
        self.assertNotIn("```", combined_answer, "Fenced code block must not leak into answer stream")

        # The final answer should arrive correctly
        self.assertIn("Here is the result.", combined_answer)

        # The second model call must include an assistant message with the pre-tool narrative text
        self.assertGreaterEqual(len(captured_calls), 2, "Expected at least 2 model calls")
        second_call_messages = captured_calls[1]
        assistant_messages = [m for m in second_call_messages if m.get("role") == "assistant"]
        pre_tool_contents = [m["content"] for m in assistant_messages if "Okay, I will check now" in m.get("content", "")]
        self.assertTrue(
            len(pre_tool_contents) > 0,
            "Pre-tool narrative text must be injected as an assistant message for the next turn",
        )

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
                "SELECT id, role, content FROM messages WHERE conversation_id = ? ORDER BY id",
                (conversation_id,),
            ).fetchall()

        self.assertEqual(
            [(row["id"], row["role"], row["content"]) for row in rows],
            [
                (user_one_id, "user", "First question"),
                (first_assistant_id, "assistant", "First answer"),
                (edited_user_id, "user", "New message"),
                (message_ids_event["assistant_message_id"], "assistant", "New answer"),
            ],
        )
        self.assertFalse(any(row["id"] == stale_assistant_id for row in rows))

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
        save_app_settings(
            {
                "user_preferences": "",
                "max_steps": "1",
                "active_tools": "[]",
                "rag_auto_inject": "false",
                "chat_summary_trigger_message_count": "40",
                "chat_summary_batch_size": "20",
            }
        )

        with get_db() as conn:
            for index in range(39):
                conn.execute(
                    "INSERT INTO messages (conversation_id, role, content, metadata) VALUES (?, 'user', ?, ?)",
                    (conversation_id, f"Message {index + 1}", None),
                )

        fake_summary = {
            "content": "Summary of the first 20 messages.",
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

        with patch("routes.chat.collect_agent_response", return_value=fake_summary), patch(
            "routes.chat.run_agent_stream", return_value=fake_events
        ):
            response = self.client.post(
                "/chat",
                json={
                    "conversation_id": conversation_id,
                    "model": "deepseek-chat",
                    "user_content": "Message 40",
                    "messages": [{"role": "user", "content": "Message 40"}],
                },
            )

        self.assertEqual(response.status_code, 200)
        streamed_events = [json.loads(line) for line in response.get_data(as_text=True).strip().splitlines()]
        conversation_response = self.client.get(f"/api/conversations/{conversation_id}")
        self.assertEqual(conversation_response.status_code, 200)
        conversation_messages = conversation_response.get_json()["messages"]
        self.assertEqual(conversation_messages[0]["role"], "summary")
        self.assertTrue(conversation_messages[0]["metadata"]["is_summary"])
        self.assertEqual(conversation_messages[0]["metadata"]["covered_message_count"], 20)
        self.assertEqual(conversation_messages[0]["metadata"]["trigger_threshold"], 40)
        self.assertEqual(conversation_messages[0]["metadata"]["summary_batch_size"], 20)
        self.assertEqual(
            conversation_messages[0]["content"],
            "Conversation summary (generated from deleted messages):\n\nSummary of the first 20 messages.",
        )
        self.assertEqual(len(conversation_messages), 22)
        self.assertEqual(conversation_messages[-1]["role"], "assistant")

        history_sync_event = next((event for event in streamed_events if event["type"] == "history_sync"), None)
        self.assertIsNotNone(history_sync_event)
        self.assertEqual(history_sync_event["messages"][0]["role"], "summary")

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

    def test_chat_summary_preserves_interleaved_tool_messages(self):
        conversation_id = self._create_conversation()
        save_app_settings(
            {
                "user_preferences": "",
                "max_steps": "1",
                "active_tools": "[]",
                "rag_auto_inject": "false",
                "chat_summary_trigger_message_count": "10",
                "chat_summary_batch_size": "5",
            }
        )

        with get_db() as conn:
            conn.execute(
                "INSERT INTO messages (conversation_id, role, content, metadata) VALUES (?, 'user', ?, ?)",
                (conversation_id, "First user message", None),
            )
            conn.execute(
                "INSERT INTO messages (conversation_id, role, content, metadata) VALUES (?, 'assistant', ?, ?)",
                (conversation_id, "First answer", None),
            )
            conn.execute(
                "INSERT INTO messages (conversation_id, role, content, tool_call_id) VALUES (?, 'tool', ?, ?)",
                (conversation_id, '{"ok":true}', "call-1"),
            )
            for index in range(7):
                conn.execute(
                    "INSERT INTO messages (conversation_id, role, content, metadata) VALUES (?, 'user', ?, ?)",
                    (conversation_id, f"Follow-up user message {index + 1}", None),
                )

        fake_summary = {
            "content": "Summary of the first five visible messages.",
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

        with patch("routes.chat.collect_agent_response", return_value=fake_summary), patch(
            "routes.chat.run_agent_stream", return_value=fake_events
        ):
            response = self.client.post(
                "/chat",
                json={
                    "conversation_id": conversation_id,
                    "model": "deepseek-chat",
                    "user_content": "Third user message",
                    "messages": [{"role": "user", "content": "Third user message"}],
                },
            )

        self.assertEqual(response.status_code, 200)
        response.get_data(as_text=True)

        with get_db() as conn:
            rows = conn.execute(
                "SELECT role, content, tool_call_id FROM messages WHERE conversation_id = ? ORDER BY position, id",
                (conversation_id,),
            ).fetchall()

        self.assertEqual(
            [(row["role"], row["tool_call_id"]) for row in rows],
            [
                ("summary", None),
                ("tool", "call-1"),
                ("user", None),
                ("user", None),
                ("user", None),
                ("user", None),
                ("user", None),
                ("assistant", None),
            ],
        )
        self.assertEqual(rows[1]["content"], '{"ok":true}')

    def test_chat_can_create_multiple_summary_passes_without_resummarizing_old_ones(self):
        conversation_id = self._create_conversation()
        save_app_settings(
            {
                "user_preferences": "",
                "max_steps": "1",
                "active_tools": "[]",
                "rag_auto_inject": "false",
                "chat_summary_trigger_message_count": "10",
                "chat_summary_batch_size": "5",
            }
        )

        with get_db() as conn:
            for index in range(13):
                conn.execute(
                    "INSERT INTO messages (conversation_id, role, content, metadata) VALUES (?, 'user', ?, ?)",
                    (conversation_id, f"Seed {index + 1}", None),
                )

        first_summary = {
            "content": "First summary block.",
            "reasoning_content": "",
            "usage": None,
            "tool_results": [],
            "errors": [],
        }
        second_summary = {
            "content": "Second summary block.",
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
            ):
                response = self.client.post(
                    "/chat",
                    json={
                        "conversation_id": conversation_id,
                        "model": "deepseek-chat",
                        "user_content": user_text,
                        "messages": [{"role": "user", "content": user_text}],
                    },
                )
            self.assertEqual(response.status_code, 200)
            response.get_data(as_text=True)

        conversation_response = self.client.get(f"/api/conversations/{conversation_id}")
        self.assertEqual(conversation_response.status_code, 200)
        messages = conversation_response.get_json()["messages"]
        summary_messages = [message for message in messages if message["role"] == "summary"]

        self.assertEqual(len(summary_messages), 2)
        self.assertEqual(
            summary_messages[0]["content"],
            "Conversation summary (generated from deleted messages):\n\nFirst summary block.",
        )
        self.assertEqual(
            summary_messages[1]["content"],
            "Conversation summary (generated from deleted messages):\n\nSecond summary block.",
        )
        self.assertEqual(summary_messages[0]["metadata"]["covered_message_count"], 5)
        self.assertEqual(summary_messages[1]["metadata"]["covered_message_count"], 5)
        first_ids = set(summary_messages[0]["metadata"]["covered_message_ids"])
        second_ids = set(summary_messages[1]["metadata"]["covered_message_ids"])
        self.assertTrue(first_ids)
        self.assertTrue(second_ids)
        self.assertTrue(first_ids.isdisjoint(second_ids))

    def test_run_agent_stream_blocks_tool_json_after_max_steps(self):
        responses = [
            iter(
                [
                    self._stream_chunk(reasoning="Need web data."),
                    self._stream_chunk(content='{"tool_calls":[{"name":"search_web","arguments":{"queries":["x"]}}]}'),
                    self._stream_chunk(usage=SimpleNamespace(prompt_tokens=2, completion_tokens=3, total_tokens=5)),
                ]
            ),
            iter(
                [
                    self._stream_chunk(reasoning="Still trying to call a tool."),
                    self._stream_chunk(content='{"tool_calls":[{"name":"search_web","arguments":{"queries":["y"]}}]}'),
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


if __name__ == "__main__":
    unittest.main()
