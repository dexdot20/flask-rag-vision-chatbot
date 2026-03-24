from __future__ import annotations

import re

from flask import Response, jsonify, request

from canvas_service import (
    build_html_download,
    build_markdown_download,
    build_pdf_download,
    clear_canvas,
    create_canvas_runtime_state,
    delete_canvas_document,
    find_latest_canvas_document,
    find_latest_canvas_documents,
    get_canvas_runtime_documents,
)
from conversation_export import (
    build_conversation_docx_download,
    build_conversation_markdown_download,
    build_conversation_pdf_download,
)
from config import (
    AVAILABLE_MODEL_IDS,
    RAG_DISABLED_FEATURE_ERROR,
    RAG_DISABLED_INGEST_ERROR,
    RAG_ENABLED,
    RAG_SEARCH_DEFAULT_TOP_K,
    RAG_SOURCE_CONVERSATION,
    RAG_SOURCE_TOOL_RESULT,
)
from db import (
    delete_conversation_file_assets,
    delete_conversation_image_assets,
    get_conversation_message_rows,
    get_db,
    insert_message,
    message_row_to_dict,
    serialize_message_metadata,
)
from rag import delete_source as rag_delete_source
from rag_service import (
    conversation_rag_source_key,
    delete_rag_document_record,
    delete_rag_source_record,
    ensure_supported_rag_sources,
    get_rag_document_record,
    list_rag_documents_db,
    search_knowledge_base_tool,
    sync_conversations_to_rag_safe,
    sync_conversations_to_rag,
)
from routes.request_utils import normalize_model_id


def _sanitize_download_filename(value: str, fallback: str = "canvas") -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value or "").strip()).strip("-._")
    return normalized[:80] or fallback


def _load_conversation_payload(conv_id: int):
    with get_db() as conn:
        conversation = conn.execute(
            "SELECT * FROM conversations WHERE id = ?",
            (conv_id,),
        ).fetchone()
        if not conversation:
            return None, None
        messages = [message_row_to_dict(message) for message in get_conversation_message_rows(conn, conv_id)]
    return conversation, messages


def register_conversation_routes(app) -> None:
    @app.route("/api/conversations", methods=["GET"])
    def list_conversations():
        with get_db() as conn:
            rows = conn.execute(
                """
                SELECT c.id, c.title, c.model, c.updated_at,
                       COUNT(m.id) AS message_count
                FROM conversations c
                  LEFT JOIN messages m ON m.conversation_id = c.id AND m.deleted_at IS NULL
                GROUP BY c.id
                ORDER BY c.updated_at DESC
                """
            ).fetchall()
        return jsonify([dict(row) for row in rows])

    @app.route("/api/conversations", methods=["POST"])
    def create_conversation():
        data = request.get_json(silent=True) or {}
        title = (data.get("title") or "New Chat")[:120]
        model = normalize_model_id(data.get("model"))
        if model not in AVAILABLE_MODEL_IDS:
            return jsonify({"error": "Invalid model."}), 400
        with get_db() as conn:
            cursor = conn.execute(
                "INSERT INTO conversations (title, model) VALUES (?, ?)",
                (title, model),
            )
            conversation_id = cursor.lastrowid
        if RAG_ENABLED:
            sync_conversations_to_rag_safe(conversation_id=conversation_id)
        return jsonify({"id": conversation_id, "title": title, "model": model}), 201

    @app.route("/api/conversations/<int:conv_id>", methods=["GET"])
    def get_conversation(conv_id):
        conversation, messages = _load_conversation_payload(conv_id)
        if not conversation:
            return jsonify({"error": "Not found."}), 404
        return jsonify(
            {
                "conversation": dict(conversation),
                "messages": messages,
            }
        )

    @app.route("/api/conversations/<int:conv_id>/export", methods=["GET"])
    def export_conversation(conv_id):
        format_name = str(request.args.get("format") or "md").strip().lower()
        conversation, messages = _load_conversation_payload(conv_id)
        if not conversation:
            return jsonify({"error": "Not found."}), 404

        base_name = _sanitize_download_filename(conversation["title"] or "conversation", fallback="conversation")
        payload_conversation = dict(conversation)
        if format_name == "md":
            payload = build_conversation_markdown_download(payload_conversation, messages)
            mime_type = "text/markdown; charset=utf-8"
            filename = f"{base_name}.md"
        elif format_name == "docx":
            payload = build_conversation_docx_download(payload_conversation, messages)
            mime_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            filename = f"{base_name}.docx"
        elif format_name == "pdf":
            payload = build_conversation_pdf_download(payload_conversation, messages)
            mime_type = "application/pdf"
            filename = f"{base_name}.pdf"
        else:
            return jsonify({"error": "format must be md, docx, or pdf."}), 400

        return Response(
            payload,
            content_type=mime_type,
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
            },
        )

    @app.route("/api/conversations/<int:conv_id>/canvas/export", methods=["GET"])
    def export_canvas_document(conv_id):
        format_name = str(request.args.get("format") or "md").strip().lower()
        document_id = str(request.args.get("document_id") or "").strip() or None

        conversation, messages = _load_conversation_payload(conv_id)
        if not conversation:
            return jsonify({"error": "Not found."}), 404

        document = find_latest_canvas_document(messages, document_id=document_id)
        if not document:
            return jsonify({"error": "Canvas document not found."}), 404

        base_name = _sanitize_download_filename(document.get("title") or conversation["title"] or "canvas")
        if format_name == "md":
            payload = build_markdown_download(document)
            mime_type = "text/markdown; charset=utf-8"
            filename = f"{base_name}.md"
        elif format_name == "html":
            payload = build_html_download(document)
            mime_type = "text/html; charset=utf-8"
            filename = f"{base_name}.html"
        elif format_name == "pdf":
            payload = build_pdf_download(document)
            mime_type = "application/pdf"
            filename = f"{base_name}.pdf"
        else:
            return jsonify({"error": "format must be md, html, or pdf."}), 400

        return Response(
            payload,
            content_type=mime_type,
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
            },
        )

    @app.route("/api/conversations/<int:conv_id>/canvas", methods=["DELETE"])
    def delete_canvas(conv_id):
        clear_all = str(request.args.get("clear_all") or "").strip().lower() in {"1", "true", "yes", "on"}
        document_id = str(request.args.get("document_id") or "").strip() or None

        conversation, messages = _load_conversation_payload(conv_id)
        if not conversation:
            return jsonify({"error": "Not found."}), 404

        runtime_state = create_canvas_runtime_state(find_latest_canvas_documents(messages))
        current_documents = get_canvas_runtime_documents(runtime_state)
        if not current_documents:
            return jsonify({"error": "Canvas document not found."}), 404

        try:
            if clear_all:
                result = clear_canvas(runtime_state)
            else:
                result = delete_canvas_document(runtime_state, document_id=document_id)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 404

        next_documents = get_canvas_runtime_documents(runtime_state)
        metadata = serialize_message_metadata(
            {
                "canvas_documents": next_documents,
                "canvas_cleared": not next_documents,
            }
        )

        with get_db() as conn:
            insert_message(
                conn,
                conv_id,
                "tool",
                "",
                metadata=metadata,
            )
            conn.execute(
                "UPDATE conversations SET updated_at = datetime('now') WHERE id = ?",
                (conv_id,),
            )

        if RAG_ENABLED:
            sync_conversations_to_rag_safe(conversation_id=conv_id)

        _, updated_messages = _load_conversation_payload(conv_id)
        active_document_id = next_documents[-1]["id"] if next_documents else None
        return jsonify(
            {
                "cleared": not next_documents,
                "documents": next_documents,
                "active_document_id": active_document_id,
                "remaining_count": len(next_documents),
                "deleted_document_id": result.get("deleted_id"),
                "messages": updated_messages,
            }
        )

    @app.route("/api/conversations/<int:conv_id>", methods=["DELETE"])
    def delete_conversation(conv_id):
        delete_conversation_image_assets(conv_id)
        delete_conversation_file_assets(conv_id)
        with get_db() as conn:
            conn.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))
        if RAG_ENABLED:
            delete_rag_source_record(conversation_rag_source_key(RAG_SOURCE_CONVERSATION, conv_id))
            delete_rag_source_record(conversation_rag_source_key(RAG_SOURCE_TOOL_RESULT, conv_id))
        return "", 204

    @app.route("/api/conversations/<int:conv_id>", methods=["PATCH"])
    def update_conversation_title(conv_id):
        data = request.get_json(silent=True) or {}
        title = (data.get("title") or "").strip()[:120]
        if not title:
            return jsonify({"error": "Title required."}), 400
        with get_db() as conn:
            conn.execute(
                "UPDATE conversations SET title = ?, updated_at = datetime('now') WHERE id = ?",
                (title, conv_id),
            )
        if RAG_ENABLED:
            sync_conversations_to_rag_safe(conversation_id=conv_id)
        return jsonify({"id": conv_id, "title": title})

    @app.route("/api/rag/documents", methods=["GET"])
    def list_rag_documents():
        if not RAG_ENABLED:
            return jsonify({"error": RAG_DISABLED_FEATURE_ERROR}), 410
        return jsonify(list_rag_documents_db())

    @app.route("/api/rag/search", methods=["GET"])
    def rag_search():
        if not RAG_ENABLED:
            return jsonify({"error": RAG_DISABLED_FEATURE_ERROR}), 410
        query = (request.args.get("q") or "").strip()
        category = (request.args.get("category") or "").strip() or None
        try:
            top_k = int(request.args.get("top_k") or RAG_SEARCH_DEFAULT_TOP_K)
        except (TypeError, ValueError):
            top_k = RAG_SEARCH_DEFAULT_TOP_K

        try:
            return jsonify(search_knowledge_base_tool(query, category=category, top_k=top_k))
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

    @app.route("/api/rag/ingest", methods=["POST"])
    def ingest_rag_document():
        if not RAG_ENABLED:
            return jsonify({"error": RAG_DISABLED_FEATURE_ERROR}), 410
        return jsonify({"error": RAG_DISABLED_INGEST_ERROR}), 410

    @app.route("/api/rag/sync-conversations", methods=["POST"])
    def sync_rag_conversations():
        if not RAG_ENABLED:
            return jsonify({"error": RAG_DISABLED_FEATURE_ERROR}), 410
        data = request.get_json(silent=True) or {}
        raw_conversation_id = data.get("conversation_id")

        try:
            conversation_id = int(raw_conversation_id) if raw_conversation_id not in (None, "", "all") else None
        except (TypeError, ValueError):
            return jsonify({"error": "conversation_id must be an integer or 'all'."}), 400

        try:
            ensure_supported_rag_sources(force=True)
            synced = sync_conversations_to_rag(conversation_id=conversation_id, force=True)
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

        return jsonify({"count": len(synced), "documents": synced})

    @app.route("/api/rag/documents/<source_key>", methods=["DELETE"])
    def delete_rag_document(source_key):
        if not RAG_ENABLED:
            return jsonify({"error": RAG_DISABLED_FEATURE_ERROR}), 410
        row = get_rag_document_record(source_key)
        if not row:
            return jsonify({"error": "Not found."}), 404

        try:
            deleted_chunks = rag_delete_source(source_key)
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

        delete_rag_document_record(source_key)
        return jsonify(
            {
                "source_key": source_key,
                "source_name": row["source_name"],
                "deleted_chunks": deleted_chunks,
            }
        )
