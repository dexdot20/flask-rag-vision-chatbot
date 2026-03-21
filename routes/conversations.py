from __future__ import annotations

from flask import jsonify, request

from config import (
    AVAILABLE_MODEL_IDS,
    RAG_DISABLED_FEATURE_ERROR,
    RAG_DISABLED_INGEST_ERROR,
    RAG_ENABLED,
    RAG_SEARCH_DEFAULT_TOP_K,
    RAG_SOURCE_CONVERSATION,
    RAG_SOURCE_TOOL_RESULT,
)
from db import delete_conversation_image_assets, get_conversation_message_rows, get_db, message_row_to_dict
from rag import delete_source as rag_delete_source
from rag_service import (
    conversation_rag_source_key,
    delete_rag_document_record,
    delete_rag_source_record,
    ensure_supported_rag_sources,
    get_rag_document_record,
    list_rag_documents_db,
    search_knowledge_base_tool,
    sync_conversations_to_rag,
)
from routes.request_utils import normalize_model_id


def register_conversation_routes(app) -> None:
    @app.route("/api/conversations", methods=["GET"])
    def list_conversations():
        with get_db() as conn:
            rows = conn.execute(
                """
                SELECT c.id, c.title, c.model, c.updated_at,
                       COUNT(m.id) AS message_count
                FROM conversations c
                LEFT JOIN messages m ON m.conversation_id = c.id
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
        return jsonify({"id": conversation_id, "title": title, "model": model}), 201

    @app.route("/api/conversations/<int:conv_id>", methods=["GET"])
    def get_conversation(conv_id):
        with get_db() as conn:
            conversation = conn.execute(
                "SELECT * FROM conversations WHERE id = ?",
                (conv_id,),
            ).fetchone()
            if not conversation:
                return jsonify({"error": "Not found."}), 404
            messages = get_conversation_message_rows(conn, conv_id)
        return jsonify(
            {
                "conversation": dict(conversation),
                "messages": [message_row_to_dict(message) for message in messages],
            }
        )

    @app.route("/api/conversations/<int:conv_id>", methods=["DELETE"])
    def delete_conversation(conv_id):
        delete_conversation_image_assets(conv_id)
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
            synced = sync_conversations_to_rag(conversation_id=conversation_id)
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
