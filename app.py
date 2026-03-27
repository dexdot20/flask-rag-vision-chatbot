from datetime import timedelta

from flask import Flask

from config import LOGIN_REMEMBER_SESSION_DAYS, RAG_ENABLED, SECRET_KEY
from db import configure_db_path, initialize_database
from rag_service import sync_conversations_to_rag_safe
from routes import (
    install_auth_guard,
    preload_dependencies,
    register_auth_routes,
    register_chat_routes,
    register_conversation_routes,
    register_page_routes,
)


def create_app(database_path: str | None = None) -> Flask:
    resolved_database_path = configure_db_path(database_path)

    app = Flask(__name__)
    app.config["DATABASE_PATH"] = resolved_database_path
    app.config["SECRET_KEY"] = SECRET_KEY
    app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(days=LOGIN_REMEMBER_SESSION_DAYS)

    initialize_database()
    register_auth_routes(app)
    install_auth_guard(app)
    register_page_routes(app)
    register_conversation_routes(app)
    register_chat_routes(app)
    if RAG_ENABLED:
        sync_conversations_to_rag_safe()
    return app


app = create_app()


if __name__ == "__main__":
    preload_dependencies(app)
    app.run(host="0.0.0.0", debug=True, use_reloader=False)
