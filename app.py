from datetime import timedelta
from threading import Lock

from flask import Flask

from config import LOGIN_REMEMBER_SESSION_DAYS, RAG_ENABLED, SECRET_KEY
from db import configure_db_path, initialize_database
from routes import (
    install_auth_guard,
    preload_dependencies,
    register_auth_routes,
    register_chat_routes,
    register_conversation_routes,
    register_page_routes,
)


_RAG_STARTUP_SYNC_LOCK = Lock()


def _sync_rag_on_startup() -> None:
    if not RAG_ENABLED:
        return
    from rag_service import sync_conversations_to_rag_safe

    sync_conversations_to_rag_safe()


def create_app(database_path: str | None = None) -> Flask:
    resolved_database_path = configure_db_path(database_path)

    app = Flask(__name__)
    app.config["DATABASE_PATH"] = resolved_database_path
    app.config["SECRET_KEY"] = SECRET_KEY
    app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(days=LOGIN_REMEMBER_SESSION_DAYS)

    @app.before_request
    def _sync_rag_once_before_request():
        if app.config.get("RAG_STARTUP_SYNC_DONE"):
            return None

        with _RAG_STARTUP_SYNC_LOCK:
            if app.config.get("RAG_STARTUP_SYNC_DONE"):
                return None
            app.config["RAG_STARTUP_SYNC_DONE"] = True

        _sync_rag_on_startup()

    initialize_database()
    register_auth_routes(app)
    install_auth_guard(app)
    register_page_routes(app)
    register_conversation_routes(app)
    register_chat_routes(app)

    return app


app = create_app()


if __name__ == "__main__":
    preload_dependencies(app)
    app.config["RAG_STARTUP_SYNC_DONE"] = True
    _sync_rag_on_startup()
    app.run(host="0.0.0.0", debug=True, use_reloader=False)
