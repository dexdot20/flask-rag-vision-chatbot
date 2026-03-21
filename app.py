from flask import Flask

from db import configure_db_path, initialize_database
from routes import preload_dependencies, register_chat_routes, register_conversation_routes, register_page_routes


def create_app(database_path: str | None = None) -> Flask:
    resolved_database_path = configure_db_path(database_path)

    app = Flask(__name__)
    app.config["DATABASE_PATH"] = resolved_database_path

    initialize_database()
    register_page_routes(app)
    register_conversation_routes(app)
    register_chat_routes(app)
    return app


app = create_app()


if __name__ == "__main__":
    preload_dependencies(app)
    app.run(host="0.0.0.0", debug=True, use_reloader=False)
