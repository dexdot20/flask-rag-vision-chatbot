from __future__ import annotations

import secrets
from datetime import datetime, timedelta, timezone

from flask import jsonify, redirect, render_template, request, session, url_for

import config

AUTH_SESSION_KEY = "auth_authenticated"
AUTH_LAST_SEEN_KEY = "auth_last_seen"
AUTH_REMEMBER_KEY = "auth_remember"
AUTH_FAILED_ATTEMPTS_KEY = "auth_failed_attempts"
AUTH_LOCKED_UNTIL_KEY = "auth_locked_until"


def is_login_pin_enabled() -> bool:
    return bool((config.LOGIN_PIN or "").strip())


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_utc_datetime(value) -> datetime | None:
    if value in (None, ""):
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _safe_next_url(raw_value) -> str | None:
    candidate = str(raw_value or "").strip()
    if not candidate:
        return None
    if not candidate.startswith("/") or candidate.startswith("//"):
        return None
    return candidate


def _clear_auth_state() -> None:
    for key in (
        AUTH_SESSION_KEY,
        AUTH_LAST_SEEN_KEY,
        AUTH_REMEMBER_KEY,
        AUTH_FAILED_ATTEMPTS_KEY,
        AUTH_LOCKED_UNTIL_KEY,
    ):
        session.pop(key, None)


def _lockout_until() -> datetime | None:
    return _parse_utc_datetime(session.get(AUTH_LOCKED_UNTIL_KEY))


def _remaining_lockout_seconds(now: datetime | None = None) -> int:
    current_time = now or _utc_now()
    lockout_until = _lockout_until()
    if lockout_until is None:
        return 0
    remaining = int((lockout_until - current_time).total_seconds())
    return max(0, remaining)


def _is_locked_out(now: datetime | None = None) -> bool:
    return _remaining_lockout_seconds(now) > 0


def _is_session_authenticated(now: datetime | None = None) -> bool:
    if not is_login_pin_enabled():
        return True
    if session.get(AUTH_SESSION_KEY) is not True:
        return False
    if session.get(AUTH_REMEMBER_KEY):
        return True

    current_time = now or _utc_now()
    last_seen = _parse_utc_datetime(session.get(AUTH_LAST_SEEN_KEY))
    if last_seen is None:
        _clear_auth_state()
        return False

    timeout = timedelta(minutes=config.LOGIN_SESSION_TIMEOUT_MINUTES)
    if current_time - last_seen > timeout:
        _clear_auth_state()
        return False

    session[AUTH_LAST_SEEN_KEY] = current_time.isoformat()
    session.modified = True
    return True


def _build_login_context(error: str | None = None, next_url: str | None = None, status_code: int = 200):
    return (
        render_template(
            "login.html",
            error=error,
            next_url=next_url,
            lockout_seconds=_remaining_lockout_seconds(),
            lockout_until=_lockout_until(),
            timeout_minutes=config.LOGIN_SESSION_TIMEOUT_MINUTES,
            remember_days=config.LOGIN_REMEMBER_SESSION_DAYS,
        ),
        status_code,
    )


def register_auth_routes(app) -> None:
    @app.route("/login", methods=["GET", "POST"])
    def login():
        if not is_login_pin_enabled():
            return redirect(url_for("index"))

        next_url = _safe_next_url(request.values.get("next"))
        if request.method == "GET":
            if _is_session_authenticated():
                return redirect(next_url or url_for("index"))
            return _build_login_context(next_url=next_url)

        if _is_locked_out():
            lockout_until = _lockout_until()
            lockout_text = lockout_until.strftime("%Y-%m-%d %H:%M:%SZ") if lockout_until else "later"
            return _build_login_context(
                error=f"Too many failed attempts. Try again after {lockout_text}.",
                next_url=next_url,
                status_code=429,
            )

        provided_pin = str(request.form.get("pin") or "").strip()
        remember = str(request.form.get("remember") or "").strip().lower() in {"1", "true", "yes", "on"}
        expected_pin = str(config.LOGIN_PIN or "").strip()

        if expected_pin and secrets.compare_digest(provided_pin, expected_pin):
            _clear_auth_state()
            session[AUTH_SESSION_KEY] = True
            session[AUTH_LAST_SEEN_KEY] = _utc_now().isoformat()
            session[AUTH_REMEMBER_KEY] = remember
            session[AUTH_FAILED_ATTEMPTS_KEY] = 0
            session[AUTH_LOCKED_UNTIL_KEY] = None
            session.permanent = remember
            session.modified = True
            return redirect(next_url or url_for("index"))

        failed_attempts = int(session.get(AUTH_FAILED_ATTEMPTS_KEY) or 0) + 1
        if failed_attempts >= config.LOGIN_MAX_FAILED_ATTEMPTS:
            lockout_until = _utc_now() + timedelta(seconds=config.LOGIN_LOCKOUT_SECONDS)
            session[AUTH_FAILED_ATTEMPTS_KEY] = 0
            session[AUTH_LOCKED_UNTIL_KEY] = lockout_until.isoformat()
            session.modified = True
            lockout_text = lockout_until.strftime("%Y-%m-%d %H:%M:%SZ")
            return _build_login_context(
                error=f"Too many failed attempts. Try again after {lockout_text}.",
                next_url=next_url,
                status_code=429,
            )

        session[AUTH_FAILED_ATTEMPTS_KEY] = failed_attempts
        session.modified = True
        return _build_login_context(error="Invalid PIN.", next_url=next_url, status_code=401)

    @app.route("/logout", methods=["POST"])
    def logout():
        _clear_auth_state()
        session.permanent = False
        session.modified = True
        if not is_login_pin_enabled():
            return redirect(url_for("index"))
        return redirect(url_for("login"))


def install_auth_guard(app) -> None:
    @app.before_request
    def _require_login_pin():
        if not is_login_pin_enabled():
            return None

        endpoint = request.endpoint or ""
        if endpoint in {"static", "login", "logout"}:
            return None

        if _is_session_authenticated():
            return None

        _clear_auth_state()
        next_target = request.full_path[:-1] if request.full_path.endswith("?") else request.full_path
        login_url = url_for("login", next=next_target)
        if request.path.startswith("/api/"):
            return jsonify({"error": "Login PIN required."}), 401
        return redirect(login_url)