from __future__ import annotations

from html import escape
from io import BytesIO
import re
from typing import Iterable
from uuid import uuid4

try:
    import markdown as markdown_lib
except ImportError:  # pragma: no cover - optional dependency fallback
    markdown_lib = None

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import ListFlowable, ListItem, Paragraph, Preformatted, SimpleDocTemplate, Spacer

# Unicode font registration for Turkish / non-Latin character support.
_FONT_PATHS = {
    "DejaVuSans": "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "DejaVuSans-Bold": "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "DejaVuSansMono": "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
}

def _try_register_fonts() -> bool:
    import os
    if not all(os.path.exists(p) for p in _FONT_PATHS.values()):
        return False
    try:
        for name, path in _FONT_PATHS.items():
            pdfmetrics.registerFont(TTFont(name, path))
        return True
    except Exception:
        return False

_UNICODE_FONTS = _try_register_fonts()
_BODY_FONT = "DejaVuSans" if _UNICODE_FONTS else "Helvetica"
_BOLD_FONT = "DejaVuSans-Bold" if _UNICODE_FONTS else "Helvetica-Bold"
_MONO_FONT = "DejaVuSansMono" if _UNICODE_FONTS else "Courier"

CANVAS_MAX_DOCUMENTS = 12
CANVAS_MAX_TITLE_LENGTH = 160
CANVAS_MAX_CONTENT_LENGTH = 120_000
CANVAS_MAX_LANGUAGE_LENGTH = 48
CANVAS_MAX_PATH_LENGTH = 240
CANVAS_MAX_SUMMARY_LENGTH = 280
CANVAS_MAX_SCOPE_ID_LENGTH = 80
CANVAS_MAX_RELATION_COUNT = 24
CANVAS_MAX_RELATION_ITEM_LENGTH = 120
CANVAS_CONTEXT_MAX_CHARS = 20_000
CANVAS_CONTEXT_MAX_LINES = 800
CANVAS_ALLOWED_FORMATS = {"markdown", "code"}
CANVAS_ALLOWED_ROLES = {"source", "config", "dependency", "docs", "test", "script", "note"}
CANVAS_MODE_DOCUMENT = "document"
CANVAS_MODE_PROJECT = "project"
CANVAS_FILE_PRIORITY = {
    "source": 10,
    "config": 20,
    "dependency": 30,
    "test": 40,
    "script": 50,
    "docs": 60,
    "note": 70,
}


def _normalize_line_endings(text: str) -> str:
    return str(text or "").replace("\r\n", "\n").replace("\r", "\n")


def _clip_text(text: str, max_length: int) -> str:
    normalized = _normalize_line_endings(text)
    if len(normalized) <= max_length:
        return normalized
    return normalized[:max_length]


def _line_count(text: str) -> int:
    if not text:
        return 0
    return len(text.split("\n"))


def _normalize_canvas_language(value) -> str | None:
    language = re.sub(r"[^a-z0-9_+.#-]", "", str(value or "").strip().lower())[:CANVAS_MAX_LANGUAGE_LENGTH]
    return language or None


def _normalize_canvas_short_text(value, max_length: int) -> str | None:
    text = re.sub(r"\s+", " ", _normalize_line_endings(str(value or "")).strip())[:max_length]
    return text or None


def _normalize_canvas_identifier(value) -> str | None:
    identifier = re.sub(r"[^a-z0-9_.:-]", "", str(value or "").strip().lower())[:CANVAS_MAX_SCOPE_ID_LENGTH]
    return identifier or None


def _normalize_canvas_path(value) -> str | None:
    raw_path = _normalize_line_endings(str(value or "")).strip().replace("\\", "/")
    if not raw_path:
        return None
    raw_path = re.sub(r"/{2,}", "/", raw_path)
    while raw_path.startswith("./"):
        raw_path = raw_path[2:]
    raw_path = raw_path.lstrip("/")

    normalized_parts = []
    for part in raw_path.split("/"):
        cleaned_part = part.strip()
        if not cleaned_part or cleaned_part == ".":
            continue
        if cleaned_part == "..":
            if normalized_parts:
                normalized_parts.pop()
            continue
        normalized_parts.append(cleaned_part)

    normalized_path = "/".join(normalized_parts)[:CANVAS_MAX_PATH_LENGTH]
    return normalized_path or None


def _normalize_canvas_role(value) -> str | None:
    role = re.sub(r"[^a-z]", "", str(value or "").strip().lower())
    if role in CANVAS_ALLOWED_ROLES:
        return role
    return None


def _normalize_canvas_string_list(values) -> list[str]:
    if not isinstance(values, (list, tuple, set)):
        return []

    normalized = []
    seen = set()
    for raw_value in values:
        item = _normalize_canvas_short_text(raw_value, CANVAS_MAX_RELATION_ITEM_LENGTH)
        if not item:
            continue
        dedupe_key = item.lower()
        if dedupe_key in seen:
            continue
        normalized.append(item)
        seen.add(dedupe_key)
        if len(normalized) >= CANVAS_MAX_RELATION_COUNT:
            break
    return normalized


def _infer_canvas_role(path: str | None, title: str, format_name: str) -> str | None:
    candidate = (path or title or "").strip().lower()
    if not candidate:
        return None
    filename = candidate.rsplit("/", 1)[-1]
    if filename.startswith("test_") or "/tests/" in f"/{candidate}" or candidate.endswith("_test.py"):
        return "test"
    if filename in {"readme", "readme.md", "readme.txt"} or candidate.startswith("docs/"):
        return "docs"
    if filename in {"requirements.txt", "requirements-dev.txt", "package.json", "pyproject.toml", ".env", ".env.example"}:
        return "dependency" if "requirements" in filename or filename == "package.json" else "config"
    if filename.endswith((".ini", ".cfg", ".toml", ".yaml", ".yml", ".json", ".env")):
        return "config"
    if filename.endswith((".sh", ".bash")):
        return "script"
    if format_name == "code":
        return "source"
    return "note"


def _build_canvas_summary(title: str, path: str | None, role: str | None, content: str) -> str:
    label = path or title or "Canvas"
    first_meaningful_line = ""
    for line in _normalize_line_endings(content).split("\n"):
        stripped = re.sub(r"\s+", " ", line.strip())
        if not stripped:
            continue
        first_meaningful_line = stripped.lstrip("#*- ").strip()
        if first_meaningful_line:
            break

    role_label = (role or "document").replace("_", " ")
    if first_meaningful_line:
        return f"{role_label.capitalize()} {label}: {first_meaningful_line}"[:CANVAS_MAX_SUMMARY_LENGTH]
    return f"{role_label.capitalize()} {label}"[:CANVAS_MAX_SUMMARY_LENGTH]


def scale_canvas_char_limit(max_lines: int | None, *, default_lines: int, default_chars: int) -> int:
    try:
        normalized_max_lines = int(max_lines or 0)
    except (TypeError, ValueError):
        return default_chars
    if normalized_max_lines <= 0 or default_lines <= 0 or default_chars <= 0:
        return default_chars
    return max(1, int(round(default_chars * (normalized_max_lines / default_lines))))


def _number_canvas_lines(
    content: str,
    *,
    max_lines: int = CANVAS_CONTEXT_MAX_LINES,
    max_chars: int | None = None,
) -> tuple[list[str], bool]:
    if max_chars is None:
        max_chars = scale_canvas_char_limit(
            max_lines,
            default_lines=CANVAS_CONTEXT_MAX_LINES,
            default_chars=CANVAS_CONTEXT_MAX_CHARS,
        )
    normalized = _normalize_line_endings(content)
    all_lines = normalized.split("\n") if normalized else []
    visible_lines = []
    visible_char_count = 0

    for index, line in enumerate(all_lines, start=1):
        numbered_line = f"{index}: {line}"
        extra_chars = len(numbered_line) + (1 if visible_lines else 0)
        if visible_lines and (len(visible_lines) >= max_lines or visible_char_count + extra_chars > max_chars):
            return visible_lines, True
        if not visible_lines and extra_chars > max_chars:
            visible_lines.append(numbered_line[:max_chars])
            return visible_lines, True
        visible_lines.append(numbered_line)
        visible_char_count += extra_chars

    return visible_lines, False


def build_canvas_relationship_map(documents: list[dict] | None) -> dict | None:
    normalized_documents = extract_canvas_documents({"canvas_documents": documents or []})
    if not normalized_documents:
        return None

    files = []
    aggregate_imports = []
    aggregate_exports = []
    aggregate_symbols = []
    aggregate_dependencies = []
    seen_buckets = {
        "imports": set(),
        "exports": set(),
        "symbols": set(),
        "dependencies": set(),
    }

    for document in sorted(normalized_documents, key=_document_sort_key):
        entry = {
            "file": document.get("path") or document.get("title") or document.get("id"),
            "role": document.get("role") or "note",
        }
        for key in ("imports", "exports", "symbols", "dependencies"):
            values = document.get(key) if isinstance(document.get(key), list) else []
            if values:
                entry[key] = values[:8]
            for value in values:
                normalized_value = str(value).strip()
                dedupe_key = normalized_value.lower()
                if not normalized_value or dedupe_key in seen_buckets[key]:
                    continue
                seen_buckets[key].add(dedupe_key)
                if key == "imports":
                    aggregate_imports.append(normalized_value)
                elif key == "exports":
                    aggregate_exports.append(normalized_value)
                elif key == "symbols":
                    aggregate_symbols.append(normalized_value)
                elif key == "dependencies":
                    aggregate_dependencies.append(normalized_value)
        files.append(entry)

    return {
        "files": files,
        "imports": aggregate_imports[:24],
        "exports": aggregate_exports[:24],
        "symbols": aggregate_symbols[:24],
        "dependencies": aggregate_dependencies[:24],
    }


def _resolve_active_canvas_document(documents: list[dict], active_document_id: str | None = None) -> dict | None:
    target_id = str(active_document_id or "").strip()
    if target_id:
        for document in documents:
            if str(document.get("id") or "") == target_id:
                return document
    return documents[-1] if documents else None


def _document_sort_key(document: dict) -> tuple[int, str, str, str]:
    role = str(document.get("role") or "note")
    priority = CANVAS_FILE_PRIORITY.get(role, 999)
    path = str(document.get("path") or "").strip().lower()
    title = str(document.get("title") or "").strip().lower()
    document_id = str(document.get("id") or "").strip()
    return priority, path, title, document_id


def _normalize_document_path_for_lookup(document_path: str | None) -> str | None:
    normalized_path = _normalize_canvas_path(document_path)
    return normalized_path or None


def _normalize_canvas_lookup_key(value) -> str | None:
    normalized_value = _normalize_document_path_for_lookup(value)
    if not normalized_value:
        return None
    return normalized_value.casefold()


def _normalize_canvas_lookup_basename(value) -> str | None:
    lookup_key = _normalize_canvas_lookup_key(value)
    if not lookup_key:
        return None
    return lookup_key.rsplit("/", 1)[-1]


def _find_canvas_document_by_path_locator(documents: list[dict], document_path: str | None) -> tuple[int, dict] | None:
    lookup_key = _normalize_canvas_lookup_key(document_path)
    if not lookup_key:
        return None

    exact_title_matches = []
    basename_matches = []
    lookup_basename = _normalize_canvas_lookup_basename(document_path)

    for index, document in enumerate(documents):
        path_key = _normalize_canvas_lookup_key(document.get("path"))
        if path_key == lookup_key:
            return index, document

        title_key = _normalize_canvas_lookup_key(document.get("title"))
        if title_key == lookup_key:
            exact_title_matches.append((index, document))
            continue

        if not lookup_basename:
            continue
        if path_key and path_key.rsplit("/", 1)[-1] == lookup_basename:
            basename_matches.append((index, document))
            continue
        if title_key and title_key.rsplit("/", 1)[-1] == lookup_basename:
            basename_matches.append((index, document))

    if len(exact_title_matches) == 1:
        return exact_title_matches[0]
    if len(basename_matches) == 1:
        return basename_matches[0]
    return None


def extract_canvas_primary_locator(document: dict | None) -> dict | None:
    if not isinstance(document, dict):
        return None
    path = _normalize_document_path_for_lookup(document.get("path"))
    if path:
        return {"type": "path", "value": path}
    document_id = str(document.get("id") or "").strip()
    if document_id:
        return {"type": "id", "value": document_id}
    return None


def extract_canvas_active_document_id(metadata: dict | None, documents: list[dict] | None = None) -> str | None:
    source = metadata if isinstance(metadata, dict) else {}
    normalized_documents = documents if isinstance(documents, list) else extract_canvas_documents(source)
    active_document_id = str(source.get("active_document_id") or "").strip()[:80]
    if active_document_id and any(str(document.get("id") or "") == active_document_id for document in normalized_documents):
        return active_document_id
    active_document = _resolve_active_canvas_document(normalized_documents)
    if not active_document:
        return None
    return str(active_document.get("id") or "").strip() or None


def determine_canvas_mode(documents: list[dict] | None) -> str:
    normalized_documents = documents if isinstance(documents, list) else []
    scope_ids = {
        str(document.get("project_id") or document.get("workspace_id") or "").strip()
        for document in normalized_documents
        if str(document.get("project_id") or document.get("workspace_id") or "").strip()
    }
    paths = {str(document.get("path") or "").strip() for document in normalized_documents if str(document.get("path") or "").strip()}
    if len(normalized_documents) > 1 or scope_ids or len(paths) > 1:
        return CANVAS_MODE_PROJECT
    return CANVAS_MODE_DOCUMENT


def _infer_canvas_target_type(documents: list[dict], active_document: dict | None) -> str:
    active_path = str((active_document or {}).get("path") or "").lower()
    dependency_paths = {
        str(document.get("path") or "").lower()
        for document in documents
        if str(document.get("role") or "") == "dependency"
    }
    if active_path.endswith(".py") or "pyproject.toml" in dependency_paths or "requirements.txt" in dependency_paths:
        return "python-project"
    if any(str(document.get("role") or "") == "source" for document in documents):
        return "multi-file-project"
    return "document-set"


def _infer_manifest_name(documents: list[dict], active_document: dict | None) -> str:
    active_document = active_document or {}
    for key in ("project_id", "workspace_id"):
        value = str(active_document.get(key) or "").strip()
        if value:
            return value
    for document in documents:
        for key in ("project_id", "workspace_id"):
            value = str(document.get(key) or "").strip()
            if value:
                return value
    active_path = str(active_document.get("path") or "").strip()
    if active_path:
        top_level = active_path.split("/", 1)[0].strip()
        if top_level:
            return top_level
    return str(active_document.get("title") or "Canvas").strip() or "Canvas"


def build_canvas_project_manifest(documents: list[dict] | None, active_document_id: str | None = None) -> dict | None:
    raw_documents = documents or []
    normalized_documents = extract_canvas_documents({"canvas_documents": raw_documents})
    if not normalized_documents:
        return None

    raw_normalized_documents = [
        cleaned
        for cleaned in (normalize_canvas_document(entry) for entry in raw_documents[:CANVAS_MAX_DOCUMENTS])
        if cleaned
    ]

    active_document = _resolve_active_canvas_document(normalized_documents, active_document_id)
    mode = determine_canvas_mode(raw_normalized_documents or normalized_documents)
    dependency_summaries = []
    seen_dependency_summaries = set()
    open_issues = []
    file_list = []

    missing_paths = 0
    missing_roles = 0
    for document in normalized_documents:
        summary = str(document.get("summary") or "").strip() or _build_canvas_summary(
            str(document.get("title") or "Canvas"),
            document.get("path"),
            document.get("role"),
            str(document.get("content") or ""),
        )
        role = str(document.get("role") or "note")
        entry = {
            "id": document["id"],
            "title": document["title"],
            "format": document["format"],
            "summary": summary,
            "line_count": int(document.get("line_count") or 0),
            "active": active_document is not None and document["id"] == active_document["id"],
            "priority": CANVAS_FILE_PRIORITY.get(role, 999),
        }
        for key in ("path", "role", "language", "project_id", "workspace_id"):
            if document.get(key):
                entry[key] = document[key]
        for key in ("imports", "exports", "symbols", "dependencies"):
            values = document.get(key) if isinstance(document.get(key), list) else []
            if values:
                entry[key] = values[:8]
        file_list.append(entry)

        if mode == CANVAS_MODE_PROJECT and not document.get("path"):
            missing_paths += 1
        if mode == CANVAS_MODE_PROJECT and not document.get("role"):
            missing_roles += 1

        dependency_values = document.get("dependencies") if isinstance(document.get("dependencies"), list) else []
        for value in dependency_values:
            normalized_value = str(value).strip()
            dedupe_key = normalized_value.lower()
            if not normalized_value or dedupe_key in seen_dependency_summaries:
                continue
            dependency_summaries.append(normalized_value)
            seen_dependency_summaries.add(dedupe_key)

    if mode == CANVAS_MODE_PROJECT and missing_paths:
        open_issues.append("Some project canvas documents are missing a path.")
    if mode == CANVAS_MODE_PROJECT and missing_roles:
        open_issues.append("Some project canvas documents are missing a role.")

    file_list.sort(key=_document_sort_key)

    validation_issues = []
    if mode == CANVAS_MODE_PROJECT:
        raw_normalized_paths = []
        for entry in raw_documents[:CANVAS_MAX_DOCUMENTS]:
            if not isinstance(entry, dict):
                continue
            path = _normalize_document_path_for_lookup(entry.get("path"))
            if path:
                raw_normalized_paths.append(path.lower())
        if len(raw_normalized_paths) != len(set(raw_normalized_paths)):
            validation_issues.append("Duplicate project paths detected.")
        active_paths = [entry.get("path") for entry in file_list if entry.get("path")]
        if not any((entry.get("role") == "source") for entry in file_list):
            validation_issues.append("No source file is marked in the project manifest.")

    manifest = {
        "mode": mode,
        "project_name": _infer_manifest_name(normalized_documents, active_document),
        "target_type": _infer_canvas_target_type(normalized_documents, active_document),
        "document_count": len(normalized_documents),
        "active_document_id": active_document["id"] if active_document else None,
        "active_path": active_document.get("path") if active_document else None,
        "active_file": active_document.get("path") or active_document.get("title") if active_document else None,
        "file_list": file_list,
        "open_issues": [*open_issues, *validation_issues],
        "last_validation_status": "ok" if not validation_issues else "needs_attention",
        "dependency_summaries": dependency_summaries[:16],
        "relationship_map": build_canvas_relationship_map(normalized_documents),
    }
    if active_document and active_document.get("project_id"):
        manifest["project_id"] = active_document["project_id"]
    if active_document and active_document.get("workspace_id"):
        manifest["workspace_id"] = active_document["workspace_id"]
    return manifest


def normalize_canvas_document(value, *, fallback_title: str = "Canvas") -> dict | None:
    if not isinstance(value, dict):
        return None

    document_id = str(value.get("id") or "").strip()[:80] or uuid4().hex
    title = str(value.get("title") or fallback_title).strip()[:CANVAS_MAX_TITLE_LENGTH] or fallback_title
    format_name = str(value.get("format") or "markdown").strip().lower() or "markdown"
    if format_name not in CANVAS_ALLOWED_FORMATS:
        format_name = "markdown"

    content = _clip_text(value.get("content") or "", CANVAS_MAX_CONTENT_LENGTH)
    language = _normalize_canvas_language(value.get("language"))
    path = _normalize_canvas_path(value.get("path"))
    created_at = str(value.get("created_at") or "").strip()[:80]
    updated_at = str(value.get("updated_at") or "").strip()[:80]
    role = _normalize_canvas_role(value.get("role")) or _infer_canvas_role(path, title, format_name)
    summary = _normalize_canvas_short_text(value.get("summary"), CANVAS_MAX_SUMMARY_LENGTH)
    imports = _normalize_canvas_string_list(value.get("imports"))
    exports = _normalize_canvas_string_list(value.get("exports"))
    symbols = _normalize_canvas_string_list(value.get("symbols"))
    dependencies = _normalize_canvas_string_list(value.get("dependencies"))
    project_id = _normalize_canvas_identifier(value.get("project_id"))
    workspace_id = _normalize_canvas_identifier(value.get("workspace_id"))

    cleaned = {
        "id": document_id,
        "title": title,
        "format": format_name,
        "content": content,
        "line_count": _line_count(content),
    }

    if path:
        cleaned["path"] = path
    if role:
        cleaned["role"] = role
    cleaned["summary"] = summary or _build_canvas_summary(title, path, role, content)

    if language:
        cleaned["language"] = language
    if imports:
        cleaned["imports"] = imports
    if exports:
        cleaned["exports"] = exports
    if symbols:
        cleaned["symbols"] = symbols
    if dependencies:
        cleaned["dependencies"] = dependencies
    if project_id:
        cleaned["project_id"] = project_id
    if workspace_id:
        cleaned["workspace_id"] = workspace_id

    if created_at:
        cleaned["created_at"] = created_at
    if updated_at:
        cleaned["updated_at"] = updated_at

    source_message_id = value.get("source_message_id")
    if isinstance(source_message_id, int) and source_message_id > 0:
        cleaned["source_message_id"] = source_message_id

    return cleaned


def extract_canvas_documents(metadata: dict | None) -> list[dict]:
    source = metadata if isinstance(metadata, dict) else {}
    raw_documents = source.get("canvas_documents")
    if not isinstance(raw_documents, list):
        return []

    normalized = []
    seen_ids = set()
    seen_paths = set()
    for entry in raw_documents[:CANVAS_MAX_DOCUMENTS]:
        cleaned = normalize_canvas_document(entry)
        if not cleaned:
            continue
        if cleaned["id"] in seen_ids:
            continue
        normalized_path = str(cleaned.get("path") or "").strip().lower()
        if normalized_path and normalized_path in seen_paths:
            continue
        normalized.append(cleaned)
        seen_ids.add(cleaned["id"])
        if normalized_path:
            seen_paths.add(normalized_path)
    return normalized


def list_canvas_lines(content: str) -> list[str]:
    normalized = _normalize_line_endings(content)
    if normalized == "":
        return []
    return normalized.split("\n")


def join_canvas_lines(lines: Iterable[str]) -> str:
    return "\n".join(str(line) for line in lines)


def create_canvas_runtime_state(initial_documents: list[dict] | None = None, active_document_id: str | None = None) -> dict:
    documents = extract_canvas_documents({"canvas_documents": initial_documents or []})
    resolved_active_document_id = extract_canvas_active_document_id({"active_document_id": active_document_id}, documents)
    runtime_state = {
        "documents": documents,
        "active_document_id": resolved_active_document_id,
    }
    runtime_state["mode"] = determine_canvas_mode(documents)
    return runtime_state


def get_canvas_runtime_active_document_id(runtime_state: dict | None) -> str | None:
    if not isinstance(runtime_state, dict):
        return None
    return extract_canvas_active_document_id(
        {"active_document_id": runtime_state.get("active_document_id")},
        runtime_state.get("documents") if isinstance(runtime_state.get("documents"), list) else [],
    )


def get_canvas_runtime_snapshot(runtime_state: dict | None) -> dict:
    documents = get_canvas_runtime_documents(runtime_state)
    active_document_id = get_canvas_runtime_active_document_id(runtime_state)
    return {
        "documents": documents,
        "active_document_id": active_document_id,
        "mode": determine_canvas_mode(documents),
        "manifest": build_canvas_project_manifest(documents, active_document_id=active_document_id),
    }


def get_canvas_runtime_documents(runtime_state: dict | None) -> list[dict]:
    if not isinstance(runtime_state, dict):
        return []
    return extract_canvas_documents({"canvas_documents": runtime_state.get("documents") or []})


def _refresh_canvas_runtime_state(runtime_state: dict) -> None:
    documents = get_canvas_runtime_documents(runtime_state)
    runtime_state["documents"] = documents
    runtime_state["active_document_id"] = extract_canvas_active_document_id(
        {"active_document_id": runtime_state.get("active_document_id")},
        documents,
    )
    runtime_state["mode"] = determine_canvas_mode(documents)


def _find_canvas_document(
    runtime_state: dict,
    document_id: str | None = None,
    document_path: str | None = None,
) -> tuple[int, dict]:
    documents = runtime_state.get("documents") if isinstance(runtime_state, dict) else None
    if not isinstance(documents, list) or not documents:
        raise ValueError("No canvas document is available yet.")

    normalized_path = _normalize_document_path_for_lookup(document_path)
    if normalized_path:
        match = _find_canvas_document_by_path_locator(documents, normalized_path)
        if match:
            return match
        raise ValueError(f"Canvas document not found for path: {normalized_path}")

    target_id = str(document_id or runtime_state.get("active_document_id") or "").strip()
    if target_id:
        for index, document in enumerate(documents):
            if str(document.get("id") or "") == target_id:
                return index, document
        raise ValueError(f"Canvas document not found for id: {target_id}")

    return len(documents) - 1, documents[-1]


def _store_canvas_document(runtime_state: dict, document: dict) -> dict:
    normalized = normalize_canvas_document(document)
    if not normalized:
        raise ValueError("Canvas document is invalid.")

    documents = runtime_state.setdefault("documents", [])
    updated = False
    for index, existing in enumerate(documents):
        if existing.get("id") == normalized["id"]:
            documents[index] = normalized
            updated = True
            break
    if not updated:
        documents.append(normalized)
        if len(documents) > CANVAS_MAX_DOCUMENTS:
            documents[:] = documents[-CANVAS_MAX_DOCUMENTS:]
    runtime_state["active_document_id"] = normalized["id"]
    _refresh_canvas_runtime_state(runtime_state)
    return normalized


def create_canvas_document(
    runtime_state: dict,
    title: str,
    content: str,
    format_name: str = "markdown",
    language_name: str | None = None,
    path: str | None = None,
    role: str | None = None,
    summary: str | None = None,
    imports: list[str] | None = None,
    exports: list[str] | None = None,
    symbols: list[str] | None = None,
    dependencies: list[str] | None = None,
    project_id: str | None = None,
    workspace_id: str | None = None,
) -> dict:
    normalized = normalize_canvas_document(
        {
            "id": uuid4().hex,
            "title": title or "Canvas",
            "format": format_name,
            "content": content,
            "language": language_name,
            "path": path,
            "role": role,
            "summary": summary,
            "imports": imports,
            "exports": exports,
            "symbols": symbols,
            "dependencies": dependencies,
            "project_id": project_id,
            "workspace_id": workspace_id,
        }
    )
    return _store_canvas_document(runtime_state, normalized)


def rewrite_canvas_document(
    runtime_state: dict,
    content: str,
    document_id: str | None = None,
    document_path: str | None = None,
    title: str | None = None,
    format_name: str | None = None,
    language_name: str | None = None,
    path: str | None = None,
    role: str | None = None,
    summary: str | None = None,
    imports: list[str] | None = None,
    exports: list[str] | None = None,
    symbols: list[str] | None = None,
    dependencies: list[str] | None = None,
    project_id: str | None = None,
    workspace_id: str | None = None,
) -> dict:
    _, document = _find_canvas_document(runtime_state, document_id=document_id, document_path=document_path)
    next_document = dict(document)
    next_document["content"] = _clip_text(content, CANVAS_MAX_CONTENT_LENGTH)
    if title is not None:
        next_document["title"] = str(title or "Canvas").strip()[:CANVAS_MAX_TITLE_LENGTH] or "Canvas"
    if format_name is not None:
        next_document["format"] = format_name
    if language_name is not None:
        next_document["language"] = language_name
    if path is not None:
        next_document["path"] = path
    if role is not None:
        next_document["role"] = role
    if summary is not None:
        next_document["summary"] = summary
    if imports is not None:
        next_document["imports"] = imports
    if exports is not None:
        next_document["exports"] = exports
    if symbols is not None:
        next_document["symbols"] = symbols
    if dependencies is not None:
        next_document["dependencies"] = dependencies
    if project_id is not None:
        next_document["project_id"] = project_id
    if workspace_id is not None:
        next_document["workspace_id"] = workspace_id
    return _store_canvas_document(runtime_state, next_document)


def replace_canvas_lines(
    runtime_state: dict,
    start_line: int,
    end_line: int,
    lines: list[str],
    document_id: str | None = None,
    document_path: str | None = None,
) -> dict:
    _, document = _find_canvas_document(runtime_state, document_id=document_id, document_path=document_path)
    existing_lines = list_canvas_lines(document.get("content") or "")
    if start_line < 1 or end_line < start_line:
        raise ValueError("start_line and end_line must define a valid 1-based inclusive range.")
    if start_line > len(existing_lines):
        raise ValueError("Line range exceeds the current canvas document.")
    if end_line > len(existing_lines):
        raise ValueError("Line range exceeds the current canvas document.")

    replacement = [str(line) for line in (lines or [])]
    next_lines = [*existing_lines[: start_line - 1], *replacement, *existing_lines[end_line:]]
    next_document = dict(document)
    next_document["content"] = join_canvas_lines(next_lines)
    return _store_canvas_document(runtime_state, next_document)


def insert_canvas_lines(
    runtime_state: dict,
    after_line: int,
    lines: list[str],
    document_id: str | None = None,
    document_path: str | None = None,
) -> dict:
    _, document = _find_canvas_document(runtime_state, document_id=document_id, document_path=document_path)
    existing_lines = list_canvas_lines(document.get("content") or "")
    if after_line < 0 or after_line > len(existing_lines):
        raise ValueError("after_line must be between 0 and the current line count.")

    additions = [str(line) for line in (lines or [])]
    next_lines = [*existing_lines[:after_line], *additions, *existing_lines[after_line:]]
    next_document = dict(document)
    next_document["content"] = join_canvas_lines(next_lines)
    return _store_canvas_document(runtime_state, next_document)


def delete_canvas_lines(
    runtime_state: dict,
    start_line: int,
    end_line: int,
    document_id: str | None = None,
    document_path: str | None = None,
) -> dict:
    return replace_canvas_lines(runtime_state, start_line, end_line, [], document_id=document_id, document_path=document_path)


def scroll_canvas_document(
    runtime_state: dict,
    start_line: int,
    end_line: int,
    document_id: str | None = None,
    document_path: str | None = None,
    max_window_lines: int = 200,
    max_chars: int | None = None,
) -> dict:
    _, document = _find_canvas_document(runtime_state, document_id=document_id, document_path=document_path)
    existing_lines = list_canvas_lines(document.get("content") or "")
    total_lines = len(existing_lines)
    if start_line < 1 or end_line < start_line:
        raise ValueError("start_line and end_line must define a valid 1-based inclusive range.")
    if total_lines == 0:
        return {
            "status": "ok",
            "action": "scrolled",
            "document_id": document.get("id"),
            "document_path": document.get("path"),
            "title": document.get("title"),
            "start_line": 1,
            "end_line_actual": 0,
            "total_lines": 0,
            "visible_lines": [],
            "has_more_above": False,
            "has_more_below": False,
        }

    window_limit = max(1, int(max_window_lines or 1))
    effective_start = min(start_line, total_lines)
    effective_end = min(total_lines, end_line, effective_start + window_limit - 1)
    if max_chars is None:
        max_chars = scale_canvas_char_limit(max_window_lines, default_lines=200, default_chars=8_000)

    visible_lines = []
    visible_char_count = 0
    for index in range(effective_start, effective_end + 1):
        numbered_line = f"{index}: {existing_lines[index - 1]}"
        extra_chars = len(numbered_line) + (1 if visible_lines else 0)
        if visible_lines and visible_char_count + extra_chars > max_chars:
            effective_end = index - 1
            break
        if not visible_lines and extra_chars > max_chars:
            visible_lines.append(numbered_line[:max_chars])
            effective_end = index
            break
        visible_lines.append(numbered_line)
        visible_char_count += extra_chars

    return {
        "status": "ok",
        "action": "scrolled",
        "document_id": document.get("id"),
        "document_path": document.get("path"),
        "title": document.get("title"),
        "start_line": effective_start,
        "end_line_actual": effective_end,
        "total_lines": total_lines,
        "visible_lines": visible_lines,
        "has_more_above": effective_start > 1,
        "has_more_below": effective_end < total_lines,
    }


def delete_canvas_document(
    runtime_state: dict,
    document_id: str | None = None,
    document_path: str | None = None,
) -> dict:
    index, document = _find_canvas_document(runtime_state, document_id=document_id, document_path=document_path)
    documents = runtime_state.get("documents") if isinstance(runtime_state, dict) else None
    if not isinstance(documents, list):
        raise ValueError("No canvas document is available yet.")

    previous_active_document_id = get_canvas_runtime_active_document_id(runtime_state)
    removed = documents.pop(index)
    if documents:
        runtime_state["active_document_id"] = (
            documents[-1]["id"]
            if str(removed.get("id") or "") == str(previous_active_document_id or "")
            else previous_active_document_id
        )
    else:
        runtime_state["active_document_id"] = None
    _refresh_canvas_runtime_state(runtime_state)
    return {
        "status": "ok",
        "action": "deleted",
        "deleted_id": removed.get("id"),
        "deleted_title": removed.get("title"),
        "remaining_count": len(documents),
    }


def clear_canvas(runtime_state: dict) -> dict:
    documents = get_canvas_runtime_documents(runtime_state)
    cleared_count = len(documents)
    runtime_state["documents"] = []
    runtime_state["active_document_id"] = None
    runtime_state["mode"] = CANVAS_MODE_DOCUMENT
    return {
        "status": "ok",
        "action": "cleared",
        "cleared_count": cleared_count,
    }


def build_canvas_tool_result(document: dict, *, action: str) -> dict:
    normalized = normalize_canvas_document(document)
    if not normalized:
        raise ValueError("Canvas document is invalid.")
    preview = normalized["content"][:2000]
    result = {
        "status": "ok",
        "action": action,
        "document": normalized,
        "document_id": normalized["id"],
        "primary_locator": extract_canvas_primary_locator(normalized),
        "title": normalized["title"],
        "format": normalized["format"],
        "line_count": normalized["line_count"],
        "content": preview,
        "content_truncated": len(normalized["content"]) > len(preview),
    }
    if normalized.get("language"):
        result["language"] = normalized["language"]
    for key in ("path", "role", "summary", "project_id", "workspace_id"):
        if normalized.get(key):
            result[key] = normalized[key]
    for key in ("imports", "exports", "symbols", "dependencies"):
        values = normalized.get(key) if isinstance(normalized.get(key), list) else []
        if values:
            result[key] = values
    return result


def build_canvas_document_context_result(
    runtime_state: dict,
    *,
    document_id: str | None = None,
    document_path: str | None = None,
    max_lines: int | None = None,
    max_chars: int | None = None,
) -> dict:
    _, document = _find_canvas_document(runtime_state, document_id=document_id, document_path=document_path)
    normalized = normalize_canvas_document(document)
    if not normalized:
        raise ValueError("Canvas document is invalid.")

    numbered_lines, is_truncated = _number_canvas_lines(
        normalized.get("content") or "",
        max_lines=max_lines or CANVAS_CONTEXT_MAX_LINES,
        max_chars=max_chars,
    )
    documents = get_canvas_runtime_documents(runtime_state)
    manifest = build_canvas_project_manifest(documents, active_document_id=get_canvas_runtime_active_document_id(runtime_state))
    relationship_map = build_canvas_relationship_map(documents)
    return {
        "status": "ok",
        "action": "expanded",
        "document": normalized,
        "document_id": normalized["id"],
        "document_path": normalized.get("path"),
        "title": normalized["title"],
        "format": normalized["format"],
        "language": normalized.get("language"),
        "role": normalized.get("role"),
        "summary": normalized.get("summary"),
        "line_count": normalized.get("line_count"),
        "visible_lines": numbered_lines,
        "visible_line_end": len(numbered_lines),
        "is_truncated": is_truncated,
        "primary_locator": extract_canvas_primary_locator(normalized),
        "manifest_excerpt": {
            "project_name": (manifest or {}).get("project_name"),
            "target_type": (manifest or {}).get("target_type"),
            "active_file": (manifest or {}).get("active_file"),
        },
        "relationship_map": relationship_map,
    }


def find_latest_canvas_state(messages: list[dict]) -> dict:
    for message in reversed(messages or []):
        metadata = message.get("metadata") if isinstance(message, dict) else None
        if isinstance(metadata, dict) and metadata.get("canvas_cleared") is True:
            return create_canvas_runtime_state([], active_document_id=None)
        documents = extract_canvas_documents(metadata)
        if not documents:
            continue
        active_document_id = extract_canvas_active_document_id(metadata, documents)
        return create_canvas_runtime_state(documents, active_document_id=active_document_id)
    return create_canvas_runtime_state()


def find_latest_canvas_documents(messages: list[dict]) -> list[dict]:
    runtime_state = find_latest_canvas_state(messages)
    message_id = None
    for message in reversed(messages or []):
        metadata = message.get("metadata") if isinstance(message, dict) else None
        if isinstance(metadata, dict) and extract_canvas_documents(metadata):
            message_id = message.get("id") if isinstance(message.get("id"), int) else None
            break
    results = []
    for document in get_canvas_runtime_documents(runtime_state):
        result = dict(document)
        if message_id is not None:
            result["source_message_id"] = message_id
        results.append(result)
    return results


def find_latest_canvas_document(
    messages: list[dict],
    document_id: str | None = None,
    document_path: str | None = None,
) -> dict | None:
    target_id = str(document_id or "").strip()
    target_path = _normalize_document_path_for_lookup(document_path)
    documents = list(reversed(find_latest_canvas_documents(messages)))
    if target_path:
        match = _find_canvas_document_by_path_locator(documents, target_path)
        if match:
            _, document = match
            return dict(document)
        return None
    for document in documents:
        if not target_id or document.get("id") == target_id:
            return dict(document)
    return None


def build_markdown_download(document: dict) -> bytes:
    normalized = normalize_canvas_document(document)
    if not normalized:
        raise ValueError("Canvas document is invalid.")
    content = _normalize_line_endings(normalized["content"])
    if normalized.get("format") == "code":
        language = normalized.get("language") or "text"
        return f"```{language}\n{content}\n```\n".encode("utf-8")
    return content.encode("utf-8")


def build_html_download(document: dict) -> bytes:
    normalized = normalize_canvas_document(document)
    if not normalized:
        raise ValueError("Canvas document is invalid.")

    content = _normalize_line_endings(normalized["content"])
    if normalized.get("format") == "code":
        language = escape(normalized.get("language") or "text")
        rendered = f'<pre><code class="language-{language}">{escape(content)}</code></pre>'
    elif markdown_lib is not None:
        rendered = markdown_lib.markdown(
            content,
            extensions=["extra", "fenced_code", "tables", "sane_lists"],
        )
    else:
        rendered = f"<pre>{escape(content)}</pre>"

    title = escape(normalized["title"])
    html = f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <title>{title}</title>
    <style>
        :root {{
            color-scheme: light;
            --bg: #f6f7fb;
            --surface: #ffffff;
            --text: #162033;
            --muted: #52607a;
            --border: #d8dfeb;
            --accent: #3157d5;
            --code-bg: #eef2fb;
        }}
        * {{ box-sizing: border-box; }}
        body {{
            margin: 0;
            background: linear-gradient(180deg, #eef2ff 0%, var(--bg) 220px);
            color: var(--text);
            font: 16px/1.7 \"Segoe UI\", Arial, sans-serif;
        }}
        main {{
            width: min(900px, calc(100vw - 32px));
            margin: 32px auto;
            padding: 32px;
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 20px;
            box-shadow: 0 24px 70px rgba(22, 32, 51, 0.08);
        }}
        h1, h2, h3, h4 {{ line-height: 1.25; color: #0f1728; }}
        p, li, blockquote {{ color: var(--text); }}
        blockquote {{ border-left: 4px solid var(--accent); margin: 1rem 0; padding: 0.1rem 0 0.1rem 1rem; color: var(--muted); }}
        pre {{ background: var(--code-bg); border: 1px solid var(--border); border-radius: 14px; padding: 14px; overflow-x: auto; }}
        code {{ background: var(--code-bg); border-radius: 6px; padding: 0.15em 0.35em; font-family: \"Cascadia Code\", Consolas, monospace; }}
        pre code {{ background: transparent; padding: 0; }}
        table {{ width: 100%; border-collapse: collapse; margin: 1rem 0; }}
        th, td {{ border: 1px solid var(--border); padding: 10px 12px; text-align: left; vertical-align: top; }}
        th {{ background: #f3f6fd; }}
        a {{ color: var(--accent); }}
    </style>
</head>
<body>
    <main>
        <article>
            {rendered}
        </article>
    </main>
</body>
</html>
"""
    return html.encode("utf-8")


def build_pdf_download(document: dict) -> bytes:
    normalized = normalize_canvas_document(document)
    if not normalized:
        raise ValueError("Canvas document is invalid.")

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "CanvasTitle",
        parent=styles["Title"],
        fontName=_BOLD_FONT,
    )
    body_style = ParagraphStyle(
        "CanvasBody",
        parent=styles["BodyText"],
        fontName=_BODY_FONT,
        fontSize=10,
        leading=14,
        spaceAfter=6,
    )
    heading1_style = ParagraphStyle(
        "CanvasH1",
        parent=styles["Heading1"],
        fontName=_BOLD_FONT,
        textColor=colors.HexColor("#1f2a44"),
        spaceAfter=8,
        spaceBefore=14,
    )
    heading_style = ParagraphStyle(
        "CanvasHeading",
        parent=styles["Heading2"],
        fontName=_BOLD_FONT,
        textColor=colors.HexColor("#1f2a44"),
        spaceAfter=8,
        spaceBefore=10,
    )
    code_style = ParagraphStyle(
        "CanvasCode",
        parent=styles["Code"],
        fontName=_MONO_FONT,
        fontSize=8.5,
        leading=11,
        leftIndent=10,
        rightIndent=10,
        backColor=colors.HexColor("#f3f5f9"),
        borderPadding=8,
    )

    story = [Paragraph(normalized["title"], title_style), Spacer(1, 6)]
    if normalized.get("format") == "code":
        story.append(Preformatted(_normalize_line_endings(normalized["content"]), code_style))
        output = BytesIO()
        doc = SimpleDocTemplate(output, pagesize=A4, topMargin=18 * mm, bottomMargin=18 * mm)
        doc.build(story)
        return output.getvalue()

    lines = list_canvas_lines(normalized["content"])
    in_code_block = False
    code_lines: list[str] = []
    list_buffer: list[str] = []

    def flush_list_buffer():
        nonlocal list_buffer
        if not list_buffer:
            return
        flowable = ListFlowable(
            [ListItem(Paragraph(item, body_style)) for item in list_buffer],
            bulletType="bullet",
            leftIndent=18,
        )
        story.append(flowable)
        story.append(Spacer(1, 4))
        list_buffer = []

    def flush_code_lines():
        nonlocal code_lines
        if not code_lines:
            return
        story.append(Preformatted("\n".join(code_lines), code_style))
        story.append(Spacer(1, 6))
        code_lines = []

    for raw_line in lines:
        line = raw_line.rstrip()
        stripped = line.strip()
        if stripped.startswith("```"):
            flush_list_buffer()
            if in_code_block:
                flush_code_lines()
            in_code_block = not in_code_block
            continue
        if in_code_block:
            code_lines.append(line)
            continue
        if not stripped:
            flush_list_buffer()
            story.append(Spacer(1, 4))
            continue
        if stripped.startswith("#"):
            flush_list_buffer()
            level = len(stripped) - len(stripped.lstrip("#"))
            heading_text = stripped[level:].strip() or "Untitled"
            style = heading1_style if level <= 1 else heading_style
            story.append(Paragraph(heading_text, style))
            continue
        if stripped.startswith(("- ", "* ")):
            list_buffer.append(stripped[2:].strip())
            continue
        flush_list_buffer()
        story.append(Paragraph(stripped.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"), body_style))

    flush_list_buffer()
    flush_code_lines()

    output = BytesIO()
    doc = SimpleDocTemplate(output, pagesize=A4, topMargin=18 * mm, bottomMargin=18 * mm)
    doc.build(story)
    return output.getvalue()