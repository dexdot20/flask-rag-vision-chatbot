from __future__ import annotations

import ast
import difflib
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys
import tomllib

from config import PROJECT_WORKSPACE_ROOT

WORKSPACE_MAX_READ_LINES = 800
WORKSPACE_MAX_SEARCH_RESULTS = 50
WORKSPACE_MAX_FILE_BYTES = 200_000
WORKSPACE_MAX_WRITE_BATCH = 100
WORKSPACE_MAX_DIFF_CHARS = 12_000
WORKSPACE_HISTORY_DIRNAME = ".workspace-history"
WORKSPACE_HISTORY_MAX_EVENTS = 20
WORKSPACE_VALIDATION_MAX_NOTES = 50
WORKFLOW_ALLOWED_STAGES = {"plan", "skeleton", "content", "validate", "fix", "validated"}
WORKFLOW_ALLOWED_TARGET_TYPES = {"python-project"}
WORKFLOW_MAX_FILES = 64
WORKFLOW_MAX_LIST_ITEMS = 24


def _slugify_workspace_name(value: str) -> str:
    normalized = "".join(ch.lower() if ch.isalnum() else "-" for ch in str(value or "").strip())
    collapsed = "-".join(part for part in normalized.split("-") if part)
    return collapsed[:80] or "project"


def _normalize_short_text(value: str | None, max_length: int = 200) -> str | None:
    normalized = " ".join(str(value or "").strip().split())[:max_length]
    return normalized or None


def _normalize_string_list(values, *, max_items: int = WORKFLOW_MAX_LIST_ITEMS, max_length: int = 120) -> list[str]:
    if not isinstance(values, (list, tuple, set)):
        return []
    cleaned = []
    seen = set()
    for raw_value in values:
        item = _normalize_short_text(raw_value, max_length=max_length)
        if not item:
            continue
        dedupe_key = item.lower()
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        cleaned.append(item)
        if len(cleaned) >= max_items:
            break
    return cleaned


def _normalize_workflow_files(values) -> list[dict]:
    if not isinstance(values, list):
        return []
    cleaned = []
    seen_paths = set()
    for entry in values[:WORKFLOW_MAX_FILES]:
        if not isinstance(entry, dict):
            continue
        path = _normalize_workspace_relative_path(entry.get("path"), allow_empty=True)
        if not path or path.lower() in seen_paths:
            continue
        seen_paths.add(path.lower())
        item = {"path": path}
        role = _normalize_short_text(entry.get("role"), max_length=24)
        purpose = _normalize_short_text(entry.get("purpose"), max_length=180)
        status = _normalize_short_text(entry.get("status"), max_length=40)
        if role:
            item["role"] = role
        if purpose:
            item["purpose"] = purpose
        if status:
            item["status"] = status
        cleaned.append(item)
    return cleaned


def normalize_project_workflow(value) -> dict | None:
    if not isinstance(value, dict):
        return None
    project_name = _normalize_short_text(value.get("project_name"), max_length=120)
    goal = _normalize_short_text(value.get("goal"), max_length=300)
    target_type = _normalize_short_text(value.get("target_type"), max_length=40)
    stage = _normalize_short_text(value.get("stage"), max_length=40)
    if stage not in WORKFLOW_ALLOWED_STAGES:
        stage = "plan"
    if target_type not in WORKFLOW_ALLOWED_TARGET_TYPES:
        target_type = "python-project"

    cleaned = {
        "stage": stage,
        "target_type": target_type,
    }
    if project_name:
        cleaned["project_name"] = project_name
    if goal:
        cleaned["goal"] = goal

    files = _normalize_workflow_files(value.get("files"))
    if files:
        cleaned["files"] = files

    dependencies = _normalize_string_list(value.get("dependencies"))
    if dependencies:
        cleaned["dependencies"] = dependencies

    open_issues = _normalize_string_list(value.get("open_issues"), max_length=180)
    if open_issues:
        cleaned["open_issues"] = open_issues

    validation = value.get("validation") if isinstance(value.get("validation"), dict) else None
    if validation:
        validation_status = _normalize_short_text(validation.get("status"), max_length=40) or "ok"
        validation_payload = {"status": validation_status}
        issues = _normalize_string_list(validation.get("issues"), max_length=200)
        warnings = _normalize_string_list(validation.get("warnings"), max_length=200)
        if issues:
            validation_payload["issues"] = issues
        if warnings:
            validation_payload["warnings"] = warnings
        cleaned["validation"] = validation_payload

    return cleaned


def create_project_workflow_runtime_state(initial_workflow: dict | None = None) -> dict:
    normalized = normalize_project_workflow(initial_workflow or {})
    return normalized or {"stage": "plan", "target_type": "python-project"}


def get_project_workflow(runtime_state: dict | None) -> dict | None:
    if not isinstance(runtime_state, dict):
        return None
    normalized = normalize_project_workflow(runtime_state)
    return normalized


def set_project_workflow(runtime_state: dict, workflow: dict) -> dict:
    normalized = normalize_project_workflow(workflow)
    if not normalized:
        raise ValueError("Project workflow is invalid.")
    runtime_state.clear()
    runtime_state.update(normalized)
    return dict(runtime_state)


def update_project_workflow(runtime_state: dict, **changes) -> dict:
    current = get_project_workflow(runtime_state) or {"stage": "plan", "target_type": "python-project"}
    merged = {**current, **changes}
    return set_project_workflow(runtime_state, merged)


def _default_workflow_files(project_name: str, target_type: str) -> list[dict]:
    root_name = _slugify_workspace_name(project_name)
    if target_type != "python-project":
        return []
    return [
        {"path": f"{root_name}/app.py", "role": "source", "purpose": "Application entry point", "status": "planned"},
        {"path": f"{root_name}/config.py", "role": "config", "purpose": "Runtime configuration", "status": "planned"},
        {"path": f"{root_name}/requirements.txt", "role": "dependency", "purpose": "Runtime dependencies", "status": "planned"},
        {"path": f"{root_name}/README.md", "role": "docs", "purpose": "Project overview and usage", "status": "planned"},
        {"path": f"{root_name}/tests/test_app.py", "role": "test", "purpose": "Basic regression coverage", "status": "planned"},
    ]


def build_project_plan(goal: str, project_name: str, target_type: str = "python-project", files: list[dict] | None = None, dependencies: list[str] | None = None) -> dict:
    normalized_target = _normalize_short_text(target_type, max_length=40) or "python-project"
    if normalized_target not in WORKFLOW_ALLOWED_TARGET_TYPES:
        normalized_target = "python-project"
    plan = {
        "stage": "plan",
        "goal": _normalize_short_text(goal, max_length=300) or f"Create {project_name}",
        "project_name": _normalize_short_text(project_name, max_length=120) or "Project",
        "target_type": normalized_target,
        "files": _normalize_workflow_files(files) or _default_workflow_files(project_name, normalized_target),
        "dependencies": _normalize_string_list(dependencies),
        "open_issues": [],
    }
    return normalize_project_workflow(plan) or {"stage": "plan", "target_type": normalized_target}


def find_latest_project_workflow(messages: list[dict]) -> dict | None:
    for message in reversed(messages or []):
        metadata = message.get("metadata") if isinstance(message, dict) else None
        workflow = normalize_project_workflow((metadata or {}).get("project_workflow")) if isinstance(metadata, dict) else None
        if workflow:
            return workflow
    return None


def get_workspace_root_for_conversation(conversation_id: int) -> str:
    return os.path.join(PROJECT_WORKSPACE_ROOT, f"conversation-{int(conversation_id)}")


def create_workspace_runtime_state(conversation_id: int | None = None, root_path: str | None = None) -> dict:
    resolved_root = str(root_path or "").strip()
    if not resolved_root and isinstance(conversation_id, int) and conversation_id > 0:
        resolved_root = get_workspace_root_for_conversation(conversation_id)
    return {
        "conversation_id": conversation_id if isinstance(conversation_id, int) and conversation_id > 0 else None,
        "root_path": resolved_root or None,
    }


def get_workspace_root(runtime_state: dict | None) -> str | None:
    if not isinstance(runtime_state, dict):
        return None
    root_path = str(runtime_state.get("root_path") or "").strip()
    return root_path or None


def ensure_workspace_available(runtime_state: dict | None) -> Path:
    root_path = get_workspace_root(runtime_state)
    if not root_path:
        raise ValueError("Workspace sandbox is not available for this conversation.")
    root = Path(root_path).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _get_history_root(root: Path) -> Path:
    history_root = root / WORKSPACE_HISTORY_DIRNAME
    history_root.mkdir(parents=True, exist_ok=True)
    return history_root


def _history_file_for_path(root: Path, normalized_path: str) -> Path:
    digest = hashlib.sha1(normalized_path.encode("utf-8")).hexdigest()
    return _get_history_root(root) / f"{digest}.json"


def _is_hidden_workspace_path(relative_path: str) -> bool:
    raw_value = str(relative_path or "").strip().replace("\\", "/")
    if not raw_value:
        return False
    normalized_parts: list[str] = []
    for part in PurePosixPath(raw_value).parts:
        if part in {"", "."}:
            continue
        if part == "..":
            return False
        normalized_parts.append(part)
    if not normalized_parts:
        return False
    return normalized_parts[0] == WORKSPACE_HISTORY_DIRNAME


def _read_text_if_exists(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("File is not valid UTF-8 text.") from exc


def _build_file_snapshot(path: Path) -> dict:
    content = _read_text_if_exists(path)
    if content is None:
        return {"exists": False, "content": ""}
    return {"exists": True, "content": content}


def _apply_file_snapshot(path: Path, snapshot: dict) -> None:
    exists = bool((snapshot or {}).get("exists"))
    if not exists:
        if path.exists():
            path.unlink()
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_clip_file_text((snapshot or {}).get("content") or ""), encoding="utf-8")


def _load_history_state(root: Path, normalized_path: str) -> dict:
    history_file = _history_file_for_path(root, normalized_path)
    if not history_file.exists():
        return {"path": normalized_path, "sequence": 0, "undo": [], "redo": []}
    try:
        payload = json.loads(history_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        payload = {}
    return {
        "path": normalized_path,
        "sequence": int(payload.get("sequence") or 0),
        "undo": payload.get("undo") if isinstance(payload.get("undo"), list) else [],
        "redo": payload.get("redo") if isinstance(payload.get("redo"), list) else [],
    }


def _save_history_state(root: Path, normalized_path: str, history_state: dict) -> None:
    history_file = _history_file_for_path(root, normalized_path)
    payload = {
        "path": normalized_path,
        "sequence": int(history_state.get("sequence") or 0),
        "undo": (history_state.get("undo") or [])[-WORKSPACE_HISTORY_MAX_EVENTS:],
        "redo": (history_state.get("redo") or [])[-WORKSPACE_HISTORY_MAX_EVENTS:],
    }
    history_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_unified_diff(path: str, before_content: str, after_content: str) -> str:
    before_lines = str(before_content or "").replace("\r\n", "\n").replace("\r", "\n").splitlines()
    after_lines = str(after_content or "").replace("\r\n", "\n").replace("\r", "\n").splitlines()
    diff_text = "\n".join(
        difflib.unified_diff(
            before_lines,
            after_lines,
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
            lineterm="",
        )
    )
    if len(diff_text) <= WORKSPACE_MAX_DIFF_CHARS:
        return diff_text
    return diff_text[:WORKSPACE_MAX_DIFF_CHARS].rstrip() + "\n...[diff truncated]"


def _describe_change_type(before_exists: bool, after_exists: bool) -> str:
    if before_exists and after_exists:
        return "modified"
    if after_exists:
        return "created"
    return "deleted"


def _record_file_change(root: Path, normalized_path: str, before: dict, after: dict, *, label: str) -> dict | None:
    if before == after:
        return None
    history_state = _load_history_state(root, normalized_path)
    next_sequence = int(history_state.get("sequence") or 0) + 1
    entry = {
        "revision_id": f"rev-{next_sequence}",
        "label": _normalize_short_text(label, max_length=80) or "workspace_change",
        "change_type": _describe_change_type(bool(before.get("exists")), bool(after.get("exists"))),
        "before": {
            "exists": bool(before.get("exists")),
            "content": _clip_file_text(before.get("content") or ""),
        },
        "after": {
            "exists": bool(after.get("exists")),
            "content": _clip_file_text(after.get("content") or ""),
        },
    }
    undo_entries = list(history_state.get("undo") or [])
    undo_entries.append(entry)
    history_state["sequence"] = next_sequence
    history_state["undo"] = undo_entries[-WORKSPACE_HISTORY_MAX_EVENTS:]
    history_state["redo"] = []
    _save_history_state(root, normalized_path, history_state)
    return entry


def _write_file_with_history(runtime_state: dict | None, normalized_path: str, content: str, *, label: str) -> dict:
    root = ensure_workspace_available(runtime_state)
    target, _ = resolve_workspace_path(runtime_state, normalized_path)
    before = _build_file_snapshot(target)
    normalized_content = _clip_file_text(content)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(normalized_content, encoding="utf-8")
    after = {"exists": True, "content": normalized_content}
    history_entry = _record_file_change(root, normalized_path, before, after, label=label)
    return {
        "path": normalized_path,
        "bytes_written": len(normalized_content.encode("utf-8")),
        "change_type": _describe_change_type(bool(before.get("exists")), True),
        "history_recorded": history_entry is not None,
        "revision_id": history_entry.get("revision_id") if history_entry else None,
    }


def preview_workspace_changes(runtime_state: dict | None, files: list[dict]) -> dict:
    previews = []
    for entry in (files or [])[:WORKSPACE_MAX_WRITE_BATCH]:
        if not isinstance(entry, dict):
            continue
        normalized_path = _normalize_workspace_relative_path(entry.get("path"))
        target, _ = resolve_workspace_path(runtime_state, normalized_path)
        before = _build_file_snapshot(target)
        after_content = _clip_file_text(entry.get("content") or "")
        previews.append(
            {
                "path": normalized_path,
                "change_type": _describe_change_type(bool(before.get("exists")), True),
                "before_exists": bool(before.get("exists")),
                "diff": _build_unified_diff(normalized_path, before.get("content") or "", after_content),
            }
        )
    return {
        "status": "ok",
        "action": "workspace_changes_previewed",
        "files": [preview["path"] for preview in previews],
        "diffs": previews,
    }


def get_workspace_file_history(runtime_state: dict | None, path: str) -> dict:
    root = ensure_workspace_available(runtime_state)
    _, normalized_path = resolve_workspace_path(runtime_state, path)
    history_state = _load_history_state(root, normalized_path)
    undo_entries = list(history_state.get("undo") or [])
    redo_entries = list(history_state.get("redo") or [])
    revisions = []
    for entry in reversed(undo_entries[-WORKSPACE_HISTORY_MAX_EVENTS:]):
        revisions.append(
            {
                "revision_id": entry.get("revision_id"),
                "label": entry.get("label"),
                "change_type": entry.get("change_type"),
            }
        )
    return {
        "status": "ok",
        "action": "workspace_file_history",
        "path": normalized_path,
        "undo_count": len(undo_entries),
        "redo_count": len(redo_entries),
        "revisions": revisions,
    }


def undo_workspace_file_change(runtime_state: dict | None, path: str) -> dict:
    root = ensure_workspace_available(runtime_state)
    target, normalized_path = resolve_workspace_path(runtime_state, path)
    history_state = _load_history_state(root, normalized_path)
    undo_entries = list(history_state.get("undo") or [])
    if not undo_entries:
        raise ValueError("No workspace change history is available for this file.")
    entry = undo_entries.pop()
    _apply_file_snapshot(target, entry.get("before") or {})
    redo_entries = list(history_state.get("redo") or [])
    redo_entries.append(entry)
    history_state["undo"] = undo_entries[-WORKSPACE_HISTORY_MAX_EVENTS:]
    history_state["redo"] = redo_entries[-WORKSPACE_HISTORY_MAX_EVENTS:]
    _save_history_state(root, normalized_path, history_state)
    current_snapshot = _build_file_snapshot(target)
    return {
        "status": "ok",
        "action": "workspace_file_undone",
        "path": normalized_path,
        "revision_id": entry.get("revision_id"),
        "change_type": entry.get("change_type"),
        "exists": bool(current_snapshot.get("exists")),
        "content": current_snapshot.get("content") or "",
        "undo_count": len(history_state.get("undo") or []),
        "redo_count": len(history_state.get("redo") or []),
    }


def redo_workspace_file_change(runtime_state: dict | None, path: str) -> dict:
    root = ensure_workspace_available(runtime_state)
    target, normalized_path = resolve_workspace_path(runtime_state, path)
    history_state = _load_history_state(root, normalized_path)
    redo_entries = list(history_state.get("redo") or [])
    if not redo_entries:
        raise ValueError("No workspace redo history is available for this file.")
    entry = redo_entries.pop()
    _apply_file_snapshot(target, entry.get("after") or {})
    undo_entries = list(history_state.get("undo") or [])
    undo_entries.append(entry)
    history_state["undo"] = undo_entries[-WORKSPACE_HISTORY_MAX_EVENTS:]
    history_state["redo"] = redo_entries[-WORKSPACE_HISTORY_MAX_EVENTS:]
    _save_history_state(root, normalized_path, history_state)
    current_snapshot = _build_file_snapshot(target)
    return {
        "status": "ok",
        "action": "workspace_file_redone",
        "path": normalized_path,
        "revision_id": entry.get("revision_id"),
        "change_type": entry.get("change_type"),
        "exists": bool(current_snapshot.get("exists")),
        "content": current_snapshot.get("content") or "",
        "undo_count": len(history_state.get("undo") or []),
        "redo_count": len(history_state.get("redo") or []),
    }


def _normalize_workspace_relative_path(value: str | None, *, allow_empty: bool = False) -> str:
    raw_value = str(value or "").strip().replace("\\", "/")
    if not raw_value:
        if allow_empty:
            return ""
        raise ValueError("A workspace-relative path is required.")
    if raw_value.startswith("/"):
        raise ValueError("Path must be relative to the workspace root.")
    normalized_parts: list[str] = []
    for part in PurePosixPath(raw_value).parts:
        if part in {"", "."}:
            continue
        if part == "..":
            raise ValueError("Path cannot escape the workspace root.")
        normalized_parts.append(part)
    normalized = "/".join(normalized_parts)
    if not normalized and not allow_empty:
        raise ValueError("A workspace-relative path is required.")
    if normalized.split("/", 1)[0] == WORKSPACE_HISTORY_DIRNAME:
        raise ValueError("Path uses a reserved workspace directory.")
    return normalized


def resolve_workspace_path(runtime_state: dict | None, relative_path: str | None, *, allow_empty: bool = False) -> tuple[Path, str]:
    root = ensure_workspace_available(runtime_state)
    normalized_relative = _normalize_workspace_relative_path(relative_path, allow_empty=allow_empty)
    target = (root / normalized_relative).resolve() if normalized_relative else root
    if root != target and root not in target.parents:
        raise ValueError("Resolved path escapes the workspace root.")
    return target, normalized_relative


def _clip_file_text(content: str) -> str:
    normalized = str(content or "").replace("\r\n", "\n").replace("\r", "\n")
    encoded = normalized.encode("utf-8")
    if len(encoded) <= WORKSPACE_MAX_FILE_BYTES:
        return normalized
    return encoded[:WORKSPACE_MAX_FILE_BYTES].decode("utf-8", errors="ignore")


def create_directory(runtime_state: dict | None, path: str) -> dict:
    target, normalized_path = resolve_workspace_path(runtime_state, path)
    target.mkdir(parents=True, exist_ok=True)
    return {
        "status": "ok",
        "action": "directory_created",
        "path": normalized_path,
    }


def create_file(runtime_state: dict | None, path: str, content: str) -> dict:
    target, normalized_path = resolve_workspace_path(runtime_state, path)
    if target.exists():
        raise ValueError("File already exists. Use update_file for existing files.")
    write_result = _write_file_with_history(runtime_state, normalized_path, content, label="create_file")
    return {
        "status": "ok",
        "action": "file_created",
        "path": normalized_path,
        "bytes_written": write_result["bytes_written"],
        "revision_id": write_result["revision_id"],
    }


def update_file(runtime_state: dict | None, path: str, content: str) -> dict:
    target, normalized_path = resolve_workspace_path(runtime_state, path)
    if not target.exists() or not target.is_file():
        raise ValueError("File not found. Use create_file for new files.")
    write_result = _write_file_with_history(runtime_state, normalized_path, content, label="update_file")
    return {
        "status": "ok",
        "action": "file_updated",
        "path": normalized_path,
        "bytes_written": write_result["bytes_written"],
        "revision_id": write_result["revision_id"],
    }


def read_file(runtime_state: dict | None, path: str, start_line: int = 1, end_line: int | None = None) -> dict:
    target, normalized_path = resolve_workspace_path(runtime_state, path)
    if not target.exists() or not target.is_file():
        raise ValueError("File not found.")
    try:
        content = target.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("File is not valid UTF-8 text.") from exc

    lines = content.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    start = max(1, int(start_line or 1))
    if end_line is None:
        end = min(len(lines), start + WORKSPACE_MAX_READ_LINES - 1)
    else:
        end = max(start, min(len(lines), int(end_line)))
    selected = lines[start - 1 : end]
    numbered_lines = [f"{index}: {line}" for index, line in enumerate(selected, start=start)]
    return {
        "status": "ok",
        "action": "file_read",
        "path": normalized_path,
        "start_line": start,
        "end_line": end,
        "line_count": len(lines),
        "content": "\n".join(numbered_lines),
        "is_truncated": end < len(lines),
    }


def list_dir(runtime_state: dict | None, path: str | None = None) -> dict:
    target, normalized_path = resolve_workspace_path(runtime_state, path, allow_empty=True)
    if not target.exists() or not target.is_dir():
        raise ValueError("Directory not found.")
    entries = []
    for child in sorted(target.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower())):
        rel = child.relative_to(ensure_workspace_available(runtime_state)).as_posix()
        if _is_hidden_workspace_path(rel):
            continue
        entries.append(
            {
                "path": rel,
                "type": "directory" if child.is_dir() else "file",
            }
        )
    return {
        "status": "ok",
        "action": "directory_listed",
        "path": normalized_path,
        "entries": entries,
    }


def search_files(runtime_state: dict | None, query: str, path_prefix: str | None = None, search_content: bool = False) -> dict:
    root = ensure_workspace_available(runtime_state)
    normalized_prefix = _normalize_workspace_relative_path(path_prefix, allow_empty=True)
    search_root = (root / normalized_prefix).resolve() if normalized_prefix else root
    if not search_root.exists() or not search_root.is_dir():
        raise ValueError("Search root directory not found.")
    needle = str(query or "").strip().lower()
    if not needle:
        raise ValueError("Search query is required.")

    matches = []
    for current_path in sorted(search_root.rglob("*")):
        rel = current_path.relative_to(root).as_posix()
        if _is_hidden_workspace_path(rel):
            continue
        if needle in rel.lower():
            matches.append({"path": rel, "match_type": "path"})
        elif search_content and current_path.is_file():
            try:
                content = current_path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            for index, line in enumerate(content.replace("\r\n", "\n").replace("\r", "\n").split("\n"), start=1):
                if needle in line.lower():
                    matches.append({"path": rel, "match_type": "content", "line": index, "excerpt": line[:200]})
                    break
        if len(matches) >= WORKSPACE_MAX_SEARCH_RESULTS:
            break
    return {
        "status": "ok",
        "action": "files_searched",
        "query": query,
        "matches": matches,
    }


def _python_project_scaffold(project_name: str) -> tuple[list[str], list[dict]]:
    root_name = _slugify_workspace_name(project_name)
    directories = [root_name, f"{root_name}/tests"]
    files = [
        {"path": f"{root_name}/README.md", "content": f"# {project_name}\n"},
        {"path": f"{root_name}/app.py", "content": "def main():\n    return 'hello'\n\n\nif __name__ == '__main__':\n    print(main())\n"},
        {"path": f"{root_name}/config.py", "content": "SETTINGS = {}\n"},
        {"path": f"{root_name}/requirements.txt", "content": "\n"},
        {"path": f"{root_name}/pyproject.toml", "content": "[project]\nname = '" + root_name.replace("-", "_") + "'\nversion = '0.1.0'\n\n"},
        {"path": f"{root_name}/tests/test_app.py", "content": "from app import main\n\n\ndef test_main():\n    assert main() == 'hello'\n"},
    ]
    return directories, files


def create_project_scaffold(runtime_state: dict | None, project_name: str, target_type: str = "python-project", confirm: bool = False) -> dict:
    normalized_name = str(project_name or "").strip() or "Project"
    normalized_target = str(target_type or "python-project").strip().lower() or "python-project"
    if normalized_target != "python-project":
        raise ValueError("Only python-project scaffold is supported right now.")
    directories, files = _python_project_scaffold(normalized_name)
    root_dir = directories[0]
    target_root, _ = resolve_workspace_path(runtime_state, root_dir)
    if target_root.exists() and any(target_root.iterdir()) and not confirm:
        preview = preview_workspace_changes(runtime_state, files)
        return {
            "status": "needs_confirmation",
            "action": "project_scaffold_preview",
            "project_root": root_dir,
            "target_type": normalized_target,
            "directories": directories,
            "files": [entry["path"] for entry in files],
            "diffs": preview["diffs"],
            "message": "Project root already exists and is not empty. Re-run with confirm=true to continue.",
        }
    written = write_project_tree(runtime_state, directories=directories, files=files, confirm=confirm)
    written.update(
        {
            "action": "project_scaffold_created",
            "project_root": root_dir,
            "target_type": normalized_target,
        }
    )
    return written


def _check_overwrites(runtime_state: dict | None, files: list[dict]) -> list[str]:
    overwrites = []
    for entry in files:
        path = str(entry.get("path") or "").strip()
        if not path:
            continue
        target, normalized_path = resolve_workspace_path(runtime_state, path)
        if target.exists():
            overwrites.append(normalized_path)
    return overwrites


def write_project_tree(runtime_state: dict | None, directories: list[str] | None = None, files: list[dict] | None = None, confirm: bool = False) -> dict:
    normalized_directories = [_normalize_workspace_relative_path(path) for path in (directories or [])[:WORKSPACE_MAX_WRITE_BATCH]]
    normalized_files = []
    for entry in (files or [])[:WORKSPACE_MAX_WRITE_BATCH]:
        if not isinstance(entry, dict):
            continue
        normalized_files.append(
            {
                "path": _normalize_workspace_relative_path(entry.get("path")),
                "content": _clip_file_text(entry.get("content") or ""),
            }
        )

    overwrites = _check_overwrites(runtime_state, normalized_files)
    preview = preview_workspace_changes(runtime_state, normalized_files)
    if overwrites and not confirm:
        return {
            "status": "needs_confirmation",
            "action": "project_tree_preview",
            "directories": normalized_directories,
            "files": [entry["path"] for entry in normalized_files],
            "overwrites": overwrites,
            "diffs": preview["diffs"],
            "message": "Some files already exist. Re-run with confirm=true to overwrite them.",
        }

    for directory in normalized_directories:
        target, _ = resolve_workspace_path(runtime_state, directory)
        target.mkdir(parents=True, exist_ok=True)
    written_files = []
    for entry in normalized_files:
        write_result = _write_file_with_history(runtime_state, entry["path"], entry["content"], label="write_project_tree")
        written_files.append(write_result)
    return {
        "status": "ok",
        "action": "project_tree_written",
        "directories": normalized_directories,
        "files": [entry["path"] for entry in normalized_files],
        "overwrites": overwrites,
        "diffs": preview["diffs"],
        "revision_ids": [item["revision_id"] for item in written_files if item.get("revision_id")],
    }


def bulk_update_workspace_files(runtime_state: dict | None, files: list[dict], confirm: bool = False) -> dict:
    normalized_files = []
    for entry in (files or [])[:WORKSPACE_MAX_WRITE_BATCH]:
        if not isinstance(entry, dict):
            continue
        normalized_files.append(
            {
                "path": _normalize_workspace_relative_path(entry.get("path")),
                "content": _clip_file_text(entry.get("content") or ""),
            }
        )
    overwrites = _check_overwrites(runtime_state, normalized_files)
    preview = preview_workspace_changes(runtime_state, normalized_files)
    if overwrites and not confirm:
        return {
            "status": "needs_confirmation",
            "action": "bulk_update_preview",
            "files": [entry["path"] for entry in normalized_files],
            "overwrites": overwrites,
            "diffs": preview["diffs"],
            "message": "Bulk update would overwrite existing files. Re-run with confirm=true to apply.",
        }
    written_files = []
    for entry in normalized_files:
        write_result = _write_file_with_history(runtime_state, entry["path"], entry["content"], label="bulk_update_workspace_files")
        written_files.append(write_result)
    return {
        "status": "ok",
        "action": "bulk_update_applied",
        "files": [entry["path"] for entry in normalized_files],
        "overwrites": overwrites,
        "diffs": preview["diffs"],
        "revision_ids": [item["revision_id"] for item in written_files if item.get("revision_id")],
    }


def _module_name_variants(relative_path: str) -> set[str]:
    normalized = str(relative_path or "").strip()
    if not normalized.endswith(".py"):
        return set()
    stem = normalized[:-3]
    variants = set()
    if stem.endswith("/__init__"):
        variants.add(stem[: -len("/__init__")].replace("/", "."))
    else:
        variants.add(stem.replace("/", "."))
    if normalized.startswith("src/"):
        trimmed = normalized[4:]
        if trimmed.endswith(".py"):
            trimmed_stem = trimmed[:-3]
            if trimmed_stem.endswith("/__init__"):
                variants.add(trimmed_stem[: -len("/__init__")].replace("/", "."))
            else:
                variants.add(trimmed_stem.replace("/", "."))
    return {variant for variant in variants if variant}


def _collect_project_module_names(relative_files: set[str]) -> set[str]:
    modules = set()
    for relative_path in relative_files:
        modules.update(_module_name_variants(relative_path))
    return modules


def _parse_requirement_names(target_root: Path) -> set[str]:
    requirements_path = target_root / "requirements.txt"
    if not requirements_path.exists():
        return set()
    names = set()
    for line in requirements_path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue
        package = raw.split(";", 1)[0].split("==", 1)[0].split(">=", 1)[0].split("[", 1)[0].strip()
        package = package.replace("-", "_").lower()
        if package:
            names.add(package)
    return names


def _safe_read_toml(path: Path) -> dict:
    if not path.exists() or not path.is_file():
        return {}
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return {}


def _has_python_entry_point(target_root: Path, relative_files: set[str]) -> bool:
    if "app.py" in relative_files or "main.py" in relative_files or "__main__.py" in relative_files:
        return True
    if any(path.endswith("/__main__.py") for path in relative_files):
        return True
    pyproject = _safe_read_toml(target_root / "pyproject.toml")
    project_table = pyproject.get("project") if isinstance(pyproject.get("project"), dict) else {}
    if isinstance(project_table.get("scripts"), dict) and project_table.get("scripts"):
        return True
    tool_table = pyproject.get("tool") if isinstance(pyproject.get("tool"), dict) else {}
    poetry_table = tool_table.get("poetry") if isinstance(tool_table.get("poetry"), dict) else {}
    if isinstance(poetry_table.get("scripts"), dict) and poetry_table.get("scripts"):
        return True
    return False


def _resolve_relative_import(file_path: str, level: int, module: str | None) -> str:
    parts = file_path[:-3].split("/") if file_path.endswith(".py") else file_path.split("/")
    if parts and parts[-1] == "__init__":
        base_parts = parts[:-1]
    else:
        base_parts = parts[:-1]
    if level > 0:
        base_parts = base_parts[: max(0, len(base_parts) - (level - 1))]
    if module:
        base_parts.extend(str(module).split("."))
    return ".".join(part for part in base_parts if part)


def _validate_python_imports(target_root: Path, python_files: list[Path], relative_files: set[str]) -> list[str]:
    module_names = _collect_project_module_names(relative_files)
    requirement_names = _parse_requirement_names(target_root)
    warnings = []
    seen = set()
    top_level_local_modules = {name.split(".", 1)[0] for name in module_names}

    for py_file in python_files:
        relative_path = py_file.relative_to(target_root).as_posix()
        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=relative_path)
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            candidate = None
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module_name = str(alias.name or "").strip()
                    if not module_name:
                        continue
                    top_level = module_name.split(".", 1)[0].replace("-", "_").lower()
                    if top_level in top_level_local_modules or top_level in requirement_names or top_level in sys.stdlib_module_names:
                        continue
                    note = f"Import may be unresolved in {relative_path}: {module_name}"
                    if note not in seen:
                        seen.add(note)
                        warnings.append(note)
            elif isinstance(node, ast.ImportFrom):
                if node.level > 0:
                    resolved = _resolve_relative_import(relative_path, node.level, node.module)
                    if resolved and resolved not in module_names:
                        note = f"Relative import target is missing in {relative_path}: {'.' * node.level}{node.module or ''}"
                        if note not in seen:
                            seen.add(note)
                            warnings.append(note)
                else:
                    module_name = str(node.module or "").strip()
                    if not module_name:
                        continue
                    top_level = module_name.split(".", 1)[0].replace("-", "_").lower()
                    if top_level in top_level_local_modules or top_level in requirement_names or top_level in sys.stdlib_module_names:
                        continue
                    note = f"Import may be unresolved in {relative_path}: {module_name}"
                    if note not in seen:
                        seen.add(note)
                        warnings.append(note)
            if len(warnings) >= WORKSPACE_VALIDATION_MAX_NOTES:
                return warnings
    return warnings


def _validate_python_project_structure(target_root: Path, relative_files: set[str]) -> list[str]:
    warnings = []
    expected = ["app.py", "pyproject.toml", "requirements.txt", "README.md"]
    for name in expected:
        if name not in relative_files:
            warnings.append(f"Missing expected file: {name}")
    if not any(file_path.startswith("tests/") for file_path in relative_files):
        warnings.append("Missing tests directory or test files.")

    config_path = target_root / "config.py"
    if config_path.exists() and not config_path.read_text(encoding="utf-8").strip():
        warnings.append("config.py is empty.")

    for relative_path in sorted(relative_files):
        file_name = relative_path.rsplit("/", 1)[-1]
        lowered_name = file_name.lower()
        if lowered_name in {".env", ".env.example", "settings.json", "config.json", "config.yaml", "config.yml"}:
            file_path = target_root / relative_path
            if file_path.exists() and not file_path.read_text(encoding="utf-8").strip():
                warnings.append(f"Config-like file is empty: {relative_path}")

    src_entries = [path for path in relative_files if path.startswith("src/")]
    if src_entries:
        package_markers = [path for path in src_entries if path.endswith("/__init__.py")]
        if not package_markers:
            warnings.append("src/ layout detected but no package __init__.py file was found under src/.")

    if not _has_python_entry_point(target_root, relative_files):
        warnings.append("No obvious Python entry point found. Add app.py, main.py, __main__.py, or declare scripts in pyproject.toml.")
    return warnings[:WORKSPACE_VALIDATION_MAX_NOTES]


def validate_project_workspace(runtime_state: dict | None, path: str | None = None) -> dict:
    target_root, normalized_path = resolve_workspace_path(runtime_state, path, allow_empty=True)
    if not target_root.exists() or not target_root.is_dir():
        raise ValueError("Workspace directory not found.")

    issues = []
    warnings = []
    checked_files = []
    python_files = sorted(target_root.rglob("*.py"))
    for py_file in python_files:
        rel = py_file.relative_to(ensure_workspace_available(runtime_state)).as_posix()
        checked_files.append(rel)
        try:
            source = py_file.read_text(encoding="utf-8")
            ast.parse(source, filename=rel)
        except (SyntaxError, UnicodeDecodeError) as exc:
            issues.append(f"Python syntax error in {rel}: {exc}")

    relative_files = {
        path.relative_to(target_root).as_posix()
        for path in target_root.rglob("*")
        if path.is_file() and not _is_hidden_workspace_path(path.relative_to(ensure_workspace_available(runtime_state)).as_posix())
    }
    looks_like_python_project = bool(python_files or {"requirements.txt", "pyproject.toml"} & relative_files)
    if looks_like_python_project:
        warnings.extend(_validate_python_project_structure(target_root, relative_files))
        warnings.extend(_validate_python_imports(target_root, python_files, relative_files))

    deduped_warnings = []
    seen_warnings = set()
    for warning in warnings:
        if warning in seen_warnings:
            continue
        seen_warnings.add(warning)
        deduped_warnings.append(warning)
        if len(deduped_warnings) >= WORKSPACE_VALIDATION_MAX_NOTES:
            break

    return {
        "status": "ok" if not issues else "needs_attention",
        "action": "workspace_validated",
        "path": normalized_path,
        "checked_files": checked_files,
        "issues": issues,
        "warnings": deduped_warnings,
        "summary": {
            "checked_file_count": len(checked_files),
            "issue_count": len(issues),
            "warning_count": len(deduped_warnings),
            "looks_like_python_project": looks_like_python_project,
        },
    }