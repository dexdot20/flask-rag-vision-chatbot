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
CANVAS_ALLOWED_FORMATS = {"markdown"}


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
    created_at = str(value.get("created_at") or "").strip()[:80]
    updated_at = str(value.get("updated_at") or "").strip()[:80]

    cleaned = {
        "id": document_id,
        "title": title,
        "format": format_name,
        "content": content,
        "line_count": _line_count(content),
    }

    if language:
        cleaned["language"] = language

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
    for entry in raw_documents[:CANVAS_MAX_DOCUMENTS]:
        cleaned = normalize_canvas_document(entry)
        if not cleaned:
            continue
        if cleaned["id"] in seen_ids:
            continue
        normalized.append(cleaned)
        seen_ids.add(cleaned["id"])
    return normalized


def list_canvas_lines(content: str) -> list[str]:
    normalized = _normalize_line_endings(content)
    if normalized == "":
        return []
    return normalized.split("\n")


def join_canvas_lines(lines: Iterable[str]) -> str:
    return "\n".join(str(line) for line in lines)


def create_canvas_runtime_state(initial_documents: list[dict] | None = None) -> dict:
    documents = extract_canvas_documents({"canvas_documents": initial_documents or []})
    active_document_id = documents[-1]["id"] if documents else None
    return {
        "documents": documents,
        "active_document_id": active_document_id,
    }


def get_canvas_runtime_documents(runtime_state: dict | None) -> list[dict]:
    if not isinstance(runtime_state, dict):
        return []
    return extract_canvas_documents({"canvas_documents": runtime_state.get("documents") or []})


def _find_canvas_document(runtime_state: dict, document_id: str | None = None) -> tuple[int, dict]:
    documents = runtime_state.get("documents") if isinstance(runtime_state, dict) else None
    if not isinstance(documents, list) or not documents:
        raise ValueError("No canvas document is available yet.")

    target_id = str(document_id or runtime_state.get("active_document_id") or "").strip()
    if target_id:
        for index, document in enumerate(documents):
            if str(document.get("id") or "") == target_id:
                return index, document

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
    return normalized


def create_canvas_document(
    runtime_state: dict,
    title: str,
    content: str,
    format_name: str = "markdown",
    language_name: str | None = None,
) -> dict:
    normalized = normalize_canvas_document(
        {
            "id": uuid4().hex,
            "title": title or "Canvas",
            "format": format_name,
            "content": content,
            "language": language_name,
        }
    )
    return _store_canvas_document(runtime_state, normalized)


def rewrite_canvas_document(
    runtime_state: dict,
    content: str,
    document_id: str | None = None,
    title: str | None = None,
    language_name: str | None = None,
) -> dict:
    _, document = _find_canvas_document(runtime_state, document_id=document_id)
    next_document = dict(document)
    next_document["content"] = _clip_text(content, CANVAS_MAX_CONTENT_LENGTH)
    if title is not None:
        next_document["title"] = str(title or "Canvas").strip()[:CANVAS_MAX_TITLE_LENGTH] or "Canvas"
    if language_name is not None:
        next_document["language"] = language_name
    return _store_canvas_document(runtime_state, next_document)


def replace_canvas_lines(runtime_state: dict, start_line: int, end_line: int, lines: list[str], document_id: str | None = None) -> dict:
    _, document = _find_canvas_document(runtime_state, document_id=document_id)
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


def insert_canvas_lines(runtime_state: dict, after_line: int, lines: list[str], document_id: str | None = None) -> dict:
    _, document = _find_canvas_document(runtime_state, document_id=document_id)
    existing_lines = list_canvas_lines(document.get("content") or "")
    if after_line < 0 or after_line > len(existing_lines):
        raise ValueError("after_line must be between 0 and the current line count.")

    additions = [str(line) for line in (lines or [])]
    next_lines = [*existing_lines[:after_line], *additions, *existing_lines[after_line:]]
    next_document = dict(document)
    next_document["content"] = join_canvas_lines(next_lines)
    return _store_canvas_document(runtime_state, next_document)


def delete_canvas_lines(runtime_state: dict, start_line: int, end_line: int, document_id: str | None = None) -> dict:
    return replace_canvas_lines(runtime_state, start_line, end_line, [], document_id=document_id)


def delete_canvas_document(runtime_state: dict, document_id: str | None = None) -> dict:
    index, document = _find_canvas_document(runtime_state, document_id=document_id)
    documents = runtime_state.get("documents") if isinstance(runtime_state, dict) else None
    if not isinstance(documents, list):
        raise ValueError("No canvas document is available yet.")

    removed = documents.pop(index)
    runtime_state["active_document_id"] = documents[-1]["id"] if documents else None
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
        "title": normalized["title"],
        "format": normalized["format"],
        "line_count": normalized["line_count"],
        "content": preview,
        "content_truncated": len(normalized["content"]) > len(preview),
    }
    if normalized.get("language"):
        result["language"] = normalized["language"]
    return result


def find_latest_canvas_documents(messages: list[dict]) -> list[dict]:
    for message in reversed(messages or []):
        metadata = message.get("metadata") if isinstance(message, dict) else None
        if isinstance(metadata, dict) and metadata.get("canvas_cleared") is True:
            return []
        documents = extract_canvas_documents(metadata)
        if not documents:
            continue
        message_id = message.get("id") if isinstance(message.get("id"), int) else None
        results = []
        for document in documents:
            result = dict(document)
            if message_id is not None:
                result["source_message_id"] = message_id
            results.append(result)
        return results
    return []


def find_latest_canvas_document(messages: list[dict], document_id: str | None = None) -> dict | None:
    target_id = str(document_id or "").strip()
    for document in reversed(find_latest_canvas_documents(messages)):
        if not target_id or document.get("id") == target_id:
            return dict(document)
    return None


def build_markdown_download(document: dict) -> bytes:
    normalized = normalize_canvas_document(document)
    if not normalized:
        raise ValueError("Canvas document is invalid.")
    return _normalize_line_endings(normalized["content"]).encode("utf-8")


def build_html_download(document: dict) -> bytes:
    normalized = normalize_canvas_document(document)
    if not normalized:
        raise ValueError("Canvas document is invalid.")

    content = _normalize_line_endings(normalized["content"])
    if markdown_lib is not None:
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