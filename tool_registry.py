from __future__ import annotations

import copy
import json

from config import RAG_ENABLED

TOOL_SPECS = [
    {
        "name": "append_scratchpad",
        "description": (
            "Append one durable user-specific fact or preference to the persistent scratchpad. "
            "Use this only for long-lived information that will likely help in future turns. "
            "Do not store temporary task details, sensitive secrets, one-off requests, or speculative inferences."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "note": {
                    "type": "string",
                    "description": "One short durable note to append to the scratchpad.",
                }
            },
            "required": ["note"],
        },
        "prompt": {
            "purpose": "Saves one short durable user fact or preference into persistent scratchpad memory.",
            "inputs": {"note": "single short durable memory line"},
            "guidance": (
                "Use very sparingly. Save only durable user-specific facts, recurring constraints, or stable preferences that will likely matter later. "
                "Do not save temporary requests, current-task details, large summaries, tool outputs, speculative guesses, or sensitive data. "
                "Prefer one short standalone line instead of paragraphs."
            ),
        },
    },
    {
        "name": "replace_scratchpad",
        "description": (
            "Completely replace the persistent scratchpad content. "
            "Use this to rewrite, reorganize, or remove outdated durable user-specific facts. "
            "Do not store temporary task details or speculative inferences."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "new_content": {
                    "type": "string",
                    "description": "The new content that will fully replace the existing scratchpad.",
                }
            },
            "required": ["new_content"],
        },
        "prompt": {
            "purpose": "Completely rewrites the persistent scratchpad memory.",
            "inputs": {"new_content": "the new complete scratchpad content"},
            "guidance": (
                "Use carefully to prune or reorganize existing facts. Ensure you do not accidentally delete important existing preferences. "
                "Keep the final text compact, usually a bulleted list of facts."
            ),
        },
    },
    {
        "name": "ask_clarifying_question",
        "description": (
            "Ask the user one or more structured clarification questions and stop answering until they reply. "
            "Use this when key requirements are missing, ambiguous, or mutually dependent and you should not guess. "
            "If the user explicitly asks you to ask questions first before answering, use this tool instead of asking inline."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "intro": {
                    "type": "string",
                    "description": "Short lead-in shown before the questions."
                },
                "questions": {
                    "type": "array",
                    "description": "List of 1-5 clarification questions.",
                    "minItems": 1,
                    "maxItems": 5,
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {
                                "type": "string",
                                "description": "Stable identifier for mapping the answer later."
                            },
                            "label": {
                                "type": "string",
                                "description": "The question shown to the user."
                            },
                            "input_type": {
                                "type": "string",
                                "enum": ["text", "single_select", "multi_select"],
                                "description": "How the user should answer this question."
                            },
                            "required": {
                                "type": "boolean",
                                "description": "Whether the user must answer this question."
                            },
                            "placeholder": {
                                "type": "string",
                                "description": "Optional placeholder for free-text answers."
                            },
                            "options": {
                                "type": "array",
                                "description": "Selectable options for single_select or multi_select questions.",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "label": {"type": "string"},
                                        "value": {"type": "string"},
                                        "description": {"type": "string"}
                                    },
                                    "required": ["label", "value"]
                                }
                            },
                            "allow_free_text": {
                                "type": "boolean",
                                "description": "Whether the user may add custom text alongside the predefined options."
                            }
                        },
                        "required": ["id", "label", "input_type"]
                    }
                },
                "submit_label": {
                    "type": "string",
                    "description": "Optional button label shown in the UI."
                }
            },
            "required": ["questions"],
        },
        "prompt": {
            "purpose": "Collects missing user requirements before continuing the answer.",
            "inputs": {
                "intro": "optional short lead-in",
                "questions": "1-5 structured questions",
                "submit_label": "optional button label"
            },
            "guidance": (
                "Use this instead of guessing when important requirements are missing. "
                "Ask only the smallest set of questions needed to continue. "
                "When the user asks you to ask questions first, this is the required tool. "
                "When you call this tool, it must be the only tool call in that assistant message and you must wait for the user's reply before answering. "
                "Each questions item must be an object with id, label, and input_type; example: {\"id\":\"scope\",\"label\":\"Which scope?\",\"input_type\":\"text\"}."
            ),
        },
    },
    {
        "name": "image_explain",
        "description": (
            "Answer a follow-up question about a previously uploaded image saved in the current conversation. "
            "Use this when the user refers back to an earlier image or screenshot and the stored visual context may matter."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "image_id": {
                    "type": "string",
                    "description": "The globally unique stored image_id for the referenced uploaded image.",
                },
                "conversation_id": {
                    "type": "integer",
                    "description": "The current conversation id used to verify that the image belongs to this chat.",
                },
                "question": {
                    "type": "string",
                    "description": "A focused follow-up question about the image. Write this question in English.",
                },
            },
            "required": ["image_id", "conversation_id", "question"],
        },
        "prompt": {
            "purpose": "Asks the vision model a new question about a stored image from this conversation.",
            "inputs": {
                "image_id": "stored image id",
                "conversation_id": "current conversation id",
                "question": "follow-up question written in English",
            },
            "guidance": (
                "Use this when the user asks about a previously uploaded image instead of relying only on the cached summary. "
                "Always send the question in English. The tool response will be in English. "
                "If the referenced image is ambiguous, ask the user to clarify which image they mean before calling the tool."
            ),
        },
    },
    {
        "name": "search_knowledge_base",
        "description": (
            "Search the internal knowledge base indexed with RAG. "
            "Use this when the answer may exist in synced conversation history or stored tool outputs. "
            "Optionally filter by category. Avoid repeating semantically overlapping searches when one good result set already answers the question."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Semantic search query for the knowledge base.",
                },
                "category": {
                    "type": "string",
                    "description": "Optional category filter: conversation or tool_result.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Maximum number of chunks to retrieve (1-12).",
                    "minimum": 1,
                    "maximum": 12,
                },
            },
            "required": ["query"],
        },
        "prompt": {
            "purpose": "Searches the internal RAG knowledge base built from files, URLs, notes, and conversations.",
            "inputs": {"query": "semantic search query", "category": "optional category", "top_k": "1-12 results"},
            "guidance": "Use at most a few focused searches and synthesize from returned chunks instead of retrying near-duplicate queries.",
        },
    },
    {
        "name": "search_tool_memory",
        "description": (
            "Search past web tool results stored from previous conversations. "
            "Use this before making a new web request when you suspect the topic was already researched. "
            "This searches remembered results from fetch_url, search_web, and news tools."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Semantic search query for past web tool results.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Maximum number of remembered results to retrieve (1-10).",
                    "minimum": 1,
                    "maximum": 10,
                },
            },
            "required": ["query"],
        },
        "prompt": {
            "purpose": "Searches memory of past web searches, URL fetches, and news lookups.",
            "inputs": {"query": "semantic search query", "top_k": "1-10 results"},
            "guidance": (
                "Use before making a new web request if similar research may already exist. "
                "If high-similarity results already answer the question, reuse them instead of repeating the search."
            ),
        },
    },
    {
        "name": "search_web",
        "description": (
            "Search the web using DuckDuckGo. Use this to find current information, facts, prices, news, or any topic requiring up-to-date data. "
            "Provide one or more search queries."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "queries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of search queries to run (1–5 queries).",
                    "minItems": 1,
                    "maxItems": 5,
                }
            },
            "required": ["queries"],
        },
        "prompt": {
            "purpose": "Runs a general web search and returns recent results.",
            "inputs": {"queries": "1-5 search queries"},
        },
    },
    {
        "name": "fetch_url",
        "description": (
            "Fetch and read the content of a specific web page. Returns cleaned text and metadata. "
            "Use after search_web to read the full content of a relevant page."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Full URL of the page (must start with http:// or https://).",
                }
            },
            "required": ["url"],
        },
        "prompt": {
            "purpose": "Reads the cleaned content of a specific URL.",
            "inputs": {"url": "full http/https URL"},
        },
    },
    {
        "name": "search_news_ddgs",
        "description": (
            "Search recent news articles using DuckDuckGo News. Returns title, link, publication time and source for each article. "
            "Use this for general news, trending topics or when a broad news index is appropriate. "
            "Optionally filter by time range and language. If you need the full article text, follow up with fetch_url on the returned links."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "queries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of news search queries (1–5).",
                    "minItems": 1,
                    "maxItems": 5,
                },
                "lang": {
                    "type": "string",
                    "enum": ["tr", "en"],
                    "description": "Search language/region. 'tr' for Turkish results, 'en' for English.",
                },
                "when": {
                    "type": "string",
                    "enum": ["d", "w", "m", "y"],
                    "description": "Optional time filter: 'd'=last day, 'w'=last week, 'm'=last month, 'y'=last year.",
                },
            },
            "required": ["queries"],
        },
        "prompt": {
            "purpose": "Searches news headlines/links/dates/sources with DuckDuckGo News.",
            "inputs": {"queries": "1-5 news queries", "lang": "tr|en", "when": "d|w|m|y"},
        },
    },
    {
        "name": "search_news_google",
        "description": (
            "Search Google News via RSS feed. Returns title, link, publication time and source for each article. "
            "Use this when Google News coverage is preferred (e.g. Turkish financial news, local outlets, or when DuckDuckGo News yields poor results). "
            "Optionally filter by time range and language. If you need the full article text, follow up with fetch_url on the returned links."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "queries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of news search queries (1–5).",
                    "minItems": 1,
                    "maxItems": 5,
                },
                "lang": {
                    "type": "string",
                    "enum": ["tr", "en"],
                    "description": "Search language/region. 'tr' for Turkish results, 'en' for English.",
                },
                "when": {
                    "type": "string",
                    "enum": ["d", "w", "m", "y"],
                    "description": "Optional time filter: 'd'=last day, 'w'=last week, 'm'=last month, 'y'=last year.",
                },
            },
            "required": ["queries"],
        },
        "prompt": {
            "purpose": "Searches news headlines/links/dates/sources with Google News RSS.",
            "inputs": {"queries": "1-5 news queries", "lang": "tr|en", "when": "d|w|m|y"},
        },
    },
    {
        "name": "create_canvas_document",
        "description": (
            "Create or replace the active canvas document for the current conversation. "
            "Use this when the user asks for a structured draft, outline, plan, article, note, or any long-form artifact that should stay editable."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Document title shown in the canvas panel."
                },
                "content": {
                    "type": "string",
                    "description": "Full markdown content for the new canvas document."
                },
                "format": {
                    "type": "string",
                    "enum": ["markdown"],
                    "description": "Canvas document format. Only markdown is currently supported."
                }
            },
            "required": ["title", "content"]
        },
        "prompt": {
            "purpose": "Creates an editable canvas document attached to the conversation.",
            "inputs": {
                "title": "document title",
                "content": "full markdown body",
                "format": "currently markdown"
            },
            "guidance": (
                "Use this when the user would benefit from a persistent artifact that can be revised. "
                "Prefer creating the document before line-level edits."
            ),
        },
    },
    {
        "name": "rewrite_canvas_document",
        "description": "Rewrite the full active canvas document while keeping the same document id.",
        "parameters": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The full replacement markdown content."
                },
                "title": {
                    "type": "string",
                    "description": "Optional replacement title."
                },
                "document_id": {
                    "type": "string",
                    "description": "Optional target canvas document id. Defaults to the active document."
                }
            },
            "required": ["content"]
        },
        "prompt": {
            "purpose": "Replaces the full content of an existing canvas document.",
            "inputs": {"content": "full markdown body", "title": "optional title", "document_id": "optional target id"},
        },
    },
    {
        "name": "replace_canvas_lines",
        "description": "Replace a 1-based inclusive line range inside the active canvas document.",
        "parameters": {
            "type": "object",
            "properties": {
                "start_line": {"type": "integer", "minimum": 1, "description": "1-based first line to replace."},
                "end_line": {"type": "integer", "minimum": 1, "description": "1-based last line to replace."},
                "lines": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Replacement lines without trailing newline characters."
                },
                "document_id": {
                    "type": "string",
                    "description": "Optional target canvas document id. Defaults to the active document."
                }
            },
            "required": ["start_line", "end_line", "lines"]
        },
        "prompt": {
            "purpose": "Replaces specific lines in the canvas document.",
            "inputs": {"start_line": "first line", "end_line": "last line", "lines": "replacement lines", "document_id": "optional target id"},
        },
    },
    {
        "name": "insert_canvas_lines",
        "description": "Insert one or more lines into the active canvas document after a given line number.",
        "parameters": {
            "type": "object",
            "properties": {
                "after_line": {"type": "integer", "minimum": 0, "description": "Insert after this 1-based line. Use 0 to insert at the top."},
                "lines": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "New lines without trailing newline characters."
                },
                "document_id": {
                    "type": "string",
                    "description": "Optional target canvas document id. Defaults to the active document."
                }
            },
            "required": ["after_line", "lines"]
        },
        "prompt": {
            "purpose": "Inserts lines into the canvas document.",
            "inputs": {"after_line": "insertion point", "lines": "new lines", "document_id": "optional target id"},
        },
    },
    {
        "name": "delete_canvas_lines",
        "description": "Delete a 1-based inclusive line range from the active canvas document.",
        "parameters": {
            "type": "object",
            "properties": {
                "start_line": {"type": "integer", "minimum": 1, "description": "1-based first line to delete."},
                "end_line": {"type": "integer", "minimum": 1, "description": "1-based last line to delete."},
                "document_id": {
                    "type": "string",
                    "description": "Optional target canvas document id. Defaults to the active document."
                }
            },
            "required": ["start_line", "end_line"]
        },
        "prompt": {
            "purpose": "Deletes specific lines from the canvas document.",
            "inputs": {"start_line": "first line", "end_line": "last line", "document_id": "optional target id"},
        },
    },
    {
        "name": "delete_canvas_document",
        "description": "Delete a canvas document. Defaults to the active document when document_id is omitted.",
        "parameters": {
            "type": "object",
            "properties": {
                "document_id": {
                    "type": "string",
                    "description": "Optional target canvas document id. Defaults to the active document."
                }
            }
        },
        "prompt": {
            "purpose": "Deletes one canvas document from the current conversation.",
            "inputs": {"document_id": "optional target id"},
        },
    },
    {
        "name": "clear_canvas",
        "description": "Delete all canvas documents for the current conversation.",
        "parameters": {
            "type": "object",
            "properties": {}
        },
        "prompt": {
            "purpose": "Clears all canvas documents from the current conversation.",
            "inputs": {},
        },
    },
]

TOOL_SPEC_BY_NAME = {tool["name"]: tool for tool in TOOL_SPECS}


def get_enabled_tool_specs(active_tool_names: list[str]) -> list[dict]:
    active_set = set(active_tool_names or [])
    specs = [tool for tool in TOOL_SPECS if tool["name"] in active_set]
    if not RAG_ENABLED:
        specs = [tool for tool in specs if tool["name"] != "search_knowledge_base"]
    return specs


def get_openai_tool_specs(active_tool_names: list[str]) -> list[dict]:
    specs = []
    for tool in get_enabled_tool_specs(active_tool_names):
        parameters = copy.deepcopy(tool.get("parameters") or {})
        if parameters.get("type") == "object":
            parameters.setdefault("additionalProperties", False)
        specs.append(
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description") or "",
                    "parameters": parameters,
                },
            }
        )
    return specs



def _compact_arg_type(arg_props: dict) -> str:
    arg_type = arg_props.get("type", "string")
    if arg_type == "array":
        item_type = (arg_props.get("items") or {}).get("type", "")
        if item_type:
            return f"array[{item_type}]"
    return arg_type


def get_prompt_tool_context(active_tool_names: list[str]) -> list[dict] | None:
    tools = []
    for tool in get_enabled_tool_specs(active_tool_names):
        parameters = tool.get("parameters") if isinstance(tool.get("parameters"), dict) else {}
        properties = parameters.get("properties") if isinstance(parameters.get("properties"), dict) else {}
        required = parameters.get("required") if isinstance(parameters.get("required"), list) else []
        prompt = tool.get("prompt") if isinstance(tool.get("prompt"), dict) else {}
        use_for = str(prompt.get("purpose") or "").strip()
        if not use_for:
            use_for = str(tool.get("description") or "").strip().split(". ")[0].strip()

        entry = {"name": tool["name"]}
        if use_for:
            entry["use_for"] = use_for
        if properties:
            args = {}
            for arg_name, arg_props in properties.items():
                parts = [_compact_arg_type(arg_props)]
                if arg_name in required:
                    parts.append("required")
                enum_values = arg_props.get("enum")
                if enum_values:
                    parts.append("one of " + json.dumps(enum_values, ensure_ascii=False))
                desc = str(arg_props.get("description") or "").strip()
                compact = ", ".join(parts)
                if desc:
                    compact += f" — {desc}"
                args[arg_name] = compact
            entry["arguments"] = args
        guidance = str(prompt.get("guidance") or "").strip()
        if guidance:
            entry["guidance"] = guidance
        tools.append(entry)
    return tools or None
