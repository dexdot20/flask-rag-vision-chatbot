from __future__ import annotations

import copy
import json

from config import RAG_ENABLED

CANVAS_DOCUMENT_TOOL_NAMES = {
    "expand_canvas_document",
    "scroll_canvas_document",
    "rewrite_canvas_document",
    "replace_canvas_lines",
    "insert_canvas_lines",
    "delete_canvas_lines",
    "delete_canvas_document",
    "clear_canvas",
}

TOOL_SPECS = [
    {
        "name": "append_scratchpad",
        "description": (
            "Append one durable user-specific fact or preference to the persistent scratchpad. "
            "Use this only for long-lived, high-signal information that will likely change future answers or actions. "
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
            "purpose": "Saves one short durable user fact or preference into persistent scratchpad memory only when it is likely to matter later.",
            "inputs": {"note": "single short durable memory line"},
            "guidance": (
                "Use very sparingly. Save only durable user-specific facts, recurring constraints, or stable preferences that are likely to matter in future conversations. "
                "Do not save temporary requests, current-task details, large summaries, tool outputs, web/search results, speculative guesses, or sensitive data. "
                "If the information would not change future responses or behavior, do not store it. Prefer one short standalone line instead of paragraphs."
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
                "Keep the final text compact and only include durable, high-signal facts. Prefer a short bulleted list over paragraphs."
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
        "name": "expand_canvas_document",
        "description": (
            "Load the full context of a specific canvas document when the active excerpt and manifest summaries are not enough. "
            "Prefer targeting by document_path in project mode."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "document_id": {
                    "type": "string",
                    "description": "Optional target canvas document id."
                },
                "document_path": {
                    "type": "string",
                    "description": "Optional target project-relative path. Prefer this in project mode."
                }
            }
        },
        "prompt": {
            "purpose": "Expands one canvas document into full line-numbered context for focused reasoning or editing.",
            "inputs": {"document_id": "optional target id", "document_path": "optional target project-relative path"},
            "guidance": (
                "Use this when the manifest and active-document excerpt are insufficient and you need another canvas file in full detail. "
                "In project mode, prefer document_path over document_id so file targeting stays stable."
            ),
        },
    },
    {
        "name": "scroll_canvas_document",
        "description": (
            "Read a targeted line range from a specific canvas document when you need lines outside the visible excerpt. "
            "Prefer targeting by document_path in project mode."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "document_id": {
                    "type": "string",
                    "description": "Optional target canvas document id."
                },
                "document_path": {
                    "type": "string",
                    "description": "Optional target project-relative path. Prefer this in project mode."
                },
                "start_line": {
                    "type": "integer",
                    "description": "1-based starting line number to read."
                },
                "end_line": {
                    "type": "integer",
                    "description": "1-based ending line number to read."
                }
            },
            "required": ["start_line", "end_line"]
        },
        "prompt": {
            "purpose": "Reads a focused line window from one canvas document without loading the entire file into the prompt.",
            "inputs": {
                "document_id": "optional target id",
                "document_path": "optional target project-relative path",
                "start_line": "1-based starting line",
                "end_line": "1-based ending line"
            },
            "guidance": (
                "Use this when you know which region you need and the active excerpt is truncated. "
                "In project mode, prefer document_path over document_id so file targeting stays stable."
            ),
        },
    },
    {
        "name": "plan_project_workspace",
        "description": "Create or revise a structured project workflow plan before writing files to the workspace sandbox.",
        "parameters": {
            "type": "object",
            "properties": {
                "goal": {
                    "type": "string",
                    "description": "User-visible project goal or outcome."
                },
                "project_name": {
                    "type": "string",
                    "description": "Human-readable project name."
                },
                "target_type": {
                    "type": "string",
                    "enum": ["python-project"],
                    "description": "Project template family."
                },
                "files": {
                    "type": "array",
                    "description": "Optional planned files with path, role, and purpose.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "role": {"type": "string"},
                            "purpose": {"type": "string"}
                        },
                        "required": ["path"]
                    }
                },
                "dependencies": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional dependency shortlist to track in the workflow."
                }
            },
            "required": ["goal", "project_name"]
        },
        "prompt": {
            "purpose": "Creates a structured project workflow plan before scaffold and file generation.",
            "inputs": {"goal": "project objective", "project_name": "human-readable project name", "target_type": "optional project family", "files": "optional planned files", "dependencies": "optional dependency shortlist"},
            "guidance": "Use this before scaffold or file writes when the user is asking for a new project or multi-file output. Keep the plan concrete and immediately actionable.",
        },
    },
    {
        "name": "get_project_workflow_status",
        "description": "Return the current project workflow stage and tracked plan state for the active conversation workspace.",
        "parameters": {
            "type": "object",
            "properties": {}
        },
        "prompt": {
            "purpose": "Reads the current project workflow state.",
            "inputs": {},
        },
    },
    {
        "name": "create_directory",
        "description": "Create a directory inside the conversation workspace sandbox.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Workspace-relative directory path to create."
                }
            },
            "required": ["path"]
        },
        "prompt": {
            "purpose": "Creates one or more directories inside the workspace sandbox.",
            "inputs": {"path": "workspace-relative directory path"},
            "guidance": "Use only workspace-relative paths. The sandbox rejects paths outside the conversation workspace.",
        },
    },
    {
        "name": "create_file",
        "description": "Create a new UTF-8 text file inside the conversation workspace sandbox.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Workspace-relative file path to create."
                },
                "content": {
                    "type": "string",
                    "description": "Full text content for the new file."
                }
            },
            "required": ["path", "content"]
        },
        "prompt": {
            "purpose": "Creates a new file in the workspace sandbox.",
            "inputs": {"path": "workspace-relative file path", "content": "full file content"},
            "guidance": "Fails if the file already exists. Use update_file for existing files.",
        },
    },
    {
        "name": "update_file",
        "description": "Replace the full content of an existing UTF-8 text file inside the workspace sandbox.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Workspace-relative file path to update."
                },
                "content": {
                    "type": "string",
                    "description": "Full replacement text content."
                }
            },
            "required": ["path", "content"]
        },
        "prompt": {
            "purpose": "Updates an existing file in the workspace sandbox.",
            "inputs": {"path": "workspace-relative file path", "content": "full replacement content"},
            "guidance": "Use this only for files that already exist.",
        },
    },
    {
        "name": "read_file",
        "description": "Read a file from the conversation workspace sandbox with optional line limits.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Workspace-relative file path to read."
                },
                "start_line": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Optional first line to include. Defaults to 1."
                },
                "end_line": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Optional last line to include."
                }
            },
            "required": ["path"]
        },
        "prompt": {
            "purpose": "Reads a file from the workspace sandbox.",
            "inputs": {"path": "workspace-relative file path", "start_line": "optional first line", "end_line": "optional last line"},
        },
    },
    {
        "name": "list_dir",
        "description": "List files and directories inside the conversation workspace sandbox.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Optional workspace-relative directory path. Defaults to the workspace root."
                }
            }
        },
        "prompt": {
            "purpose": "Lists workspace files and directories.",
            "inputs": {"path": "optional workspace-relative directory path"},
        },
    },
    {
        "name": "search_files",
        "description": "Search workspace file paths and optionally file contents inside the conversation sandbox.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Case-insensitive search text."
                },
                "path_prefix": {
                    "type": "string",
                    "description": "Optional workspace-relative directory to search under."
                },
                "search_content": {
                    "type": "boolean",
                    "description": "Whether to search inside file contents in addition to file paths."
                }
            },
            "required": ["query"]
        },
        "prompt": {
            "purpose": "Searches file paths or contents inside the workspace sandbox.",
            "inputs": {"query": "case-insensitive search text", "path_prefix": "optional subdirectory", "search_content": "optional boolean"},
        },
    },
    {
        "name": "create_project_scaffold",
        "description": "Create a starter project skeleton inside the conversation workspace sandbox.",
        "parameters": {
            "type": "object",
            "properties": {
                "project_name": {
                    "type": "string",
                    "description": "Human-readable project name."
                },
                "target_type": {
                    "type": "string",
                    "enum": ["python-project"],
                    "description": "Starter scaffold type."
                },
                "confirm": {
                    "type": "boolean",
                    "description": "Set true only after the user approves overwriting an existing non-empty project root."
                }
            },
            "required": ["project_name"]
        },
        "prompt": {
            "purpose": "Creates a starter project structure in the workspace sandbox.",
            "inputs": {"project_name": "project name", "target_type": "optional scaffold type", "confirm": "optional overwrite confirmation"},
            "guidance": "If the tool reports needs_confirmation, show the preview to the user and ask before re-running with confirm=true.",
        },
    },
    {
        "name": "write_project_tree",
        "description": "Create or overwrite many directories and files inside the workspace sandbox in one operation.",
        "parameters": {
            "type": "object",
            "properties": {
                "directories": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Workspace-relative directories to create."
                },
                "files": {
                    "type": "array",
                    "description": "Files to write with path and content.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "content": {"type": "string"}
                        },
                        "required": ["path", "content"]
                    }
                },
                "confirm": {
                    "type": "boolean",
                    "description": "Set true only after the user approves overwriting existing files."
                }
            }
        },
        "prompt": {
            "purpose": "Writes a batch of project directories and files into the workspace sandbox.",
            "inputs": {"directories": "optional directories", "files": "optional file entries", "confirm": "optional overwrite confirmation"},
            "guidance": "If the tool reports needs_confirmation, review the returned diffs with the user and do not re-run with confirm=true until the overwrite set is approved.",
        },
    },
    {
        "name": "bulk_update_workspace_files",
        "description": "Overwrite many workspace files in one operation after explicit confirmation when required.",
        "parameters": {
            "type": "object",
            "properties": {
                "files": {
                    "type": "array",
                    "description": "Files to overwrite with path and content.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "content": {"type": "string"}
                        },
                        "required": ["path", "content"]
                    }
                },
                "confirm": {
                    "type": "boolean",
                    "description": "Set true only after the user approves overwriting the listed files."
                }
            },
            "required": ["files"]
        },
        "prompt": {
            "purpose": "Applies a batch file update inside the workspace sandbox.",
            "inputs": {"files": "file entries with path and content", "confirm": "optional overwrite confirmation"},
            "guidance": "If the tool reports needs_confirmation, review the returned diffs and wait for user approval before applying overwrites.",
        },
    },
    {
        "name": "preview_workspace_changes",
        "description": "Preview unified diffs for one or more workspace file writes without applying them.",
        "parameters": {
            "type": "object",
            "properties": {
                "files": {
                    "type": "array",
                    "description": "Files to preview with path and proposed content.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "content": {"type": "string"}
                        },
                        "required": ["path", "content"]
                    }
                }
            },
            "required": ["files"]
        },
        "prompt": {
            "purpose": "Shows file-by-file unified diffs before a workspace change is applied.",
            "inputs": {"files": "file entries with path and proposed content"},
            "guidance": "Use this before larger rewrites or when you need an explicit diff review without modifying the workspace.",
        },
    },
    {
        "name": "get_workspace_file_history",
        "description": "Inspect stored undo and redo history for one workspace file.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Workspace-relative file path."
                }
            },
            "required": ["path"]
        },
        "prompt": {
            "purpose": "Shows available revision history for a workspace file.",
            "inputs": {"path": "workspace-relative file path"},
        },
    },
    {
        "name": "undo_workspace_file_change",
        "description": "Undo the latest recorded change for one workspace file.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Workspace-relative file path."
                }
            },
            "required": ["path"]
        },
        "prompt": {
            "purpose": "Restores the previous file content from workspace history.",
            "inputs": {"path": "workspace-relative file path"},
        },
    },
    {
        "name": "redo_workspace_file_change",
        "description": "Re-apply the latest undone change for one workspace file.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Workspace-relative file path."
                }
            },
            "required": ["path"]
        },
        "prompt": {
            "purpose": "Reapplies the next available redo entry for a workspace file.",
            "inputs": {"path": "workspace-relative file path"},
        },
    },
    {
        "name": "validate_project_workspace",
        "description": "Run lightweight validation checks against the workspace sandbox or one project subdirectory.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Optional workspace-relative directory path. Defaults to the workspace root."
                }
            }
        },
        "prompt": {
            "purpose": "Validates project files in the workspace sandbox.",
            "inputs": {"path": "optional workspace-relative directory path"},
        },
    },
    {
        "name": "create_canvas_document",
        "description": (
            "Create a canvas document for the current conversation. "
            "Use this for a single editable draft or as one file inside a multi-file canvas project workspace."
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
                    "enum": ["markdown", "code"],
                    "description": "Canvas document format. Use code for a raw code document without markdown wrappers."
                },
                "language": {
                    "type": "string",
                    "description": "Optional dominant code language for the document, such as python, javascript, or sql."
                },
                "path": {
                    "type": "string",
                    "description": "Optional project-relative path such as src/app.py, README.md, or tests/test_app.py."
                },
                "role": {
                    "type": "string",
                    "enum": ["source", "config", "dependency", "docs", "test", "script", "note"],
                    "description": "Optional semantic role for the document inside a project workspace."
                },
                "summary": {
                    "type": "string",
                    "description": "Optional short semantic summary of the document's responsibility."
                },
                "imports": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional imported modules, files, or config keys referenced by this document."
                },
                "exports": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional exported entry points, functions, classes, or files produced by this document."
                },
                "symbols": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional important symbols defined in this document."
                },
                "dependencies": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional package or file dependencies associated with this document."
                },
                "project_id": {
                    "type": "string",
                    "description": "Optional stable project identifier grouping related canvas documents."
                },
                "workspace_id": {
                    "type": "string",
                    "description": "Optional stable workspace identifier grouping related canvas documents."
                }
            },
            "required": ["title", "content"]
        },
        "prompt": {
            "purpose": "Creates an editable canvas document attached to the conversation, optionally as part of a project workspace.",
            "inputs": {
                "title": "document title",
                "content": "full markdown body",
                "format": "currently markdown",
                "language": "optional dominant code language",
                "path": "optional project-relative file path",
                "role": "optional semantic document role",
                "summary": "optional short responsibility summary",
                "imports": "optional referenced modules, files, or config keys",
                "exports": "optional exported entry points or files",
                "symbols": "optional key symbols defined in the document",
                "dependencies": "optional package or file dependencies",
                "project_id": "optional project identifier",
                "workspace_id": "optional workspace identifier"
            },
            "guidance": (
                "Use this when the user would benefit from a persistent artifact that can be revised. "
                "Prefer creating the document before line-level edits. In project mode, set path and role so the workspace manifest stays coherent."
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
                "format": {
                    "type": "string",
                    "enum": ["markdown", "code"],
                    "description": "Optional replacement format for the document."
                },
                "language": {
                    "type": "string",
                    "description": "Optional dominant code language for the updated document."
                },
                "path": {
                    "type": "string",
                    "description": "Optional replacement project-relative path for the document."
                },
                "role": {
                    "type": "string",
                    "enum": ["source", "config", "dependency", "docs", "test", "script", "note"],
                    "description": "Optional replacement semantic role for the document."
                },
                "summary": {
                    "type": "string",
                    "description": "Optional replacement short semantic summary."
                },
                "imports": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional replacement import list."
                },
                "exports": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional replacement export list."
                },
                "symbols": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional replacement symbol list."
                },
                "dependencies": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional replacement dependency list."
                },
                "project_id": {
                    "type": "string",
                    "description": "Optional replacement project identifier."
                },
                "workspace_id": {
                    "type": "string",
                    "description": "Optional replacement workspace identifier."
                },
                "document_id": {
                    "type": "string",
                    "description": "Optional target canvas document id. Defaults to the active document."
                },
                "document_path": {
                    "type": "string",
                    "description": "Optional target project-relative path. Prefer this over document_id in project mode."
                }
            },
            "required": ["content"]
        },
        "prompt": {
            "purpose": "Replaces the full content of an existing canvas document.",
            "inputs": {"content": "full document body", "title": "optional title", "format": "optional markdown or code", "language": "optional dominant code language", "path": "optional project-relative file path", "role": "optional semantic role", "summary": "optional short responsibility summary", "imports": "optional import list", "exports": "optional export list", "symbols": "optional symbol list", "dependencies": "optional dependency list", "project_id": "optional project identifier", "workspace_id": "optional workspace identifier", "document_id": "optional target id", "document_path": "optional target project-relative path"},
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
                },
                "document_path": {
                    "type": "string",
                    "description": "Optional target project-relative path. Prefer this over document_id in project mode."
                }
            },
            "required": ["start_line", "end_line", "lines"]
        },
        "prompt": {
            "purpose": "Replaces specific lines in the canvas document.",
            "inputs": {"start_line": "first line", "end_line": "last line", "lines": "replacement lines", "document_id": "optional target id", "document_path": "optional target project-relative path"},
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
                },
                "document_path": {
                    "type": "string",
                    "description": "Optional target project-relative path. Prefer this over document_id in project mode."
                }
            },
            "required": ["after_line", "lines"]
        },
        "prompt": {
            "purpose": "Inserts lines into the canvas document.",
            "inputs": {"after_line": "insertion point", "lines": "new lines", "document_id": "optional target id", "document_path": "optional target project-relative path"},
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
                },
                "document_path": {
                    "type": "string",
                    "description": "Optional target project-relative path. Prefer this over document_id in project mode."
                }
            },
            "required": ["start_line", "end_line"]
        },
        "prompt": {
            "purpose": "Deletes specific lines from the canvas document.",
            "inputs": {"start_line": "first line", "end_line": "last line", "document_id": "optional target id", "document_path": "optional target project-relative path"},
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
                },
                "document_path": {
                    "type": "string",
                    "description": "Optional target project-relative path. Prefer this over document_id in project mode."
                }
            }
        },
        "prompt": {
            "purpose": "Deletes one canvas document from the current conversation.",
            "inputs": {"document_id": "optional target id", "document_path": "optional target project-relative path"},
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


def resolve_runtime_tool_names(active_tool_names: list[str], canvas_documents: list[dict] | None = None) -> list[str]:
    names = list(active_tool_names or [])
    if canvas_documents:
        return names
    return [name for name in names if name not in CANVAS_DOCUMENT_TOOL_NAMES]


def get_openai_tool_specs(active_tool_names: list[str], canvas_documents: list[dict] | None = None) -> list[dict]:
    specs = []
    runtime_tool_names = resolve_runtime_tool_names(active_tool_names, canvas_documents=canvas_documents)
    for tool in get_enabled_tool_specs(runtime_tool_names):
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


def get_prompt_tool_context(active_tool_names: list[str], canvas_documents: list[dict] | None = None) -> list[dict] | None:
    tools = []
    runtime_tool_names = resolve_runtime_tool_names(active_tool_names, canvas_documents=canvas_documents)
    for tool in get_enabled_tool_specs(runtime_tool_names):
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
