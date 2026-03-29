# Flask ChatBot: DeepSeek + Tools + RAG + Vision + Canvas + Memory

This is a single-page Flask chat application built around DeepSeek models, multi-step tool use, local RAG, local vision OCR, conversation summarization, pruning, persistent memory, editable canvas documents, and a per-conversation workspace sandbox.

It is not a minimal prompt/response demo. The app keeps conversation history in SQLite, restores assistant metadata when a conversation is reopened, supports editing earlier user messages, streams tool progress and reasoning, can enrich a user turn with local OCR or extracted document text before the model sees it, and can compact older content with summaries and pruning.

## Contents

- [What the app does](#what-the-app-does)
- [Architecture overview](#architecture-overview)
- [Project structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Using the app](#using-the-app)
- [Available tools](#available-tools)
- [HTTP endpoints](#http-endpoints)
- [Data storage](#data-storage)
- [Development](#development)
- [Security and operational notes](#security-and-operational-notes)
- [Troubleshooting](#troubleshooting)
- [FAQ](#faq)
- [License](#license)

## What the app does

### Chat and conversation workflow

- Create, open, rename, and delete conversations
- Stream assistant output to the browser as NDJSON events
- Persist messages, usage metadata, tool traces, reasoning content, and canvas state in SQLite
- Cancel an active response mid-stream
- Clear the current chat view without deleting stored conversations
- Automatically generate a short title after the first exchange or on demand
- Edit a previous user message, delete later turns, and regenerate from that branch
- Restore assistant metadata, reasoning, tool results, and canvas state when reopening a conversation
- Show a separate Fix action that rewrites the current draft before sending
- Manually summarize a conversation, undo an inserted summary, and prune older visible messages

### Model and agent behavior

- Supports these models:
  - `deepseek-chat`
  - `deepseek-reasoner`
- Uses the DeepSeek API through the OpenAI Python client
- Validates tool names and tool argument schemas before execution
- Supports native function calls from the model
- Supports model-emitted tool JSON fallback handling
- Limits tool rounds with configurable `max_steps` from 1 to 10
- Forces a final-answer phase when the tool budget is exhausted
- Tracks estimated prompt composition locally across runtime instructions, tool specs, canvas context, scratchpad, tool trace, tool memory, RAG context, message history, tool calls, tool results, and provider overhead
- Estimates per-turn and session cost with the app's configured DeepSeek pricing logic
- Writes rotating agent trace logs to `logs/agent-trace.log` by default

### Attachments

- Image uploads are analyzed locally with Qwen2.5-VL when vision is enabled
- Document uploads are extracted locally and injected into the conversation context
- Supported image formats:
  - PNG
  - JPEG
  - WEBP
- Supported document formats:
  - DOCX
  - PDF
  - TXT
  - CSV
  - Markdown
  - Common code and config files such as Python, JavaScript, TypeScript, JSON, HTML, CSS, YAML, SQL, and shell scripts

### Memory and retrieval

- Persistent scratchpad for durable user-specific facts and preferences
- Persistent user profile memory extracted from structured conversation summaries
- Tool memory for successful web/news/URL results from earlier sessions
- RAG knowledge base built from stored conversations, successful text-like tool results, remembered web results, and uploaded documents
- Optional auto-injection of retrieved RAG context into each turn
- Optional auto-injection of remembered tool results into each turn
- RAG source pools can be scoped in Settings to conversations, tool results, tool memory, and uploaded documents
- Structured clarification tool for cases where the request is underspecified

### Canvas documents

- The model can create and edit Markdown canvas documents attached to the current conversation
- The model can also create code-format canvas documents with language metadata, path/role metadata, and project summaries when working in project mode
- The UI can display multiple canvas documents, search within them, filter them, and export them
- Canvas documents can be edited line-by-line by the model
- Targeted reads use `scroll_canvas_document`, while wider reads use `expand_canvas_document`
- Project-mode canvas sessions include a file tree with active-file highlighting
- Canvas documents can be downloaded as Markdown, HTML, or PDF

Manual smoke test checklist for the Canvas UI is available in [docs/canvas-ui-smoke-test.md](docs/canvas-ui-smoke-test.md)

### Observability

- Usage panel separates provider-reported usage from local input-source estimates
- Panel shows provider session totals, latest-turn totals, peak prompt size, configured per-call caps, and per-turn cost alongside non-zero input-source chips
- Stored assistant metadata is used to rebuild the panel after reload
- Summary inspector surfaces trigger thresholds, token gaps, source-message counts, and recent summary status

## Architecture overview

1. The browser sends JSON or multipart form data to `/chat`.
2. The backend loads persisted settings from SQLite.
3. If an image is attached, local vision analysis runs first.
4. If a document is attached, its text is extracted and added to the turn context.
5. If RAG auto-injection is enabled, the user message is searched against the knowledge base.
6. If tool-memory auto-injection is enabled, the same query searches remembered web results.
7. A runtime system message is built with the current time, preferences, scratchpad, user profile facts, tool guidance, and any retrieved context.
8. The agent streams model output.
9. Tool calls are validated, executed, cached, and appended to the transcript.
10. Tool progress, reasoning deltas, answer deltas, usage, and message IDs are streamed back as NDJSON.
11. The final assistant message is stored with metadata such as reasoning, usage, tool trace, canvas state, and stored tool results.
12. After a turn finishes, the app may summarize older context, prune older visible messages, and sync conversations or tool results into the RAG store.

## Project structure

```text
.
├── app.py                  # Flask app factory and entrypoint
├── agent.py                # Streaming agent loop, tool execution, usage tracking, trace logging
├── canvas_service.py       # Canvas document storage and line-level editing
├── config.py               # Environment variables, defaults, feature flags, runtime limits
├── conversation_export.py  # Conversation and canvas export utilities
├── db.py                   # SQLite schema, settings, assets, cache, metadata helpers
├── doc_service.py          # Document upload and text extraction
├── messages.py             # Runtime prompt construction and API message preparation
├── prune_service.py        # Message pruning helpers
├── rag_service.py          # RAG sync/search orchestration and tool-memory storage
├── token_utils.py          # Token counting and prompt-source estimation
├── tool_registry.py        # Tool definitions and schemas exposed to the model
├── vision.py               # Local Qwen2.5-VL OCR and image analysis pipeline
├── web_tools.py            # Web search, news search, safe URL fetch, proxy rotation
├── routes/
│   ├── chat.py             # /chat, /api/fix-text, title generation, summarization, pruning helpers
│   ├── conversations.py    # Conversation CRUD, export, RAG maintenance, canvas maintenance
│   ├── pages.py            # Main page, settings page, settings API
│   └── request_utils.py    # Request parsing helpers
├── rag/
│   ├── chunker.py          # Chunk splitting and chunk metadata
│   ├── embedder.py         # BGE-M3 loading and embedding generation
│   ├── ingestor.py         # Record-to-chunk conversion helpers
│   └── store.py            # ChromaDB collection/query/delete helpers
├── static/
│   ├── app.js              # Frontend application logic
│   └── style.css           # UI styling
├── templates/
│   ├── index.html          # Chat UI
│   └── settings.html       # Dedicated settings page
├── tests/
│   └── test_app.py         # Backend, streaming, tool, RAG, pruning, and UI bootstrap tests
├── proxies.example.txt     # Sample proxy file
├── requirements.txt        # Runtime dependencies
├── requirements-dev.txt    # Runtime + development tooling
└── pyproject.toml          # Ruff configuration
```

## Installation

Quick start:

```bash
bash install.sh
```

The installer asks for a system profile and accelerator, writes `.env`, and installs runtime dependencies. If you prefer a manual setup, follow the steps below.

### 1) Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

Runtime:

```bash
pip install -r requirements.txt
```

Development:

```bash
pip install -r requirements-dev.txt
```

`requirements.txt` includes the heavier RAG, OCR, export, and document-processing dependencies.

### 3) Hardware and runtime requirements

The RAG embedder and the local vision model are GPU-backed in this codebase.

- RAG embeddings require PyTorch with CUDA support and a CUDA-capable GPU.
- Local Qwen2.5-VL vision inference requires CUDA and a local model directory.
- There is no CPU fallback in the current codebase for either local RAG or local vision.
- If you do not have the required GPU stack, disable the features explicitly in `.env` instead of leaving them enabled.

Example overrides for a lighter setup:

```env
RAG_ENABLED=false
VISION_ENABLED=false
```

### 4) Create `.env`

Copy the included `.env.example` file to `.env` and fill in your values.

Minimum required:

```env
DEEPSEEK_API_KEY=your-deepseek-api-key
```

### 5) Optional proxy setup

```bash
cp proxies.example.txt proxies.txt
```

Add one proxy per line. Supported schemes:

- `http://`
- `https://`
- `socks5://`
- `socks5h://`

### 6) Run the app

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

Running `python app.py` also triggers optional preload hooks for the local vision model and the embedder. Importing `create_app()` alone does not run those preload hooks.

## Configuration

Some settings come from environment variables, and some are stored in SQLite through the Settings page.

### Required

| Variable | Description |
| --- | --- |
| `DEEPSEEK_API_KEY` | DeepSeek API key used by the OpenAI-compatible client |

### Runtime and storage

| Variable | Default | Description |
| --- | --- | --- |
| `AGENT_TRACE_LOG_PATH` | `logs/agent-trace.log` | Rotating agent trace log file |
| `IMAGE_STORAGE_DIR` | `./data/images` | Directory used for uploaded image assets |
| `DOCUMENT_STORAGE_DIR` | `./data/documents` | Directory used for uploaded document assets |
| `PROJECT_WORKSPACE_ROOT` | `./data/workspaces` | Root directory for per-conversation workspace sandboxes |
| `CANVAS_PROMPT_DEFAULT_MAX_LINES` | `800` | Default number of canvas lines injected into prompts |
| `CANVAS_EXPAND_DEFAULT_MAX_LINES` | `1600` | Default number of canvas lines returned by expand |
| `CANVAS_SCROLL_WINDOW_LINES` | `200` | Default targeted canvas scroll window |
| `SCRATCHPAD_ADMIN_EDITING_ENABLED` | `false` | Shows scratchpad editing in the UI |

### Login and session protection

| Variable | Default | Description |
| --- | --- | --- |
| `FLASK_SECRET_KEY` | `dev-only-change-me` | Secret key used for Flask sessions; override in production |
| `LOGIN_PIN` | empty | Enables the login gate when set |
| `LOGIN_SESSION_TIMEOUT_MINUTES` | `30` | Inactivity timeout for normal sessions |
| `LOGIN_MAX_FAILED_ATTEMPTS` | `3` | Failed PIN attempts before temporary lockout |
| `LOGIN_LOCKOUT_SECONDS` | `300` | Lockout duration after repeated failures |
| `LOGIN_REMEMBER_SESSION_DAYS` | `3650` | Lifetime used for remembered devices |

### RAG and embedding

| Variable | Default | Description |
| --- | --- | --- |
| `RAG_ENABLED` | `true` | Enables RAG endpoints, sync, and retrieval |
| `CHROMA_DB_PATH` | `./chroma_db` | ChromaDB persistence directory |
| `BGE_M3_MODEL_PATH` | `BAAI/bge-m3` | Embedding model name or local path |
| `BGE_M3_DEVICE` | `cuda` | Device used by the embedder in this codebase |
| `BGE_M3_LOCAL_FILES_ONLY` | `false` | Load the embedding model only from local files |
| `BGE_M3_TRUST_REMOTE_CODE` | `false` | Allow Sentence Transformers remote code |
| `BGE_M3_BATCH_SIZE` | `32` | Embedding batch size |
| `BGE_M3_PRELOAD` | `true` | Preload the embedder on startup |
| `RAG_AUTO_INJECT_TOP_K` | `3` | Seed value used to derive the default context-size preset |
| `RAG_SEARCH_DEFAULT_TOP_K` | `5` | Default knowledge-base search size |
| `RAG_AUTO_INJECT_THRESHOLD` | `0.50` | Seed value used to derive the default sensitivity preset |
| `RAG_SEARCH_MIN_SIMILARITY` | `0.35` | Minimum similarity shown in search results |
| `RAG_QUERY_EXPANSION_ENABLED` | `true` | Expands some search queries before retrieval |
| `RAG_QUERY_EXPANSION_MAX_VARIANTS` | `3` | Maximum query expansion variants |
| `RAG_TEMPORAL_DECAY_ALPHA` | `0.15` | Score decay factor for recency weighting |
| `RAG_TEMPORAL_DECAY_LAMBDA` | `0.05` | Score decay factor for time-based weighting |

Note: `rag_context_size` and `rag_sensitivity` are the runtime settings used during retrieval. The corresponding env vars above only seed the default presets stored in SQLite.

### Vision and OCR

| Variable | Default | Description |
| --- | --- | --- |
| `VISION_ENABLED` | `true` | Enables image analysis and image follow-up |
| `QWEN_VL_MODEL_PATH` | empty | Required local Qwen2.5-VL model directory |
| `QWEN_VL_ATTENTION` | empty | Optional Transformers attention implementation |
| `QWEN_VL_LOAD_IN_4BIT` | `true` | Use 4-bit loading when supported |
| `QWEN_VL_TORCH_DTYPE` | `float16` | `float16`, `bfloat16`, or `float32` |
| `QWEN_VL_MAX_NEW_TOKENS` | `768` | Generation limit for image analysis |
| `QWEN_VL_MIN_PIXELS` | `256 * 28 * 28` | Processor min-pixels setting |
| `QWEN_VL_MAX_PIXELS` | `896 * 28 * 28` | Processor max-pixels setting |
| `QWEN_VL_MAX_IMAGE_SIDE` | `1280` | Resize limit before local inference |
| `QWEN_VL_PRELOAD` | `true` | Preload the vision model on startup |

### Fetch, summarization, and prompt budgets

| Variable | Default | Description |
| --- | --- | --- |
| `FETCH_SUMMARY_TOKEN_THRESHOLD` | `3500` | Token threshold before fetched content is clipped |
| `FETCH_SUMMARY_MAX_CHARS` | `8000` | Maximum characters kept for clipped fetch content |
| `FETCH_SUMMARY_GENERAL_TOP_K` | `3` | Top-K sentences used by fetch summarization |
| `FETCH_SUMMARY_QUERY_TOP_K` | `4` | Query-aware sentence count for fetch summarization |
| `FETCH_SUMMARY_EXCERPT_MAX_CHARS` | `500` | Maximum excerpt length in summaries |
| `CHAT_SUMMARY_TRIGGER_TOKEN_COUNT` | `80000` | Visible-token count that triggers automatic summarization |
| `CHAT_SUMMARY_MODE` | `auto` | `auto`, `never`, or `aggressive` |
| `CHAT_SUMMARY_MODEL` | `deepseek-chat` | Model used for summarization |
| `PROMPT_MAX_INPUT_TOKENS` | `100000` | Upper bound for prompt budgeting |
| `PROMPT_RESPONSE_TOKEN_RESERVE` | `8000` | Reserve for model output when budgeting input |
| `PROMPT_RECENT_HISTORY_MAX_TOKENS` | `70000` | Max recent-history budget |
| `PROMPT_SUMMARY_MAX_TOKENS` | `15000` | Max summary budget |
| `PROMPT_RAG_MAX_TOKENS` | `6000` | Max RAG budget |
| `PROMPT_TOOL_MEMORY_MAX_TOKENS` | `4000` | Max tool-memory budget |
| `PROMPT_PREFLIGHT_SUMMARY_TOKEN_COUNT` | `90000` | Preflight summary trigger budget |
| `AGENT_CONTEXT_COMPACTION_THRESHOLD` | `0.85` | Fraction of budget that triggers context compaction |
| `AGENT_CONTEXT_COMPACTION_KEEP_RECENT_ROUNDS` | `2` | How many recent exchanges are preserved during compaction |
| `AGENT_TOOL_RESULT_TRANSCRIPT_MAX_CHARS` | `16000` | Maximum transcript length retained for tool results |
| `SUMMARY_SOURCE_TARGET_TOKENS` | `6000` | Target source size for summarization |
| `SUMMARY_RETRY_MIN_SOURCE_TOKENS` | `1500` | Minimum source size before retrying summary |
| `FETCH_RAW_TOOL_RESULT_MAX_TEXT_CHARS` | `24000` | Maximum raw tool-result text kept for fetch-style results |

### Scratchpad and memory

| Variable | Default | Description |
| --- | --- | --- |
| `MAX_USER_PREFERENCES_LENGTH` | `2000` | Internal limit for stored user preferences |
| `MAX_SCRATCHPAD_LENGTH` | `4000` | Internal limit for scratchpad text |

### Built-in runtime limits from code

- fetch timeout: 20 seconds
- max fetch size: 5 MB
- max redirects: 5
- web cache TTL: 24 hours
- max search/news results per query: 5
- supported image types: PNG, JPEG, WEBP
- max upload image size: 10 MB
- document upload max size: 20 MB
- document max extracted text: 50,000 characters
- canvas document limit: 12 documents per conversation
- canvas title length limit: 160 characters

### Settings stored in SQLite via the UI

The Settings page persists these values in `app_settings`:

- `user_preferences`
- `scratchpad`
- `max_steps`
- `active_tools`
- `rag_auto_inject`
- `rag_sensitivity`
- `rag_context_size`
- `rag_source_types`
- `tool_memory_auto_inject`
- `canvas_prompt_max_lines`
- `canvas_expand_max_lines`
- `canvas_scroll_window_lines`
- `chat_summary_mode`
- `chat_summary_trigger_token_count`
- `summary_skip_first`
- `summary_skip_last`
- `pruning_enabled`
- `pruning_token_threshold`
- `pruning_batch_size`
- `fetch_url_token_threshold`
- `fetch_url_clip_aggressiveness`

## Using the app

### Basic chat flow

1. Open the app.
2. Pick a model.
3. Type a message.
4. Optionally click Fix to rewrite the draft before sending.
5. Press Enter to send, or Shift+Enter for a new line.
6. Watch tool progress, reasoning, and answer text stream live.

### Title, summary, and pruning actions

- Use Generate Title to refresh a conversation title manually.
- Use Summarize to force a summary pass for the current conversation.
- Use Undo on a summary message to restore the summarized messages.
- Use Prune history to prune the first N eligible unpruned messages in the current conversation.

### Settings page

The app includes a dedicated `/settings` page.

- Assistant tab: user preferences, tool-step budget, fetch clipping, canvas limits, summarization, and pruning
- Memory tab: scratchpad, tool-memory auto-injection, RAG auto-injection, RAG source pools, and user profile memory behavior
- Tools tab: active tool permissions, including canvas and project-workspace tools
- Knowledge tab: knowledge-base uploads, RAG maintenance, and sync controls

Use the settings page when you want to change global behavior without opening layered panels on the chat screen.

### Reading the Usage and Cost panel

Use the panel as two parallel views of the same conversation:

- Provider-reported usage: what the model API returned for completed assistant turns
- Local input-source estimates: how the app thinks each prompt was assembled before those API calls were made

Section by section:

- Header badge: cumulative provider total tokens for completed assistant turns in the current conversation
- Provider totals (session): session-level sum of prompt tokens, completion tokens, total tokens, and estimated cost
- Provider totals (latest assistant turn): the most recent completed assistant reply only, including all model calls used during that reply if tools were involved
- Provider totals (latest assistant turn) also show the peak prompt seen in a single model call and the configured per-call prompt cap
- Estimated billed input sources (session): cumulative local breakdown across completed assistant turns, aligned to the provider billed prompt total
- Estimated billed input sources (latest assistant turn): local breakdown for the most recent completed assistant reply only
- Completed assistant turns: one row per completed assistant reply, with model, provider token totals, optional per-turn cost, and expandable per-call details

Important interpretation details:

- A single assistant turn can contain multiple model calls, so one row may include several prompt/response cycles
- The panel is rebuilt from stored assistant-message metadata when conversation history is reloaded
- Zero-value breakdown categories are hidden
- Local input-source totals are explanatory only and can differ from billed prompt tokens
- Cost uses the app's configured DeepSeek cache-miss input pricing because the usage payload does not expose cache-hit prompt-token counts

### Editing a previous user message

When you edit a stored user message:

1. The old user message is loaded back into the input.
2. Sending updates that message in the database.
3. All later messages in that conversation are deleted.
4. Generation restarts from that branch.

### Image-assisted messages

If you attach an image:

1. The frontend validates file type and size.
2. The backend revalidates and reads the upload.
3. The image is optimized locally.
4. Qwen2.5-VL extracts OCR text and visual context.
5. That context is injected into the user message before the main model call.

The backend also stores the analysis so follow-up questions about the same image can use the `image_explain` tool.

### Document upload workflow

If you attach a document (DOCX, PDF, TXT, CSV, Markdown, or a common code/config file):

1. The frontend validates file type and size.
2. The backend extracts plain text from the document.
3. The extracted text is stored as a file asset and can be opened in Canvas.
4. The text is injected into the user message as a context block.
5. If the extracted text is large, it is truncated before it enters the model context.

The same extraction path is also used by the knowledge-base upload form in Settings, where the uploaded file can be indexed as an `uploaded_document` source with a title, description, and auto-inject preference.

### Scratchpad workflow

- The scratchpad is a persistent text area in the Settings page when `SCRATCHPAD_ADMIN_EDITING_ENABLED` is true.
- The model can append one durable fact with `append_scratchpad` or replace the full scratchpad with `replace_scratchpad`.
- The scratchpad is included in the runtime system prompt for every turn.
- Use it for long-term user facts, preferences, or constraints that should survive across conversations.

### User profile memory workflow

- Structured conversation summaries can contain `facts`, `decisions`, `open_issues`, `entities`, and `tool_outcomes`.
- Facts that look like durable user preferences or stable constraints are written to the persistent `user_profile` table.
- Those facts are injected back into the runtime system context as a compact bullet list.
- This is separate from the scratchpad and complements it with automatically extracted memory.

### Tool memory workflow

- Successful web search, news search, and URL fetch results can be stored as tool memory when RAG is enabled.
- Enable `tool_memory_auto_inject` in Settings to inject relevant past results automatically.
- The model can explicitly search tool memory with `search_tool_memory`.
- This reduces redundant web requests and helps the model build on previously fetched information.

### Canvas documents workflow

- The model can create a canvas document with `create_canvas_document`.
- Canvas documents may be markdown or code artifacts, and in project mode they can carry `path`, `role`, `summary`, `imports`, `exports`, `symbols`, `dependencies`, `project_id`, and `workspace_id` metadata.
- The model can expand a non-active canvas file with `expand_canvas_document` when project summaries are insufficient.
- Targeted reads use `scroll_canvas_document`; existing documents can then be rewritten with `rewrite_canvas_document`.
- Line-level edits use `replace_canvas_lines`, `insert_canvas_lines`, and `delete_canvas_lines`.
- Canvas documents can be deleted with `delete_canvas_document` or cleared with `clear_canvas`.
- Canvas documents are stored in SQLite and attached to the current conversation.
- The UI exposes a collapsible canvas panel with search, role/path filters, a project tree, tabs, copy, delete, edit, and download actions.
- Canvas exports are available as Markdown, HTML, and PDF.

### Workspace project workflow

- Project-mode turns can track a structured workflow state with plan, skeleton, content, validate, fix, and validated stages.
- Workspace tools operate in a conversation-scoped sandbox rooted under `PROJECT_WORKSPACE_ROOT`.
- `plan_project_workspace` creates or revises the project plan, and `get_project_workflow_status` reports the current stage.
- `create_project_scaffold` creates a Python-project starter structure when you want a quick baseline.
- `preview_workspace_changes` shows unified diffs before a write.
- `write_project_tree` and `bulk_update_workspace_files` handle batch writes with overwrite confirmation when needed.
- `create_directory`, `create_file`, `update_file`, `read_file`, `list_dir`, and `search_files` operate inside the workspace sandbox.
- Workspace file history supports undo and redo for recorded file changes, and `validate_project_workspace` runs lightweight checks.

### Exporting conversations

You can export a conversation in three formats:

- Markdown: `/api/conversations/<id>/export?format=md`
- DOCX: `/api/conversations/<id>/export?format=docx`
- PDF: `/api/conversations/<id>/export?format=pdf`

Canvas documents can also be exported individually with `/api/conversations/<id>/canvas/export` in Markdown, HTML, or PDF.

### Chat summarization workflow

- When a conversation grows beyond the configured visible-token threshold, the app can automatically summarize older messages.
- `CHAT_SUMMARY_MODE` can be `auto`, `never`, or `aggressive`.
- Summaries are stored as special system messages so the original flow is preserved while context size is reduced.
- Summary behavior also respects the `summary_skip_first` and `summary_skip_last` settings from the UI.
- Summary generation also feeds durable facts into the user profile memory when the output includes usable facts.

### Pruning workflow

- Pruning is separate from summarization and targets visible user and assistant messages.
- The background post-response task can prune older visible messages once the prunable-token count crosses `pruning_token_threshold`.
- `pruning_batch_size` controls how many messages are compacted per pass.
- Messages that already contain tool calls, summaries, or prior pruning markers are skipped.
- A manual prune endpoint exists for individual messages.

### Knowledge base workflow

RAG is a structured retrieval layer in this codebase.

Supported behavior:

- sync existing conversations into RAG
- sync successful text-like tool results into RAG
- sync tool-memory entries into RAG
- index uploaded documents as `uploaded_document` sources via the Settings page or `/api/rag/ingest`
- search the knowledge base from the API or from the model tool
- auto-inject retrieved context into each chat turn
- delete indexed sources one by one

The Settings page can scope retrieval to conversation, tool result, tool memory, and uploaded-document source pools.

## Available tools

Only tools enabled in Settings are exposed to the model. If RAG is disabled, `search_knowledge_base` is removed from the tool list even if it is enabled in settings.

### Memory and personalization

#### `append_scratchpad`

Append one durable user-specific fact or preference to the persistent scratchpad.

- Arguments:
  - `note` (string, required) - one short durable note to append

#### `replace_scratchpad`

Completely replace the persistent scratchpad content.

- Arguments:
  - `new_content` (string, required) - the new content that fully replaces the scratchpad

#### `ask_clarifying_question`

Ask one or more structured clarification questions and stop answering until the user replies.

- Arguments:
  - `questions` (array, required) - 1 to 5 questions, each with `id`, `label`, and `input_type`
  - `intro` (string, optional) - short lead-in shown before the questions
  - `submit_label` (string, optional) - optional button label in the UI

Each question item can also include `required`, `placeholder`, `options`, and `allow_free_text`.

#### `image_explain`

Answer a follow-up question about a previously uploaded image saved in the current conversation.

- Arguments:
  - `image_id` (string, required) - stored image id
  - `conversation_id` (integer, required) - current conversation id
  - `question` (string, required) - focused follow-up question about the image, written in English

### Knowledge base and tool memory

#### `search_knowledge_base`

Semantic search over synced conversations, stored tool results, remembered web results, and uploaded documents.

- Arguments:
  - `query` (string, required) - semantic search query
  - `category` (string, optional) - optional category filter
  - `top_k` (integer, optional, 1-12) - maximum number of chunks to retrieve

The current code accepts conversation, tool_result, tool_memory, and uploaded_document categories.

#### `search_tool_memory`

Search past web tool results stored from previous conversations.

- Arguments:
  - `query` (string, required) - semantic search query for past web tool results
  - `top_k` (integer, optional, 1-10) - maximum number of remembered results to retrieve

### Web search and browsing

#### `search_web`

DuckDuckGo text search.

- Arguments:
  - `queries` (array, required, 1-5 strings) - list of search queries to run

#### `fetch_url`

Fetch and read the content of a specific web page. Returns cleaned text and metadata.

- Arguments:
  - `url` (string, required) - full HTTP or HTTPS URL

### News search

#### `search_news_ddgs`

DuckDuckGo News search.

- Arguments:
  - `queries` (array, required, 1-5 strings)
  - `lang` (string, optional) - `tr` or `en`
  - `when` (string, optional) - `d`, `w`, `m`, or `y`

#### `search_news_google`

Google News RSS search.

- Arguments:
  - `queries` (array, required, 1-5 strings)
  - `lang` (string, optional) - `tr` or `en`
  - `when` (string, optional) - `d`, `w`, `m`, or `y`

### Canvas document editing

#### `create_canvas_document`

Create a new canvas document for the current conversation.

- Arguments:
  - `title` (string, required) - document title
  - `content` (string, required) - full document content
  - `format` (string, optional) - `markdown` or `code`
  - `language` (string, optional) - optional dominant code language
  - `path` (string, optional) - optional project-relative path
  - `role` (string, optional) - optional semantic role inside a project workspace
  - `summary` (string, optional) - optional short responsibility summary

#### `expand_canvas_document`

Load the full context of a canvas document when the current excerpt is not enough.

- Arguments:
  - `document_id` (string, optional) - target canvas document id
  - `document_path` (string, optional) - target project-relative path

#### `scroll_canvas_document`

Read a targeted 1-based line range from a canvas document.

- Arguments:
  - `document_id` (string, optional) - target canvas document id
  - `document_path` (string, optional) - target project-relative path
  - `start_line` (integer, required) - 1-based starting line number
  - `end_line` (integer, required) - 1-based ending line number

#### `rewrite_canvas_document`

Rewrite the full active canvas document while keeping the same document id.

- Arguments:
  - `content` (string, required) - full replacement content
  - `title` (string, optional) - optional replacement title
  - `format` (string, optional) - `markdown` or `code`
  - `language` (string, optional) - optional dominant code language
  - `path` (string, optional) - optional project-relative path
  - `role` (string, optional) - optional semantic role inside a project workspace
  - `summary` (string, optional) - optional short responsibility summary
  - `document_id` (string, optional) - optional target document id

#### `replace_canvas_lines`

Replace a 1-based inclusive line range inside the active canvas document.

- Arguments:
  - `start_line` (integer, required)
  - `end_line` (integer, required)
  - `lines` (array, required) - replacement lines without trailing newlines
  - `document_id` (string, optional)

#### `insert_canvas_lines`

Insert one or more lines into the active canvas document after a given line number.

- Arguments:
  - `after_line` (integer, required)
  - `lines` (array, required)
  - `document_id` (string, optional)

#### `delete_canvas_lines`

Delete a 1-based inclusive line range from the active canvas document.

- Arguments:
  - `start_line` (integer, required)
  - `end_line` (integer, required)
  - `document_id` (string, optional)

#### `delete_canvas_document`

Delete a single canvas document.

- Arguments:
  - `document_id` (string, optional)

#### `clear_canvas`

Delete all canvas documents for the current conversation.

- Arguments: none

### Project workspace tools

Workspace tools let the assistant plan, preview, scaffold, validate, and edit files in the conversation sandbox rooted under `PROJECT_WORKSPACE_ROOT`.

- `plan_project_workspace` creates or revises a structured project plan before file writes.
- `get_project_workflow_status` returns the current project workflow stage.
- `create_project_scaffold` creates a starter Python-project layout.
- `preview_workspace_changes` previews unified diffs for proposed writes.
- `write_project_tree` and `bulk_update_workspace_files` apply batch file writes with overwrite confirmation when needed.
- `create_directory`, `create_file`, `update_file`, `read_file`, `list_dir`, and `search_files` operate inside the workspace sandbox.
- `get_workspace_file_history`, `undo_workspace_file_change`, and `redo_workspace_file_change` manage file history.
- `validate_project_workspace` runs lightweight validation on the workspace or a subdirectory.

## HTTP endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/` | Main chat UI |
| `GET` | `/settings` | Dedicated settings page |
| `GET` | `/api/settings` | Read persisted settings |
| `PATCH` | `/api/settings` | Update persisted settings |
| `POST` | `/api/fix-text` | Rewrite the current draft before sending |
| `POST` | `/chat` | Main streamed chat endpoint; accepts JSON or multipart uploads |
| `POST` | `/api/conversations/<id>/generate-title` | Generate a title from conversation content |
| `POST` | `/api/conversations/<id>/summarize` | Manually summarize a conversation |
| `POST` | `/api/conversations/<id>/summaries/<summary_id>/undo` | Undo a summary message |
| `POST` | `/api/messages/<id>/prune` | Prune a visible user or assistant message |
| `GET` | `/api/conversations` | List conversations |
| `POST` | `/api/conversations` | Create a conversation |
| `GET` | `/api/conversations/<id>` | Load one conversation and all messages |
| `PATCH` | `/api/conversations/<id>` | Rename a conversation |
| `DELETE` | `/api/conversations/<id>` | Delete a conversation |
| `GET` | `/api/conversations/<id>/export` | Export a conversation as Markdown, DOCX, or PDF |
| `GET` | `/api/conversations/<id>/canvas/export` | Export a canvas document as Markdown, HTML, or PDF |
| `DELETE` | `/api/conversations/<id>/canvas` | Delete one canvas document or clear all canvas documents |
| `GET` | `/api/rag/documents` | List indexed RAG sources |
| `DELETE` | `/api/rag/documents/<source_key>` | Delete one indexed RAG source |
| `GET` | `/api/rag/search?q=...` | Search the knowledge base |
| `POST` | `/api/rag/sync-conversations` | Sync one conversation or all conversations into RAG |
| `POST` | `/api/rag/upload-metadata` | Suggest a title and description for an uploaded knowledge-base file |
| `POST` | `/api/rag/ingest` | Upload text or a file into RAG as an `uploaded_document` source |

## Data storage

### SQLite

The app creates and uses these tables:

- `conversations`
- `messages`
- `app_settings`
- `user_profile`
- `image_assets`
- `file_assets`
- `web_cache`
- `rag_documents`

`messages.metadata` can contain:

- image metadata
- OCR text
- vision summary
- assistant guidance
- key points
- reasoning content
- tool trace
- stored tool results
- canvas documents
- pending clarification payloads
- usage data
- summary metadata
- pruning metadata

### ChromaDB

RAG data is stored in a persistent Chroma collection under `CHROMA_DB_PATH`.

### Files on disk

- SQLite database defaults to `chatbot.db`
- Chroma persistence defaults to `chroma_db/`
- agent logs default to `logs/agent-trace.log`
- optional proxies are loaded from `proxies.txt`
- uploaded images are stored in `data/images/`
- uploaded documents are stored in `data/documents/`
- project workspace files default to `data/workspaces/`
- workspace history metadata is stored alongside each workspace under `.workspace-history/`

## Development

### Run tests

```bash
/home/ricky/Desktop/os-chatbot/venv/bin/python -m pytest tests/test_app.py
```

### Lint

```bash
ruff check .
```

### Format

```bash
ruff format .
```

### App factory usage

You can create isolated app instances with a separate database path:

```python
from app import create_app

app = create_app(database_path="/tmp/chatbot-test.db")
```

That is how the test suite keeps databases isolated.

### Pre-commit hooks (optional)

If you have pre-commit installed, you can add a hook to run Ruff formatting and linting before each commit:

```bash
pre-commit install
```

A sample `.pre-commit-config.yaml` is not included in the repository.

## Security and operational notes

- `fetch_url` rejects localhost and private-network targets
- only enabled tools are exposed to the model
- tool arguments are schema-validated before execution
- uploaded images and documents are MIME-checked and size-limited
- assistant Markdown is sanitized in the browser
- repeated web and fetch calls benefit from caching
- knowledge-base ingestion is limited to synced conversations, tool outputs, remembered web results, and explicit uploads through Settings
- the app stores conversation content locally; plan backups accordingly
- local vision requires a local model download and enough hardware to run it
- scratchpad admin editing is disabled by default; enable it only if you trust the UI users
- pruning rewrites message content in place and stores the original text in message metadata
- when `LOGIN_PIN` is set, the app requires a PIN before any page or API route is accessible

## Troubleshooting

**RAG or vision fails with CUDA errors**

Set `RAG_ENABLED=false` and/or `VISION_ENABLED=false` in `.env` to disable GPU-dependent features. Otherwise, make sure you have a compatible NVIDIA GPU, CUDA toolkit, and a PyTorch build with CUDA support.

**Proxy rotation not working**

Ensure `proxies.txt` exists and contains valid proxy entries, one per line. The file is read on startup, so restart the app after making changes.

**Tool memory auto-injection is not happening**

Check that `tool_memory_auto_inject` is enabled in Settings and that RAG is enabled. Tool-memory entries are only created for successful web-tool results.

**Chat summarization seems too aggressive**

Increase `CHAT_SUMMARY_TRIGGER_TOKEN_COUNT` or set `chat_summary_mode` to `never` in Settings.

**Pruning is too aggressive**

Raise `pruning_token_threshold`, lower `pruning_batch_size`, or disable pruning in the Settings page.

**Scratchpad is not visible in the UI**

The scratchpad editor is only shown when `SCRATCHPAD_ADMIN_EDITING_ENABLED` is true. The scratchpad still exists for the model even when the UI editor is hidden.

**Canvas documents are missing**

Canvas documents are attached to the current conversation. Make sure you are in the same conversation where the document was created, and open the Canvas panel from the top bar.

**Vision uploads are rejected**

Image uploads require an existing saved conversation and a supported image type. Vision must also be enabled in `.env`.

**Document uploads fail**

Only DOCX, PDF, TXT, CSV, Markdown, and common code/config files are accepted. The document must also contain extractable text.

## FAQ

**How do I reset the database?**

Delete `chatbot.db` and restart the app. The schema will be recreated automatically.

**How do I clear the RAG vector store?**

Delete the `chroma_db` directory or the path configured in `CHROMA_DB_PATH`, then restart.

**Why is the vision model not loading?**

Make sure `QWEN_VL_MODEL_PATH` points to a local directory that contains the Qwen2.5-VL model files.

**How do I enable debug logging?**

Set `AGENT_TRACE_LOG_PATH` to a writable file path and inspect the rotating log file.

**How can I disable a specific tool?**

Use the Settings page to uncheck the tool in the active tool list. The tool will no longer be exposed to the model.

## License

MIT