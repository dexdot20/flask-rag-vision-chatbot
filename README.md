# Flask ChatBot: DeepSeek + Tools + RAG + Local Vision OCR

![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.x-000000?logo=flask&logoColor=white)
![RAG](https://img.shields.io/badge/RAG-ChromaDB%20%2B%20BGE--M3-6E44FF)
![Vision](https://img.shields.io/badge/Vision-Qwen2.5--VL-0EA5E9)
![License](https://img.shields.io/badge/License-MIT-green)

This project is a single-page Flask chat application that combines:

- DeepSeek chat models via an OpenAI-compatible client
- multi-step tool use with live streaming status
- persistent conversations in SQLite
- conversation/tool-result RAG with ChromaDB + BGE-M3
- optional local image analysis with Qwen2.5-VL
- provider-reported token usage, estimated cost, reasoning, and tool-trace visibility in the UI

It is not a minimal “send prompt, get answer” demo. The app keeps chat history, restores assistant metadata when a conversation is reopened, lets you edit older user turns, syncs chats into a searchable knowledge base, and can enrich a message with local OCR/vision context before the model sees it.

## Table of Contents

- [What the app can do](#what-the-app-can-do)
- [Architecture overview](#architecture-overview)
- [Project structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Using the app](#using-the-app)
- [Available tools](#available-tools)
- [RAG behavior](#rag-behavior)
- [Vision behavior](#vision-behavior)
- [HTTP endpoints](#http-endpoints)
- [Data storage](#data-storage)
- [Development](#development)
- [Security and operational notes](#security-and-operational-notes)
- [License](#license)

## What the app can do

### Chat and conversation UX

- Create, open, rename, and delete conversations
- Persist all messages in SQLite
- Stream assistant output to the browser as NDJSON
- Show reasoning content in a separate collapsible panel when the model provides it
- Render assistant messages as Markdown with DOMPurify sanitization
- Cancel an active response mid-stream
- Clear the current chat view without deleting stored conversations
- Automatically generate a short conversation title after the first user/assistant exchange
- Edit an older user message, delete everything after that point, and regenerate from there
- Restore stored reasoning, token usage data, tool traces, and tool-result badges when reopening a conversation

### Model and agent behavior

- Supports:
  - `deepseek-chat`
  - `deepseek-reasoner`
- Uses the DeepSeek API through the OpenAI Python client
- Supports both native tool-call responses and custom JSON tool-call responses from the model
- Validates tool names and tool argument schemas before execution
- Limits tool rounds with configurable `max_steps` (1-10)
- Forces a final answer phase when the tool budget is exhausted
- Tracks estimated token-source breakdown locally:
  - system prompt
  - user messages
  - assistant history
  - tool results
  - RAG context
  - final instruction
- Estimates per-turn and session cost with current DeepSeek V3.2 API pricing
- Writes rotating agent trace logs to `logs/agent-trace.log` by default

### Tooling and web access

- Tool permissions are configurable from the UI and stored in SQLite
- In-turn in-memory tool-result cache to avoid duplicate executions in the same response
- Persistent SQLite web cache for repeated search/fetch requests
- DuckDuckGo web search
- URL fetching with readable-content extraction
- DuckDuckGo News search
- Google News RSS search
- Proxy rotation from `proxies.txt` with direct fallback
- Per-tool progress UI with step number, preview, duration, cached status, and final summary

### URL fetch behavior

`fetch_url` is more than a raw HTTP GET:

- supports HTML, plain text, JSON, XML, and PDF
- blocks localhost and private-network targets
- applies browser-like headers and retry variants
- follows redirects with limits
- caps downloaded content size
- cleans extracted text aggressively before sending it to the model
- marks partial-content recoveries when a connection ends early but some text was recovered
- clips oversized page text before it enters the model context
- stores fetch diagnostics and cleanup metadata with assistant messages
- exposes “cleaned”, “clipped”, or “summarized” state back to the UI

### RAG / knowledge base

- Uses ChromaDB as a persistent vector store
- Uses BGE-M3 embeddings through Sentence Transformers
- Indexes only these source types:
  - conversation history
  - successful text-like tool results
- Can auto-inject retrieved context into each chat turn
- Can expose `search_knowledge_base` as a tool to the model
- Lets you list indexed sources in the UI
- Lets you delete indexed sources individually
- Syncs one conversation or all conversations into RAG via API
- Keeps conversation content and tool-result content in separate source categories
- Filters search results by similarity threshold

### Local vision / OCR

- Accepts `PNG`, `JPEG`, and `WEBP`
- Enforces a 10 MB upload limit
- Validates MIME type and file presence
- Optimizes and optionally resizes images before local inference
- Runs a local Qwen2.5-VL pipeline if configured
- Produces:
  - `ocr_text`
  - `vision_summary`
  - `key_points`
  - `assistant_guidance`
- Injects that vision context into the actual user message sent to the chat model
- Shows attachment badges and a compact vision note in the UI

### Observability in the UI

The Token Usage panel separates provider-reported usage from local estimates.

- Provider totals (session): prompt, completion, total tokens, and estimated cost across all completed assistant turns in the current conversation
- Provider totals (latest assistant turn): prompt, completion, total tokens, and model for the most recent completed assistant reply
- Latest-turn totals include every model call made while producing that reply, including tool-execution rounds
- Estimated input sources (session and latest turn): local prompt-source breakdown for these categories:
  - system prompt
  - user messages
  - assistant history
  - tool results
  - RAG context
  - final instruction
- Estimated input-source totals are inferred locally and may differ from provider-billed input tokens
- Estimated cost uses the current DeepSeek V3.2 API prices shown on the pricing page: $0.28 per 1M input tokens on a cache-miss basis and $0.42 per 1M output tokens; cache-hit discounts are not included because the API usage payload does not expose cache-hit counts
- Completed assistant turns list: one row per assistant reply with provider totals, model, optional cost, and the local input-source estimate for that turn
- Live tool-trace panel for assistant messages
- Visual badge when fetched web content was cleaned/clipped/summarized

## Architecture overview

1. The browser sends either JSON or `multipart/form-data` to `/chat`.
2. The backend loads saved settings from SQLite.
3. If an image is attached, local vision analysis runs first.
4. If RAG auto-inject is enabled, the latest user message is searched against ChromaDB.
5. A runtime system message is prepended with:
   - current date/time and timezone
   - user preferences
   - enabled tool context
   - tool-call contract
   - auto-injected RAG context, if any
6. The agent streams model output.
7. If the model emits tool calls, they are validated, executed, cached, and appended to the transcript as tool messages.
8. Tool progress, reasoning deltas, answer deltas, usage, and message IDs are streamed back to the browser as NDJSON events.
9. The final assistant message is stored with metadata such as reasoning, usage, tool trace, and storable tool results.
10. Conversations and successful tool text can later be synced into the RAG store.

## Project structure

```text
.
├── app.py                  # Flask app factory and local entrypoint
├── agent.py                # Streaming agent loop, tool execution, usage tracking, trace logging
├── config.py               # Environment variables, defaults, model list, limits
├── db.py                   # SQLite schema, settings, cache, metadata serialization
├── messages.py             # Runtime prompt context and API-message preparation
├── rag_service.py          # RAG sync/search orchestration for conversations and tool results
├── tool_registry.py        # Tool definitions and JSON schemas exposed to the model
├── vision.py               # Local Qwen2.5-VL image analysis pipeline
├── web_tools.py            # Web/news search, proxy loading, safe URL fetch, content extraction
├── routes/
│   ├── chat.py             # /chat, /api/fix-text, title generation, preload helpers
│   ├── conversations.py    # conversation CRUD and RAG endpoints
│   └── pages.py            # index page and settings endpoints
├── rag/
│   ├── chunker.py          # chunk splitting and chunk metadata
│   ├── embedder.py         # BGE-M3 loading and embedding generation
│   ├── ingestor.py         # record-to-chunk conversion helpers
│   └── store.py            # ChromaDB collection/query/delete helpers
├── static/
│   ├── app.js              # full frontend app logic
│   └── style.css           # UI styles
├── templates/
│   └── index.html          # single-page UI template
├── tests/
│   └── test_app.py         # backend, stream, tool, RAG, and frontend bootstrap tests
├── proxies.example.txt     # sample proxy file
├── requirements.txt        # runtime dependencies
├── requirements-dev.txt    # runtime + dev tooling
└── pyproject.toml          # Ruff configuration
```

## Installation

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

> `requirements.txt` already includes the heavier RAG and vision dependencies. If you want the simplest install, the codebase does not currently provide a separate lightweight requirements file.

### 3) Create `.env`

There is no `.env.example` in the repository. Create `.env` manually.

Minimum required:

```env
DEEPSEEK_API_KEY=your-deepseek-api-key
```

### 4) Optional proxy setup

```bash
cp proxies.example.txt proxies.txt
```

Add one proxy per line. Supported schemes:

- `http://`
- `https://`
- `socks5://`
- `socks5h://`

### 5) Run the app

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

`python app.py` also triggers optional preload hooks for the local vision model and the embedder. If you only import `create_app()`, those preload hooks are not run automatically.

## Configuration

Some settings come from environment variables, and some are stored in SQLite through the Settings panel.

### Required

| Variable | Description |
| --- | --- |
| `DEEPSEEK_API_KEY` | DeepSeek API key used by the OpenAI-compatible client |

### General / logging

| Variable | Default | Description |
| --- | --- | --- |
| `AGENT_TRACE_LOG_PATH` | `logs/agent-trace.log` | Rotating agent trace log file |

### RAG / embedding

| Variable | Default | Description |
| --- | --- | --- |
| `CHROMA_DB_PATH` | `./chroma_db` | ChromaDB persistence directory |
| `BGE_M3_MODEL_PATH` | `BAAI/bge-m3` | Embedding model name or local path |
| `BGE_M3_DEVICE` | auto | Force device such as `cpu` or `cuda` |
| `BGE_M3_LOCAL_FILES_ONLY` | `false` | Load embedding model only from local files |
| `BGE_M3_TRUST_REMOTE_CODE` | `false` | Sentence Transformers remote-code permission |
| `BGE_M3_BATCH_SIZE` | `32` | Embedding batch size |
| `BGE_M3_PRELOAD` | `true` | Preload embedder on startup |
| `RAG_AUTO_INJECT_TOP_K` | `5` | Auto-inject candidate count |
| `RAG_SEARCH_DEFAULT_TOP_K` | `5` | Default knowledge-base search size |
| `RAG_AUTO_INJECT_THRESHOLD` | `0.35` | Similarity threshold for auto-injected context |
| `RAG_SEARCH_MIN_SIMILARITY` | `0.2` | Minimum similarity shown in RAG search results |

### Local vision / OCR

| Variable | Default | Description |
| --- | --- | --- |
| `QWEN_VL_MODEL_PATH` | empty | Required local Qwen2.5-VL model directory |
| `QWEN_VL_ATTENTION` | empty | Optional Transformers attention implementation |
| `QWEN_VL_LOAD_IN_4BIT` | `true` | Use 4-bit loading when CUDA is available |
| `QWEN_VL_TORCH_DTYPE` | `float16` | `float16`, `bfloat16`, or `float32` |
| `QWEN_VL_MAX_NEW_TOKENS` | `768` | Generation limit for image analysis |
| `QWEN_VL_MIN_PIXELS` | `256 * 28 * 28` | Processor min-pixels setting |
| `QWEN_VL_MAX_PIXELS` | `896 * 28 * 28` | Processor max-pixels setting |
| `QWEN_VL_MAX_IMAGE_SIDE` | `1280` | Resize limit before local inference |
| `QWEN_VL_PRELOAD` | `true` | Preload the vision model on startup |

### Fetch / clipping behavior

| Variable | Default | Description |
| --- | --- | --- |
| `FETCH_SUMMARY_TOKEN_THRESHOLD` | `3500` | Token threshold before long fetched content is clipped |
| `FETCH_RAW_TOOL_RESULT_MAX_TEXT_CHARS` | `24000` | Max raw fetch text preserved in stored tool metadata |

### Built-in runtime limits from code

- fetch timeout: **20 seconds**
- max fetch size: **5 MB**
- max redirects: **5**
- web cache TTL: **24 hours**
- max search/news results per query: **5**
- supported image types: **PNG / JPEG / WEBP**
- max upload image size: **10 MB**

### Settings stored in SQLite via the UI

The Settings panel persists these values in `app_settings`:

- `user_preferences`
- `max_steps`
- `active_tools`
- `rag_auto_inject`
- `fetch_url_token_threshold`
- `fetch_url_clip_aggressiveness`

## Using the app

### Basic chat flow

1. Open the app.
2. Pick a model.
3. Type a message.
4. Optionally click `Fix` to rewrite your draft before sending.
5. Press `Enter` to send, or `Shift+Enter` for a new line.
6. Watch tool progress, reasoning, and answer text stream live.

### Reading the Token Usage panel

- The header badge shows the sum of provider-reported total tokens for completed assistant turns in the current conversation
- `Provider totals` come from usage metadata returned by the model API
- `Estimated cost` currently uses DeepSeek V3.2 cache-miss input pricing because the returned usage payload does not indicate which prompt tokens were billed as cache hits
- `Estimated input sources` are computed locally from the messages sent to the model and are meant to explain where input context came from, not to replicate billing exactly
- `Completed assistant turns` shows one row per finished assistant reply; a single row can already include multiple model calls if tools were used during that reply

### Editing a previous user message

When you click `Edit` on a stored user message:

1. the old user message is loaded back into the input,
2. sending updates that message in the database,
3. all later messages in that conversation are deleted,
4. generation restarts from that branch.

### Image-assisted messages

If you attach an image:

1. the frontend validates file type and size,
2. the backend re-validates and reads the upload,
3. the image is optimized locally,
4. Qwen2.5-VL extracts OCR and visual context,
5. that context is injected into the user message before the main model call.

### Knowledge-base workflow

RAG is not a generic file-ingestion system in the current codebase.

What is supported:

- sync existing conversations into RAG
- sync successful text-like tool results into RAG
- search the knowledge base from the API
- enable auto-injection per chat turn
- delete indexed sources one by one

What is intentionally disabled:

- manual `/api/rag/ingest` document ingestion

## Available tools

Only tools enabled in Settings are exposed to the model.

### `search_knowledge_base`

Semantic search over synced conversations and stored tool results.

Arguments:

- `query` *(required)*
- `category` *(optional: `conversation` or `tool_result`)*
- `top_k` *(optional, 1-12)*

### `search_web`

DuckDuckGo text search.

Arguments:

- `queries` *(required, 1-5 strings)*

### `fetch_url`

Fetches and cleans readable content from one URL.

Arguments:

- `url` *(required)*

Behavior summary:

- safety checks against localhost/private networks
- content extraction for HTML, text, JSON, XML, PDF
- content cleanup
- clipping for long pages
- diagnostics attached to stored tool results

### `search_news_ddgs`

DuckDuckGo News search.

Arguments:

- `queries` *(required, 1-5 strings)*
- `lang` *(optional: `tr` or `en`)*
- `when` *(optional: `d`, `w`, `m`, `y`)*

### `search_news_google`

Google News RSS search.

Arguments:

- `queries` *(required, 1-5 strings)*
- `lang` *(optional: `tr` or `en`)*
- `when` *(optional: `d`, `w`, `m`, `y`)*

## RAG behavior

- Chunks are created from stored conversation records and stored tool results
- Conversation sources and tool-result sources are indexed separately
- Chunks are normalized and deduplicated before storage
- Similarity search uses cosine space in ChromaDB
- Search results include:
  - source key
  - source name
  - source type
  - category
  - chunk index
  - similarity
  - clipped excerpt
- Auto-injected context is added to the runtime system message, not directly appended as a visible user message

## Vision behavior

- Vision is optional, but if enabled it is fully local
- The app expects `QWEN_VL_MODEL_PATH` to point to an existing local directory
- If the model path is missing, vision requests fail with a clear runtime error
- The image-analysis prompt asks the vision model to return strict JSON
- If the vision model returns malformed output, the backend normalizes it into a safe fallback structure
- OCR/summary/guidance are stored in message metadata and shown again when the conversation is reopened

## HTTP endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/` | Main single-page UI |
| `GET` | `/api/settings` | Read persisted app settings |
| `PATCH` | `/api/settings` | Update persisted settings |
| `POST` | `/api/fix-text` | Rewrite the current input before sending |
| `POST` | `/chat` | Main streamed chat endpoint; accepts JSON or multipart for image uploads |
| `GET` | `/api/conversations` | List conversations |
| `POST` | `/api/conversations` | Create a conversation |
| `GET` | `/api/conversations/<id>` | Load one conversation and all messages |
| `PATCH` | `/api/conversations/<id>` | Rename a conversation |
| `DELETE` | `/api/conversations/<id>` | Delete a conversation |
| `POST` | `/api/conversations/<id>/generate-title` | Generate a short title from the first turn |
| `GET` | `/api/rag/documents` | List indexed RAG sources |
| `DELETE` | `/api/rag/documents/<source_key>` | Delete one indexed RAG source |
| `GET` | `/api/rag/search?q=...` | Search the knowledge base |
| `POST` | `/api/rag/sync-conversations` | Sync one or all conversations into RAG |
| `POST` | `/api/rag/ingest` | Disabled on purpose; returns `410 Gone` |

## Data storage

### SQLite

The app creates and uses these tables:

- `conversations`
- `messages`
- `app_settings`
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
- normalized usage data

### ChromaDB

RAG data is stored in a persistent Chroma collection named `knowledge_base` under `CHROMA_DB_PATH`.

### Files on disk

- SQLite database defaults to `chatbot.db`
- Chroma persistence defaults to `chroma_db/`
- agent logs default to `logs/agent-trace.log`
- optional proxies are loaded from `proxies.txt`

## Development

### Run tests

```bash
python -m unittest discover -s tests
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

## Security and operational notes

- `fetch_url` rejects localhost and private-network targets
- only enabled tools are exposed to the model
- tool arguments are schema-validated before execution
- uploaded images are MIME-checked and size-limited
- assistant Markdown is sanitized in the browser
- repeated web/tool calls benefit from caching
- manual RAG ingestion is disabled to keep indexed sources limited to internal conversation/tool data
- the app stores conversation content locally; plan storage and backups accordingly
- local vision requires a local model download and enough hardware to run it

## License

MIT
