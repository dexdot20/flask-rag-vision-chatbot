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
- conversation/tool‑result/tool‑memory RAG with ChromaDB + BGE‑M3
- optional local image analysis with Qwen2.5‑VL
- provider‑reported token usage, estimated cost, reasoning, and tool‑trace visibility in the UI
- persistent scratchpad for long‑term user facts
- tool‑memory auto‑injection of past web/search results
- editable canvas documents for structured long‑form artifacts
- chat‑history summarization to keep context manageable

It is not a minimal “send prompt, get answer” demo. The app keeps chat history, restores assistant metadata when a conversation is reopened, lets you edit older user turns, syncs chats into a searchable knowledge base, and can enrich a message with local OCR/vision context before the model sees it.

## Table of Contents

- [What the app can do](#what-the-app-can-do)
  - [Chat and conversation UX](#chat-and-conversation-ux)
  - [Model and agent behavior](#model-and-agent-behavior)
  - [Tooling and web access](#tooling-and-web-access)
  - [URL fetch behavior](#url-fetch-behavior)
  - [RAG / knowledge base](#rag--knowledge-base)
  - [Local vision / OCR](#local-vision--ocr)
  - [Scratchpad (persistent memory)](#scratchpad-persistent-memory)
  - [Tool memory (web‑result memory)](#tool-memory-webresult-memory)
  - [Canvas documents](#canvas-documents)
  - [Chat summarization](#chat-summarization)
  - [Observability in the UI](#observability-in-the-ui)
- [Architecture overview](#architecture-overview)
- [Project structure](#project-structure)
- [Installation](#installation)
  - [1) Create a virtual environment](#1-create-a-virtual-environment)
  - [2) Install dependencies](#2-install-dependencies)
  - [3) GPU and CUDA requirements](#3-gpu-and-cuda-requirements)
  - [4) Create `.env`](#4-create-env)
  - [5) Optional proxy setup](#5-optional-proxy-setup)
  - [6) Run the app](#6-run-the-app)
- [Configuration](#configuration)
  - [Required](#required)
  - [General / logging](#general--logging)
  - [RAG / embedding](#rag--embedding)
  - [Local vision / OCR](#local-vision--ocr-1)
  - [Fetch / clipping behavior](#fetch--clipping-behavior)
  - [Chat summarization](#chat-summarization-1)
  - [Scratchpad administration](#scratchpad-administration)
  - [Built‑in runtime limits from code](#builtin-runtime-limits-from-code)
  - [Settings stored in SQLite via the UI](#settings-stored-in-sqlite-via-the-ui)
- [Using the app](#using-the-app)
  - [Basic chat flow](#basic-chat-flow)
  - [Reading the Token Usage panel](#reading-the-token-usage-panel)
  - [Editing a previous user message](#editing-a-previous-user-message)
  - [Image‑assisted messages](#imageassisted-messages)
  - [Scratchpad workflow](#scratchpad-workflow)
  - [Tool memory workflow](#tool-memory-workflow)
  - [Canvas documents workflow](#canvas-documents-workflow)
  - [Chat summarization workflow](#chat-summarization-workflow)
  - [Knowledge‑base workflow](#knowledgebase-workflow)
- [Available tools](#available-tools)
  - [Memory & personalization](#memory--personalization)
  - [Knowledge base & tool memory](#knowledge-base--tool-memory)
  - [Web search & browsing](#web-search--browsing)
  - [News search](#news-search)
  - [Canvas document editing](#canvas-document-editing)
- [RAG behavior](#rag-behavior)
- [Vision behavior](#vision-behavior)
- [HTTP endpoints](#http-endpoints)
- [Data storage](#data-storage)
  - [SQLite](#sqlite)
  - [ChromaDB](#chromadb)
  - [Files on disk](#files-on-disk)
- [Development](#development)
  - [Run tests](#run-tests)
  - [Lint](#lint)
  - [Format](#format)
  - [App factory usage](#app-factory-usage)
  - [Pre‑commit hooks (optional)](#precommit-hooks-optional)
- [Security and operational notes](#security-and-operational-notes)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## What the app can do

### Chat and conversation UX

- Create, open, rename, and delete conversations
- Persist all messages in SQLite
- Stream assistant output to the browser as NDJSON
- Show reasoning content in a separate collapsible panel when the model provides it
- Render assistant messages as Markdown with DOMPurify sanitization
- Cancel an active response mid‑stream
- Clear the current chat view without deleting stored conversations
- Automatically generate a short conversation title after the first user/assistant exchange
- Edit an older user message, delete everything after that point, and regenerate from there
- Restore stored reasoning, token usage data, tool traces, and tool‑result badges when reopening a conversation

### Model and agent behavior

- Supports:
  - `deepseek‑chat`
  - `deepseek‑reasoner`
- Uses the DeepSeek API through the OpenAI Python client
- Supports both native tool‑call responses and custom JSON tool‑call responses from the model
- Validates tool names and tool argument schemas before execution
- Limits tool rounds with configurable `max_steps` (1‑10)
- Forces a final answer phase when the tool budget is exhausted
- Tracks estimated token‑source breakdown locally:
  - system prompt
  - user messages
  - assistant history
  - tool results
  - RAG context
  - final instruction
- Estimates per‑turn and session cost with current DeepSeek V3.2 API pricing
- Writes rotating agent trace logs to `logs/agent‑trace.log` by default

### Tooling and web access

- Tool permissions are configurable from the UI and stored in SQLite
- In‑turn in‑memory tool‑result cache to avoid duplicate executions in the same response
- Persistent SQLite web cache for repeated search/fetch requests
- DuckDuckGo web search
- URL fetching with readable‑content extraction
- DuckDuckGo News search
- Google News RSS search
- Proxy rotation from `proxies.txt` with direct fallback
- Per‑tool progress UI with step number, preview, duration, cached status, and final summary

### URL fetch behavior

`fetch_url` is more than a raw HTTP GET:

- supports HTML, plain text, JSON, XML, and PDF
- blocks localhost and private‑network targets
- applies browser‑like headers and retry variants
- follows redirects with limits
- caps downloaded content size
- cleans extracted text aggressively before sending it to the model
- marks partial‑content recoveries when a connection ends early but some text was recovered
- clips oversized page text before it enters the model context
- stores fetch diagnostics and cleanup metadata with assistant messages
- exposes “cleaned”, “clipped”, or “summarized” state back to the UI

### RAG / knowledge base

- Uses ChromaDB as a persistent vector store
- Uses BGE‑M3 embeddings through Sentence Transformers
- Indexes these source types:
  - conversation history
  - successful text‑like tool results
  - tool‑memory entries (past web/search results)
- Can auto‑inject retrieved context into each chat turn
- Can expose `search_knowledge_base` as a tool to the model
- Lets you list indexed sources in the UI
- Lets you delete indexed sources individually
- Syncs one conversation or all conversations into RAG via API
- Keeps conversation content, tool‑result content, and tool‑memory content in separate source categories
- Filters search results by similarity threshold

### Local vision / OCR

- Accepts `PNG`, `JPEG`, and `WEBP`
- Enforces a 10 MB upload limit
- Validates MIME type and file presence
- Optimizes and optionally resizes images before local inference
- Runs a local Qwen2.5‑VL pipeline if configured
- Produces:
  - `ocr_text`
  - `vision_summary`
  - `key_points`
  - `assistant_guidance`
- Injects that vision context into the actual user message sent to the chat model
- Shows attachment badges and a compact vision note in the UI

### Scratchpad (persistent memory)

- Provides a persistent, user‑specific text area that the model can read and update
- Two dedicated tools: `append_scratchpad` (adds one durable fact) and `replace_scratchpad` (rewrites the whole scratchpad)
- Scratchpad content is included in the runtime system prompt for every turn
- Deduplication prevents duplicate entries
- Admin‑editing toggle (`SCRATCHPAD_ADMIN_EDITING_ENABLED`) allows manual editing via the UI
- Useful for storing long‑term user preferences, constraints, or facts that should be remembered across conversations

### Tool memory (web‑result memory)

- Automatically stores successful web‑search, news‑search, and URL‑fetch results in a dedicated RAG category (`tool_memory`)
- `search_tool_memory` tool lets the model query past web results before repeating a request
- Auto‑injection can be enabled/disabled via the `tool_memory_auto_inject` setting
- Reduces redundant web calls and helps the model build on previously fetched information
- Each tool‑memory entry includes the original query/URL, a cleaned content snippet, and a short summary

### Canvas documents

- Enables the model to create, rewrite, and edit structured Markdown documents attached to the conversation
- Tools: `create_canvas_document`, `rewrite_canvas_document`, `replace_canvas_lines`, `insert_canvas_lines`, `delete_canvas_lines`, `delete_canvas_document`, `clear_canvas`
- Documents are stored in SQLite and can be edited collaboratively (user ↔ model)
- Useful for outlines, plans, articles, notes, or any long‑form artifact that should stay editable across turns

### Chat summarization

- When a conversation grows beyond a configurable token threshold (`CHAT_SUMMARY_TRIGGER_TOKEN_COUNT`), the app can automatically summarize older messages
- Summarization mode can be `auto` (default), `never`, or `aggressive`
- Summaries are stored as special system messages, preserving the original conversational flow while reducing token consumption
- Helps keep long conversations within the model’s context window without losing important context

### Observability in the UI

The Usage & Cost panel separates provider‑reported usage from local prompt‑source estimates and rebuilds its totals from stored assistant‑message metadata when you reopen a conversation.

- Header badge: shows the sum of provider‑reported `total_tokens` across completed assistant turns in the current conversation
- Provider totals (session): sums `prompt_tokens`, `completion_tokens`, `total_tokens`, and estimated USD cost across all completed assistant turns currently loaded for that conversation
- Provider totals (latest assistant turn): shows the provider totals and model name for the most recent completed assistant reply in the conversation
- Latest‑turn provider totals include every model call used to produce that assistant reply, including multi‑step tool rounds and the final answer call
- Estimated input sources (session and latest assistant turn): local prompt‑source estimates grouped into these fixed categories:
  - system prompt
  - user messages
  - assistant history
  - tool results
  - RAG context
  - final instruction
- Estimated input totals are inferred before each model call from the messages actually sent to the API, then accumulated across all calls in that assistant turn
- `RAG context` is estimated by isolating the `## Knowledge Base` section inside the runtime system prompt; the rest of that system message is counted as `system prompt`
- `tool results` includes both tool‑role transcript content and synthetic system instructions that contain tool execution results
- `final instruction` covers the extra system instructions used to force or recover a final answer after tool use
- Frontend normalization keeps the six‑category breakdown non‑negative and uses the greater of `estimated_input_tokens` and the summed breakdown categories as the displayed estimated‑input total for a turn
- Estimated input‑source totals are explanatory only and can differ from billed prompt tokens
- Estimated cost uses the current DeepSeek V3.2 pricing configured in the app: $0.28 per 1M input tokens and $0.42 per 1M output tokens on a cache‑miss basis; cache‑hit discounts are not applied because the API usage payload does not expose cache‑hit token counts
- Completed assistant turns list: one row per completed assistant reply with model, provider input/output/total tokens, optional per‑turn cost, and inline chips for every non‑zero estimated input‑source category
- Live tool‑trace panel for assistant messages
- Visual badge when fetched web content was cleaned, clipped, or summarized

## Architecture overview

1. The browser sends either JSON or `multipart/form‑data` to `/chat`.
2. The backend loads saved settings from SQLite.
3. If an image is attached, local vision analysis runs first.
4. If RAG auto‑inject is enabled, the latest user message is searched against ChromaDB.
5. If tool‑memory auto‑inject is enabled, the same query searches the tool‑memory collection.
6. A runtime system message is prepended with:
   - current date/time and timezone
   - user preferences
   - enabled tool context
   - tool‑call contract
   - auto‑injected RAG context, if any
   - auto‑injected tool‑memory context, if any
   - scratchpad content, if any
7. The agent streams model output.
8. If the model emits tool calls, they are validated, executed, cached, and appended to the transcript as tool messages.
9. Tool progress, reasoning deltas, answer deltas, usage, and message IDs are streamed back to the browser as NDJSON events.
10. The final assistant message is stored with metadata such as reasoning, usage, tool trace, and storable tool results.
11. Conversations, successful tool text, and web‑tool results can later be synced into the RAG store.

## Project structure

```text
.
├── app.py                  # Flask app factory and local entrypoint
├── agent.py                # Streaming agent loop, tool execution, usage tracking, trace logging
├── config.py               # Environment variables, defaults, model list, limits
├── db.py                   # SQLite schema, settings, cache, metadata serialization
├── messages.py             # Runtime prompt context and API‑message preparation
├── rag_service.py          # RAG sync/search orchestration for conversations and tool results
├── tool_registry.py        # Tool definitions and JSON schemas exposed to the model
├── vision.py               # Local Qwen2.5‑VL image analysis pipeline
├── web_tools.py            # Web/news search, proxy loading, safe URL fetch, content extraction
├── canvas_service.py       # Canvas document storage and line‑level editing
├── doc_service.py          # Document upload and text extraction (future extension)
├── token_utils.py          # Token counting and prompt‑source estimation
├── conversation_export.py  # Conversation export utilities
├── routes/
│   ├── chat.py             # /chat, /api/fix‑text, title generation, preload helpers
│   ├── conversations.py    # conversation CRUD and RAG endpoints
│   └── pages.py            # index page and settings endpoints
├── rag/
│   ├── chunker.py          # chunk splitting and chunk metadata
│   ├── embedder.py         # BGE‑M3 loading and embedding generation
│   ├── ingestor.py         # record‑to‑chunk conversion helpers
│   └── store.py            # ChromaDB collection/query/delete helpers
├── static/
│   ├── app.js              # full frontend app logic
│   └── style.css           # UI styles
├── templates/
│   └── index.html          # single‑page UI template
├── tests/
│   └── test_app.py         # backend, stream, tool, RAG, and frontend bootstrap tests
├── proxies.example.txt     # sample proxy file
├── requirements.txt        # runtime dependencies
├── requirements‑dev.txt    # runtime + dev tooling
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

### 3) GPU and CUDA requirements

The default dependency set assumes GPU‑backed RAG and local vision are available.

- RAG embeddings use Sentence Transformers with CUDA‑only loading in this codebase.
- Local vision/OCR uses Qwen2.5‑VL with Torch, Transformers, and optional 4‑bit quantization.
- If your machine does not have a supported NVIDIA GPU and working CUDA runtime, disable these features explicitly instead of leaving them enabled.

Example `.env` overrides for CPU‑only or lightweight setups:

```env
RAG_ENABLED=false
VISION_ENABLED=false
```

If `RAG_ENABLED=true` or `VISION_ENABLED=true` without the required CUDA stack, those features will fail at runtime.

### 4) Create `.env`

There is no `.env.example` in the repository. Create `.env` manually.

Minimum required:

```env
DEEPSEEK_API_KEY=your‑deepseek‑api‑key
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

`python app.py` also triggers optional preload hooks for the local vision model and the embedder. If you only import `create_app()`, those preload hooks are not run automatically.

## Configuration

Some settings come from environment variables, and some are stored in SQLite through the Settings panel.

### Required

| Variable | Description |
| --- | --- |
| `DEEPSEEK_API_KEY` | DeepSeek API key used by the OpenAI‑compatible client |

### General / logging

| Variable | Default | Description |
| --- | --- | --- |
| `AGENT_TRACE_LOG_PATH` | `logs/agent‑trace.log` | Rotating agent trace log file |

### RAG / embedding

| Variable | Default | Description |
| --- | --- | --- |
| `CHROMA_DB_PATH` | `./chroma_db` | ChromaDB persistence directory |
| `BGE_M3_MODEL_PATH` | `BAAI/bge‑m3` | Embedding model name or local path |
| `BGE_M3_DEVICE` | auto | Force device such as `cpu` or `cuda` |
| `BGE_M3_LOCAL_FILES_ONLY` | `false` | Load embedding model only from local files |
| `BGE_M3_TRUST_REMOTE_CODE` | `false` | Sentence Transformers remote‑code permission |
| `BGE_M3_BATCH_SIZE` | `32` | Embedding batch size |
| `BGE_M3_PRELOAD` | `true` | Preload embedder on startup |
| `RAG_AUTO_INJECT_TOP_K` | `5` | Auto‑inject candidate count |
| `RAG_SEARCH_DEFAULT_TOP_K` | `5` | Default knowledge‑base search size |
| `RAG_AUTO_INJECT_THRESHOLD` | `0.35` | Similarity threshold for auto‑injected context |
| `RAG_SEARCH_MIN_SIMILARITY` | `0.2` | Minimum similarity shown in RAG search results |
| `RAG_TOOL_RESULT_MAX_TEXT_CHARS` | `12000` | Maximum characters of tool‑result text stored in RAG |
| `RAG_TOOL_RESULT_SUMMARY_MAX_CHARS` | `300` | Maximum characters of tool‑result summary stored in RAG |
| `RAG_SENSITIVITY_PRESETS` | `{"flexible":0.20,"normal":0.35,"strict":0.55}` | Named sensitivity presets for UI dropdown |
| `RAG_CONTEXT_SIZE_PRESETS` | `{"small":2,"medium":5,"large":8}` | Named context‑size presets for UI dropdown |

### Local vision / OCR

| Variable | Default | Description |
| --- | --- | --- |
| `QWEN_VL_MODEL_PATH` | empty | Required local Qwen2.5‑VL model directory |
| `QWEN_VL_ATTENTION` | empty | Optional Transformers attention implementation |
| `QWEN_VL_LOAD_IN_4BIT` | `true` | Use 4‑bit loading when CUDA is available |
| `QWEN_VL_TORCH_DTYPE` | `float16` | `float16`, `bfloat16`, or `float32` |
| `QWEN_VL_MAX_NEW_TOKENS` | `768` | Generation limit for image analysis |
| `QWEN_VL_MIN_PIXELS` | `256 * 28 * 28` | Processor min‑pixels setting |
| `QWEN_VL_MAX_PIXELS` | `896 * 28 * 28` | Processor max‑pixels setting |
| `QWEN_VL_MAX_IMAGE_SIDE` | `1280` | Resize limit before local inference |
| `QWEN_VL_PRELOAD` | `true` | Preload the vision model on startup |

### Fetch / clipping behavior

| Variable | Default | Description |
| --- | --- | --- |
| `FETCH_SUMMARY_TOKEN_THRESHOLD` | `3500` | Token threshold before long fetched content is clipped |
| `FETCH_RAW_TOOL_RESULT_MAX_TEXT_CHARS` | `24000` | Max raw fetch text preserved in stored tool metadata |
| `FETCH_SUMMARY_MAX_CHARS` | `8000` | Maximum characters of generated fetch summary |
| `FETCH_SUMMARY_GENERAL_TOP_K` | `3` | Top‑K sentences for general summarization |
| `FETCH_SUMMARY_QUERY_TOP_K` | `4` | Top‑K sentences for query‑aware summarization |
| `FETCH_SUMMARY_EXCERPT_MAX_CHARS` | `500` | Maximum excerpt length in summaries |

### Chat summarization

| Variable | Default | Description |
| --- | --- | --- |
| `CHAT_SUMMARY_TRIGGER_TOKEN_COUNT` | `6000` | Token count that triggers automatic summarization |
| `CHAT_SUMMARY_BATCH_SIZE` | `20` | Number of messages to summarize at once |
| `CHAT_SUMMARY_MODE` | `auto` | `auto`, `never`, or `aggressive` |
| `CHAT_SUMMARY_MODEL` | `deepseek‑chat` | Model used for summarization (must be available) |

### Scratchpad administration

| Variable | Default | Description |
| --- | --- | --- |
| `SCRATCHPAD_ADMIN_EDITING_ENABLED` | `false` | Enable manual scratchpad editing in the UI (admin feature) |
| `MAX_USER_PREFERENCES_LENGTH` | `2000` | Maximum length of user preferences text |
| `MAX_SCRATCHPAD_LENGTH` | `4000` | Maximum length of scratchpad text |

### Built‑in runtime limits from code

- fetch timeout: **20 seconds**
- max fetch size: **5 MB**
- max redirects: **5**
- web cache TTL: **24 hours**
- max search/news results per query: **5**
- supported image types: **PNG / JPEG / WEBP**
- max upload image size: **10 MB**
- document upload max size: **20 MB**
- document max text characters: **50 000**

### Settings stored in SQLite via the UI

The Settings panel persists these values in `app_settings`:

- `user_preferences`
- `scratchpad`
- `max_steps`
- `active_tools`
- `rag_auto_inject`
- `rag_sensitivity` (flexible / normal / strict)
- `rag_context_size` (small / medium / large)
- `tool_memory_auto_inject`
- `fetch_url_token_threshold`
- `fetch_url_clip_aggressiveness`
- `chat_summary_trigger_token_count`
- `chat_summary_batch_size`
- `chat_summary_mode`

## Using the app

### Basic chat flow

1. Open the app.
2. Pick a model.
3. Type a message.
4. Optionally click `Fix` to rewrite your draft before sending.
5. Press `Enter` to send, or `Shift+Enter` for a new line.
6. Watch tool progress, reasoning, and answer text stream live.

### Reading the Token Usage panel

Use the panel as two parallel views of the same conversation:

- Provider‑reported usage: what the model API returned for completed assistant turns
- Local input‑source estimates: what the app inferred about how each prompt was assembled before those API calls were made

Section by section:

- Header badge: cumulative provider `total_tokens` for completed assistant turns in the current conversation
- `Provider totals (session)`: conversation‑level sum of provider `prompt_tokens`, `completion_tokens`, `total_tokens`, and estimated cost
- `Provider totals (latest assistant turn)`: the most recent completed assistant reply only, including all model calls used during that reply if tools were involved
- `Estimated input sources (session)`: conversation‑level sum of the six local breakdown categories across completed assistant turns
- `Estimated input sources (latest assistant turn)`: the local breakdown for the most recent completed assistant reply only
- `Completed assistant turns`: one row per completed assistant reply in order, with the model, provider token totals, optional per‑turn cost, and non‑zero local breakdown chips

Important interpretation details:

- A single assistant turn can contain multiple model calls, so one row can already include several prompt/response cycles
- The panel is rebuilt from stored assistant‑message usage metadata when conversation history is reloaded, so totals remain available after refresh or reopening a conversation
- Zero‑value breakdown categories are hidden in the lists and per‑turn chips
- Local estimated input totals are meant to explain prompt composition, not to reproduce billing exactly
- Estimated cost uses DeepSeek V3.2 cache‑miss input pricing because the API usage payload does not reveal cache‑hit prompt‑token counts

### Editing a previous user message

When you click `Edit` on a stored user message:

1. the old user message is loaded back into the input,
2. sending updates that message in the database,
3. all later messages in that conversation are deleted,
4. generation restarts from that branch.

### Image‑assisted messages

If you attach an image:

1. the frontend validates file type and size,
2. the backend re‑validates and reads the upload,
3. the image is optimized locally,
4. Qwen2.5‑VL extracts OCR and visual context,
5. that context is injected into the user message before the main model call.

### Scratchpad workflow

- The scratchpad is a persistent text area visible in the Settings panel (if `SCRATCHPAD_ADMIN_EDITING_ENABLED` is true).
- The model can add entries via `append_scratchpad` or replace the entire content via `replace_scratchpad`.
- The scratchpad is included in the system prompt for every turn, so the model always sees its current content.
- Use the scratchpad for long‑term user facts, preferences, or constraints that should be remembered across conversations.

### Tool memory workflow

- When a web search, news search, or URL fetch returns useful content, the result is automatically stored in the tool‑memory RAG collection (if RAG is enabled).
- Enable `tool_memory_auto_inject` in Settings to have relevant past results automatically injected into the system prompt.
- The model can also explicitly search tool memory with the `search_tool_memory` tool.
- This reduces redundant web requests and helps the model build on previously fetched information.

### Canvas documents workflow

- The model can create a canvas document with `create_canvas_document` (title + Markdown content).
- Existing documents can be rewritten (`rewrite_canvas_document`), edited line‑by‑line (`replace_canvas_lines`, `insert_canvas_lines`, `delete_canvas_lines`), or deleted (`delete_canvas_document`, `clear_canvas`).
- Canvas documents are attached to the current conversation and are stored in SQLite.
- The UI shows a collapsible canvas panel with the document’s content, allowing the user to view and manually edit it as well.

### Chat summarization workflow

- When the conversation grows long, the app may automatically summarize older messages to stay within the model’s context window.
- The trigger token count (`CHAT_SUMMARY_TRIGGER_TOKEN_COUNT`) and summarization mode (`chat_summary_mode`) control this behavior.
- Summaries appear as special system messages in the conversation history, preserving the original flow while reducing token consumption.
- You can disable summarization by setting `chat_summary_mode` to `never`.

### Knowledge‑base workflow

RAG is not a generic file‑ingestion system in the current codebase.

What is supported:

- sync existing conversations into RAG
- sync successful text‑like tool results into RAG
- sync tool‑memory entries (past web results) into RAG
- search the knowledge base from the API
- enable auto‑injection per chat turn
- delete indexed sources one by one

What is intentionally disabled:

- manual `/api/rag/ingest` document ingestion

## Available tools

Only tools enabled in Settings are exposed to the model.

### Memory & personalization

#### `append_scratchpad`
Append one durable user‑specific fact or preference to the persistent scratchpad.

- **Arguments**:
  - `note` (string, required) – One short durable note to append.

#### `replace_scratchpad`
Completely replace the persistent scratchpad content.

- **Arguments**:
  - `new_content` (string, required) – The new content that will fully replace the existing scratchpad.

#### `ask_clarifying_question`
Ask the user one or more structured clarification questions and stop answering until they reply.

- **Arguments**:
  - `questions` (array, required) – List of 1‑5 clarification questions, each with `id`, `label`, `input_type`, etc.
  - `intro` (string, optional) – Short lead‑in shown before the questions.
  - `submit_label` (string, optional) – Optional button label shown in the UI.

#### `image_explain`
Answer a follow‑up question about a previously uploaded image saved in the current conversation.

- **Arguments**:
  - `image_id` (string, required) – The globally unique stored image_id for the referenced uploaded image.
  - `conversation_id` (integer, required) – The current conversation id used to verify that the image belongs to this chat.
  - `question` (string, required) – A focused follow‑up question about the image (written in English).

### Knowledge base & tool memory

#### `search_knowledge_base`
Semantic search over synced conversations and stored tool results.

- **Arguments**:
  - `query` (string, required) – Semantic search query for the knowledge base.
  - `category` (string, optional) – Optional category filter: `conversation` or `tool_result`.
  - `top_k` (integer, optional, 1‑12) – Maximum number of chunks to retrieve.

#### `search_tool_memory`
Search past web tool results stored from previous conversations.

- **Arguments**:
  - `query` (string, required) – Semantic search query for past web tool results.
  - `top_k` (integer, optional, 1‑10) – Maximum number of remembered results to retrieve.

### Web search & browsing

#### `search_web`
DuckDuckGo text search.

- **Arguments**:
  - `queries` (array, required, 1‑5 strings) – List of search queries to run.

#### `fetch_url`
Fetch and read the content of a specific web page. Returns cleaned text and metadata.

- **Arguments**:
  - `url` (string, required) – Full URL of the page (must start with `http://` or `https://`).

### News search

#### `search_news_ddgs`
DuckDuckGo News search.

- **Arguments**:
  - `queries` (array, required, 1‑5 strings) – List of news search queries.
  - `lang` (string, optional) – Search language/region: `tr` (Turkish) or `en` (English).
  - `when` (string, optional) – Time filter: `d` (last day), `w` (last week), `m` (last month), `y` (last year).

#### `search_news_google`
Google News RSS search.

- **Arguments**:
  - `queries` (array, required, 1‑5 strings) – List of news search queries.
  - `lang` (string, optional) – Search language/region: `tr` (Turkish) or `en` (English).
  - `when` (string, optional) – Time filter: `d`, `w`, `m`, `y`.

### Canvas document editing

#### `create_canvas_document`
Create or replace the active canvas document for the current conversation.

- **Arguments**:
  - `title` (string, required) – Document title shown in the canvas panel.
  - `content` (string, required) – Full Markdown content for the new canvas document.
  - `format` (string, optional) – Canvas document format (currently only `markdown`).

#### `rewrite_canvas_document`
Rewrite the full active canvas document while keeping the same document id.

- **Arguments**:
  - `content` (string, required) – The full replacement Markdown content.
  - `title` (string, optional) – Optional replacement title.
  - `document_id` (string, optional) – Optional target canvas document id (defaults to the active document).

#### `replace_canvas_lines`
Replace a 1‑based inclusive line range inside the active canvas document.

- **Arguments**:
  - `start_line` (integer, required) – 1‑based first line to replace.
  - `end_line` (integer, required) – 1‑based last line to replace.
  - `lines` (array, required) – Replacement lines without trailing newline characters.
  - `document_id` (string, optional) – Optional target canvas document id.

#### `insert_canvas_lines`
Insert one or more lines into the active canvas document after a given line number.

- **Arguments**:
  - `after_line` (integer, required) – Insert after this 1‑based line (0 to insert at the top).
  - `lines` (array, required) – New lines without trailing newline characters.
  - `document_id` (string, optional) – Optional target canvas document id.

#### `delete_canvas_lines`
Delete a 1‑based inclusive line range from the active canvas document.

- **Arguments**:
  - `start_line` (integer, required) – 1‑based first line to delete.
  - `end_line` (integer, required) – 1‑based last line to delete.
  - `document_id` (string, optional) – Optional target canvas document id.

#### `delete_canvas_document`
Delete a canvas document. Defaults to the active document when document_id is omitted.

- **Arguments**:
  - `document_id` (string, optional) – Optional target canvas document id.

#### `clear_canvas`
Delete all canvas documents for the current conversation.

- **Arguments**: none.

## RAG behavior

- Chunks are created from stored conversation records, stored tool results, and tool‑memory entries.
- Conversation sources, tool‑result sources, and tool‑memory sources are indexed separately.
- Chunks are normalized and deduplicated before storage.
- Similarity search uses cosine space in ChromaDB.
- Search results include:
  - source key
  - source name
  - source type
  - category
  - chunk index
  - similarity
  - clipped excerpt
- Auto‑injected context is added to the runtime system message, not directly appended as a visible user message.

## Vision behavior

- Vision is optional, but if enabled it is fully local.
- The app expects `QWEN_VL_MODEL_PATH` to point to an existing local directory.
- If the model path is missing, vision requests fail with a clear runtime error.
- The image‑analysis prompt asks the vision model to return strict JSON.
- If the vision model returns malformed output, the backend normalizes it into a safe fallback structure.
- OCR/summary/guidance are stored in message metadata and shown again when the conversation is reopened.

## HTTP endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/` | Main single‑page UI |
| `GET` | `/api/settings` | Read persisted app settings |
| `PATCH` | `/api/settings` | Update persisted settings |
| `POST` | `/api/fix‑text` | Rewrite the current input before sending |
| `POST` | `/chat` | Main streamed chat endpoint; accepts JSON or multipart for image uploads |
| `GET` | `/api/conversations` | List conversations |
| `POST` | `/api/conversations` | Create a conversation |
| `GET` | `/api/conversations/<id>` | Load one conversation and all messages |
| `PATCH` | `/api/conversations/<id>` | Rename a conversation |
| `DELETE` | `/api/conversations/<id>` | Delete a conversation |
| `POST` | `/api/conversations/<id>/generate‑title` | Generate a short title from the first turn |
| `GET` | `/api/rag/documents` | List indexed RAG sources |
| `DELETE` | `/api/rag/documents/<source_key>` | Delete one indexed RAG source |
| `GET` | `/api/rag/search?q=...` | Search the knowledge base |
| `POST` | `/api/rag/sync‑conversations` | Sync one or all conversations into RAG |
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
- agent logs default to `logs/agent‑trace.log`
- optional proxies are loaded from `proxies.txt`
- uploaded images are stored in `data/images/` (subdirectories by hash prefix)
- uploaded documents are stored in `data/documents/` (subdirectories by hash prefix)

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

app = create_app(database_path="/tmp/chatbot‑test.db")
```

That is how the test suite keeps databases isolated.

### Pre‑commit hooks (optional)

If you have pre‑commit installed, you can add a hook to run Ruff formatting and linting before each commit:

```bash
pre‑commit install
```

A sample `.pre‑commit‑config.yaml` is not included in the repository but can be added easily.

## Security and operational notes

- `fetch_url` rejects localhost and private‑network targets
- only enabled tools are exposed to the model
- tool arguments are schema‑validated before execution
- uploaded images are MIME‑checked and size‑limited
- assistant Markdown is sanitized in the browser
- repeated web/tool calls benefit from caching
- manual RAG ingestion is disabled to keep indexed sources limited to internal conversation/tool data
- the app stores conversation content locally; plan storage and backups accordingly
- local vision requires a local model download and enough hardware to run it
- scratchpad admin editing is disabled by default; enable only if you trust the UI users

## Troubleshooting

**RAG or vision fails with CUDA errors**

Set `RAG_ENABLED=false` and/or `VISION_ENABLED=false` in `.env` to disable GPU‑dependent features. Alternatively, ensure you have a compatible NVIDIA GPU, CUDA toolkit, and PyTorch with CUDA support installed.

**Proxy rotation not working**

Ensure `proxies.txt` exists and contains valid proxy entries (one per line). The file is read on startup; changes require a restart.

**Tool memory auto‑injection not happening**

Check that `tool_memory_auto_inject` is enabled in Settings and that RAG is enabled (`RAG_ENABLED=true`). Tool‑memory entries are only created for successful web‑tool results.

**Chat summarization seems too aggressive**

Adjust `CHAT_SUMMARY_TRIGGER_TOKEN_COUNT` (increase) or set `chat_summary_mode` to `never` in Settings.

**Scratchpad not visible in the UI**

The scratchpad UI is only shown when `SCRATCHPAD_ADMIN_EDITING_ENABLED` is `true`. Otherwise, the scratchpad is still accessible to the model but not editable manually.

**Canvas documents not appearing**

Canvas documents are attached to the current conversation. Ensure you are in the same conversation where the document was created. The canvas panel may be collapsed; look for the “Canvas” toggle in the UI.

## Frequently Asked Questions

**How do I reset the database?**
Delete `chatbot.db` and restart the app. The schema will be recreated automatically.

**How can I clear the RAG vector store?**
Delete the `chroma_db` directory (or the path set in `CHROMA_DB_PATH`) and restart. All indexed content will be lost.

**Why is the vision model not loading?**
Ensure `QWEN_VL_MODEL_PATH` points to a local directory containing the Qwen2.5‑VL model files. The model must be downloaded from Hugging Face separately.

**How do I enable debug logging?**
Set `AGENT_TRACE_LOG_PATH` to a writable file path and check the logs for detailed agent traces.

**Can I use a different embedding model?**
Change `BGE_M3_MODEL_PATH` to another Sentence Transformers model name or local path. Note that the chunking and retrieval logic is tuned for BGE‑M3.

**How can I disable a specific tool?**
Use the Settings panel to uncheck the tool from the “Active tools” list. The tool will no longer be exposed to the model.

## License

MIT
