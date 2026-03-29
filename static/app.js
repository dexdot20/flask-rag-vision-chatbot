const bootstrapEl = document.getElementById("app-bootstrap");
const bootstrapData = (() => {
  if (!bootstrapEl) {
    return { settings: {} };
  }
  try {
    return JSON.parse(bootstrapEl.textContent || "{}") || { settings: {} };
  } catch (_) {
    return { settings: {} };
  }
})();

const messagesEl = document.getElementById("messages");
const inputEl = document.getElementById("user-input");
const imageInputEl = document.getElementById("image-input");
const docInputEl = document.getElementById("doc-input");
const attachBtn = document.getElementById("attach-btn");
const attachmentPreviewEl = document.getElementById("attachment-preview");
const summaryNowBtn = document.getElementById("summary-now-btn");
const summaryUndoBtn = document.getElementById("summary-undo-btn");
const kbSyncBtn = document.getElementById("kb-sync-btn");
const kbStatusEl = document.getElementById("kb-status");
const kbDocumentsListEl = document.getElementById("kb-documents-list");
const cancelBtn = document.getElementById("cancel-btn");
const fixBtn = document.getElementById("fix-btn");
const sendBtn = document.getElementById("send-btn");
const modelSel = document.getElementById("model-select");
const mobileModelSel = document.getElementById("mobile-model-select");
const emptyState = document.getElementById("empty-state");
const errorArea = document.getElementById("error-area");
const editBanner = document.getElementById("edit-banner");
const editBannerText = document.getElementById("edit-banner-text");
const editBannerCancelBtn = document.getElementById("edit-banner-cancel");
const summaryInspectorBadge = document.getElementById("summary-inspector-badge");
const summaryInspectorHeadline = document.getElementById("summary-inspector-headline");
const summaryInspectorCurrent = document.getElementById("summary-inspector-current");
const summaryInspectorTrigger = document.getElementById("summary-inspector-trigger");
const summaryInspectorGap = document.getElementById("summary-inspector-gap");
const summaryInspectorDetail = document.getElementById("summary-inspector-detail");
const summaryInspectorToolMessages = document.getElementById("summary-inspector-tool-messages");
const summaryInspectorReason = document.getElementById("summary-inspector-reason");
const summaryInspectorLast = document.getElementById("summary-inspector-last");
const canvasToggleBtn = document.getElementById("canvas-toggle-btn");
const canvasPanel = document.getElementById("canvas-panel");
const canvasOverlay = document.getElementById("canvas-overlay");
const canvasClose = document.getElementById("canvas-close");
const canvasSearchInput = document.getElementById("canvas-search-input");
const canvasSearchStatus = document.getElementById("canvas-search-status");
const canvasFormatSelect = document.getElementById("canvas-format-select");
const canvasRoleFilter = document.getElementById("canvas-role-filter");
const canvasPathFilter = document.getElementById("canvas-path-filter");
const canvasTreePanel = document.getElementById("canvas-tree-panel");
const canvasTreeCount = document.getElementById("canvas-tree-count");
const canvasTreeEl = document.getElementById("canvas-tree");
const canvasSubtitle = document.getElementById("canvas-subtitle");
const canvasStatus = document.getElementById("canvas-status");
const canvasHint = document.getElementById("canvas-hint");
const canvasDiffEl = document.getElementById("canvas-diff");
const canvasEmptyState = document.getElementById("canvas-empty-state");
const canvasEditorEl = document.getElementById("canvas-editor");
const canvasDocumentEl = document.getElementById("canvas-document");
const canvasDocumentTabsEl = document.getElementById("canvas-document-tabs");
const canvasMetaBar = document.getElementById("canvas-meta-bar");
const canvasMetaChips = document.getElementById("canvas-meta-chips");
const canvasCopyRefBtn = document.getElementById("canvas-copy-ref-btn");
const canvasResetFiltersBtn = document.getElementById("canvas-reset-filters-btn");
const canvasEditBtn = document.getElementById("canvas-edit-btn");
const canvasSaveBtn = document.getElementById("canvas-save-btn");
const canvasCancelBtn = document.getElementById("canvas-cancel-btn");
const canvasCopyBtn = document.getElementById("canvas-copy-btn");
const canvasDeleteBtn = document.getElementById("canvas-delete-btn");
const canvasClearBtn = document.getElementById("canvas-clear-btn");
const canvasDownloadHtmlBtn = document.getElementById("canvas-download-html-btn");
const canvasDownloadMdBtn = document.getElementById("canvas-download-md-btn");
const canvasDownloadPdfBtn = document.getElementById("canvas-download-pdf-btn");
const canvasResizeHandle = document.getElementById("canvas-resize-handle");
const canvasBtnIndicator = document.getElementById("canvas-btn-indicator");
const canvasConfirmModal = document.getElementById("canvas-confirm-modal");
const canvasConfirmOverlay = document.getElementById("canvas-confirm-overlay");
const canvasConfirmTitle = document.getElementById("canvas-confirm-title");
const canvasConfirmMessage = document.getElementById("canvas-confirm-message");
const canvasConfirmOpenBtn = document.getElementById("canvas-confirm-open");
const canvasConfirmLaterBtn = document.getElementById("canvas-confirm-later");
const canvasConfirmCloseBtn = document.getElementById("canvas-confirm-close");
const tokensBadge = document.getElementById("tokens-badge");
const statsPanel = document.getElementById("stats-panel");
const statsOverlay = document.getElementById("stats-overlay");
const statsClose = document.getElementById("stats-close");
const headerEl = document.querySelector("header");
const sidebarList = document.getElementById("sidebar-list");
const sidebarToggleBtn = document.getElementById("sidebar-toggle-btn");
const sidebarOverlay = document.getElementById("sidebar-overlay");
const newChatBtn = document.getElementById("new-chat-btn");
const mobileToolsBtn = document.getElementById("mobile-tools-btn");
const mobileToolsPanel = document.getElementById("mobile-tools-panel");
const mobileToolsOverlay = document.getElementById("mobile-tools-overlay");
const mobileToolsClose = document.getElementById("mobile-tools-close");
const mobileCanvasBtn = document.getElementById("mobile-canvas-btn");
const mobileExportBtn = document.getElementById("mobile-export-btn");
const mobilePruneBtn = document.getElementById("mobile-prune-btn");
const mobileSettingsBtn = document.getElementById("mobile-settings-btn");
const mobileLogoutBtn = document.getElementById("mobile-logout-btn");
const mobileTokensBtn = document.getElementById("mobile-tokens-btn");
const exportPanel = document.getElementById("export-panel");
const exportOverlay = document.getElementById("export-overlay");
const exportClose = document.getElementById("export-close");
const exportSubtitle = document.getElementById("export-subtitle");
const exportStatus = document.getElementById("export-status");
const conversationExportMdBtn = document.getElementById("conversation-export-md-btn");
const conversationExportDocxBtn = document.getElementById("conversation-export-docx-btn");
const conversationExportPdfBtn = document.getElementById("conversation-export-pdf-btn");

let history = [];
let isStreaming = false;
let isFixing = false;
let currentConvId = null;
let currentConvTitle = "New Chat";
let activeAbortController = null;
let selectedImageFiles = [];
let selectedDocumentFiles = [];
let pendingDocumentCanvasOpen = null;
let editingMessageId = null;
let activeCanvasDocumentId = null;
let streamingCanvasDocuments = [];
let isCanvasEditing = false;
let editingCanvasDocumentId = null;
let pendingCanvasDiff = null;
let canvasHasUnreadUpdates = false;
let lastCanvasTriggerEl = null;
let lastCanvasConfirmTriggerEl = null;
let lastExportTriggerEl = null;
let streamingCanvasPreview = null;
let pendingCanvasPreviewTimer = 0;
let lastCanvasStructureSignature = "";
let latestSummaryStatus = null;
let conversationRefreshGeneration = 0;
let pendingConversationRefreshTimers = new Set();
let lastConversationSignature = "";
let userScrolledUp = false;
let pendingCanvasConfirmAction = null;
let activeSidebarRename = null;
let collapsedCanvasFolders = new Set();
let lastCanvasTreeTypeAheadValue = "";
let lastCanvasTreeTypeAheadAt = 0;
const appSettings = bootstrapData.settings || {};
const featureFlags = bootstrapData.features || appSettings.features || {};
const MAX_IMAGE_BYTES = 10 * 1024 * 1024;
const ALLOWED_IMAGE_TYPES = new Set(["image/png", "image/jpeg", "image/webp"]);
const MAX_DOCUMENT_BYTES = 20 * 1024 * 1024;
const ALLOWED_DOCUMENT_TYPES = new Set([
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
  "application/pdf",
  "text/plain",
  "text/csv",
  "text/markdown",
]);
const DOCUMENT_EXTENSIONS = new Set([".docx", ".pdf", ".txt", ".csv", ".md"]);
const STREAM_TYPING_INTERVAL_MS = 24;
const STREAM_TYPING_MIN_STEP = 3;
const STREAM_TYPING_MAX_STEP = 48;
const CANVAS_PANEL_WIDTH_STORAGE_KEY = "chatbot.canvasPanelWidth";
const CANVAS_PANEL_DEFAULT_WIDTH = 620;
const CANVAS_PANEL_MIN_WIDTH = 420;
const CANVAS_PANEL_MAX_WIDTH = 1100;
const CANVAS_ROOT_PATH_FILTER = "__root__";
const CANVAS_PREVIEW_RENDER_INTERVAL_MS = 150;
const CANVAS_CODE_FILE_EXTENSIONS = new Set([
  ".bat",
  ".c",
  ".cc",
  ".cfg",
  ".conf",
  ".cpp",
  ".cs",
  ".css",
  ".env",
  ".go",
  ".h",
  ".hpp",
  ".html",
  ".ini",
  ".java",
  ".js",
  ".json",
  ".jsx",
  ".kt",
  ".kts",
  ".less",
  ".lua",
  ".mjs",
  ".php",
  ".ps1",
  ".py",
  ".rb",
  ".rs",
  ".sass",
  ".scss",
  ".sh",
  ".sql",
  ".swift",
  ".toml",
  ".ts",
  ".tsx",
  ".vue",
  ".xml",
  ".yaml",
  ".yml",
  ".zsh",
]);

function isDocumentFile(file) {
  if (ALLOWED_DOCUMENT_TYPES.has(file.type)) return true;
  const ext = (file.name || "").toLowerCase().match(/\.[^.]+$/);
  return ext ? DOCUMENT_EXTENSIONS.has(ext[0]) : false;
}

function getAttachmentFileKey(file) {
  return [file?.name || "", file?.size || 0, file?.type || "", file?.lastModified || 0].join("::");
}

function dedupeFiles(files) {
  const deduped = [];
  const seen = new Set();
  (files || []).forEach((file) => {
    if (!file) {
      return;
    }
    const key = getAttachmentFileKey(file);
    if (seen.has(key)) {
      return;
    }
    seen.add(key);
    deduped.push(file);
  });
  return deduped;
}

function normalizeMessageAttachment(entry) {
  if (!entry || typeof entry !== "object") {
    return null;
  }

  const kind = String(entry.kind || "").trim().toLowerCase();
  if (kind !== "image" && kind !== "document") {
    return null;
  }

  if (kind === "image") {
    const imageId = String(entry.image_id || "").trim();
    const imageName = String(entry.image_name || "").trim();
    if (!imageId && !imageName) {
      return null;
    }
    return {
      kind,
      image_id: imageId,
      image_name: imageName,
      image_mime_type: String(entry.image_mime_type || "").trim(),
      ocr_text: String(entry.ocr_text || "").trim(),
      vision_summary: String(entry.vision_summary || "").trim(),
      assistant_guidance: String(entry.assistant_guidance || "").trim(),
      key_points: Array.isArray(entry.key_points) ? entry.key_points.filter(Boolean).map((value) => String(value)) : [],
    };
  }

  const fileId = String(entry.file_id || "").trim();
  const fileName = String(entry.file_name || "").trim();
  if (!fileId && !fileName) {
    return null;
  }
  return {
    kind,
    file_id: fileId,
    file_name: fileName,
    file_mime_type: String(entry.file_mime_type || "").trim(),
    file_text_truncated: entry.file_text_truncated === true,
    file_context_block: String(entry.file_context_block || "").trim(),
  };
}

function getMessageAttachments(metadata) {
  if (!metadata || typeof metadata !== "object") {
    return [];
  }

  const attachments = [];
  const seen = new Set();

  const appendAttachment = (entry) => {
    const normalized = normalizeMessageAttachment(entry);
    if (!normalized) {
      return;
    }
    const dedupeKey = normalized.kind === "image"
      ? `image:${normalized.image_id || normalized.image_name}`
      : `document:${normalized.file_id || normalized.file_name}`;
    if (seen.has(dedupeKey)) {
      return;
    }
    seen.add(dedupeKey);
    attachments.push(normalized);
  };

  if (Array.isArray(metadata.attachments)) {
    metadata.attachments.forEach((entry) => appendAttachment(entry));
  }
  appendAttachment({
    kind: "image",
    image_id: metadata.image_id,
    image_name: metadata.image_name,
    image_mime_type: metadata.image_mime_type,
    ocr_text: metadata.ocr_text,
    vision_summary: metadata.vision_summary,
    assistant_guidance: metadata.assistant_guidance,
    key_points: metadata.key_points,
  });
  appendAttachment({
    kind: "document",
    file_id: metadata.file_id,
    file_name: metadata.file_name,
    file_mime_type: metadata.file_mime_type,
    file_text_truncated: metadata.file_text_truncated === true,
    file_context_block: metadata.file_context_block,
  });

  return attachments;
}

function buildLegacyAttachmentMetadata(attachments) {
  const legacy = {};
  const primaryImage = (attachments || []).find((entry) => entry.kind === "image") || null;
  const primaryDocument = (attachments || []).find((entry) => entry.kind === "document") || null;

  if (primaryImage) {
    if (primaryImage.image_id) legacy.image_id = primaryImage.image_id;
    if (primaryImage.image_name) legacy.image_name = primaryImage.image_name;
    if (primaryImage.image_mime_type) legacy.image_mime_type = primaryImage.image_mime_type;
    if (primaryImage.ocr_text) legacy.ocr_text = primaryImage.ocr_text;
    if (primaryImage.vision_summary) legacy.vision_summary = primaryImage.vision_summary;
    if (primaryImage.assistant_guidance) legacy.assistant_guidance = primaryImage.assistant_guidance;
    if (primaryImage.key_points?.length) legacy.key_points = [...primaryImage.key_points];
  }

  if (primaryDocument) {
    if (primaryDocument.file_id) legacy.file_id = primaryDocument.file_id;
    if (primaryDocument.file_name) legacy.file_name = primaryDocument.file_name;
    if (primaryDocument.file_mime_type) legacy.file_mime_type = primaryDocument.file_mime_type;
    if (primaryDocument.file_text_truncated) legacy.file_text_truncated = true;
  }

  const contextBlocks = (attachments || [])
    .filter((entry) => entry.kind === "document" && entry.file_context_block)
    .map((entry) => entry.file_context_block);
  if (contextBlocks.length) {
    legacy.file_context_block = contextBlocks.join("\n\n");
  }

  return legacy;
}

function mergeAttachmentMetadata(metadata, attachment) {
  const base = metadata && typeof metadata === "object" ? { ...metadata } : {};
  const blockedKeys = [
    "attachments",
    "image_id",
    "image_name",
    "image_mime_type",
    "ocr_text",
    "vision_summary",
    "assistant_guidance",
    "key_points",
    "file_id",
    "file_name",
    "file_mime_type",
    "file_text_truncated",
    "file_context_block",
  ];
  blockedKeys.forEach((key) => delete base[key]);

  const attachments = getMessageAttachments(metadata);
  const normalized = normalizeMessageAttachment(attachment);
  const nextAttachments = normalized
    ? [...attachments.filter((entry) => {
        if (normalized.kind === "image") {
          return !(entry.kind === "image" && (entry.image_id || entry.image_name) === (normalized.image_id || normalized.image_name));
        }
        return !(entry.kind === "document" && (entry.file_id || entry.file_name) === (normalized.file_id || normalized.file_name));
      }), normalized]
    : attachments;

  return {
    ...base,
    ...(nextAttachments.length ? { attachments: nextAttachments } : {}),
    ...buildLegacyAttachmentMetadata(nextAttachments),
  };
}

function buildPendingAttachmentMetadata(imageFiles, documentFiles) {
  const attachments = [
    ...(imageFiles || []).map((file) => ({ kind: "image", image_name: file.name })),
    ...(documentFiles || []).map((file) => ({ kind: "document", file_name: file.name })),
  ];
  return attachments.length
    ? {
        attachments,
        ...buildLegacyAttachmentMetadata(getMessageAttachments({ attachments })),
      }
    : null;
}
const ragDisabledNoteEl = document.getElementById("rag-disabled-note");
const visionDisabledNoteEl = document.getElementById("vision-disabled-note");
const RAG_SENSITIVITY_HINTS = {
  flexible: "Flexible: lower threshold around 0.20, so the system injects broader matches.",
  normal: "Normal: balanced matching with an approximate threshold of 0.35.",
  strict: "Strict: higher threshold around 0.55, so only stronger matches are injected.",
};

const markdownEngine = globalThis.marked || null;
const sanitizer = globalThis.DOMPurify || null;
const highlighter = globalThis.hljs || null;
const SIDEBAR_STORAGE_KEY = "chatbot.sidebarOpen";
const CANVAS_STREAMING_PREVIEW_TOOLS = new Set(["create_canvas_document", "rewrite_canvas_document"]);

function isCanvasStreamingPreviewTool(toolName) {
  return CANVAS_STREAMING_PREVIEW_TOOLS.has(String(toolName || "").trim());
}

function normalizeCanvasDocument(document) {
  if (!document || typeof document !== "object") {
    return null;
  }
  const format = String(document.format || "markdown").trim().toLowerCase();
  const normalizedFormat = format === "code" ? "code" : "markdown";
  const content = String(document.content || "").replace(/\r\n?/g, "\n");
  return {
    id: String(document.id || "").trim(),
    title: String(document.title || "Canvas").trim() || "Canvas",
    path: String(document.path || "").trim().replace(/\\/g, "/"),
    role: String(document.role || "").trim().toLowerCase(),
    summary: String(document.summary || "").trim(),
    format: normalizedFormat,
    language: String(document.language || "").trim().toLowerCase(),
    content,
    line_count: Number.isInteger(Number(document.line_count)) ? Number(document.line_count) : content.split("\n").length,
    source_message_id: Number.isInteger(Number(document.source_message_id)) ? Number(document.source_message_id) : null,
  };
}

function getCanvasMode(documents) {
  return Array.isArray(documents) && documents.some((document) => document.path || document.role) ? "project" : "document";
}

function getCanvasPreferredActiveDocumentId(entries = history) {
  for (let index = entries.length - 1; index >= 0; index -= 1) {
    const metadata = entries[index]?.metadata;
    const candidate = typeof metadata?.active_document_id === "string"
      ? metadata.active_document_id.trim()
      : "";
    if (candidate) {
      return candidate;
    }
  }
  return "";
}

function getCanvasDocumentLabel(document) {
  if (!document) {
    return "";
  }
  return String(document.path || document.title || "").trim();
}

function getCanvasDocumentReference(document) {
  return getCanvasDocumentLabel(document);
}

function getCanvasDocumentDisplayName(document) {
  return getCanvasDocumentReference(document) || String(document?.title || "Canvas").trim() || "Canvas";
}

function getCanvasFileName(document) {
  const label = getCanvasDocumentLabel(document);
  const parts = label.split("/");
  return parts[parts.length - 1] || label;
}

function shouldRenderCanvasAsCode(document) {
  if (!document || typeof document !== "object") {
    return false;
  }

  const explicitFormat = String(document.format || "").trim().toLowerCase();
  if (explicitFormat === "code") {
    return true;
  }

  const language = String(document.language || "").trim().toLowerCase();
  if (language && !["markdown", "md", "plain", "text", "txt"].includes(language)) {
    return true;
  }

  const candidateLabel = String(document.path || document.title || "").trim().toLowerCase();
  const extensionMatch = candidateLabel.match(/\.[^.\/]+$/);
  return Boolean(extensionMatch && CANVAS_CODE_FILE_EXTENSIONS.has(extensionMatch[0]));
}

function normalizeStreamingCanvasPreviewDocument(document) {
  const normalized = normalizeCanvasDocument(document);
  if (!normalized) {
    return null;
  }
  if (shouldRenderCanvasAsCode(normalized)) {
    normalized.format = "code";
  }
  return normalized;
}

function getCanvasPathFilterValue() {
  return String(canvasPathFilter?.value || "").trim();
}

function resetCanvasWorkspaceState() {
  isCanvasEditing = false;
  editingCanvasDocumentId = null;
  pendingCanvasDiff = null;
  resetStreamingCanvasPreview();
  lastCanvasStructureSignature = "";
  collapsedCanvasFolders = new Set();
  lastCanvasTreeTypeAheadValue = "";
  lastCanvasTreeTypeAheadAt = 0;
  setCanvasAttention(false);
  setCanvasSearchStatus("");
  setCanvasStatus("Canvas idle", "muted");
  if (canvasSearchInput) {
    canvasSearchInput.value = "";
  }
  if (canvasRoleFilter) {
    canvasRoleFilter.value = "";
  }
  if (canvasPathFilter) {
    canvasPathFilter.value = "";
  }
}

function hasActiveCanvasFilters() {
  return Boolean(
    String(canvasSearchInput?.value || "").trim()
    || String(canvasRoleFilter?.value || "").trim()
    || getCanvasPathFilterValue()
  );
}

function resetCanvasMetaBar() {
  if (canvasMetaBar) {
    canvasMetaBar.hidden = true;
  }
  if (canvasMetaChips) {
    canvasMetaChips.innerHTML = "";
  }
  if (canvasCopyRefBtn) {
    canvasCopyRefBtn.disabled = true;
    canvasCopyRefBtn.textContent = "Copy reference";
  }
  if (canvasResetFiltersBtn) {
    canvasResetFiltersBtn.disabled = true;
  }
}

function resetCanvasFilters({ silent = false } = {}) {
  if (canvasSearchInput) {
    canvasSearchInput.value = "";
  }
  if (canvasRoleFilter) {
    canvasRoleFilter.value = "";
  }
  if (canvasPathFilter) {
    canvasPathFilter.value = "";
  }
  renderCanvasPanel();
  if (!silent) {
    setCanvasSearchStatus("Canvas filters cleared.", "muted");
  }
}

function documentMatchesCanvasFilters(document, searchTerm, roleValue, pathValue) {
  if (!document) {
    return false;
  }

  if (document.isStreamingPreview) {
    return true;
  }

  const normalizedRole = String(roleValue || "").trim().toLowerCase();
  const normalizedPath = String(pathValue || "").trim();
  const normalizedSearch = String(searchTerm || "").trim().toLowerCase();

  if (normalizedRole && document.role !== normalizedRole) {
    return false;
  }

  if (normalizedPath === CANVAS_ROOT_PATH_FILTER) {
    if ((document.path || "").includes("/")) {
      return false;
    }
  } else if (normalizedPath) {
    const candidatePath = getCanvasDocumentLabel(document);
    if (!(candidatePath === normalizedPath || candidatePath.startsWith(`${normalizedPath}/`))) {
      return false;
    }
  }

  if (!normalizedSearch) {
    return true;
  }

  const haystack = [document.title, document.path, document.role, document.summary, document.content]
    .filter(Boolean)
    .join("\n")
    .toLowerCase();
  return haystack.includes(normalizedSearch);
}

function getCanvasVisibleDocuments(documents) {
  const searchTerm = String(canvasSearchInput?.value || "").trim();
  const roleValue = String(canvasRoleFilter?.value || "").trim();
  const pathValue = getCanvasPathFilterValue();
  return (documents || []).filter((document) => documentMatchesCanvasFilters(document, searchTerm, roleValue, pathValue));
}

function buildCanvasPathFilterOptions(documents) {
  const options = [{ value: "", label: "All paths" }];
  const seen = new Set([""]);
  let hasRootFile = false;

  (documents || []).forEach((document) => {
    const path = String(document.path || "").trim();
    if (!path || !path.includes("/")) {
      hasRootFile = true;
      return;
    }

    const parts = path.split("/");
    let prefix = "";
    parts.slice(0, -1).forEach((part) => {
      prefix = prefix ? `${prefix}/${part}` : part;
      if (!seen.has(prefix)) {
        seen.add(prefix);
        options.push({ value: prefix, label: prefix });
      }
    });
  });

  if (hasRootFile) {
    options.push({ value: CANVAS_ROOT_PATH_FILTER, label: "Root files" });
  }

  return options;
}

function syncCanvasFilterControls(documents) {
  if (canvasRoleFilter) {
    const currentValue = String(canvasRoleFilter.value || "").trim();
    const roles = Array.from(new Set((documents || []).map((document) => document.role).filter(Boolean))).sort();
    canvasRoleFilter.innerHTML = '<option value="">All roles</option>' + roles.map((role) => `<option value="${escHtml(role)}">${escHtml(role)}</option>`).join("");
    canvasRoleFilter.value = roles.includes(currentValue) ? currentValue : "";
  }

  if (canvasPathFilter) {
    const currentValue = getCanvasPathFilterValue();
    const options = buildCanvasPathFilterOptions(documents);
    canvasPathFilter.innerHTML = options.map((option) => `<option value="${escHtml(option.value)}">${escHtml(option.label)}</option>`).join("");
    canvasPathFilter.value = options.some((option) => option.value === currentValue) ? currentValue : "";
  }
}

function buildCanvasTreeNodes(documents) {
  const root = { folders: new Map(), files: [] };

  (documents || []).forEach((document) => {
    const path = String(document.path || "").trim();
    if (!path || !path.includes("/")) {
      root.files.push({ name: getCanvasFileName(document), document });
      return;
    }

    const parts = path.split("/");
    let cursor = root;
    let prefix = "";
    parts.slice(0, -1).forEach((part) => {
      prefix = prefix ? `${prefix}/${part}` : part;
      if (!cursor.folders.has(part)) {
        cursor.folders.set(part, { name: part, path: prefix, folders: new Map(), files: [] });
      }
      cursor = cursor.folders.get(part);
    });

    cursor.files.push({ name: parts[parts.length - 1], document });
  });

  return root;
}

function getCanvasTreeItems() {
  if (!canvasTreeEl) {
    return [];
  }
  return Array.from(canvasTreeEl.querySelectorAll('[data-canvas-tree-item="true"]')).filter((item) => item instanceof HTMLElement && !item.hidden);
}

function syncCanvasTreeTabStops(preferredItem = null) {
  const items = getCanvasTreeItems().filter((item) => !item.disabled);
  if (!items.length) {
    return null;
  }

  const preferredActiveId = String(activeCanvasDocumentId || getCanvasPreferredActiveDocumentId() || "").trim();
  const nextItem = preferredItem instanceof HTMLElement
    ? preferredItem
    : items.find((item) => item.dataset.canvasDocumentId === preferredActiveId)
      || items[0];

  items.forEach((item) => {
    item.tabIndex = item === nextItem ? 0 : -1;
  });
  return nextItem;
}

function focusCanvasTreeItem(targetItem) {
  const nextItem = syncCanvasTreeTabStops(targetItem);
  if (nextItem && typeof nextItem.focus === "function") {
    nextItem.focus();
  }
  return nextItem;
}

function getCanvasTreeDocumentItem(documentId) {
  const targetId = String(documentId || "").trim();
  if (!targetId) {
    return null;
  }
  return getCanvasTreeItems().find((item) => item.dataset.canvasDocumentId === targetId) || null;
}

function getCanvasTreeFolderItem(folderPath) {
  const targetPath = String(folderPath || "").trim();
  if (!targetPath) {
    return null;
  }
  return getCanvasTreeItems().find((item) => item.dataset.canvasTreeFolder === "true" && item.dataset.folderPath === targetPath) || null;
}

function getCanvasTreeParentItem(treeItem) {
  if (!(treeItem instanceof HTMLElement)) {
    return null;
  }
  const parentGroup = treeItem.closest('[role="group"]');
  if (!(parentGroup instanceof HTMLElement)) {
    return null;
  }
  const parentSection = parentGroup.parentElement;
  if (!(parentSection instanceof HTMLElement)) {
    return null;
  }
  return parentSection.querySelector(':scope > [data-canvas-tree-folder="true"]');
}

function getCanvasTreeFirstChildItem(treeItem) {
  if (!(treeItem instanceof HTMLElement)) {
    return null;
  }
  const section = treeItem.closest('.canvas-tree-node');
  if (!(section instanceof HTMLElement)) {
    return null;
  }
  return section.querySelector(':scope > [role="group"] [data-canvas-tree-item="true"]');
}

function restoreCanvasTreeFocus({ documentId = "", folderPath = "", firstChild = false } = {}) {
  globalThis.requestAnimationFrame(() => {
    let targetItem = null;
    if (documentId) {
      targetItem = getCanvasTreeDocumentItem(documentId);
    } else if (folderPath) {
      targetItem = getCanvasTreeFolderItem(folderPath);
      if (firstChild) {
        targetItem = getCanvasTreeFirstChildItem(targetItem) || targetItem;
      }
    }
    focusCanvasTreeItem(targetItem);
  });
}

function setCanvasTreeFolderExpanded(folderPath, expanded = null, { focusTarget = "self" } = {}) {
  const normalizedPath = String(folderPath || "").trim();
  if (!normalizedPath) {
    return;
  }
  const isExpanded = !collapsedCanvasFolders.has(normalizedPath);
  const nextExpanded = typeof expanded === "boolean" ? expanded : !isExpanded;
  if (nextExpanded) {
    collapsedCanvasFolders.delete(normalizedPath);
  } else {
    collapsedCanvasFolders.add(normalizedPath);
  }
  renderCanvasPanel();
  restoreCanvasTreeFocus({ folderPath: normalizedPath, firstChild: focusTarget === "child" });
}

function handleCanvasTreeItemKeydown(event) {
  const currentItem = event.currentTarget instanceof HTMLElement ? event.currentTarget : null;
  if (!currentItem) {
    return;
  }

  const items = getCanvasTreeItems().filter((item) => !item.disabled);
  if (!items.length) {
    return;
  }

  const currentIndex = items.indexOf(currentItem);
  const folderPath = String(currentItem.dataset.folderPath || "").trim();
  const isFolder = currentItem.dataset.canvasTreeFolder === "true";
  const isExpanded = currentItem.getAttribute("aria-expanded") === "true";

  if (event.key === "ArrowDown") {
    event.preventDefault();
    focusCanvasTreeItem(items[Math.min(currentIndex + 1, items.length - 1)]);
    return;
  }
  if (event.key === "ArrowUp") {
    event.preventDefault();
    focusCanvasTreeItem(items[Math.max(currentIndex - 1, 0)]);
    return;
  }
  if (event.key === "Home") {
    event.preventDefault();
    focusCanvasTreeItem(items[0]);
    return;
  }
  if (event.key === "End") {
    event.preventDefault();
    focusCanvasTreeItem(items[items.length - 1]);
    return;
  }
  if (event.key === "ArrowRight") {
    if (isFolder && !isExpanded) {
      event.preventDefault();
      setCanvasTreeFolderExpanded(folderPath, true);
      return;
    }
    if (isFolder && isExpanded) {
      const firstChild = getCanvasTreeFirstChildItem(currentItem);
      if (firstChild) {
        event.preventDefault();
        focusCanvasTreeItem(firstChild);
      }
    }
    return;
  }
  if (event.key === "ArrowLeft") {
    if (isFolder && isExpanded) {
      event.preventDefault();
      setCanvasTreeFolderExpanded(folderPath, false);
      return;
    }
    const parentItem = getCanvasTreeParentItem(currentItem);
    if (parentItem) {
      event.preventDefault();
      focusCanvasTreeItem(parentItem);
    }
    return;
  }
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    currentItem.click();
    return;
  }

  const isTypeAheadKey = event.key.length === 1 && !event.altKey && !event.ctrlKey && !event.metaKey && /\S/.test(event.key);
  if (!isTypeAheadKey) {
    return;
  }

  const now = Date.now();
  const resetWindowMs = 700;
  lastCanvasTreeTypeAheadValue = now - lastCanvasTreeTypeAheadAt > resetWindowMs
    ? event.key.toLowerCase()
    : `${lastCanvasTreeTypeAheadValue}${event.key.toLowerCase()}`;
  lastCanvasTreeTypeAheadAt = now;

  const normalizedQuery = lastCanvasTreeTypeAheadValue;
  const searchPool = [...items.slice(currentIndex + 1), ...items.slice(0, currentIndex + 1)];
  const matchedItem = searchPool.find((item) => {
    const label = String(item.dataset.treeLabel || item.textContent || "").trim().toLowerCase();
    return label.startsWith(normalizedQuery);
  });
  if (matchedItem) {
    event.preventDefault();
    focusCanvasTreeItem(matchedItem);
  }
}

function renderCanvasTreeFile(document, depth, activeDocument) {
  const button = globalThis.document.createElement("button");
  const isActive = Boolean(activeDocument && activeDocument.id === document.id);
  const roleBadge = document.role ? `<span class="canvas-tree-file__role">${escHtml(document.role)}</span>` : "";
  const pathLabel = document.path ? `<span class="canvas-tree-file__path">${escHtml(document.path)}</span>` : "";

  button.type = "button";
  button.className = `canvas-tree-file${isActive ? " active" : ""}`;
  button.style.setProperty("--canvas-tree-depth", String(depth));
  button.disabled = isCanvasEditing && !isActive;
  button.dataset.canvasTreeItem = "true";
  button.dataset.canvasDocumentId = document.id;
  button.dataset.treeLabel = getCanvasFileName(document).toLowerCase();
  button.setAttribute("role", "treeitem");
  button.setAttribute("aria-level", String(depth + 1));
  button.setAttribute("aria-selected", isActive ? "true" : "false");
  button.tabIndex = -1;
  button.innerHTML = `<span class="canvas-tree-file__name">${escHtml(getCanvasFileName(document))}</span>${roleBadge}${pathLabel}`;
  button.title = getCanvasDocumentLabel(document);
  button.addEventListener("click", () => {
    activeCanvasDocumentId = document.id;
    renderCanvasPanel();
    restoreCanvasTreeFocus({ documentId: document.id });
  });
  button.addEventListener("keydown", handleCanvasTreeItemKeydown);
  return button;
}

function renderCanvasTree(documents, activeDocument) {
  if (!canvasTreePanel || !canvasTreeEl) {
    return;
  }

  const shouldShowTree = getCanvasMode(documents) === "project" || (documents || []).length > 1;
  canvasTreePanel.hidden = !shouldShowTree;
  if (!shouldShowTree) {
    canvasTreeEl.innerHTML = "";
    if (canvasTreeCount) {
      canvasTreeCount.textContent = "";
    }
    return;
  }

  const visibleDocuments = getCanvasVisibleDocuments(documents);
  if (canvasTreeCount) {
    canvasTreeCount.textContent = `${visibleDocuments.length} shown`;
  }
  if (!visibleDocuments.length) {
    canvasTreeEl.innerHTML = '<div class="canvas-tree-empty">No files match the current filters.</div>';
    return;
  }

  const tree = buildCanvasTreeNodes(visibleDocuments);
  const fragment = document.createDocumentFragment();

  const renderFolder = (folder, depth = 0) => {
    const section = document.createElement("section");
    const isCollapsed = collapsedCanvasFolders.has(folder.path);
    section.className = "canvas-tree-node";

    const header = document.createElement("button");
    const bodyId = `canvas-tree-group-${encodeURIComponent(String(folder.path || "root"))}`;
    header.type = "button";
    header.className = `canvas-tree-folder${isCollapsed ? " collapsed" : ""}`;
    header.style.setProperty("--canvas-tree-depth", String(depth));
    header.dataset.canvasTreeItem = "true";
    header.dataset.canvasTreeFolder = "true";
    header.dataset.folderPath = folder.path;
    header.dataset.treeLabel = folder.name.toLowerCase();
    header.setAttribute("role", "treeitem");
    header.setAttribute("aria-expanded", isCollapsed ? "false" : "true");
    header.setAttribute("aria-level", String(depth + 1));
    header.setAttribute("aria-controls", bodyId);
    header.tabIndex = -1;
    header.innerHTML = `<span class="canvas-tree-folder__caret">▾</span><span class="canvas-tree-folder__label">${escHtml(folder.name)}</span>`;
    header.addEventListener("click", () => {
      setCanvasTreeFolderExpanded(folder.path);
    });
    header.addEventListener("keydown", handleCanvasTreeItemKeydown);
    section.appendChild(header);

    if (!isCollapsed) {
      const body = document.createElement("div");
      body.id = bodyId;
      body.className = "canvas-tree-children";
      body.setAttribute("role", "group");
      Array.from(folder.folders.values())
        .sort((left, right) => left.name.localeCompare(right.name))
        .forEach((childFolder) => body.appendChild(renderFolder(childFolder, depth + 1)));
      folder.files
        .sort((left, right) => left.name.localeCompare(right.name))
        .forEach((entry) => body.appendChild(renderCanvasTreeFile(entry.document, depth + 1, activeDocument)));
      section.appendChild(body);
    }

    return section;
  };

  Array.from(tree.folders.values())
    .sort((left, right) => left.name.localeCompare(right.name))
    .forEach((folder) => fragment.appendChild(renderFolder(folder, 0)));
  tree.files
    .sort((left, right) => left.name.localeCompare(right.name))
    .forEach((entry) => fragment.appendChild(renderCanvasTreeFile(entry.document, 0, activeDocument)));

  canvasTreeEl.innerHTML = "";
  canvasTreeEl.appendChild(fragment);
  syncCanvasTreeTabStops();
}

function renderHighlightedCodeBlock(codeText, rawLang = null) {
  const normalizedCode = String(codeText || "").replace(/\r\n?/g, "\n");
  const lines = normalizedCode.split("\n");
  const lang = rawLang && highlighter && highlighter.getLanguage(rawLang) ? rawLang : null;
  const renderedLines = lines.map((line, index) => {
    let highlightedLine = line ? escHtml(line) : "&nbsp;";
    if (highlighter) {
      try {
        const sourceLine = line || " ";
        highlightedLine = lang
          ? highlighter.highlight(sourceLine, { language: lang, ignoreIllegals: true }).value
          : highlighter.highlightAuto(sourceLine).value;
      } catch (_) {
        highlightedLine = line ? escHtml(line) : "&nbsp;";
      }
    }
    return `<span class="canvas-code-line"><span class="canvas-code-line__number">${index + 1}</span><span class="canvas-code-line__content">${highlightedLine}</span></span>`;
  }).join("");
  const langClass = lang ? ` language-${lang}` : "";
  const langLabel = lang ? `<span class="canvas-code-lang">${escHtml(lang)}</span>` : "";
  return `<pre class="canvas-code-block">${langLabel}<code class="hljs${langClass}">${renderedLines}</code></pre>`;
}

if (markdownEngine && typeof markdownEngine.use === "function") {
  markdownEngine.use({
    breaks: true,
    gfm: true,
  });
  if (highlighter) {
    markdownEngine.use({
      renderer: {
        // Compatible with both marked v4 (code: string, language: string)
        // and marked v5+ (code: token object with .text and .lang properties).
        code(tokenOrCode, languageHint) {
          const isToken = tokenOrCode !== null && typeof tokenOrCode === "object";
          const codeText = isToken ? String(tokenOrCode.text || "") : String(tokenOrCode || "");
          const rawLang = isToken ? (tokenOrCode.lang || null) : (languageHint || null);
          return `${renderHighlightedCodeBlock(codeText, rawLang)}\n`;
        },
      },
    });
  }
}

function sanitizeHtml(html) {
  const rawHtml = String(html || "");
  if (sanitizer && typeof sanitizer.sanitize === "function") {
    return sanitizer.sanitize(rawHtml);
  }
  return rawHtml;
}

function closeUnclosedCodeFences(text) {
  const fenceCount = (text.match(/^```/gm) || []).length;
  return fenceCount % 2 !== 0 ? text + "\n```" : text;
}

function renderMarkdown(text) {
  const rawText = closeUnclosedCodeFences(String(text || ""));
  if (markdownEngine && typeof markdownEngine.parse === "function") {
    try {
      return sanitizeHtml(markdownEngine.parse(rawText));
    } catch (_) {
      // Fall through to plain-text fallback if the markdown engine throws.
    }
  }
  return sanitizeHtml(escHtml(rawText).replace(/\n/g, "<br>"));
}

function renderCanvasDocumentBody(document) {
  if (!document) {
    return "";
  }
  if (document.format === "code") {
    return sanitizeHtml(`<div class="canvas-code-document">${renderHighlightedCodeBlock(document.content, document.language || null)}</div>`);
  }
  return renderMarkdown(document.content);
}

function getCanvasDocumentById(documents, documentId) {
  const targetId = String(documentId || "").trim();
  if (!targetId) {
    return null;
  }
  return documents.find((document) => document.id === targetId) || null;
}

function buildCanvasDiff(previousContent, nextContent) {
  const previousLines = String(previousContent || "").replace(/\r\n?/g, "\n").split("\n");
  const nextLines = String(nextContent || "").replace(/\r\n?/g, "\n").split("\n");
  let start = 0;
  while (start < previousLines.length && start < nextLines.length && previousLines[start] === nextLines[start]) {
    start += 1;
  }

  let previousEnd = previousLines.length - 1;
  let nextEnd = nextLines.length - 1;
  while (previousEnd >= start && nextEnd >= start && previousLines[previousEnd] === nextLines[nextEnd]) {
    previousEnd -= 1;
    nextEnd -= 1;
  }

  const removed = previousEnd >= start ? previousLines.slice(start, previousEnd + 1) : [];
  const added = nextEnd >= start ? nextLines.slice(start, nextEnd + 1) : [];
  if (!removed.length && !added.length) {
    return null;
  }

  return {
    startLine: start + 1,
    removed,
    added,
  };
}

function renderCanvasDiffPreview(activeDocument) {
  if (!canvasDiffEl) {
    return;
  }
  if (!pendingCanvasDiff || pendingCanvasDiff.documentId !== activeDocument?.id || isCanvasEditing || activeDocument?.isStreamingPreview) {
    canvasDiffEl.hidden = true;
    canvasDiffEl.innerHTML = "";
    return;
  }

  const diff = pendingCanvasDiff.diff;
  const changedLineCount = diff.removed.length + diff.added.length;
  const lines = [
    ...diff.removed.map((line, index) => ({ kind: "removed", lineNumber: diff.startLine + index, text: line })),
    ...diff.added.map((line, index) => ({ kind: "added", lineNumber: diff.startLine + index, text: line })),
  ].slice(0, 120);

  canvasDiffEl.hidden = false;
  canvasDiffEl.innerHTML =
    `<div class="canvas-diff__header">` +
      `<div>` +
        `<div class="canvas-diff__title">Recent AI change</div>` +
        `<div class="canvas-diff__meta">${changedLineCount} changed line${changedLineCount === 1 ? "" : "s"} around line ${diff.startLine}</div>` +
      `</div>` +
      `<button class="canvas-diff__close" type="button" data-action="dismiss-canvas-diff">Dismiss</button>` +
    `</div>` +
    `<div class="canvas-diff__body">` +
      lines.map((line) =>
        `<div class="canvas-diff__line canvas-diff__line--${line.kind}"><span class="canvas-diff__line-num">${line.kind === "added" ? "+" : "-"}${line.lineNumber}</span><span>${escHtml(line.text || " ")}</span></div>`
      ).join("") +
    `</div>`;

  canvasDiffEl.querySelector('[data-action="dismiss-canvas-diff"]')?.addEventListener("click", () => {
    pendingCanvasDiff = null;
    renderCanvasDiffPreview(getActiveCanvasDocument());
  });
}

function setCanvasEditing(enabled) {
  const activeDocument = getActiveCanvasDocument();
  isCanvasEditing = Boolean(enabled && activeDocument);
  editingCanvasDocumentId = isCanvasEditing ? activeDocument.id : null;
  if (isCanvasEditing && canvasEditorEl) {
    canvasEditorEl.value = activeDocument.content || "";
  }
  renderCanvasPanel();
}

function readCanvasWidthPreference() {
  try {
    const value = Number.parseInt(localStorage.getItem(CANVAS_PANEL_WIDTH_STORAGE_KEY) || "", 10);
    return Number.isFinite(value) ? value : CANVAS_PANEL_DEFAULT_WIDTH;
  } catch (_) {
    return CANVAS_PANEL_DEFAULT_WIDTH;
  }
}

function clampCanvasWidth(width) {
  const viewportLimit = Math.max(CANVAS_PANEL_MIN_WIDTH, globalThis.innerWidth - 24);
  return Math.min(Math.max(width, CANVAS_PANEL_MIN_WIDTH), Math.min(CANVAS_PANEL_MAX_WIDTH, viewportLimit));
}

function applyCanvasPanelWidth(width, persist = true) {
  if (!canvasPanel || globalThis.innerWidth <= 900) {
    if (canvasPanel) {
      canvasPanel.style.width = "";
    }
    return;
  }
  const nextWidth = clampCanvasWidth(width);
  canvasPanel.style.width = `${nextWidth}px`;
  if (persist) {
    try {
      localStorage.setItem(CANVAS_PANEL_WIDTH_STORAGE_KEY, String(nextWidth));
    } catch (_) {
      // Ignore storage errors.
    }
  }
}

function getCanvasDocuments(metadata) {
  if (!metadata || typeof metadata !== "object" || !Array.isArray(metadata.canvas_documents)) {
    return [];
  }

  return metadata.canvas_documents
    .map((document) => normalizeCanvasDocument(document))
    .filter((document) => document.id);
}

function getCanvasDocumentCollection(entries = history) {
  if (streamingCanvasDocuments.length) {
    return streamingCanvasDocuments;
  }

  for (let index = entries.length - 1; index >= 0; index -= 1) {
    const message = entries[index];
    if (message?.metadata && message.metadata.canvas_cleared === true) {
      return [];
    }
    const documents = getCanvasDocuments(message?.metadata);
    if (!documents.length) {
      continue;
    }
    return documents;
  }

  return [];
}

function resetStreamingCanvasPreview() {
  streamingCanvasPreview = null;
  if (pendingCanvasPreviewTimer) {
    globalThis.clearTimeout(pendingCanvasPreviewTimer);
    pendingCanvasPreviewTimer = 0;
  }
}

function buildStreamingCanvasPreviewDocument(toolName, previewKey = "", snapshot = {}) {
  const normalizedToolName = String(toolName || "").trim();
  const normalizedPreviewKey = String(previewKey || "").trim() || "canvas-call-0";
  const snapshotData = snapshot && typeof snapshot === "object" ? snapshot : {};
  const activeDocument = getActiveCanvasDocument(history);
  const isRewritePreview = normalizedToolName === "rewrite_canvas_document" && activeDocument;
  const normalized = normalizeStreamingCanvasPreviewDocument({
    id: isRewritePreview ? activeDocument.id : `streaming-canvas-preview-${normalizedPreviewKey}`,
    title: String(snapshotData.title || (isRewritePreview ? activeDocument.title : "Canvas draft")).trim() || "Canvas draft",
    path: String(snapshotData.path || (isRewritePreview ? activeDocument.path : "")).trim(),
    role: String(snapshotData.role || (isRewritePreview ? activeDocument.role : "note")).trim(),
    summary: isRewritePreview ? String(activeDocument.summary || "") : "",
    format: String(snapshotData.format || (isRewritePreview ? activeDocument.format : "markdown")).trim() || "markdown",
    language: String(snapshotData.language || (isRewritePreview ? activeDocument.language : "")).trim(),
    content: "",
    source_message_id: isRewritePreview ? activeDocument.source_message_id : null,
  });
  return normalized ? { ...normalized, isStreamingPreview: true, tool: normalizedToolName, previewKey: normalizedPreviewKey } : null;
}

function applyStreamingCanvasPreviewSnapshot(snapshot = {}) {
  if (!streamingCanvasPreview || !snapshot || typeof snapshot !== "object") {
    return false;
  }
  let changed = false;
  if (typeof snapshot.title === "string" && snapshot.title.trim()) {
    const nextTitle = snapshot.title.trim();
    if (nextTitle !== streamingCanvasPreview.title) {
      streamingCanvasPreview.title = nextTitle;
      changed = true;
    }
  }
  if (typeof snapshot.path === "string") {
    const nextPath = snapshot.path.trim().replace(/\\/g, "/");
    if (nextPath && nextPath !== streamingCanvasPreview.path) {
      streamingCanvasPreview.path = nextPath;
      changed = true;
    }
  }
  if (typeof snapshot.role === "string") {
    const nextRole = snapshot.role.trim().toLowerCase();
    if (nextRole && nextRole !== streamingCanvasPreview.role) {
      streamingCanvasPreview.role = nextRole;
      changed = true;
    }
  }
  if (typeof snapshot.format === "string") {
    const normalizedFormat = snapshot.format.trim().toLowerCase();
    const nextFormat = normalizedFormat === "code" ? "code" : "markdown";
    if (nextFormat !== streamingCanvasPreview.format) {
      streamingCanvasPreview.format = nextFormat;
      changed = true;
    }
  }
  if (typeof snapshot.language === "string") {
    const nextLanguage = snapshot.language.trim().toLowerCase();
    if (nextLanguage && nextLanguage !== streamingCanvasPreview.language) {
      streamingCanvasPreview.language = nextLanguage;
      changed = true;
    }
  }

  const normalizedPreview = normalizeStreamingCanvasPreviewDocument(streamingCanvasPreview);
  if (normalizedPreview) {
    ["title", "path", "role", "format", "language", "summary"].forEach((key) => {
      if (normalizedPreview[key] !== streamingCanvasPreview[key]) {
        streamingCanvasPreview[key] = normalizedPreview[key];
        changed = true;
      }
    });
  }

  return changed;
}

function ensureStreamingCanvasPreview(toolName, previewKey = "", snapshot = {}) {
  const normalizedToolName = String(toolName || "").trim();
  const normalizedPreviewKey = String(previewKey || "").trim() || "canvas-call-0";
  if (!normalizedToolName) {
    return null;
  }
  if (
    !streamingCanvasPreview
    || streamingCanvasPreview.tool !== normalizedToolName
    || streamingCanvasPreview.previewKey !== normalizedPreviewKey
  ) {
    streamingCanvasPreview = buildStreamingCanvasPreviewDocument(normalizedToolName, normalizedPreviewKey, snapshot);
  }
  if (!streamingCanvasPreview) {
    return null;
  }
  applyStreamingCanvasPreviewSnapshot(snapshot);
  activeCanvasDocumentId = streamingCanvasPreview.id;
  return streamingCanvasPreview;
}

function getCanvasRenderableDocuments(entries = history) {
  const documents = getCanvasDocumentCollection(entries);
  if (!streamingCanvasPreview?.id) {
    return documents;
  }
  const previewIndex = documents.findIndex((document) => document.id === streamingCanvasPreview.id);
  if (previewIndex >= 0) {
    return [
      ...documents.slice(0, previewIndex),
      streamingCanvasPreview,
      ...documents.slice(previewIndex + 1),
    ];
  }
  return [...documents, streamingCanvasPreview];
}

function buildCanvasStructureSignature(documents, visibleDocuments = documents) {
  const documentSignature = (documents || []).map((document) => [
    String(document.id || "").trim(),
    String(document.title || "").trim(),
    String(document.path || "").trim(),
    String(document.role || "").trim(),
    String(document.format || "").trim(),
    String(document.language || "").trim(),
    document.isStreamingPreview ? "preview" : "stored",
  ].join("\u241f")).join("\u241e");
  const visibleSignature = (visibleDocuments || []).map((document) => String(document.id || "").trim()).join("\u241e");
  const filterSignature = [
    String(canvasSearchInput?.value || "").trim(),
    String(canvasRoleFilter?.value || "").trim(),
    getCanvasPathFilterValue(),
    isCanvasEditing ? "editing" : "view",
  ].join("\u241f");
  return [documentSignature, visibleSignature, filterSignature].join("\u241d");
}

function buildCanvasRenderState(documents = getCanvasRenderableDocuments()) {
  const visibleDocuments = getCanvasVisibleDocuments(documents);
  const preferredActiveId = [
    String(activeCanvasDocumentId || "").trim(),
    String(getCanvasPreferredActiveDocumentId() || "").trim(),
  ].find(Boolean) || "";
  const activeDocument = visibleDocuments.length
    ? getCanvasDocumentById(visibleDocuments, preferredActiveId) || visibleDocuments[visibleDocuments.length - 1]
    : null;

  return {
    isCanvasPanelOpen: isCanvasOpen(),
    documents,
    visibleDocuments,
    activeDocument,
    isStreamingPreviewActive: Boolean(activeDocument?.isStreamingPreview),
    searchTerm: String(canvasSearchInput?.value || "").trim(),
    structureSignature: buildCanvasStructureSignature(documents, visibleDocuments),
  };
}

function scheduleCanvasPreviewRender() {
  if (pendingCanvasPreviewTimer) {
    return;
  }
  pendingCanvasPreviewTimer = globalThis.setTimeout(() => {
    pendingCanvasPreviewTimer = 0;
    renderCanvasPreviewFrame();
  }, CANVAS_PREVIEW_RENDER_INTERVAL_MS);
}

function getActiveCanvasDocument(entries = history) {
  const documents = getCanvasDocumentCollection(entries);
  if (!documents.length) {
    return null;
  }

  const preferredId = String(activeCanvasDocumentId || getCanvasPreferredActiveDocumentId(entries) || "").trim();
  if (preferredId) {
    const matched = documents.find((document) => document.id === preferredId);
    if (matched) {
      return matched;
    }
  }

  return documents[documents.length - 1];
}

function setCanvasStatus(message, tone = "muted") {
  if (!canvasStatus) {
    return;
  }
  canvasStatus.textContent = String(message || "").trim() || "Canvas idle";
  canvasStatus.dataset.tone = tone;
}

function setCanvasSearchStatus(message, tone = "muted") {
  if (!canvasSearchStatus) {
    return;
  }

  const text = String(message || "").trim();
  canvasSearchStatus.dataset.tone = tone;
  canvasSearchStatus.hidden = !text;
  canvasSearchStatus.textContent = text;
}

function updateCanvasSearchFeedback(renderState, matchCount = 0) {
  const {
    documents,
    visibleDocuments,
    isStreamingPreviewActive,
    searchTerm,
  } = renderState;

  if (!documents.length || isCanvasEditing || isStreamingPreviewActive) {
    setCanvasSearchStatus("");
    return;
  }

  const roleValue = String(canvasRoleFilter?.value || "").trim();
  const pathValue = getCanvasPathFilterValue();
  if (!searchTerm && !roleValue && !pathValue) {
    setCanvasSearchStatus("");
    return;
  }

  if (!visibleDocuments.length) {
    const filterParts = [];
    if (searchTerm) {
      filterParts.push(`search \"${searchTerm}\"`);
    }
    if (roleValue) {
      filterParts.push(`role ${roleValue}`);
    }
    if (pathValue) {
      filterParts.push(pathValue === CANVAS_ROOT_PATH_FILTER ? "root files" : `path ${pathValue}`);
    }
    setCanvasSearchStatus(`No canvas files match ${filterParts.join(" · ")}.`, "warning");
    return;
  }

  if (searchTerm) {
    setCanvasSearchStatus(
      matchCount
        ? `${matchCount} search match${matchCount === 1 ? "" : "es"} across ${visibleDocuments.length} file${visibleDocuments.length === 1 ? "" : "s"}.`
        : `No text matches in ${visibleDocuments.length} filtered file${visibleDocuments.length === 1 ? "" : "s"}.`,
      matchCount ? "muted" : "warning"
    );
    return;
  }

  const filterCount = visibleDocuments.length;
  setCanvasSearchStatus(
    `${filterCount} file${filterCount === 1 ? "" : "s"} shown after filtering.`,
    "muted"
  );
}

function describeCanvasActiveDocumentChange(previousDocument, nextDocument, requestedDocumentId = "") {
  if (!nextDocument) {
    return "";
  }

  const previousId = String(previousDocument?.id || "").trim();
  const nextId = String(nextDocument.id || "").trim();
  const requestedId = String(requestedDocumentId || "").trim();
  const nextLabel = getCanvasDocumentDisplayName(nextDocument);
  if (requestedId && requestedId === nextId && requestedId !== previousId) {
    return `Active canvas switched to ${nextLabel}.`;
  }
  if (previousId && previousId !== nextId) {
    return `Previous active canvas is unavailable. Focus moved to ${nextLabel}.`;
  }
  return "";
}

function setCanvasHint(message, tone = "muted") {
  if (!canvasHint) {
    return;
  }

  const text = String(message || "").trim();
  if (!text) {
    canvasHint.hidden = true;
    canvasHint.textContent = "";
    canvasHint.dataset.tone = tone;
    return;
  }

  canvasHint.hidden = false;
  canvasHint.textContent = text;
  canvasHint.dataset.tone = tone;
}

function setPendingDocumentCanvasOpen(files) {
  if (!files || !files.length) {
    pendingDocumentCanvasOpen = null;
    return;
  }

  pendingDocumentCanvasOpen = {
    fileCount: files.length,
    fileName: String(files[0]?.name || "Document").trim() || "Document",
  };
}

function renderCanvasMetaBar(renderState) {
  if (!canvasMetaBar || !canvasMetaChips) {
    return;
  }

  const { activeDocument, documents, isStreamingPreviewActive, visibleDocuments } = renderState;
  if (!activeDocument || !(documents || []).length) {
    resetCanvasMetaBar();
    return;
  }

  const modeLabel = getCanvasMode(documents) === "project" ? "Project mode" : "Document mode";
  const countLabel = visibleDocuments.length === documents.length
    ? `${documents.length} file${documents.length === 1 ? "" : "s"}`
    : `${visibleDocuments.length}/${documents.length} shown`;
  const chips = [
    { label: modeLabel, className: "canvas-meta-chip canvas-meta-chip--primary" },
    { label: countLabel, className: "canvas-meta-chip" },
  ];

  if (isStreamingPreviewActive) {
    chips.push({ label: "Live preview", className: "canvas-meta-chip canvas-meta-chip--live" });
  }
  if (activeDocument.role) {
    chips.push({ label: activeDocument.role, className: "canvas-meta-chip" });
  }
  chips.push({ label: activeDocument.format === "code" ? "Code" : "Markdown", className: "canvas-meta-chip" });
  if (activeDocument.language) {
    chips.push({ label: activeDocument.language, className: "canvas-meta-chip" });
  }

  const reference = getCanvasDocumentReference(activeDocument);
  if (reference) {
    chips.push({
      label: reference,
      className: "canvas-meta-chip canvas-meta-chip--path",
      title: reference,
    });
  }

  canvasMetaChips.innerHTML = chips.map((chip) => {
    const titleAttr = chip.title ? ` title="${escHtml(chip.title)}"` : "";
    return `<span class="${chip.className}"${titleAttr}>${escHtml(chip.label)}</span>`;
  }).join("");
  canvasMetaBar.hidden = false;

  if (canvasCopyRefBtn) {
    canvasCopyRefBtn.disabled = !reference;
    canvasCopyRefBtn.textContent = activeDocument.path ? "Copy path" : "Copy title";
  }
  if (canvasResetFiltersBtn) {
    canvasResetFiltersBtn.disabled = !hasActiveCanvasFilters();
  }
}

function renderCanvasDocumentTabs(visibleDocuments) {
  if (!canvasDocumentTabsEl) {
    return;
  }

  canvasDocumentTabsEl.hidden = visibleDocuments.length <= 1;
  canvasDocumentTabsEl.innerHTML = "";
  visibleDocuments.forEach((entry) => {
    const button = globalThis.document.createElement("button");
    button.type = "button";
    button.className = `canvas-document-tab${entry.id === activeCanvasDocumentId ? " active" : ""}`;
    button.textContent = getCanvasFileName(entry);
    button.title = `${getCanvasDocumentLabel(entry)} · ${entry.line_count} lines`;
    button.disabled = isCanvasEditing && entry.id !== activeCanvasDocumentId;
    button.addEventListener("click", () => {
      activeCanvasDocumentId = entry.id;
      renderCanvasPanel();
    });
    canvasDocumentTabsEl.appendChild(button);
  });
}

function updateCanvasActiveDocumentDisplay(renderState) {
  const {
    activeDocument,
    documents,
    isCanvasPanelOpen,
    isStreamingPreviewActive,
    searchTerm,
    visibleDocuments,
  } = renderState;

  activeCanvasDocumentId = activeDocument.id;
  const modeLabel = getCanvasMode(documents) === "project" ? "Project mode" : "Document mode";
  const detailLabel = activeDocument.path || activeDocument.title;
  const roleLabel = activeDocument.role ? ` · ${activeDocument.role}` : "";
  const languageLabel = activeDocument.language ? ` · ${activeDocument.language}` : "";
  canvasSubtitle.textContent = `${modeLabel} · ${visibleDocuments.length}/${documents.length} files · ${detailLabel} · ${activeDocument.line_count} lines${roleLabel}${languageLabel}`;
  renderCanvasMetaBar(renderState);
  const promptLineLimit = Number(appSettings.canvas_prompt_max_lines || 800);
  const expandLineLimit = Number(appSettings.canvas_expand_max_lines || 1600);
  if (Number.isFinite(activeDocument.line_count) && activeDocument.line_count > promptLineLimit) {
    const hasExpandedRoom = activeDocument.line_count > expandLineLimit;
    setCanvasHint(
      hasExpandedRoom
        ? "Large canvas detected. The default view is truncated. Use scroll_canvas_document for a targeted range or expand_canvas_document for a wider slice."
        : `Large canvas detected. The default view is truncated to the first ${promptLineLimit} lines; use scroll_canvas_document for a targeted range.`,
      "warning"
    );
  } else {
    setCanvasHint("");
  }
  canvasEmptyState.hidden = true;
  canvasEmptyState.innerHTML = "<h3>No canvas document yet</h3><p>Ask the assistant to draft something substantial, then continue refining it with line-based edits.</p>";
  if (canvasFormatSelect) {
    canvasFormatSelect.disabled = !isCanvasEditing || isStreamingPreviewActive;
    canvasFormatSelect.value = activeDocument.format || "markdown";
  }
  if (canvasSearchInput) {
    canvasSearchInput.disabled = isCanvasEditing || isStreamingPreviewActive;
  }
  if (canvasRoleFilter) {
    canvasRoleFilter.disabled = isCanvasEditing || isStreamingPreviewActive;
  }
  if (canvasPathFilter) {
    canvasPathFilter.disabled = isCanvasEditing || isStreamingPreviewActive;
  }
  if (canvasEditBtn) {
    canvasEditBtn.hidden = isCanvasEditing;
    canvasEditBtn.disabled = isStreamingPreviewActive;
  }
  if (canvasSaveBtn) {
    canvasSaveBtn.hidden = !isCanvasEditing;
    canvasSaveBtn.disabled = isStreamingPreviewActive;
  }
  if (canvasCancelBtn) {
    canvasCancelBtn.hidden = !isCanvasEditing;
    canvasCancelBtn.disabled = isStreamingPreviewActive;
  }

  if (isCanvasEditing && canvasEditorEl) {
    if (editingCanvasDocumentId !== activeDocument.id) {
      editingCanvasDocumentId = activeDocument.id;
      canvasEditorEl.value = activeDocument.content || "";
    }
    canvasEditorEl.hidden = false;
    canvasDocumentEl.hidden = true;
    canvasDocumentEl.innerHTML = "";
  } else {
    canvasDocumentEl.hidden = false;
    canvasDocumentEl.innerHTML = renderCanvasDocumentBody(activeDocument);
    if (canvasEditorEl) {
      canvasEditorEl.hidden = true;
    }
  }

  renderCanvasDiffPreview(activeDocument);
  const matchCount = !isCanvasEditing && !isStreamingPreviewActive ? applyCanvasSearchHighlight(searchTerm) : 0;
  updateCanvasSearchFeedback(renderState, matchCount);
  if (canvasCopyBtn) {
    canvasCopyBtn.disabled = !String(activeDocument.content || "").length;
    canvasCopyBtn.hidden = !isCanvasPanelOpen;
  }
  if (canvasDeleteBtn) {
    canvasDeleteBtn.disabled = isStreamingPreviewActive;
  }
  if (canvasClearBtn) {
    canvasClearBtn.disabled = isStreamingPreviewActive || documents.length === 0;
  }
  if (canvasDownloadHtmlBtn) {
    canvasDownloadHtmlBtn.disabled = isStreamingPreviewActive;
  }
  if (canvasDownloadMdBtn) {
    canvasDownloadMdBtn.disabled = isStreamingPreviewActive;
  }
  if (canvasDownloadPdfBtn) {
    canvasDownloadPdfBtn.disabled = isStreamingPreviewActive;
  }
}

function renderCanvasPreviewFrame() {
  if (!canvasDocumentEl || !canvasEmptyState || !canvasSubtitle) {
    return;
  }

  const renderState = buildCanvasRenderState();
  if (!renderState.documents.length || !renderState.activeDocument || isCanvasEditing || !renderState.isStreamingPreviewActive) {
    renderCanvasPanel();
    return;
  }

  if (renderState.structureSignature !== lastCanvasStructureSignature) {
    renderCanvasPanel();
    return;
  }

  updateCanvasActiveDocumentDisplay(renderState);
}

function consumePendingDocumentCanvasOpen() {
  const pendingRequest = pendingDocumentCanvasOpen;
  pendingDocumentCanvasOpen = null;
  return pendingRequest;
}

function isCanvasConfirmOpen() {
  return Boolean(canvasConfirmModal?.classList.contains("open"));
}

function closeCanvasConfirmModal(action = "cancel", executeHandler = true) {
  if (!canvasConfirmModal) {
    return;
  }

  const pendingAction = pendingCanvasConfirmAction;
  pendingCanvasConfirmAction = null;
  canvasConfirmModal.classList.remove("open");
  canvasConfirmOverlay?.classList.remove("open");
  canvasConfirmModal.setAttribute("aria-hidden", "true");

  if (lastCanvasConfirmTriggerEl && typeof lastCanvasConfirmTriggerEl.focus === "function") {
    lastCanvasConfirmTriggerEl.focus();
  }

  if (!executeHandler || !pendingAction) {
    return;
  }

  if (action === "confirm") {
    pendingAction.onConfirm?.();
    return;
  }

  pendingAction.onCancel?.();
}

function openCanvasConfirmModal(options = {}) {
  if (!canvasConfirmModal || !canvasConfirmTitle || !canvasConfirmMessage) {
    options.onConfirm?.();
    return;
  }

  if (isCanvasConfirmOpen()) {
    closeCanvasConfirmModal("cancel", false);
  }

  closeMobileTools();
  closeExportPanel();
  closeStats();
  lastCanvasConfirmTriggerEl = document.activeElement instanceof HTMLElement ? document.activeElement : attachBtn;
  pendingCanvasConfirmAction = {
    onConfirm: typeof options.onConfirm === "function" ? options.onConfirm : null,
    onCancel: typeof options.onCancel === "function" ? options.onCancel : null,
  };
  canvasConfirmTitle.textContent = String(options.title || "Open document in Canvas?").trim() || "Open document in Canvas?";
  canvasConfirmMessage.textContent = String(options.message || "Your uploaded document is ready in Canvas.").trim() || "Your uploaded document is ready in Canvas.";
  canvasConfirmModal.classList.add("open");
  canvasConfirmOverlay?.classList.add("open");
  canvasConfirmModal.setAttribute("aria-hidden", "false");
  canvasConfirmOpenBtn?.focus();
}

function confirmCanvasOpenForDocument(pendingRequest, documentCount, callbacks = {}) {
  const fileCount = Number(pendingRequest?.fileCount || 1);
  const fileName = String(pendingRequest?.fileName || "document").trim() || "document";
  const requestLabel = fileCount > 1 ? `${fileCount} uploaded documents` : fileName;
  const documentLabel = documentCount === 1 ? "canvas document" : `${documentCount} canvas documents`;
  openCanvasConfirmModal({
    title: "Open document in Canvas?",
    message: `${requestLabel} ${fileCount === 1 ? "is" : "are"} ready in Canvas. ${documentLabel.charAt(0).toUpperCase()}${documentLabel.slice(1)} ${documentCount === 1 ? "is" : "are"} available now.`,
    onConfirm: callbacks.onConfirm,
    onCancel: callbacks.onCancel,
  });
}

function escapeRegExp(text) {
  return String(text || "").replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function setCanvasAttention(enabled) {
  canvasHasUnreadUpdates = Boolean(enabled);
  if (canvasBtnIndicator) {
    canvasBtnIndicator.hidden = !canvasHasUnreadUpdates;
  }
}

function setExportStatus(message, tone = "muted") {
  if (!exportStatus) {
    return;
  }
  exportStatus.textContent = String(message || "").trim() || "Export idle";
  exportStatus.dataset.tone = tone;
}

function updateExportPanel() {
  if (!exportSubtitle) {
    return;
  }
  exportSubtitle.textContent = currentConvId
    ? `Current conversation: ${currentConvTitle || `Chat #${currentConvId}`}`
    : "Open or create a conversation before exporting.";
}

function isCanvasOpen() {
  return Boolean(canvasPanel?.classList.contains("open"));
}

function syncCanvasToggleButton() {
  if (!canvasToggleBtn) {
    return;
  }
  canvasToggleBtn.setAttribute("aria-expanded", String(isCanvasOpen()));
}

function getCanvasFocusableElements() {
  if (!canvasPanel) {
    return [];
  }
  return Array.from(
    canvasPanel.querySelectorAll(
      'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])'
    )
  ).filter((element) => !element.hasAttribute("hidden") && element.getAttribute("aria-hidden") !== "true");
}

function applyCanvasSearchHighlight(query) {
  if (!canvasDocumentEl) {
    return 0;
  }

  const normalizedQuery = String(query || "").trim();
  if (!normalizedQuery) {
    return 0;
  }

  const pattern = escapeRegExp(normalizedQuery);
  const selectorMatcher = new RegExp(pattern, "i");
  const walker = document.createTreeWalker(canvasDocumentEl, NodeFilter.SHOW_TEXT, {
    acceptNode(node) {
      const parentName = node.parentNode?.nodeName;
      if (!node.textContent?.trim()) {
        return NodeFilter.FILTER_REJECT;
      }
      if (parentName === "SCRIPT" || parentName === "STYLE" || parentName === "MARK") {
        return NodeFilter.FILTER_REJECT;
      }
      return selectorMatcher.test(node.textContent) ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_REJECT;
    },
  });

  const textNodes = [];
  let currentNode;
  while ((currentNode = walker.nextNode())) {
    textNodes.push(currentNode);
  }

  let matchCount = 0;
  textNodes.forEach((textNode) => {
    const source = textNode.textContent || "";
    const fragment = document.createDocumentFragment();
    const highlightMatcher = new RegExp(pattern, "gi");
    let lastIndex = 0;

    source.replace(highlightMatcher, (matched, offset) => {
      if (offset > lastIndex) {
        fragment.appendChild(document.createTextNode(source.slice(lastIndex, offset)));
      }
      const mark = document.createElement("mark");
      mark.textContent = matched;
      fragment.appendChild(mark);
      lastIndex = offset + matched.length;
      matchCount += 1;
      return matched;
    });

    if (lastIndex < source.length) {
      fragment.appendChild(document.createTextNode(source.slice(lastIndex)));
    }

    textNode.parentNode.replaceChild(fragment, textNode);
  });

  return matchCount;
}

function renderCanvasPanel() {
  if (!canvasDocumentEl || !canvasEmptyState || !canvasSubtitle) {
    return;
  }

  const documents = getCanvasRenderableDocuments();
  syncCanvasFilterControls(documents);
  const renderState = buildCanvasRenderState(documents);
  const {
    activeDocument,
    documents: renderDocuments,
    visibleDocuments,
  } = renderState;
  lastCanvasStructureSignature = renderState.structureSignature;

  renderCanvasTree(renderDocuments, activeDocument);
  if (!renderDocuments.length) {
    isCanvasEditing = false;
    editingCanvasDocumentId = null;
    resetCanvasMetaBar();
    canvasSubtitle.textContent = "No canvas document yet.";
    setCanvasHint("");
    canvasEmptyState.hidden = false;
    if (canvasEditorEl) {
      canvasEditorEl.hidden = true;
      canvasEditorEl.value = "";
    }
    canvasDocumentEl.hidden = true;
    canvasDocumentEl.innerHTML = "";
    if (canvasDiffEl) {
      canvasDiffEl.hidden = true;
      canvasDiffEl.innerHTML = "";
    }
    if (canvasDocumentTabsEl) {
      canvasDocumentTabsEl.hidden = true;
      canvasDocumentTabsEl.innerHTML = "";
    }
    if (canvasEditBtn) {
      canvasEditBtn.disabled = true;
      canvasEditBtn.hidden = false;
    }
    if (canvasSaveBtn) {
      canvasSaveBtn.disabled = true;
      canvasSaveBtn.hidden = true;
    }
    if (canvasCancelBtn) {
      canvasCancelBtn.disabled = true;
      canvasCancelBtn.hidden = true;
    }
    if (canvasCopyBtn) {
      canvasCopyBtn.disabled = true;
      canvasCopyBtn.hidden = true;
    }
    if (canvasDeleteBtn) {
      canvasDeleteBtn.disabled = true;
    }
    if (canvasClearBtn) {
      canvasClearBtn.disabled = true;
    }
    if (canvasDownloadHtmlBtn) {
      canvasDownloadHtmlBtn.disabled = true;
    }
    if (canvasDownloadMdBtn) {
      canvasDownloadMdBtn.disabled = true;
    }
    if (canvasDownloadPdfBtn) {
      canvasDownloadPdfBtn.disabled = true;
    }
    if (canvasSearchInput) {
      canvasSearchInput.disabled = true;
    }
    if (canvasFormatSelect) {
      canvasFormatSelect.disabled = true;
      canvasFormatSelect.value = "markdown";
    }
    if (canvasRoleFilter) {
      canvasRoleFilter.disabled = true;
    }
    if (canvasPathFilter) {
      canvasPathFilter.disabled = true;
    }
    setCanvasSearchStatus("");
    return;
  }

  if (!activeDocument) {
    isCanvasEditing = false;
    editingCanvasDocumentId = null;
    resetCanvasMetaBar();
    canvasSubtitle.textContent = `Project mode · ${renderDocuments.length} files · no matches`;
    setCanvasHint("");
    canvasEmptyState.hidden = false;
    canvasEmptyState.innerHTML = "<h3>No files match the current filters</h3><p>Adjust the search term, role, or path filter to bring files back into view.</p>";
    if (canvasEditorEl) {
      canvasEditorEl.hidden = true;
      canvasEditorEl.value = "";
    }
    canvasDocumentEl.hidden = true;
    canvasDocumentEl.innerHTML = "";
    if (canvasDiffEl) {
      canvasDiffEl.hidden = true;
      canvasDiffEl.innerHTML = "";
    }
    if (canvasEditBtn) {
      canvasEditBtn.disabled = true;
      canvasEditBtn.hidden = false;
    }
    if (canvasSaveBtn) {
      canvasSaveBtn.disabled = true;
      canvasSaveBtn.hidden = true;
    }
    if (canvasCancelBtn) {
      canvasCancelBtn.disabled = true;
      canvasCancelBtn.hidden = true;
    }
    if (canvasCopyBtn) {
      canvasCopyBtn.disabled = true;
      canvasCopyBtn.hidden = true;
    }
    if (canvasDeleteBtn) {
      canvasDeleteBtn.disabled = true;
    }
    if (canvasClearBtn) {
      canvasClearBtn.disabled = renderDocuments.length === 0;
    }
    if (canvasDownloadHtmlBtn) {
      canvasDownloadHtmlBtn.disabled = true;
    }
    if (canvasDownloadMdBtn) {
      canvasDownloadMdBtn.disabled = true;
    }
    if (canvasDownloadPdfBtn) {
      canvasDownloadPdfBtn.disabled = true;
    }
    if (canvasRoleFilter) {
      canvasRoleFilter.disabled = false;
    }
    if (canvasPathFilter) {
      canvasPathFilter.disabled = false;
    }
    updateCanvasSearchFeedback(renderState, 0);
    return;
  }

  updateCanvasActiveDocumentDisplay(renderState);
  renderCanvasDocumentTabs(visibleDocuments);
}

function openCanvas(triggerEl = null) {
  closeMobileTools();
  closeCanvasConfirmModal("cancel", false);
  closeStats();
  closeExportPanel();
  canvasPanel?.classList.add("open");
  canvasOverlay?.classList.add("open");
  canvasPanel?.setAttribute("aria-hidden", "false");
  syncCanvasToggleButton();
  lastCanvasTriggerEl = triggerEl instanceof HTMLElement
    ? triggerEl
    : (document.activeElement instanceof HTMLElement ? document.activeElement : mobileToolsBtn);
  setCanvasAttention(false);
  applyCanvasPanelWidth(readCanvasWidthPreference(), false);
  renderCanvasPanel();
  canvasClose?.focus();
}

function closeCanvas() {
  isCanvasEditing = false;
  editingCanvasDocumentId = null;
  canvasPanel?.classList.remove("open");
  canvasOverlay?.classList.remove("open");
  canvasPanel?.setAttribute("aria-hidden", "true");
  syncCanvasToggleButton();
  if (canvasCopyBtn) {
    canvasCopyBtn.hidden = true;
  }
  if (lastCanvasTriggerEl && typeof lastCanvasTriggerEl.focus === "function") {
    lastCanvasTriggerEl.focus();
  }
}

function openExportPanel(triggerEl = null) {
  closeMobileTools();
  closeStats();
  closeCanvas();
  updateExportPanel();
  exportPanel?.classList.add("open");
  exportOverlay?.classList.add("open");
  exportPanel?.setAttribute("aria-hidden", "false");
  lastExportTriggerEl = triggerEl instanceof HTMLElement
    ? triggerEl
    : (document.activeElement instanceof HTMLElement ? document.activeElement : mobileToolsBtn);
  exportClose?.focus();
}

async function pruneConversationHistory() {
  if (!currentConvId) {
    showToast("No active conversation.", "warning");
    return;
  }
  const raw = window.prompt("How many of the first unpruned messages should be pruned?", "5");
  if (raw === null) {
    return;
  }
  const count = parseInt(raw, 10);
  if (!Number.isInteger(count) || count < 1 || count > 50) {
    showToast("Please enter a number between 1 and 50.", "warning");
    return;
  }
  if (mobilePruneBtn) {
    mobilePruneBtn.disabled = true;
  }
  try {
    const response = await fetch(`/api/conversations/${currentConvId}/prune-batch`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ count }),
    });
    const data = await response.json().catch(() => null);
    if (!response.ok) {
      throw new Error(data?.error || "Pruning failed.");
    }
    if (Array.isArray(data.messages)) {
      history = data.messages.map(normalizeHistoryEntry);
      rebuildTokenStatsFromHistory();
      renderConversationHistory();
    }
    loadSidebar();
    showToast(
      data.pruned_count > 0
        ? `${data.pruned_count} message${data.pruned_count === 1 ? " was" : "s were"} pruned.`
        : "No eligible messages found.",
      "success",
    );
  } catch (error) {
    showError(error.message || "Pruning failed.");
  } finally {
    if (mobilePruneBtn) {
      mobilePruneBtn.disabled = false;
    }
  }
}

function closeExportPanel() {
  exportPanel?.classList.remove("open");
  exportOverlay?.classList.remove("open");
  exportPanel?.setAttribute("aria-hidden", "true");
  if (lastExportTriggerEl && typeof lastExportTriggerEl.focus === "function") {
    lastExportTriggerEl.focus();
  }
}

async function downloadConversation(format) {
  if (!currentConvId) {
    setExportStatus("Conversation is not available yet.", "warning");
    return;
  }

  setExportStatus(`Preparing ${format.toUpperCase()} export…`, "muted");
  try {
    const response = await fetch(`/api/conversations/${currentConvId}/export?format=${encodeURIComponent(format)}`);
    if (!response.ok) {
      const payload = await response.json().catch(() => ({}));
      throw new Error(payload.error || "Conversation export failed.");
    }

    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `${currentConvTitle || "conversation"}.${format}`;
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    URL.revokeObjectURL(url);
    setExportStatus(`${format.toUpperCase()} download is ready.`, "success");
  } catch (error) {
    setExportStatus(error.message || "Conversation export failed.", "danger");
  }
}

async function downloadCanvasDocument(format) {
  const canvasDocument = getActiveCanvasDocument();
  if (!canvasDocument || !currentConvId) {
    setCanvasStatus("Canvas document is not available yet.", "warning");
    return;
  }

  setCanvasStatus(`Preparing ${format.toUpperCase()} download…`, "muted");
  try {
    const response = await fetch(`/api/conversations/${currentConvId}/canvas/export?format=${encodeURIComponent(format)}&document_id=${encodeURIComponent(canvasDocument.id)}`);
    if (!response.ok) {
      const payload = await response.json().catch(() => ({}));
      throw new Error(payload.error || "Canvas export failed.");
    }

    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `${canvasDocument.title}.${format}`;
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    URL.revokeObjectURL(url);
    setCanvasStatus(`${format.toUpperCase()} download is ready.`, "success");
  } catch (error) {
    setCanvasStatus(error.message || "Canvas export failed.", "danger");
  }
}

async function deleteCanvasDocuments({ documentId = null, clearAll = false } = {}) {
  if (!currentConvId) {
    setCanvasStatus("Canvas is not available yet.", "warning");
    return;
  }

  const activeDocument = getActiveCanvasDocument();
  const targetDocumentId = documentId || activeDocument?.id || null;
  if (!clearAll && !targetDocumentId) {
    setCanvasStatus("No canvas document is available to delete.", "warning");
    return;
  }

  const confirmationMessage = clearAll
    ? "Delete all canvas documents?"
    : `Delete ${activeDocument?.title || "this canvas document"}?`;
  if (!globalThis.confirm(confirmationMessage)) {
    return;
  }

  setCanvasStatus(clearAll ? "Deleting all canvas documents..." : "Deleting canvas document...", "muted");
  try {
    const params = new URLSearchParams();
    if (targetDocumentId) {
      params.set("document_id", targetDocumentId);
    }
    if (clearAll) {
      params.set("clear_all", "true");
    }

    const query = params.toString();
    const response = await fetch(`/api/conversations/${currentConvId}/canvas${query ? `?${query}` : ""}`, {
      method: "DELETE",
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(payload.error || "Canvas delete failed.");
    }

    history = Array.isArray(payload.messages) ? payload.messages.map(normalizeHistoryEntry) : history;
    streamingCanvasDocuments = [];
    pendingCanvasDiff = null;
    isCanvasEditing = false;
    editingCanvasDocumentId = null;
    activeCanvasDocumentId = payload.cleared
      ? null
      : String(payload.active_document_id || getActiveCanvasDocument(history)?.id || "").trim() || null;
    renderConversationHistory();
    renderCanvasPanel();

    if (payload.cleared) {
      setCanvasAttention(false);
      setCanvasStatus("Canvas cleared.", "success");
      return;
    }

    setCanvasStatus("Canvas document deleted.", "success");
  } catch (error) {
    setCanvasStatus(error.message || "Canvas delete failed.", "danger");
  }
}

async function saveCanvasEdits() {
  const activeDocument = getActiveCanvasDocument();
  if (!currentConvId || !activeDocument || !canvasEditorEl) {
    setCanvasStatus("Canvas document is not available yet.", "warning");
    return;
  }

  const nextContent = canvasEditorEl.value.replace(/\r\n?/g, "\n");
  const nextFormat = canvasFormatSelect?.value === "code" ? "code" : "markdown";
  setCanvasStatus("Saving canvas edits...", "muted");

  try {
    const response = await fetch(`/api/conversations/${currentConvId}/canvas`, {
      method: "PATCH",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        document_id: activeDocument.id,
        content: nextContent,
        format: nextFormat,
        language: activeDocument.language || null,
      }),
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(payload.error || "Canvas save failed.");
    }

    history = Array.isArray(payload.messages) ? payload.messages.map(normalizeHistoryEntry) : history;
    streamingCanvasDocuments = [];
    pendingCanvasDiff = null;
    activeCanvasDocumentId = String(payload.active_document_id || activeDocument.id).trim() || activeDocument.id;
    isCanvasEditing = false;
    editingCanvasDocumentId = null;
    renderConversationHistory();
    renderCanvasPanel();
    setCanvasStatus("Canvas saved.", "success");
  } catch (error) {
    setCanvasStatus(error.message || "Canvas save failed.", "danger");
  }
}

function renderBubbleWithCursor(bubbleEl, text) {
  if (!bubbleEl) {
    return;
  }

  bubbleEl.classList.add("streaming-text");
  bubbleEl.innerHTML = renderMarkdown(text);

  const cursorEl = document.createElement("span");
  cursorEl.className = "stream-cursor";
  cursorEl.textContent = "▋";

  const textWalker = document.createTreeWalker(bubbleEl, NodeFilter.SHOW_TEXT, {
    acceptNode(node) {
      return node.nodeValue && node.nodeValue.trim() ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_SKIP;
    },
  });

  let lastTextNode = null;
  while (textWalker.nextNode()) {
    lastTextNode = textWalker.currentNode;
  }

  if (lastTextNode && lastTextNode.parentNode) {
    lastTextNode.parentNode.insertBefore(cursorEl, lastTextNode.nextSibling);
    return;
  }

  bubbleEl.appendChild(cursorEl);
}

function renderBubbleMarkdown(bubbleEl, text) {
  if (!bubbleEl) {
    return;
  }

  bubbleEl.classList.remove("streaming-text");
  bubbleEl.innerHTML = renderMarkdown(text);
}

const INPUT_BREAKDOWN_ORDER = [
  "core_instructions",
  "tool_specs",
  "canvas",
  "scratchpad",
  "tool_trace",
  "tool_memory",
  "rag_context",
  "internal_state",
  "user_messages",
  "assistant_history",
  "assistant_tool_calls",
  "tool_results",
  "unknown_provider_overhead",
];

const INPUT_BREAKDOWN_LABELS = {
  core_instructions: "Core instructions",
  tool_specs: "Tool definitions",
  canvas: "Canvas",
  scratchpad: "Scratchpad",
  tool_trace: "Tool trace",
  tool_memory: "Tool memory",
  rag_context: "RAG context",
  internal_state: "Agent working state",
  user_messages: "User messages",
  assistant_history: "Assistant history",
  assistant_tool_calls: "Assistant tool calls",
  tool_results: "Tool results",
  unknown_provider_overhead: "Unknown/Provider overhead",
};

const INPUT_BREAKDOWN_HELP_TEXT = {
  tool_specs: "Prompt tool list plus API function schema sent with the request.",
  internal_state: "Short internal working-memory instructions added during blocker handling or recovery.",
  unknown_provider_overhead: "The remaining billed prompt tokens left after local content, tool, and request-framing estimates are aligned to the provider total.",
};

const BREAKDOWN_WARNING_RATIO = 0.03;

const BREAKDOWN_REDUCTION_ORDER = [
  "tool_specs",
  "internal_state",
  "canvas",
  "scratchpad",
  "tool_trace",
  "tool_memory",
  "rag_context",
  "assistant_tool_calls",
  "tool_results",
  "assistant_history",
  "user_messages",
  "core_instructions",
];

const BREAKDOWN_FLOOR_KEYS = ["user_messages", "tool_results"];

const MODEL_CALL_TYPE_LABELS = {
  agent_step: "Agent step",
  final_answer: "Final answer",
};

const tokenTurns = [];

function createEmptyBreakdown() {
  return INPUT_BREAKDOWN_ORDER.reduce((acc, key) => {
    acc[key] = 0;
    return acc;
  }, {});
}

function toFiniteNumber(value, fallback = 0) {
  return Number.isFinite(Number(value)) ? Number(value) : fallback;
}

function toNonNegativeIntOrNull(value) {
  if (value === null || value === undefined || value === "") {
    return null;
  }
  const normalized = Number(value);
  if (!Number.isFinite(normalized)) {
    return null;
  }
  return Math.max(0, Math.round(normalized));
}

function getProtectedBreakdownKeys(breakdown, targetTotal) {
  const parsedTarget = toNonNegativeIntOrNull(targetTotal);
  if (parsedTarget === null || parsedTarget <= 0) {
    return new Set();
  }

  const presentKeys = BREAKDOWN_FLOOR_KEYS.filter((key) => toFiniteNumber(breakdown[key], 0) > 0);
  return new Set(presentKeys.slice(0, Math.min(presentKeys.length, parsedTarget)));
}

function alignBreakdownToTotal(breakdown, targetTotal) {
  const normalized = createEmptyBreakdown();
  INPUT_BREAKDOWN_ORDER.forEach((key) => {
    normalized[key] = Math.max(0, Math.round(toFiniteNumber(breakdown[key], 0)));
  });

  const parsedTarget = toNonNegativeIntOrNull(targetTotal);
  if (parsedTarget === null) {
    return normalized;
  }

  let currentTotal = sumBreakdown(normalized);
  if (currentTotal < parsedTarget) {
    normalized.unknown_provider_overhead += parsedTarget - currentTotal;
    return normalized;
  }

  let overflow = currentTotal - parsedTarget;
  if (overflow <= 0) {
    return normalized;
  }

  const protectedKeys = getProtectedBreakdownKeys(normalized, parsedTarget);

  BREAKDOWN_REDUCTION_ORDER.forEach((key) => {
    if (overflow <= 0) {
      return;
    }
    const floor = protectedKeys.has(key) ? 1 : 0;
    const available = (normalized[key] || 0) - floor;
    if (available <= 0) {
      return;
    }
    const reduction = Math.min(available, overflow);
    normalized[key] = available - reduction + floor;
    overflow -= reduction;
  });

  if (overflow > 0) {
    INPUT_BREAKDOWN_ORDER
      .slice()
      .sort((left, right) => (normalized[right] || 0) - (normalized[left] || 0))
      .forEach((key) => {
        if (overflow <= 0) {
          return;
        }
        const floor = protectedKeys.has(key) ? 1 : 0;
        const available = (normalized[key] || 0) - floor;
        if (available <= 0) {
          return;
        }
        const reduction = Math.min(available, overflow);
        normalized[key] = available - reduction + floor;
        overflow -= reduction;
      });
  }

  return normalized;
}

function normalizeBreakdown(rawBreakdown, targetTotal = null) {
  const normalized = createEmptyBreakdown();
  const source = rawBreakdown && typeof rawBreakdown === "object" ? rawBreakdown : {};
  const legacyCoreInstructions =
    toFiniteNumber(source.core_instructions, 0) +
    toFiniteNumber(source.system_prompt, 0) +
    toFiniteNumber(source.final_instruction, 0);
  INPUT_BREAKDOWN_ORDER.forEach((key) => {
    if (key === "core_instructions") {
      normalized[key] = Math.max(0, Math.round(toFiniteNumber(legacyCoreInstructions, 0)));
      return;
    }
    normalized[key] = Math.max(0, Math.round(toFiniteNumber(source[key], 0)));
  });
  return alignBreakdownToTotal(normalized, targetTotal);
}

function sumBreakdown(breakdown) {
  return INPUT_BREAKDOWN_ORDER.reduce((sum, key) => sum + toFiniteNumber(breakdown[key], 0), 0);
}

function getModelCallInputTokens(call) {
  if (!call || typeof call !== "object") {
    return 0;
  }

  const promptTokens = toNonNegativeIntOrNull(call.prompt_tokens);
  if (promptTokens !== null) {
    return promptTokens;
  }

  return toNonNegativeIntOrNull(call.estimated_input_tokens) ?? 0;
}

function getMaxInputTokensPerCall(modelCalls, fallbackPromptTokens = 0) {
  const peak = (Array.isArray(modelCalls) ? modelCalls : []).reduce(
    (maxValue, call) => Math.max(maxValue, getModelCallInputTokens(call)),
    0,
  );
  if (peak > 0) {
    return peak;
  }
  return Math.max(0, Math.round(toFiniteNumber(fallbackPromptTokens, 0)));
}

function normalizeModelCallPayload(callEntry) {
  const source = callEntry && typeof callEntry === "object" ? callEntry : {};
  const promptTokens = toNonNegativeIntOrNull(source.prompt_tokens);
  const completionTokens = toNonNegativeIntOrNull(source.completion_tokens);
  const totalTokens = toNonNegativeIntOrNull(source.total_tokens);
  const estimatedTarget = promptTokens ?? toNonNegativeIntOrNull(source.estimated_input_tokens);
  const inputBreakdown = normalizeBreakdown(source.input_breakdown, estimatedTarget);

  return {
    index: toNonNegativeIntOrNull(source.index),
    step: toNonNegativeIntOrNull(source.step),
    call_type: String(source.call_type || "agent_step") || "agent_step",
    is_retry: source.is_retry === true,
    retry_reason: String(source.retry_reason || "").trim(),
    message_count: toNonNegativeIntOrNull(source.message_count),
    tool_schema_tokens: toNonNegativeIntOrNull(source.tool_schema_tokens),
    prompt_tokens: promptTokens,
    completion_tokens: completionTokens,
    total_tokens: totalTokens,
    estimated_input_tokens: estimatedTarget ?? sumBreakdown(inputBreakdown),
    input_breakdown: inputBreakdown,
    missing_provider_usage: source.missing_provider_usage === true,
  };
}

function normalizeUsagePayload(usage) {
  const source = usage && typeof usage === "object" ? usage : {};
  const promptTokens = Math.max(0, Math.round(toFiniteNumber(source.prompt_tokens, 0)));
  const estimatedSourceTokens = Math.max(0, Math.round(toFiniteNumber(source.estimated_input_tokens, 0)));
  const inputBreakdown = normalizeBreakdown(source.input_breakdown, promptTokens || estimatedSourceTokens || null);
  const modelCalls = Array.isArray(source.model_calls)
    ? source.model_calls.map(normalizeModelCallPayload)
    : [];
  const modelCallCount = Math.max(
    modelCalls.length,
    Math.round(toFiniteNumber(source.model_call_count, 0)),
  );
  const estimatedInputTokens = promptTokens || sumBreakdown(inputBreakdown) || estimatedSourceTokens;
  const configuredPromptMaxInputTokens = toNonNegativeIntOrNull(source.configured_prompt_max_input_tokens);
  const maxInputTokensPerCall =
    toNonNegativeIntOrNull(source.max_input_tokens_per_call) ??
    getMaxInputTokensPerCall(modelCalls, promptTokens || estimatedInputTokens);

  return {
    prompt_tokens: promptTokens,
    completion_tokens: Math.max(0, Math.round(toFiniteNumber(source.completion_tokens, 0))),
    total_tokens: Math.max(0, Math.round(toFiniteNumber(source.total_tokens, 0))),
    estimated_input_tokens: estimatedInputTokens,
    input_breakdown: inputBreakdown,
    model_call_count: modelCallCount,
    model_calls: modelCalls,
    max_input_tokens_per_call: maxInputTokensPerCall,
    configured_prompt_max_input_tokens: configuredPromptMaxInputTokens,
    cost: Math.max(0, toFiniteNumber(source.cost, 0)),
    currency: String(source.currency || "USD") || "USD",
    model: String(source.model || "—") || "—",
  };
}

function aggregateBreakdown(turns) {
  const aggregate = createEmptyBreakdown();
  turns.forEach((turn) => {
    INPUT_BREAKDOWN_ORDER.forEach((key) => {
      aggregate[key] += toFiniteNumber(turn.input_breakdown[key], 0);
    });
  });
  return aggregate;
}

function getBreakdownWarningRatio(breakdown, totalTokens) {
  const total = Math.max(0, Math.round(toFiniteNumber(totalTokens, 0)));
  if (!total) {
    return 0;
  }
  return toFiniteNumber(breakdown.unknown_provider_overhead, 0) / total;
}

function renderBreakdownWarning(breakdown, totalTokens) {
  const ratio = getBreakdownWarningRatio(breakdown, totalTokens);
  if (ratio < BREAKDOWN_WARNING_RATIO) {
    return "";
  }

  return (
    `<div class="breakdown-warning">` +
      `Unknown/Provider overhead is ${Math.round(ratio * 1000) / 10}% of the billed prompt total.` +
    `</div>`
  );
}

function renderBreakdownList(containerId, breakdown, options = {}) {
  const container = document.getElementById(containerId);
  if (!container) {
    return;
  }

  const entries = INPUT_BREAKDOWN_ORDER.filter((key) => toFiniteNumber(breakdown[key], 0) > 0);
  if (!entries.length) {
    container.innerHTML = '<div class="breakdown-empty">No input-source estimate available yet.</div>';
    return;
  }

  container.innerHTML = entries
    .map(
      (key) => {
        const helpText = INPUT_BREAKDOWN_HELP_TEXT[key];
        const labelAttrs = helpText ? ` title="${escHtml(helpText)}"` : "";
        return (
        `<div class="breakdown-row">` +
          `<span class="breakdown-label"${labelAttrs}>${escHtml(INPUT_BREAKDOWN_LABELS[key] || key)}</span>` +
          `<span class="breakdown-value">${fmt(breakdown[key])}</span>` +
        `</div>`
        );
      },
    )
    .join("") + renderBreakdownWarning(breakdown, options.totalTokens);
}

function renderBreakdownChips(breakdown, className = "turn-breakdown-chip") {
  const entries = INPUT_BREAKDOWN_ORDER.filter((key) => toFiniteNumber(breakdown[key], 0) > 0);
  return entries
    .map(
      (key) =>
        `<span class="${className}">${escHtml(INPUT_BREAKDOWN_LABELS[key] || key)}: ${fmt(breakdown[key])}</span>`,
    )
    .join("");
}

function renderTurnBreakdownInline(breakdown) {
  const entries = INPUT_BREAKDOWN_ORDER.filter((key) => toFiniteNumber(breakdown[key], 0) > 0);
  if (!entries.length) {
    return "";
  }

  return (
    `<div class="turn-breakdown">` +
      renderBreakdownChips(breakdown) +
    `</div>`
  );
}

function renderUnknownWarningBadge(breakdown, totalTokens) {
  const ratio = getBreakdownWarningRatio(breakdown, totalTokens);
  if (ratio < BREAKDOWN_WARNING_RATIO) {
    return "";
  }
  return `<span class="turn-warning-badge">Unknown ${Math.round(ratio * 1000) / 10}%</span>`;
}

function renderModelCallItem(call) {
  const callTypeLabel = MODEL_CALL_TYPE_LABELS[call.call_type] || "Model call";
  const stepLabel = call.step ? ` · step ${call.step}` : "";
  const retryReason = call.retry_reason ? ` · ${call.retry_reason.replaceAll("_", " ")}` : "";
  const promptStat = call.prompt_tokens !== null
    ? `<span class="turn-call-stat">${fmt(call.prompt_tokens)} prompt</span>`
    : `<span class="turn-call-stat">${fmt(call.estimated_input_tokens)} estimated prompt</span>`;
  const completionStat = call.completion_tokens !== null
    ? `<span class="turn-call-stat">${fmt(call.completion_tokens)} completion</span>`
    : "";
  const messageCountStat = call.message_count !== null
    ? `<span class="turn-call-stat">${fmt(call.message_count)} messages</span>`
    : "";
  const schemaStat = call.tool_schema_tokens !== null && call.tool_schema_tokens > 0
    ? `<span class="turn-call-stat">${fmt(call.tool_schema_tokens)} tool schema</span>`
    : "";
  const missingBadge = call.missing_provider_usage
    ? `<span class="turn-call-badge">Missing provider usage</span>`
    : "";

  return (
    `<div class="turn-call-item">` +
      `<div class="turn-call-title-row">` +
        `<span class="turn-call-title">Call ${fmt(call.index || 0)} · ${escHtml(callTypeLabel)}${escHtml(stepLabel)}${escHtml(retryReason)}</span>` +
        missingBadge +
      `</div>` +
      `<div class="turn-call-meta">` +
        promptStat + completionStat + messageCountStat + schemaStat +
      `</div>` +
      `<div class="turn-call-breakdown">${renderBreakdownChips(call.input_breakdown, "turn-call-breakdown-chip")}</div>` +
    `</div>`
  );
}

function renderModelCallSection(title, calls) {
  if (!calls.length) {
    return "";
  }
  return (
    `<div class="turn-call-section">` +
      `<div class="turn-call-section-title">${escHtml(title)}</div>` +
      `<div class="turn-call-list">${calls.map(renderModelCallItem).join("")}</div>` +
    `</div>`
  );
}

function renderModelCallDrawer(turn) {
  const calls = Array.isArray(turn.model_calls) ? turn.model_calls : [];
  if (!calls.length) {
    return "";
  }

  const primaryCalls = calls.filter((call) => !call.is_retry);
  const retryCalls = calls.filter((call) => call.is_retry);
  return (
    `<details class="turn-call-drawer">` +
      `<summary class="turn-call-summary">View ${fmt(calls.length)} model calls</summary>` +
      `<div class="turn-call-sections">` +
        renderModelCallSection("Primary calls", primaryCalls) +
        renderModelCallSection("Retry and recovery calls", retryCalls) +
      `</div>` +
    `</details>`
  );
}

function renderTokenStats() {
  const totalUser = tokenTurns.reduce((sum, turn) => sum + turn.prompt_tokens, 0);
  const totalAsst = tokenTurns.reduce((sum, turn) => sum + turn.completion_tokens, 0);
  const grandTotal = tokenTurns.reduce((sum, turn) => sum + turn.total_tokens, 0);
  const totalCost = tokenTurns.reduce((sum, turn) => sum + turn.cost, 0);
  const sessionBreakdown = aggregateBreakdown(tokenTurns);
  const lastTurn = tokenTurns.length ? tokenTurns[tokenTurns.length - 1] : null;

  document.getElementById("stat-user").textContent = fmt(totalUser);
  document.getElementById("stat-asst").textContent = fmt(totalAsst);
  document.getElementById("stat-total").textContent = fmt(grandTotal);
  document.getElementById("stat-cost").textContent = "$" + totalCost.toFixed(6);
  document.getElementById("stat-last-input").textContent = lastTurn ? fmt(lastTurn.prompt_tokens) : "—";
  document.getElementById("stat-last-peak-input").textContent = lastTurn
    ? fmt(lastTurn.max_input_tokens_per_call)
    : "—";
  document.getElementById("stat-last-call-count").textContent = lastTurn
    ? fmt(lastTurn.model_call_count)
    : "—";
  document.getElementById("stat-last-prompt-cap").textContent = lastTurn && lastTurn.configured_prompt_max_input_tokens !== null
    ? fmt(lastTurn.configured_prompt_max_input_tokens)
    : "—";
  document.getElementById("stat-last-output").textContent = lastTurn ? fmt(lastTurn.completion_tokens) : "—";
  document.getElementById("stat-last-total").textContent = lastTurn ? fmt(lastTurn.total_tokens) : "—";
  document.getElementById("stat-last-model").textContent = lastTurn ? lastTurn.model : "—";
  document.getElementById("stat-breakdown-session-total").textContent = fmt(sumBreakdown(sessionBreakdown));
  document.getElementById("stat-breakdown-latest-total").textContent = lastTurn
    ? fmt(lastTurn.estimated_input_tokens)
    : "—";
  tokensBadge.textContent = fmt(grandTotal);

  renderBreakdownList("session-breakdown-list", sessionBreakdown, { totalTokens: totalUser });
  renderBreakdownList("latest-breakdown-list", lastTurn ? lastTurn.input_breakdown : createEmptyBreakdown(), {
    totalTokens: lastTurn ? lastTurn.prompt_tokens : 0,
  });

  const list = document.getElementById("turns-list");
  if (!tokenTurns.length) {
    list.innerHTML = '<div class="breakdown-empty">No completed assistant turns yet.</div>';
    return;
  }

  list.innerHTML = tokenTurns
    .map(
      (turn, index) => {
        const callCount = Math.max(turn.model_call_count || 0, Array.isArray(turn.model_calls) ? turn.model_calls.length : 0);
        const peakPromptStat = turn.max_input_tokens_per_call
          ? `<span class="turn-stat">${fmt(turn.max_input_tokens_per_call)} peak call prompt</span>`
          : "";
        const promptCapStat = turn.configured_prompt_max_input_tokens !== null
          ? `<span class="turn-stat">${fmt(turn.configured_prompt_max_input_tokens)} per-call prompt cap</span>`
          : "";
        return (
        `<div class="turn-item">` +
          `<div class="turn-header">` +
            `<span class="turn-label">Assistant turn ${index + 1}</span>` +
            `<div class="turn-header-meta">` +
              (callCount ? `<span class="turn-call-count">${fmt(callCount)} calls</span>` : "") +
              renderUnknownWarningBadge(turn.input_breakdown, turn.prompt_tokens) +
              `<span class="turn-model">${escHtml(turn.model || "—")}</span>` +
            `</div>` +
          `</div>` +
          `<div class="turn-details">` +
            `<span class="turn-stat"><span class="stats-dot dot-user"></span>${fmt(turn.prompt_tokens)} prompt (all calls)</span>` +
            peakPromptStat +
            promptCapStat +
            `<span class="turn-stat"><span class="stats-dot dot-asst"></span>${fmt(turn.completion_tokens)} completion</span>` +
            `<span class="turn-stat">${fmt(turn.total_tokens)} total</span>` +
            (turn.cost ? `<span class="turn-stat cost-stat">$${turn.cost.toFixed(6)}</span>` : "") +
          `</div>` +
          renderTurnBreakdownInline(turn.input_breakdown) +
          renderModelCallDrawer(turn) +
        `</div>`
        );
      },
    )
    .join("");
}

function normalizeHistoryEntry(entry) {
  const source = entry && typeof entry === "object" ? entry : {};
  const normalizedId = Number(source.id);
  const usage = source.usage && typeof source.usage === "object" ? normalizeUsagePayload(source.usage) : null;
  const role = ["assistant", "user", "tool", "system", "summary"].includes(source.role) ? source.role : "user";
  const toolCalls = Array.isArray(source.tool_calls) ? source.tool_calls : [];
  const toolCallId = typeof source.tool_call_id === "string" && source.tool_call_id.trim()
    ? source.tool_call_id.trim()
    : null;
  return {
    id: Number.isInteger(normalizedId) ? normalizedId : null,
    role,
    content: String(source.content || ""),
    metadata: source.metadata && typeof source.metadata === "object" ? source.metadata : null,
    tool_calls: toolCalls,
    tool_call_id: toolCallId,
    usage,
  };
}

function buildRequestMessagesFromHistory(entries = history) {
  return entries.map((item) => ({
    role: item.role,
    content: item.content,
    metadata: item.metadata || null,
    tool_calls: Array.isArray(item.tool_calls) ? item.tool_calls : [],
    tool_call_id: item.tool_call_id || null,
  }));
}

function isRenderableHistoryEntry(message) {
  if (!message) {
    return false;
  }

  if (message.role === "assistant" && Array.isArray(message.tool_calls) && message.tool_calls.length > 0) {
    return false;
  }

  return message.role === "user" || message.role === "assistant" || message.role === "summary";
}

function getVisibleHistoryEntries(entries = history) {
  return entries.filter(isRenderableHistoryEntry);
}

function getConversationSignature(entries = history) {
  return getVisibleHistoryEntries(entries)
    .map((message) => {
      const metadata = message.metadata ? JSON.stringify(message.metadata) : "";
      return `${message.role}:${message.content}:${metadata}`;
    })
    .join("\u0001");
}

function buildAssistantMetadata({
  reasoning = "",
  toolTrace = [],
  tool_trace = null,
  toolResults = [],
  tool_results = null,
  canvasDocuments = [],
  canvas_documents = null,
  usage = null,
  pendingClarification = null,
  pending_clarification = null,
} = {}) {
  const normalizedToolTrace = Array.isArray(tool_trace) ? tool_trace : toolTrace;
  const normalizedToolResults = Array.isArray(tool_results) ? tool_results : toolResults;
  const normalizedCanvasDocuments = Array.isArray(canvas_documents) ? canvas_documents : canvasDocuments;
  const normalizedPendingClarification = pending_clarification && typeof pending_clarification === "object"
    ? pending_clarification
    : pendingClarification && typeof pendingClarification === "object"
      ? pendingClarification
      : null;

  return reasoning || usage || normalizedToolResults.length || normalizedToolTrace.length || normalizedCanvasDocuments.length || normalizedPendingClarification
    ? {
        ...(reasoning ? { reasoning_content: reasoning } : {}),
        ...(normalizedToolTrace.length ? { tool_trace: normalizedToolTrace } : {}),
        ...(normalizedToolResults.length ? { tool_results: normalizedToolResults } : {}),
        ...(normalizedCanvasDocuments.length ? { canvas_documents: normalizedCanvasDocuments } : {}),
        ...(normalizedPendingClarification ? { pending_clarification: normalizedPendingClarification } : {}),
        ...(usage ? { usage } : {}),
      }
    : null;
}

function normalizeClarificationQuestion(question, index) {
  if (!question || typeof question !== "object") {
    return null;
  }

  const inputType = String(question.input_type || "text").trim();
  if (!["text", "single_select", "multi_select"].includes(inputType)) {
    return null;
  }

  const label = String(question.label || "").trim();
  if (!label) {
    return null;
  }

  const normalized = {
    id: String(question.id || `question_${index + 1}`).trim() || `question_${index + 1}`,
    label,
    input_type: inputType,
    required: question.required !== false,
    placeholder: String(question.placeholder || "").trim(),
    allow_free_text: question.allow_free_text === true,
    options: [],
  };

  const rawOptions = Array.isArray(question.options) ? question.options : [];
  normalized.options = rawOptions
    .map((option) => {
      if (!option || typeof option !== "object") {
        return null;
      }
      const optionLabel = String(option.label || option.value || "").trim();
      const optionValue = String(option.value || option.label || "").trim();
      const optionDescription = String(option.description || "").trim();
      if (!optionLabel || !optionValue) {
        return null;
      }
      return {
        label: optionLabel,
        value: optionValue,
        ...(optionDescription ? { description: optionDescription } : {}),
      };
    })
    .filter(Boolean);

  return normalized;
}

function getPendingClarification(metadata) {
  if (!metadata || typeof metadata !== "object") {
    return null;
  }

  const payload = metadata.pending_clarification;
  if (!payload || typeof payload !== "object") {
    return null;
  }

  const questions = Array.isArray(payload.questions)
    ? payload.questions.map(normalizeClarificationQuestion).filter(Boolean)
    : [];
  if (!questions.length) {
    return null;
  }

  return {
    intro: String(payload.intro || "").trim(),
    submit_label: String(payload.submit_label || "").trim() || "Send answers",
    questions,
  };
}

function formatClarificationResponse(clarification, answers) {
  const lines = ["Clarification answers:"];
  clarification.questions.forEach((question) => {
    const answer = answers[question.id];
    if (!answer || !String(answer.display || "").trim()) {
      return;
    }
    lines.push(`- ${question.label}: ${String(answer.display).trim()}`);
  });
  return lines.join("\n");
}

function collectClarificationAnswers(form, clarification) {
  const answers = {};

  for (let index = 0; index < clarification.questions.length; index += 1) {
    const question = clarification.questions[index];
    const fieldName = `clarify_${index}`;
    const freeTextName = `${fieldName}_free`;
    let display = "";
    let value = null;

    if (question.input_type === "text") {
      const input = form.elements[fieldName];
      display = String(input?.value || "").trim();
      value = display;
    } else if (question.input_type === "single_select") {
      const selected = form.querySelector(`input[name="${fieldName}"]:checked`);
      value = selected ? String(selected.value || "").trim() : "";
      const selectedOption = question.options.find((option) => option.value === value);
      display = selectedOption ? selectedOption.label : value;
    } else if (question.input_type === "multi_select") {
      const selected = Array.from(form.querySelectorAll(`input[name="${fieldName}"]:checked`));
      value = selected.map((element) => String(element.value || "").trim()).filter(Boolean);
      display = value
        .map((entry) => question.options.find((option) => option.value === entry)?.label || entry)
        .filter(Boolean)
        .join(", ");
    }

    const freeTextInput = form.elements[freeTextName];
    const freeText = String(freeTextInput?.value || "").trim();
    if (freeText) {
      display = display ? `${display} (${freeText})` : freeText;
      if (Array.isArray(value)) {
        value = [...value, freeText];
      } else if (value) {
        value = { selection: value, free_text: freeText };
      } else {
        value = freeText;
      }
    }

    if (question.required && !String(display || "").trim()) {
      return { error: `${question.label} is required.` };
    }

    answers[question.id] = { value, display };
  }

  return {
    answers,
    text: formatClarificationResponse(clarification, answers),
  };
}

function shouldGenerateConversationTitle() {
  const visibleEntries = getVisibleHistoryEntries();
  return Boolean(
    currentConvId &&
    visibleEntries.length === 2 &&
    visibleEntries[0]?.role === "user" &&
    visibleEntries[1]?.role === "assistant" &&
    !getPendingClarification(visibleEntries[1]?.metadata),
  );
}

async function streamNdjsonResponse(response, onEvent) {
  if (!response.body) {
    throw new Error("The server returned an empty response stream.");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  const processLine = (line) => {
    if (!line.trim()) {
      return;
    }
    try {
      onEvent(JSON.parse(line));
    } catch (_) {
      // Ignore malformed partial chunks.
    }
  };

  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      break;
    }
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";
    lines.forEach(processLine);
  }

  buffer += decoder.decode();
  processLine(buffer);
}

function getHistoryMessageIndex(messageId) {
  const normalizedId = Number(messageId);
  if (!Number.isInteger(normalizedId) || normalizedId <= 0) {
    return -1;
  }
  return history.findIndex((item) => Number(item.id) === normalizedId);
}

function isPersistedMessageId(messageId) {
  const normalizedId = Number(messageId);
  return Number.isInteger(normalizedId) && normalizedId > 0;
}

function getHistoryMessage(messageId) {
  const index = getHistoryMessageIndex(messageId);
  return index >= 0 ? history[index] : null;
}

function isPrunableHistoryMessage(message) {
  if (!message || (message.role !== "user" && message.role !== "assistant")) {
    return false;
  }
  if (message.role === "assistant" && Array.isArray(message.tool_calls) && message.tool_calls.length > 0) {
    return false;
  }
  const metadata = message.metadata && typeof message.metadata === "object" ? message.metadata : null;
  if (metadata?.is_summary === true || metadata?.is_pruned === true) {
    return false;
  }
  return String(message.content || "").trim().length > 0;
}

function clearEditTarget() {
  editingMessageId = null;
  editBanner.hidden = true;
  editBannerText.textContent = "";
}

function refreshEditBanner() {
  const message = getHistoryMessage(editingMessageId);
  if (!message || message.role !== "user") {
    clearEditTarget();
    return;
  }

  editBanner.hidden = false;
  editBannerText.textContent = "Editing an earlier message. Sending now will replace that turn and continue from there.";
}

function beginEditingMessage(messageId) {
  if (isStreaming || isFixing) {
    return;
  }

  const message = getHistoryMessage(messageId);
  if (!message || message.role !== "user") {
    return;
  }

  editingMessageId = Number(message.id);
  inputEl.value = message.content;
  autoResize(inputEl);
  clearSelectedImage();
  refreshEditBanner();
  renderConversationHistory();
  inputEl.focus();
  inputEl.setSelectionRange(inputEl.value.length, inputEl.value.length);
}

function createMessageActions(message, options = {}) {
  if (!message) {
    return null;
  }

  const actions = document.createElement("div");
  actions.className = "msg-actions";

  const messageId = message.id;

  if (message.role === "user" && options.editable) {
    const editBtn = document.createElement("button");
    editBtn.type = "button";
    editBtn.className = "msg-action-btn";
    editBtn.textContent = "Edit";
    editBtn.disabled = !isPersistedMessageId(messageId);
    editBtn.addEventListener("click", () => beginEditingMessage(messageId));
    actions.appendChild(editBtn);
  }

  const documents = getCanvasDocuments(message.metadata);
  if (message.role === "assistant" && documents.length) {
    const openBtn = document.createElement("button");
    openBtn.type = "button";
    openBtn.className = "msg-action-btn";
    openBtn.textContent = "Open canvas";
    openBtn.addEventListener("click", () => {
      activeCanvasDocumentId = documents[documents.length - 1].id;
      openCanvas();
      setCanvasStatus("Canvas ready.", "muted");
    });
    actions.appendChild(openBtn);
  }

  if (isPrunableHistoryMessage(message)) {
    // per-message prune removed; use the global Prune history button instead
  }

  if (!actions.childElementCount) {
    return null;
  }
  return actions;
}

async function pruneMessage(messageId) {
  if (isStreaming || isFixing) {
    return;
  }

  const message = getHistoryMessage(messageId);
  if (!isPrunableHistoryMessage(message)) {
    return;
  }

  try {
    const response = await fetch(`/api/messages/${messageId}/prune`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ conversation_id: currentConvId }),
    });
    const data = await response.json().catch(() => null);
    if (!response.ok) {
      throw new Error(data?.error || "Message could not be pruned.");
    }
    if (!data?.message) {
      throw new Error("Pruned message payload is missing.");
    }

    const index = getHistoryMessageIndex(messageId);
    if (index >= 0) {
      history[index] = normalizeHistoryEntry(data.message);
      renderConversationHistory();
    } else {
      await refreshConversationFromServer();
    }
    showToast("Message pruned.", "success");
  } catch (error) {
    showError(error.message || "Message could not be pruned.");
  }
}

function renderConversationHistory(options = {}) {
  const preserveScroll = options && options.preserveScroll === true;
  const previousDistanceFromBottom = preserveScroll
    ? messagesEl.scrollHeight - messagesEl.scrollTop - messagesEl.clientHeight
    : 0;
  const fragment = document.createDocumentFragment();
  fragment.appendChild(emptyState);

  if (!history.length) {
    emptyState.style.display = "";
    messagesEl.replaceChildren(fragment);
    scrollToBottom();
    renderSummaryInspector();
    return;
  }

  emptyState.style.display = "none";
  const visibleEntries = getVisibleHistoryEntries();
  visibleEntries.forEach((message, index) => {
    if (!isRenderableHistoryEntry(message)) {
      return;
    }
    fragment.appendChild(createMessageGroup(message.role, message.content, message.metadata || null, {
      messageId: message.id,
      editable: message.role === "user",
      isEditingTarget: isPersistedMessageId(message.id)
        && isPersistedMessageId(editingMessageId)
        && Number(message.id) === Number(editingMessageId),
      isLatestVisible: index === visibleEntries.length - 1,
      toolCalls: message.tool_calls,
    }));
  });
  messagesEl.replaceChildren(fragment);
  if (preserveScroll) {
    if (previousDistanceFromBottom <= 100) {
      scrollToBottom();
    } else {
      messagesEl.scrollTop = Math.max(0, messagesEl.scrollHeight - messagesEl.clientHeight - previousDistanceFromBottom);
    }
  } else {
    scrollToBottom();
  }
  renderSummaryInspector();
}

async function refreshConversationFromServer() {
  if (!currentConvId) {
    return false;
  }

  const response = await fetch(`/api/conversations/${currentConvId}`);
  if (!response.ok) {
    return false;
  }

  const data = await response.json().catch(() => null);
  if (!data || Number(data.conversation?.id) !== Number(currentConvId)) {
    return false;
  }

  const serverHistory = Array.isArray(data.messages) ? data.messages.map(normalizeHistoryEntry) : [];
  const serverSignature = getConversationSignature(serverHistory);
  if (serverSignature === lastConversationSignature) {
    return false;
  }

  history = serverHistory;
  currentConvTitle = String(data.conversation?.title || currentConvTitle || "New Chat").trim() || "New Chat";
  latestSummaryStatus = null;
  streamingCanvasDocuments = [];
  resetStreamingCanvasPreview();
  activeCanvasDocumentId = getActiveCanvasDocument(history)?.id || null;
  lastConversationSignature = serverSignature;
  renderConversationHistory({ preserveScroll: true });
  renderCanvasPanel();
  updateExportPanel();
  rebuildTokenStatsFromHistory();
  loadSidebar();
  return true;
}

function scheduleConversationRefreshAfterStream() {
  if (!currentConvId) {
    return;
  }

  const refreshGeneration = ++conversationRefreshGeneration;
  pendingConversationRefreshTimers.forEach((timerId) => window.clearTimeout(timerId));
  pendingConversationRefreshTimers.clear();

  [800, 2000, 5000, 10000].forEach((delay) => {
    const timerId = window.setTimeout(async () => {
      pendingConversationRefreshTimers.delete(timerId);
      if (refreshGeneration !== conversationRefreshGeneration || !currentConvId || isStreaming || isFixing) {
        return;
      }

      try {
        const refreshed = await refreshConversationFromServer();
        if (refreshed) {
          pendingConversationRefreshTimers.forEach((pendingTimerId) => window.clearTimeout(pendingTimerId));
          pendingConversationRefreshTimers.clear();
        }
      } catch (_) {
        // Ignore transient refresh errors and keep polling.
      }
    }, delay);
    pendingConversationRefreshTimers.add(timerId);
  });
}

function rebuildTokenStatsFromHistory() {
  resetTokenStats();
  history.forEach((message) => {
    if (message.role === "assistant" && message.usage) {
      updateStats(message.usage);
    }
  });
}

function createAssistantStreamingGroup() {
  const asstGroup = document.createElement("div");
  asstGroup.className = "msg-group assistant";

  const metaRow = document.createElement("div");
  metaRow.className = "msg-meta-row";

  const asstLabel = document.createElement("div");
  asstLabel.className = "msg-label";
  asstLabel.textContent = "Assistant";

  metaRow.appendChild(asstLabel);

  const stepLog = document.createElement("div");
  stepLog.className = "step-log";
  stepLog.style.display = "none";

  const asstBubble = document.createElement("div");
  asstBubble.className = "bubble thinking cursor";
  asstBubble.textContent = "Working...";

  asstGroup.appendChild(metaRow);
  asstGroup.appendChild(stepLog);
  asstGroup.appendChild(asstBubble);
  messagesEl.appendChild(asstGroup);
  scrollToBottom();

  return { asstGroup, stepLog, asstBubble };
}

function finalizeAssistantStreamingGroup(asstGroup, stepLog, metadata) {
  if (!asstGroup) {
    return;
  }

  if (stepLog) {
    stepLog.style.display = "none";
  }

  updateAssistantFetchBadge(asstGroup, metadata);
  updateAssistantToolTrace(asstGroup, metadata);
  updateReasoningPanel(asstGroup, getReasoningText(metadata));
  appendClarificationPanel(asstGroup, metadata, {});
}

function applyPersistedMessageIds(persistedIds, assistantEntry) {
  if (!persistedIds || typeof persistedIds !== "object") {
    return;
  }

  const userId = Number(persistedIds.user_message_id);
  if (isPersistedMessageId(userId)) {
    for (let index = history.length - 1; index >= 0; index -= 1) {
      if (history[index].role === "user") {
        history[index].id = userId;
        break;
      }
    }
  }

  const assistantId = Number(persistedIds.assistant_message_id);
  if (assistantEntry && isPersistedMessageId(assistantId)) {
    assistantEntry.id = assistantId;
  }
}

function updateStats(usage) {
  tokenTurns.push(normalizeUsagePayload(usage));
  renderTokenStats();
}

function fmt(value) {
  return Number.isFinite(value) ? value.toLocaleString() : "—";
}

function estimateLocalTokens(text) {
  const normalized = String(text || "").trim();
  if (!normalized) {
    return 0;
  }

  const words = normalized.split(/\s+/).filter(Boolean).length;
  const charEstimate = normalized.length / 4;
  const wordEstimate = words * 1.35;
  return Math.max(1, Math.round(Math.max(charEstimate, wordEstimate)));
}

function getSummaryModeValue() {
  return String(appSettings.chat_summary_mode || "auto").trim() || "auto";
}

function getSummaryTriggerValue() {
  const rawValue = parseInt(appSettings.chat_summary_trigger_token_count || 80000, 10);
  return Number.isFinite(rawValue) ? rawValue : 80000;
}

function getEffectiveSummaryTriggerValue() {
  const baseTrigger = getSummaryTriggerValue();
  return getSummaryModeValue() === "aggressive"
    ? Math.max(1000, Math.floor(baseTrigger / 2))
    : baseTrigger;
}

function estimateSummaryTriggerTokens(entries = history) {
  return (entries || []).reduce((total, entry) => {
    const role = String(entry?.role || "").trim();
    if (!role) {
      return total;
    }
    if (role === "assistant" && Array.isArray(entry?.tool_calls) && entry.tool_calls.length > 0) {
      return total;
    }
    if (!["user", "assistant", "tool", "summary"].includes(role)) {
      return total;
    }
    return total + estimateLocalTokens(entry.content);
  }, 0);
}

function findLatestSummaryEntry(entries = history) {
  for (let index = (entries || []).length - 1; index >= 0; index -= 1) {
    const entry = entries[index];
    if (entry?.role === "summary") {
      return entry;
    }
  }
  return null;
}

function formatSummaryTimestamp(value) {
  const timestamp = String(value || "").trim();
  if (!timestamp) {
    return "—";
  }
  const date = new Date(timestamp);
  if (Number.isNaN(date.getTime())) {
    return timestamp;
  }
  return date.toLocaleString();
}

function describeSummaryFailure(status) {
  const reason = String(status?.reason || "").trim();
  const stage = String(status?.failure_stage || "").trim();
  const detail = String(status?.failure_detail || "").trim();

  if (reason === "mode_never") {
    return "Auto summary is disabled by settings.";
  }
  if (reason === "below_threshold") {
    const tokenGap = Number(status?.token_gap || 0);
    return tokenGap > 0
      ? `Below threshold by ${fmt(tokenGap)} counted tokens.`
      : "Still below the summary trigger threshold.";
  }
  if (reason === "no_source_messages") {
    return "There are no older unsummarized user or assistant messages left to compress.";
  }
  if (reason === "no_prompt_messages") {
    return "Candidate messages existed, but all of them were empty or invalid after prompt sanitization.";
  }
  if (reason === "locked") {
    return "Another summary pass was already running, so this turn skipped a duplicate summary attempt.";
  }
  if (reason !== "summary_generation_failed") {
    return detail || "Waiting for the next completed assistant turn to evaluate summary conditions.";
  }

  if (stage === "context_too_large") {
    return "The provider rejected the summary request because the summary prompt itself exceeded the model context limit.";
  }
  if (stage === "invalid_message_sequence") {
    return "The provider rejected the summary prompt because the message sequence was invalid.";
  }
  if (stage === "tool_call_unexpected") {
    return "The model attempted a tool-style response during summary generation, so the result was rejected for safety.";
  }
  if (stage === "empty_output") {
    return "The provider returned no assistant summary content.";
  }
  if (stage === "too_short") {
    return "The provider returned a summary that was too short to keep as reliable compressed context.";
  }
  if (stage === "provider_error") {
    return "The provider returned an error while generating the summary.";
  }
  return detail || "The summary attempt failed validation, so no messages were compressed.";
}

function renderSummaryInspector() {
  if (!summaryInspectorBadge || !summaryInspectorHeadline) {
    return;
  }

  const mode = getSummaryModeValue();
  const effectiveTrigger = getEffectiveSummaryTriggerValue();
  const currentEstimate = estimateSummaryTriggerTokens(history);
  const latestSummary = findLatestSummaryEntry(history);
  const lastSummaryMeta = latestSummary?.metadata && typeof latestSummary.metadata === "object"
    ? latestSummary.metadata
    : null;
  const canUndoLatestSummary = Boolean(currentConvId && Number.isInteger(Number(latestSummary?.id)) && lastSummaryMeta?.is_summary);
  const remaining = Math.max(0, effectiveTrigger - currentEstimate);
  const overBy = Math.max(0, currentEstimate - effectiveTrigger);
  const latestServerCheck = Number(latestSummaryStatus?.visible_token_count || 0);
  const mergedAssistantCount = Number(latestSummaryStatus?.merged_assistant_message_count || 0);
  const skippedErrorCount = Number(latestSummaryStatus?.skipped_error_message_count || 0);

  let badgeTone = "muted";
  let badgeText = "Waiting";
  let headline = "Start chatting to see when automatic summary will run.";

  if (mode === "never") {
    badgeTone = "warning";
    badgeText = "Disabled";
    headline = "Auto summary is disabled. Long chats will keep growing until you re-enable it.";
  } else if (!currentConvId) {
    badgeTone = "muted";
    badgeText = "Idle";
    headline = "Auto summary watches each conversation after messages are saved.";
  } else if (latestSummaryStatus?.reason === "summary_generation_failed") {
    badgeTone = "error";
    badgeText = "Failed";
    headline = describeSummaryFailure(latestSummaryStatus);
  } else if (latestSummaryStatus?.reason === "locked") {
    badgeTone = "warning";
    badgeText = "Busy";
    headline = "A summary pass was already in progress for this conversation, so this turn skipped a duplicate run.";
  } else if (currentEstimate >= effectiveTrigger) {
    badgeTone = "accent";
    badgeText = "Ready";
    headline = "This conversation is large enough for auto summary. The next completed turn can compress older user and assistant messages.";
  } else if (latestSummary) {
    badgeTone = "success";
    badgeText = "Tracked";
    headline = "This conversation has already been summarized before and is being monitored for the next pass.";
  } else {
    badgeTone = "muted";
    badgeText = "Tracking";
    headline = `Current conversation is ${fmt(remaining)} estimated tokens below the next auto-summary pass.`;
  }

  summaryInspectorBadge.dataset.tone = badgeTone;
  summaryInspectorBadge.textContent = badgeText;
  summaryInspectorHeadline.textContent = headline;
  summaryInspectorCurrent.textContent = fmt(currentEstimate);
  summaryInspectorTrigger.textContent = fmt(effectiveTrigger);
  summaryInspectorGap.textContent = overBy > 0 ? `${fmt(overBy)} over` : `${fmt(remaining)} left`;

  const detailParts = [
    "Counts user, assistant, tool, and summary history toward the summary trigger, while ignoring assistant tool-call placeholders.",
    "Does not count runtime system prompt, RAG context, or final-answer instructions.",
  ];
  if (latestServerCheck > 0) {
    detailParts.push(`Last server-side check saw ${fmt(latestServerCheck)} counted tokens.`);
  }
  if (mergedAssistantCount > 0) {
    detailParts.push(`Merged ${fmt(mergedAssistantCount)} consecutive assistant blocks before sending the summary prompt.`);
  }
  if (skippedErrorCount > 0) {
    detailParts.push(`Skipped ${fmt(skippedErrorCount)} assistant error placeholders during prompt cleanup.`);
  }
  summaryInspectorDetail.textContent = detailParts.join(" ");

  if (summaryInspectorToolMessages) {
    const toolMessageParts = [
      "Assistant tool-call placeholders are excluded from the trigger estimate.",
      "Tool outputs and cleaned assistant content still count toward summary readiness.",
    ];
    if (mergedAssistantCount > 0) {
      toolMessageParts.push(`${fmt(mergedAssistantCount)} assistant blocks were merged during the latest cleanup pass.`);
    }
    if (skippedErrorCount > 0) {
      toolMessageParts.push(`${fmt(skippedErrorCount)} assistant error placeholders were skipped.`);
    }
    summaryInspectorToolMessages.textContent = toolMessageParts.join(" ");
  }

  if (summaryInspectorReason) {
    let reasonText = "The latest summary decision will appear here after each completed assistant turn.";
    if (latestSummaryStatus) {
      if (latestSummaryStatus.reason === "summary_generation_failed" || latestSummaryStatus.reason === "locked" || latestSummaryStatus.reason === "below_threshold" || latestSummaryStatus.reason === "mode_never") {
        reasonText = describeSummaryFailure(latestSummaryStatus);
      } else if (latestSummaryStatus.applied) {
        reasonText = "Latest summary pass completed successfully.";
      } else {
        reasonText = "Latest summary check completed.";
      }
      const failureDetail = String(latestSummaryStatus.failure_detail || "").trim();
      const failureSummary = describeSummaryFailure(latestSummaryStatus);
      if (failureDetail && failureDetail !== failureSummary) {
        reasonText = `${reasonText} ${failureDetail}`;
      }
    }
    summaryInspectorReason.textContent = reasonText;
  }

  if (lastSummaryMeta?.is_summary) {
    const generatedAt = formatSummaryTimestamp(lastSummaryMeta.generated_at);
    const coveredCount = Number(lastSummaryMeta.covered_message_count || 0);
    const summaryModel = String(lastSummaryMeta.summary_model || latestSummaryStatus?.summary_model || "—").trim() || "—";
    summaryInspectorLast.textContent = `Last pass: ${fmt(coveredCount)} messages compressed on ${generatedAt} using ${summaryModel}.`;
  } else if (latestSummaryStatus?.reason === "summary_generation_failed") {
    const checkedAt = formatSummaryTimestamp(latestSummaryStatus.checked_at);
    summaryInspectorLast.textContent = `Last attempt: failed on ${checkedAt}. No messages were deleted because failed summaries are never applied.`;
  } else if (latestSummaryStatus?.reason === "below_threshold") {
    summaryInspectorLast.textContent = "No summary pass was needed on the latest turn because the conversation stayed below the trigger.";
  } else if (latestSummaryStatus?.reason === "mode_never") {
    summaryInspectorLast.textContent = "No summary pass will run while summary mode is set to Never.";
  } else {
    summaryInspectorLast.textContent = "No summary pass has run in this conversation yet.";
  }

  if (summaryUndoBtn) {
    summaryUndoBtn.disabled = !canUndoLatestSummary;
    summaryUndoBtn.title = canUndoLatestSummary
      ? "Restore the messages covered by the latest summary."
      : "No summary is available to undo in this conversation.";
  }
}

function openStats() {
  closeMobileTools();
  closeCanvas();
  closeExportPanel();
  statsPanel.classList.add("open");
  statsOverlay.classList.add("open");
}

function closeStats() {
  statsPanel.classList.remove("open");
  statsOverlay.classList.remove("open");
}

function openMobileTools() {
  closeStats();
  closeCanvas();
  closeExportPanel();
  mobileToolsPanel?.classList.add("open");
  mobileToolsOverlay?.classList.add("open");
  mobileToolsBtn?.setAttribute("aria-expanded", "true");
  mobileToolsPanel?.setAttribute("aria-hidden", "false");
}

function closeMobileTools() {
  mobileToolsPanel?.classList.remove("open");
  mobileToolsOverlay?.classList.remove("open");
  mobileToolsBtn?.setAttribute("aria-expanded", "false");
  mobileToolsPanel?.setAttribute("aria-hidden", "true");
}

function syncModelSelectors(value) {
  const nextValue = String(value || "");
  if (modelSel && modelSel.value !== nextValue) {
    modelSel.value = nextValue;
  }
  if (mobileModelSel && mobileModelSel.value !== nextValue) {
    mobileModelSel.value = nextValue;
  }
}

function isMobileViewport() {
  return window.matchMedia("(max-width: 980px)").matches;
}

function updateHeaderOffset() {
  if (!headerEl) {
    return;
  }
  document.documentElement.style.setProperty("--header-offset", `${headerEl.offsetHeight}px`);
}

function readSidebarPreference() {
  try {
    const stored = localStorage.getItem(SIDEBAR_STORAGE_KEY);
    if (stored === null) {
      return null;
    }
    return stored === "true";
  } catch (_) {
    return null;
  }
}

function writeSidebarPreference(isOpen) {
  try {
    localStorage.setItem(SIDEBAR_STORAGE_KEY, String(Boolean(isOpen)));
  } catch (_) {
    // Ignore storage errors.
  }
}

function updateSidebarToggleLabel(isOpen) {
  if (!sidebarToggleBtn) {
    return;
  }
  sidebarToggleBtn.setAttribute("aria-expanded", String(Boolean(isOpen)));
  sidebarToggleBtn.title = isOpen ? "Hide conversations" : "Show conversations";
}

function setSidebarOpen(isOpen, persist = true) {
  document.body.classList.toggle("sidebar-collapsed", !isOpen);
  updateSidebarToggleLabel(isOpen);
  if (persist) {
    writeSidebarPreference(isOpen);
  }
}

function toggleSidebar() {
  const isOpen = !document.body.classList.contains("sidebar-collapsed");
  setSidebarOpen(!isOpen);
}

function closeSidebarOnMobile() {
  if (isMobileViewport()) {
    setSidebarOpen(false);
  }
}

statsClose.addEventListener("click", closeStats);
statsOverlay.addEventListener("click", closeStats);
if (canvasClose) {
  canvasClose.addEventListener("click", closeCanvas);
}
if (canvasOverlay) {
  canvasOverlay.addEventListener("click", closeCanvas);
}
if (canvasEditBtn) {
  canvasEditBtn.addEventListener("click", () => setCanvasEditing(true));
}
if (canvasSaveBtn) {
  canvasSaveBtn.addEventListener("click", () => {
    void saveCanvasEdits();
  });
}
if (canvasCancelBtn) {
  canvasCancelBtn.addEventListener("click", () => {
    isCanvasEditing = false;
    editingCanvasDocumentId = null;
    renderCanvasPanel();
    setCanvasStatus("Canvas edit cancelled.", "muted");
  });
}
if (mobileCanvasBtn) {
  mobileCanvasBtn.addEventListener("click", () => openCanvas(mobileToolsBtn || mobileCanvasBtn));
}
if (canvasToggleBtn) {
  canvasToggleBtn.addEventListener("click", () => {
    if (isCanvasOpen()) {
      closeCanvas();
    } else {
      openCanvas(canvasToggleBtn);
    }
  });
}
if (mobileExportBtn) {
  mobileExportBtn.addEventListener("click", () => openExportPanel(mobileToolsBtn || mobileExportBtn));
}
if (mobilePruneBtn) {
  mobilePruneBtn.addEventListener("click", () => {
    closeMobileTools();
    void pruneConversationHistory();
  });
}
if (exportClose) {
  exportClose.addEventListener("click", closeExportPanel);
}
if (exportOverlay) {
  exportOverlay.addEventListener("click", closeExportPanel);
}
if (conversationExportMdBtn) {
  conversationExportMdBtn.addEventListener("click", () => downloadConversation("md"));
}
if (conversationExportDocxBtn) {
  conversationExportDocxBtn.addEventListener("click", () => downloadConversation("docx"));
}
if (conversationExportPdfBtn) {
  conversationExportPdfBtn.addEventListener("click", () => downloadConversation("pdf"));
}
if (canvasCopyBtn) {
  canvasCopyBtn.addEventListener("click", async () => {
    const document = getCanvasDocumentById(getCanvasRenderableDocuments(), activeCanvasDocumentId) || getActiveCanvasDocument();
    if (!document || !navigator.clipboard) {
      setCanvasStatus("Clipboard is not available.", "warning");
      return;
    }
    try {
      await navigator.clipboard.writeText(document.content || "");
      setCanvasStatus("Canvas copied to clipboard.", "success");
    } catch (_) {
      setCanvasStatus("Copy failed.", "danger");
    }
  });
}
if (canvasCopyRefBtn) {
  canvasCopyRefBtn.addEventListener("click", async () => {
    const document = getCanvasDocumentById(getCanvasRenderableDocuments(), activeCanvasDocumentId) || getActiveCanvasDocument();
    const reference = getCanvasDocumentReference(document);
    if (!reference || !navigator.clipboard) {
      setCanvasStatus("Reference copy is not available.", "warning");
      return;
    }
    try {
      await navigator.clipboard.writeText(reference);
      setCanvasStatus(document?.path ? "Canvas path copied." : "Canvas title copied.", "success");
    } catch (_) {
      setCanvasStatus("Reference copy failed.", "danger");
    }
  });
}
if (canvasResetFiltersBtn) {
  canvasResetFiltersBtn.addEventListener("click", () => resetCanvasFilters());
}
if (canvasDeleteBtn) {
  canvasDeleteBtn.addEventListener("click", () => {
    void deleteCanvasDocuments();
  });
}
if (canvasClearBtn) {
  canvasClearBtn.addEventListener("click", () => {
    void deleteCanvasDocuments({ clearAll: true });
  });
}
if (canvasDownloadHtmlBtn) {
  canvasDownloadHtmlBtn.addEventListener("click", () => downloadCanvasDocument("html"));
}
if (canvasDownloadMdBtn) {
  canvasDownloadMdBtn.addEventListener("click", () => downloadCanvasDocument("md"));
}
if (canvasDownloadPdfBtn) {
  canvasDownloadPdfBtn.addEventListener("click", () => downloadCanvasDocument("pdf"));
}
if (canvasSearchInput) {
  canvasSearchInput.addEventListener("input", () => renderCanvasPanel());
}
if (canvasRoleFilter) {
  canvasRoleFilter.addEventListener("change", () => renderCanvasPanel());
}
if (canvasPathFilter) {
  canvasPathFilter.addEventListener("change", () => renderCanvasPanel());
}
if (canvasEditorEl) {
  canvasEditorEl.addEventListener("keydown", (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "s") {
      event.preventDefault();
      void saveCanvasEdits();
    }
  });
}
if (sidebarToggleBtn) {
  sidebarToggleBtn.addEventListener("click", toggleSidebar);
}
if (sidebarOverlay) {
  sidebarOverlay.addEventListener("click", () => setSidebarOpen(false));
}
if (mobileToolsBtn) {
  mobileToolsBtn.addEventListener("click", () => {
    if (mobileToolsPanel?.classList.contains("open")) {
      closeMobileTools();
    } else {
      openMobileTools();
    }
  });
}
if (mobileToolsClose) {
  mobileToolsClose.addEventListener("click", closeMobileTools);
}
if (mobileToolsOverlay) {
  mobileToolsOverlay.addEventListener("click", closeMobileTools);
}
if (canvasConfirmOverlay) {
  canvasConfirmOverlay.addEventListener("click", () => closeCanvasConfirmModal("cancel"));
}
if (canvasConfirmCloseBtn) {
  canvasConfirmCloseBtn.addEventListener("click", () => closeCanvasConfirmModal("cancel"));
}
if (canvasConfirmLaterBtn) {
  canvasConfirmLaterBtn.addEventListener("click", () => closeCanvasConfirmModal("cancel"));
}
if (canvasConfirmOpenBtn) {
  canvasConfirmOpenBtn.addEventListener("click", () => closeCanvasConfirmModal("confirm"));
}
if (mobileTokensBtn) {
  mobileTokensBtn.addEventListener("click", () => {
    openStats();
    closeMobileTools();
  });
}
if (mobileSettingsBtn) {
  mobileSettingsBtn.addEventListener("click", closeMobileTools);
}
if (mobileLogoutBtn) {
  mobileLogoutBtn.addEventListener("click", closeMobileTools);
}
if (modelSel) {
  modelSel.addEventListener("change", () => syncModelSelectors(modelSel.value));
}
if (mobileModelSel) {
  mobileModelSel.addEventListener("change", () => syncModelSelectors(mobileModelSel.value));
}

window.addEventListener("resize", () => {
  updateHeaderOffset();
  if (!isMobileViewport()) {
    closeMobileTools();
  }
  applyCanvasPanelWidth(readCanvasWidthPreference(), false);
  syncCanvasToggleButton();
}, { passive: true });

if (canvasResizeHandle) {
  canvasResizeHandle.addEventListener("mousedown", (event) => {
    if (isMobileViewport()) {
      return;
    }
    event.preventDefault();
    const onMouseMove = (moveEvent) => {
      const nextWidth = window.innerWidth - moveEvent.clientX;
      applyCanvasPanelWidth(nextWidth);
    };
    const onMouseUp = () => {
      window.removeEventListener("mousemove", onMouseMove);
      window.removeEventListener("mouseup", onMouseUp);
    };
    window.addEventListener("mousemove", onMouseMove);
    window.addEventListener("mouseup", onMouseUp);
  });
}

window.addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.shiftKey && event.key.toLowerCase() === "c") {
    event.preventDefault();
    if (isCanvasOpen()) {
      closeCanvas();
    } else {
      openCanvas();
    }
    return;
  }
  if (event.key === "Escape") {
    if (isCanvasConfirmOpen()) {
      closeCanvasConfirmModal("cancel");
      return;
    }
    if (isCanvasOpen()) {
      if (isCanvasEditing) {
        isCanvasEditing = false;
        editingCanvasDocumentId = null;
        renderCanvasPanel();
        setCanvasStatus("Canvas edit cancelled.", "muted");
      } else if (canvasSearchInput?.value) {
        canvasSearchInput.value = "";
        renderCanvasPanel();
        setCanvasSearchStatus("Canvas search cleared.", "muted");
      } else {
        closeCanvas();
      }
      return;
    }
    if (mobileToolsPanel?.classList.contains("open")) {
      closeMobileTools();
      return;
    }
  }
  if (event.key === "Tab" && isCanvasOpen()) {
    const focusable = getCanvasFocusableElements();
    if (!focusable.length) {
      return;
    }
    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    const activeElement = document.activeElement;
    if (event.shiftKey && activeElement === first) {
      event.preventDefault();
      last.focus();
      return;
    }
    if (!event.shiftKey && activeElement === last) {
      event.preventDefault();
      first.focus();
    }
  }
  if (event.key === "Escape" && !document.body.classList.contains("sidebar-collapsed") && isMobileViewport()) {
    setSidebarOpen(false);
  }
});

async function loadSidebar() {
  cancelSidebarRename();
  const response = await fetch("/api/conversations");
  const list = await response.json();
  sidebarList.innerHTML = "";
  if (list.length === 0) {
    sidebarList.innerHTML = '<p class="sidebar-empty">No conversations yet.</p>';
    return;
  }
  list.forEach((conversation) => {
    if (conversation.id === currentConvId) {
      currentConvTitle = String(conversation.title || "New Chat").trim() || "New Chat";
    }
    const item = document.createElement("div");
    item.className = "sidebar-item" + (conversation.id === currentConvId ? " active" : "");
    item.dataset.id = conversation.id;
    item.innerHTML =
      `<span class="sidebar-title">${escHtml(conversation.title)}</span>` +
      `<button class="sidebar-edit" title="Rename" aria-label="Rename" data-id="${conversation.id}">` +
      `  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.3" stroke-linecap="round" stroke-linejoin="round">` +
      `    <path d="M12 20h9"/><path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4Z"/>` +
      `  </svg>` +
      `</button>` +
      `<button class="sidebar-del" title="Delete" data-id="${conversation.id}">` +
      `  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round">` +
      `    <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>` +
      `  </svg>` +
      `</button>`;
    item.addEventListener("click", (event) => {
      if (event.target.closest(".sidebar-del") || event.target.closest(".sidebar-edit")) {
        return;
      }
      if (conversation.id !== currentConvId) {
        openConversation(conversation.id);
        closeSidebarOnMobile();
      }
    });
    item.querySelector(".sidebar-edit").addEventListener("click", (event) => {
      event.stopPropagation();
      startSidebarRename(conversation, item);
    });
    item.querySelector(".sidebar-del").addEventListener("click", (event) => {
      event.stopPropagation();
      if (!window.confirm("Are you sure you want to delete this conversation?")) {
        return;
      }
      deleteConversation(conversation.id);
    });
    sidebarList.appendChild(item);
  });
}

function updateConversationTitleInState(conversationId, title) {
  const normalizedTitle = String(title || "New Chat").trim() || "New Chat";
  if (Number(conversationId) === Number(currentConvId)) {
    currentConvTitle = normalizedTitle;
    updateExportPanel();
  }
}

function cancelSidebarRename() {
  if (!activeSidebarRename) {
    return;
  }

  const { item, originalTitle } = activeSidebarRename;
  const titleInput = item.querySelector(".sidebar-title-input");
  if (titleInput) {
    titleInput.replaceWith(createSidebarTitleSpan(originalTitle));
  }
  item.classList.remove("editing");
  activeSidebarRename = null;
}

function createSidebarTitleSpan(title) {
  const span = document.createElement("span");
  span.className = "sidebar-title";
  span.textContent = String(title || "New Chat").trim() || "New Chat";
  return span;
}

function startSidebarRename(conversation, item) {
  if (!conversation || !item) {
    return;
  }

  if (activeSidebarRename && activeSidebarRename.item !== item) {
    cancelSidebarRename();
  }

  const titleNode = item.querySelector(".sidebar-title");
  if (!titleNode || item.querySelector(".sidebar-title-input")) {
    return;
  }

  const originalTitle = String(conversation.title || "New Chat").trim() || "New Chat";
  const titleInput = document.createElement("input");
  titleInput.type = "text";
  titleInput.className = "sidebar-title-input";
  titleInput.value = originalTitle;
  titleInput.spellcheck = false;
  titleInput.autocomplete = "off";
  titleInput.setAttribute("aria-label", "Rename conversation");

  item.classList.add("editing");
  titleNode.replaceWith(titleInput);
  activeSidebarRename = {
    item,
    conversationId: conversation.id,
    originalTitle,
    committing: false,
  };

  const submitRename = async () => {
    if (!activeSidebarRename || activeSidebarRename.item !== item || activeSidebarRename.committing) {
      return;
    }

    const nextTitle = titleInput.value.trim();
    if (!nextTitle) {
      cancelSidebarRename();
      return;
    }

    activeSidebarRename.committing = true;
    try {
      const response = await fetch(`/api/conversations/${conversation.id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title: nextTitle }),
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(data.error || "Unable to rename conversation.");
      }

      updateConversationTitleInState(conversation.id, data.title || nextTitle);
      activeSidebarRename = null;
      await loadSidebar();
    } catch (error) {
      activeSidebarRename = null;
      showError(error.message || "Unable to rename conversation.");
      await loadSidebar();
    }
  };

  const cancelRename = () => {
    if (!activeSidebarRename || activeSidebarRename.item !== item || activeSidebarRename.committing) {
      return;
    }
    cancelSidebarRename();
  };

  titleInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      submitRename();
    } else if (event.key === "Escape") {
      event.preventDefault();
      cancelRename();
    }
  });
  titleInput.addEventListener("blur", () => {
    window.setTimeout(cancelRename, 0);
  });

  titleInput.focus();
  titleInput.select();
}

async function openConversation(id) {
  const response = await fetch(`/api/conversations/${id}`);
  const data = await response.json();
  if (!response.ok) {
    return;
  }

  resetTokenStats();
  history = [];
  latestSummaryStatus = null;
  userScrolledUp = false;
  currentConvId = id;
  currentConvTitle = String(data.conversation?.title || "New Chat").trim() || "New Chat";
  syncModelSelectors(data.conversation.model);
  clearEditTarget();
  resetCanvasWorkspaceState();

  history = Array.isArray(data.messages) ? data.messages.map(normalizeHistoryEntry) : [];
  streamingCanvasDocuments = [];
  resetStreamingCanvasPreview();
  activeCanvasDocumentId = getActiveCanvasDocument(history)?.id || null;
  lastConversationSignature = getConversationSignature(history);
  renderConversationHistory();
  renderCanvasPanel();
  updateExportPanel();
  rebuildTokenStatsFromHistory();

  loadSidebar();
  scrollToBottom();
  inputEl.focus();
}

async function deleteConversation(id) {
  await fetch(`/api/conversations/${id}`, { method: "DELETE" });
  if (id === currentConvId) {
    startNewChat();
  } else {
    loadSidebar();
  }
}

function startNewChat() {
  conversationRefreshGeneration += 1;
  pendingConversationRefreshTimers.forEach((timerId) => window.clearTimeout(timerId));
  pendingConversationRefreshTimers.clear();
  userScrolledUp = false;
  currentConvId = null;
  currentConvTitle = "New Chat";
  history = [];
  latestSummaryStatus = null;
  streamingCanvasDocuments = [];
  resetStreamingCanvasPreview();
  activeCanvasDocumentId = null;
  lastConversationSignature = "";
  clearEditTarget();
  resetCanvasWorkspaceState();
  clearSelectedImage();
  resetTokenStats();
  renderConversationHistory();
  renderCanvasPanel();
  updateExportPanel();
  errorArea.innerHTML = "";
  loadSidebar();
  inputEl.focus();
  closeSidebarOnMobile();
}

function escHtml(str) {
  return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

const TOOL_UI_CONFIG = {
  search_knowledge_base: {
    icon: "🧠",
    label: "Knowledge Base",
    runningTitle: "Searching knowledge base",
    doneTitle: "Knowledge results ready",
    errorTitle: "Knowledge search failed",
    fallbackDetail: "Looking through indexed context and synced chat memory.",
  },
  search_web: {
    icon: "🔎",
    label: "Web Search",
    runningTitle: "Searching the web",
    doneTitle: "Web results ready",
    errorTitle: "Web search failed",
    fallbackDetail: "Collecting live sources from the open web.",
  },
  fetch_url: {
    icon: "🌐",
    label: "Web Fetch",
    runningTitle: "Reading page",
    doneTitle: "Page content extracted",
    errorTitle: "Page read failed",
    fallbackDetail: "Opening the source and extracting readable content.",
  },
  search_news_ddgs: {
    icon: "📰",
    label: "News Search",
    runningTitle: "Scanning news sources",
    doneTitle: "News results ready",
    errorTitle: "News search failed",
    fallbackDetail: "Checking recent headlines and source coverage.",
  },
  search_news_google: {
    icon: "🗞️",
    label: "Google News",
    runningTitle: "Scanning Google News",
    doneTitle: "Google News results ready",
    errorTitle: "Google News search failed",
    fallbackDetail: "Checking recent headlines and publisher coverage.",
  },
};

function getToolUiConfig(toolName) {
  return TOOL_UI_CONFIG[toolName] || {
    icon: "⚙️",
    label: "Tool",
    runningTitle: "Running tool",
    doneTitle: "Tool completed",
    errorTitle: "Tool failed",
    fallbackDetail: "Processing tool call.",
  };
}

function formatToolDuration(durationMs) {
  if (!Number.isFinite(durationMs) || durationMs < 0) {
    return "";
  }
  if (durationMs < 1000) {
    return `${Math.max(1, Math.round(durationMs))} ms`;
  }
  if (durationMs < 10_000) {
    return `${(durationMs / 1000).toFixed(1)} s`;
  }
  return `${Math.round(durationMs / 1000)} s`;
}

function extractHost(value) {
  const text = String(value || "").trim();
  if (!text) {
    return "";
  }
  try {
    return new URL(text).hostname.replace(/^www\./i, "");
  } catch (_) {
    return "";
  }
}

function normalizeToolSummary(summary) {
  const raw = String(summary || "").trim();
  if (!raw) {
    return { text: "", cached: false, isError: false };
  }

  const cached = /\(cached\)$/i.test(raw);
  const withoutCached = raw.replace(/\s*\(cached\)$/i, "").trim();
  const isError = /^error:/i.test(withoutCached) || /^failed:/i.test(withoutCached) || /^[^:]{0,120}\bfailed:\s*/i.test(withoutCached);
  const text = isError ? withoutCached.replace(/^error:\s*/i, "").trim() : withoutCached;
  return { text, cached, isError };
}

function buildToolMeta(toolName, preview, options = {}) {
  const meta = [];
  const detail = String(preview || "").trim();
  const { cached = false, durationMs = null } = options;

  if (toolName === "fetch_url") {
    const host = extractHost(detail);
    if (host) {
      meta.push(host);
    }
    if (detail) {
      meta.push("URL");
    }
  } else if (["search_web", "search_news_ddgs", "search_news_google"].includes(toolName) && detail) {
    const queryCount = detail
      .split(",")
      .map((value) => value.trim())
      .filter(Boolean).length;
    if (queryCount > 0) {
      meta.push(`${queryCount} quer${queryCount === 1 ? "y" : "ies"}`);
    }
  } else if (toolName === "search_knowledge_base") {
    meta.push("semantic retrieval");
  }

  if (cached) {
    meta.push("cached");
  }

  const durationText = formatToolDuration(durationMs);
  if (durationText) {
    meta.push(durationText);
  }

  return meta;
}

function ensureToolStepSection(stepLog, stepSections, step, maxSteps) {
  const stepKey = String(step || 1);
  if (stepSections[stepKey]) {
    return stepSections[stepKey];
  }

  const section = document.createElement("section");
  section.className = "step-section";

  const header = document.createElement("div");
  header.className = "step-section-header";

  const title = document.createElement("div");
  title.className = "step-section-title";
  title.textContent = `Step ${stepKey}`;

  const caption = document.createElement("div");
  caption.className = "step-section-caption";
  caption.textContent = maxSteps ? `Tool round ${stepKey}/${maxSteps}` : "Tool round";

  header.appendChild(title);
  header.appendChild(caption);

  const items = document.createElement("div");
  items.className = "step-section-items";

  section.appendChild(header);
  section.appendChild(items);
  stepLog.appendChild(section);

  stepSections[stepKey] = items;
  return items;
}

function createToolStepItem(toolName) {
  const config = getToolUiConfig(toolName);
  const item = document.createElement("details");
  item.className = "step-item step-running";
  item.open = true;
  item.innerHTML = [
    '<summary class="step-item-summary">',
    '  <div class="step-item-icon"></div>',
    '  <div class="step-item-body">',
    '    <div class="step-item-top">',
    '      <span class="step-status-badge"></span>',
    '      <span class="step-item-label"></span>',
    '      <span class="step-time"></span>',
    "    </div>",
    '    <div class="step-title"></div>',
    "  </div>",
    "</summary>",
    '<div class="step-item-content">',
    '  <div class="step-detail"></div>',
    '  <div class="step-meta"></div>',
    '  <div class="step-summary"></div>',
    "</div>",
  ].join("");
  item.querySelector(".step-item-icon").textContent = config.icon;
  return item;
}

function setToolStepState(item, payload) {
  const config = getToolUiConfig(payload.toolName);
  const state = payload.state || "running";
  const preview = String(payload.preview || "").trim();
  const durationMs = Number.isFinite(payload.durationMs) ? payload.durationMs : null;
  const metaItems = buildToolMeta(payload.toolName, preview, {
    cached: Boolean(payload.cached),
    durationMs,
  });

  item.classList.remove("step-running", "step-done", "step-error");
  item.classList.add(`step-${state}`);
  item.open = state !== "done";

  const badge = item.querySelector(".step-status-badge");
  const label = item.querySelector(".step-item-label");
  const time = item.querySelector(".step-time");
  const title = item.querySelector(".step-title");
  const detail = item.querySelector(".step-detail");
  const meta = item.querySelector(".step-meta");
  const summary = item.querySelector(".step-summary");
  const icon = item.querySelector(".step-item-icon");

  badge.textContent = state === "running" ? "Running" : state === "error" ? "Failed" : payload.cached ? "Cached" : "Done";
  label.textContent = config.label;
  time.textContent = state === "running" ? "" : formatToolDuration(durationMs);
  title.textContent = state === "running" ? config.runningTitle : state === "error" ? config.errorTitle : config.doneTitle;

  const detailText = preview || config.fallbackDetail;
  detail.textContent = detailText;
  detail.style.display = detailText ? "" : "none";

  meta.innerHTML = metaItems.map((value) => `<span class="step-chip">${escHtml(value)}</span>`).join("");
  meta.style.display = metaItems.length ? "" : "none";

  summary.textContent = String(payload.summary || "").trim();
  summary.style.display = summary.textContent ? "" : "none";

  icon.textContent = config.icon;
  item.dataset.state = state;
}

newChatBtn.addEventListener("click", startNewChat);

function autoResize(element) {
  element.style.height = "auto";
  element.style.height = element.scrollHeight + "px";
}
inputEl.addEventListener("input", () => {
  autoResize(inputEl);
});

messagesEl.addEventListener("scroll", () => {
  if (!isStreaming) {
    return;
  }
  const distanceFromBottom = messagesEl.scrollHeight - messagesEl.scrollTop - messagesEl.clientHeight;
  userScrolledUp = distanceFromBottom > 100;
});

if (summaryNowBtn) {
  summaryNowBtn.addEventListener("click", async () => {
    if (!currentConvId) {
      showToast("No active conversation to summarize.", "warning");
      return;
    }
    summaryNowBtn.disabled = true;
    summaryNowBtn.textContent = "Summarizing…";
    try {
      const response = await fetch(`/api/conversations/${currentConvId}/summarize`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ force: true }),
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || "Failed to summarize.");
      }
      if (data.applied) {
        if (Array.isArray(data.messages)) {
          history = data.messages.map(normalizeHistoryEntry);
          rebuildTokenStatsFromHistory();
          renderConversationHistory();
        }
        const coveredCount = Number(data.covered_message_count || 0);
        showToast(
          coveredCount > 0
            ? `${coveredCount} message${coveredCount === 1 ? " was" : "s were"} summarized.`
            : "Summary completed.",
          "success"
        );
        latestSummaryStatus = { applied: true, reason: "applied", failure_stage: null, failure_detail: "Manual summary completed." };
      } else {
        showToast(data.failure_detail || data.reason || "Summary was not applied.", "warning");
        latestSummaryStatus = { applied: false, reason: data.reason, failure_detail: data.failure_detail };
      }
      renderSummaryInspector();
    } catch (error) {
      showToast(error.message, "error");
    } finally {
      summaryNowBtn.disabled = false;
      summaryNowBtn.textContent = "Summarize now";
    }
  });
}

if (summaryUndoBtn) {
  summaryUndoBtn.addEventListener("click", async () => {
    const latestSummary = findLatestSummaryEntry(history);
    const summaryId = Number(latestSummary?.id || 0);
    if (!currentConvId || !summaryId) {
      showToast("No summary is available to undo.", "warning");
      return;
    }

    summaryUndoBtn.disabled = true;
    summaryUndoBtn.textContent = "Restoring…";
    try {
      const response = await fetch(`/api/conversations/${currentConvId}/summaries/${summaryId}/undo`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || "Failed to undo summary.");
      }

      if (Array.isArray(data.messages)) {
        history = data.messages.map(normalizeHistoryEntry);
        rebuildTokenStatsFromHistory();
        renderConversationHistory();
      }

      latestSummaryStatus = {
        applied: false,
        reason: "summary_undone",
        failure_stage: null,
        failure_detail: "The latest summary was reverted and the covered messages were restored.",
      };
      const restoredCount = Number(data.restored_message_count || 0);
      showToast(
        restoredCount > 0
          ? `${restoredCount} message${restoredCount === 1 ? " was" : "s were"} restored.`
          : "Summary was undone.",
        "success"
      );
      renderSummaryInspector();
    } catch (error) {
      showToast(error.message || "Failed to undo summary.", "error");
      renderSummaryInspector();
    } finally {
      summaryUndoBtn.textContent = "Undo last summary";
      renderSummaryInspector();
    }
  });
}

inputEl.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    if ("ontouchstart" in window || navigator.maxTouchPoints > 0) return;
    event.preventDefault();
    if (!isStreaming && !isFixing) {
      sendMessage();
    }
  }
});

cancelBtn.addEventListener("click", () => {
  if (activeAbortController) {
    activeAbortController.abort();
  }
});

editBannerCancelBtn.addEventListener("click", () => {
  clearEditTarget();
  inputEl.focus();
});

sendBtn.addEventListener("click", () => {
  if (!isStreaming && !isFixing) {
    sendMessage();
  }
});
fixBtn.addEventListener("click", () => {
  if (!isStreaming && !isFixing) {
    fixMessage();
  }
});
attachBtn.addEventListener("click", () => {
  if (isStreaming || isFixing) return;
  imageInputEl.click();
});

attachBtn.addEventListener("contextmenu", (e) => {
  if (isStreaming || isFixing) return;
  e.preventDefault();
  docInputEl.click();
});
kbSyncBtn?.addEventListener("click", syncKnowledgeBaseConversations);

imageInputEl.addEventListener("change", () => {
  const files = Array.from(imageInputEl.files || []);
  imageInputEl.value = "";
  if (!files.length) {
    return;
  }
  handleSelectedFiles(files);
});

docInputEl.addEventListener("change", () => {
  const files = Array.from(docInputEl.files || []);
  docInputEl.value = "";
  if (!files.length) return;
  handleSelectedFiles(files, { documentsOnly: true });
});

function handleSelectedFiles(files, options = {}) {
  const documentsOnly = options.documentsOnly === true;
  const nextImages = [...selectedImageFiles];
  const nextDocuments = [...selectedDocumentFiles];

  for (const file of files || []) {
    if (!file) {
      continue;
    }
    if (isDocumentFile(file)) {
      if (file.size > MAX_DOCUMENT_BYTES) {
        showError(`Document ${file.name} is too large. Upload a maximum of 20 MB.`);
        continue;
      }
      nextDocuments.push(file);
      continue;
    }
    if (documentsOnly) {
      showError(`Unsupported document type: ${file.name}`);
      continue;
    }
    if (!featureFlags.image_uploads_enabled) {
      showError("Image uploads are disabled. Only documents can be attached.");
      continue;
    }
    if (!ALLOWED_IMAGE_TYPES.has(file.type)) {
      showError(`Unsupported file type: ${file.name}`);
      continue;
    }
    if (file.size > MAX_IMAGE_BYTES) {
      showError(`Image ${file.name} is too large. Upload a maximum of 10 MB.`);
      continue;
    }
    nextImages.push(file);
  }

  selectedImageFiles = dedupeFiles(nextImages);
  selectedDocumentFiles = dedupeFiles(nextDocuments);
  renderAttachmentPreview();
}

function resetTokenStats() {
  tokenTurns.length = 0;
  renderTokenStats();
}

function setStreaming(active) {
  isStreaming = active;
  if (!active) {
    userScrolledUp = false;
  }
  sendBtn.style.display = active ? "none" : "";
  cancelBtn.style.display = active ? "" : "none";
  fixBtn.disabled = active;
  inputEl.disabled = active;
  attachBtn.disabled = active;
  if (mobilePruneBtn) {
    mobilePruneBtn.disabled = active;
  }
}

function setFixing(active) {
  isFixing = active;
  sendBtn.disabled = active;
  fixBtn.disabled = active;
  inputEl.disabled = active;
  attachBtn.disabled = active;
}

function showToast(message, tone = "error") {
  errorArea.innerHTML = `<div class="error-toast" data-tone="${escHtml(String(tone || "error"))}">${escHtml(String(message || "An unexpected event occurred."))}</div>`;
  setTimeout(() => {
    errorArea.innerHTML = "";
  }, 5000);
}

function showError(message) {
  showToast(message, "error");
}

function setKbStatus(message, tone = "muted") {
  if (!kbStatusEl) {
    return;
  }
  kbStatusEl.textContent = message;
  kbStatusEl.dataset.tone = tone;
}

async function loadKnowledgeBaseDocuments() {
  if (!kbDocumentsListEl || !kbStatusEl) {
    return;
  }
  if (!Boolean(featureFlags.rag_enabled)) {
    renderKnowledgeBaseDocuments([]);
    setKbStatus("RAG disabled in .env", "warning");
    return;
  }
  try {
    const response = await fetch("/api/rag/documents");
    if (response.status === 410) {
      renderKnowledgeBaseDocuments([]);
      setKbStatus("RAG disabled in .env", "warning");
      return;
    }
    const docs = await response.json();
    renderKnowledgeBaseDocuments(Array.isArray(docs) ? docs : []);
  } catch (_) {
    renderKnowledgeBaseDocuments([]);
  }
}

function renderKnowledgeBaseDocuments(docs) {
  if (!kbDocumentsListEl) {
    return;
  }
  kbDocumentsListEl.innerHTML = "";

  if (!docs.length) {
    kbDocumentsListEl.innerHTML = '<p class="kb-empty">No indexed sources yet.</p>';
    return;
  }

  docs.forEach((doc) => {
    const item = document.createElement("div");
    item.className = "kb-doc-item";

    const meta = document.createElement("div");
    meta.className = "kb-doc-meta";

    const title = document.createElement("div");
    title.className = "kb-doc-title";
    title.textContent = doc.source_name || "Untitled source";

    const sub = document.createElement("div");
    sub.className = "kb-doc-subtitle";
    sub.textContent = `${doc.source_type || "document"} · ${doc.category || "general"} · ${doc.chunk_count || 0} chunks`;

    meta.appendChild(title);
    meta.appendChild(sub);

    const del = document.createElement("button");
    del.type = "button";
    del.className = "kb-doc-delete";
    del.textContent = "Delete";
    del.addEventListener("click", () => deleteKnowledgeBaseDocument(doc.source_key));

    item.appendChild(meta);
    item.appendChild(del);
    kbDocumentsListEl.appendChild(item);
  });
}

async function deleteKnowledgeBaseDocument(sourceKey) {
  if (!sourceKey) {
    return;
  }
  setKbStatus("Deleting source…");
  try {
    const response = await fetch(`/api/rag/documents/${encodeURIComponent(sourceKey)}`, { method: "DELETE" });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(data.error || "Delete failed.");
    }
    setKbStatus("Source deleted", "success");
    loadKnowledgeBaseDocuments();
  } catch (error) {
    setKbStatus(error.message, "error");
  }
}

async function syncKnowledgeBaseConversations() {
  if (!kbSyncBtn) {
    return;
  }
  if (!Boolean(featureFlags.rag_enabled)) {
    setKbStatus("RAG disabled in .env", "warning");
    return;
  }
  setKbStatus("Syncing conversations into RAG…");
  if (kbSyncBtn) {
    kbSyncBtn.disabled = true;
  }
  try {
    const response = await fetch("/api/rag/sync-conversations", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(data.error || "Conversation sync failed.");
    }
    setKbStatus(`${data.count || 0} RAG sources synced`, "success");
    loadKnowledgeBaseDocuments();
  } catch (error) {
    setKbStatus(error.message, "error");
  } finally {
    if (kbSyncBtn) {
      kbSyncBtn.disabled = false;
    }
  }
}

function formatFileSize(size) {
  if (!Number.isFinite(size) || size <= 0) {
    return "0 KB";
  }
  if (size < 1024 * 1024) {
    return `${Math.max(1, Math.round(size / 1024))} KB`;
  }
  return `${(size / (1024 * 1024)).toFixed(1)} MB`;
}

function clearSelectedImage() {
  selectedImageFiles = [];
  imageInputEl.value = "";
  renderAttachmentPreview();
}

function clearSelectedDocument() {
  selectedDocumentFiles = [];
  docInputEl.value = "";
  renderAttachmentPreview();
}

function removeSelectedAttachment(kind, fileKey) {
  if (kind === "image") {
    selectedImageFiles = selectedImageFiles.filter((file) => getAttachmentFileKey(file) !== fileKey);
  } else {
    selectedDocumentFiles = selectedDocumentFiles.filter((file) => getAttachmentFileKey(file) !== fileKey);
  }
  renderAttachmentPreview();
}

function clearAllAttachments() {
  selectedImageFiles = [];
  selectedDocumentFiles = [];
  imageInputEl.value = "";
  docInputEl.value = "";
  renderAttachmentPreview();
}

function renderAttachmentPreview() {
  const attachments = [
    ...selectedImageFiles.map((file) => ({ kind: "image", file })),
    ...selectedDocumentFiles.map((file) => ({ kind: "document", file })),
  ];

  if (!attachments.length) {
    attachmentPreviewEl.hidden = true;
    attachmentPreviewEl.innerHTML = "";
    return;
  }

  attachmentPreviewEl.hidden = false;

  attachmentPreviewEl.innerHTML = attachments.map(({ kind, file }) => {
    const fileKey = getAttachmentFileKey(file);
    const icon = kind === "image" ? "🖼️" : "📄";
    const description = kind === "image"
      ? `Ready for Qwen vision analysis · ${formatFileSize(file.size)}`
      : `${((file.name || "").split(".").pop() || "FILE").toUpperCase()} document · ${formatFileSize(file.size)}`;
    const removeLabel = kind === "image" ? "Remove image" : "Remove document";
    return (
      `<div class="attachment-chip">` +
        `<span class="attachment-chip__icon">${icon}</span>` +
        `<span class="attachment-chip__meta">` +
          `<strong>${escHtml(file.name)}</strong>` +
          `<small>${escHtml(description)}</small>` +
        `</span>` +
        `<button type="button" class="attachment-chip__remove" data-kind="${escHtml(kind)}" data-file-key="${escHtml(fileKey)}" title="${removeLabel}">×</button>` +
      `</div>`
    );
  }).join("");

  attachmentPreviewEl.querySelectorAll(".attachment-chip__remove").forEach((button) => {
    button.addEventListener("click", () => {
      removeSelectedAttachment(button.dataset.kind, button.dataset.fileKey);
    });
  });
}

function appendAttachmentBadge(group, metadata) {
  const attachments = getMessageAttachments(metadata);
  if (!attachments.length) {
    return;
  }

  group.querySelectorAll(".message-attachment").forEach((node) => node.remove());

  attachments.forEach((attachment) => {
    const badge = document.createElement("div");
    badge.className = "message-attachment";
    if (attachment.kind === "document") {
      const fileId = attachment.file_id ? String(attachment.file_id).trim() : "";
      const fileName = String(attachment.file_name || "Document").trim() || "Document";
      const label = fileId ? `${fileName} · ${fileId}` : fileName;
      badge.innerHTML =
        `<span class="message-attachment__icon">📄</span>` +
        `<span class="message-attachment__name">${escHtml(label)}</span>` +
        `<span class="message-attachment__state">Document uploaded · Canvas</span>`;
      group.appendChild(badge);
      return;
    }

    const imageName = String(attachment.image_name || "Image").trim() || "Image";
    const imageId = attachment.image_id ? String(attachment.image_id).trim() : "";
    const hasVisionContext = attachment.ocr_text || attachment.vision_summary || attachment.assistant_guidance;
    const stateLabel = hasVisionContext ? "Qwen vision context added" : "Image to be processed";
    const label = imageId ? `${imageName} · ${imageId}` : imageName;
    badge.innerHTML =
      `<span class="message-attachment__icon">🖼️</span>` +
      `<span class="message-attachment__name">${escHtml(label)}</span>` +
      `<span class="message-attachment__state">${stateLabel}</span>`;
    group.appendChild(badge);
  });
}

function updateAttachmentBadge(group, metadata) {
  appendAttachmentBadge(group, metadata);
}

function buildVisionNoteHtml(metadata) {
  const imageAttachments = getMessageAttachments(metadata).filter((attachment) => attachment.kind === "image");
  if (!imageAttachments.length) {
    return "";
  }

  const hasVisionContent = imageAttachments.some((attachment) => {
    const summary = String(attachment.vision_summary || "").trim();
    const guidance = String(attachment.assistant_guidance || "").trim();
    const ocrText = String(attachment.ocr_text || "").trim();
    const keyPoints = Array.isArray(attachment.key_points) ? attachment.key_points.filter(Boolean) : [];
    return Boolean(summary || guidance || ocrText || keyPoints.length);
  });
  if (!hasVisionContent) {
    return "";
  }

  return imageAttachments.map((attachment, index) => {
    const summary = String(attachment.vision_summary || "").trim();
    const guidance = String(attachment.assistant_guidance || "").trim();
    const keyPoints = Array.isArray(attachment.key_points) ? attachment.key_points.filter(Boolean) : [];
    const ocrText = String(attachment.ocr_text || "").trim();
    const imageId = String(attachment.image_id || "").trim();
    const imageName = String(attachment.image_name || "Image").trim() || "Image";
    const parts = [];

    if (imageAttachments.length > 1 || imageName) {
      parts.push(`<div><strong>Image:</strong> ${escHtml(imageName)}</div>`);
    }
    if (imageId) {
      parts.push(`<div><strong>Image ID:</strong> ${escHtml(imageId)}</div>`);
    }
    if (summary) {
      parts.push(`<div><strong>Summary:</strong> ${escHtml(summary)}</div>`);
    }
    if (keyPoints.length) {
      parts.push(
        `<div><strong>Highlights:</strong><ul>` +
          keyPoints.slice(0, 4).map((point) => `<li>${escHtml(String(point))}</li>`).join("") +
        `</ul></div>`
      );
    }
    if (guidance) {
      parts.push(`<div><strong>Guidance note:</strong> ${escHtml(guidance)}</div>`);
    }
    if (ocrText) {
      parts.push(`<div><strong>OCR:</strong> ${escHtml(ocrText.slice(0, 240))}${ocrText.length > 240 ? "…" : ""}</div>`);
    }

    return `<div class="message-vision-note__item" data-index="${index}">${parts.join("")}</div>`;
  }).join("");
}

function appendVisionDetails(group, metadata) {
  const noteHtml = buildVisionNoteHtml(metadata);
  if (!noteHtml) {
    return;
  }

  const note = document.createElement("div");
  note.className = "message-vision-note";
  note.innerHTML = noteHtml;
  group.appendChild(note);
}

function updateVisionDetails(group, metadata) {
  const existing = group.querySelector(".message-vision-note");
  const noteHtml = buildVisionNoteHtml(metadata);

  if (!noteHtml) {
    if (existing) {
      existing.remove();
    }
    return;
  }

  if (existing) {
    existing.innerHTML = noteHtml;
    return;
  }

  appendVisionDetails(group, metadata);
}

function getReasoningText(metadata) {
  if (!metadata || typeof metadata !== "object") {
    return "";
  }
  return String(metadata.reasoning_content || "").trim();
}

function getToolTraceEntries(metadata) {
  if (!metadata || typeof metadata !== "object" || !Array.isArray(metadata.tool_trace)) {
    return [];
  }

  return metadata.tool_trace
    .filter((entry) => entry && typeof entry === "object" && String(entry.tool_name || "").trim())
    .map((entry) => ({
      step: Number.isFinite(Number(entry.step)) ? Math.max(1, Number(entry.step)) : 1,
      tool_name: String(entry.tool_name || "").trim(),
      preview: String(entry.preview || "").trim(),
      summary: String(entry.summary || "").trim(),
      state: ["running", "done", "error"].includes(String(entry.state || "").trim())
        ? String(entry.state || "").trim()
        : "done",
      cached: entry.cached === true,
    }));
}

function getAssistantFetchIndicator(metadata) {
  if (!metadata || typeof metadata !== "object" || !Array.isArray(metadata.tool_results)) {
    return null;
  }

  const fetchResults = metadata.tool_results.filter(
    (entry) => entry && typeof entry === "object" && entry.tool_name === "fetch_url",
  );
  if (!fetchResults.length) {
    return null;
  }

  const clippedEntry = fetchResults.find((entry) => String(entry.content_mode || "").trim() === "clipped_text");
  if (clippedEntry) {
    return {
      label: fetchResults.length > 1 ? `${fetchResults.length} web sources clipped` : "Web source clipped",
      title: String(clippedEntry.summary_notice || "").trim()
        || "Long fetched content was cleaned and clipped before the model used it.",
      tone: "summary",
    };
  }

  const summarizedEntry = fetchResults.find((entry) => String(entry.content_mode || "").trim() === "rag_summary");
  if (summarizedEntry) {
    return {
      label: fetchResults.length > 1 ? `${fetchResults.length} web sources summarized` : "Web source summarized",
      title: String(summarizedEntry.summary_notice || "").trim()
        || "Long fetched content was cleaned and summarized before the model used it.",
      tone: "summary",
    };
  }

  const cleanedEntry = fetchResults.find((entry) => entry.cleanup_applied === true);
  if (cleanedEntry) {
    return {
      label: fetchResults.length > 1 ? `${fetchResults.length} web sources cleaned` : "Web source cleaned",
      title: "Fetched web content was cleaned before the model used it.",
      tone: "clean",
    };
  }

  return null;
}

function updateAssistantFetchBadge(group, metadata) {
  const indicator = getAssistantFetchIndicator(metadata);
  const existing = group.querySelector(".assistant-context-badge");

  if (!indicator) {
    if (existing) {
      existing.remove();
    }
    return;
  }

  const badge = existing || document.createElement("div");
  badge.className = `assistant-context-badge assistant-context-badge--${indicator.tone}`;
  badge.title = indicator.title;
  badge.innerHTML =
    `<span class="assistant-context-badge__icon">${indicator.tone === "summary" ? "✦" : "🧹"}</span>` +
    `<span class="assistant-context-badge__label">${escHtml(indicator.label)}</span>`;

  if (!existing) {
    const anchor = group.querySelector(".tool-trace-panel") || group.querySelector(".reasoning-panel") || group.querySelector(".bubble");
    if (anchor) {
      group.insertBefore(badge, anchor);
    } else {
      group.appendChild(badge);
    }
  }
}

function updateAssistantToolTrace(group, metadata) {
  const entries = getToolTraceEntries(metadata);
  const existing = group.querySelector(".tool-trace-panel");

  if (!entries.length) {
    if (existing) {
      existing.remove();
    }
    return;
  }

  const panel = existing || document.createElement("section");
  panel.className = "tool-trace-panel";

  const title = document.createElement("div");
  title.className = "tool-trace-panel__title";
  title.textContent = entries.length === 1 ? "Tool used" : `Tools used (${entries.length})`;

  const body = document.createElement("div");
  body.className = "tool-trace-panel__body";

  const sections = {};
  entries.forEach((entry) => {
    const sectionItems = ensureToolStepSection(body, sections, entry.step, null);
    const item = createToolStepItem(entry.tool_name);
    const normalizedSummary = normalizeToolSummary(entry.summary);
    setToolStepState(item, {
      toolName: entry.tool_name,
      preview: entry.preview,
      summary: normalizedSummary.text,
      state: normalizedSummary.isError ? "error" : entry.state || "done",
      cached: entry.cached || normalizedSummary.cached,
    });
    sectionItems.appendChild(item);
  });

  panel.innerHTML = "";
  panel.appendChild(title);
  panel.appendChild(body);

  if (!existing) {
    const anchor = group.querySelector(".reasoning-panel") || group.querySelector(".bubble");
    if (anchor) {
      group.insertBefore(panel, anchor);
    } else {
      group.appendChild(panel);
    }
  }
}

function buildReasoningPanel(reasoningText) {
  const text = String(reasoningText || "").trim();
  if (!text) {
    return null;
  }

  const details = document.createElement("details");
  details.className = "reasoning-panel";
  details.open = true;

  const summary = document.createElement("summary");
  summary.textContent = "Reasoning";

  const body = document.createElement("div");
  body.className = "reasoning-body";
  body.innerHTML = renderMarkdown(text);

  details.appendChild(summary);
  details.appendChild(body);
  return details;
}

function updateReasoningPanel(group, reasoningText) {
  const text = String(reasoningText || "").trim();
  const existing = group.querySelector(".reasoning-panel");

  if (!text) {
    if (existing) {
      existing.remove();
    }
    return;
  }

  if (existing) {
    const body = existing.querySelector(".reasoning-body");
    if (body) {
      body.innerHTML = renderMarkdown(text);
    }
    return;
  }

  const panel = buildReasoningPanel(text);
  if (!panel) {
    return;
  }

  const bubble = group.querySelector(".bubble");
  if (bubble) {
    group.insertBefore(panel, bubble);
  } else {
    group.appendChild(panel);
  }
}

function appendClarificationPanel(group, metadata, options = {}) {
  const clarification = getPendingClarification(metadata);
  if (!clarification) {
    return;
  }

  const panel = document.createElement("section");
  panel.className = "clarification-card";

  const title = document.createElement("div");
  title.className = "clarification-card__title";
  title.textContent = clarification.questions.length === 1 ? "Clarification needed" : "Clarifications needed";
  panel.appendChild(title);

  const isInteractive = Boolean(options.isLatestVisible && Number.isInteger(Number(options.messageId)));
  if (!isInteractive) {
    const state = document.createElement("div");
    state.className = "clarification-card__state";
    state.textContent = "Waiting for a reply in this thread.";
    panel.appendChild(state);
    group.appendChild(panel);
    return;
  }

  const form = document.createElement("form");
  form.className = "clarification-form";

  clarification.questions.forEach((question, index) => {
    const field = document.createElement("div");
    field.className = "clarification-field";

    const label = document.createElement("label");
    label.className = "clarification-field__label";
    label.textContent = question.label;
    field.appendChild(label);

    const fieldName = `clarify_${index}`;
    const freeTextName = `${fieldName}_free`;

    if (question.input_type === "text") {
      const input = document.createElement("textarea");
      input.name = fieldName;
      input.rows = 2;
      input.placeholder = question.placeholder || "Type your answer";
      input.className = "clarification-field__textarea";
      input.addEventListener("input", () => autoResize(input));
      field.appendChild(input);
    } else {
      const optionsList = document.createElement("div");
      optionsList.className = "clarification-options";
      question.options.forEach((option) => {
        const optionLabel = document.createElement("label");
        optionLabel.className = "clarification-option";

        const input = document.createElement("input");
        input.type = question.input_type === "single_select" ? "radio" : "checkbox";
        input.name = fieldName;
        input.value = option.value;

        const textBlock = document.createElement("span");
        textBlock.className = "clarification-option__text";
        textBlock.innerHTML = `<strong>${escHtml(option.label)}</strong>${option.description ? `<small>${escHtml(option.description)}</small>` : ""}`;

        optionLabel.appendChild(input);
        optionLabel.appendChild(textBlock);
        optionsList.appendChild(optionLabel);
      });
      field.appendChild(optionsList);

      if (question.allow_free_text) {
        const freeTextInput = document.createElement("input");
        freeTextInput.type = "text";
        freeTextInput.name = freeTextName;
        freeTextInput.className = "clarification-field__input";
        freeTextInput.placeholder = question.placeholder || "Add details if needed";
        field.appendChild(freeTextInput);
      }
    }

    form.appendChild(field);
  });

  const error = document.createElement("div");
  error.className = "clarification-form__error";
  error.hidden = true;
  form.appendChild(error);

  const submitButton = document.createElement("button");
  submitButton.type = "submit";
  submitButton.className = "msg-action-btn clarification-form__submit";
  submitButton.textContent = clarification.submit_label;
  form.appendChild(submitButton);

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (isStreaming || isFixing) {
      return;
    }

    const collected = collectClarificationAnswers(form, clarification);
    if (collected.error) {
      error.hidden = false;
      error.textContent = collected.error;
      return;
    }

    error.hidden = true;
    await sendMessage({
      forcedText: collected.text,
      forcedMetadata: {
        clarification_response: {
          assistant_message_id: Number(options.messageId),
          answers: collected.answers,
        },
      },
    });
  });

  panel.appendChild(form);
  group.appendChild(panel);
}

function createMessageGroup(role, text, metadata = null, options = {}) {
  emptyState.style.display = "none";

  const group = document.createElement("div");
  group.className = `msg-group ${role}`;
  if (Number.isInteger(Number(options.messageId))) {
    group.dataset.messageId = String(options.messageId);
  }
  if (options.isEditingTarget) {
    group.classList.add("editing-target");
  }

  const metaRow = document.createElement("div");
  metaRow.className = "msg-meta-row";

  const normalizedMetadata = metadata && typeof metadata === "object" ? metadata : null;
  const labelGroup = document.createElement("div");
  labelGroup.className = "msg-meta-label-group";

  const label = document.createElement("div");
  label.className = "msg-label";
  label.textContent = role === "user" ? "You" : role === "summary" ? "Summary" : "Assistant";

  labelGroup.appendChild(label);
  if (normalizedMetadata?.is_pruned === true) {
    const prunedBadge = document.createElement("span");
    prunedBadge.className = "pruned-badge";
    prunedBadge.textContent = "Pruned";
    labelGroup.appendChild(prunedBadge);
  }

  metaRow.appendChild(labelGroup);
  const actions = createMessageActions(
    {
      id: options.messageId,
      role,
      content: text,
      metadata: normalizedMetadata,
      tool_calls: Array.isArray(options.toolCalls) ? options.toolCalls : [],
    },
    options,
  );
  if (actions) {
    metaRow.appendChild(actions);
  }

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  const attachments = getMessageAttachments(metadata);
  const hasImage = attachments.some((attachment) => attachment.kind === "image");
  const hasDocument = attachments.some((attachment) => attachment.kind === "document");
  const displayText = text || (attachments.length ? "Attachments uploaded." : hasImage ? "Image uploaded." : hasDocument ? "Document uploaded." : "");

  if ((role === "assistant" || role === "summary") && text !== "Working…") {
    bubble.innerHTML = renderMarkdown(text);
  } else {
    bubble.textContent = displayText;
  }

  group.appendChild(metaRow);
  if (role === "assistant") {
    updateAssistantFetchBadge(group, metadata);
    updateAssistantToolTrace(group, metadata);
    updateReasoningPanel(group, getReasoningText(metadata));
  }
  group.appendChild(bubble);
  if (role === "assistant") {
    appendClarificationPanel(group, metadata, options);
  }
  if (role === "user" && attachments.length) {
    appendAttachmentBadge(group, metadata);
    if (hasImage) {
      appendVisionDetails(group, metadata);
    }
  }
  return group;
}

function appendGroup(role, text, metadata = null, options = {}) {
  const group = createMessageGroup(role, text, metadata, options);
  messagesEl.appendChild(group);
  scrollToBottom();
  return group;
}

let scrollToBottomFrame = null;

function scrollToBottom() {
  if (userScrolledUp) {
    return;
  }
  if (scrollToBottomFrame !== null) {
    return;
  }

  const flushScroll = () => {
    scrollToBottomFrame = null;
    messagesEl.scrollTop = messagesEl.scrollHeight;
  };

  if (typeof window !== "undefined" && typeof window.requestAnimationFrame === "function") {
    scrollToBottomFrame = window.requestAnimationFrame(flushScroll);
    return;
  }

  scrollToBottomFrame = window.setTimeout(flushScroll, 16);
}

async function fixMessage() {
  const text = inputEl.value.trim();
  if (!text) {
    return;
  }

  errorArea.innerHTML = "";
  setFixing(true);

  try {
    const response = await fetch("/api/fix-text", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });

    const data = await response.json().catch(() => ({ error: "An unexpected error occurred." }));
    if (!response.ok) {
      throw new Error(data.error || "An unexpected error occurred.");
    }

    inputEl.value = data.text || text;
    autoResize(inputEl);
    inputEl.focus();
    inputEl.setSelectionRange(inputEl.value.length, inputEl.value.length);
  } catch (error) {
    showError(error.message);
  } finally {
    setFixing(false);
  }
}

async function sendMessage(options = {}) {
  const forcedText = typeof options.forcedText === "string" ? options.forcedText.trim() : "";
  const forcedMetadata = options.forcedMetadata && typeof options.forcedMetadata === "object"
    ? options.forcedMetadata
    : null;
  const text = forcedText || inputEl.value.trim();
  const pendingImages = [...selectedImageFiles];
  const pendingDocuments = [...selectedDocumentFiles];
  if (!text && !pendingImages.length && !pendingDocuments.length) {
    return;
  }

  setPendingDocumentCanvasOpen(pendingDocuments);

  if (pendingImages.length && !Boolean(featureFlags.image_uploads_enabled)) {
    clearSelectedImage();
    showError("Image uploads are disabled in .env.");
    return;
  }

  const editingEntry = getHistoryMessage(editingMessageId);
  const isEditing = Boolean(editingEntry && editingEntry.role === "user");
  const editedMessageId = isEditing ? Number(editingEntry.id) : null;
  if (!isEditing) {
    clearEditTarget();
  }

  errorArea.innerHTML = "";
  inputEl.value = "";
  inputEl.style.height = "auto";
  clearAllAttachments();

  if (!currentConvId) {
    const response = await fetch("/api/conversations", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title: "New Chat", model: modelSel.value }),
    });
    const conversation = await response.json();
    currentConvId = conversation.id;
    currentConvTitle = String(conversation.title || "New Chat").trim() || "New Chat";
    loadSidebar();
    updateExportPanel();
  }

  let userMetadata = buildPendingAttachmentMetadata(pendingImages, pendingDocuments);
  if (forcedMetadata) {
    userMetadata = {
      ...(userMetadata || {}),
      ...forcedMetadata,
    };
  }
  if (userMetadata && !Object.keys(userMetadata).length) {
    userMetadata = null;
  }
  let userGroup;

  if (isEditing) {
    const editIndex = getHistoryMessageIndex(editedMessageId);
    if (editIndex < 0) {
      clearEditTarget();
      showError("The selected message could not be edited.");
      return;
    }

    if (!pendingImages.length && !pendingDocuments.length) {
      userMetadata = editingEntry.metadata || null;
    }

    history = history.slice(0, editIndex + 1).map((item) => ({
      ...normalizeHistoryEntry(item),
      metadata: item.metadata && typeof item.metadata === "object" ? { ...item.metadata } : null,
    }));
    history[editIndex] = {
      ...history[editIndex],
      content: text,
      metadata: userMetadata,
    };
    streamingCanvasDocuments = [];
    resetStreamingCanvasPreview();
    pendingCanvasDiff = null;
    isCanvasEditing = false;
    editingCanvasDocumentId = null;
    activeCanvasDocumentId = getActiveCanvasDocument(history)?.id || null;
    rebuildTokenStatsFromHistory();
    renderConversationHistory();
    renderCanvasPanel();
    userGroup = messagesEl.querySelector(".msg-group.user:last-of-type");
    clearEditTarget();
  } else {
    const userEntry = { id: null, role: "user", content: text, metadata: userMetadata };
    history.push(userEntry);
    userGroup = appendGroup("user", text, userMetadata, { editable: false, messageId: null });
  }

  const { asstGroup, stepLog, asstBubble } = createAssistantStreamingGroup();

  const controller = new AbortController();
  activeAbortController = controller;
  conversationRefreshGeneration += 1;
  pendingConversationRefreshTimers.forEach((timerId) => window.clearTimeout(timerId));
  pendingConversationRefreshTimers.clear();
  setStreaming(true);

  let rawAnswer = "";
  let rawReasoning = "";
  let fullAnswer = "";
  let latestUsage = null;
  let assistantToolResults = [];
  let assistantToolTrace = [];
  let assistantToolHistory = [];
  let pendingClarification = null;
  let persistedMessageIds = null;
  let receivedHistorySync = false;
  const stepItems = {};
  const stepSections = {};
  const assistantTraceByKey = {};
  let latestStepInfo = { step: 1, maxSteps: null };
  let pendingAnswerRenderTimer = null;
  let visibleAnswer = "";

  const getTypingStepSize = () => {
    const remainingLength = Math.max(0, fullAnswer.length - visibleAnswer.length);
    if (!remainingLength) {
      return 0;
    }

    return Math.max(
      STREAM_TYPING_MIN_STEP,
      Math.min(STREAM_TYPING_MAX_STEP, Math.ceil(remainingLength * 0.18)),
    );
  };

  const scheduleAnswerRender = () => {
    if (pendingAnswerRenderTimer !== null) {
      return;
    }

    pendingAnswerRenderTimer = window.setTimeout(() => {
      pendingAnswerRenderTimer = null;
      const stepSize = getTypingStepSize();
      if (!stepSize) {
        return;
      }

      visibleAnswer = fullAnswer.slice(0, visibleAnswer.length + stepSize);
      renderBubbleWithCursor(asstBubble, visibleAnswer);
      scrollToBottom();
      if (visibleAnswer.length < fullAnswer.length) {
        scheduleAnswerRender();
      }
    }, STREAM_TYPING_INTERVAL_MS);
  };

  const flushAnswerRender = () => {
    if (pendingAnswerRenderTimer !== null) {
      window.clearTimeout(pendingAnswerRenderTimer);
      pendingAnswerRenderTimer = null;
    }

    visibleAnswer = fullAnswer;
    renderBubbleWithCursor(asstBubble, visibleAnswer);
  };

  try {
    const requestMessages = buildRequestMessagesFromHistory();

    let response;
    if (pendingImages.length || pendingDocuments.length) {
      const formData = new FormData();
      formData.append("messages", JSON.stringify(requestMessages));
      formData.append("model", modelSel.value);
      formData.append("conversation_id", String(currentConvId));
      formData.append("user_content", text);
      if (editedMessageId !== null) {
        formData.append("edited_message_id", String(editedMessageId));
      }
      pendingImages.forEach((file) => formData.append("image", file));
      pendingDocuments.forEach((file) => formData.append("document", file));

      response = await fetch("/chat", {
        method: "POST",
        signal: controller.signal,
        body: formData,
      });
    } else {
      response = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: controller.signal,
        body: JSON.stringify({
          messages: requestMessages,
          model: modelSel.value,
          conversation_id: currentConvId,
          edited_message_id: editedMessageId,
          user_content: text,
        }),
      });
    }

    if (!response.ok) {
      const error = await response.json().catch(() => ({ error: "An unexpected error occurred." }));
      throw new Error(error.error || "An unexpected error occurred.");
    }

    await streamNdjsonResponse(response, (event) => {
      if (event.type === "step_started") {
        latestStepInfo = {
          step: event.step || latestStepInfo.step,
          maxSteps: event.max_steps || latestStepInfo.maxSteps,
        };
      } else if (event.type === "vision_complete" || event.type === "ocr_complete") {
        const lastMessage = history[history.length - 1];
        if (lastMessage && lastMessage.role === "user") {
          const attachment = event.attachment || {
            kind: "image",
            image_id: event.image_id,
            image_name: event.image_name,
            ocr_text: event.ocr_text,
            vision_summary: event.vision_summary,
            assistant_guidance: event.assistant_guidance,
            key_points: Array.isArray(event.key_points) ? event.key_points : [],
          };
          lastMessage.metadata = mergeAttachmentMetadata(lastMessage.metadata, attachment);
          updateAttachmentBadge(userGroup, lastMessage.metadata);
          updateVisionDetails(userGroup, lastMessage.metadata);
        }
        scrollToBottom();
      } else if (event.type === "document_processed") {
        const lastMessage = history[history.length - 1];
        if (lastMessage && lastMessage.role === "user") {
          const attachment = event.attachment || {
            kind: "document",
            file_id: event.file_id,
            file_name: event.file_name,
            file_mime_type: event.file_mime_type,
          };
          lastMessage.metadata = mergeAttachmentMetadata(lastMessage.metadata, attachment);
          appendAttachmentBadge(userGroup, lastMessage.metadata);
        }

        const pendingCanvasRequest = consumePendingDocumentCanvasOpen();
        if (event.canvas_document && pendingCanvasRequest && !isCanvasOpen()) {
          const documentCount = Number(pendingCanvasRequest.fileCount || 1);
          const requestLabel = Number(pendingCanvasRequest.fileCount || 1) > 1
            ? `${pendingCanvasRequest.fileCount} documents`
            : pendingCanvasRequest.fileName;
          confirmCanvasOpenForDocument(pendingCanvasRequest, documentCount, {
            onConfirm: () => {
              openCanvas();
              setCanvasStatus(`${requestLabel} opened in Canvas.`, "success");
            },
            onCancel: () => {
              setCanvasAttention(true);
              setCanvasStatus(`${requestLabel} ${Number(pendingCanvasRequest.fileCount || 1) > 1 ? "are" : "is"} ready in Canvas. Open the panel when needed.`, "muted");
            },
          });
        }
        scrollToBottom();
      } else if (event.type === "step_update") {
        stepLog.style.display = "";
        const toolKey = event.call_id || event.tool || "__generic__";
        const sectionItems = ensureToolStepSection(
          stepLog,
          stepSections,
          event.step || latestStepInfo.step,
          event.max_steps || latestStepInfo.maxSteps,
        );
        if (!stepItems[toolKey]) {
          const item = createToolStepItem(event.tool);
          sectionItems.appendChild(item);
          stepItems[toolKey] = {
            el: item,
            toolName: event.tool,
            preview: event.preview || "",
            startedAt: performance.now(),
          };
        }
        const itemRef = stepItems[toolKey];
        itemRef.toolName = event.tool || itemRef.toolName;
        itemRef.preview = event.preview || itemRef.preview;
        setToolStepState(itemRef.el, {
          toolName: itemRef.toolName,
          preview: itemRef.preview,
          state: "running",
        });
        if (event.tool) {
          const traceEntry = assistantTraceByKey[toolKey] || {
            tool_name: event.tool,
            step: event.step || latestStepInfo.step || 1,
            preview: event.preview || "",
            summary: "",
            state: "running",
            cached: false,
          };
          traceEntry.tool_name = event.tool || traceEntry.tool_name;
          traceEntry.step = event.step || traceEntry.step || 1;
          traceEntry.preview = event.preview || traceEntry.preview || "";
          traceEntry.state = "running";
          assistantTraceByKey[toolKey] = traceEntry;
          if (!assistantToolTrace.includes(traceEntry)) {
            assistantToolTrace.push(traceEntry);
          }
        }
        scrollToBottom();
      } else if (event.type === "tool_result") {
        const toolKey = event.call_id || event.tool || "__generic__";
        const itemRef = stepItems[toolKey];
        if (itemRef) {
          const normalizedSummary = normalizeToolSummary(event.summary);
          const durationMs = performance.now() - itemRef.startedAt;
          setToolStepState(itemRef.el, {
            toolName: event.tool || itemRef.toolName,
            preview: itemRef.preview,
            summary: normalizedSummary.text,
            state: normalizedSummary.isError ? "error" : "done",
            cached: normalizedSummary.cached,
            durationMs,
          });
          const traceEntry = assistantTraceByKey[toolKey] || {
            tool_name: event.tool || itemRef.toolName,
            step: event.step || latestStepInfo.step || 1,
            preview: itemRef.preview || "",
            summary: "",
            state: "done",
            cached: false,
          };
          traceEntry.tool_name = event.tool || traceEntry.tool_name;
          traceEntry.step = event.step || traceEntry.step || 1;
          traceEntry.preview = itemRef.preview || traceEntry.preview || "";
          traceEntry.summary = normalizedSummary.text;
          traceEntry.state = normalizedSummary.isError ? "error" : "done";
          traceEntry.cached = normalizedSummary.cached;
          assistantTraceByKey[toolKey] = traceEntry;
          if (!assistantToolTrace.includes(traceEntry)) {
            assistantToolTrace.push(traceEntry);
          }
          scrollToBottom();
        }
      } else if (event.type === "tool_error") {
        const toolKey = event.call_id || event.tool || "__generic__";
        let itemRef = stepItems[toolKey];
        if (!itemRef) {
          const sectionItems = ensureToolStepSection(
            stepLog,
            stepSections,
            event.step || latestStepInfo.step,
            latestStepInfo.maxSteps,
          );
          const item = createToolStepItem(event.tool);
          sectionItems.appendChild(item);
          itemRef = {
            el: item,
            toolName: event.tool,
            preview: "",
            startedAt: performance.now(),
          };
          stepItems[toolKey] = itemRef;
        }

        if (itemRef) {
          const durationMs = performance.now() - itemRef.startedAt;
          stepLog.style.display = "";
          setToolStepState(itemRef.el, {
            toolName: event.tool || itemRef.toolName,
            preview: itemRef.preview,
            summary: event.error || "Error",
            state: "error",
            durationMs,
          });
          if (event.tool) {
            const traceEntry = assistantTraceByKey[toolKey] || {
              tool_name: event.tool || itemRef.toolName,
              step: event.step || latestStepInfo.step || 1,
              preview: itemRef.preview || "",
              summary: "",
              state: "error",
              cached: false,
            };
            traceEntry.tool_name = event.tool || traceEntry.tool_name;
            traceEntry.step = event.step || traceEntry.step || 1;
            traceEntry.preview = itemRef.preview || traceEntry.preview || "";
            traceEntry.summary = event.error || "Error";
            traceEntry.state = "error";
            assistantTraceByKey[toolKey] = traceEntry;
            if (!assistantToolTrace.includes(traceEntry)) {
              assistantToolTrace.push(traceEntry);
            }
          }
        } else {
          const errItem = document.createElement("div");
          errItem.className = "step-item step-error";
          errItem.textContent = event.error || "Error";
          stepLog.style.display = "";
          stepLog.appendChild(errItem);
        }
        scrollToBottom();
      } else if (event.type === "answer_start") {
        const wasThinking = asstBubble.classList.contains("thinking");
        asstBubble.classList.remove("thinking");
        if (wasThinking) {
          asstBubble.textContent = "";
        }
      } else if (event.type === "reasoning_start") {
        updateReasoningPanel(asstGroup, rawReasoning);
        scrollToBottom();
      } else if (event.type === "reasoning_delta") {
        rawReasoning += event.text || "";
        updateReasoningPanel(asstGroup, rawReasoning);
        scrollToBottom();
      } else if (event.type === "answer_sync") {
        const syncedAnswer = String(event.text || "").trim();
        if (!syncedAnswer) {
          return;
        }
        rawAnswer = syncedAnswer;
        fullAnswer = rawAnswer;
        flushAnswerRender();
        asstBubble.classList.remove("thinking");
        asstBubble.classList.remove("cursor");
        renderBubbleMarkdown(asstBubble, fullAnswer);
        scrollToBottom();
      } else if (event.type === "answer_delta") {
        rawAnswer += event.text || "";
        fullAnswer = rawAnswer;
        scheduleAnswerRender();
      } else if (event.type === "clarification_request") {
        pendingClarification = event.clarification && typeof event.clarification === "object" ? event.clarification : null;
        rawAnswer = String(event.text || "").trim();
        fullAnswer = rawAnswer;
        flushAnswerRender();
        asstBubble.classList.remove("thinking");
        asstBubble.classList.remove("cursor");
        renderBubbleMarkdown(asstBubble, fullAnswer);
        scrollToBottom();
      } else if (event.type === "usage") {
        latestUsage = normalizeUsagePayload(event);
        updateStats(latestUsage);
      } else if (event.type === "assistant_tool_results") {
        assistantToolResults = Array.isArray(event.tool_results) ? event.tool_results : [];
        updateAssistantFetchBadge(asstGroup, { tool_results: assistantToolResults });
        scrollToBottom();
      } else if (event.type === "assistant_tool_history") {
        const nextToolHistory = Array.isArray(event.messages)
          ? event.messages.map(normalizeHistoryEntry).filter((item) => item.role === "assistant" || item.role === "tool")
          : [];
        assistantToolHistory.push(...nextToolHistory);
      } else if (event.type === "canvas_loading") {
        if (!isCanvasStreamingPreviewTool(event.tool)) {
          return;
        }
        ensureStreamingCanvasPreview(event.tool, event.preview_key, event.snapshot);
        if (!isCanvasOpen()) {
          openCanvas();
        } else {
          renderCanvasPanel();
        }
        setCanvasStatus("Preparing canvas...", "muted");
      } else if (event.type === "canvas_content_delta") {
        if (!isCanvasStreamingPreviewTool(event.tool)) {
          return;
        }
        const previewDocument = ensureStreamingCanvasPreview(event.tool, event.preview_key, event.snapshot);
        if (previewDocument) {
          previewDocument.content += String(event.delta || "");
          previewDocument.line_count = previewDocument.content ? previewDocument.content.split("\n").length : 0;
          if (!isCanvasOpen()) {
            openCanvas();
          }
          setCanvasStatus("Generating live canvas...", "muted");
          scheduleCanvasPreviewRender();
        }
      } else if (event.type === "canvas_sync") {
        const previousDocuments = getCanvasDocumentCollection();
        const nextDocuments = Array.isArray(event.documents)
          ? event.documents.map((document) => normalizeCanvasDocument(document)).filter((document) => document.id)
          : [];
        const previousActiveId = String(activeCanvasDocumentId || "").trim();
        const requestedActiveId = String(event.active_document_id || "").trim();
        const nextActiveCandidate = getCanvasDocumentById(nextDocuments, requestedActiveId)
          || getCanvasDocumentById(nextDocuments, previousActiveId)
          || nextDocuments[nextDocuments.length - 1]
          || null;
        const previousSelectedDocument = getCanvasDocumentById(previousDocuments, previousActiveId);
        const previousVersionOfNextDocument = getCanvasDocumentById(previousDocuments, nextActiveCandidate?.id || previousActiveId);
        if (previousVersionOfNextDocument && nextActiveCandidate && previousVersionOfNextDocument.content !== nextActiveCandidate.content) {
          pendingCanvasDiff = {
            documentId: nextActiveCandidate.id,
            diff: buildCanvasDiff(previousVersionOfNextDocument.content, nextActiveCandidate.content),
          };
          if (!pendingCanvasDiff.diff) {
            pendingCanvasDiff = null;
          }
        }
        resetStreamingCanvasPreview();
        streamingCanvasDocuments = nextDocuments;
        if (streamingCanvasDocuments.length) {
          activeCanvasDocumentId = String(nextActiveCandidate?.id || "").trim() || streamingCanvasDocuments[streamingCanvasDocuments.length - 1].id;
          renderCanvasPanel();
          const pendingCanvasRequest = pendingDocumentCanvasOpen;
          const activeDocumentChangeMessage = describeCanvasActiveDocumentChange(previousSelectedDocument, nextActiveCandidate, requestedActiveId);
          if (pendingCanvasRequest) {
            consumePendingDocumentCanvasOpen();
          }

          if (pendingCanvasRequest && event.auto_open && !isCanvasOpen()) {
            const requestLabel = Number(pendingCanvasRequest.fileCount || 1) > 1
              ? `${pendingCanvasRequest.fileCount} documents`
              : pendingCanvasRequest.fileName;
            confirmCanvasOpenForDocument(pendingCanvasRequest, streamingCanvasDocuments.length, {
              onConfirm: () => {
                openCanvas();
                setCanvasStatus(`${requestLabel} opened in Canvas.`, "success");
              },
              onCancel: () => {
                setCanvasAttention(true);
                setCanvasStatus(`${requestLabel} ${Number(pendingCanvasRequest.fileCount || 1) > 1 ? "are" : "is"} ready in Canvas. Open the panel when needed.`, "muted");
              },
            });
          } else if (event.auto_open && !isCanvasOpen()) {
            openCanvas();
            setCanvasStatus("Document opened in Canvas.", "success");
          } else if (activeDocumentChangeMessage) {
            setCanvasStatus(activeDocumentChangeMessage, "muted");
          } else if (isCanvasOpen()) {
            setCanvasStatus("Canvas updated.", "success");
          } else {
            setCanvasAttention(true);
            setCanvasStatus("Canvas updated. Open the panel to review.", "success");
          }
        } else if (event.cleared) {
          pendingCanvasDiff = null;
          isCanvasEditing = false;
          editingCanvasDocumentId = null;
          activeCanvasDocumentId = null;
          renderCanvasPanel();
          if (isCanvasOpen()) {
            closeCanvas();
          }
          setCanvasAttention(false);
          setCanvasStatus("Canvas cleared.", "success");
        }
      } else if (event.type === "history_sync") {
        receivedHistorySync = true;
        history = Array.isArray(event.messages) ? event.messages.map(normalizeHistoryEntry) : [];
        streamingCanvasDocuments = [];
        resetStreamingCanvasPreview();
        activeCanvasDocumentId = getActiveCanvasDocument(history)?.id || null;
        rebuildTokenStatsFromHistory();
        renderConversationHistory();
        renderCanvasPanel();
      } else if (event.type === "conversation_summary_status") {
        latestSummaryStatus = event && typeof event === "object" ? { ...event } : null;
        renderSummaryInspector();
      } else if (event.type === "conversation_summary_applied") {
        latestSummaryStatus = event && typeof event === "object"
          ? { ...event, applied: true, reason: "applied", failure_stage: null, failure_detail: "Summary completed successfully." }
          : { applied: true, reason: "applied", failure_stage: null, failure_detail: "Summary completed successfully." };
        const coveredCount = Number(event.covered_message_count || 0);
        const mode = String(event.mode || "auto").trim() || "auto";
        const tokenCount = Number(event.visible_token_count || 0);
        const parts = [
          coveredCount > 0
            ? `${coveredCount} older message${coveredCount === 1 ? " was" : "s were"} summarized`
            : "Conversation summary updated",
        ];
        parts.push(`mode: ${mode}`);
        if (tokenCount > 0) {
          parts.push(`visible tokens: ${tokenCount}`);
        }
        showToast(parts.join(" • "), "success");
      } else if (event.type === "message_ids") {
        persistedMessageIds = event;
      } else if (event.type === "done") {
        // no-op
      }
    });

    if (pendingAnswerRenderTimer !== null) {
      flushAnswerRender();
    }
    pendingDocumentCanvasOpen = null;
    asstBubble.classList.remove("cursor");
    renderBubbleMarkdown(asstBubble, fullAnswer);
    const assistantEntry = {
      id: null,
      role: "assistant",
      content: fullAnswer,
      usage: latestUsage,
      metadata: buildAssistantMetadata({
        reasoning: rawReasoning,
        tool_trace: assistantToolTrace,
        tool_results: assistantToolResults,
        canvas_documents: streamingCanvasDocuments,
        usage: latestUsage,
        pending_clarification: pendingClarification,
      }),
    };
    if (!receivedHistorySync) {
      history.push(...assistantToolHistory, assistantEntry);
      applyPersistedMessageIds(persistedMessageIds, assistantEntry);
    }
    finalizeAssistantStreamingGroup(asstGroup, stepLog, assistantEntry.metadata);
    clearEditTarget();
    renderSummaryInspector();

    if (shouldGenerateConversationTitle()) {
      generateTitle(currentConvId);
    } else {
      loadSidebar();
    }
    lastConversationSignature = getConversationSignature(history);
    scheduleConversationRefreshAfterStream();
  } catch (error) {
    if (pendingAnswerRenderTimer !== null) {
      flushAnswerRender();
    }
    pendingDocumentCanvasOpen = null;
    if (fullAnswer.trim()) {
      asstBubble.classList.remove("cursor");
      renderBubbleMarkdown(asstBubble, fullAnswer);

      const assistantEntry = {
        id: null,
        role: "assistant",
        content: fullAnswer,
        usage: latestUsage,
        metadata: buildAssistantMetadata({
          reasoning: rawReasoning,
          tool_trace: assistantToolTrace,
          tool_results: assistantToolResults,
          canvas_documents: streamingCanvasDocuments,
          usage: latestUsage,
          pending_clarification: pendingClarification,
        }),
      };
      if (!receivedHistorySync) {
        history.push(...assistantToolHistory, assistantEntry);
        applyPersistedMessageIds(persistedMessageIds, assistantEntry);
      }
      finalizeAssistantStreamingGroup(asstGroup, stepLog, assistantEntry.metadata);
      clearEditTarget();
      renderSummaryInspector();
      loadSidebar();

      if (error.name !== "AbortError") {
        showError("Connection was interrupted. The partial answer was preserved.");
      }
      lastConversationSignature = getConversationSignature(history);
      scheduleConversationRefreshAfterStream();
    } else {
      if (currentConvId) {
        await openConversation(currentConvId);
      } else {
        startNewChat();
      }
      if (error.name !== "AbortError") {
        showError(error.message);
      }
    }
  } finally {
    activeAbortController = null;
    setStreaming(false);
    refreshEditBanner();
    inputEl.focus();
  }
}

async function generateTitle(convId) {
  try {
    const response = await fetch(`/api/conversations/${convId}/generate-title`, { method: "POST" });
    if (!response.ok) {
      const data = await response.json().catch(() => ({}));
      console.warn(data.error || "Conversation title generation failed.");
    }
  } finally {
    loadSidebar();
  }
}

setKbStatus("Knowledge base idle");
clearEditTarget();
updateHeaderOffset();
applyCanvasPanelWidth(readCanvasWidthPreference(), false);
syncCanvasToggleButton();
const initialSidebarPref = readSidebarPreference();
setSidebarOpen(initialSidebarPref === null ? !isMobileViewport() : initialSidebarPref, false);
syncModelSelectors(modelSel ? modelSel.value : "");
loadSidebar();
updateExportPanel();
void loadKnowledgeBaseDocuments();
