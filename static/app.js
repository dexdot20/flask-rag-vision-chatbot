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
const exportBtn = document.getElementById("export-btn");
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
const canvasBtn = document.getElementById("canvas-btn");
const canvasPanel = document.getElementById("canvas-panel");
const canvasOverlay = document.getElementById("canvas-overlay");
const canvasClose = document.getElementById("canvas-close");
const canvasSearchInput = document.getElementById("canvas-search-input");
const canvasSubtitle = document.getElementById("canvas-subtitle");
const canvasStatus = document.getElementById("canvas-status");
const canvasEmptyState = document.getElementById("canvas-empty-state");
const canvasDocumentEl = document.getElementById("canvas-document");
const canvasDocumentTabsEl = document.getElementById("canvas-document-tabs");
const canvasCopyBtn = document.getElementById("canvas-copy-btn");
const canvasDeleteBtn = document.getElementById("canvas-delete-btn");
const canvasClearBtn = document.getElementById("canvas-clear-btn");
const canvasDownloadHtmlBtn = document.getElementById("canvas-download-html-btn");
const canvasDownloadMdBtn = document.getElementById("canvas-download-md-btn");
const canvasDownloadPdfBtn = document.getElementById("canvas-download-pdf-btn");
const canvasBtnIndicator = document.getElementById("canvas-btn-indicator");
const canvasConfirmModal = document.getElementById("canvas-confirm-modal");
const canvasConfirmOverlay = document.getElementById("canvas-confirm-overlay");
const canvasConfirmTitle = document.getElementById("canvas-confirm-title");
const canvasConfirmMessage = document.getElementById("canvas-confirm-message");
const canvasConfirmOpenBtn = document.getElementById("canvas-confirm-open");
const canvasConfirmLaterBtn = document.getElementById("canvas-confirm-later");
const canvasConfirmCloseBtn = document.getElementById("canvas-confirm-close");
const tokensBtn = document.getElementById("tokens-btn");
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
const mobileExportBtn = document.getElementById("mobile-export-btn");
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
let selectedImageFile = null;
let selectedDocumentFile = null;
let pendingDocumentCanvasOpen = null;
let editingMessageId = null;
let activeCanvasDocumentId = null;
let streamingCanvasDocuments = [];
let canvasHasUnreadUpdates = false;
let lastCanvasTriggerEl = null;
let lastCanvasConfirmTriggerEl = null;
let lastExportTriggerEl = null;
let latestSummaryStatus = null;
let conversationRefreshGeneration = 0;
let pendingConversationRefreshTimers = new Set();
let lastConversationSignature = "";
let pendingCanvasConfirmAction = null;
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

function isDocumentFile(file) {
  if (ALLOWED_DOCUMENT_TYPES.has(file.type)) return true;
  const ext = (file.name || "").toLowerCase().match(/\.[^.]+$/);
  return ext ? DOCUMENT_EXTENSIONS.has(ext[0]) : false;
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
          const lang = rawLang && highlighter.getLanguage(rawLang) ? rawLang : null;
          let highlighted;
          try {
            highlighted = lang
              ? highlighter.highlight(codeText, { language: lang }).value
              : highlighter.highlightAuto(codeText).value;
          } catch (_) {
            highlighted = escHtml(codeText);
          }
          const langClass = lang ? ` language-${lang}` : "";
          const langLabel = lang ? `<span class="canvas-code-lang">${lang}</span>` : "";
          return `<pre>${langLabel}<code class="hljs${langClass}">${highlighted}</code></pre>\n`;
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

function getCanvasDocuments(metadata) {
  if (!metadata || typeof metadata !== "object" || !Array.isArray(metadata.canvas_documents)) {
    return [];
  }

  return metadata.canvas_documents
    .filter((document) => document && typeof document === "object")
    .map((document) => ({
      id: String(document.id || "").trim(),
      title: String(document.title || "Canvas").trim() || "Canvas",
      format: String(document.format || "markdown").trim() || "markdown",
      content: String(document.content || ""),
      line_count: Number.isInteger(Number(document.line_count)) ? Number(document.line_count) : String(document.content || "").split("\n").length,
      source_message_id: Number.isInteger(Number(document.source_message_id)) ? Number(document.source_message_id) : null,
    }))
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

function getActiveCanvasDocument(entries = history) {
  const documents = getCanvasDocumentCollection(entries);
  if (!documents.length) {
    return null;
  }

  const preferredId = String(activeCanvasDocumentId || "").trim();
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

function setPendingDocumentCanvasOpen(file) {
  if (!file) {
    pendingDocumentCanvasOpen = null;
    return;
  }

  pendingDocumentCanvasOpen = {
    fileName: String(file.name || "Document").trim() || "Document",
  };
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
  const fileName = String(pendingRequest?.fileName || "document").trim() || "document";
  const documentLabel = documentCount === 1 ? "canvas document" : `${documentCount} canvas documents`;
  openCanvasConfirmModal({
    title: "Open document in Canvas?",
    message: `${fileName} is ready in Canvas. ${documentLabel.charAt(0).toUpperCase()}${documentLabel.slice(1)} ${documentCount === 1 ? "is" : "are"} available now.`,
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

  const documents = getCanvasDocumentCollection();
  const activeDocument = getActiveCanvasDocument();
  const searchTerm = String(canvasSearchInput?.value || "").trim();
  if (!activeDocument) {
    canvasSubtitle.textContent = "No canvas document yet.";
    canvasEmptyState.hidden = false;
    canvasDocumentEl.hidden = true;
    canvasDocumentEl.innerHTML = "";
    if (canvasDocumentTabsEl) {
      canvasDocumentTabsEl.hidden = true;
      canvasDocumentTabsEl.innerHTML = "";
    }
    if (canvasCopyBtn) {
      canvasCopyBtn.disabled = true;
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
    return;
  }

  activeCanvasDocumentId = activeDocument.id;
  canvasSubtitle.textContent = `${documents.length} doc${documents.length === 1 ? "" : "s"} · ${activeDocument.title} · ${activeDocument.line_count} lines`;
  canvasEmptyState.hidden = true;
  canvasDocumentEl.hidden = false;
  canvasDocumentEl.innerHTML = renderMarkdown(activeDocument.content);
  if (canvasDocumentTabsEl) {
    canvasDocumentTabsEl.hidden = documents.length <= 1;
    canvasDocumentTabsEl.innerHTML = "";
    documents.forEach((entry) => {
      const button = globalThis.document.createElement("button");
      button.type = "button";
      button.className = `canvas-document-tab${entry.id === activeCanvasDocumentId ? " active" : ""}`;
      button.textContent = entry.title;
      button.title = `${entry.title} · ${entry.line_count} lines`;
      button.addEventListener("click", () => {
        activeCanvasDocumentId = entry.id;
        renderCanvasPanel();
      });
      canvasDocumentTabsEl.appendChild(button);
    });
  }
  const matchCount = applyCanvasSearchHighlight(searchTerm);
  if (canvasCopyBtn) {
    canvasCopyBtn.disabled = false;
  }
  if (canvasDeleteBtn) {
    canvasDeleteBtn.disabled = false;
  }
  if (canvasClearBtn) {
    canvasClearBtn.disabled = documents.length === 0;
  }
  if (canvasDownloadHtmlBtn) {
    canvasDownloadHtmlBtn.disabled = false;
  }
  if (canvasDownloadMdBtn) {
    canvasDownloadMdBtn.disabled = false;
  }
  if (canvasDownloadPdfBtn) {
    canvasDownloadPdfBtn.disabled = false;
  }

  if (searchTerm) {
    setCanvasStatus(matchCount ? `${matchCount} match${matchCount === 1 ? "" : "es"} found.` : "No matches found.", matchCount ? "muted" : "warning");
  }
}

function openCanvas() {
  closeMobileTools();
  closeCanvasConfirmModal("cancel", false);
  closeStats();
  closeExportPanel();
  canvasPanel?.classList.add("open");
  canvasOverlay?.classList.add("open");
  canvasPanel?.setAttribute("aria-hidden", "false");
  lastCanvasTriggerEl = document.activeElement instanceof HTMLElement ? document.activeElement : canvasBtn;
  setCanvasAttention(false);
  renderCanvasPanel();
  canvasClose?.focus();
}

function closeCanvas() {
  canvasPanel?.classList.remove("open");
  canvasOverlay?.classList.remove("open");
  canvasPanel?.setAttribute("aria-hidden", "true");
  if (lastCanvasTriggerEl && typeof lastCanvasTriggerEl.focus === "function") {
    lastCanvasTriggerEl.focus();
  }
}

function openExportPanel() {
  closeMobileTools();
  closeStats();
  closeCanvas();
  updateExportPanel();
  exportPanel?.classList.add("open");
  exportOverlay?.classList.add("open");
  exportPanel?.setAttribute("aria-hidden", "false");
  lastExportTriggerEl = document.activeElement instanceof HTMLElement ? document.activeElement : exportBtn;
  exportClose?.focus();
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

function renderBubbleWithCursor(bubbleEl, text) {
  if (!bubbleEl) {
    return;
  }

  bubbleEl.classList.add("streaming-text");
  bubbleEl.innerHTML = renderMarkdown(text);

  const existingCursor = bubbleEl.querySelector(".stream-cursor");
  if (existingCursor) {
    existingCursor.remove();
  }

  const cursorEl = document.createElement("span");
  cursorEl.className = "stream-cursor";
  cursorEl.textContent = "▋";

  const cursorTarget = bubbleEl.querySelector(
    "pre code, p:last-child, li:last-child, blockquote:last-child, h1:last-child, h2:last-child, h3:last-child, h4:last-child, h5:last-child, h6:last-child, td:last-child, th:last-child"
  );
  if (cursorTarget) {
    cursorTarget.appendChild(cursorEl);
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
  "system_prompt",
  "user_messages",
  "assistant_history",
  "tool_results",
  "rag_context",
  "final_instruction",
];

const INPUT_BREAKDOWN_LABELS = {
  system_prompt: "System prompt",
  user_messages: "User messages",
  assistant_history: "Assistant history",
  tool_results: "Tool results",
  rag_context: "RAG context",
  final_instruction: "Final instruction",
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

function normalizeBreakdown(rawBreakdown) {
  const normalized = createEmptyBreakdown();
  const source = rawBreakdown && typeof rawBreakdown === "object" ? rawBreakdown : {};
  INPUT_BREAKDOWN_ORDER.forEach((key) => {
    normalized[key] = Math.max(0, Math.round(toFiniteNumber(source[key], 0)));
  });
  return normalized;
}

function sumBreakdown(breakdown) {
  return INPUT_BREAKDOWN_ORDER.reduce((sum, key) => sum + toFiniteNumber(breakdown[key], 0), 0);
}

function normalizeUsagePayload(usage) {
  const source = usage && typeof usage === "object" ? usage : {};
  const inputBreakdown = normalizeBreakdown(source.input_breakdown);
  const estimatedInputTokens = Math.max(
    sumBreakdown(inputBreakdown),
    Math.round(toFiniteNumber(source.estimated_input_tokens, 0)),
  );

  return {
    prompt_tokens: Math.max(0, Math.round(toFiniteNumber(source.prompt_tokens, 0))),
    completion_tokens: Math.max(0, Math.round(toFiniteNumber(source.completion_tokens, 0))),
    total_tokens: Math.max(0, Math.round(toFiniteNumber(source.total_tokens, 0))),
    estimated_input_tokens: estimatedInputTokens,
    input_breakdown: inputBreakdown,
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

function renderBreakdownList(containerId, breakdown) {
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
      (key) =>
        `<div class="breakdown-row">` +
          `<span class="breakdown-label">${escHtml(INPUT_BREAKDOWN_LABELS[key] || key)}</span>` +
          `<span class="breakdown-value">${fmt(breakdown[key])}</span>` +
        `</div>`,
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
      entries
        .map(
          (key) =>
            `<span class="turn-breakdown-chip">${escHtml(INPUT_BREAKDOWN_LABELS[key] || key)}: ${fmt(breakdown[key])}</span>`,
        )
        .join("") +
    `</div>`
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
  document.getElementById("stat-last-output").textContent = lastTurn ? fmt(lastTurn.completion_tokens) : "—";
  document.getElementById("stat-last-total").textContent = lastTurn ? fmt(lastTurn.total_tokens) : "—";
  document.getElementById("stat-last-model").textContent = lastTurn ? lastTurn.model : "—";
  document.getElementById("stat-breakdown-session-total").textContent = fmt(sumBreakdown(sessionBreakdown));
  document.getElementById("stat-breakdown-latest-total").textContent = lastTurn
    ? fmt(lastTurn.estimated_input_tokens)
    : "—";
  tokensBadge.textContent = fmt(grandTotal);

  renderBreakdownList("session-breakdown-list", sessionBreakdown);
  renderBreakdownList("latest-breakdown-list", lastTurn ? lastTurn.input_breakdown : createEmptyBreakdown());

  const list = document.getElementById("turns-list");
  if (!tokenTurns.length) {
    list.innerHTML = '<div class="breakdown-empty">No completed assistant turns yet.</div>';
    return;
  }

  list.innerHTML = tokenTurns
    .map(
      (turn, index) =>
        `<div class="turn-item">` +
          `<div class="turn-header"><span class="turn-label">Assistant turn ${index + 1}</span> <span class="turn-model">${escHtml(turn.model || "—")}</span></div>` +
          `<div class="turn-details">` +
            `<span class="turn-stat"><span class="stats-dot dot-user"></span>${fmt(turn.prompt_tokens)} in</span>` +
            `<span class="turn-stat"><span class="stats-dot dot-asst"></span>${fmt(turn.completion_tokens)} out</span>` +
            `<span class="turn-stat">${fmt(turn.total_tokens)} total</span>` +
            (turn.cost ? `<span class="turn-stat cost-stat">$${turn.cost.toFixed(6)}</span>` : "") +
          `</div>` +
          renderTurnBreakdownInline(turn.input_breakdown) +
        `</div>`,
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
  if (!Number.isInteger(normalizedId)) {
    return -1;
  }
  return history.findIndex((item) => Number(item.id) === normalizedId);
}

function getHistoryMessage(messageId) {
  const index = getHistoryMessageIndex(messageId);
  return index >= 0 ? history[index] : null;
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

function createMessageActions(messageId) {
  const actions = document.createElement("div");
  actions.className = "msg-actions";

  const editBtn = document.createElement("button");
  editBtn.type = "button";
  editBtn.className = "msg-action-btn";
  editBtn.textContent = "Edit";
  editBtn.disabled = !Number.isInteger(Number(messageId));
  editBtn.addEventListener("click", () => beginEditingMessage(messageId));

  actions.appendChild(editBtn);
  return actions;
}

function createAssistantCanvasActions(metadata) {
  const documents = getCanvasDocuments(metadata);
  if (!documents.length) {
    return null;
  }

  const actions = document.createElement("div");
  actions.className = "msg-actions";

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
  return actions;
}

function renderConversationHistory() {
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
      isEditingTarget: Number(message.id) === Number(editingMessageId),
      isLatestVisible: index === visibleEntries.length - 1,
    }));
  });
  messagesEl.replaceChildren(fragment);
  scrollToBottom();
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
  activeCanvasDocumentId = getActiveCanvasDocument(history)?.id || null;
  lastConversationSignature = serverSignature;
  renderConversationHistory();
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

function applyPersistedMessageIds(persistedIds, assistantEntry) {
  if (!persistedIds || typeof persistedIds !== "object") {
    return;
  }

  const userId = Number(persistedIds.user_message_id);
  if (Number.isInteger(userId)) {
    for (let index = history.length - 1; index >= 0; index -= 1) {
      if (history[index].role === "user") {
        history[index].id = userId;
        break;
      }
    }
  }

  const assistantId = Number(persistedIds.assistant_message_id);
  if (assistantEntry && Number.isInteger(assistantId)) {
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

tokensBtn.addEventListener("click", openStats);
statsClose.addEventListener("click", closeStats);
statsOverlay.addEventListener("click", closeStats);
if (canvasBtn) {
  canvasBtn.addEventListener("click", openCanvas);
}
if (canvasClose) {
  canvasClose.addEventListener("click", closeCanvas);
}
if (canvasOverlay) {
  canvasOverlay.addEventListener("click", closeCanvas);
}
if (exportBtn) {
  exportBtn.addEventListener("click", openExportPanel);
}
if (mobileExportBtn) {
  mobileExportBtn.addEventListener("click", openExportPanel);
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
    const document = getActiveCanvasDocument();
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
}, { passive: true });

window.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    if (isCanvasConfirmOpen()) {
      closeCanvasConfirmModal("cancel");
      return;
    }
    if (isCanvasOpen()) {
      if (canvasSearchInput?.value) {
        canvasSearchInput.value = "";
        renderCanvasPanel();
        setCanvasStatus("Canvas search cleared.", "muted");
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
      `<button class="sidebar-del" title="Delete" data-id="${conversation.id}">` +
      `  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round">` +
      `    <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>` +
      `  </svg>` +
      `</button>`;
    item.addEventListener("click", (event) => {
      if (event.target.closest(".sidebar-del")) {
        return;
      }
      if (conversation.id !== currentConvId) {
        openConversation(conversation.id);
        closeSidebarOnMobile();
      }
    });
    item.querySelector(".sidebar-del").addEventListener("click", (event) => {
      event.stopPropagation();
      deleteConversation(conversation.id);
    });
    sidebarList.appendChild(item);
  });
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
  currentConvId = id;
  currentConvTitle = String(data.conversation?.title || "New Chat").trim() || "New Chat";
  syncModelSelectors(data.conversation.model);
  clearEditTarget();

  history = Array.isArray(data.messages) ? data.messages.map(normalizeHistoryEntry) : [];
  streamingCanvasDocuments = [];
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
  currentConvId = null;
  currentConvTitle = "New Chat";
  history = [];
  latestSummaryStatus = null;
  streamingCanvasDocuments = [];
  activeCanvasDocumentId = null;
  lastConversationSignature = "";
  clearEditTarget();
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
  const isError = /^error:/i.test(withoutCached);
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
kbSyncBtn.addEventListener("click", syncKnowledgeBaseConversations);

imageInputEl.addEventListener("change", () => {
  const file = imageInputEl.files && imageInputEl.files[0];
  if (!file) {
    return;
  }
  if (isDocumentFile(file)) {
    imageInputEl.value = "";
    handleDocumentSelection(file);
    return;
  }
  if (!featureFlags.vision_enabled) {
    showError("Image uploads are disabled. Only documents can be attached.");
    clearSelectedImage();
    return;
  }
  if (!ALLOWED_IMAGE_TYPES.has(file.type)) {
    showError("Unsupported file type. Upload PNG, JPG, WEBP images or DOCX, PDF, TXT, CSV, MD documents.");
    clearSelectedImage();
    return;
  }
  if (file.size > MAX_IMAGE_BYTES) {
    showError("Image is too large. Upload a maximum of 10 MB.");
    clearSelectedImage();
    return;
  }
  selectedImageFile = file;
  clearSelectedDocument();
  renderAttachmentPreview();
});

docInputEl.addEventListener("change", () => {
  const file = docInputEl.files && docInputEl.files[0];
  if (!file) return;
  handleDocumentSelection(file);
});

function handleDocumentSelection(file) {
  if (!isDocumentFile(file)) {
    showError("Unsupported document type. Upload DOCX, PDF, TXT, CSV or MD.");
    clearSelectedDocument();
    return;
  }
  if (file.size > MAX_DOCUMENT_BYTES) {
    showError("Document is too large. Upload a maximum of 20 MB.");
    clearSelectedDocument();
    return;
  }
  selectedDocumentFile = file;
  clearSelectedImage();
  renderAttachmentPreview();
}

function resetTokenStats() {
  tokenTurns.length = 0;
  renderTokenStats();
}

function setStreaming(active) {
  isStreaming = active;
  sendBtn.style.display = active ? "none" : "";
  cancelBtn.style.display = active ? "" : "none";
  fixBtn.disabled = active;
  inputEl.disabled = active;
  attachBtn.disabled = active;
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
  kbStatusEl.textContent = message;
  kbStatusEl.dataset.tone = tone;
}

async function loadKnowledgeBaseDocuments() {
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
  if (!Boolean(featureFlags.rag_enabled)) {
    setKbStatus("RAG disabled in .env", "warning");
    return;
  }
  setKbStatus("Syncing conversations into RAG…");
  kbSyncBtn.disabled = true;
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
    kbSyncBtn.disabled = false;
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
  selectedImageFile = null;
  imageInputEl.value = "";
  renderAttachmentPreview();
}

function clearSelectedDocument() {
  selectedDocumentFile = null;
  docInputEl.value = "";
  renderAttachmentPreview();
}

function clearAllAttachments() {
  selectedImageFile = null;
  selectedDocumentFile = null;
  imageInputEl.value = "";
  docInputEl.value = "";
  renderAttachmentPreview();
}

function renderAttachmentPreview() {
  if (!selectedImageFile && !selectedDocumentFile) {
    attachmentPreviewEl.hidden = true;
    attachmentPreviewEl.innerHTML = "";
    return;
  }

  attachmentPreviewEl.hidden = false;

  if (selectedImageFile) {
    attachmentPreviewEl.innerHTML =
      `<div class="attachment-chip">` +
        `<span class="attachment-chip__icon">🖼️</span>` +
        `<span class="attachment-chip__meta">` +
          `<strong>${escHtml(selectedImageFile.name)}</strong>` +
          `<small>Ready for Qwen vision analysis · ${formatFileSize(selectedImageFile.size)}</small>` +
        `</span>` +
        `<button type="button" class="attachment-chip__remove" title="Remove image">×</button>` +
      `</div>`;
    attachmentPreviewEl.querySelector(".attachment-chip__remove").addEventListener("click", clearAllAttachments);
  } else if (selectedDocumentFile) {
    const ext = (selectedDocumentFile.name || "").split(".").pop().toUpperCase() || "FILE";
    attachmentPreviewEl.innerHTML =
      `<div class="attachment-chip">` +
        `<span class="attachment-chip__icon">📄</span>` +
        `<span class="attachment-chip__meta">` +
          `<strong>${escHtml(selectedDocumentFile.name)}</strong>` +
          `<small>${ext} document · ${formatFileSize(selectedDocumentFile.size)} · Will open in Canvas</small>` +
        `</span>` +
        `<button type="button" class="attachment-chip__remove" title="Remove document">×</button>` +
      `</div>`;
    attachmentPreviewEl.querySelector(".attachment-chip__remove").addEventListener("click", clearAllAttachments);
  }
}

function appendAttachmentBadge(group, metadata) {
  const imageName = metadata && metadata.image_name;
  const fileName = metadata && metadata.file_name;
  if (!imageName && !fileName) {
    return;
  }

  if (fileName) {
    const fileId = metadata.file_id ? String(metadata.file_id).trim() : "";
    const label = fileId ? `${fileName} · ${fileId}` : fileName;
    const badge = document.createElement("div");
    badge.className = "message-attachment";
    badge.innerHTML =
      `<span class="message-attachment__icon">📄</span>` +
      `<span class="message-attachment__name">${escHtml(label)}</span>` +
      `<span class="message-attachment__state">Document uploaded · Canvas</span>`;
    group.appendChild(badge);
    return;
  }

  const imageId = metadata && metadata.image_id ? String(metadata.image_id).trim() : "";
  const hasVisionContext = metadata && (metadata.ocr_text || metadata.vision_summary || metadata.assistant_guidance);
  const stateLabel = hasVisionContext ? "Qwen vision context added" : "Image to be processed";
  const label = imageId ? `${imageName} · ${imageId}` : imageName;

  const badge = document.createElement("div");
  badge.className = "message-attachment";
  badge.innerHTML =
    `<span class="message-attachment__icon">🖼️</span>` +
    `<span class="message-attachment__name">${escHtml(label)}</span>` +
    `<span class="message-attachment__state">${stateLabel}</span>`;
  group.appendChild(badge);
}

function updateAttachmentBadge(group, metadata) {
  const nameEl = group.querySelector(".message-attachment__name");
  const stateEl = group.querySelector(".message-attachment__state");
  if (!stateEl || !nameEl) {
    return;
  }
  const imageName = metadata && metadata.image_name ? String(metadata.image_name) : "";
  const imageId = metadata && metadata.image_id ? String(metadata.image_id).trim() : "";
  nameEl.textContent = imageId ? `${imageName} · ${imageId}` : imageName;
  stateEl.textContent = metadata && (metadata.ocr_text || metadata.vision_summary || metadata.assistant_guidance)
    ? "Qwen vision context added"
    : "Image to be processed";
}

function buildVisionNoteHtml(metadata) {
  if (!metadata) {
    return "";
  }

  const summary = String(metadata.vision_summary || "").trim();
  const guidance = String(metadata.assistant_guidance || "").trim();
  const keyPoints = Array.isArray(metadata.key_points) ? metadata.key_points.filter(Boolean) : [];
  const ocrText = String(metadata.ocr_text || "").trim();
  const imageId = String(metadata.image_id || "").trim();
  const parts = [];

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

  return parts.join("");
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
      state: entry.state || (normalizedSummary.isError ? "error" : "done"),
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

  const label = document.createElement("div");
  label.className = "msg-label";
  label.textContent = role === "user" ? "You" : role === "summary" ? "Summary" : "Assistant";

  metaRow.appendChild(label);
  if (role === "user" && options.editable) {
    metaRow.appendChild(createMessageActions(options.messageId));
  } else if (role === "assistant") {
    const canvasActions = createAssistantCanvasActions(metadata);
    if (canvasActions) {
      metaRow.appendChild(canvasActions);
    }
  }

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  const hasImage = Boolean(metadata && metadata.image_name);
  const hasDocument = Boolean(metadata && metadata.file_name);
  const displayText = text || (hasImage ? "Image uploaded." : hasDocument ? "Document uploaded." : "");

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
  if (role === "user" && hasImage) {
    appendAttachmentBadge(group, metadata);
    appendVisionDetails(group, metadata);
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
  const pendingImage = selectedImageFile;
  const pendingDocument = selectedDocumentFile;
  if (!text && !pendingImage && !pendingDocument) {
    return;
  }

  setPendingDocumentCanvasOpen(pendingDocument);

  if (pendingImage && !Boolean(featureFlags.vision_enabled)) {
    clearSelectedImage();
    showError("Image uploads are disabled in .env.");
    return;
  }

  const editingEntry = getHistoryMessage(editingMessageId);
  const isEditing = Boolean(editingEntry && editingEntry.role === "user");
  const editedMessageId = isEditing ? Number(editingEntry.id) : null;

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

  let userMetadata = pendingImage
    ? { image_name: pendingImage.name }
    : pendingDocument
      ? { file_name: pendingDocument.name }
      : null;
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

    if (!pendingImage && !pendingDocument) {
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
    rebuildTokenStatsFromHistory();
    renderConversationHistory();
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
    if (pendingImage || pendingDocument) {
      const formData = new FormData();
      formData.append("messages", JSON.stringify(requestMessages));
      formData.append("model", modelSel.value);
      formData.append("conversation_id", String(currentConvId));
      formData.append("user_content", text);
      if (editedMessageId !== null) {
        formData.append("edited_message_id", String(editedMessageId));
      }
      if (pendingImage) {
        formData.append("image", pendingImage);
      }
      if (pendingDocument) {
        formData.append("document", pendingDocument);
      }

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
          lastMessage.metadata = {
            ...(lastMessage.metadata || {}),
            image_id: event.image_id,
            image_name: event.image_name,
            ocr_text: event.ocr_text,
            vision_summary: event.vision_summary,
            assistant_guidance: event.assistant_guidance,
            key_points: Array.isArray(event.key_points) ? event.key_points : [],
          };
          updateAttachmentBadge(userGroup, lastMessage.metadata);
          updateVisionDetails(userGroup, lastMessage.metadata);
        }
        scrollToBottom();
      } else if (event.type === "document_processed") {
        const lastMessage = history[history.length - 1];
        if (lastMessage && lastMessage.role === "user") {
          lastMessage.metadata = {
            ...(lastMessage.metadata || {}),
            file_id: event.file_id,
            file_name: event.file_name,
            file_mime_type: event.file_mime_type,
          };
          appendAttachmentBadge(userGroup, lastMessage.metadata);
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
      } else if (event.type === "canvas_sync") {
        streamingCanvasDocuments = Array.isArray(event.documents) ? event.documents.map((document) => ({
          id: String(document.id || "").trim(),
          title: String(document.title || "Canvas").trim() || "Canvas",
          format: String(document.format || "markdown").trim() || "markdown",
          content: String(document.content || ""),
          line_count: Number.isInteger(Number(document.line_count)) ? Number(document.line_count) : String(document.content || "").split("\n").length,
        })).filter((document) => document.id) : [];
        if (streamingCanvasDocuments.length) {
          const activeStillExists = streamingCanvasDocuments.some((document) => document.id === activeCanvasDocumentId);
          activeCanvasDocumentId = activeStillExists
            ? activeCanvasDocumentId
            : streamingCanvasDocuments[streamingCanvasDocuments.length - 1].id;
          renderCanvasPanel();
          const pendingCanvasRequest = pendingDocumentCanvasOpen;
          if (pendingCanvasRequest) {
            consumePendingDocumentCanvasOpen();
          }

          if (pendingCanvasRequest && event.auto_open && !isCanvasOpen()) {
            confirmCanvasOpenForDocument(pendingCanvasRequest, streamingCanvasDocuments.length, {
              onConfirm: () => {
                openCanvas();
                setCanvasStatus(`${pendingCanvasRequest.fileName} opened in Canvas.`, "success");
              },
              onCancel: () => {
                setCanvasAttention(true);
                setCanvasStatus(`${pendingCanvasRequest.fileName} is ready in Canvas. Open the panel when needed.`, "muted");
              },
            });
          } else if (event.auto_open && !isCanvasOpen()) {
            openCanvas();
            setCanvasStatus("Document opened in Canvas.", "success");
          } else if (isCanvasOpen()) {
            setCanvasStatus("Canvas updated.", "success");
          } else {
            setCanvasAttention(true);
            setCanvasStatus("Canvas updated. Open the panel to review.", "success");
          }
        } else if (event.cleared) {
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
    clearEditTarget();
    renderConversationHistory();

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
      clearEditTarget();
      renderConversationHistory();
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
    await fetch(`/api/conversations/${convId}/generate-title`, { method: "POST" });
  } finally {
    loadSidebar();
  }
}

setKbStatus("Knowledge base idle");
clearEditTarget();
updateHeaderOffset();
const initialSidebarPref = readSidebarPreference();
setSidebarOpen(initialSidebarPref === null ? !isMobileViewport() : initialSidebarPref, false);
syncModelSelectors(modelSel ? modelSel.value : "");
loadSidebar();
updateExportPanel();
loadKnowledgeBaseDocuments();
