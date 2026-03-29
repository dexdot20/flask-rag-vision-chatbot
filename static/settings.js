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

const appSettings = bootstrapData.settings || {};
const featureFlags = bootstrapData.features || appSettings.features || {};

const preferencesEl = document.getElementById("preferences-input");
const scratchpadListEl = document.getElementById("scratchpad-list");
const scratchpadAddBtn = document.getElementById("scratchpad-add-btn");
const scratchpadCountEl = document.getElementById("scratchpad-count");
const scratchpadReadonlyNoteEl = document.getElementById("scratchpad-readonly-note");
const maxStepsEl = document.getElementById("max-steps-input");
const summaryModeEl = document.getElementById("summary-mode-select");
const summaryTriggerEl = document.getElementById("summary-trigger-input");
const summarySkipFirstEl = document.getElementById("summary-skip-first-input");
const summarySkipLastEl = document.getElementById("summary-skip-last-input");
const pruningEnabledEl = document.getElementById("pruning-enabled-toggle");
const pruningTokenThresholdEl = document.getElementById("pruning-token-threshold-input");
const pruningBatchSizeEl = document.getElementById("pruning-batch-size-input");
const fetchThresholdEl = document.getElementById("fetch-threshold-input");
const fetchAggressivenessEl = document.getElementById("fetch-aggressiveness-input");
const canvasPromptLinesEl = document.getElementById("canvas-prompt-lines-input");
const canvasExpandLinesEl = document.getElementById("canvas-expand-lines-input");
const canvasScrollLinesEl = document.getElementById("canvas-scroll-lines-input");
const ragAutoInjectEl = document.getElementById("rag-auto-inject-toggle");
const ragInjectOptionsEl = document.getElementById("rag-inject-options");
const ragSensitivityEl = document.getElementById("rag-sensitivity-select");
const ragSensitivityHintEl = document.getElementById("rag-sensitivity-hint");
const ragContextSizeEl = document.getElementById("rag-context-size-select");
const ragSourceTypeEls = Array.from(document.querySelectorAll("input[name='rag-source-type']"));
const ragSourceSummaryEl = document.getElementById("rag-source-summary");
const toolMemoryAutoInjectEl = document.getElementById("tool-memory-auto-inject-toggle");
const toolMemoryDisabledNoteEl = document.getElementById("tool-memory-disabled-note");
const ragDisabledNoteEl = document.getElementById("rag-disabled-note");
const toolToggleEls = Array.from(document.querySelectorAll("#tool-toggles input[type='checkbox']"));
const kbSyncBtn = document.getElementById("kb-sync-btn");
const kbStatusEl = document.getElementById("kb-status");
const kbDocumentsListEl = document.getElementById("kb-documents-list");
const kbUploadFileEl = document.getElementById("kb-upload-file");
const kbUploadTitleEl = document.getElementById("kb-upload-title");
const kbUploadDescriptionEl = document.getElementById("kb-upload-description");
const kbUploadAutoInjectEl = document.getElementById("kb-upload-auto-inject-toggle");
const kbSuggestBtn = document.getElementById("kb-suggest-btn");
const kbUploadBtn = document.getElementById("kb-upload-btn");
const kbUploadStatusEl = document.getElementById("kb-upload-status");
const settingsStatus = document.getElementById("settings-status");
const saveButtons = Array.from(document.querySelectorAll(".settings-save-trigger"));
const dirtyPillEl = document.getElementById("settings-dirty-pill");
const statScratchpadEl = document.getElementById("settings-stat-scratchpad");
const statToolsEl = document.getElementById("settings-stat-tools");
const statRagEl = document.getElementById("settings-stat-rag");
const tabButtons = Array.from(document.querySelectorAll("[data-settings-tab]"));
const tabPanels = Array.from(document.querySelectorAll("[data-settings-panel]"));

const RAG_SENSITIVITY_HINTS = {
  flexible: "Flexible: lower threshold around 0.20, so the system injects broader matches.",
  normal: "Normal: balanced matching with an approximate threshold of 0.35.",
  strict: "Strict: higher threshold around 0.55, so only stronger matches are injected.",
};

const RAG_SOURCE_TYPE_LABELS = {
  conversation: "Chats",
  tool_result: "Tool results",
  tool_memory: "Tool memory",
  uploaded_document: "Uploaded documents",
};

let hasUnsavedChanges = false;

function autoResize(element) {
  if (!element) {
    return;
  }
  element.style.height = "auto";
  element.style.height = `${element.scrollHeight}px`;
}

function setSettingsStatus(message, tone = "muted") {
  if (!settingsStatus) {
    return;
  }
  settingsStatus.textContent = message;
  settingsStatus.dataset.tone = tone;
}

function setDirtyPill(message, tone = "muted") {
  if (!dirtyPillEl) {
    return;
  }
  dirtyPillEl.textContent = message;
  dirtyPillEl.dataset.tone = tone;
}

function markDirty() {
  hasUnsavedChanges = true;
  setSettingsStatus("Unsaved changes", "warning");
  setDirtyPill("Unsaved changes", "warning");
}

function clearDirtyState() {
  hasUnsavedChanges = false;
  setSettingsStatus("Saved", "success");
  setDirtyPill("All changes saved", "success");
}

function normalizeScratchpadNote(value) {
  return String(value || "").replace(/\s+/g, " ").trim();
}

function readNumericSetting(element, defaultValue, { allowZero = true } = {}) {
  if (!element) {
    return defaultValue;
  }
  const rawValue = String(element.value || "").trim();
  if (!rawValue) {
    return defaultValue;
  }
  const parsed = Number.parseInt(rawValue, 10);
  if (Number.isNaN(parsed)) {
    return defaultValue;
  }
  if (!allowZero && parsed === 0) {
    return defaultValue;
  }
  return parsed;
}

function readScratchpadNotesFromList() {
  if (!scratchpadListEl) {
    return [];
  }

  const notes = [];
  const seen = new Set();
  for (const input of scratchpadListEl.querySelectorAll(".scratchpad-note-input")) {
    const note = normalizeScratchpadNote(input.value);
    if (!note || seen.has(note)) {
      continue;
    }
    seen.add(note);
    notes.push(note);
  }

  return notes;
}

function getScratchpadNotesFromSettings() {
  return String(appSettings.scratchpad || "")
    .replace(/\r\n/g, "\n")
    .replace(/\r/g, "\n")
    .split("\n")
    .map((line) => normalizeScratchpadNote(line))
    .filter((line) => line.length > 0);
}

function getVisibleScratchpadNotes() {
  return Boolean(featureFlags.scratchpad_admin_editing)
    ? readScratchpadNotesFromList()
    : getScratchpadNotesFromSettings();
}

function updateScratchpadCount() {
  if (!scratchpadCountEl) {
    return;
  }
  const count = getVisibleScratchpadNotes().length;
  scratchpadCountEl.textContent = count === 1 ? "1 note" : `${count} notes`;
}

function setScratchpadEmptyState(message = "No scratchpad entries yet.") {
  if (!scratchpadListEl) {
    return;
  }

  scratchpadListEl.replaceChildren();
  const emptyState = document.createElement("div");
  emptyState.className = "scratchpad-empty-state";
  emptyState.textContent = message;
  scratchpadListEl.append(emptyState);
  updateScratchpadCount();
  syncOverviewStats();
}

function createScratchpadNoteRow(note = "") {
  const row = document.createElement("div");
  row.className = "scratchpad-note-row";

  const input = document.createElement("input");
  input.type = "text";
  input.className = "settings-text scratchpad-note-input";
  input.placeholder = "One durable note";
  input.value = note;
  input.addEventListener("input", () => {
    markDirty();
    updateScratchpadCount();
    syncOverviewStats();
  });

  const removeBtn = document.createElement("button");
  removeBtn.type = "button";
  removeBtn.className = "btn-ghost btn-ghost--danger scratchpad-note-remove";
  removeBtn.textContent = "Remove";
  removeBtn.addEventListener("click", () => {
    row.remove();
    if (!scratchpadListEl || !scratchpadListEl.querySelector(".scratchpad-note-row")) {
      setScratchpadEmptyState();
    } else {
      updateScratchpadCount();
      syncOverviewStats();
    }
    markDirty();
  });

  row.append(input, removeBtn);
  return row;
}

function createScratchpadReadonlyRow(note = "") {
  const row = document.createElement("div");
  row.className = "scratchpad-note-static";
  row.textContent = note;
  return row;
}

function renderScratchpadList(notes, { editable = true } = {}) {
  if (!scratchpadListEl) {
    return;
  }

  scratchpadListEl.replaceChildren();
  if (!Array.isArray(notes) || !notes.length) {
    setScratchpadEmptyState(editable ? "No scratchpad entries yet. Add a note to get started." : "No scratchpad entries stored yet.");
    return;
  }

  for (const note of notes) {
    scratchpadListEl.append(editable ? createScratchpadNoteRow(note) : createScratchpadReadonlyRow(note));
  }

  updateScratchpadCount();
  syncOverviewStats();
}

function renderScratchpad(editable = true) {
  renderScratchpadList(getScratchpadNotesFromSettings(), { editable });
}

function addScratchpadNote(note = "") {
  if (!scratchpadListEl) {
    return;
  }

  const emptyState = scratchpadListEl.querySelector(".scratchpad-empty-state");
  if (emptyState) {
    emptyState.remove();
  }

  const row = createScratchpadNoteRow(note);
  scratchpadListEl.append(row);
  row.querySelector(".scratchpad-note-input")?.focus();
  updateScratchpadCount();
  syncOverviewStats();
  markDirty();
}

function updateRagSensitivityHint() {
  if (!ragSensitivityHintEl || !ragSensitivityEl) {
    return;
  }
  const sensitivity = ragSensitivityEl.value || "normal";
  ragSensitivityHintEl.textContent = RAG_SENSITIVITY_HINTS[sensitivity] || RAG_SENSITIVITY_HINTS.normal;
}

function getSelectedTools() {
  return toolToggleEls.filter((element) => element.checked).map((element) => element.value);
}

function getSelectedRagSourceTypes() {
  return ragSourceTypeEls.filter((element) => element.checked).map((element) => element.value);
}

function applySelectedTools(selected) {
  const active = new Set(Array.isArray(selected) ? selected : []);
  toolToggleEls.forEach((element) => {
    element.checked = active.has(element.value);
  });
}

function applySelectedRagSourceTypes(selected) {
  const active = new Set(Array.isArray(selected) ? selected : []);
  ragSourceTypeEls.forEach((element) => {
    element.checked = active.has(element.value);
  });
  updateRagSourceSummary();
}

function updateRagSourceSummary() {
  if (!ragSourceSummaryEl) {
    return;
  }

  if (!Boolean(featureFlags.rag_enabled)) {
    ragSourceSummaryEl.textContent = "RAG is disabled in .env, so source pool selection is inactive.";
    return;
  }

  const selected = getSelectedRagSourceTypes();
  if (!selected.length) {
    ragSourceSummaryEl.textContent = "No RAG source pool is selected. The assistant will not retrieve memory context.";
    return;
  }

  const labels = selected.map((value) => RAG_SOURCE_TYPE_LABELS[value] || value).join(", ");
  ragSourceSummaryEl.textContent = `Assistant can search and auto-inject: ${labels}.`;
}

function syncOverviewStats() {
  if (statScratchpadEl) {
    const noteCount = getVisibleScratchpadNotes().length;
    statScratchpadEl.textContent = noteCount === 1 ? "1 note" : `${noteCount} notes`;
  }

  if (statToolsEl) {
    const toolCount = getSelectedTools().length;
    statToolsEl.textContent = toolCount === 1 ? "1 enabled" : `${toolCount} enabled`;
  }

  if (statRagEl) {
    if (!featureFlags.rag_enabled) {
      statRagEl.textContent = "Disabled";
    } else {
      const sourceCount = getSelectedRagSourceTypes().length;
      statRagEl.textContent = sourceCount === 1 ? "1 source" : `${sourceCount} sources`;
    }
  }
}

function applySettingsToForm() {
  const scratchpadAdminEditingEnabled = Boolean(featureFlags.scratchpad_admin_editing);

  if (preferencesEl) {
    preferencesEl.value = appSettings.user_preferences || "";
    autoResize(preferencesEl);
  }
  if (maxStepsEl) maxStepsEl.value = String(appSettings.max_steps || 5);
  if (summaryModeEl) summaryModeEl.value = appSettings.chat_summary_mode || "auto";
  if (summaryTriggerEl) summaryTriggerEl.value = String(appSettings.chat_summary_trigger_token_count || 80000);
  if (summarySkipFirstEl) summarySkipFirstEl.value = String(appSettings.summary_skip_first ?? 2);
  if (summarySkipLastEl) summarySkipLastEl.value = String(appSettings.summary_skip_last ?? 1);
  if (pruningEnabledEl) pruningEnabledEl.checked = Boolean(appSettings.pruning_enabled);
  if (pruningTokenThresholdEl) pruningTokenThresholdEl.value = String(appSettings.pruning_token_threshold || 80000);
  if (pruningBatchSizeEl) pruningBatchSizeEl.value = String(appSettings.pruning_batch_size || 10);
  if (fetchThresholdEl) fetchThresholdEl.value = String(appSettings.fetch_url_token_threshold || 3500);
  if (fetchAggressivenessEl) fetchAggressivenessEl.value = String(appSettings.fetch_url_clip_aggressiveness || 50);
  if (canvasPromptLinesEl) canvasPromptLinesEl.value = String(appSettings.canvas_prompt_max_lines || 800);
  if (canvasExpandLinesEl) canvasExpandLinesEl.value = String(appSettings.canvas_expand_max_lines || 1600);
  if (canvasScrollLinesEl) canvasScrollLinesEl.value = String(appSettings.canvas_scroll_window_lines || 200);
  applySelectedTools(appSettings.active_tools || []);
  if (ragAutoInjectEl) {
    ragAutoInjectEl.checked = Boolean(featureFlags.rag_enabled ? appSettings.rag_auto_inject : false);
  }
  if (ragSensitivityEl) {
    ragSensitivityEl.value = appSettings.rag_sensitivity || "normal";
  }
  if (ragContextSizeEl) {
    ragContextSizeEl.value = appSettings.rag_context_size || "medium";
  }
  applySelectedRagSourceTypes(appSettings.rag_source_types || []);
  if (toolMemoryAutoInjectEl) {
    toolMemoryAutoInjectEl.checked = Boolean(featureFlags.rag_enabled ? appSettings.tool_memory_auto_inject : false);
  }
  updateRagSensitivityHint();
  if (scratchpadListEl || scratchpadAddBtn || scratchpadCountEl) {
    if (scratchpadAdminEditingEnabled) {
      renderScratchpad(true);
    } else {
      renderScratchpad(false);
    }
  }
  syncOverviewStats();
}

function applyFeatureAvailability() {
  const ragEnabled = Boolean(featureFlags.rag_enabled);
  const scratchpadAdminEditingEnabled = Boolean(featureFlags.scratchpad_admin_editing);

  if (ragAutoInjectEl) ragAutoInjectEl.disabled = !ragEnabled;
  if (ragSensitivityEl) ragSensitivityEl.disabled = !ragEnabled;
  if (ragContextSizeEl) ragContextSizeEl.disabled = !ragEnabled;
  ragSourceTypeEls.forEach((element) => {
    element.disabled = !ragEnabled;
  });
  if (kbSyncBtn) kbSyncBtn.disabled = !ragEnabled;
  if (toolMemoryAutoInjectEl) toolMemoryAutoInjectEl.disabled = !ragEnabled;
  if (kbUploadFileEl) kbUploadFileEl.disabled = !ragEnabled;
  if (kbUploadTitleEl) kbUploadTitleEl.disabled = !ragEnabled;
  if (kbUploadDescriptionEl) kbUploadDescriptionEl.disabled = !ragEnabled;
  if (kbUploadAutoInjectEl) kbUploadAutoInjectEl.disabled = !ragEnabled;
  if (kbUploadBtn) kbUploadBtn.disabled = !ragEnabled;
  if (ragInjectOptionsEl) {
    ragInjectOptionsEl.classList.toggle("is-disabled", !ragEnabled);
  }
  if (ragDisabledNoteEl) {
    ragDisabledNoteEl.hidden = ragEnabled;
  }
  if (toolMemoryDisabledNoteEl) {
    toolMemoryDisabledNoteEl.hidden = ragEnabled;
  }
  if (scratchpadAddBtn) {
    scratchpadAddBtn.hidden = !scratchpadAdminEditingEnabled;
  }
  if (scratchpadReadonlyNoteEl) {
    scratchpadReadonlyNoteEl.hidden = scratchpadAdminEditingEnabled;
  }
  if (!ragEnabled) {
    setKbStatus("RAG disabled in .env", "warning");
    setKbUploadStatus("Upload disabled because RAG is off", "warning");
  } else {
    setKbUploadStatus("Ready to upload", "muted");
  }
  updateRagSourceSummary();
  syncOverviewStats();
}

async function refreshSettings() {
  try {
    const response = await fetch("/api/settings");
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "Failed to load settings.");
    }

    appSettings.user_preferences = data.user_preferences || "";
    appSettings.scratchpad = data.scratchpad || "";
    appSettings.max_steps = data.max_steps || 5;
    appSettings.chat_summary_mode = data.chat_summary_mode || "auto";
    appSettings.chat_summary_trigger_token_count = data.chat_summary_trigger_token_count || 80000;
    appSettings.summary_skip_first = data.summary_skip_first ?? 2;
    appSettings.summary_skip_last = data.summary_skip_last ?? 1;
    appSettings.pruning_enabled = Boolean(data.pruning_enabled);
    appSettings.pruning_token_threshold = data.pruning_token_threshold || 80000;
    appSettings.pruning_batch_size = data.pruning_batch_size || 10;
    appSettings.fetch_url_token_threshold = data.fetch_url_token_threshold || 3500;
    appSettings.fetch_url_clip_aggressiveness = data.fetch_url_clip_aggressiveness ?? 50;
    appSettings.canvas_prompt_max_lines = data.canvas_prompt_max_lines || 800;
    appSettings.canvas_expand_max_lines = data.canvas_expand_max_lines || 1600;
    appSettings.canvas_scroll_window_lines = data.canvas_scroll_window_lines || 200;
    appSettings.active_tools = Array.isArray(data.active_tools) ? data.active_tools : [];
    appSettings.rag_auto_inject = Boolean(data.rag_auto_inject);
    appSettings.rag_sensitivity = data.rag_sensitivity || "normal";
    appSettings.rag_context_size = data.rag_context_size || "medium";
    appSettings.rag_source_types = Array.isArray(data.rag_source_types) ? data.rag_source_types : [];
    appSettings.tool_memory_auto_inject = Boolean(data.tool_memory_auto_inject);
    if (data.features && typeof data.features === "object") {
      Object.assign(featureFlags, data.features);
    }

    applySettingsToForm();
    applyFeatureAvailability();
    setSettingsStatus("Ready");
    setDirtyPill("All changes saved", "muted");
  } catch (error) {
    setSettingsStatus(error.message || "Failed to load settings.", "error");
    setDirtyPill("Load failed", "error");
  }
}

async function saveSettings() {
  const scratchpadAdminEditingEnabled = Boolean(featureFlags.scratchpad_admin_editing);
  const payload = {
    user_preferences: preferencesEl?.value.trim() || "",
    max_steps: readNumericSetting(maxStepsEl, 5, { allowZero: false }),
    chat_summary_mode: summaryModeEl?.value || "auto",
    chat_summary_trigger_token_count: readNumericSetting(summaryTriggerEl, 80000, { allowZero: false }),
    summary_skip_first: readNumericSetting(summarySkipFirstEl, 0),
    summary_skip_last: readNumericSetting(summarySkipLastEl, 1),
    pruning_enabled: Boolean(pruningEnabledEl?.checked),
    pruning_token_threshold: readNumericSetting(pruningTokenThresholdEl, 80000, { allowZero: false }),
    pruning_batch_size: readNumericSetting(pruningBatchSizeEl, 10, { allowZero: false }),
    fetch_url_token_threshold: readNumericSetting(fetchThresholdEl, 3500, { allowZero: false }),
    fetch_url_clip_aggressiveness: readNumericSetting(fetchAggressivenessEl, 50),
    canvas_prompt_max_lines: readNumericSetting(canvasPromptLinesEl, 800, { allowZero: false }),
    canvas_expand_max_lines: readNumericSetting(canvasExpandLinesEl, 1600, { allowZero: false }),
    canvas_scroll_window_lines: readNumericSetting(canvasScrollLinesEl, 200, { allowZero: false }),
    active_tools: getSelectedTools(),
    rag_auto_inject: featureFlags.rag_enabled ? Boolean(ragAutoInjectEl?.checked) : false,
    rag_sensitivity: ragSensitivityEl?.value || "normal",
    rag_context_size: ragContextSizeEl?.value || "medium",
    rag_source_types: featureFlags.rag_enabled ? getSelectedRagSourceTypes() : [],
    tool_memory_auto_inject: featureFlags.rag_enabled ? Boolean(toolMemoryAutoInjectEl?.checked) : false,
  };

  if (scratchpadAdminEditingEnabled) {
    payload.scratchpad = readScratchpadNotesFromList().join("\n");
  }

  saveButtons.forEach((button) => {
    button.disabled = true;
  });
  setSettingsStatus("Saving...");
  setDirtyPill("Saving...", "warning");

  try {
    const response = await fetch("/api/settings", {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "Failed to save settings.");
    }

    appSettings.user_preferences = data.user_preferences || "";
    appSettings.scratchpad = data.scratchpad || "";
    appSettings.max_steps = data.max_steps || 5;
    appSettings.chat_summary_mode = data.chat_summary_mode || "auto";
    appSettings.chat_summary_trigger_token_count = data.chat_summary_trigger_token_count || 80000;
    appSettings.summary_skip_first = data.summary_skip_first ?? 2;
    appSettings.summary_skip_last = data.summary_skip_last ?? 1;
    appSettings.pruning_enabled = Boolean(data.pruning_enabled);
    appSettings.pruning_token_threshold = data.pruning_token_threshold || 80000;
    appSettings.pruning_batch_size = data.pruning_batch_size || 10;
    appSettings.fetch_url_token_threshold = data.fetch_url_token_threshold || 3500;
    appSettings.fetch_url_clip_aggressiveness = data.fetch_url_clip_aggressiveness ?? 50;
    appSettings.canvas_prompt_max_lines = data.canvas_prompt_max_lines || 800;
    appSettings.canvas_expand_max_lines = data.canvas_expand_max_lines || 1600;
    appSettings.canvas_scroll_window_lines = data.canvas_scroll_window_lines || 200;
    appSettings.active_tools = Array.isArray(data.active_tools) ? data.active_tools : [];
    appSettings.rag_auto_inject = Boolean(data.rag_auto_inject);
    appSettings.rag_sensitivity = data.rag_sensitivity || "normal";
    appSettings.rag_context_size = data.rag_context_size || "medium";
    appSettings.rag_source_types = Array.isArray(data.rag_source_types) ? data.rag_source_types : [];
    appSettings.tool_memory_auto_inject = Boolean(data.tool_memory_auto_inject);
    if (data.features && typeof data.features === "object") {
      Object.assign(featureFlags, data.features);
    }

    applySettingsToForm();
    applyFeatureAvailability();
    clearDirtyState();
  } catch (error) {
    setSettingsStatus(error.message || "Failed to save settings.", "error");
    setDirtyPill("Save failed", "error");
  } finally {
    saveButtons.forEach((button) => {
      button.disabled = false;
    });
  }
}

function setKbStatus(message, tone = "muted") {
  if (!kbStatusEl) {
    return;
  }
  kbStatusEl.textContent = message;
  kbStatusEl.dataset.tone = tone;
}

function setKbUploadStatus(message, tone = "muted") {
  if (!kbUploadStatusEl) {
    return;
  }
  kbUploadStatusEl.textContent = message;
  kbUploadStatusEl.dataset.tone = tone;
}

function syncKbUploadActionState() {
  const ragEnabled = Boolean(featureFlags.rag_enabled);
  const hasFile = Boolean(kbUploadFileEl?.files?.length);

  if (kbSuggestBtn) {
    kbSuggestBtn.disabled = !ragEnabled || !hasFile;
  }
  if (kbUploadBtn) {
    kbUploadBtn.disabled = !ragEnabled || !hasFile;
  }
}

function summarizeKbDocument(doc) {
  const metadata = doc && typeof doc.metadata === "object" ? doc.metadata : {};
  const parts = [RAG_SOURCE_TYPE_LABELS[doc.source_type] || doc.source_type || "Document"];
  if (doc.category) {
    parts.push(doc.category);
  }
  parts.push(`${doc.chunk_count || 0} chunks`);
  if (metadata.file_name) {
    parts.unshift(metadata.file_name);
  }
  return parts.join(" · ");
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
    sub.textContent = summarizeKbDocument(doc);

    meta.append(title, sub);

    const metadata = doc && typeof doc.metadata === "object" ? doc.metadata : {};
    const description = String(metadata.description || "").trim();
    if (description) {
      const descriptionEl = document.createElement("div");
      descriptionEl.className = "kb-doc-description";
      descriptionEl.textContent = description;
      meta.append(descriptionEl);
    }

    const badges = document.createElement("div");
    badges.className = "kb-doc-badges";
    if (doc.source_type === "uploaded_document") {
      const uploadBadge = document.createElement("span");
      uploadBadge.className = "kb-doc-badge";
      uploadBadge.textContent = "manual upload";
      badges.append(uploadBadge);
    }
    const autoInjectBadge = document.createElement("span");
    autoInjectBadge.className = "kb-doc-badge";
    autoInjectBadge.dataset.tone = metadata.auto_inject_enabled === false ? "muted" : "success";
    autoInjectBadge.textContent = metadata.auto_inject_enabled === false ? "manual only" : "auto inject on";
    badges.append(autoInjectBadge);
    meta.append(badges);

    const del = document.createElement("button");
    del.type = "button";
    del.className = "kb-doc-delete";
    del.textContent = "Delete";
    del.addEventListener("click", () => {
      void deleteKnowledgeBaseDocument(doc.source_key);
    });

    item.append(meta, del);
    kbDocumentsListEl.append(item);
  });
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
    setKbStatus("Failed to load indexed sources.", "error");
  }
}

async function deleteKnowledgeBaseDocument(sourceKey) {
  if (!sourceKey) {
    return;
  }
  setKbStatus("Deleting source...");
  try {
    const response = await fetch(`/api/rag/documents/${encodeURIComponent(sourceKey)}`, { method: "DELETE" });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(data.error || "Delete failed.");
    }
    setKbStatus("Source deleted", "success");
    await loadKnowledgeBaseDocuments();
  } catch (error) {
    setKbStatus(error.message || "Delete failed.", "error");
  }
}

async function uploadKnowledgeBaseDocument() {
  if (!Boolean(featureFlags.rag_enabled)) {
    setKbUploadStatus("RAG disabled in .env", "warning");
    return;
  }

  const file = kbUploadFileEl?.files?.[0];
  if (!file) {
    setKbUploadStatus("Choose a document to upload.", "warning");
    return;
  }

  const formData = new FormData();
  formData.append("document", file);
  formData.append("source_name", kbUploadTitleEl?.value.trim() || "");
  formData.append("description", kbUploadDescriptionEl?.value.trim() || "");
  formData.append("auto_inject_enabled", kbUploadAutoInjectEl?.checked ? "true" : "false");

  if (kbUploadBtn) {
    kbUploadBtn.disabled = true;
  }
  if (kbSuggestBtn) {
    kbSuggestBtn.disabled = true;
  }
  setKbUploadStatus("Uploading document...");

  try {
    const response = await fetch("/api/rag/ingest", {
      method: "POST",
      body: formData,
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(data.error || "Upload failed.");
    }

    if (kbUploadFileEl) {
      kbUploadFileEl.value = "";
    }
    if (kbUploadTitleEl) {
      kbUploadTitleEl.value = "";
    }
    if (kbUploadDescriptionEl) {
      kbUploadDescriptionEl.value = "";
      autoResize(kbUploadDescriptionEl);
    }
    if (kbUploadAutoInjectEl) {
      kbUploadAutoInjectEl.checked = true;
    }

    const sourceName = data.document?.source_name || data.file_name || "Document";
    setKbUploadStatus(`${sourceName} indexed`, "success");
    await loadKnowledgeBaseDocuments();
  } catch (error) {
    setKbUploadStatus(error.message || "Upload failed.", "error");
  } finally {
    syncKbUploadActionState();
  }
}

async function generateKnowledgeBaseMetadata() {
  if (!Boolean(featureFlags.rag_enabled)) {
    setKbUploadStatus("RAG disabled in .env", "warning");
    return;
  }

  const file = kbUploadFileEl?.files?.[0];
  if (!file) {
    setKbUploadStatus("Choose a document first.", "warning");
    return;
  }

  const formData = new FormData();
  formData.append("document", file);
  formData.append("source_name", kbUploadTitleEl?.value.trim() || "");
  formData.append("description", kbUploadDescriptionEl?.value.trim() || "");

  if (kbSuggestBtn) {
    kbSuggestBtn.disabled = true;
  }
  if (kbUploadBtn) {
    kbUploadBtn.disabled = true;
  }
  setKbUploadStatus("Generating title and description...", "muted");

  try {
    const response = await fetch("/api/rag/upload-metadata", {
      method: "POST",
      body: formData,
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(data.error || "Metadata generation failed.");
    }

    if (kbUploadTitleEl && typeof data.title === "string") {
      kbUploadTitleEl.value = data.title;
    }
    if (kbUploadDescriptionEl && typeof data.description === "string") {
      kbUploadDescriptionEl.value = data.description;
      autoResize(kbUploadDescriptionEl);
    }

    setKbUploadStatus("Title and description generated.", "success");
  } catch (error) {
    setKbUploadStatus(error.message || "Metadata generation failed.", "error");
  } finally {
    syncKbUploadActionState();
  }
}

async function syncKnowledgeBaseConversations() {
  if (!Boolean(featureFlags.rag_enabled)) {
    setKbStatus("RAG disabled in .env", "warning");
    return;
  }

  if (kbSyncBtn) {
    kbSyncBtn.disabled = true;
  }
  setKbStatus("Syncing conversations into RAG...");

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
    await loadKnowledgeBaseDocuments();
  } catch (error) {
    setKbStatus(error.message || "Conversation sync failed.", "error");
  } finally {
    if (kbSyncBtn) {
      kbSyncBtn.disabled = false;
    }
  }
}

function activateTab(tabId, updateHash = true) {
  const nextId = String(tabId || "assistant");

  tabButtons.forEach((button) => {
    const isActive = button.dataset.settingsTab === nextId;
    button.classList.toggle("active", isActive);
    button.setAttribute("aria-selected", String(isActive));
  });

  tabPanels.forEach((panel) => {
    const isActive = panel.dataset.settingsPanel === nextId;
    panel.classList.toggle("active", isActive);
    panel.toggleAttribute("hidden", !isActive);
  });

  if (updateHash) {
    history.replaceState(null, "", `#${nextId}`);
  }
}

function initializeTabs() {
  tabButtons.forEach((button) => {
    button.addEventListener("click", () => activateTab(button.dataset.settingsTab));
  });

  const hash = String(window.location.hash || "").replace(/^#/, "");
  const initialTab = tabButtons.some((button) => button.dataset.settingsTab === hash) ? hash : "assistant";
  activateTab(initialTab, false);
}

function registerDirtyListeners() {
  preferencesEl?.addEventListener("input", () => {
    autoResize(preferencesEl);
    markDirty();
  });
  maxStepsEl?.addEventListener("input", markDirty);
  summaryModeEl?.addEventListener("change", markDirty);
  summaryTriggerEl?.addEventListener("input", markDirty);
  summarySkipFirstEl?.addEventListener("input", markDirty);
  summarySkipLastEl?.addEventListener("input", markDirty);
  pruningEnabledEl?.addEventListener("change", markDirty);
  pruningTokenThresholdEl?.addEventListener("input", markDirty);
  pruningBatchSizeEl?.addEventListener("input", markDirty);
  fetchThresholdEl?.addEventListener("input", markDirty);
  fetchAggressivenessEl?.addEventListener("input", markDirty);
  canvasPromptLinesEl?.addEventListener("input", markDirty);
  canvasExpandLinesEl?.addEventListener("input", markDirty);
  canvasScrollLinesEl?.addEventListener("input", markDirty);
  ragAutoInjectEl?.addEventListener("change", markDirty);
  ragSensitivityEl?.addEventListener("change", () => {
    updateRagSensitivityHint();
    markDirty();
  });
  ragContextSizeEl?.addEventListener("change", markDirty);
  ragSourceTypeEls.forEach((element) => {
    element.addEventListener("change", () => {
      updateRagSourceSummary();
      syncOverviewStats();
      markDirty();
    });
  });
  toolMemoryAutoInjectEl?.addEventListener("change", markDirty);
  toolToggleEls.forEach((element) => {
    element.addEventListener("change", () => {
      markDirty();
      syncOverviewStats();
    });
  });
}

scratchpadAddBtn?.addEventListener("click", () => addScratchpadNote());
kbUploadFileEl?.addEventListener("change", () => {
  const filename = kbUploadFileEl.files?.[0]?.name || "";
  if (filename && kbUploadTitleEl && !kbUploadTitleEl.value.trim()) {
    kbUploadTitleEl.value = filename.replace(/\.[^.]+$/, "");
  }
  syncKbUploadActionState();
});
kbUploadDescriptionEl?.addEventListener("input", () => autoResize(kbUploadDescriptionEl));
kbSuggestBtn?.addEventListener("click", () => {
  void generateKnowledgeBaseMetadata();
});
kbUploadBtn?.addEventListener("click", () => {
  void uploadKnowledgeBaseDocument();
});
kbSyncBtn?.addEventListener("click", () => {
  void syncKnowledgeBaseConversations();
});
saveButtons.forEach((button) => {
  button.addEventListener("click", () => {
    void saveSettings();
  });
});

window.addEventListener("beforeunload", (event) => {
  if (!hasUnsavedChanges) {
    return;
  }
  event.preventDefault();
  event.returnValue = "";
});

window.addEventListener("keydown", (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "s") {
    event.preventDefault();
    void saveSettings();
  }
});

initializeTabs();
registerDirtyListeners();
applySettingsToForm();
applyFeatureAvailability();
syncKbUploadActionState();
setSettingsStatus("Ready");
setDirtyPill("All changes saved", "muted");
void refreshSettings();
void loadKnowledgeBaseDocuments();
