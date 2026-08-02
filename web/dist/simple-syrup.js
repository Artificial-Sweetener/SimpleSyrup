// web/src/main.ts
import { app } from "../../../scripts/app.js";

// web/src/api.ts
var SETTINGS_ROUTE = "/simple-syrup/settings";
var EXTERNAL_LLM_SETTINGS_ROUTE = "/simple-syrup/external-llm/settings";
var EXTERNAL_LLM_API_KEY_ROUTE = "/simple-syrup/external-llm/api-key";
var EXTERNAL_LLM_MODELS_REFRESH_ROUTE = "/simple-syrup/external-llm/models/refresh";
var MASK_BATCH_PREVIEW_ROUTE = "/simple-syrup/mask-batch/preview";
async function getMaskBatchPreview(files, channel, fetchImpl = fetch) {
  const response = await fetchImpl(MASK_BATCH_PREVIEW_ROUTE, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ files, channel })
  });
  if (!response.ok) {
    throw new Error(
      await backendErrorMessage(
        response,
        `Could not render Load Mask Batch preview. Backend returned ${String(response.status)}.`
      )
    );
  }
  return parseMaskBatchPreview(await response.json());
}
function parseMaskBatchPreview(payload) {
  if (!isMaskBatchPreviewPayload(payload)) {
    throw new Error(
      "SimpleSyrup mask batch preview payload is invalid. Expected native images and animation flags."
    );
  }
  return {
    images: payload.images.map((image) => ({ ...image })),
    animated: [...payload.animated]
  };
}
async function getSettings(fetchImpl = fetch) {
  const response = await fetchImpl(SETTINGS_ROUTE);
  if (!response.ok) {
    throw new Error(
      await backendErrorMessage(
        response,
        `Could not load SimpleSyrup settings. Backend returned ${String(response.status)}.`
      )
    );
  }
  return parseSettings(await response.json());
}
async function saveSettings(settings, fetchImpl = fetch) {
  const response = await fetchImpl(SETTINGS_ROUTE, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(settings)
  });
  if (!response.ok) {
    throw new Error(
      await backendErrorMessage(
        response,
        `Could not save SimpleSyrup settings. Backend returned ${String(response.status)}.`
      )
    );
  }
  return parseSettings(await response.json());
}
function parseSettings(payload) {
  if (!isSettingsPayload(payload)) {
    throw new Error(
      "SimpleSyrup settings payload is invalid. Expected show_downloadable_models to be a boolean."
    );
  }
  return {
    show_downloadable_models: payload.show_downloadable_models
  };
}
async function getExternalLLMSettings(fetchImpl = fetch) {
  const response = await fetchImpl(EXTERNAL_LLM_SETTINGS_ROUTE);
  if (!response.ok) {
    throw new Error(
      await backendErrorMessage(
        response,
        `Could not load external LLM settings. Backend returned ${String(response.status)}.`
      )
    );
  }
  return parseExternalLLMSettings(await response.json());
}
async function saveExternalLLMSettings(settings, fetchImpl = fetch) {
  const response = await fetchImpl(EXTERNAL_LLM_SETTINGS_ROUTE, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(settings)
  });
  if (!response.ok) {
    throw new Error(
      await backendErrorMessage(
        response,
        `Could not save external LLM settings. Backend returned ${String(response.status)}.`
      )
    );
  }
  return parseExternalLLMSettings(await response.json());
}
async function saveExternalLLMApiKey(payload, fetchImpl = fetch) {
  const response = await fetchImpl(EXTERNAL_LLM_API_KEY_ROUTE, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  if (!response.ok) {
    throw new Error(
      await backendErrorMessage(
        response,
        `Could not save external LLM API key. Backend returned ${String(response.status)}.`
      )
    );
  }
  return parseExternalLLMSettings(await response.json());
}
async function refreshExternalLLMModels(fetchImpl = fetch) {
  const response = await fetchImpl(EXTERNAL_LLM_MODELS_REFRESH_ROUTE, {
    method: "POST"
  });
  if (!response.ok) {
    throw new Error(
      await backendErrorMessage(
        response,
        `Could not refresh external LLM models. Backend returned ${String(response.status)}.`
      )
    );
  }
  return parseExternalLLMSettings(await response.json());
}
function parseExternalLLMSettings(payload) {
  if (!isExternalLLMSettingsPayload(payload)) {
    throw new Error(
      "External LLM settings payload is invalid. Expected base_url, cached_models, default_model, and has_api_key."
    );
  }
  return {
    base_url: payload.base_url,
    cached_models: [...payload.cached_models],
    default_model: payload.default_model,
    has_api_key: payload.has_api_key
  };
}
function isSettingsPayload(payload) {
  return typeof payload === "object" && payload !== null && typeof payload.show_downloadable_models === "boolean";
}
function isExternalLLMSettingsPayload(payload) {
  return typeof payload === "object" && payload !== null && typeof payload.base_url === "string" && Array.isArray(payload.cached_models) && payload.cached_models?.every(
    (model) => typeof model === "string"
  ) === true && typeof payload.default_model === "string" && typeof payload.has_api_key === "boolean";
}
function isMaskBatchPreviewPayload(payload) {
  if (typeof payload !== "object" || payload === null) return false;
  const candidate = payload;
  return Array.isArray(candidate.images) && candidate.images.every(isComfyImageResult) && Array.isArray(candidate.animated) && candidate.animated.every((value) => typeof value === "boolean");
}
function isComfyImageResult(value) {
  if (typeof value !== "object" || value === null) return false;
  const candidate = value;
  return typeof candidate.filename === "string" && typeof candidate.subfolder === "string" && (candidate.type === "input" || candidate.type === "output" || candidate.type === "temp");
}
async function backendErrorMessage(response, fallback) {
  try {
    const payload = await response.json();
    if (typeof payload === "object" && payload !== null && typeof payload.error === "string") {
      return payload.error;
    }
  } catch {
    return fallback;
  }
  return fallback;
}

// web/src/settings.ts
var SIMPLE_SYRUP_SETTING_ID = "SimpleSyrup.ShowDownloadableModels";
var SIMPLE_SYRUP_SETTING_LABEL = "SimpleSyrup: Show downloadable models in loader dropdowns";
var SIMPLE_SYRUP_SETTING_DESCRIPTION = "Show known downloadable SAM, GroundingDINO, and ViTMatte models even when they are not installed locally.";
var EXTERNAL_LLM_ENDPOINT_SETTING_ID = "SimpleSyrup.ExternalLLM.Endpoint";
var EXTERNAL_LLM_ENDPOINT_SETTING_LABEL = "SimpleSyrup: External LLM endpoint";
var EXTERNAL_LLM_ENDPOINT_SETTING_DESCRIPTION = "OpenAI-compatible endpoint base URL used by SimpleSyrup prompt nodes.";
var EXTERNAL_LLM_API_KEY_SETTING_ID = "SimpleSyrup.ExternalLLM.ApiKey";
var EXTERNAL_LLM_API_KEY_SETTING_LABEL = "SimpleSyrup: External LLM API key";
var EXTERNAL_LLM_API_KEY_SETTING_DESCRIPTION = "Stores the API key for the configured external LLM endpoint in OS credential storage.";
var DEFAULT_SETTINGS = {
  show_downloadable_models: true
};
async function registerSimpleSyrupSettings(app2, api = {
  getSettings,
  saveSettings,
  getExternalLLMSettings,
  saveExternalLLMSettings,
  saveExternalLLMApiKey
}, logger = console) {
  let initialSettings = DEFAULT_SETTINGS;
  let externalLLMSettings = {
    base_url: "",
    cached_models: [],
    default_model: "",
    has_api_key: false
  };
  try {
    initialSettings = await api.getSettings();
  } catch (error) {
    logger.warn(
      "Could not load SimpleSyrup settings. Using the default setting until the backend is available.",
      error
    );
  }
  try {
    externalLLMSettings = await api.getExternalLLMSettings();
  } catch (error) {
    logger.warn(
      "Could not load SimpleSyrup external LLM settings. Using empty endpoint settings until the backend is available.",
      error
    );
  }
  let savedSettings = initialSettings;
  let savedExternalLLMSettings = externalLLMSettings;
  installSimpleSyrupSettingsStyle();
  const setting = app2.ui.settings.addSetting({
    id: SIMPLE_SYRUP_SETTING_ID,
    name: SIMPLE_SYRUP_SETTING_LABEL,
    type: "boolean",
    defaultValue: initialSettings.show_downloadable_models,
    tooltip: SIMPLE_SYRUP_SETTING_DESCRIPTION,
    onChange: async (value) => {
      try {
        const saved = await api.saveSettings({
          show_downloadable_models: value
        });
        savedSettings = saved;
        setting.value = saved.show_downloadable_models;
      } catch (error) {
        logger.warn(
          "Could not save SimpleSyrup settings. The backend rejected the setting update.",
          error
        );
        setting.value = savedSettings.show_downloadable_models;
      }
    }
  });
  setting.value = initialSettings.show_downloadable_models;
  app2.ui.settings.addSetting({
    id: EXTERNAL_LLM_ENDPOINT_SETTING_ID,
    name: EXTERNAL_LLM_ENDPOINT_SETTING_LABEL,
    sortOrder: 320,
    type: () => createExternalLLMEndpointControl({
      api,
      logger,
      refreshModelChoices: () => refreshExternalLLMModelChoices(app2, logger),
      getSettings: () => savedExternalLLMSettings,
      setSettings: (settings) => {
        savedExternalLLMSettings = settings;
      }
    }),
    defaultValue: externalLLMSettings.base_url,
    tooltip: EXTERNAL_LLM_ENDPOINT_SETTING_DESCRIPTION
  });
  app2.ui.settings.addSetting({
    id: EXTERNAL_LLM_API_KEY_SETTING_ID,
    name: EXTERNAL_LLM_API_KEY_SETTING_LABEL,
    sortOrder: 319,
    type: () => createExternalLLMApiKeyControl({
      api,
      logger,
      refreshModelChoices: () => refreshExternalLLMModelChoices(app2, logger),
      getSettings: () => savedExternalLLMSettings,
      setSettings: (settings) => {
        savedExternalLLMSettings = settings;
      }
    }),
    defaultValue: "",
    tooltip: EXTERNAL_LLM_API_KEY_SETTING_DESCRIPTION
  });
}
function endpointShouldBeSaved(value) {
  const endpoint = value.trim();
  if (!endpoint) {
    return true;
  }
  try {
    const parsed = new URL(endpoint);
    return (parsed.protocol === "http:" || parsed.protocol === "https:") && parsed.hostname.length > 0;
  } catch {
    return false;
  }
}
function installSimpleSyrupSettingsStyle() {
  if (document.getElementById("simple-syrup-settings-style")) {
    return;
  }
  const style = document.createElement("style");
  style.id = "simple-syrup-settings-style";
  style.textContent = `
    .simple-syrup-settings-row {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      min-width: min(38rem, 100%);
    }
    .simple-syrup-settings-row[data-pending="true"] {
      opacity: 0.75;
    }
    .simple-syrup-settings-input {
      min-width: 16rem;
      flex: 1 1 auto;
    }
    .simple-syrup-settings-button {
      flex: 0 0 auto;
      white-space: nowrap;
    }
    .simple-syrup-settings-status {
      color: var(--fg-color);
      opacity: 0.8;
      white-space: normal;
      overflow-wrap: anywhere;
    }
    .simple-syrup-dialog-backdrop {
      position: fixed;
      inset: 0;
      z-index: 2147483647;
      display: flex;
      align-items: center;
      justify-content: center;
      background: rgb(0 0 0 / 45%);
    }
    .simple-syrup-dialog {
      display: grid;
      gap: 0.75rem;
      min-width: min(28rem, calc(100vw - 2rem));
      padding: 1rem;
      background: var(--comfy-menu-bg);
      color: var(--fg-color);
    }
    .simple-syrup-dialog-title {
      margin: 0;
      font-size: 1rem;
    }
    .simple-syrup-dialog-actions {
      display: flex;
      justify-content: flex-end;
      gap: 0.5rem;
    }
  `;
  document.head.appendChild(style);
}
function createExternalLLMEndpointControl(context) {
  const wrapper = createElement("div", "simple-syrup-settings-row");
  const input = createElement("input", "simple-syrup-settings-input");
  input.type = "text";
  input.value = context.getSettings().base_url;
  input.autocomplete = "off";
  const saveButton = createElement("button", "simple-syrup-settings-button");
  saveButton.type = "button";
  saveButton.textContent = "Save Endpoint";
  const status = createElement("span", "simple-syrup-settings-status");
  const saveEndpoint = async () => {
    const value = input.value.trim();
    if (!endpointShouldBeSaved(value)) {
      status.textContent = "Enter an http:// or https:// endpoint.";
      return;
    }
    setPending(wrapper, true);
    try {
      const saved = await context.api.saveExternalLLMSettings({
        base_url: value,
        default_model: context.getSettings().default_model
      });
      context.setSettings(saved);
      input.value = saved.base_url;
      await context.refreshModelChoices();
      status.textContent = "Endpoint saved.";
    } catch (error) {
      context.logger.warn(
        "Could not save SimpleSyrup external LLM endpoint settings. The backend rejected the setting update.",
        error
      );
      status.textContent = errorMessage(error, "Endpoint was not saved.");
    } finally {
      setPending(wrapper, false);
    }
  };
  saveButton.addEventListener("click", () => {
    void saveEndpoint();
  });
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      void saveEndpoint();
    }
  });
  wrapper.append(input, saveButton, status);
  return wrapper;
}
function createExternalLLMApiKeyControl(context) {
  const wrapper = createElement("div", "simple-syrup-settings-row");
  const button = createElement("button", "simple-syrup-settings-button");
  button.type = "button";
  const status = createElement("span", "simple-syrup-settings-status");
  const render = () => {
    const remembered = context.getSettings().has_api_key;
    button.textContent = remembered ? "Replace API Key" : "Add API Key";
    status.textContent = remembered ? "API key remembered." : "";
  };
  button.addEventListener("click", () => {
    if (!context.getSettings().base_url.trim()) {
      status.textContent = "Save endpoint first.";
      return;
    }
    openExternalLLMApiKeyDialog({
      replacing: context.getSettings().has_api_key,
      onSubmit: async (apiKey) => {
        setPending(wrapper, true);
        try {
          const saved = await context.api.saveExternalLLMApiKey({
            api_key: apiKey
          });
          context.setSettings(saved);
          await context.refreshModelChoices();
          render();
          status.textContent = context.getSettings().has_api_key ? "API key remembered." : "API key was not saved.";
        } catch (error) {
          context.logger.warn(
            "Could not save SimpleSyrup external LLM API key. The backend rejected the credential update.",
            error
          );
          status.textContent = apiKeyErrorMessage(error);
        } finally {
          setPending(wrapper, false);
        }
      }
    });
  });
  wrapper.append(button, status);
  render();
  return wrapper;
}
function openExternalLLMApiKeyDialog(options) {
  const overlay = createElement("div", "simple-syrup-dialog-backdrop");
  const dialog = createElement("div", "simple-syrup-dialog comfy-dialog");
  const title = createElement("h3", "simple-syrup-dialog-title");
  title.textContent = options.replacing ? "Replace API Key" : "Add API Key";
  const input = createElement("input", "simple-syrup-settings-input");
  input.type = "password";
  input.autocomplete = "off";
  input.placeholder = "API key";
  input.setAttribute("data-1p-ignore", "true");
  input.setAttribute("data-lpignore", "true");
  input.setAttribute("data-bwignore", "true");
  const actions = createElement("div", "simple-syrup-dialog-actions");
  const submitButton = createElement("button", "simple-syrup-settings-button");
  submitButton.type = "button";
  submitButton.textContent = options.replacing ? "Replace Key" : "Store Key";
  const cancelButton = createElement("button", "simple-syrup-settings-button");
  cancelButton.type = "button";
  cancelButton.textContent = "Cancel";
  const close = () => {
    overlay.remove();
  };
  const submit = async () => {
    const apiKey = input.value.trim();
    if (!apiKey) {
      input.focus();
      return;
    }
    submitButton.disabled = true;
    await options.onSubmit(apiKey);
    close();
  };
  submitButton.addEventListener("click", () => {
    void submit();
  });
  cancelButton.addEventListener("click", close);
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      void submit();
    }
    if (event.key === "Escape") {
      event.preventDefault();
      close();
    }
  });
  actions.append(submitButton, cancelButton);
  dialog.append(title, input, actions);
  overlay.append(dialog);
  document.body.appendChild(overlay);
  input.focus();
}
function createElement(tagName, className) {
  const element = document.createElement(tagName);
  element.className = className;
  return element;
}
function setPending(element, pending) {
  element.dataset.pending = pending ? "true" : "false";
  for (const control of Array.from(element.querySelectorAll("input, button"))) {
    if (control instanceof HTMLInputElement || control instanceof HTMLButtonElement) {
      control.disabled = pending;
    }
  }
}
function errorMessage(error, fallback) {
  if (error instanceof Error && error.message) {
    return error.message;
  }
  return fallback;
}
function apiKeyErrorMessage(error) {
  const message = errorMessage(error, "API key was not saved.");
  if (message.includes("Configure an external LLM endpoint")) {
    return "Save endpoint first.";
  }
  return message;
}
async function refreshExternalLLMModelChoices(app2, logger) {
  try {
    await app2.refreshComboInNodes?.();
  } catch (error) {
    logger.warn(
      "Could not refresh Comfy node definitions after saving external LLM settings.",
      error
    );
  }
}

// web/src/refresh.ts
var REFRESH_WRAPPED = /* @__PURE__ */ Symbol.for("SimpleSyrup.ExternalLLM.RefreshWrapped");
function registerExternalLLMRefreshHook(app2, api = { refreshExternalLLMModels }, logger = console) {
  if (!app2.refreshComboInNodes) {
    return;
  }
  const refreshOwner = app2;
  if (refreshOwner[REFRESH_WRAPPED]) {
    return;
  }
  const originalRefresh = app2.refreshComboInNodes.bind(app2);
  refreshOwner[REFRESH_WRAPPED] = true;
  app2.refreshComboInNodes = async () => {
    try {
      await api.refreshExternalLLMModels();
    } catch (error) {
      logger.warn("Could not refresh SimpleSyrup external LLM models.", error);
    }
    await originalRefresh();
  };
}

// web/src/maskBatchPreview.ts
var EMPTY_PREVIEW = {
  images: [],
  animated: []
};
var PREVIEW_PROMPT_ID = "simple-syrup-mask-batch-preview";
var MaskBatchPreviewController = class {
  constructor(executionEvents, node, loadPreview = getMaskBatchPreview, logger = console, clearNativePreview2 = () => void 0) {
    this.executionEvents = executionEvents;
    this.node = node;
    this.loadPreview = loadPreview;
    this.logger = logger;
    this.clearNativePreview = clearNativePreview2;
  }
  executionEvents;
  node;
  loadPreview;
  logger;
  clearNativePreview;
  requestVersion = 0;
  /** Replace the visible preview with the exact selected-channel mask output. */
  refresh(files, channel) {
    if (this.node.id === void 0 || channel === void 0) return;
    const requestVersion = ++this.requestVersion;
    this.clearNativePreview();
    this.publish(EMPTY_PREVIEW);
    if (files.length === 0) return;
    void this.loadPreview([...files], channel).then((output) => {
      if (requestVersion === this.requestVersion) this.publish(output);
    }).catch((error) => {
      if (requestVersion !== this.requestVersion) return;
      this.logger.warn("Could not refresh Load Mask Batch preview.", error);
    });
  }
  /** Publish an execution-shaped result through Comfy's native preview. */
  publish(output) {
    if (this.node.id === void 0) return;
    this.executionEvents.dispatchEvent(
      new CustomEvent("executed", {
        detail: {
          node: String(this.node.id),
          output,
          merge: false,
          prompt_id: PREVIEW_PROMPT_ID
        }
      })
    );
    this.node.graph?.setDirtyCanvas?.(true, true);
  }
};

// web/src/maskBatchUpload.ts
var LOAD_MASK_BATCH_NODE_ID = "SimpleSyrup.LoadMaskBatch";
var REPLACE_MASKS_LABEL = "Replace masks...";
var ADD_MASKS_LABEL = "Add masks...";
var REMOVE_MASK_LABEL = "Remove selected mask";
var EMPTY_MASK_SELECTION_LABEL = "No masks loaded";
function registerMaskBatchUpload(app2, executionEvents, loadPreview, logger = console) {
  const extension = {
    name: "SimpleSyrup.LoadMaskBatchUpload",
    nodeCreated(node) {
      try {
        configureMaskBatchNode(node, executionEvents, loadPreview, logger, app2);
      } catch (error) {
        logger.warn(
          `Could not configure Load Mask Batch native controls: ${errorMessage2(error)}`,
          error
        );
        throw error;
      }
    }
  };
  app2.registerExtension(extension);
}
function errorMessage2(error) {
  return error instanceof Error && error.message ? error.message : String(error);
}
function configureMaskBatchNode(candidate, executionEvents, loadPreview, logger = console, app2) {
  if (!isMaskBatchNode(candidate)) return;
  const imageWidget = findWidget(candidate, "image");
  const channelWidget = findWidget(candidate, "channel");
  const uploadWidget = findNativeUploadWidget(candidate);
  if (!imageWidget || !channelWidget || !uploadWidget?.callback) return;
  hideInternalWidget(imageWidget);
  hideInternalWidget(uploadWidget);
  const preview = new MaskBatchPreviewController(
    executionEvents,
    candidate,
    loadPreview,
    logger,
    () => {
      clearNativePreview(candidate, app2);
    }
  );
  let selectionIntent = "replace";
  let appendBase = [];
  let uploadPending = false;
  let programmaticSelectionUpdate = false;
  let ignoredUploadCallbackFiles;
  const nativeUploadCallback = uploadWidget.callback;
  const replaceWidget = candidate.addWidget(
    "button",
    "simple_syrup_replace_masks",
    "image",
    () => {
      selectionIntent = "replace";
      appendBase = [];
      uploadPending = true;
      nativeUploadCallback.call(uploadWidget);
    },
    nativeButtonOptions(
      "Choose one or more masks and replace the current ordered list."
    )
  );
  replaceWidget.label = REPLACE_MASKS_LABEL;
  const setSelectedMasks = (files) => {
    programmaticSelectionUpdate = true;
    try {
      imageWidget.value = [...files];
    } finally {
      programmaticSelectionUpdate = false;
    }
  };
  uploadWidget.callback = (value) => {
    selectionIntent = "replace";
    appendBase = [];
    uploadPending = true;
    nativeUploadCallback.call(uploadWidget, value);
  };
  const addWidget = candidate.addWidget(
    "button",
    "simple_syrup_add_masks",
    "image",
    () => {
      selectionIntent = "append";
      appendBase = selectedMaskFiles(imageWidget);
      uploadPending = true;
      nativeUploadCallback.call(uploadWidget);
    },
    nativeButtonOptions(
      "Upload one or more masks and append them after the current ordered list."
    )
  );
  addWidget.label = ADD_MASKS_LABEL;
  const selectedMaskWidget = candidate.addWidget(
    "combo",
    "simple_syrup_selected_mask",
    EMPTY_MASK_SELECTION_LABEL,
    (value) => {
      const selectedIndex = selectedMaskIndex(selectedMaskWidget, value);
      candidate.imageIndex = selectedIndex;
      candidate.graph?.setDirtyCanvas?.(true, true);
    },
    {
      serialize: false,
      tooltip: "Select one loaded mask by its ordered position for preview or removal.",
      values: []
    }
  );
  selectedMaskWidget.label = "selected mask";
  const removeWidget = candidate.addWidget(
    "button",
    "simple_syrup_remove_mask",
    "image",
    () => {
      const previous = selectedMaskFiles(imageWidget);
      const index = activeMaskIndex(candidate, selectedMaskWidget, previous);
      if (index === null) return;
      const current = previous.toSpliced(index, 1);
      setSelectedMasks(current);
      updateMaskSelection(selectedMaskWidget, current, index);
      candidate.imageIndex = current.length === 0 ? null : Math.min(index, current.length - 1);
      notifySelectionChanged(candidate, imageWidget, { current, previous });
      updateRemoveAvailability(removeWidget, current.length);
      preview.refresh(current, selectedChannel(channelWidget));
    },
    nativeButtonOptions(
      "Open a mask in the preview gallery, then remove that position from the loaded list."
    )
  );
  removeWidget.label = REMOVE_MASK_LABEL;
  imageWidget.callback = (value) => {
    if (programmaticSelectionUpdate) return;
    const callbackValue = value ?? imageWidget.value;
    if (uploadPending && !Array.isArray(callbackValue)) return;
    const incoming = normalizeMaskFiles(callbackValue);
    if (ignoredUploadCallbackFiles && sameMaskFiles(incoming, ignoredUploadCallbackFiles)) {
      ignoredUploadCallbackFiles = void 0;
      return;
    }
    ignoredUploadCallbackFiles = uploadPending ? [...incoming] : void 0;
    uploadPending = false;
    const current = selectionIntent === "append" ? [...appendBase, ...incoming] : incoming;
    const appended = selectionIntent === "append";
    selectionIntent = "replace";
    appendBase = [];
    if (appended) setSelectedMasks(current);
    candidate.imageIndex = null;
    updateMaskSelection(selectedMaskWidget, current);
    updateRemoveAvailability(removeWidget, current.length);
    preview.refresh(current, selectedChannel(channelWidget));
  };
  const originalChannelCallback = channelWidget.callback;
  channelWidget.callback = (value) => {
    originalChannelCallback?.call(channelWidget, value);
    const files = selectedMaskFiles(imageWidget);
    if (files.length > 0) {
      preview.refresh(files, selectedChannel(channelWidget, value));
    }
  };
  resetIntentForExternalUploads(candidate, () => {
    selectionIntent = "replace";
    appendBase = [];
    uploadPending = false;
  });
  const originalOnGraphConfigured = candidate.onGraphConfigured;
  candidate.onGraphConfigured = function(...args) {
    const result = originalOnGraphConfigured?.apply(this, args);
    const restored = selectedMaskFiles(imageWidget);
    if (!Array.isArray(imageWidget.value)) setSelectedMasks(restored);
    updateMaskSelection(selectedMaskWidget, restored);
    updateRemoveAvailability(removeWidget, restored.length);
    if (restored.length > 0) {
      preview.refresh(restored, selectedChannel(channelWidget));
    }
    return result;
  };
  const initial = selectedMaskFiles(imageWidget);
  if (!Array.isArray(imageWidget.value)) setSelectedMasks(initial);
  updateMaskSelection(selectedMaskWidget, initial);
  updateRemoveAvailability(removeWidget, initial.length);
  if (initial.length > 0) {
    preview.refresh(initial, selectedChannel(channelWidget));
  }
}
function hideInternalWidget(widget) {
  widget.options ??= {};
  widget.options.hidden = true;
  widget.hidden = true;
  widget.computeSize = () => [0, -4];
}
function updateMaskSelection(widget, files, preferredIndex = 0) {
  const values = files.map(
    (file, index) => `${String(index + 1)}. ${file}`
  );
  const hasMasks = values.length > 0;
  widget.options ??= {};
  widget.options.values = hasMasks ? values : [EMPTY_MASK_SELECTION_LABEL];
  widget.disabled = !hasMasks;
  widget.options.disabled = !hasMasks;
  widget.value = hasMasks ? values[Math.min(Math.max(preferredIndex, 0), values.length - 1)] : EMPTY_MASK_SELECTION_LABEL;
}
function selectedMaskIndex(widget, callbackValue) {
  if (widget.disabled) return null;
  const value = callbackValue ?? widget.value;
  const values = widget.options?.values ?? [];
  const index = typeof value === "string" ? values.indexOf(value) : -1;
  return index >= 0 ? index : null;
}
function clearNativePreview(node, app2) {
  node.imgs = void 0;
  node.images = [];
  if (node.id !== void 0) delete app2?.nodeOutputs?.[String(node.id)];
  node.graph?.setDirtyCanvas?.(true, true);
}
function activeMaskIndex(node, selectedMaskWidget, files) {
  const galleryIndex = node.imageIndex;
  if (galleryIndex !== null && galleryIndex !== void 0 && galleryIndex >= 0 && galleryIndex < files.length) {
    return galleryIndex;
  }
  const selectedIndex = selectedMaskIndex(selectedMaskWidget);
  return selectedIndex !== null && selectedIndex < files.length ? selectedIndex : null;
}
function findWidget(node, name) {
  return node.widgets?.find((widget) => widget.name === name);
}
function findNativeUploadWidget(node) {
  return node.widgets?.find(
    (widget) => widget.type === "button" && widget.value === "image" && widget.options?.serialize === false && widget.options.canvasOnly === true
  );
}
function nativeButtonOptions(tooltip) {
  return { serialize: false, tooltip };
}
function selectedChannel(channelWidget, callbackValue) {
  const value = callbackValue ?? channelWidget.value;
  return typeof value === "string" && value.length > 0 ? value : void 0;
}
function selectedMaskFiles(widget) {
  return normalizeMaskFiles(widget.value);
}
function normalizeMaskFiles(value) {
  const values = Array.isArray(value) ? value : [value];
  return values.filter(
    (item) => typeof item === "string" && item.length > 0
  );
}
function sameMaskFiles(left, right) {
  return left.length === right.length && left.every((file, index) => file === right[index]);
}
function updateRemoveAvailability(widget, count) {
  const disabled = count === 0;
  widget.disabled = disabled;
  widget.options ??= {};
  widget.options.disabled = disabled;
}
function notifySelectionChanged(node, imageWidget, change) {
  node.onWidgetChanged?.(
    imageWidget.name,
    [...change.current],
    [...change.previous],
    imageWidget
  );
  node.graph?.setDirtyCanvas?.(true, true);
}
function resetIntentForExternalUploads(node, reset) {
  const originalPasteFiles = node.pasteFiles;
  const originalOnDragDrop = node.onDragDrop;
  const originalOnRemoved = node.onRemoved;
  const wrappedPasteFiles = originalPasteFiles ? (...args) => {
    reset();
    return originalPasteFiles.apply(node, args);
  } : void 0;
  const wrappedOnDragDrop = originalOnDragDrop ? (...args) => {
    reset();
    return originalOnDragDrop.apply(node, args);
  } : void 0;
  if (wrappedPasteFiles) node.pasteFiles = wrappedPasteFiles;
  if (wrappedOnDragDrop) node.onDragDrop = wrappedOnDragDrop;
  node.onRemoved = function(...args) {
    if (node.pasteFiles === wrappedPasteFiles) {
      if (originalPasteFiles) node.pasteFiles = originalPasteFiles;
      else delete node.pasteFiles;
    }
    if (node.onDragDrop === wrappedOnDragDrop) {
      if (originalOnDragDrop) node.onDragDrop = originalOnDragDrop;
      else delete node.onDragDrop;
    }
    return originalOnRemoved?.apply(this, args);
  };
}
function isMaskBatchNode(candidate) {
  if (typeof candidate !== "object" || candidate === null) return false;
  const node = candidate;
  return node.constructor?.comfyClass === LOAD_MASK_BATCH_NODE_ID && typeof node.addWidget === "function";
}

// web/src/interactiveInspector.ts
var InteractiveInspectorController = class {
  constructor(view, parse, logger = console) {
    this.view = view;
    this.parse = parse;
    this.logger = logger;
  }
  view;
  parse;
  logger;
  version = 0;
  disposed = false;
  committed;
  /** Parse and asynchronously display one execution output. */
  update(output) {
    if (this.disposed) return;
    const version = ++this.version;
    let document2;
    try {
      document2 = this.parse(output);
    } catch (error) {
      this.fail(error);
      return;
    }
    if (!document2) return;
    this.view.setLoading();
    void this.view.prepare(document2).then((prepared) => {
      if (this.disposed || version !== this.version) {
        prepared.dispose();
        return;
      }
      this.committed?.dispose();
      this.committed = prepared;
      prepared.commit();
    }).catch((error) => {
      if (!this.disposed && version === this.version) this.fail(error);
    });
  }
  /** Release committed state and prevent pending work from publishing. */
  dispose() {
    if (this.disposed) return;
    this.disposed = true;
    this.version += 1;
    this.committed?.dispose();
    this.committed = void 0;
    this.view.dispose();
  }
  fail(error) {
    const message = error instanceof Error ? error.message : String(error);
    this.logger.warn(`Could not display Simple Preview SEGS: ${message}`, error);
    this.view.showError(message);
  }
};

// web/src/imageViewport.ts
var ImageViewport = class {
  constructor(source, display) {
    this.source = source;
    this.display = display;
  }
  source;
  display;
  /** Convert browser pointer coordinates into source-image coordinates. */
  sourcePoint(clientX, clientY, bounds) {
    const normalizedX = clamp((clientX - bounds.left) / Math.max(1, bounds.width));
    const normalizedY = clamp((clientY - bounds.top) / Math.max(1, bounds.height));
    return {
      x: normalizedX * this.source.width,
      y: normalizedY * this.source.height
    };
  }
  /** Project one source rectangle into the preview canvas. */
  displayRectangle(sourceRectangle) {
    const scaleX = this.display.width / this.source.width;
    const scaleY = this.display.height / this.source.height;
    return {
      x: sourceRectangle.x * scaleX,
      y: sourceRectangle.y * scaleY,
      width: sourceRectangle.width * scaleX,
      height: sourceRectangle.height * scaleY
    };
  }
};
function clamp(value) {
  return Math.max(0, Math.min(1, value));
}

// web/src/maskAtlas.ts
var MaskAtlas = class {
  constructor(document2, pixels) {
    this.document = document2;
    this.pixels = pixels;
    const expected = document2.atlas.width * document2.atlas.height * 4;
    if (pixels.length !== expected) {
      throw new Error("Simple Preview SEGS atlas pixels do not match its dimensions.");
    }
  }
  document;
  pixels;
  /** Return every mask containing a source point, most specific first. */
  hitsAt(sourceX, sourceY) {
    return this.document.regions.filter((region) => this.contains(region, sourceX, sourceY)).sort((left, right) => left.area - right.area || left.index - right.index);
  }
  /** Return one colored RGBA image for a packed region mask. */
  coloredMask(region) {
    const color = parseColor(region.color);
    const data = new Uint8ClampedArray(region.atlas.width * region.atlas.height * 4);
    for (let y = 0; y < region.atlas.height; y += 1) {
      for (let x = 0; x < region.atlas.width; x += 1) {
        const atlasOffset = this.offset(region.atlas.x + x, region.atlas.y + y);
        const mask = this.pixels[atlasOffset] ?? 0;
        const outputOffset = (y * region.atlas.width + x) * 4;
        data[outputOffset] = color[0];
        data[outputOffset + 1] = color[1];
        data[outputOffset + 2] = color[2];
        data[outputOffset + 3] = mask;
      }
    }
    return new ImageData(data, region.atlas.width, region.atlas.height);
  }
  contains(region, sourceX, sourceY) {
    const crop = region.crop;
    if (sourceX < crop.x || sourceY < crop.y || sourceX >= crop.x + crop.width || sourceY >= crop.y + crop.height) {
      return false;
    }
    const localX = Math.min(
      region.atlas.width - 1,
      Math.floor((sourceX - crop.x) / crop.width * region.atlas.width)
    );
    const localY = Math.min(
      region.atlas.height - 1,
      Math.floor((sourceY - crop.y) / crop.height * region.atlas.height)
    );
    return (this.pixels[this.offset(region.atlas.x + localX, region.atlas.y + localY)] ?? 0) >= 128;
  }
  offset(x, y) {
    return (y * this.document.atlas.width + x) * 4;
  }
};
function maskAtlasFromImage(previewDocument, image) {
  const canvas = globalThis.document.createElement("canvas");
  canvas.width = previewDocument.atlas.width;
  canvas.height = previewDocument.atlas.height;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) throw new Error("Simple Preview SEGS could not read its mask atlas.");
  context.drawImage(image, 0, 0, canvas.width, canvas.height);
  return new MaskAtlas(
    previewDocument,
    context.getImageData(0, 0, canvas.width, canvas.height).data
  );
}
function parseColor(value) {
  return [
    Number.parseInt(value.slice(1, 3), 16),
    Number.parseInt(value.slice(3, 5), 16),
    Number.parseInt(value.slice(5, 7), 16)
  ];
}

// web/src/segPreviewTypes.ts
var SEG_PREVIEW_OUTPUT_KEY = "simple_syrup_segs_preview";
function parseSegPreviewDocument(output) {
  if (!isRecord(output)) return void 0;
  const candidates = output[SEG_PREVIEW_OUTPUT_KEY];
  if (!Array.isArray(candidates) || candidates.length === 0) return void 0;
  const candidate = candidates[candidates.length - 1];
  if (!isRecord(candidate) || candidate.version !== 1) {
    throw new Error("Simple Preview SEGS received an unsupported preview payload.");
  }
  const source = parseDimensions(candidate.source, "source");
  const preview = parseAssetDimensions(candidate.preview, "preview");
  const atlas = parseAssetDimensions(candidate.atlas, "atlas");
  if (!Array.isArray(candidate.regions)) {
    throw new Error("Simple Preview SEGS regions must be a list.");
  }
  const regions = candidate.regions.map(parseRegion);
  return { version: 1, source, preview, atlas, regions };
}
function previewAssetUrl(reference, apiURL = (path) => path) {
  const query = new URLSearchParams({
    filename: reference.filename,
    subfolder: reference.subfolder,
    type: reference.type
  });
  return apiURL(`/view?${query.toString()}`);
}
function parseRegion(value, index) {
  if (!isRecord(value)) {
    throw new Error(
      `Simple Preview SEGS region ${String(index + 1)} must be an object.`
    );
  }
  const id = stringValue(value.id, "region id");
  const label = stringValue(value.label, "region label");
  const color = stringValue(value.color, "region color");
  if (!/^#[0-9a-f]{6}$/i.test(color)) {
    throw new Error(`Simple Preview SEGS region '${id}' has an invalid color.`);
  }
  return {
    id,
    index: integerValue(value.index, "region index", 0),
    label,
    confidence: numberValue(value.confidence, "region confidence", 0),
    area: integerValue(value.area, "region area", 0),
    color,
    crop: parseRectangle(value.crop, `region '${id}' crop`),
    atlas: parseRectangle(value.atlas, `region '${id}' atlas`)
  };
}
function parseAssetDimensions(value, name) {
  if (!isRecord(value)) {
    throw new Error(`Simple Preview SEGS ${name} must be an object.`);
  }
  return {
    ...parseDimensions(value, name),
    image: parseImageReference(value.image, name)
  };
}
function parseDimensions(value, name) {
  if (!isRecord(value)) {
    throw new Error(`Simple Preview SEGS ${name} dimensions must be an object.`);
  }
  return {
    width: integerValue(value.width, `${name} width`, 1),
    height: integerValue(value.height, `${name} height`, 1)
  };
}
function parseRectangle(value, name) {
  if (!isRecord(value)) {
    throw new Error(`Simple Preview SEGS ${name} must be an object.`);
  }
  return {
    x: integerValue(value.x, `${name} x`, 0),
    y: integerValue(value.y, `${name} y`, 0),
    width: integerValue(value.width, `${name} width`, 1),
    height: integerValue(value.height, `${name} height`, 1)
  };
}
function parseImageReference(value, name) {
  if (!isRecord(value)) {
    throw new Error(`Simple Preview SEGS ${name} image reference is invalid.`);
  }
  const type = stringValue(value.type, `${name} image type`);
  if (type !== "input" && type !== "output" && type !== "temp") {
    throw new Error(`Simple Preview SEGS ${name} image type is invalid.`);
  }
  return {
    filename: stringValue(value.filename, `${name} image filename`),
    subfolder: stringValue(value.subfolder, `${name} image subfolder`),
    type
  };
}
function integerValue(value, name, minimum) {
  if (!Number.isInteger(value) || value < minimum) {
    throw new Error(
      `Simple Preview SEGS ${name} must be at least ${String(minimum)}.`
    );
  }
  return value;
}
function numberValue(value, name, minimum) {
  if (typeof value !== "number" || !Number.isFinite(value) || value < minimum) {
    throw new Error(
      `Simple Preview SEGS ${name} must be at least ${String(minimum)}.`
    );
  }
  return value;
}
function stringValue(value, name) {
  if (typeof value !== "string") {
    throw new Error(`Simple Preview SEGS ${name} must be text.`);
  }
  return value;
}
function isRecord(value) {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

// web/src/selectionModel.ts
var SelectionModel = class {
  hovered = null;
  selected = null;
  candidates = [];
  subscribers = /* @__PURE__ */ new Set();
  /** Subscribe to selection changes and receive the current state immediately. */
  subscribe(subscriber) {
    this.subscribers.add(subscriber);
    subscriber(this.state());
    return () => this.subscribers.delete(subscriber);
  }
  /** Replace the hover target and all regions currently beneath the pointer. */
  hover(candidates) {
    this.candidates = [...candidates];
    this.hovered = candidates[0] ?? null;
    this.publish();
  }
  /** Clear transient pointer state without changing the pinned region. */
  clearHover() {
    this.candidates = [];
    this.hovered = null;
    this.publish();
  }
  /** Pin one region, or clear the pinned selection with null. */
  select(id) {
    this.selected = id;
    this.publish();
  }
  /** Cycle through overlapping pointer candidates and pin the result. */
  selectNextCandidate() {
    if (this.candidates.length === 0) return this.selected;
    const currentIndex = this.selected === null ? -1 : this.candidates.indexOf(this.selected);
    const next = this.candidates[(currentIndex + 1) % this.candidates.length] ?? null;
    this.select(next);
    return next;
  }
  /** Return immutable selection state for rendering or tests. */
  state() {
    return {
      hovered: this.hovered,
      selected: this.selected,
      candidates: [...this.candidates],
      active: this.hovered ?? this.selected
    };
  }
  publish() {
    const state = this.state();
    for (const subscriber of this.subscribers) subscriber(state);
  }
};

// web/src/segPreviewInspector.ts
var SegPreviewInspector = class {
  constructor(apiURL = (path) => path, loadImage = loadPreviewImage) {
    this.apiURL = apiURL;
    this.loadImage = loadImage;
    installStyles();
    this.element.className = "ss-segs-preview";
    this.body.className = "ss-segs-preview__body";
    this.status.className = "ss-segs-preview__status";
    this.status.setAttribute("aria-live", "polite");
    const toolbar = document.createElement("nav");
    toolbar.className = "ss-segs-preview__toolbar";
    toolbar.setAttribute("aria-label", "SEGS preview mode");
    toolbar.append(this.overlayButton, this.gridButton);
    this.overlayButton.addEventListener("click", () => {
      this.setMode("overlay");
    });
    this.gridButton.addEventListener("click", () => {
      this.setMode("grid");
    });
    this.element.append(toolbar, this.body, this.status);
    this.updateModeButtons();
    this.showMessage("Run the workflow to inspect SEGS.");
  }
  apiURL;
  loadImage;
  element = document.createElement("section");
  body = document.createElement("div");
  status = document.createElement("div");
  overlayButton = modeButton("Overlay");
  gridButton = modeButton("Grid");
  mode = "overlay";
  committed;
  unsubscribeSelection;
  highlightCanvas;
  focusContainer;
  gridButtons = /* @__PURE__ */ new Map();
  /** Show a native loading state while execution assets are decoded. */
  setLoading() {
    this.showMessage("Loading SEGS preview\u2026");
  }
  /** Load source and atlas assets without publishing partial state. */
  async prepare(document2) {
    const [image, atlasImage] = await Promise.all([
      this.loadImage(previewAssetUrl(document2.preview.image, this.apiURL)),
      this.loadImage(previewAssetUrl(document2.atlas.image, this.apiURL))
    ]);
    const atlas = maskAtlasFromImage(document2, atlasImage);
    let disposed = false;
    return {
      commit: () => {
        if (!disposed) this.commit(document2, image, atlas);
      },
      dispose: () => {
        disposed = true;
      }
    };
  }
  /** Replace the inspector with an actionable failure message. */
  showError(message) {
    this.showMessage(message, true);
  }
  /** Release node-owned DOM and interaction subscriptions. */
  dispose() {
    this.unsubscribeSelection?.();
    this.unsubscribeSelection = void 0;
    this.committed = void 0;
    this.body.replaceChildren();
    this.status.textContent = "";
  }
  commit(document2, image, atlas) {
    this.unsubscribeSelection?.();
    const selection = new SelectionModel();
    const masks = new Map(
      document2.regions.map((region) => [
        region.id,
        maskCanvas(atlas, region)
      ])
    );
    this.committed = {
      document: document2,
      image,
      atlas,
      viewport: new ImageViewport(document2.source, document2.preview),
      masks,
      selection
    };
    this.unsubscribeSelection = selection.subscribe((state) => {
      this.renderSelection(state);
    });
    this.renderMode();
  }
  setMode(mode) {
    if (this.mode === mode) return;
    this.mode = mode;
    this.updateModeButtons();
    this.renderMode();
  }
  updateModeButtons() {
    this.overlayButton.setAttribute(
      "aria-pressed",
      String(this.mode === "overlay")
    );
    this.gridButton.setAttribute("aria-pressed", String(this.mode === "grid"));
  }
  renderMode() {
    this.highlightCanvas = void 0;
    this.focusContainer = void 0;
    this.gridButtons.clear();
    if (!this.committed) return;
    this.body.replaceChildren(
      this.mode === "overlay" ? this.overlayView() : this.gridView()
    );
    this.renderSelection(this.committed.selection.state());
  }
  overlayView() {
    const committed = required(this.committed);
    const container = document.createElement("div");
    container.className = "ss-segs-preview__overlay-view";
    const stack = document.createElement("div");
    stack.className = "ss-segs-preview__canvas-stack";
    const base = sizedCanvas(committed.document.preview);
    const highlight = sizedCanvas(committed.document.preview);
    highlight.className = "ss-segs-preview__highlight";
    this.highlightCanvas = highlight;
    const context = context2d(base);
    context.drawImage(
      committed.image,
      0,
      0,
      committed.document.preview.width,
      committed.document.preview.height
    );
    context.globalAlpha = 0.38;
    for (const region of committed.document.regions) {
      drawRegionMask(context, committed, region);
    }
    context.globalAlpha = 1;
    highlight.addEventListener("pointermove", (event) => {
      const point = committed.viewport.sourcePoint(
        event.clientX,
        event.clientY,
        highlight.getBoundingClientRect()
      );
      const hits = committed.atlas.hitsAt(point.x, point.y);
      committed.selection.hover(hits.map((region) => region.id));
    });
    highlight.addEventListener("pointerleave", () => {
      committed.selection.clearHover();
    });
    highlight.addEventListener("click", () => {
      committed.selection.selectNextCandidate();
    });
    stack.append(base, highlight);
    this.focusContainer = document.createElement("div");
    this.focusContainer.className = "ss-segs-preview__focus";
    container.append(stack, this.focusContainer);
    return container;
  }
  gridView() {
    const committed = required(this.committed);
    const grid = document.createElement("div");
    grid.className = "ss-segs-preview__grid";
    grid.setAttribute("role", "listbox");
    if (committed.document.regions.length === 0) {
      const empty = document.createElement("div");
      empty.className = "ss-segs-preview__empty";
      empty.textContent = "No SEGS to display.";
      grid.append(empty);
      return grid;
    }
    for (const region of committed.document.regions) {
      const card = document.createElement("button");
      card.type = "button";
      card.className = "ss-segs-preview__card";
      card.setAttribute("role", "option");
      card.dataset.regionId = region.id;
      const canvas = regionPreviewCanvas(committed, region, 160, 132);
      const label = document.createElement("span");
      label.textContent = regionTitle(region);
      card.append(canvas, label);
      card.addEventListener("pointerenter", () => {
        committed.selection.hover([region.id]);
      });
      card.addEventListener("pointerleave", () => {
        committed.selection.clearHover();
      });
      card.addEventListener("click", () => {
        committed.selection.select(region.id);
      });
      this.gridButtons.set(region.id, card);
      grid.append(card);
    }
    return grid;
  }
  renderSelection(state) {
    const committed = this.committed;
    if (!committed) return;
    const active = state.active ? committed.document.regions.find((region) => region.id === state.active) : void 0;
    if (this.highlightCanvas) {
      const context = context2d(this.highlightCanvas);
      context.clearRect(
        0,
        0,
        this.highlightCanvas.width,
        this.highlightCanvas.height
      );
      if (active) {
        context.save();
        context.globalAlpha = 0.9;
        context.shadowColor = "rgba(255, 255, 255, 0.95)";
        context.shadowBlur = 8;
        drawRegionMask(context, committed, active);
        context.restore();
      }
    }
    for (const [id, button] of this.gridButtons) {
      const selected = id === state.selected;
      const highlighted = id === active?.id;
      button.setAttribute("aria-selected", String(selected));
      button.dataset.highlighted = String(highlighted);
    }
    this.renderFocus(active);
    if (!active) {
      this.status.textContent = `${String(committed.document.regions.length)} regions`;
    } else if (state.candidates.length > 1) {
      this.status.textContent = `${regionTitle(active)} \xB7 ${String(state.candidates.length)} overlapping regions \xB7 click to cycle`;
    } else {
      this.status.textContent = regionTitle(active);
    }
  }
  renderFocus(region) {
    if (!this.focusContainer || !this.committed) return;
    if (!region) {
      this.focusContainer.replaceChildren();
      return;
    }
    const title = document.createElement("strong");
    title.textContent = regionTitle(region);
    const details = document.createElement("span");
    details.textContent = `${String(Math.round(region.confidence * 100))}% confidence \xB7 ${region.area.toLocaleString()} px`;
    this.focusContainer.replaceChildren(
      regionPreviewCanvas(this.committed, region, 320, 220),
      title,
      details
    );
  }
  showMessage(message, error = false) {
    const content = document.createElement("div");
    content.className = "ss-segs-preview__message";
    content.dataset.error = String(error);
    content.textContent = message;
    this.body.replaceChildren(content);
    this.status.textContent = "";
  }
};
function drawRegionMask(context, committed, region) {
  const placement = committed.viewport.displayRectangle(region.crop);
  const mask = committed.masks.get(region.id);
  if (!mask) return;
  context.drawImage(
    mask,
    placement.x,
    placement.y,
    placement.width,
    placement.height
  );
}
function regionPreviewCanvas(committed, region, maximumWidth, maximumHeight) {
  const aspect = region.crop.width / region.crop.height;
  const width = Math.max(80, Math.min(maximumWidth, Math.round(maximumHeight * aspect)));
  const height = Math.max(64, Math.min(maximumHeight, Math.round(width / aspect)));
  const canvas = sizedCanvas({ width, height });
  const context = context2d(canvas);
  const source = committed.viewport.displayRectangle(region.crop);
  context.drawImage(
    committed.image,
    source.x,
    source.y,
    source.width,
    source.height,
    0,
    0,
    width,
    height
  );
  context.globalAlpha = 0.45;
  const mask = committed.masks.get(region.id);
  if (mask) context.drawImage(mask, 0, 0, width, height);
  context.globalAlpha = 1;
  return canvas;
}
function maskCanvas(atlas, region) {
  const canvas = sizedCanvas(region.atlas);
  context2d(canvas).putImageData(atlas.coloredMask(region), 0, 0);
  return canvas;
}
function sizedCanvas(size) {
  const canvas = document.createElement("canvas");
  canvas.width = size.width;
  canvas.height = size.height;
  return canvas;
}
function context2d(canvas) {
  const context = canvas.getContext("2d");
  if (!context) throw new Error("Simple Preview SEGS requires canvas rendering.");
  return context;
}
function regionTitle(region) {
  const label = region.label.trim();
  return `${String(region.index + 1)}. ${label || "region"}`;
}
function modeButton(label) {
  const button = document.createElement("button");
  button.type = "button";
  button.textContent = label;
  return button;
}
function required(value) {
  if (!value) throw new Error("Simple Preview SEGS has no committed document.");
  return value;
}
function loadPreviewImage(url) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.addEventListener("load", () => {
      resolve(image);
    }, { once: true });
    image.addEventListener(
      "error",
      () => {
        reject(new Error("Simple Preview SEGS could not load a preview asset."));
      },
      { once: true }
    );
    image.src = url;
  });
}
var stylesInstalled = false;
function installStyles() {
  if (stylesInstalled) return;
  const style = document.createElement("style");
  style.dataset.simpleSyrupSegsPreview = "true";
  style.textContent = `
    .ss-segs-preview { box-sizing: border-box; width: 100%; min-height: 360px; color: var(--fg-color, #ddd); font: 12px sans-serif; }
    .ss-segs-preview * { box-sizing: border-box; }
    .ss-segs-preview__toolbar { display: flex; gap: 4px; margin: 0 0 6px; }
    .ss-segs-preview__toolbar button { flex: 1; min-height: 26px; border: 1px solid var(--border-color, #555); border-radius: 5px; color: inherit; background: var(--comfy-input-bg, #222); cursor: pointer; }
    .ss-segs-preview__toolbar button[aria-pressed="true"] { border-color: var(--p-primary-color, #6aa9ff); background: color-mix(in srgb, var(--p-primary-color, #6aa9ff) 28%, var(--comfy-input-bg, #222)); }
    .ss-segs-preview__body { min-height: 320px; overflow: hidden; border: 1px solid var(--border-color, #444); border-radius: 6px; background: var(--comfy-menu-bg, #181818); }
    .ss-segs-preview__canvas-stack { position: relative; line-height: 0; background: #111; }
    .ss-segs-preview__canvas-stack canvas { display: block; width: 100%; height: auto; }
    .ss-segs-preview__highlight { position: absolute; inset: 0; cursor: crosshair; }
    .ss-segs-preview__status { min-height: 22px; padding: 5px 2px 0; color: var(--descrip-text, #aaa); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .ss-segs-preview__focus { display: grid; grid-template-columns: minmax(96px, 42%) 1fr; gap: 3px 9px; align-items: start; padding: 7px; border-top: 1px solid var(--border-color, #444); }
    .ss-segs-preview__focus canvas { grid-row: 1 / span 2; width: 100%; height: auto; border-radius: 4px; background: #111; }
    .ss-segs-preview__focus span { color: var(--descrip-text, #aaa); }
    .ss-segs-preview__grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(124px, 1fr)); gap: 6px; max-height: 430px; padding: 7px; overflow: auto; }
    .ss-segs-preview__card { min-width: 0; padding: 4px; border: 1px solid var(--border-color, #444); border-radius: 5px; color: inherit; background: var(--comfy-input-bg, #222); cursor: pointer; text-align: left; }
    .ss-segs-preview__card[data-highlighted="true"], .ss-segs-preview__card[aria-selected="true"] { border-color: var(--p-primary-color, #6aa9ff); box-shadow: 0 0 0 1px var(--p-primary-color, #6aa9ff); }
    .ss-segs-preview__card canvas { display: block; width: 100%; height: 94px; object-fit: contain; margin-bottom: 4px; background: #111; }
    .ss-segs-preview__card span { display: block; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .ss-segs-preview__message, .ss-segs-preview__empty { display: grid; min-height: 320px; place-items: center; padding: 18px; color: var(--descrip-text, #aaa); text-align: center; }
    .ss-segs-preview__message[data-error="true"] { color: var(--error-text, #ff8a80); }
  `;
  document.head.append(style);
  stylesInstalled = true;
}

// web/src/segPreviewNode.ts
var SIMPLE_PREVIEW_SEGS_NODE_ID = "SimpleSyrup.SimplePreviewSEGS";
function registerSimplePreviewSEGS(app2, api, loadImage, logger = console) {
  const controllers = /* @__PURE__ */ new Map();
  const extension = {
    name: "SimpleSyrup.SimplePreviewSEGS",
    nodeCreated(candidate) {
      if (!isPreviewNode(candidate)) return;
      const inspector = new SegPreviewInspector(
        (path) => api.apiURL?.(path) ?? path,
        loadImage
      );
      const controller = new InteractiveInspectorController(
        inspector,
        parseSegPreviewDocument,
        logger
      );
      const widget = candidate.addDOMWidget(
        "simple_syrup_segs_preview",
        "simple_syrup_segs_preview",
        inspector.element,
        { serialize: false, canvasOnly: false }
      );
      widget.serialize = false;
      widget.options ??= {};
      widget.options.serialize = false;
      widget.options.canvasOnly = false;
      widget.computeSize = (width = 420) => [Math.max(360, width), 470];
      const registerId = () => {
        if (candidate.id !== void 0) {
          controllers.set(String(candidate.id), controller);
        }
      };
      registerId();
      const originalExecuted = candidate.onExecuted;
      candidate.onExecuted = function(output) {
        originalExecuted?.call(this, output);
        controller.update(output);
      };
      const originalGraphConfigured = candidate.onGraphConfigured;
      candidate.onGraphConfigured = function(...args) {
        const result = originalGraphConfigured?.apply(this, args);
        registerId();
        if (candidate.id !== void 0) {
          controller.update(app2.nodeOutputs?.[String(candidate.id)]);
        }
        return result;
      };
      const originalRemoved = candidate.onRemoved;
      candidate.onRemoved = function(...args) {
        if (candidate.id !== void 0) controllers.delete(String(candidate.id));
        controller.dispose();
        return originalRemoved?.apply(this, args);
      };
      const computed = candidate.computeSize?.();
      const current = candidate.size ?? computed;
      if (current && candidate.setSize) {
        candidate.setSize([
          Math.max(420, current[0]),
          Math.max(520, computed?.[1] ?? current[1])
        ]);
      }
    },
    onNodeOutputsUpdated(outputs) {
      for (const [nodeId, output] of Object.entries(outputs)) {
        controllers.get(nodeId)?.update(output);
      }
    }
  };
  app2.registerExtension(extension);
}
function isPreviewNode(value) {
  if (typeof value !== "object" || value === null) return false;
  const node = value;
  return node.constructor?.comfyClass === SIMPLE_PREVIEW_SEGS_NODE_ID && typeof node.addDOMWidget === "function";
}

// web/src/main.ts
var comfyApp = app;
var comfyApi = window.comfyAPI.api.api;
comfyApp.registerExtension({
  name: "SimpleSyrup.Settings",
  async setup(appInstance) {
    await registerSimpleSyrupSettings(appInstance);
    registerExternalLLMRefreshHook(appInstance);
  }
});
registerMaskBatchUpload(comfyApp, comfyApi);
registerSimplePreviewSEGS(comfyApp, comfyApi);
