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

// web/src/main.ts
var comfyApp = app;
var comfyExecutionEvents = window.comfyAPI.api.api;
comfyApp.registerExtension({
  name: "SimpleSyrup.Settings",
  async setup(appInstance) {
    await registerSimpleSyrupSettings(appInstance);
    registerExternalLLMRefreshHook(appInstance);
  }
});
registerMaskBatchUpload(comfyApp, comfyExecutionEvents);
