// web/src/main.ts
import { app } from "../../../scripts/app.js";

// web/src/api.ts
var SETTINGS_ROUTE = "/simple-syrup/settings";
var QUANT_CACHE_ROUTE = "/simple-syrup/quant-cache";
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
    show_downloadable_models: payload.show_downloadable_models,
    quant_cache_limit_gib: payload.quant_cache_limit_gib
  };
}
async function getQuantCacheStatus(fetchImpl = fetch) {
  const response = await fetchImpl(QUANT_CACHE_ROUTE);
  if (!response.ok) {
    throw new Error(
      await backendErrorMessage(
        response,
        `Could not load quant cache status. Backend returned ${String(response.status)}.`
      )
    );
  }
  return parseQuantCacheStatus(await response.json());
}
async function clearQuantCache(fetchImpl = fetch) {
  const response = await fetchImpl(QUANT_CACHE_ROUTE, { method: "DELETE" });
  if (!response.ok) {
    throw new Error(
      await backendErrorMessage(
        response,
        `Could not clear quant cache. Backend returned ${String(response.status)}.`
      )
    );
  }
  return parseQuantCacheStatus(await response.json());
}
async function enforceQuantCacheLimit(fetchImpl = fetch) {
  const response = await fetchImpl(QUANT_CACHE_ROUTE, { method: "POST" });
  if (!response.ok) {
    throw new Error(
      await backendErrorMessage(
        response,
        `Could not enforce quant cache limit. Backend returned ${String(response.status)}.`
      )
    );
  }
  return parseQuantCacheStatus(await response.json());
}
function parseQuantCacheStatus(payload) {
  if (!isQuantCacheStatusPayload(payload)) {
    throw new Error(
      "SimpleSyrup quant cache status is invalid. Expected path, byte usage, limit, and artifact counts."
    );
  }
  return { ...payload };
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
  return typeof payload === "object" && payload !== null && typeof payload.show_downloadable_models === "boolean" && Number.isInteger(
    payload.quant_cache_limit_gib
  ) && Number(payload.quant_cache_limit_gib) > 0;
}
function isQuantCacheStatusPayload(payload) {
  if (typeof payload !== "object" || payload === null) return false;
  const candidate = payload;
  return typeof candidate.path === "string" && isNonNegativeInteger(candidate.usage_bytes) && isNonNegativeInteger(candidate.limit_bytes) && isNonNegativeInteger(candidate.artifact_count) && isNonNegativeInteger(candidate.active_artifact_count) && (candidate.removed_artifacts === void 0 || isNonNegativeInteger(candidate.removed_artifacts)) && (candidate.removed_bytes === void 0 || isNonNegativeInteger(candidate.removed_bytes));
}
function isNonNegativeInteger(value) {
  return typeof value === "number" && Number.isInteger(value) && value >= 0;
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

// web/src/downloadableModelsSetting.ts
var SIMPLE_SYRUP_SETTING_ID = "SimpleSyrup.ShowDownloadableModels";
var SIMPLE_SYRUP_SETTING_LABEL = "SimpleSyrup: Show downloadable models in loader dropdowns";
var SIMPLE_SYRUP_SETTING_DESCRIPTION = "Show known downloadable SAM, GroundingDINO, and ViTMatte models even when they are not installed locally.";
function registerDownloadableModelsSetting(app2, context, logger) {
  const setting = app2.ui.settings.addSetting({
    id: SIMPLE_SYRUP_SETTING_ID,
    name: SIMPLE_SYRUP_SETTING_LABEL,
    type: "boolean",
    defaultValue: context.getSettings().show_downloadable_models,
    tooltip: SIMPLE_SYRUP_SETTING_DESCRIPTION,
    onChange: async (value) => {
      const previous = context.getSettings();
      try {
        const saved = await context.saveSettings({
          ...previous,
          show_downloadable_models: value
        });
        context.setSettings(saved);
        setting.value = saved.show_downloadable_models;
      } catch (error) {
        logger.warn(
          "Could not save SimpleSyrup settings. The backend rejected the setting update.",
          error
        );
        setting.value = previous.show_downloadable_models;
      }
    }
  });
  setting.value = context.getSettings().show_downloadable_models;
}

// web/src/settingsUi.ts
function installSimpleSyrupSettingsStyle() {
  if (document.getElementById("simple-syrup-settings-style")) return;
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

// web/src/externalLlmSettings.ts
var EXTERNAL_LLM_ENDPOINT_SETTING_ID = "SimpleSyrup.ExternalLLM.Endpoint";
var EXTERNAL_LLM_ENDPOINT_SETTING_LABEL = "SimpleSyrup: External LLM endpoint";
var EXTERNAL_LLM_ENDPOINT_SETTING_DESCRIPTION = "OpenAI-compatible endpoint base URL used by SimpleSyrup prompt nodes.";
var EXTERNAL_LLM_API_KEY_SETTING_ID = "SimpleSyrup.ExternalLLM.ApiKey";
var EXTERNAL_LLM_API_KEY_SETTING_LABEL = "SimpleSyrup: External LLM API key";
var EXTERNAL_LLM_API_KEY_SETTING_DESCRIPTION = "Stores the API key for the configured external LLM endpoint in OS credential storage.";
async function registerExternalLLMSettings(app2, api = {
  getExternalLLMSettings,
  saveExternalLLMSettings,
  saveExternalLLMApiKey
}, logger = console) {
  let externalLLMSettings = {
    base_url: "",
    cached_models: [],
    default_model: "",
    has_api_key: false
  };
  try {
    externalLLMSettings = await api.getExternalLLMSettings();
  } catch (error) {
    logger.warn(
      "Could not load SimpleSyrup external LLM settings. Using empty endpoint settings until the backend is available.",
      error
    );
  }
  let savedExternalLLMSettings = externalLLMSettings;
  installSimpleSyrupSettingsStyle();
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

// web/src/quantCacheSetting.ts
var QUANT_CACHE_SETTING_ID = "SimpleSyrup.QuantCache";
var QUANT_CACHE_SETTING_LABEL = "SimpleSyrup: Quantized model cache";
var QUANT_CACHE_SETTING_DESCRIPTION = "Sets the global models/SyrupQuants cache limit in GiB for every SimpleSyrup loader; least-recently-used inactive copies are removed automatically.";
async function registerQuantCacheSetting(app2, settings, api, logger) {
  installSimpleSyrupSettingsStyle();
  let initialStatus = null;
  try {
    initialStatus = await api.getQuantCacheStatus();
  } catch (error) {
    logger.warn("Could not load SimpleSyrup quant cache status.", error);
  }
  app2.ui.settings.addSetting({
    id: QUANT_CACHE_SETTING_ID,
    name: QUANT_CACHE_SETTING_LABEL,
    sortOrder: 321,
    type: () => createQuantCacheControl({ settings, api, logger, initialStatus }),
    defaultValue: settings.getSettings().quant_cache_limit_gib,
    tooltip: QUANT_CACHE_SETTING_DESCRIPTION
  });
}
function createQuantCacheControl(context) {
  const wrapper = createElement("div", "simple-syrup-settings-row");
  const limitInput = createElement("input", "simple-syrup-settings-input");
  limitInput.type = "number";
  limitInput.min = "1";
  limitInput.max = "2048";
  limitInput.step = "1";
  limitInput.value = String(context.settings.getSettings().quant_cache_limit_gib);
  limitInput.setAttribute("aria-label", "Quant cache limit in GiB");
  const unitLabel = createElement("span", "simple-syrup-settings-status");
  unitLabel.textContent = "GiB limit";
  const saveButton = createElement("button", "simple-syrup-settings-button");
  saveButton.type = "button";
  saveButton.textContent = "Save Limit";
  const clearButton = createElement("button", "simple-syrup-settings-button");
  clearButton.type = "button";
  clearButton.textContent = "Clear Inactive";
  const status = createElement("span", "simple-syrup-settings-status");
  const renderStatus = (cacheStatus) => {
    status.textContent = cacheStatus ? `${formatGiB(cacheStatus.usage_bytes)} GiB used in ${cacheStatus.path} (${String(cacheStatus.artifact_count)} cached, ${String(cacheStatus.active_artifact_count)} active).` : "Cache status unavailable. Generated models are stored in models/SyrupQuants.";
  };
  renderStatus(context.initialStatus);
  saveButton.addEventListener("click", () => {
    void saveLimit();
  });
  limitInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      void saveLimit();
    }
  });
  clearButton.addEventListener("click", () => {
    void clearInactive();
  });
  const saveLimit = async () => {
    const limit = Number(limitInput.value);
    if (!Number.isInteger(limit) || limit < 1 || limit > 2048) {
      status.textContent = "Enter a whole-number cache limit from 1 to 2048 GiB.";
      return;
    }
    const previous = context.settings.getSettings();
    setPending(wrapper, true);
    try {
      const saved = await context.settings.saveSettings({
        ...previous,
        quant_cache_limit_gib: limit
      });
      context.settings.setSettings(saved);
      limitInput.value = String(saved.quant_cache_limit_gib);
      const refreshed = await context.api.enforceQuantCacheLimit();
      renderStatus(refreshed);
    } catch (error) {
      context.logger.warn("Could not save SimpleSyrup quant cache limit.", error);
      limitInput.value = String(previous.quant_cache_limit_gib);
      status.textContent = "Cache limit was not saved.";
    } finally {
      setPending(wrapper, false);
    }
  };
  const clearInactive = async () => {
    setPending(wrapper, true);
    try {
      const cleared = await context.api.clearQuantCache();
      renderStatus(cleared);
    } catch (error) {
      context.logger.warn("Could not clear SimpleSyrup quant cache.", error);
      status.textContent = "Inactive cached models were not cleared.";
    } finally {
      setPending(wrapper, false);
    }
  };
  wrapper.append(limitInput, unitLabel, saveButton, clearButton, status);
  return wrapper;
}
function formatGiB(bytes) {
  return (bytes / 1024 ** 3).toFixed(2);
}

// web/src/settingsRegistration.ts
var DEFAULT_SETTINGS = {
  show_downloadable_models: true,
  quant_cache_limit_gib: 20
};
async function registerSimpleSyrupSettings(app2, api = defaultApi(), logger = console) {
  let savedSettings = DEFAULT_SETTINGS;
  try {
    savedSettings = await api.getSettings();
  } catch (error) {
    logger.warn(
      "Could not load SimpleSyrup settings. Using defaults until the backend is available.",
      error
    );
  }
  const settingsContext = {
    getSettings: () => savedSettings,
    saveSettings: (settings) => api.saveSettings(settings),
    setSettings: (settings) => {
      savedSettings = settings;
    }
  };
  registerDownloadableModelsSetting(app2, settingsContext, logger);
  await registerQuantCacheSetting(app2, settingsContext, api, logger);
  await registerExternalLLMSettings(app2, api, logger);
}
function defaultApi() {
  return {
    getSettings,
    saveSettings,
    getQuantCacheStatus,
    enforceQuantCacheLimit,
    clearQuantCache,
    getExternalLLMSettings,
    saveExternalLLMSettings,
    saveExternalLLMApiKey
  };
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

// web/src/nativeNodePreview.ts
var NativeNodePreview = class {
  constructor(app2, api, node) {
    this.app = app2;
    this.api = api;
    this.node = node;
  }
  app;
  api;
  node;
  publishListeners = /* @__PURE__ */ new Set();
  /** Publish media as an execution-shaped output for both node renderers. */
  publish(output) {
    const localNodeId = this.localNodeId();
    if (!localNodeId) return;
    const outputLocator = this.outputLocator(localNodeId);
    this.app.nodeOutputs ??= {};
    this.app.nodeOutputs[localNodeId] = output;
    this.app.nodeOutputs[outputLocator] = output;
    for (const executionId of this.executionIds(localNodeId)) {
      this.api.dispatchEvent(
        new CustomEvent("executed", {
          detail: { node: executionId, display_node: executionId, output }
        })
      );
    }
    for (const listener of this.publishListeners) listener();
  }
  /** Notify renderer adapters after native output changes. */
  subscribe(listener) {
    this.publishListeners.add(listener);
    return () => this.publishListeners.delete(listener);
  }
  /** Remove loader-owned media from Comfy's native preview surface. */
  clear() {
    this.publish({ images: [], animated: [] });
  }
  localNodeId() {
    if (this.node.id === void 0) return void 0;
    return String(this.node.id);
  }
  /** Match the output key selected by Comfy's Nodes 2.0 subgraph renderer. */
  outputLocator(localNodeId) {
    const graphId = this.node.graph?.id;
    const isSubgraph = this.node.graph && this.node.graph !== this.app.rootGraph;
    return isSubgraph && graphId !== void 0 ? `${String(graphId)}:${localNodeId}` : localNodeId;
  }
  /** Resolve instance paths because Comfy events consume execution IDs, not graph locators. */
  executionIds(localNodeId) {
    const graph = this.node.graph;
    const rootGraph = this.app.rootGraph;
    if (!graph || !rootGraph || graph === rootGraph || graph.isRootGraph) {
      return [localNodeId];
    }
    const parentPaths = findGraphInstancePaths(rootGraph, graph);
    return parentPaths.length > 0 ? parentPaths.map((path) => `${path}:${localNodeId}`) : [];
  }
};
function findGraphInstancePaths(root, target, visited = /* @__PURE__ */ new Set()) {
  if (visited.has(root)) return [];
  const nextVisited = new Set(visited);
  nextVisited.add(root);
  const paths = [];
  for (const node of root.nodes ?? []) {
    const nodeId = graphNodeId(node);
    if (!nodeId || !node.subgraph) continue;
    if (node.subgraph === target) paths.push(nodeId);
    for (const nestedPath of findGraphInstancePaths(
      node.subgraph,
      target,
      nextVisited
    )) {
      paths.push(`${nodeId}:${nestedPath}`);
    }
  }
  return paths;
}
function graphNodeId(node) {
  if (node.id === void 0) return void 0;
  const identity = String(node.id);
  return identity.length > 0 ? identity : void 0;
}

// web/src/orderedMediaPreview.ts
var OrderedMediaPreviewController = class {
  constructor(target, label, logger = console) {
    this.target = target;
    this.label = label;
    this.logger = logger;
  }
  target;
  label;
  logger;
  requestVersion = 0;
  disposed = false;
  hasPreview = false;
  /** Resolve one ordered selection without allowing stale previews to win. */
  refresh(files, loadPreview) {
    if (this.disposed) return;
    const requestVersion = ++this.requestVersion;
    if (files.length === 0) {
      this.target.clear();
      this.hasPreview = false;
      return;
    }
    void loadPreview().then(async (output) => {
      await nextTask();
      if (this.disposed || requestVersion !== this.requestVersion) return;
      if (!Array.isArray(output.images) || output.images.length !== files.length) {
        throw new Error(
          `${this.label} preview returned ${String(output.images?.length ?? 0)} images for ${String(files.length)} files.`
        );
      }
      this.target.publish(output);
      this.hasPreview = true;
    }).catch((error) => {
      if (this.disposed || requestVersion !== this.requestVersion) return;
      const message = error instanceof Error ? error.message : String(error);
      this.logger.warn(`Could not refresh ${this.label} preview: ${message}`, error);
      if (!this.hasPreview) this.target.clear();
    });
  }
  /** Clear native media and invalidate pending work owned by a removed node. */
  dispose() {
    if (this.disposed) return;
    this.disposed = true;
    this.requestVersion += 1;
    this.target.clear();
  }
};
function nextTask() {
  return new Promise((resolve) => {
    setTimeout(resolve, 0);
  });
}

// web/src/nativePreviewLifecycle.ts
var listeners = /* @__PURE__ */ new Set();
var observer = null;
function subscribeNativePreviewLifecycle(listener) {
  listeners.add(listener);
  ensureObserver();
  return () => {
    listeners.delete(listener);
    if (listeners.size === 0) {
      observer?.disconnect();
      observer = null;
    }
  };
}
function ensureObserver() {
  if (observer) return;
  observer = new MutationObserver(() => {
    for (const listener of listeners) listener();
  });
  observer.observe(document.body, {
    childList: true,
    subtree: true,
    attributes: true,
    attributeFilter: ["aria-current", "src"]
  });
}

// web/src/orderedMediaPreviewAffordances.ts
var MEDIA_MOVE_EARLIER_ATTRIBUTE = "data-ss-media-move-earlier";
var MEDIA_MOVE_LATER_ATTRIBUTE = "data-ss-media-move-later";
var MEDIA_REMOVE_ATTRIBUTE = "data-ss-media-remove";
var MEDIA_INDEX_ATTRIBUTE = "data-ss-media-index";
var OrderedMediaPreviewAffordances = class {
  constructor(options) {
    this.options = options;
    window.addEventListener("resize", this.requestRefresh, true);
    document.addEventListener("click", this.requestRefresh, true);
    document.addEventListener("keydown", this.requestRefresh, true);
    document.addEventListener("pointerdown", this.suspendViewportControls, true);
    document.addEventListener("pointermove", this.requestRefresh, true);
    document.addEventListener("pointerup", this.resumeViewportControls, true);
    document.addEventListener("pointercancel", this.resumeViewportControls, true);
    document.addEventListener("wheel", this.requestRefresh, true);
    this.refresh();
  }
  options;
  elements = [];
  positionedContainers = /* @__PURE__ */ new Map();
  animationFrame = null;
  remainingSyncFrames = 0;
  viewportControlsSuspended = false;
  /** Recheck native layout across several frames after Comfy rerenders output. */
  refresh() {
    this.remainingSyncFrames = Math.max(this.remainingSyncFrames, 4);
    this.scheduleSync();
  }
  /** Remove every overlay control and global layout listener. */
  dispose() {
    window.removeEventListener("resize", this.requestRefresh, true);
    document.removeEventListener("click", this.requestRefresh, true);
    document.removeEventListener("keydown", this.requestRefresh, true);
    document.removeEventListener("pointerdown", this.suspendViewportControls, true);
    document.removeEventListener("pointermove", this.requestRefresh, true);
    document.removeEventListener("pointerup", this.resumeViewportControls, true);
    document.removeEventListener("pointercancel", this.resumeViewportControls, true);
    document.removeEventListener("wheel", this.requestRefresh, true);
    if (this.animationFrame !== null) {
      cancelAnimationFrame(this.animationFrame);
      this.animationFrame = null;
    }
    for (const element of this.elements) element.root.remove();
    this.elements = [];
    this.releasePositionedContainers(/* @__PURE__ */ new Set());
  }
  requestRefresh = () => {
    this.remainingSyncFrames = Math.max(this.remainingSyncFrames, 2);
    this.scheduleSync();
  };
  suspendViewportControls = (event) => {
    if (!(event.target instanceof HTMLCanvasElement)) return;
    this.viewportControlsSuspended = true;
    for (const elements of this.elements) {
      if (elements.root.parentElement === document.body) elements.root.hidden = true;
    }
  };
  resumeViewportControls = () => {
    if (!this.viewportControlsSuspended) return;
    this.viewportControlsSuspended = false;
    this.refresh();
  };
  scheduleSync() {
    if (this.animationFrame !== null) return;
    this.animationFrame = requestAnimationFrame(() => {
      this.animationFrame = null;
      this.sync();
      this.remainingSyncFrames -= 1;
      if (this.remainingSyncFrames > 0) this.scheduleSync();
    });
  }
  sync() {
    const slots = this.options.getSlots();
    const activeContainers = /* @__PURE__ */ new Set();
    this.resizeElements(slots.length);
    for (const [index, elements] of this.elements.entries()) {
      const actionSlot = slots[index];
      const slot = actionSlot?.bounds;
      if (!actionSlot || !slot || slot.width <= 0 || slot.height <= 0 || !actionSlot.container && this.viewportControlsSuspended) {
        elements.root.hidden = true;
        continue;
      }
      const itemIndex = actionSlot.itemIndex;
      elements.root.hidden = false;
      const position = this.mount(elements.root, actionSlot, activeContainers);
      const stripHeight = Math.max(18, Math.min(26, position.height * 0.16));
      Object.assign(elements.root.style, {
        left: `${String(position.left)}px`,
        top: `${String(position.top)}px`,
        width: `${String(position.width)}px`,
        height: `${String(stripHeight)}px`
      });
      elements.root.dataset.ssMediaIndex = String(itemIndex);
      elements.earlier.dataset.ssMediaIndex = String(itemIndex);
      elements.later.dataset.ssMediaIndex = String(itemIndex);
      elements.remove.dataset.ssMediaIndex = String(itemIndex);
      setActionAvailability(elements.earlier, itemIndex > 0);
      setActionAvailability(
        elements.later,
        itemIndex < this.options.getItemCount() - 1
      );
      elements.earlier.setAttribute(
        "aria-label",
        `Move ${this.options.itemLabel} ${String(itemIndex + 1)} earlier`
      );
      elements.later.setAttribute(
        "aria-label",
        `Move ${this.options.itemLabel} ${String(itemIndex + 1)} later`
      );
      elements.remove.setAttribute(
        "aria-label",
        `Remove ${this.options.itemLabel} ${String(itemIndex + 1)}`
      );
    }
    this.releasePositionedContainers(activeContainers);
  }
  /** Mount one control strip in the preview surface that owns its geometry. */
  mount(root, actionSlot, activeContainers) {
    const container = actionSlot.container;
    if (!container) {
      if (root.parentElement !== document.body) document.body.append(root);
      root.style.position = "fixed";
      root.style.zIndex = "2";
      return actionSlot.bounds;
    }
    activeContainers.add(container);
    this.positionContainer(container);
    if (root.parentElement !== container) container.append(root);
    root.style.position = "absolute";
    root.style.zIndex = "1";
    const containerRect = container.getBoundingClientRect();
    const scaleX = containerScale(containerRect.width, container.offsetWidth);
    const scaleY = containerScale(containerRect.height, container.offsetHeight);
    return {
      left: (actionSlot.bounds.left - containerRect.left) / scaleX - container.clientLeft + container.scrollLeft,
      top: (actionSlot.bounds.top - containerRect.top) / scaleY - container.clientTop + container.scrollTop,
      width: actionSlot.bounds.width / scaleX,
      height: actionSlot.bounds.height / scaleY
    };
  }
  /** Establish a local containing block without overriding authored positioning. */
  positionContainer(container) {
    if (this.positionedContainers.has(container)) return;
    const position = getComputedStyle(container).position;
    if (position !== "" && position !== "static") return;
    this.positionedContainers.set(container, container.style.position);
    container.style.position = "relative";
  }
  /** Restore preview surfaces that no longer contain loader controls. */
  releasePositionedContainers(activeContainers) {
    for (const [container, originalPosition] of this.positionedContainers) {
      if (activeContainers.has(container)) continue;
      container.style.position = originalPosition;
      this.positionedContainers.delete(container);
    }
  }
  resizeElements(count) {
    while (this.elements.length > count) this.elements.pop()?.root.remove();
    while (this.elements.length < count) {
      this.elements.push(this.createElements());
    }
  }
  createElements() {
    const root = document.createElement("div");
    root.className = "ss-native-preview-affordance";
    Object.assign(root.style, {
      position: "fixed",
      zIndex: "2",
      display: "flex",
      alignItems: "stretch",
      pointerEvents: "none",
      overflow: "hidden",
      background: "rgba(20, 20, 20, 0.72)"
    });
    const earlier = controlButton("pi-arrow-left", "Move earlier");
    earlier.setAttribute(MEDIA_MOVE_EARLIER_ATTRIBUTE, "true");
    const later = controlButton("pi-arrow-right", "Move later");
    later.setAttribute(MEDIA_MOVE_LATER_ATTRIBUTE, "true");
    const remove = controlButton("pi-times", "Remove");
    remove.setAttribute(MEDIA_REMOVE_ATTRIBUTE, "true");
    bindAction(earlier, (index) => {
      this.options.moveEarlier(index);
    });
    bindAction(later, (index) => {
      this.options.moveLater(index);
    });
    bindAction(remove, (index) => {
      this.options.remove(index);
    });
    root.append(earlier, later, remove);
    document.body.append(root);
    return { root, earlier, later, remove };
  }
};
function mediaIndex(element) {
  const value = element.getAttribute(MEDIA_INDEX_ATTRIBUTE);
  if (value === null) return null;
  const index = Number(value);
  return Number.isInteger(index) && index >= 0 ? index : null;
}
function controlButton(iconClass, title) {
  const button = document.createElement("button");
  button.type = "button";
  button.title = title;
  const icon = document.createElement("i");
  icon.className = `pi ${iconClass}`;
  icon.setAttribute("aria-hidden", "true");
  button.append(icon);
  Object.assign(button.style, {
    appearance: "none",
    border: "0",
    padding: "0",
    margin: "0",
    minWidth: "0",
    flex: "1 1 0",
    color: "rgba(255, 255, 255, 0.92)",
    background: "transparent",
    fontSize: "13px",
    cursor: "pointer",
    pointerEvents: "auto"
  });
  return button;
}
function bindAction(button, action) {
  button.addEventListener("pointerdown", stopControlPointerEvent);
  button.addEventListener("click", (event) => {
    event.preventDefault();
    event.stopPropagation();
    const index = mediaIndex(button);
    if (index !== null) action(index);
  });
}
function setActionAvailability(button, available) {
  button.disabled = !available;
  button.style.cursor = available ? "pointer" : "default";
  button.style.opacity = available ? "1" : "0.35";
}
function stopControlPointerEvent(event) {
  event.preventDefault();
  event.stopPropagation();
}
function containerScale(renderedSize, localSize) {
  if (renderedSize <= 0 || localSize <= 0) return 1;
  const scale = renderedSize / localSize;
  return Number.isFinite(scale) && scale > 0 ? scale : 1;
}

// web/src/orderedMediaPreviewTransaction.ts
var OrderedMediaPreviewTransaction = class {
  constructor(options) {
    this.options = options;
  }
  options;
  surface = null;
  items = null;
  handoffFrame = null;
  /** Move an already-loaded native item without rebuilding its gallery. */
  move(index, destination) {
    if (!this.begin()) return false;
    const items = this.items;
    if (!items || index < 0 || index >= items.length || destination < 0 || destination >= items.length || index === destination) {
      return false;
    }
    const [moved] = items.splice(index, 1);
    if (!moved) return false;
    items.splice(destination, 0, moved);
    this.apply();
    return true;
  }
  /** Remove an already-loaded item and let the native gallery compact itself. */
  remove(index) {
    if (!this.begin()) return false;
    const items = this.items;
    if (!items || index < 0 || index >= items.length) return false;
    items.splice(index, 1);
    this.apply();
    return true;
  }
  /** Return live native-cell geometry while an optimistic transaction is active. */
  activeSlots() {
    if (!this.surface || !this.items) return null;
    return this.surface.slots.slice(0, this.items.length).map((slot) => slot());
  }
  /** End the transaction after Comfy's authoritative native preview is ready. */
  authoritativePublished() {
    if (!this.surface || this.handoffFrame !== null) return;
    this.scheduleHandoff();
  }
  /** Release native surface changes and stop any pending handoff check. */
  dispose() {
    if (this.handoffFrame !== null) cancelAnimationFrame(this.handoffFrame);
    this.handoffFrame = null;
    this.finish();
  }
  begin() {
    if (this.surface && this.items) return true;
    const surface = this.options.captureSurface();
    if (!surface || surface.items.length === 0) return false;
    this.surface = surface;
    this.items = [...surface.items];
    return true;
  }
  apply() {
    if (!this.surface || !this.items) return;
    this.surface.apply(this.items);
    this.options.stateChanged();
  }
  scheduleHandoff() {
    this.handoffFrame = requestAnimationFrame(() => {
      this.handoffFrame = null;
      if (!this.surface) return;
      if (this.options.authoritativeReady()) {
        this.finish();
        this.options.stateChanged();
        return;
      }
      this.scheduleHandoff();
    });
  }
  finish() {
    this.surface?.release();
    this.surface = null;
    this.items = null;
  }
};

// web/src/comfyImageReference.ts
function comfyImageReferenceKey(reference) {
  return `${reference.type}
${reference.subfolder}
${reference.filename}`;
}
function comfyImageSourceKey(sourceUrl) {
  try {
    const url = new URL(sourceUrl, window.location.href);
    return `${url.searchParams.get("type") ?? ""}
${url.searchParams.get("subfolder") ?? ""}
${url.searchParams.get("filename") ?? ""}`;
  } catch {
    return "";
  }
}
function loadedImagesMatchReferences(images, references) {
  return images?.length === references.length && images.every((image, index) => {
    const reference = references[index];
    return reference !== void 0 && comfyImageSourceKey(image.currentSrc || image.src) === comfyImageReferenceKey(reference);
  });
}

// web/src/orderedMediaPreviewActions.ts
var CUBE_FACE_PROJECTION_SYMBOL = /* @__PURE__ */ Symbol.for(
  "sugarcubes.cube-face-projection.v1"
);
var OrderedMediaPreviewActions = class {
  constructor(options) {
    this.options = options;
    this.transaction = new OrderedMediaPreviewTransaction({
      captureSurface: () => this.captureSurface(),
      authoritativeReady: () => this.authoritativeReady(),
      stateChanged: () => {
        this.affordances.refresh();
      }
    });
    const moveEarlier = (index) => {
      const destination = index - 1;
      this.transaction.move(index, destination);
      this.followMovedDetail(index, destination);
      options.moveEarlier(index);
    };
    const moveLater = (index) => {
      const destination = index + 1;
      this.transaction.move(index, destination);
      this.followMovedDetail(index, destination);
      options.moveLater(index);
    };
    const remove = (index) => {
      const removedDetail = this.selectedItemIndex() === index;
      this.transaction.remove(index);
      options.remove(index);
      this.followRemovedDetail(index, removedDetail);
    };
    const actionOptions = {
      getItemCount: () => this.itemCount(),
      moveEarlier,
      moveLater,
      remove
    };
    this.affordances = new OrderedMediaPreviewAffordances({
      itemLabel: options.itemLabel,
      getSlots: () => this.nativeActionSlots(),
      ...actionOptions
    });
    this.unsubscribePreview = options.preview.subscribe(() => {
      this.affordances.refresh();
      this.transaction.authoritativePublished();
      this.restorePendingDetail();
    });
    this.unsubscribeLifecycle = subscribeNativePreviewLifecycle(() => {
      this.affordances.refresh();
    });
  }
  options;
  affordances;
  transaction;
  unsubscribePreview;
  unsubscribeLifecycle;
  lastCanvasPreviewRect = null;
  pendingDetailIndex = null;
  detailRestoreFrame = null;
  /** Remove layout listeners and every loader-owned overlay control. */
  dispose() {
    this.unsubscribePreview();
    this.unsubscribeLifecycle();
    this.transaction.dispose();
    this.affordances.dispose();
    if (this.detailRestoreFrame !== null) {
      cancelAnimationFrame(this.detailRestoreFrame);
      this.detailRestoreFrame = null;
    }
  }
  itemCount() {
    const files = this.options.getFiles();
    const images = this.currentImages();
    return images?.length === files.length ? files.length : 0;
  }
  currentImages() {
    const nodeId = this.nodeId();
    return nodeId ? this.options.app.nodeOutputs?.[nodeId]?.images : void 0;
  }
  domImages() {
    const root = this.domRoot();
    const references = this.currentImages();
    if (!root || !references || references.length !== this.itemCount()) return [];
    const remainingKeys = references.map(comfyImageReferenceKey);
    const matched = [];
    for (const image of Array.from(
      root.querySelectorAll("img")
    )) {
      const key = comfyImageSourceKey(image.src);
      const referenceIndex = remainingKeys.indexOf(key);
      if (referenceIndex < 0) continue;
      matched.push(image);
      remainingKeys.splice(referenceIndex, 1);
    }
    return remainingKeys.length === 0 ? matched : [];
  }
  domRoot() {
    const nodeId = this.nodeId();
    if (!nodeId) return null;
    return Array.from(document.querySelectorAll("[data-node-id]")).find(
      (element) => element.dataset.nodeId === nodeId
    ) ?? null;
  }
  canvasSlots() {
    const imageRects = this.options.node.imageRects;
    if (!imageRects) return [];
    if (imageRects.length > 0) {
      this.lastCanvasPreviewRect = unionImageRects(imageRects);
    }
    const slots = [];
    for (const rect of imageRects) {
      const slot = this.canvasLocalSlot(rect);
      if (slot) slots.push(slot);
    }
    return slots;
  }
  nativeActionSlots() {
    if (!this.belongsToActiveWorkflow()) return [];
    if (this.options.node.flags?.collapsed) return [];
    const selectedIndex = this.selectedItemIndex();
    if (selectedIndex !== null) {
      return this.detailActionSlots(selectedIndex);
    }
    const activeSlots = this.transaction.activeSlots();
    const domImages = this.domImages();
    if (activeSlots) {
      return indexedSlots(
        activeSlots,
        this.domPreviewContainer(domImages) ?? this.cubeFaceProjection()?.container ?? null
      );
    }
    if (!this.isGridVisible()) return [];
    const domSlots = domImages.map((image) => {
      const rect = image.getBoundingClientRect();
      return {
        left: rect.left,
        top: rect.top,
        width: rect.width,
        height: rect.height
      };
    });
    if (domSlots.length === this.itemCount()) {
      return indexedSlots(domSlots, this.domPreviewContainer(domImages));
    }
    return indexedSlots(
      this.canvasSlots(),
      this.cubeFaceProjection()?.container ?? null
    );
  }
  /** Prevent inactive workflow tabs with reused node IDs from claiming active Nodes 2 DOM. */
  belongsToActiveWorkflow() {
    const nodeGraph = this.options.node.graph;
    const nodeRoot = nodeGraph?._rootGraph ?? nodeGraph;
    const activeRoot = this.options.app.rootGraph ?? this.options.app.canvas?.graph;
    return !nodeRoot || !activeRoot || nodeRoot === activeRoot;
  }
  /** Expose one selected-item control strip in native detail mode. */
  detailActionSlots(selectedIndex) {
    const domSlot = this.domDetailSlot(selectedIndex);
    if (domSlot) return [domSlot];
    const canvasSlot = this.canvasDetailSlot();
    if (!canvasSlot) return [];
    const container = this.cubeFaceProjection()?.container;
    return [
      container ? { itemIndex: selectedIndex, bounds: canvasSlot, container } : { itemIndex: selectedIndex, bounds: canvasSlot }
    ];
  }
  /** Locate the selected Nodes 2.0 preview without replacing native UI. */
  domDetailSlot(selectedIndex) {
    const root = this.domRoot();
    const reference = this.currentImages()?.[selectedIndex];
    if (!root || !reference) return null;
    const expectedKey = comfyImageReferenceKey(reference);
    const candidates = Array.from(
      root.querySelectorAll("img")
    ).filter((image2) => {
      const rect = image2.getBoundingClientRect();
      return comfyImageSourceKey(image2.src) === expectedKey && rect.width > 0 && rect.height > 0;
    });
    const image = candidates.reduce(
      (largest, candidate) => !largest || imageArea(candidate) > imageArea(largest) ? candidate : largest,
      null
    );
    const previewRegion = root.querySelector(
      '[role="region"][aria-label^="Image preview"]'
    );
    const previewElement = image ?? previewRegion;
    if (!previewElement) return null;
    return {
      itemIndex: selectedIndex,
      bounds: elementSlot(previewElement),
      container: previewRegion ?? this.domPreviewContainer(image ? [image] : []) ?? root
    };
  }
  /** Return the closest native preview surface shared by rendered media. */
  domPreviewContainer(images) {
    const root = this.domRoot();
    if (!root || images.length === 0) return null;
    const previewRegion = root.querySelector(
      '[role="region"][aria-label^="Image preview"]'
    );
    if (previewRegion && images.every((image) => previewRegion.contains(image))) {
      return previewRegion;
    }
    let candidate = images[0]?.parentElement ?? null;
    while (candidate && candidate !== root) {
      if (images.every((image) => candidate?.contains(image) === true)) {
        return candidate;
      }
      candidate = candidate.parentElement;
    }
    return root;
  }
  /** Resolve selected preview geometry in node-local canvas coordinates. */
  canvasDetailLocalSlot() {
    const previewWidget = this.options.node.widgets?.find(
      (widget) => widget.options?.canvasOnly === true && typeof widget.y === "number" && typeof widget.computedHeight === "number" && widget.computedHeight > 0
    );
    const nodeWidth = this.options.node.size?.[0];
    const previewY = previewWidget?.y;
    const previewHeight = previewWidget?.computedHeight;
    if (typeof nodeWidth !== "number" || nodeWidth <= 0 || typeof previewY !== "number" || typeof previewHeight !== "number") {
      if (!this.lastCanvasPreviewRect) return null;
      const [left, top, width, height] = this.lastCanvasPreviewRect;
      return { left, top, width, height };
    }
    return { left: 0, top: previewY, width: nodeWidth, height: previewHeight };
  }
  /** Locate the selected Nodes 1.0 preview from its canvas widget geometry. */
  canvasDetailSlot() {
    const localSlot = this.canvasDetailLocalSlot();
    return localSlot ? this.canvasLocalSlot([
      localSlot.left,
      localSlot.top,
      localSlot.width,
      localSlot.height
    ]) : null;
  }
  /** Project one node-local canvas rectangle into viewport coordinates. */
  canvasLocalSlot(rect) {
    const projection = this.cubeFaceProjection();
    if (projection) {
      try {
        const slot = projection.projectRect(rect);
        if (validSlot(slot)) return slot;
      } catch {
        return null;
      }
    }
    const canvasApi = this.options.app.canvas;
    const nodePosition = this.options.node.pos;
    if (!canvasApi || !nodePosition) return null;
    const [x, y, width, height] = rect;
    const canvasRect = canvasApi.canvas.getBoundingClientRect();
    const topLeft = canvasApi.convertOffsetToCanvas([
      nodePosition[0] + x,
      nodePosition[1] + y
    ]);
    const bottomRight = canvasApi.convertOffsetToCanvas([
      nodePosition[0] + x + width,
      nodePosition[1] + y + height
    ]);
    return {
      left: canvasRect.left + topLeft[0],
      top: canvasRect.top + topLeft[1],
      width: bottomRight[0] - topLeft[0],
      height: bottomRight[1] - topLeft[1]
    };
  }
  /** Read SugarCubes' renderer-neutral projection contract when this node is embedded. */
  cubeFaceProjection() {
    const value = Reflect.get(
      this.options.node,
      CUBE_FACE_PROJECTION_SYMBOL
    );
    if (typeof value !== "object" || value === null) return null;
    const candidate = value;
    if (!(candidate.container instanceof HTMLElement) || typeof candidate.projectRect !== "function") {
      return null;
    }
    return candidate;
  }
  captureSurface() {
    const domImages = this.domImages();
    if (domImages.length > 0) return this.domSurface(domImages);
    const canvasImages = this.options.node.imgs;
    if (!canvasImages || canvasImages.length !== this.currentImages()?.length) {
      return null;
    }
    return {
      items: previewItems(canvasImages),
      slots: canvasImages.map(
        (_, index) => () => this.canvasSlots()[index]
      ),
      apply: (items) => {
        this.options.node.imgs = items.map((item) => item.image);
        delete this.options.node.imageRects;
        this.options.node.graph?.setDirtyCanvas?.(true, true);
      },
      release: () => void 0
    };
  }
  domSurface(images) {
    const slots = images.map((image) => imageSlot(image));
    const targets = images.map((image) => image.closest("button") ?? image);
    const originalDisplays = targets.map((target) => target.style.display);
    return {
      items: previewItems(images),
      slots,
      apply: (items) => {
        for (const [index, image] of images.entries()) {
          const item = items[index];
          const target = targets[index];
          if (!target) continue;
          if (!item) {
            target.style.display = "none";
            continue;
          }
          target.style.display = originalDisplays[index] ?? "";
          if (image.src !== item.sourceUrl) image.src = item.sourceUrl;
        }
      },
      release: () => {
        for (const [index, target] of targets.entries()) {
          target.style.display = originalDisplays[index] ?? "";
        }
      }
    };
  }
  authoritativeReady() {
    const references = this.currentImages();
    if (!references || references.length !== this.options.getFiles().length) {
      return false;
    }
    const expectedKeys = references.map(comfyImageReferenceKey);
    const rendered = this.authoritativeImages(expectedKeys);
    return rendered.length === expectedKeys.length && rendered.every(
      (image, index) => comfyImageSourceKey(image.src) === expectedKeys[index] && image.complete && image.naturalWidth > 0
    );
  }
  authoritativeImages(expectedKeys) {
    const root = this.domRoot();
    const domImages = root ? Array.from(root.querySelectorAll("img")) : [];
    if (domImages.length === expectedKeys.length) return domImages;
    return this.options.node.imgs ?? [];
  }
  isGridVisible() {
    return !this.options.node.flags?.collapsed && this.options.node.imageIndex == null && this.itemCount() > 1;
  }
  /** Read detail selection from the renderer that currently owns it. */
  selectedItemIndex() {
    const index = this.options.node.imageIndex;
    if (typeof index === "number" && Number.isInteger(index) && index >= 0 && index < this.itemCount()) {
      return index;
    }
    const currentButton = this.domDetailButtons().find(
      (button) => button.getAttribute("aria-current") === "true"
    );
    if (!currentButton) return null;
    const detailIndex = this.domDetailButtons().indexOf(currentButton);
    return detailIndex >= 0 && detailIndex < this.itemCount() ? detailIndex : null;
  }
  /** Keep the moved item selected across both native renderer state models. */
  followMovedDetail(index, destination) {
    if (this.selectedItemIndex() !== index) return;
    this.pendingDetailIndex = destination;
    if (this.options.node.imageIndex === index) {
      this.options.node.imageIndex = destination;
    } else {
      this.domDetailButtons()[destination]?.click();
    }
    this.affordances.refresh();
  }
  /** Select the nearest remaining item after removing an inspected item. */
  followRemovedDetail(index, removedDetail) {
    if (!removedDetail) return;
    const remaining = this.itemCount();
    const destination = remaining > 0 ? Math.min(index, remaining - 1) : null;
    this.pendingDetailIndex = destination;
    if (this.options.node.imageIndex === index) {
      this.options.node.imageIndex = destination;
    } else if (destination !== null) {
      this.domDetailButtons()[destination]?.click();
    }
    this.affordances.refresh();
  }
  /** Re-enter Nodes 2.0 detail mode after output publication resets its grid. */
  restorePendingDetail() {
    if (this.pendingDetailIndex === null || this.detailRestoreFrame !== null) {
      return;
    }
    let stableFrames = 0;
    let remainingFrames = 12;
    const restore = () => {
      this.detailRestoreFrame = null;
      const destination = this.pendingDetailIndex;
      if (destination === null) return;
      const buttons = this.domDetailButtons();
      const selected = buttons.findIndex(
        (button) => button.getAttribute("aria-current") === "true"
      );
      if (selected === destination) {
        stableFrames += 1;
      } else {
        stableFrames = 0;
        buttons[destination]?.click();
      }
      remainingFrames -= 1;
      if (stableFrames >= 3 || remainingFrames <= 0) {
        this.pendingDetailIndex = null;
        this.affordances.refresh();
        return;
      }
      this.detailRestoreFrame = requestAnimationFrame(restore);
    };
    this.detailRestoreFrame = requestAnimationFrame(restore);
  }
  /** Return Comfy's ordered detail navigation controls for this node. */
  domDetailButtons() {
    const root = this.domRoot();
    if (!root) return [];
    return Array.from(
      root.querySelectorAll(
        'button[aria-label^="View image "]'
      )
    );
  }
  nodeId() {
    const nodeId = this.options.node.id;
    return nodeId === void 0 ? void 0 : String(nodeId);
  }
};
function previewItems(images) {
  return images.map((image) => ({
    sourceUrl: image.currentSrc || image.src,
    image
  }));
}
function imageSlot(image) {
  return () => elementSlot(image);
}
function elementSlot(element) {
  const rect = element.getBoundingClientRect();
  return {
    left: rect.left,
    top: rect.top,
    width: rect.width,
    height: rect.height
  };
}
function validSlot(slot) {
  return Number.isFinite(slot.left) && Number.isFinite(slot.top) && Number.isFinite(slot.width) && Number.isFinite(slot.height) && slot.width >= 0 && slot.height >= 0;
}
function imageArea(image) {
  const rect = image.getBoundingClientRect();
  return rect.width * rect.height;
}
function indexedSlots(slots, container = null) {
  return slots.map(
    (bounds, itemIndex) => container ? { itemIndex, bounds, container } : { itemIndex, bounds }
  );
}
function unionImageRects(rects) {
  const left = Math.min(...rects.map(([x]) => x));
  const top = Math.min(...rects.map(([, y]) => y));
  const right = Math.max(...rects.map(([x, , width]) => x + width));
  const bottom = Math.max(...rects.map(([, y, , height]) => y + height));
  return [left, top, right - left, bottom - top];
}

// web/src/orderedMediaSelection.ts
var OrderedMediaSelection = class {
  files;
  constructor(value = []) {
    this.files = normalizeMediaFiles(value);
  }
  /** Return a defensive snapshot in authored order. */
  snapshot() {
    return [...this.files];
  }
  /** Replace every position with a normalized persisted widget value. */
  replace(value) {
    this.files = normalizeMediaFiles(value);
    return this.snapshot();
  }
  /** Append every incoming position without deduplicating filenames. */
  append(value) {
    this.files.push(...normalizeMediaFiles(value));
    return this.snapshot();
  }
  /** Remove one exact position when it exists. */
  remove(index) {
    if (validIndex(index, this.files.length)) this.files.splice(index, 1);
    return this.snapshot();
  }
  /** Move one exact position and retain all duplicate filenames. */
  move(from, to) {
    if (!validIndex(from, this.files.length) || !validIndex(to, this.files.length)) {
      return this.snapshot();
    }
    const [moved] = this.files.splice(from, 1);
    if (moved !== void 0) this.files.splice(to, 0, moved);
    return this.snapshot();
  }
};
function normalizeMediaFiles(value) {
  const values = Array.isArray(value) ? value : [value];
  return values.filter(
    (item) => typeof item === "string" && item.length > 0
  );
}
function validIndex(index, length) {
  return Number.isInteger(index) && index >= 0 && index < length;
}

// web/src/orderedMediaNode.ts
function configureOrderedMediaNode(candidate, app2, api, config, logger = console) {
  if (!isOrderedMediaNode(candidate, config.nodeId)) return;
  const imageWidget = findWidget(candidate, "image");
  const uploadWidget = findNativeUploadWidget(candidate);
  if (!imageWidget || !uploadWidget?.callback) return;
  hideInternalWidget(imageWidget);
  hideInternalWidget(uploadWidget);
  const selection = new OrderedMediaSelection(imageWidget.value);
  const nativePreview = new NativeNodePreview(app2, api, candidate);
  const preview = new OrderedMediaPreviewController(
    nativePreview,
    `${config.labels.singular} loader`,
    logger
  );
  let programmaticSelectionUpdate = false;
  let selectionIntent = "replace";
  let appendBase = [];
  let uploadPending = false;
  let ignoredUploadCallbackFiles;
  const nativeUploadCallback = uploadWidget.callback;
  const setPersistedFiles = (files) => {
    programmaticSelectionUpdate = true;
    try {
      imageWidget.value = [...files];
    } finally {
      programmaticSelectionUpdate = false;
    }
  };
  const refresh = (files) => {
    config.onSelectionChanged?.([...files]);
    preview.refresh(files, () => config.preview([...files], candidate));
  };
  const commit = (current, previous) => {
    setPersistedFiles(current);
    candidate.onWidgetChanged?.(
      imageWidget.name,
      [...current],
      [...previous],
      imageWidget
    );
    candidate.graph?.setDirtyCanvas?.(true, true);
    refresh(current);
  };
  const removeAt = (index) => {
    const previous = selection.snapshot();
    if (index < 0 || index >= previous.length) return;
    const current = selection.remove(index);
    commit(current, previous);
  };
  const moveTo = (index, destination) => {
    const previous = selection.snapshot();
    if (index < 0 || index >= previous.length || destination < 0 || destination >= previous.length || index === destination) {
      return;
    }
    const current = selection.move(index, destination);
    commit(current, previous);
  };
  const previewActions = new OrderedMediaPreviewActions({
    app: app2,
    node: candidate,
    preview: nativePreview,
    itemLabel: config.labels.singular,
    getFiles: () => selection.snapshot(),
    moveEarlier: (index) => {
      moveTo(index, index - 1);
    },
    moveLater: (index) => {
      moveTo(index, index + 1);
    },
    remove: removeAt
  });
  const beginReplace = () => {
    selectionIntent = "replace";
    appendBase = [];
    uploadPending = true;
  };
  const beginAppend = () => {
    selectionIntent = "append";
    appendBase = selection.snapshot();
    uploadPending = true;
  };
  nativeButton(candidate, "simple_syrup_replace_media", config.labels.replace, () => {
    beginReplace();
    nativeUploadCallback.call(uploadWidget);
  });
  nativeButton(candidate, "simple_syrup_add_media", config.labels.add, () => {
    beginAppend();
    nativeUploadCallback.call(uploadWidget);
  });
  uploadWidget.callback = (value) => {
    beginReplace();
    nativeUploadCallback.call(uploadWidget, value);
  };
  imageWidget.callback = (value) => {
    if (programmaticSelectionUpdate) return;
    const callbackValue = value ?? imageWidget.value;
    if (uploadPending && !Array.isArray(callbackValue)) return;
    const incoming = normalizeMediaFiles(callbackValue);
    if (ignoredUploadCallbackFiles && sameMediaFiles(incoming, ignoredUploadCallbackFiles)) {
      ignoredUploadCallbackFiles = void 0;
      return;
    }
    ignoredUploadCallbackFiles = uploadPending ? [...incoming] : void 0;
    uploadPending = false;
    const current = selectionIntent === "append" ? selection.replace([...appendBase, ...incoming]) : selection.replace(incoming);
    selectionIntent = "replace";
    appendBase = [];
    if (!sameMediaFiles(current, normalizeMediaFiles(imageWidget.value))) {
      setPersistedFiles(current);
    }
    candidate.graph?.setDirtyCanvas?.(true, true);
    refresh(current);
  };
  for (const widgetName of config.previewWidgetNames ?? []) {
    const widget = findWidget(candidate, widgetName);
    if (!widget) continue;
    const originalCallback = widget.callback;
    widget.callback = (value) => {
      originalCallback?.call(widget, value);
      if (value !== void 0) widget.value = value;
      refresh(selection.snapshot());
    };
  }
  const restoreExternalUploads = wrapExternalAppendUploads(candidate, beginAppend);
  const originalOnGraphConfigured = candidate.onGraphConfigured;
  candidate.onGraphConfigured = function(...args) {
    const configuredValue = imageWidget.value;
    const serializedValue = configuredWidgetValue(candidate, imageWidget);
    const result = originalOnGraphConfigured?.apply(this, args);
    const restored = selection.replace(
      Array.isArray(configuredValue) ? configuredValue : Array.isArray(serializedValue) ? serializedValue : imageWidget.value
    );
    if (!Array.isArray(imageWidget.value)) setPersistedFiles(restored);
    refresh(restored);
    return result;
  };
  const originalOnRemoved = candidate.onRemoved;
  candidate.onRemoved = function(...args) {
    restoreExternalUploads();
    previewActions.dispose();
    preview.dispose();
    return originalOnRemoved?.apply(this, args);
  };
  const initial = selection.snapshot();
  if (!Array.isArray(imageWidget.value)) setPersistedFiles(initial);
  refresh(initial);
}
function configuredWidgetValue(node, widget) {
  const index = node.widgets?.indexOf(widget) ?? -1;
  return index >= 0 ? node.widgets_values?.[index] : void 0;
}
function registerOrderedMediaNode(app2, api, extensionName, config, logger = console) {
  app2.registerExtension({
    name: extensionName,
    nodeCreated(candidate) {
      try {
        configureOrderedMediaNode(candidate, app2, api, config, logger);
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        logger.warn(
          `Could not configure ${config.labels.singular} loader: ${message}`,
          error
        );
        throw error;
      }
    }
  });
}
function nativeButton(node, name, label, callback) {
  const widget = node.addWidget("button", name, "ordered_media", callback, {
    serialize: false,
    tooltip: label
  });
  widget.label = label;
  return widget;
}
function findWidget(node, name) {
  return node.widgets?.find((widget) => widget.name === name);
}
function findNativeUploadWidget(node) {
  return node.widgets?.find(
    (widget) => widget.type === "button" && widget.value === "image" && widget.options?.serialize === false && widget.options.canvasOnly === true
  );
}
function hideInternalWidget(widget) {
  widget.options ??= {};
  widget.options.hidden = true;
  widget.hidden = true;
  widget.computeSize = () => [0, -4];
}
function sameMediaFiles(left, right) {
  return left.length === right.length && left.every((file, index) => file === right[index]);
}
function wrapExternalAppendUploads(node, beginAppend) {
  const originalPasteFiles = node.pasteFiles;
  const originalOnDragDrop = node.onDragDrop;
  const wrappedPasteFiles = originalPasteFiles ? (...args) => {
    beginAppend();
    return originalPasteFiles.apply(node, args);
  } : void 0;
  const wrappedOnDragDrop = originalOnDragDrop ? (...args) => {
    beginAppend();
    return originalOnDragDrop.apply(node, args);
  } : void 0;
  if (wrappedPasteFiles) node.pasteFiles = wrappedPasteFiles;
  if (wrappedOnDragDrop) node.onDragDrop = wrappedOnDragDrop;
  return () => {
    if (node.pasteFiles === wrappedPasteFiles) {
      if (originalPasteFiles) node.pasteFiles = originalPasteFiles;
      else delete node.pasteFiles;
    }
    if (node.onDragDrop === wrappedOnDragDrop) {
      if (originalOnDragDrop) node.onDragDrop = originalOnDragDrop;
      else delete node.onDragDrop;
    }
  };
}
function isOrderedMediaNode(candidate, nodeId) {
  if (typeof candidate !== "object" || candidate === null) return false;
  const node = candidate;
  return node.constructor?.comfyClass === nodeId && typeof node.addWidget === "function";
}

// web/src/maskBatchUpload.ts
var LOAD_MASK_BATCH_NODE_ID = "SimpleSyrup.LoadMaskBatch";
function registerMaskBatchUpload(app2, api, loadPreview = getMaskBatchPreview, logger = console) {
  registerOrderedMediaNode(
    app2,
    api,
    "SimpleSyrup.LoadMaskBatchUpload",
    maskConfig(loadPreview),
    logger
  );
}
function maskConfig(loadPreview) {
  return {
    nodeId: LOAD_MASK_BATCH_NODE_ID,
    labels: {
      singular: "mask",
      plural: "masks",
      replace: "Replace masks...",
      add: "Add masks..."
    },
    previewWidgetNames: ["channel"],
    preview: (files, node) => loadPreview(files, selectedChannel(node))
  };
}
function selectedChannel(node) {
  const value = node.widgets?.find((widget) => widget.name === "channel")?.value;
  return typeof value === "string" && value.length > 0 ? value : "alpha";
}

// web/src/comfyImageUrl.ts
function comfyImageUrl(reference, apiURL = (path) => path) {
  const query = new URLSearchParams({
    filename: reference.filename,
    subfolder: reference.subfolder,
    type: reference.type
  });
  return apiURL(`/view?${query.toString()}`);
}
function inputImageReference(path) {
  const annotated = path.trim().replace(/\s+\[input\]$/, "");
  const normalized = annotated.replaceAll("\\", "/");
  const separator = normalized.lastIndexOf("/");
  return {
    filename: normalized.slice(separator + 1),
    subfolder: separator >= 0 ? normalized.slice(0, separator) : "",
    type: "input"
  };
}

// web/src/imageListUpload.ts
function registerImageListUpload(app2, api, logger = console) {
  registerOrderedMediaNode(
    app2,
    api,
    "SimpleSyrup.LoadImageListUpload",
    {
      nodeId: "SimpleSyrup.LoadImageList",
      labels: {
        singular: "image",
        plural: "images",
        replace: "Replace images...",
        add: "Add images..."
      },
      preview: (files) => Promise.resolve({
        images: files.map(inputImageReference),
        animated: files.map(() => false)
      })
    },
    logger
  );
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

// web/src/domWidgetLayout.ts
var FixedDomWidgetLayout = class {
  constructor(node, widget, measure) {
    this.node = node;
    Object.defineProperty(widget, "computeLayoutSize", {
      configurable: true,
      value: void 0,
      writable: true
    });
    widget.computeSize = measure;
  }
  node;
  /** Recompute legacy widget allocation after the content mode changes. */
  reflow() {
    if (!this.node.graph) return;
    this.node.arrange?.();
    this.node.graph.setDirtyCanvas?.(true, true);
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
  constructor(inspectRegion = () => void 0, changeMode = () => void 0, apiURL = (path) => path, loadImage = loadPreviewImage) {
    this.inspectRegion = inspectRegion;
    this.changeMode = changeMode;
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
    this.updateMode();
    this.showMessage("Run the workflow to inspect SEGS.");
  }
  inspectRegion;
  changeMode;
  apiURL;
  loadImage;
  element = document.createElement("section");
  overlayButton = modeButton("Overlay");
  gridButton = modeButton("Grid");
  body = document.createElement("div");
  status = document.createElement("div");
  mode = "overlay";
  committed;
  unsubscribeSelection;
  highlightCanvas;
  /** Return the renderer height needed by the active preview surface. */
  preferredHeight() {
    return this.mode === "overlay" ? 360 : 38;
  }
  /** Select the overlay or native gallery and optionally notify its owner. */
  setMode(mode, notifyOwner = true) {
    if (mode === this.mode) {
      if (notifyOwner) this.changeMode(mode);
      return;
    }
    this.mode = mode;
    this.updateMode();
    this.renderMode();
    if (notifyOwner) this.changeMode(mode);
  }
  /** Show a loading state while execution assets are decoded. */
  setLoading() {
    this.showMessage("Loading SEGS preview\u2026");
  }
  /** Load source and atlas assets without publishing partial state. */
  async prepare(document2) {
    const [image, atlasImage] = await Promise.all([
      this.loadImage(comfyImageUrl(document2.preview.image, this.apiURL)),
      this.loadImage(comfyImageUrl(document2.atlas.image, this.apiURL))
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
  /** Show an actionable error without breaking the node lifecycle. */
  showError(message) {
    this.showMessage(message, true);
  }
  /** Release selection subscriptions and rendered state. */
  dispose() {
    this.unsubscribeSelection?.();
    this.unsubscribeSelection = void 0;
    this.committed = void 0;
    this.highlightCanvas = void 0;
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
  renderMode() {
    const committed = this.committed;
    if (!committed) return;
    if (this.mode === "overlay") {
      this.renderOverlay();
      return;
    }
    this.highlightCanvas = void 0;
    this.body.replaceChildren();
    this.status.textContent = `${String(committed.document.regions.length)} regions`;
  }
  renderOverlay() {
    const committed = required(this.committed);
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
      const selected = committed.selection.selectNextCandidate();
      const region = committed.document.regions.find(
        (candidate) => candidate.id === selected
      );
      if (region) {
        this.setMode("grid", false);
        this.inspectRegion(region.index);
      }
      committed.selection.select(null);
    });
    stack.append(base, highlight);
    this.body.replaceChildren(stack);
    this.renderSelection(committed.selection.state());
  }
  renderSelection(state) {
    const committed = this.committed;
    const highlight = this.highlightCanvas;
    if (!committed || !highlight) return;
    const active = state.active ? committed.document.regions.find((region) => region.id === state.active) : void 0;
    const context = context2d(highlight);
    context.clearRect(0, 0, highlight.width, highlight.height);
    if (active) {
      context.save();
      context.globalAlpha = 0.9;
      context.shadowColor = "rgba(255, 255, 255, 0.95)";
      context.shadowBlur = 8;
      drawRegionMask(context, committed, active);
      context.restore();
    }
    if (!active) {
      this.status.textContent = `${String(committed.document.regions.length)} regions`;
    } else if (state.candidates.length > 1) {
      this.status.textContent = `${regionTitle(active)} \xB7 ${String(state.candidates.length)} overlapping regions \xB7 click to inspect`;
    } else {
      this.status.textContent = `${regionTitle(active)} \xB7 click to inspect`;
    }
  }
  showMessage(message, error = false) {
    const content = document.createElement("div");
    content.className = "ss-segs-preview__message";
    content.dataset.error = String(error);
    content.textContent = message;
    this.body.replaceChildren(content);
    this.status.textContent = "";
  }
  updateMode() {
    this.element.dataset.mode = this.mode;
    this.overlayButton.setAttribute(
      "aria-pressed",
      String(this.mode === "overlay")
    );
    this.gridButton.setAttribute("aria-pressed", String(this.mode === "grid"));
  }
};
function modeButton(label) {
  const button = document.createElement("button");
  button.type = "button";
  button.textContent = label;
  return button;
}
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
    .ss-segs-preview { box-sizing: border-box; width: 100%; color: var(--fg-color, #ddd); font: 12px sans-serif; }
    .ss-segs-preview * { box-sizing: border-box; }
    .ss-segs-preview__toolbar { display: flex; gap: 4px; margin: 0 0 6px; }
    .ss-segs-preview__toolbar button { flex: 1; min-height: 26px; border: 1px solid var(--border-color, #555); border-radius: 5px; color: inherit; background: var(--comfy-input-bg, #222); cursor: pointer; }
    .ss-segs-preview__toolbar button[aria-pressed="true"] { border-color: var(--p-primary-color, #6aa9ff); background: color-mix(in srgb, var(--p-primary-color, #6aa9ff) 28%, var(--comfy-input-bg, #222)); }
    .ss-segs-preview__body { overflow: hidden; border: 1px solid var(--border-color, #444); border-radius: 6px; background: var(--comfy-menu-bg, #181818); }
    .ss-segs-preview[data-mode="grid"] .ss-segs-preview__body { display: none; }
    .ss-segs-preview[data-mode="grid"] .ss-segs-preview__status { display: none; }
    .ss-segs-preview__canvas-stack { position: relative; line-height: 0; background: #111; }
    .ss-segs-preview__canvas-stack canvas { display: block; width: 100%; height: auto; }
    .ss-segs-preview__highlight { position: absolute; inset: 0; cursor: crosshair; }
    .ss-segs-preview__status { min-height: 22px; padding: 5px 2px 0; color: var(--descrip-text, #aaa); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .ss-segs-preview__message { display: grid; min-height: 160px; place-items: center; padding: 18px; color: var(--descrip-text, #aaa); text-align: center; }
    .ss-segs-preview__message[data-error="true"] { color: var(--error-text, #ff8a80); }
  `;
  document.head.append(style);
  stylesInstalled = true;
}

// web/src/nativePreviewNavigator.ts
var NativePreviewNavigator = class {
  constructor(app2, node) {
    this.app = app2;
    this.node = node;
  }
  app;
  node;
  pendingReference;
  legacyInspectionFrame;
  unsubscribeLifecycle;
  /** Report whether Comfy currently owns a DOM-backed preview surface. */
  usesDomPreview() {
    return this.domRoot() !== null;
  }
  /** Inspect one zero-based native preview item in either node renderer. */
  inspect(index) {
    if (!Number.isInteger(index) || index < 0) return;
    if (this.domRoot()) {
      const images = this.images();
      if (!images || index >= images.length) return;
      const reference = images[index];
      if (this.clickDomImage(reference)) return;
      this.pendingReference = reference;
      this.unsubscribeLifecycle ??= subscribeNativePreviewLifecycle(() => {
        this.completePendingInspection();
      });
      return;
    }
    this.scheduleLegacyImage(index);
  }
  /** Wait for Comfy to restore canvas images before selecting native detail. */
  scheduleLegacyImage(index) {
    this.cancelLegacyInspection();
    const inspectWhenReady = () => {
      const images = this.node.imgs;
      if (images !== void 0 && images.length > index) {
        this.legacyInspectionFrame = void 0;
        this.openLegacyImage(index);
        return;
      }
      this.legacyInspectionFrame = requestAnimationFrame(inspectWhenReady);
    };
    this.legacyInspectionFrame = requestAnimationFrame(inspectWhenReady);
  }
  /** Open one loaded image through Comfy's canvas detail state. */
  openLegacyImage(index) {
    this.node.imageIndex = index;
    const position = this.app.canvas?.graph_mouse;
    if (position) {
      this.node.pointerDown = { index, pos: [position[0], position[1]] };
    }
    delete this.node.imageRects;
    this.node.graph?.setDirtyCanvas?.(true, true);
  }
  /** Return the native preview to its one authoritative gallery. */
  showGrid() {
    this.cancelLegacyInspection();
    this.pendingReference = void 0;
    if (this.clickDomGrid()) return;
    this.node.imageIndex = null;
    delete this.node.imageRects;
    this.node.graph?.setDirtyCanvas?.(true, true);
  }
  /** Cancel pending navigation when native images are hidden. */
  hide() {
    this.cancelLegacyInspection();
    this.pendingReference = void 0;
    this.node.imageIndex = null;
    delete this.node.imageRects;
    this.node.graph?.setDirtyCanvas?.(true, true);
  }
  /** Release native-preview lifecycle observation. */
  dispose() {
    this.cancelLegacyInspection();
    this.pendingReference = void 0;
    this.unsubscribeLifecycle?.();
    this.unsubscribeLifecycle = void 0;
  }
  cancelLegacyInspection() {
    if (this.legacyInspectionFrame === void 0) return;
    cancelAnimationFrame(this.legacyInspectionFrame);
    this.legacyInspectionFrame = void 0;
  }
  completePendingInspection() {
    const reference = this.pendingReference;
    if (!reference || !this.clickDomImage(reference)) return;
    this.pendingReference = void 0;
  }
  clickDomImage(reference) {
    const root = this.domRoot();
    if (!root) return false;
    const image = Array.from(root.querySelectorAll("img")).find(
      (candidate) => comfyImageSourceKey(candidate.src) === comfyImageReferenceKey(reference)
    );
    const button = image?.closest("button");
    if (!button) return false;
    button.click();
    return true;
  }
  clickDomGrid() {
    const root = this.domRoot();
    if (!root) return false;
    const button = Array.from(root.querySelectorAll("button")).find(
      (candidate) => candidate.getAttribute("aria-label") === "Grid view" || candidate.title === "Grid view"
    );
    if (!button) return false;
    button.click();
    return true;
  }
  domRoot() {
    const nodeId = this.nodeId();
    if (!nodeId) return null;
    return Array.from(document.querySelectorAll("[data-node-id]")).find(
      (element) => element.dataset.nodeId === nodeId
    ) ?? null;
  }
  images() {
    const nodeId = this.nodeId();
    return nodeId ? this.app.nodeOutputs?.[nodeId]?.images : void 0;
  }
  nodeId() {
    return this.node.id === void 0 ? void 0 : String(this.node.id);
  }
};

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

// web/src/segPreviewNativeSurface.ts
var SegPreviewNativeSurface = class {
  constructor(app2, api, node) {
    this.app = app2;
    this.node = node;
    this.preview = new NativeNodePreview(app2, api, node);
    this.navigator = new NativePreviewNavigator(app2, node);
  }
  app;
  node;
  preview;
  navigator;
  authoritativeOutput;
  mode = "overlay";
  /** Accept a complete backend result and apply the active surface mode. */
  update(value) {
    const output = executionOutput(value);
    if (!output) return;
    const document2 = parseSegPreviewDocument(value);
    if (!document2) return;
    if (output.images.length !== document2.regions.length) return;
    this.authoritativeOutput = output;
    if (this.mode === "overlay") {
      this.hideNativePreview();
    } else if (this.mode === "grid") {
      this.navigator.showGrid();
    }
  }
  /** Show only the custom semantic overlay. */
  showOverlay() {
    this.mode = "overlay";
    this.hideNativePreview();
  }
  /** Show Comfy's native gallery and return it from native detail mode. */
  showGrid() {
    this.mode = "grid";
    this.setCanvasPreviewVisible(true);
    if (this.nativeImagesNeedPublishing()) {
      this.publishAuthoritativeOutput();
    }
    this.navigator.showGrid();
  }
  /** Show one SEG through Comfy's native detail viewer. */
  inspect(index) {
    const output = this.authoritativeOutput;
    if (!output || index < 0 || index >= output.images.length) return;
    this.mode = "detail";
    this.setCanvasPreviewVisible(true);
    if (this.nativeImagesNeedPublishing()) {
      this.publishAuthoritativeOutput();
    }
    this.navigator.inspect(index);
  }
  /** Release native-preview observers owned by this node. */
  dispose() {
    this.navigator.dispose();
  }
  hideNativePreview() {
    const usesDomPreview = this.navigator.usesDomPreview();
    this.navigator.hide();
    this.setCanvasPreviewVisible(false);
    if (!usesDomPreview) return;
    this.node.imgs = [];
    const output = this.authoritativeOutput;
    if (!output || output.images.length === 0) return;
    this.preview.publish({
      ...output,
      images: [],
      animated: []
    });
  }
  /** Hide the persistent Nodes 1.0 preview widget outside native modes. */
  setCanvasPreviewVisible(visible) {
    const widget = this.node.widgets?.find(
      (candidate) => candidate.name === "$$canvas-image-preview"
    );
    if (!widget) return;
    widget.hidden = !visible;
    widget.options ??= {};
    widget.options.hidden = !visible;
  }
  publishAuthoritativeOutput() {
    const output = this.authoritativeOutput;
    if (!output) return;
    delete this.node.images;
    delete this.node.imgs;
    this.preview.publish({
      ...output,
      images: [...output.images]
    });
  }
  /** Check whether Comfy's current output still contains every SEG image. */
  hasAuthoritativeImages() {
    const output = this.authoritativeOutput;
    if (!output || this.node.id === void 0) return false;
    return this.app.nodeOutputs?.[String(this.node.id)]?.images?.length === output.images.length;
  }
  /** Decide whether the active renderer needs its native images republished. */
  nativeImagesNeedPublishing() {
    const output = this.authoritativeOutput;
    if (!output) return false;
    if (this.navigator.usesDomPreview()) return !this.hasAuthoritativeImages();
    return !loadedImagesMatchReferences(this.node.imgs, output.images);
  }
};
function executionOutput(value) {
  if (typeof value !== "object" || value === null) return void 0;
  const output = value;
  if (!Array.isArray(output.images) || !output.images.every(isImageResult)) {
    return void 0;
  }
  return output;
}
function isImageResult(value) {
  if (typeof value !== "object" || value === null) return false;
  const image = value;
  return typeof image.filename === "string" && typeof image.subfolder === "string" && (image.type === "input" || image.type === "output" || image.type === "temp");
}

// web/src/segPreviewNode.ts
var SIMPLE_PREVIEW_SEGS_NODE_ID = "SimpleSyrup.SimplePreviewSEGS";
function registerSimplePreviewSEGS(app2, api, loadImage, logger = console) {
  const controllers = /* @__PURE__ */ new Map();
  const nativeSurfaces = /* @__PURE__ */ new Map();
  const widgetLayouts = /* @__PURE__ */ new Map();
  const extension = {
    name: "SimpleSyrup.SimplePreviewSEGS",
    nodeCreated(candidate) {
      if (!isPreviewNode(candidate)) return;
      const nativeSurface = new SegPreviewNativeSurface(app2, api, candidate);
      const layoutOwner = {};
      const inspector = new SegPreviewInspector(
        (index) => {
          nativeSurface.inspect(index);
          layoutOwner.current?.reflow();
        },
        (mode) => {
          if (mode === "overlay") nativeSurface.showOverlay();
          else nativeSurface.showGrid();
          layoutOwner.current?.reflow();
        },
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
      const widgetLayout = new FixedDomWidgetLayout(candidate, widget, (width = 420) => [
        Math.max(360, width),
        inspector.preferredHeight()
      ]);
      layoutOwner.current = widgetLayout;
      const registerId = () => {
        if (candidate.id !== void 0) {
          const nodeId = String(candidate.id);
          controllers.set(nodeId, controller);
          nativeSurfaces.set(nodeId, nativeSurface);
          widgetLayouts.set(nodeId, widgetLayout);
        }
      };
      registerId();
      const originalExecuted = candidate.onExecuted;
      candidate.onExecuted = function(output) {
        originalExecuted?.call(this, output);
        controller.update(output);
        nativeSurface.update(output);
        widgetLayout.reflow();
      };
      const originalGraphConfigured = candidate.onGraphConfigured;
      candidate.onGraphConfigured = function(...args) {
        const result = originalGraphConfigured?.apply(this, args);
        registerId();
        if (candidate.id !== void 0) {
          const output = app2.nodeOutputs?.[String(candidate.id)];
          controller.update(output);
          nativeSurface.update(output);
          widgetLayout.reflow();
        }
        return result;
      };
      const originalRemoved = candidate.onRemoved;
      candidate.onRemoved = function(...args) {
        if (candidate.id !== void 0) {
          const nodeId = String(candidate.id);
          controllers.delete(nodeId);
          nativeSurfaces.delete(nodeId);
          widgetLayouts.delete(nodeId);
        }
        controller.dispose();
        nativeSurface.dispose();
        return originalRemoved?.apply(this, args);
      };
      const computed = candidate.computeSize?.();
      const current = candidate.size ?? computed;
      if (current && candidate.setSize) {
        candidate.setSize([
          Math.max(420, current[0]),
          Math.max(420, computed?.[1] ?? current[1])
        ]);
      }
    },
    onNodeOutputsUpdated(outputs) {
      for (const [nodeId, output] of Object.entries(outputs)) {
        controllers.get(nodeId)?.update(output);
        nativeSurfaces.get(nodeId)?.update(output);
        widgetLayouts.get(nodeId)?.reflow();
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
registerImageListUpload(comfyApp, comfyApi);
registerSimplePreviewSEGS(comfyApp, comfyApi);
