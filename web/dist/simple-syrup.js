// web/src/main.ts
import { app } from "../../../scripts/app.js";

// web/src/api.ts
var SETTINGS_ROUTE = "/simple-syrup/settings";
var EXTERNAL_LLM_SETTINGS_ROUTE = "/simple-syrup/external-llm/settings";
var EXTERNAL_LLM_API_KEY_ROUTE = "/simple-syrup/external-llm/api-key";
var EXTERNAL_LLM_MODELS_REFRESH_ROUTE = "/simple-syrup/external-llm/models/refresh";
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

// web/src/main.ts
var comfyApp = app;
comfyApp.registerExtension({
  name: "SimpleSyrup.Settings",
  async setup(appInstance) {
    await registerSimpleSyrupSettings(appInstance);
    registerExternalLLMRefreshHook(appInstance);
  }
});
