// SimpleSyrup - workflow-focused ComfyUI extensions for image generation
// Copyright (C) 2026  Artificial Sweetener and contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

// Owns the external LLM endpoint and credential settings presentation.

import {
  getExternalLLMSettings,
  saveExternalLLMApiKey,
  saveExternalLLMSettings
} from "./api";
import type { ExternalLLMSettings } from "./api";
import {
  createElement,
  installSimpleSyrupSettingsStyle,
  setPending
} from "./settingsUi";
import type { ComfyApp, Logger } from "./types";

export const EXTERNAL_LLM_ENDPOINT_SETTING_ID =
  "SimpleSyrup.ExternalLLM.Endpoint";
export const EXTERNAL_LLM_ENDPOINT_SETTING_LABEL =
  "SimpleSyrup: External LLM endpoint";
export const EXTERNAL_LLM_ENDPOINT_SETTING_DESCRIPTION =
  "OpenAI-compatible endpoint base URL used by SimpleSyrup prompt nodes.";
export const EXTERNAL_LLM_API_KEY_SETTING_ID =
  "SimpleSyrup.ExternalLLM.ApiKey";
export const EXTERNAL_LLM_API_KEY_SETTING_LABEL =
  "SimpleSyrup: External LLM API key";
export const EXTERNAL_LLM_API_KEY_SETTING_DESCRIPTION =
  "Stores the API key for the configured external LLM endpoint in OS credential storage.";

export interface ExternalLLMSettingsApi {
  getExternalLLMSettings(): Promise<ExternalLLMSettings>;
  saveExternalLLMSettings(
    settings: Pick<ExternalLLMSettings, "base_url" | "default_model">
  ): Promise<ExternalLLMSettings>;
  saveExternalLLMApiKey(settings: { api_key: string }): Promise<ExternalLLMSettings>;
}

export async function registerExternalLLMSettings(
  app: ComfyApp,
  api: ExternalLLMSettingsApi = {
    getExternalLLMSettings,
    saveExternalLLMSettings,
    saveExternalLLMApiKey
  },
  logger: Logger = console
): Promise<void> {
  let externalLLMSettings: ExternalLLMSettings = {
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

  app.ui.settings.addSetting({
    id: EXTERNAL_LLM_ENDPOINT_SETTING_ID,
    name: EXTERNAL_LLM_ENDPOINT_SETTING_LABEL,
    sortOrder: 320,
    type: () =>
      createExternalLLMEndpointControl({
        api,
        logger,
        refreshModelChoices: () =>
          refreshExternalLLMModelChoices(app, logger),
        getSettings: () => savedExternalLLMSettings,
        setSettings: (settings) => {
          savedExternalLLMSettings = settings;
        }
      }),
    defaultValue: externalLLMSettings.base_url,
    tooltip: EXTERNAL_LLM_ENDPOINT_SETTING_DESCRIPTION
  });

  app.ui.settings.addSetting({
    id: EXTERNAL_LLM_API_KEY_SETTING_ID,
    name: EXTERNAL_LLM_API_KEY_SETTING_LABEL,
    sortOrder: 319,
    type: () =>
      createExternalLLMApiKeyControl({
        api,
        logger,
        refreshModelChoices: () =>
          refreshExternalLLMModelChoices(app, logger),
        getSettings: () => savedExternalLLMSettings,
        setSettings: (settings) => {
          savedExternalLLMSettings = settings;
        }
      }),
    defaultValue: "",
    tooltip: EXTERNAL_LLM_API_KEY_SETTING_DESCRIPTION
  });
}

function endpointShouldBeSaved(value: string): boolean {
  const endpoint = value.trim();
  if (!endpoint) {
    return true;
  }
  try {
    const parsed = new URL(endpoint);
    return (
      (parsed.protocol === "http:" || parsed.protocol === "https:") &&
      parsed.hostname.length > 0
    );
  } catch {
    return false;
  }
}

interface ExternalLLMControlContext {
  api: ExternalLLMSettingsApi;
  logger: Logger;
  refreshModelChoices(): Promise<void>;
  getSettings(): ExternalLLMSettings;
  setSettings(settings: ExternalLLMSettings): void;
}

function createExternalLLMEndpointControl(
  context: ExternalLLMControlContext
): HTMLElement {
  const wrapper = createElement("div", "simple-syrup-settings-row");
  const input = createElement("input", "simple-syrup-settings-input");
  input.type = "text";
  input.value = context.getSettings().base_url;
  input.autocomplete = "off";

  const saveButton = createElement("button", "simple-syrup-settings-button");
  saveButton.type = "button";
  saveButton.textContent = "Save Endpoint";

  const status = createElement("span", "simple-syrup-settings-status");

  const saveEndpoint = async (): Promise<void> => {
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

function createExternalLLMApiKeyControl(
  context: ExternalLLMControlContext
): HTMLElement {
  const wrapper = createElement("div", "simple-syrup-settings-row");
  const button = createElement("button", "simple-syrup-settings-button");
  button.type = "button";
  const status = createElement("span", "simple-syrup-settings-status");

  const render = (): void => {
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
          status.textContent = context.getSettings().has_api_key
            ? "API key remembered."
            : "API key was not saved.";
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

function openExternalLLMApiKeyDialog(options: {
  replacing: boolean;
  onSubmit(apiKey: string): Promise<void>;
}): void {
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

  const close = (): void => {
    overlay.remove();
  };

  const submit = async (): Promise<void> => {
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

function errorMessage(error: unknown, fallback: string): string {
  if (error instanceof Error && error.message) {
    return error.message;
  }
  return fallback;
}

function apiKeyErrorMessage(error: unknown): string {
  const message = errorMessage(error, "API key was not saved.");
  if (message.includes("Configure an external LLM endpoint")) {
    return "Save endpoint first.";
  }
  return message;
}

async function refreshExternalLLMModelChoices(
  app: ComfyApp,
  logger: Logger
): Promise<void> {
  try {
    await app.refreshComboInNodes?.();
  } catch (error) {
    logger.warn(
      "Could not refresh Comfy node definitions after saving external LLM settings.",
      error
    );
  }
}
