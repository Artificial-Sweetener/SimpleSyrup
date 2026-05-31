// SimpleSyrup - workflow-focused ComfyUI extensions for image generation
// Copyright (C) 2026  Artificial Sweetener and contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

import { describe, expect, it, vi } from "vitest";

import {
  SIMPLE_SYRUP_SETTING_ID,
  SIMPLE_SYRUP_SETTING_LABEL,
  EXTERNAL_LLM_API_KEY_SETTING_ID,
  EXTERNAL_LLM_ENDPOINT_SETTING_ID,
  registerSimpleSyrupSettings
} from "../src/settings";
import type { SimpleSyrupSettingsApi } from "../src/settings";
import { createFakeComfyApp } from "./testUtils";

describe("Comfy settings registration", () => {
  it("registers the ShowDownloadableModels setting from backend state", async () => {
    const app = createFakeComfyApp();
    const api = fakeSettingsApi(false);

    await registerSimpleSyrupSettings(app, api);

    expect(app.ui.settings.definitions).toHaveLength(3);
    expect(app.ui.settings.definitions[0]).toMatchObject({
      id: SIMPLE_SYRUP_SETTING_ID,
      name: SIMPLE_SYRUP_SETTING_LABEL,
      type: "boolean",
      defaultValue: false
    });
    expect(app.ui.settings.settings[0]?.value).toBe(false);
    expect(app.ui.settings.definitions[1]).toMatchObject({
      id: EXTERNAL_LLM_ENDPOINT_SETTING_ID,
      sortOrder: 320
    });
    expect(typeof app.ui.settings.definitions[1]?.type).toBe("function");
    expect(app.ui.settings.definitions[2]).toMatchObject({
      id: EXTERNAL_LLM_API_KEY_SETTING_ID,
      sortOrder: 319
    });
    expect(typeof app.ui.settings.definitions[2]?.type).toBe("function");
  });

  it("saves setting changes to the backend", async () => {
    const app = createFakeComfyApp();
    const saveSettings = vi
      .fn<SimpleSyrupSettingsApi["saveSettings"]>()
      .mockResolvedValue({ show_downloadable_models: true });
    const api: SimpleSyrupSettingsApi = {
      getSettings: vi.fn().mockResolvedValue({ show_downloadable_models: false }),
      saveSettings,
      getExternalLLMSettings: vi.fn().mockResolvedValue(defaultExternalLLMSettings()),
      saveExternalLLMSettings: vi.fn().mockResolvedValue(defaultExternalLLMSettings()),
      saveExternalLLMApiKey: vi.fn().mockResolvedValue(defaultExternalLLMSettings())
    };

    await registerSimpleSyrupSettings(app, api);
    await app.ui.settings.definitions[0]?.onChange?.(true);

    expect(saveSettings).toHaveBeenCalledWith({
      show_downloadable_models: true
    });
    expect(app.ui.settings.settings[0]?.value).toBe(true);
  });

  it("falls back to the default and warns when backend load fails", async () => {
    const app = createFakeComfyApp();
    const logger = { warn: vi.fn() };
    const api: SimpleSyrupSettingsApi = {
      getSettings: vi.fn().mockRejectedValue(new Error("offline")),
      saveSettings: vi.fn().mockResolvedValue({ show_downloadable_models: true }),
      getExternalLLMSettings: vi.fn().mockResolvedValue(defaultExternalLLMSettings()),
      saveExternalLLMSettings: vi.fn().mockResolvedValue(defaultExternalLLMSettings()),
      saveExternalLLMApiKey: vi.fn().mockResolvedValue(defaultExternalLLMSettings())
    };

    await registerSimpleSyrupSettings(app, api, logger);

    expect(app.ui.settings.settings[0]?.value).toBe(true);
    expect(logger.warn).toHaveBeenCalledWith(
      expect.stringContaining("Could not load SimpleSyrup settings"),
      expect.any(Error)
    );
  });

  it("warns and restores the previous value when backend save fails", async () => {
    const app = createFakeComfyApp();
    const logger = { warn: vi.fn() };
    const saveSettings = vi
      .fn<SimpleSyrupSettingsApi["saveSettings"]>()
      .mockResolvedValueOnce({ show_downloadable_models: true })
      .mockRejectedValueOnce(new Error("rejected"));
    const api: SimpleSyrupSettingsApi = {
      getSettings: vi.fn().mockResolvedValue({ show_downloadable_models: false }),
      saveSettings,
      getExternalLLMSettings: vi.fn().mockResolvedValue(defaultExternalLLMSettings()),
      saveExternalLLMSettings: vi.fn().mockResolvedValue(defaultExternalLLMSettings()),
      saveExternalLLMApiKey: vi.fn().mockResolvedValue(defaultExternalLLMSettings())
    };

    await registerSimpleSyrupSettings(app, api, logger);
    await app.ui.settings.definitions[0]?.onChange?.(true);
    await app.ui.settings.definitions[0]?.onChange?.(false);

    expect(logger.warn).toHaveBeenCalledWith(
      expect.stringContaining("Could not save SimpleSyrup settings"),
      expect.any(Error)
    );
    expect(app.ui.settings.settings[0]?.value).toBe(true);
  });

  it("saves external LLM endpoint changes to the backend", async () => {
    const app = createFakeComfyApp();
    const refreshComboInNodes = vi.fn().mockResolvedValue(undefined);
    app.refreshComboInNodes = refreshComboInNodes;
    const saveExternalLLMSettings =
      vi.fn<SimpleSyrupSettingsApi["saveExternalLLMSettings"]>().mockResolvedValue({
        ...defaultExternalLLMSettings(),
        base_url: "https://provider.example/v1"
      });
    const api = fakeSettingsApi(true);
    api.saveExternalLLMSettings = saveExternalLLMSettings;

    await registerSimpleSyrupSettings(app, api);
    const control = renderSetting(getDefinition(app, 1));
    const input = requiredInput(control, "input");
    const button = requiredButton(control, "button");

    input.value = "https://provider.example/v1";
    button.click();
    await flushPromises();

    expect(saveExternalLLMSettings).toHaveBeenCalledWith({
      base_url: "https://provider.example/v1",
      default_model: ""
    });
    expect(refreshComboInNodes).toHaveBeenCalledOnce();
    expect(control.textContent).toContain("Endpoint saved.");
  });

  it("keeps partial endpoint text visible while the user is typing", async () => {
    const app = createFakeComfyApp();
    const saveExternalLLMSettings =
      vi.fn<SimpleSyrupSettingsApi["saveExternalLLMSettings"]>();
    const api = fakeSettingsApi(true);
    api.saveExternalLLMSettings = saveExternalLLMSettings;

    await registerSimpleSyrupSettings(app, api);
    const control = renderSetting(getDefinition(app, 1));
    const input = requiredInput(control, "input");
    const button = requiredButton(control, "button");

    input.value = "https://";
    button.click();
    await Promise.resolve();

    expect(saveExternalLLMSettings).not.toHaveBeenCalled();
    expect(input.value).toBe("https://");
  });

  it("saves API keys through an add dialog without showing the stored key", async () => {
    const app = createFakeComfyApp();
    const refreshComboInNodes = vi.fn().mockResolvedValue(undefined);
    app.refreshComboInNodes = refreshComboInNodes;
    const saveExternalLLMApiKey =
      vi.fn<SimpleSyrupSettingsApi["saveExternalLLMApiKey"]>().mockResolvedValue({
        ...configuredExternalLLMSettings(),
        has_api_key: true
      });
    const api = fakeSettingsApi(true);
    api.getExternalLLMSettings = vi
      .fn()
      .mockResolvedValue(configuredExternalLLMSettings());
    api.saveExternalLLMApiKey = saveExternalLLMApiKey;

    await registerSimpleSyrupSettings(app, api);
    const control = renderSetting(getDefinition(app, 2));
    const addButton = requiredButton(control, "button");
    expect(addButton.textContent).toBe("Add API Key");

    addButton.click();
    const dialogInput = requiredInput(document.body, ".simple-syrup-dialog input");
    const submitButton = requiredButton(
      document.body,
      ".simple-syrup-dialog button"
    );

    dialogInput.value = "secret";
    submitButton.click();
    await flushPromises();

    expect(saveExternalLLMApiKey).toHaveBeenCalledWith({ api_key: "secret" });
    expect(refreshComboInNodes).toHaveBeenCalledOnce();
    expect(document.body.querySelector(".simple-syrup-dialog")).toBeNull();
    expect(control.textContent).toContain("API key remembered.");
    expect(control.textContent).not.toContain("secret");
  });

  it("keeps save success visible if model choice refresh fails", async () => {
    const app = createFakeComfyApp();
    const logger = { warn: vi.fn() };
    app.refreshComboInNodes = vi.fn().mockRejectedValue(new Error("offline"));
    const saveExternalLLMSettings =
      vi.fn<SimpleSyrupSettingsApi["saveExternalLLMSettings"]>().mockResolvedValue({
        ...defaultExternalLLMSettings(),
        base_url: "https://provider.example/v1"
      });
    const api = fakeSettingsApi(true);
    api.saveExternalLLMSettings = saveExternalLLMSettings;

    await registerSimpleSyrupSettings(app, api, logger);
    const control = renderSetting(getDefinition(app, 1));
    const input = requiredInput(control, "input");
    const button = requiredButton(control, "button");

    input.value = "https://provider.example/v1";
    button.click();
    await flushPromises();

    expect(control.textContent).toContain("Endpoint saved.");
    expect(logger.warn).toHaveBeenCalledWith(
      expect.stringContaining("Could not refresh Comfy node definitions"),
      expect.any(Error)
    );
  });

  it("requires a saved endpoint before opening the API key dialog", async () => {
    const app = createFakeComfyApp();
    const api = fakeSettingsApi(true);
    const saveExternalLLMApiKey = vi.fn<
      SimpleSyrupSettingsApi["saveExternalLLMApiKey"]
    >();
    api.saveExternalLLMApiKey = saveExternalLLMApiKey;

    await registerSimpleSyrupSettings(app, api);
    const control = renderSetting(getDefinition(app, 2));
    const addButton = requiredButton(control, "button");

    addButton.click();

    expect(document.body.querySelector(".simple-syrup-dialog")).toBeNull();
    expect(control.textContent).toContain("Save endpoint first.");
    expect(saveExternalLLMApiKey).not.toHaveBeenCalled();
  });

  it("shows backend API key save errors in the setting row", async () => {
    const app = createFakeComfyApp();
    const api = fakeSettingsApi(true);
    api.getExternalLLMSettings = vi
      .fn()
      .mockResolvedValue(configuredExternalLLMSettings());
    api.saveExternalLLMApiKey = vi
      .fn<SimpleSyrupSettingsApi["saveExternalLLMApiKey"]>()
      .mockRejectedValue(new Error("Configure an external LLM endpoint first."));

    await registerSimpleSyrupSettings(app, api);
    const control = renderSetting(getDefinition(app, 2));
    const addButton = requiredButton(control, "button");

    addButton.click();
    const dialogInput = requiredInput(document.body, ".simple-syrup-dialog input");
    const submitButton = requiredButton(
      document.body,
      ".simple-syrup-dialog button"
    );

    dialogInput.value = "secret";
    submitButton.click();
    await Promise.resolve();
    await Promise.resolve();

    expect(control.textContent).toContain("Save endpoint first.");
    expect(control.textContent).not.toContain("secret");
  });

  it("offers replacement when an API key is remembered", async () => {
    const app = createFakeComfyApp();
    const api = fakeSettingsApi(true);
    api.getExternalLLMSettings = vi
      .fn()
      .mockResolvedValue({ ...defaultExternalLLMSettings(), has_api_key: true });

    await registerSimpleSyrupSettings(app, api);
    const control = renderSetting(getDefinition(app, 2));
    const button = control.querySelector("button");

    expect(button?.textContent).toBe("Replace API Key");
    expect(control.textContent).toContain("API key remembered.");
  });
});

function fakeSettingsApi(
  showDownloadableModels: boolean
): SimpleSyrupSettingsApi {
  return {
    getSettings: vi.fn().mockResolvedValue({
      show_downloadable_models: showDownloadableModels
    }),
    saveSettings: vi.fn().mockImplementation((settings) => Promise.resolve(settings)),
    getExternalLLMSettings: vi.fn().mockResolvedValue(defaultExternalLLMSettings()),
    saveExternalLLMSettings: vi
      .fn()
      .mockImplementation((settings) =>
        Promise.resolve({ ...defaultExternalLLMSettings(), ...settings })
      ),
    saveExternalLLMApiKey: vi.fn().mockResolvedValue(defaultExternalLLMSettings())
  };
}

function defaultExternalLLMSettings() {
  return {
    base_url: "",
    cached_models: [],
    default_model: "",
    has_api_key: false
  };
}

function configuredExternalLLMSettings() {
  return {
    base_url: "https://provider.example/v1",
    cached_models: [],
    default_model: "",
    has_api_key: false
  };
}

function renderSetting(definition: {
  type: "boolean" | "text" | (() => HTMLElement);
}): HTMLElement {
  expect(typeof definition.type).toBe("function");
  return (definition.type as () => HTMLElement)();
}

function getDefinition(app: ReturnType<typeof createFakeComfyApp>, index: number) {
  const definition = app.ui.settings.definitions[index];
  if (!definition) {
    throw new Error(`Expected setting definition at index ${String(index)}.`);
  }
  return definition;
}

function requiredInput(parent: ParentNode, selector: string): HTMLInputElement {
  const input = parent.querySelector(selector);
  if (!(input instanceof HTMLInputElement)) {
    throw new Error(`Expected input for selector ${selector}.`);
  }
  return input;
}

function requiredButton(parent: ParentNode, selector: string): HTMLButtonElement {
  const button = parent.querySelector(selector);
  if (!(button instanceof HTMLButtonElement)) {
    throw new Error(`Expected button for selector ${selector}.`);
  }
  return button;
}

async function flushPromises(): Promise<void> {
  for (let index = 0; index < 6; index += 1) {
    await Promise.resolve();
  }
}
