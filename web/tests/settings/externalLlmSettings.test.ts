// SimpleSyrup - workflow-focused ComfyUI extensions for image generation
// Copyright (C) 2026  Artificial Sweetener and contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

import { describe, expect, it, vi } from "vitest";

import { registerSimpleSyrupSettings } from "../../src/settingsRegistration";
import type { SimpleSyrupSettingsApi } from "../../src/settingsRegistration";
import { createFakeComfyApp } from "../support/testUtils";
import {
  configuredExternalLLMSettings,
  defaultExternalLLMSettings,
  fakeSettingsApi,
  flushPromises,
  getDefinition,
  renderSetting,
  requiredButton,
  requiredInput
} from "./settingsTestSupport";

describe("external LLM settings registration", () => {
  it("saves endpoint changes to the backend", async () => {
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
    const control = renderSetting(getDefinition(app, 2));
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
    const control = renderSetting(getDefinition(app, 2));
    const input = requiredInput(control, "input");
    const button = requiredButton(control, "button");
    input.value = "https://";
    button.click();
    await Promise.resolve();

    expect(saveExternalLLMSettings).not.toHaveBeenCalled();
    expect(input.value).toBe("https://");
  });

  it("saves API keys without showing the stored key", async () => {
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
    const control = renderSetting(getDefinition(app, 3));
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
    const control = renderSetting(getDefinition(app, 2));
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
    const control = renderSetting(getDefinition(app, 3));
    requiredButton(control, "button").click();

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
    const control = renderSetting(getDefinition(app, 3));
    requiredButton(control, "button").click();
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
    const control = renderSetting(getDefinition(app, 3));
    expect(control.querySelector("button")?.textContent).toBe("Replace API Key");
    expect(control.textContent).toContain("API key remembered.");
  });
});
