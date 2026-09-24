// SimpleSyrup - workflow-focused ComfyUI extensions for image generation
// Copyright (C) 2026  Artificial Sweetener and contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

import { describe, expect, it, vi } from "vitest";

import {
  SIMPLE_SYRUP_SETTING_DESCRIPTION,
  SIMPLE_SYRUP_SETTING_ID,
  SIMPLE_SYRUP_SETTING_LABEL
} from "../../src/downloadableModelsSetting";
import {
  EXTERNAL_LLM_API_KEY_SETTING_ID,
  EXTERNAL_LLM_ENDPOINT_SETTING_ID
} from "../../src/externalLlmSettings";
import { QUANT_CACHE_SETTING_ID } from "../../src/quantCacheSetting";
import { registerSimpleSyrupSettings } from "../../src/settingsRegistration";
import type { SimpleSyrupSettingsApi } from "../../src/settingsRegistration";
import { createFakeComfyApp } from "../support/testUtils";
import {
  defaultExternalLLMSettings,
  defaultQuantCacheStatus,
  fakeSettingsApi,
  flushPromises,
  getDefinition,
  renderSetting,
  requiredButton,
  requiredInput
} from "./settingsTestSupport";

describe("general and quant-cache settings registration", () => {
  it("registers every setting from backend state", async () => {
    const app = createFakeComfyApp();
    await registerSimpleSyrupSettings(app, fakeSettingsApi(false));

    expect(app.ui.settings.definitions).toHaveLength(4);
    expect(app.ui.settings.definitions[0]).toMatchObject({
      id: SIMPLE_SYRUP_SETTING_ID,
      name: SIMPLE_SYRUP_SETTING_LABEL,
      type: "boolean",
      defaultValue: false,
      tooltip: SIMPLE_SYRUP_SETTING_DESCRIPTION
    });
    expect(SIMPLE_SYRUP_SETTING_DESCRIPTION).toContain("WD14 tagger");
    expect(SIMPLE_SYRUP_SETTING_DESCRIPTION).toContain("Ultralytics");
    expect(app.ui.settings.settings[0]?.value).toBe(false);
    expect(app.ui.settings.definitions[1]).toMatchObject({
      id: QUANT_CACHE_SETTING_ID,
      sortOrder: 321
    });
    expect(app.ui.settings.definitions[2]).toMatchObject({
      id: EXTERNAL_LLM_ENDPOINT_SETTING_ID,
      sortOrder: 320
    });
    expect(app.ui.settings.definitions[3]).toMatchObject({
      id: EXTERNAL_LLM_API_KEY_SETTING_ID,
      sortOrder: 319
    });
  });

  it("saves downloadable-model setting changes", async () => {
    const app = createFakeComfyApp();
    const refreshComboInNodes = vi.fn().mockResolvedValue(undefined);
    app.refreshComboInNodes = refreshComboInNodes;
    const saveSettings = vi
      .fn<SimpleSyrupSettingsApi["saveSettings"]>()
      .mockResolvedValue({
        show_downloadable_models: true,
        quant_cache_limit_gib: 20
      });
    const api = fakeSettingsApi(false);
    api.saveSettings = saveSettings;

    await registerSimpleSyrupSettings(app, api);
    await app.ui.settings.definitions[0]?.onChange?.(true);

    expect(saveSettings).toHaveBeenCalledWith({
      show_downloadable_models: true,
      quant_cache_limit_gib: 20
    });
    expect(app.ui.settings.settings[0]?.value).toBe(true);
    expect(refreshComboInNodes).toHaveBeenCalledOnce();
  });

  it("falls back to the default and warns when backend load fails", async () => {
    const app = createFakeComfyApp();
    const logger = { warn: vi.fn() };
    const api = fakeSettingsApi(true);
    api.getSettings = vi.fn().mockRejectedValue(new Error("offline"));
    api.getExternalLLMSettings = vi
      .fn()
      .mockResolvedValue(defaultExternalLLMSettings());

    await registerSimpleSyrupSettings(app, api, logger);

    expect(app.ui.settings.settings[0]?.value).toBe(true);
    expect(logger.warn).toHaveBeenCalledWith(
      expect.stringContaining("Could not load SimpleSyrup settings"),
      expect.any(Error)
    );
  });

  it("restores the previous downloadable-model value when save fails", async () => {
    const app = createFakeComfyApp();
    const logger = { warn: vi.fn() };
    const api = fakeSettingsApi(false);
    api.saveSettings = vi
      .fn<SimpleSyrupSettingsApi["saveSettings"]>()
      .mockResolvedValueOnce({
        show_downloadable_models: true,
        quant_cache_limit_gib: 20
      })
      .mockRejectedValueOnce(new Error("rejected"));

    await registerSimpleSyrupSettings(app, api, logger);
    await app.ui.settings.definitions[0]?.onChange?.(true);
    await app.ui.settings.definitions[0]?.onChange?.(false);

    expect(app.ui.settings.settings[0]?.value).toBe(true);
    expect(logger.warn).toHaveBeenCalledWith(
      expect.stringContaining("Could not save SimpleSyrup settings"),
      expect.any(Error)
    );
  });

  it("keeps a saved setting when live model-choice refresh fails", async () => {
    const app = createFakeComfyApp();
    const logger = { warn: vi.fn() };
    app.refreshComboInNodes = vi.fn().mockRejectedValue(new Error("offline"));

    await registerSimpleSyrupSettings(app, fakeSettingsApi(false), logger);
    await app.ui.settings.definitions[0]?.onChange?.(true);

    expect(app.ui.settings.settings[0]?.value).toBe(true);
    expect(logger.warn).toHaveBeenCalledWith(
      expect.stringContaining("Could not refresh Comfy loader model choices"),
      expect.any(Error)
    );
  });

  it("shows global quant cache usage and saves its GiB limit", async () => {
    const app = createFakeComfyApp();
    const api = fakeSettingsApi(true);
    const saveSettings = vi
      .fn<SimpleSyrupSettingsApi["saveSettings"]>()
      .mockResolvedValue({
        show_downloadable_models: true,
        quant_cache_limit_gib: 30
      });
    api.saveSettings = saveSettings;

    await registerSimpleSyrupSettings(app, api);
    const control = renderSetting(getDefinition(app, 1));
    const input = requiredInput(control, "input[type=number]");
    expect(input.value).toBe("20");
    expect(control.textContent).toContain("models/SyrupQuants");
    input.value = "30";
    requiredButton(control, "button").click();
    await flushPromises();

    expect(saveSettings).toHaveBeenCalledWith({
      show_downloadable_models: true,
      quant_cache_limit_gib: 30
    });
    expect(input.value).toBe("30");
  });

  it("clears inactive quant artifacts and refreshes cache status", async () => {
    const app = createFakeComfyApp();
    const api = fakeSettingsApi(true);
    const clearQuantCache = vi.fn().mockResolvedValue({
      ...defaultQuantCacheStatus(),
      removed_artifacts: 2,
      removed_bytes: 1024
    });
    api.clearQuantCache = clearQuantCache;

    await registerSimpleSyrupSettings(app, api);
    const control = renderSetting(getDefinition(app, 1));
    const clearButton = control.querySelectorAll("button")[1];
    if (!(clearButton instanceof HTMLButtonElement)) {
      throw new Error("Expected quant cache clear button.");
    }
    clearButton.click();
    await flushPromises();

    expect(clearQuantCache).toHaveBeenCalledOnce();
    expect(control.textContent).toContain("0.00 GiB used");
  });

  it("restores the previous quant limit when backend saving fails", async () => {
    const app = createFakeComfyApp();
    const logger = { warn: vi.fn() };
    const api = fakeSettingsApi(true);
    api.saveSettings = vi.fn().mockRejectedValue(new Error("rejected"));

    await registerSimpleSyrupSettings(app, api, logger);
    const control = renderSetting(getDefinition(app, 1));
    const input = requiredInput(control, "input[type=number]");
    input.value = "30";
    requiredButton(control, "button").click();
    await flushPromises();

    expect(input.value).toBe("20");
    expect(control.textContent).toContain("was not saved");
    expect(logger.warn).toHaveBeenCalledWith(
      expect.stringContaining("quant cache limit"),
      expect.any(Error)
    );
  });
});
