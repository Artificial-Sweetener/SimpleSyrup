// SimpleSyrup - workflow-focused ComfyUI extensions for image generation
// Copyright (C) 2026  Artificial Sweetener and contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

import { describe, expect, it, vi } from "vitest";

import {
  SIMPLE_SYRUP_SETTING_ID,
  SIMPLE_SYRUP_SETTING_LABEL,
  registerSimpleSyrupSettings
} from "../src/settings";
import type { SimpleSyrupSettingsApi } from "../src/settings";
import { createFakeComfyApp } from "./testUtils";

describe("Comfy settings registration", () => {
  it("registers the ShowDownloadableModels setting from backend state", async () => {
    const app = createFakeComfyApp();
    const api = fakeSettingsApi(false);

    await registerSimpleSyrupSettings(app, api);

    expect(app.ui.settings.definitions).toHaveLength(1);
    expect(app.ui.settings.definitions[0]).toMatchObject({
      id: SIMPLE_SYRUP_SETTING_ID,
      name: SIMPLE_SYRUP_SETTING_LABEL,
      type: "boolean",
      defaultValue: false
    });
    expect(app.ui.settings.settings[0]?.value).toBe(false);
  });

  it("saves setting changes to the backend", async () => {
    const app = createFakeComfyApp();
    const saveSettings = vi
      .fn<SimpleSyrupSettingsApi["saveSettings"]>()
      .mockResolvedValue({ show_downloadable_models: true });
    const api: SimpleSyrupSettingsApi = {
      getSettings: vi.fn().mockResolvedValue({ show_downloadable_models: false }),
      saveSettings
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
      saveSettings: vi.fn().mockResolvedValue({ show_downloadable_models: true })
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
      saveSettings
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
});

function fakeSettingsApi(
  showDownloadableModels: boolean
): SimpleSyrupSettingsApi {
  return {
    getSettings: vi.fn().mockResolvedValue({
      show_downloadable_models: showDownloadableModels
    }),
    saveSettings: vi.fn().mockImplementation((settings) => Promise.resolve(settings))
  };
}
