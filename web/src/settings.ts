// SimpleSyrup - workflow-focused ComfyUI extensions for image generation
// Copyright (C) 2026  Artificial Sweetener and contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

import { getSettings, saveSettings } from "./api";
import type { SimpleSyrupSettings } from "./api";
import type { ComfyApp, Logger } from "./types";

export const SIMPLE_SYRUP_SETTING_ID = "SimpleSyrup.ShowDownloadableModels";
export const SIMPLE_SYRUP_SETTING_LABEL =
  "SimpleSyrup: Show downloadable models in loader dropdowns";
export const SIMPLE_SYRUP_SETTING_DESCRIPTION =
  "Show known downloadable SAM, GroundingDINO, and ViTMatte models even when they are not installed locally.";

const DEFAULT_SETTINGS: SimpleSyrupSettings = {
  show_downloadable_models: true
};

export interface SimpleSyrupSettingsApi {
  getSettings(): Promise<SimpleSyrupSettings>;
  saveSettings(settings: SimpleSyrupSettings): Promise<SimpleSyrupSettings>;
}

export async function registerSimpleSyrupSettings(
  app: ComfyApp,
  api: SimpleSyrupSettingsApi = { getSettings, saveSettings },
  logger: Logger = console
): Promise<void> {
  let initialSettings = DEFAULT_SETTINGS;

  try {
    initialSettings = await api.getSettings();
  } catch (error) {
    logger.warn(
      "Could not load SimpleSyrup settings. Using the default setting until the backend is available.",
      error
    );
  }
  let savedSettings = initialSettings;

  const setting = app.ui.settings.addSetting({
    id: SIMPLE_SYRUP_SETTING_ID,
    name: SIMPLE_SYRUP_SETTING_LABEL,
    type: "boolean",
    defaultValue: initialSettings.show_downloadable_models,
    tooltip: SIMPLE_SYRUP_SETTING_DESCRIPTION,
    onChange: async (value: boolean) => {
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
}
