// SimpleSyrup - workflow-focused ComfyUI extensions for image generation
// Copyright (C) 2026  Artificial Sweetener and contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

import type { SimpleSyrupSettings } from "./api";
import type { ComfyApp, Logger } from "./types";

export const SIMPLE_SYRUP_SETTING_ID = "SimpleSyrup.ShowDownloadableModels";
export const SIMPLE_SYRUP_SETTING_LABEL =
  "SimpleSyrup: Show downloadable models in loader dropdowns";
export const SIMPLE_SYRUP_SETTING_DESCRIPTION =
  "Show known downloadable SAM, GroundingDINO, and ViTMatte models even when they are not installed locally.";

export interface GeneralSettingsContext {
  getSettings(): SimpleSyrupSettings;
  saveSettings(settings: SimpleSyrupSettings): Promise<SimpleSyrupSettings>;
  setSettings(settings: SimpleSyrupSettings): void;
}

export function registerDownloadableModelsSetting(
  app: ComfyApp,
  context: GeneralSettingsContext,
  logger: Logger
): void {
  const setting = app.ui.settings.addSetting({
    id: SIMPLE_SYRUP_SETTING_ID,
    name: SIMPLE_SYRUP_SETTING_LABEL,
    type: "boolean",
    defaultValue: context.getSettings().show_downloadable_models,
    tooltip: SIMPLE_SYRUP_SETTING_DESCRIPTION,
    onChange: async (value: boolean) => {
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
