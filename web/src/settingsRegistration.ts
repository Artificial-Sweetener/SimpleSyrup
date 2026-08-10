// SimpleSyrup - workflow-focused ComfyUI extensions for image generation
// Copyright (C) 2026  Artificial Sweetener and contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

import {
  clearQuantCache,
  enforceQuantCacheLimit,
  getExternalLLMSettings,
  getQuantCacheStatus,
  getSettings,
  saveExternalLLMApiKey,
  saveExternalLLMSettings,
  saveSettings
} from "./api";
import type {
  ExternalLLMSettings,
  QuantCacheStatus,
  SimpleSyrupSettings
} from "./api";
import {
  registerDownloadableModelsSetting,
  type GeneralSettingsContext
} from "./downloadableModelsSetting";
import {
  registerExternalLLMSettings,
  type ExternalLLMSettingsApi
} from "./externalLlmSettings";
import {
  registerQuantCacheSetting,
  type QuantCacheSettingsApi
} from "./quantCacheSetting";
import type { ComfyApp, Logger } from "./types";

const DEFAULT_SETTINGS: SimpleSyrupSettings = {
  show_downloadable_models: true,
  quant_cache_limit_gib: 20
};

export interface SimpleSyrupSettingsApi
  extends ExternalLLMSettingsApi,
    QuantCacheSettingsApi {
  getSettings(): Promise<SimpleSyrupSettings>;
  saveSettings(settings: SimpleSyrupSettings): Promise<SimpleSyrupSettings>;
}

export async function registerSimpleSyrupSettings(
  app: ComfyApp,
  api: SimpleSyrupSettingsApi = defaultApi(),
  logger: Logger = console
): Promise<void> {
  let savedSettings = DEFAULT_SETTINGS;
  try {
    savedSettings = await api.getSettings();
  } catch (error) {
    logger.warn(
      "Could not load SimpleSyrup settings. Using defaults until the backend is available.",
      error
    );
  }

  const settingsContext: GeneralSettingsContext = {
    getSettings: () => savedSettings,
    saveSettings: (settings) => api.saveSettings(settings),
    setSettings: (settings) => {
      savedSettings = settings;
    }
  };
  registerDownloadableModelsSetting(app, settingsContext, logger);
  await registerQuantCacheSetting(app, settingsContext, api, logger);
  await registerExternalLLMSettings(app, api, logger);
}

function defaultApi(): SimpleSyrupSettingsApi {
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

export type { ExternalLLMSettings, QuantCacheStatus, SimpleSyrupSettings };
