// SimpleSyrup - workflow-focused ComfyUI extensions for image generation
// Copyright (C) 2026  Artificial Sweetener and contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

import type { QuantCacheStatus } from "./api";
import type { ComfyApp, Logger } from "./types";
import type { GeneralSettingsContext } from "./downloadableModelsSetting";
import {
  createElement,
  installSimpleSyrupSettingsStyle,
  setPending
} from "./settingsUi";

export const QUANT_CACHE_SETTING_ID = "SimpleSyrup.QuantCache";
export const QUANT_CACHE_SETTING_LABEL = "SimpleSyrup: Quantized model cache";
export const QUANT_CACHE_SETTING_DESCRIPTION =
  "Sets the global models/SyrupQuants cache limit in GiB for every SimpleSyrup loader; least-recently-used inactive copies are removed automatically.";

export interface QuantCacheSettingsApi {
  getQuantCacheStatus(): Promise<QuantCacheStatus>;
  enforceQuantCacheLimit(): Promise<QuantCacheStatus>;
  clearQuantCache(): Promise<QuantCacheStatus>;
}

interface QuantCacheControlContext {
  settings: GeneralSettingsContext;
  api: QuantCacheSettingsApi;
  logger: Logger;
  initialStatus: QuantCacheStatus | null;
}

export async function registerQuantCacheSetting(
  app: ComfyApp,
  settings: GeneralSettingsContext,
  api: QuantCacheSettingsApi,
  logger: Logger
): Promise<void> {
  installSimpleSyrupSettingsStyle();
  let initialStatus: QuantCacheStatus | null = null;
  try {
    initialStatus = await api.getQuantCacheStatus();
  } catch (error) {
    logger.warn("Could not load SimpleSyrup quant cache status.", error);
  }
  app.ui.settings.addSetting({
    id: QUANT_CACHE_SETTING_ID,
    name: QUANT_CACHE_SETTING_LABEL,
    sortOrder: 321,
    type: () =>
      createQuantCacheControl({ settings, api, logger, initialStatus }),
    defaultValue: settings.getSettings().quant_cache_limit_gib,
    tooltip: QUANT_CACHE_SETTING_DESCRIPTION
  });
}

function createQuantCacheControl(context: QuantCacheControlContext): HTMLElement {
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

  const renderStatus = (cacheStatus: QuantCacheStatus | null): void => {
    status.textContent = cacheStatus
      ? `${formatGiB(cacheStatus.usage_bytes)} GiB used in ${cacheStatus.path} (${String(cacheStatus.artifact_count)} cached, ${String(cacheStatus.active_artifact_count)} active).`
      : "Cache status unavailable. Generated models are stored in models/SyrupQuants.";
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

  const saveLimit = async (): Promise<void> => {
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

  const clearInactive = async (): Promise<void> => {
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

function formatGiB(bytes: number): string {
  return (bytes / 1024 ** 3).toFixed(2);
}
