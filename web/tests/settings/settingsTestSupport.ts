// SimpleSyrup - workflow-focused ComfyUI extensions for image generation
// Copyright (C) 2026  Artificial Sweetener and contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

import { expect, vi } from "vitest";

import type { SimpleSyrupSettingsApi } from "../../src/settingsRegistration";
import type { createFakeComfyApp } from "../support/testUtils";

export function fakeSettingsApi(
  showDownloadableModels: boolean
): SimpleSyrupSettingsApi {
  return {
    getSettings: vi.fn().mockResolvedValue({
      show_downloadable_models: showDownloadableModels,
      quant_cache_limit_gib: 20
    }),
    saveSettings: vi.fn().mockImplementation((settings) => Promise.resolve(settings)),
    getQuantCacheStatus: vi.fn().mockResolvedValue(defaultQuantCacheStatus()),
    enforceQuantCacheLimit: vi.fn().mockResolvedValue(defaultQuantCacheStatus()),
    clearQuantCache: vi.fn().mockResolvedValue(defaultQuantCacheStatus()),
    getExternalLLMSettings: vi.fn().mockResolvedValue(defaultExternalLLMSettings()),
    saveExternalLLMSettings: vi
      .fn()
      .mockImplementation((settings) =>
        Promise.resolve({ ...defaultExternalLLMSettings(), ...settings })
      ),
    saveExternalLLMApiKey: vi.fn().mockResolvedValue(defaultExternalLLMSettings())
  };
}

export function defaultQuantCacheStatus() {
  return {
    path: "models/SyrupQuants",
    usage_bytes: 0,
    limit_bytes: 20 * 1024 ** 3,
    artifact_count: 0,
    active_artifact_count: 0
  };
}

export function defaultExternalLLMSettings() {
  return {
    base_url: "",
    cached_models: [],
    default_model: "",
    has_api_key: false
  };
}

export function configuredExternalLLMSettings() {
  return {
    base_url: "https://provider.example/v1",
    cached_models: [],
    default_model: "",
    has_api_key: false
  };
}

export function renderSetting(definition: {
  type: "boolean" | "text" | (() => HTMLElement);
}): HTMLElement {
  expect(typeof definition.type).toBe("function");
  return (definition.type as () => HTMLElement)();
}

export function getDefinition(
  app: ReturnType<typeof createFakeComfyApp>,
  index: number
) {
  const definition = app.ui.settings.definitions[index];
  if (!definition) {
    throw new Error(`Expected setting definition at index ${String(index)}.`);
  }
  return definition;
}

export function requiredInput(
  parent: ParentNode,
  selector: string
): HTMLInputElement {
  const input = parent.querySelector(selector);
  if (!(input instanceof HTMLInputElement)) {
    throw new Error(`Expected input for selector ${selector}.`);
  }
  return input;
}

export function requiredButton(
  parent: ParentNode,
  selector: string
): HTMLButtonElement {
  const button = parent.querySelector(selector);
  if (!(button instanceof HTMLButtonElement)) {
    throw new Error(`Expected button for selector ${selector}.`);
  }
  return button;
}

export async function flushPromises(): Promise<void> {
  for (let index = 0; index < 6; index += 1) {
    await Promise.resolve();
  }
}
