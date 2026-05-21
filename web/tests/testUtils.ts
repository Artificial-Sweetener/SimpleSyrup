// SimpleSyrup - workflow-focused ComfyUI extensions for image generation
// Copyright (C) 2026  Artificial Sweetener and contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

import type {
  ComfyApp,
  ComfyExtension,
  ComfySetting,
  ComfySettingDefinition,
  SettingValue
} from "../src/types";

export interface FakeComfySettingsApi {
  definitions: Array<ComfySettingDefinition<boolean>>;
  settings: Array<ComfySetting<boolean>>;
  addSetting<TValue extends SettingValue>(
    definition: ComfySettingDefinition<TValue>
  ): ComfySetting<TValue>;
}

export interface FakeComfyApp extends ComfyApp {
  extensions: ComfyExtension[];
  ui: {
    settings: FakeComfySettingsApi;
  };
}

export function createFakeComfyApp(): FakeComfyApp {
  const settingsApi: FakeComfySettingsApi = {
    definitions: [],
    settings: [],
    addSetting<TValue extends SettingValue>(
      definition: ComfySettingDefinition<TValue>
    ): ComfySetting<TValue> {
      const setting: ComfySetting<TValue> = {
        value: definition.defaultValue
      };
      this.definitions.push(
        definition as unknown as ComfySettingDefinition<boolean>
      );
      this.settings.push(setting as ComfySetting<boolean>);
      return setting;
    }
  };

  return {
    extensions: [],
    ui: {
      settings: settingsApi
    },
    registerExtension(extension: ComfyExtension): void {
      this.extensions.push(extension);
    }
  };
}

export function createJsonResponse(
  payload: unknown,
  init: ResponseInit = {}
): Response {
  return new Response(JSON.stringify(payload), {
    headers: { "Content-Type": "application/json" },
    ...init
  });
}
