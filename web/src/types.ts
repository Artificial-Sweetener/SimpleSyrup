// SimpleSyrup - workflow-focused ComfyUI extensions for image generation
// Copyright (C) 2026  Artificial Sweetener and contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

export type Logger = Pick<Console, "warn">;

export type SettingValue = boolean | string | number | null;

export interface ComfySetting<TValue extends SettingValue> {
  value: TValue;
}

export interface ComfySettingDefinition<TValue extends SettingValue> {
  id: string;
  name: string;
  type: "boolean" | "text" | (() => HTMLElement);
  defaultValue: TValue;
  sortOrder?: number;
  tooltip?: string;
  onChange?: (value: TValue) => void | Promise<void>;
}

export interface ComfySettingsApi {
  addSetting<TValue extends SettingValue>(
    definition: ComfySettingDefinition<TValue>
  ): ComfySetting<TValue>;
}

export interface ComfyApp {
  ui: {
    settings: ComfySettingsApi;
  };
  refreshComboInNodes?: () => Promise<void>;
  registerExtension(extension: ComfyExtension): void;
}

export interface ComfyExtension {
  name: string;
  setup(app: ComfyApp): void | Promise<void>;
}
