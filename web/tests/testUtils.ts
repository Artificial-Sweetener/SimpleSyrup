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
import { vi } from "vitest";

export interface FakeComfySettingsApi {
  definitions: Array<ComfySettingDefinition<SettingValue>>;
  settings: Array<ComfySetting<SettingValue>>;
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
        definition as unknown as ComfySettingDefinition<SettingValue>
      );
      this.settings.push(setting);
      return setting;
    }
  };

  return {
    extensions: [],
    ui: {
      settings: settingsApi
    },
    refreshComboInNodes: async (): Promise<void> => {},
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

/** Install a deterministic 2D canvas surface for DOM inspector tests. */
export function installCanvasMock(maskValue = 255): void {
  const context = {
    globalAlpha: 1,
    shadowColor: "",
    shadowBlur: 0,
    clearRect: vi.fn(),
    drawImage: vi.fn(),
    fillRect: vi.fn(),
    getImageData: vi.fn((x: number, y: number, width: number, height: number) => {
      void x;
      void y;
      const data = new Uint8ClampedArray(width * height * 4);
      for (let offset = 0; offset < data.length; offset += 4) {
        data[offset] = maskValue;
        data[offset + 1] = maskValue;
        data[offset + 2] = maskValue;
        data[offset + 3] = 255;
      }
      return { data, width, height, colorSpace: "srgb" };
    }),
    putImageData: vi.fn(),
    restore: vi.fn(),
    save: vi.fn()
  } as unknown as CanvasRenderingContext2D;
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(context);
  vi.stubGlobal(
    "ImageData",
    class {
      readonly colorSpace = "srgb";

      constructor(
        readonly data: Uint8ClampedArray,
        readonly width: number,
        readonly height: number
      ) {}
    }
  );
}
