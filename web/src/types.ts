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
  nodeOutputs?: Record<string, ComfyNodeExecutionOutput>;
  canvas?: ComfyCanvasApi;
  ui: {
    settings: ComfySettingsApi;
  };
  refreshComboInNodes?: () => Promise<void>;
  registerExtension(extension: ComfyExtension): void;
}

/** Stable coordinate operations exposed by Comfy's graph canvas. */
export interface ComfyCanvasApi {
  canvas: HTMLCanvasElement;
  graph_mouse?: readonly [number, number];
  convertEventToCanvasOffset(event: MouseEvent): [number, number];
  convertOffsetToCanvas(
    position: readonly [number, number]
  ): [number, number];
}

export interface ComfyApi extends ComfyExecutionEvents {
  apiURL?(path: string): string;
}

/** Native Comfy event target used to publish execution-shaped node output. */
export type ComfyExecutionEvents = Pick<EventTarget, "dispatchEvent">;

export interface ComfyImageResult {
  filename: string;
  subfolder: string;
  type: "input" | "output" | "temp";
}

export interface ComfyNodeExecutionOutput {
  images?: ComfyImageResult[];
  animated?: boolean[];
}

export interface ComfyExtension {
  name: string;
  setup?(app: ComfyApp): void | Promise<void>;
  nodeCreated?(node: unknown): void | Promise<void>;
  onNodeOutputsUpdated?(
    outputs: Record<string, ComfyNodeExecutionOutput>
  ): void | Promise<void>;
}
