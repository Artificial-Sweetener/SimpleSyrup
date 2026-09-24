// SimpleSyrup - workflow-focused ComfyUI extensions for image generation
// Copyright (C) 2026  Artificial Sweetener and contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

/** Provide focused ordered-media node fixtures shared across capability tests. */

import { vi } from "vitest";

import { configureOrderedMediaNode } from "../../src/orderedMediaNode";
import type {
  ComfyApi,
  ComfyApp,
  ComfyImageResult
} from "../../src/types";

export const CONFIG = {
  nodeId: "SimpleSyrup.LoadImageList",
  labels: {
    singular: "image",
    plural: "images",
    replace: "Replace images...",
    add: "Add images..."
  },
  preview: (files: string[]) =>
    Promise.resolve({
      images: references(...files),
      animated: files.map(() => false)
    })
};

interface TestWidget {
  name: string;
  value: unknown;
  type?: string;
  label?: string;
  hidden?: boolean;
  disabled?: boolean;
  callback?: (value?: unknown) => void;
  computeSize?: (width?: number) => [number, number];
  options?: {
    canvasOnly?: boolean;
    serialize?: boolean;
    tooltip?: string;
    hidden?: boolean;
  };
}

const activeNodes: Array<{ onRemoved: () => unknown }> = [];

export function resetOrderedMediaFixtures(): void {
  for (const node of activeNodes.splice(0)) node.onRemoved();
  document.body.replaceChildren();
}

export function configured(value: unknown, reactive = false) {
  const fixture = createFixture(value, reactive);
  configureOrderedMediaNode(fixture.node, fixture.app, fixture.api, CONFIG);
  return fixture;
}

export function createFixture(value: unknown, reactive = false) {
  const nativeUpload = vi.fn();
  const imageWidget: TestWidget = {
    name: "image",
    value,
    type: "combo",
    options: { canvasOnly: true }
  };
  if (reactive) {
    let stored = value;
    Object.defineProperty(imageWidget, "value", {
      configurable: true,
      get: () => stored,
      set: (next: unknown) => {
        stored = next;
        imageWidget.callback?.(next);
      }
    });
  }
  const uploadWidget: TestWidget = {
    name: "upload",
    value: "image",
    type: "button",
    callback: nativeUpload,
    options: { serialize: false, canvasOnly: true }
  };
  const widgets = [imageWidget, uploadWidget];
  const originalRemoved = vi.fn();
  const onWidgetChanged = vi.fn();
  const node = {
    constructor: { comfyClass: "SimpleSyrup.LoadImageList" },
    id: 7,
    widgets,
    widgets_values: undefined as unknown[] | undefined,
    imgs: undefined as HTMLImageElement[] | undefined,
    graph: { setDirtyCanvas: vi.fn() },
    pasteFiles: vi.fn(() => true) as (...args: unknown[]) => unknown,
    onDragDrop: vi.fn(() => true) as (...args: unknown[]) => unknown,
    onRemoved: originalRemoved as (...args: unknown[]) => unknown,
    onGraphConfigured: undefined as ((...args: unknown[]) => unknown) | undefined,
    onWidgetChanged,
    addWidget: vi.fn(
      (
        type: "button" | "combo",
        name: string,
        widgetValue: string,
        callback: (value?: unknown) => void,
        options?: TestWidget["options"]
      ): TestWidget => {
        const created: TestWidget = {
          type,
          name,
          value: widgetValue,
          callback
        };
        if (options) created.options = options;
        widgets.push(created);
        return created;
      }
    )
  };
  activeNodes.push(node);
  const app = {
    nodeOutputs: {},
    ui: { settings: { addSetting: vi.fn() } },
    registerExtension: vi.fn()
  } as unknown as ComfyApp;
  const events = new EventTarget();
  const api = events as ComfyApi;
  const executed = vi.fn();
  events.addEventListener("executed", (event: Event) => {
    executed((event as CustomEvent).detail);
  });
  return {
    api,
    app,
    executed,
    imageWidget,
    nativeUpload,
    node,
    onWidgetChanged,
    originalRemoved,
    uploadWidget
  };
}

export function nativePreviewImage(
  filename: string,
  left: number
): HTMLImageElement {
  const image = document.createElement("img");
  image.src = `/api/view?filename=${filename}&subfolder=&type=input&preview=webp`;
  Object.defineProperties(image, {
    complete: { configurable: true, value: true },
    naturalWidth: { configurable: true, value: 80 }
  });
  mockRect(image, left, 0, 80, 80);
  return image;
}

export function nativePreviewButton(
  filename: string,
  left: number
): HTMLButtonElement {
  const button = document.createElement("button");
  button.append(nativePreviewImage(filename, left));
  mockRect(button, left, 0, 80, 80);
  return button;
}

export function visiblePreviewFiles(root: HTMLElement): string[] {
  return Array.from(root.querySelectorAll<HTMLImageElement>("img"))
    .filter((image) => image.style.display !== "none")
    .map((image) => new URL(image.src).searchParams.get("filename") ?? "");
}

export function imageFiles(images: HTMLImageElement[] | undefined): string[] {
  return (images ?? []).map(
    (image) => new URL(image.src).searchParams.get("filename") ?? ""
  );
}

export function mockRect(
  element: Element,
  left: number,
  top: number,
  width: number,
  height: number
): void {
  element.getBoundingClientRect = () => new DOMRect(left, top, width, height);
}

export function widget(
  fixture: ReturnType<typeof createFixture>,
  name: string
): TestWidget {
  const found = fixture.node.widgets.find((candidate) => candidate.name === name);
  if (!found) throw new Error(`Missing test widget ${name}.`);
  return found;
}

export function selectFiles(
  fixture: ReturnType<typeof createFixture>,
  files: string[]
): void {
  fixture.imageWidget.value = files;
  fixture.imageWidget.callback?.(files);
}

export function references(...filenames: string[]): ComfyImageResult[] {
  return filenames.map((filename) => ({
    filename,
    subfolder: "",
    type: "input"
  }));
}

export function deferred<T>(): {
  readonly promise: Promise<T>;
  readonly resolve: (value: T) => void;
} {
  let resolvePromise: ((value: T) => void) | undefined;
  const promise = new Promise<T>((resolve) => {
    resolvePromise = resolve;
  });
  return {
    promise,
    resolve: (value) => {
      if (!resolvePromise) throw new Error("Deferred promise is unavailable.");
      resolvePromise(value);
    }
  };
}
