// SimpleSyrup - workflow-focused ComfyUI extensions for image generation
// Copyright (C) 2026  Artificial Sweetener and contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

import { afterEach, describe, expect, it, vi } from "vitest";

import {
  configureMaskBatchNode,
  registerMaskBatchUpload
} from "../src/maskBatchUpload";
import type { ComfyApi, ComfyApp, ComfyExtension } from "../src/types";

afterEach(() => {
  document.body.replaceChildren();
});

describe("Load Mask Batch ordered-media integration", () => {
  it("registers its node-created integration", () => {
    let extension: ComfyExtension | undefined;
    const app = comfyApp();
    app.registerExtension = (value: ComfyExtension) => {
      extension = value;
    };
    registerMaskBatchUpload(app, api(), vi.fn());

    expect(extension?.name).toBe("SimpleSyrup.LoadMaskBatchUpload");
    expect(typeof extension?.nodeCreated).toBe("function");
  });

  it("renders the selected mask channel through native preview output", async () => {
    const fixture = maskNode(["mask.png"]);
    const app = comfyApp();
    const preview = vi.fn().mockResolvedValue({
      images: [{ filename: "rendered.png", subfolder: "", type: "temp" }]
    });
    configureMaskBatchNode(fixture.node, app, api(), preview);

    await vi.waitFor(() => {
      expect(preview).toHaveBeenCalledWith(["mask.png"], "alpha");
    });
    fixture.channelWidget.callback?.("red");

    await vi.waitFor(() => {
      expect(preview).toHaveBeenLastCalledWith(["mask.png"], "red");
      expect(app.nodeOutputs?.["9"]?.images).toEqual([
        { filename: "rendered.png", subfolder: "", type: "temp" }
      ]);
    });
  });

  it("removes an exact mask through its thumbnail action", async () => {
    const fixture = maskNode(["first.png", "second.png"]);
    const preview = maskPreview();
    const app = comfyApp();
    configureMaskBatchNode(fixture.node, app, api(), preview);
    const root = document.createElement("section");
    root.dataset.nodeId = "9";
    root.append(
      previewImage("first-preview.png", 0),
      previewImage("second-preview.png", 90)
    );
    document.body.append(root);

    await vi.waitFor(() => {
      expect(document.querySelectorAll("[data-ss-media-move-earlier]")).toHaveLength(2);
    });
    document
      .querySelector<HTMLButtonElement>("[data-ss-media-remove][data-ss-media-index='1']")
      ?.click();

    expect(fixture.imageWidget.value).toEqual(["first.png"]);
    expect(visiblePreviewFiles(root)).toEqual(["first-preview.png"]);
    await vi.waitFor(() => {
      expect(app.nodeOutputs?.["9"]?.images).toEqual([
        { filename: "first-preview.png", subfolder: "", type: "temp" }
      ]);
    });
    (fixture.node as { onRemoved?: () => unknown }).onRemoved?.();
    root.remove();
  });

  it("moves masks through per-thumbnail arrow actions", async () => {
    const fixture = maskNode(["first.png", "second.png"]);
    const preview = maskPreview();
    const app = comfyApp();
    configureMaskBatchNode(fixture.node, app, api(), preview);
    const root = document.createElement("section");
    root.dataset.nodeId = "9";
    root.append(
      previewImage("first-preview.png", 0),
      previewImage("second-preview.png", 90)
    );
    document.body.append(root);

    await vi.waitFor(() => {
      expect(document.querySelectorAll("[data-ss-media-move-later]")).toHaveLength(2);
    });
    document
      .querySelectorAll<HTMLButtonElement>("[data-ss-media-move-later]")[0]
      ?.click();

    expect(fixture.imageWidget.value).toEqual(["second.png", "first.png"]);
    expect(visiblePreviewFiles(root)).toEqual([
      "second-preview.png",
      "first-preview.png"
    ]);
    await vi.waitFor(() => {
      expect(app.nodeOutputs?.["9"]?.images).toEqual([
        { filename: "second-preview.png", subfolder: "", type: "temp" },
        { filename: "first-preview.png", subfolder: "", type: "temp" }
      ]);
    });
    (fixture.node as { onRemoved?: () => unknown }).onRemoved?.();
    root.remove();
  });
});

function api(): ComfyApi {
  const value = new EventTarget() as ComfyApi;
  value.apiURL = (path) => `/base${path}`;
  return value;
}

function comfyApp(): ComfyApp {
  return {
    nodeOutputs: {},
    registerExtension: vi.fn(),
    ui: { settings: { addSetting: vi.fn() } }
  };
}

function maskNode(files: string[]) {
  const imageWidget = widget("image", files, "combo", { canvasOnly: true });
  const channelWidget = widget("channel", "alpha", "combo");
  const uploadWidget = widget("upload", "image", "button", {
    serialize: false,
    canvasOnly: true
  });
  uploadWidget.callback = vi.fn();
  const widgets = [imageWidget, channelWidget, uploadWidget];
  const node = {
    constructor: { comfyClass: "SimpleSyrup.LoadMaskBatch" },
    id: 9,
    widgets,
    graph: { setDirtyCanvas: vi.fn() },
    addWidget(
      type: "button" | "combo",
      name: string,
      value: string,
      callback: (value?: unknown) => void,
      options?: TestWidget["options"]
    ) {
      const result = widget(name, value, type, options);
      result.callback = callback;
      widgets.push(result);
      return result;
    }
  };
  return { channelWidget, imageWidget, node };
}

function previewImage(filename: string, left: number): HTMLImageElement {
  const image = document.createElement("img");
  image.src = `/api/view?filename=${filename}&subfolder=&type=temp&preview=webp`;
  image.getBoundingClientRect = () => new DOMRect(left, 0, 80, 80);
  Object.defineProperties(image, {
    complete: { configurable: true, value: true },
    naturalWidth: { configurable: true, value: 80 }
  });
  return image;
}

function maskPreview() {
  return vi.fn((files: string[]) =>
    Promise.resolve({
      images: files.map((file) => ({
        filename: file.replace(".png", "-preview.png"),
        subfolder: "",
        type: "temp" as const
      }))
    })
  );
}

function visiblePreviewFiles(root: HTMLElement): string[] {
  return Array.from(root.querySelectorAll<HTMLImageElement>("img"))
    .filter((image) => image.style.display !== "none")
    .map((image) => new URL(image.src).searchParams.get("filename") ?? "");
}

interface TestWidget {
  name: string;
  value: unknown;
  type?: string;
  label?: string;
  hidden?: boolean;
  callback?: (value?: unknown) => void;
  computeSize?: (width?: number) => [number, number];
  options?: {
    canvasOnly?: boolean;
    serialize?: boolean;
    tooltip?: string;
    hidden?: boolean;
    values?: string[] | (() => string[]);
  };
}

function widget(
  name: string,
  value: unknown,
  type?: string,
  options?: TestWidget["options"]
): TestWidget {
  const result: TestWidget = { name, value };
  if (type) result.type = type;
  if (options) result.options = options;
  return result;
}
