// SimpleSyrup - workflow-focused ComfyUI extensions for image generation
// Copyright (C) 2026  Artificial Sweetener and contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

import { afterEach, describe, expect, it, vi } from "vitest";

import { SegPreviewNativeSurface } from "../src/segPreviewNativeSurface";
import type { ComfyApi } from "../src/types";
import { createFakeComfyApp } from "./testUtils";

describe("SegPreviewNativeSurface", () => {
  afterEach(() => {
    vi.useRealTimers();
    document.body.replaceChildren();
  });

  it("keeps overlay and native grid mutually exclusive", () => {
    const app = createFakeComfyApp();
    const apiTarget = new EventTarget();
    const api = apiTarget as ComfyApi;
    const node = previewNode();
    const root = document.createElement("section");
    root.dataset.nodeId = "7";
    document.body.append(root);
    const surface = new SegPreviewNativeSurface(app, api, node);
    const output = executionOutput();

    surface.update(output);

    expect(app.nodeOutputs?.["7"]?.images).toEqual([]);
    expect(node.imgs).toEqual([]);
    expect(node.widgets[0]?.hidden).toBe(true);

    surface.showGrid();

    expect(app.nodeOutputs?.["7"]?.images).toEqual(output.images);
    expect(node.imageIndex).toBeNull();
    expect(node.widgets[0]?.hidden).toBe(false);

    surface.showOverlay();

    expect(app.nodeOutputs?.["7"]?.images).toEqual([]);
    expect(node.imgs).toEqual([]);
    expect(node.widgets[0]?.hidden).toBe(true);
    surface.dispose();
  });

  it("opens native detail and returns to the same native grid", async () => {
    const app = createFakeComfyApp();
    const apiTarget = new EventTarget();
    const api = apiTarget as ComfyApi;
    const dispatchEvent = vi.spyOn(api, "dispatchEvent");
    const node = previewNode();
    const surface = new SegPreviewNativeSurface(app, api, node);
    const output = executionOutput();
    apiTarget.addEventListener("executed", () => {
      node.imgs = output.images.map(loadedImage);
    });
    surface.update(output);

    surface.inspect(1);

    expect(app.nodeOutputs?.["7"]?.images).toEqual(output.images);
    await vi.waitFor(() => {
      expect(node.imageIndex).toBe(1);
    });
    surface.update(app.nodeOutputs?.["7"]);
    expect(node.imageIndex).toBe(1);

    surface.showGrid();

    expect(node.imageIndex).toBeNull();
    expect(node.images).toBeUndefined();
    expect(app.nodeOutputs?.["7"]?.images).not.toBe(output.images);
    expect(dispatchEvent).toHaveBeenCalledOnce();
    expect(node.graph.setDirtyCanvas).toHaveBeenCalledWith(true, true);
    surface.dispose();
  });

  it("preserves loaded Nodes 1.0 images while the canvas preview is hidden", async () => {
    const app = createFakeComfyApp();
    const api = new EventTarget() as ComfyApi;
    const dispatchEvent = vi.spyOn(api, "dispatchEvent");
    const node = previewNode();
    const output = executionOutput();
    const loadedImages = output.images.map(loadedImage);
    node.imgs = loadedImages;
    app.nodeOutputs = { "7": output };
    const surface = new SegPreviewNativeSurface(app, api, node);

    surface.update(output);

    expect(node.widgets[0]?.hidden).toBe(true);
    expect(node.imgs).toBe(loadedImages);
    expect(app.nodeOutputs["7"]?.images).toEqual(output.images);
    expect(dispatchEvent).not.toHaveBeenCalled();

    surface.inspect(1);

    await vi.waitFor(() => {
      expect(node.imageIndex).toBe(1);
    });
    expect(node.imgs).toBe(loadedImages);
    expect(dispatchEvent).not.toHaveBeenCalled();
    surface.dispose();
  });
});

function previewNode() {
  return {
    id: 7,
    imageIndex: null as number | null,
    imageRects: [[0, 0, 10, 10]] as readonly unknown[],
    images: [reference("stale.png")],
    imgs: [document.createElement("img")],
    widgets: [
      {
        name: "$$canvas-image-preview",
        hidden: false,
        options: { hidden: false }
      }
    ],
    graph: { setDirtyCanvas: vi.fn() }
  };
}

function executionOutput() {
  const image = { filename: "asset.png", subfolder: "", type: "temp" as const };
  return {
    images: [reference("first.png"), reference("second.png")],
    simple_syrup_segs_preview: [
      {
        version: 1,
        source: { width: 4, height: 4 },
        preview: { width: 4, height: 4, image },
        atlas: { width: 2, height: 1, image },
        regions: [region(0), region(1)]
      }
    ]
  };
}

function reference(filename: string) {
  return { filename, subfolder: "", type: "temp" as const };
}

function loadedImage(reference: ReturnType<typeof executionOutput>["images"][number]) {
  const image = document.createElement("img");
  image.src = `/view?filename=${reference.filename}&subfolder=${reference.subfolder}&type=${reference.type}`;
  return image;
}

function region(index: number) {
  return {
    id: `seg-${String(index + 1).padStart(4, "0")}`,
    index,
    label: `region ${String(index + 1)}`,
    confidence: 0.9,
    area: 8,
    color: "#f24236",
    crop: { x: index * 2, y: 0, width: 2, height: 4 },
    atlas: { x: index, y: 0, width: 1, height: 1 }
  };
}
