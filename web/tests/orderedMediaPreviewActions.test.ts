// SimpleSyrup - workflow-focused ComfyUI extensions for image generation
// Copyright (C) 2026  Artificial Sweetener and contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

import { afterEach, describe, expect, it, vi } from "vitest";

import { NativeNodePreview } from "../src/nativeNodePreview";
import { OrderedMediaPreviewActions } from "../src/orderedMediaPreviewActions";
import type { ComfyApi, ComfyApp, ComfyImageResult } from "../src/types";

const adapters: OrderedMediaPreviewActions[] = [];

afterEach(() => {
  for (const adapter of adapters.splice(0)) adapter.dispose();
  document.body.replaceChildren();
});

describe("OrderedMediaPreviewActions", () => {
  it("discovers Nodes 2.0 thumbnails by output reference", async () => {
    const fixture = actionFixture(["one.png", "two.png"]);
    const root = document.createElement("section");
    root.dataset.nodeId = "7";
    const unrelated = document.createElement("img");
    unrelated.src = "/icon.png";
    root.append(unrelated, previewImage("one.png", 0), previewImage("two.png", 100));
    document.body.append(root);

    await vi.waitFor(() => {
      expect(document.querySelectorAll("[data-ss-media-move-earlier]")).toHaveLength(2);
    });
    document
      .querySelectorAll<HTMLButtonElement>("[data-ss-media-move-earlier]")[1]
      ?.click();
    document
      .querySelectorAll<HTMLButtonElement>("[data-ss-media-move-later]")[0]
      ?.click();
    document
      .querySelectorAll<HTMLButtonElement>("[data-ss-media-remove]")[1]
      ?.click();

    expect(fixture.moveEarlier).toHaveBeenCalledWith(1);
    expect(fixture.moveLater).toHaveBeenCalledWith(0);
    expect(fixture.remove).toHaveBeenCalledWith(1);
  });

  it("positions controls from Nodes 1.0 imageRects", async () => {
    const canvas = document.createElement("canvas");
    mockRect(canvas, 10, 20, 500, 500);
    const fixture = actionFixture(["one.png", "two.png"], {
      pos: [100, 200],
      imageIndex: null,
      imageRects: [
        [0, 30, 80, 80],
        [80, 30, 80, 80]
      ]
    });
    fixture.app.canvas = {
      canvas,
      convertEventToCanvasOffset: (event) => [
        event.clientX - 10,
        event.clientY - 20
      ],
      convertOffsetToCanvas: (position) => [...position]
    };
    document.body.append(canvas);

    await vi.waitFor(() => {
      expect(document.querySelectorAll(".ss-native-preview-affordance")).toHaveLength(2);
    });
    const first = document.querySelector<HTMLElement>(
      ".ss-native-preview-affordance"
    );
    expect(first?.style.left).toBe("110px");
    expect(first?.style.top).toBe("250px");
    expect(first?.style.width).toBe("80px");
  });

  it("restores controls when native detail view returns to the grid", async () => {
    const node = { id: 7, imageIndex: null as number | null };
    const fixture = actionFixture(["one.png", "two.png"], node);
    const root = document.createElement("section");
    root.dataset.nodeId = "7";
    root.append(previewImage("one.png", 0), previewImage("two.png", 100));
    document.body.append(root);
    await vi.waitFor(() => {
      expect(document.querySelectorAll(".ss-native-preview-affordance")).toHaveLength(2);
    });

    fixture.node.imageIndex = 0;
    document.dispatchEvent(new MouseEvent("click"));
    await vi.waitFor(() => {
      expect(document.querySelectorAll(".ss-native-preview-affordance")).toHaveLength(0);
    });

    fixture.node.imageIndex = null;
    document.dispatchEvent(new MouseEvent("click"));
    await vi.waitFor(() => {
      expect(document.querySelectorAll(".ss-native-preview-affordance")).toHaveLength(2);
    });
  });

  it("hides actions when Comfy output cannot map to every file", async () => {
    const fixture = actionFixture(["one.png", "two.png"]);
    fixture.app.nodeOutputs = { "7": { images: references("one.png") } };
    document.dispatchEvent(new MouseEvent("click"));

    await vi.waitFor(() => {
      expect(document.querySelectorAll(".ss-native-preview-affordance")).toHaveLength(0);
    });
  });
});

function actionFixture(
  files: string[],
  nodeOverrides: Record<string, unknown> = {}
) {
  const app: ComfyApp = {
    nodeOutputs: {
      "7": { images: references(...files), animated: files.map(() => false) }
    },
    ui: { settings: { addSetting: vi.fn() } },
    registerExtension: vi.fn()
  };
  const api = new EventTarget() as ComfyApi;
  const node = { id: 7, ...nodeOverrides } as {
    id: number;
    imageIndex?: number | null;
  };
  const preview = new NativeNodePreview(app, api, node);
  const moveEarlier = vi.fn();
  const moveLater = vi.fn();
  const remove = vi.fn();
  const adapter = new OrderedMediaPreviewActions({
    app,
    node,
    preview,
    itemLabel: "image",
    getFiles: () => [...files],
    moveEarlier,
    moveLater,
    remove
  });
  adapters.push(adapter);
  return { adapter, app, moveEarlier, moveLater, node, remove };
}

function previewImage(filename: string, left: number): HTMLImageElement {
  const image = document.createElement("img");
  image.src = `/api/view?filename=${filename}&subfolder=&type=input&preview=webp`;
  mockRect(image, left, 0, 80, 80);
  return image;
}

function mockRect(
  element: Element,
  left: number,
  top: number,
  width: number,
  height: number
): void {
  element.getBoundingClientRect = () => new DOMRect(left, top, width, height);
}

function references(...filenames: string[]): ComfyImageResult[] {
  return filenames.map((filename) => ({
    filename,
    subfolder: "",
    type: "input"
  }));
}
