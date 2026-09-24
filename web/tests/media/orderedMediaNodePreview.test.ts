// SimpleSyrup - workflow-focused ComfyUI extensions for image generation
// Copyright (C) 2026  Artificial Sweetener and contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

import { afterEach, describe, expect, it, vi } from "vitest";

import { configureOrderedMediaNode } from "../../src/orderedMediaNode";
import type { ComfyNodeExecutionOutput } from "../../src/types";
import {
  CONFIG,
  configured,
  createFixture,
  deferred,
  imageFiles,
  mockRect,
  nativePreviewButton,
  nativePreviewImage,
  references,
  resetOrderedMediaFixtures,
  visiblePreviewFiles
} from "./orderedMediaNodeTestSupport";

afterEach(resetOrderedMediaFixtures);

describe("ordered-media native preview integration", () => {
  it("attaches actions when a multi-image native gallery mounts later", async () => {
    const fixture = configured(["one.png", "two.png"]);
    await vi.waitFor(() => {
      expect(fixture.app.nodeOutputs?.["7"]?.images).toHaveLength(2);
    });
    expect(document.querySelectorAll("[data-ss-media-move-later]")).toHaveLength(0);
    const root = document.createElement("section");
    root.dataset.nodeId = "7";
    root.append(
      nativePreviewButton("one.png", 0),
      nativePreviewButton("two.png", 90)
    );

    document.body.append(root);

    await vi.waitFor(() => {
      expect(document.querySelectorAll("[data-ss-media-move-later]")).toHaveLength(2);
    });
  });

  it("moves and removes exact positions through thumbnail actions", async () => {
    const fixture = configured(["same.png", "middle.png", "same.png"]);
    await vi.waitFor(() => {
      expect(fixture.app.nodeOutputs?.["7"]?.images).toHaveLength(3);
    });
    const root = document.createElement("section");
    root.dataset.nodeId = "7";
    root.append(
      nativePreviewImage("same.png", 0),
      nativePreviewImage("middle.png", 90),
      nativePreviewImage("same.png", 180)
    );
    document.body.append(root);
    await vi.waitFor(() => {
      expect(document.querySelectorAll("[data-ss-media-move-earlier]")).toHaveLength(3);
    });
    document
      .querySelectorAll<HTMLButtonElement>("[data-ss-media-move-earlier]")[2]
      ?.click();

    expect(fixture.imageWidget.value).toEqual([
      "same.png",
      "same.png",
      "middle.png"
    ]);
    expect(visiblePreviewFiles(root)).toEqual([
      "same.png",
      "same.png",
      "middle.png"
    ]);
    await vi.waitFor(() => {
      expect(fixture.app.nodeOutputs?.["7"]?.images).toEqual(
        references("same.png", "same.png", "middle.png")
      );
    });
    expect(fixture.onWidgetChanged).toHaveBeenCalledWith(
      "image",
      ["same.png", "same.png", "middle.png"],
      ["same.png", "middle.png", "same.png"],
      fixture.imageWidget
    );

    document
      .querySelectorAll<HTMLButtonElement>("[data-ss-media-remove]")[1]
      ?.click();
    expect(fixture.imageWidget.value).toEqual(["same.png", "middle.png"]);
    expect(visiblePreviewFiles(root)).toEqual(["same.png", "middle.png"]);
    await vi.waitFor(() => {
      expect(fixture.app.nodeOutputs?.["7"]?.images).toEqual(
        references("same.png", "middle.png")
      );
    });
  });

  it("keeps exactly the native cells in visual order during a slow rebuild", async () => {
    const delayed = deferred<ComfyNodeExecutionOutput>();
    let previewCalls = 0;
    const config = {
      ...CONFIG,
      preview: (files: string[]) => {
        previewCalls += 1;
        return previewCalls === 1
          ? Promise.resolve({ images: references(...files) })
          : delayed.promise;
      }
    };
    const fixture = createFixture(["one.png", "two.png", "three.png"]);
    configureOrderedMediaNode(fixture.node, fixture.app, fixture.api, config);
    await vi.waitFor(() => {
      expect(fixture.app.nodeOutputs?.["7"]?.images).toEqual(
        references("one.png", "two.png", "three.png")
      );
    });
    const root = document.createElement("section");
    root.dataset.nodeId = "7";
    root.append(
      nativePreviewImage("one.png", 0),
      nativePreviewImage("two.png", 90),
      nativePreviewImage("three.png", 180)
    );
    document.body.append(root);
    await vi.waitFor(() => {
      expect(document.querySelectorAll("[data-ss-media-move-later]")).toHaveLength(3);
    });

    document
      .querySelectorAll<HTMLButtonElement>("[data-ss-media-move-later]")[0]
      ?.click();

    expect(fixture.imageWidget.value).toEqual([
      "two.png",
      "one.png",
      "three.png"
    ]);
    expect(visiblePreviewFiles(root)).toEqual([
      "two.png",
      "one.png",
      "three.png"
    ]);
    expect(root.querySelectorAll("img")).toHaveLength(3);
    expect(fixture.app.nodeOutputs?.["7"]?.images).toEqual(
      references("one.png", "two.png", "three.png")
    );
    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(visiblePreviewFiles(root)).toEqual([
      "two.png",
      "one.png",
      "three.png"
    ]);

    delayed.resolve({ images: references("two.png", "one.png", "three.png") });
    await vi.waitFor(() => {
      expect(fixture.app.nodeOutputs?.["7"]?.images).toEqual(
        references("two.png", "one.png", "three.png")
      );
    });
    expect(visiblePreviewFiles(root)).toEqual([
      "two.png",
      "one.png",
      "three.png"
    ]);

    root.replaceChildren(
      nativePreviewImage("two.png", 0),
      nativePreviewImage("one.png", 90),
      nativePreviewImage("three.png", 180)
    );
    expect(visiblePreviewFiles(root)).toEqual([
      "two.png",
      "one.png",
      "three.png"
    ]);
    expect(root.querySelectorAll("img")).toHaveLength(3);
  });

  it("keeps the native visual and serialized order when rebuilding fails", async () => {
    const logger = { warn: vi.fn() };
    let previewCalls = 0;
    const config = {
      ...CONFIG,
      preview: (files: string[]) => {
        previewCalls += 1;
        return previewCalls === 1
          ? Promise.resolve({ images: references(...files) })
          : Promise.reject(new Error("preview unavailable"));
      }
    };
    const fixture = createFixture(["one.png", "two.png"]);
    configureOrderedMediaNode(
      fixture.node,
      fixture.app,
      fixture.api,
      config,
      logger
    );
    await vi.waitFor(() => {
      expect(fixture.app.nodeOutputs?.["7"]?.images).toEqual(
        references("one.png", "two.png")
      );
    });
    const root = document.createElement("section");
    root.dataset.nodeId = "7";
    root.append(
      nativePreviewImage("one.png", 0),
      nativePreviewImage("two.png", 90)
    );
    document.body.append(root);
    await vi.waitFor(() => {
      expect(document.querySelectorAll("[data-ss-media-move-later]")).toHaveLength(2);
    });

    document
      .querySelectorAll<HTMLButtonElement>("[data-ss-media-move-later]")[0]
      ?.click();

    await vi.waitFor(() => {
      expect(logger.warn).toHaveBeenCalledWith(
        expect.stringContaining("preview unavailable"),
        expect.any(Error)
      );
    });
    expect(fixture.imageWidget.value).toEqual(["two.png", "one.png"]);
    expect(visiblePreviewFiles(root)).toEqual(["two.png", "one.png"]);
    expect(root.querySelectorAll("img")).toHaveLength(2);
    expect(fixture.app.nodeOutputs?.["7"]?.images).toEqual(
      references("one.png", "two.png")
    );
  });

  it("preserves native thumbnail inspection clicks in Nodes 2.0", async () => {
    const fixture = configured(["one.png", "two.png", "three.png"]);
    await vi.waitFor(() => {
      expect(fixture.app.nodeOutputs?.["7"]?.images).toHaveLength(3);
    });
    const root = document.createElement("section");
    root.dataset.nodeId = "7";
    const images = ["one.png", "two.png", "three.png"].map(
      (filename, index) => nativePreviewImage(filename, index * 90)
    );
    root.append(...images);
    document.body.append(root);
    const clicked = vi.fn();
    images[1]?.addEventListener("click", clicked);

    images[1]?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    expect(clicked).toHaveBeenCalledOnce();
  });

  it("moves Nodes 1.0 native canvas cells through imageRects", async () => {
    const fixture = createFixture(["one.png", "two.png"]);
    const canvas = document.createElement("canvas");
    mockRect(canvas, 0, 0, 500, 500);
    document.body.append(canvas);
    fixture.app.canvas = {
      canvas,
      convertEventToCanvasOffset: (event) => [event.clientX, event.clientY],
      convertOffsetToCanvas: (position) => [...position]
    };
    Object.assign(fixture.node, {
      pos: [10, 100],
      imageIndex: null,
      imageRects: [
        [0, 50, 80, 80],
        [80, 50, 80, 80]
      ],
      imgs: [nativePreviewImage("one.png", 0), nativePreviewImage("two.png", 90)]
    });
    configureOrderedMediaNode(fixture.node, fixture.app, fixture.api, CONFIG);
    await vi.waitFor(() => {
      expect(fixture.app.nodeOutputs?.["7"]?.images).toHaveLength(2);
    });
    await vi.waitFor(() => {
      expect(document.querySelectorAll("[data-ss-media-move-later]")).toHaveLength(2);
    });
    document
      .querySelectorAll<HTMLButtonElement>("[data-ss-media-move-later]")[0]
      ?.click();

    expect(fixture.imageWidget.value).toEqual(["two.png", "one.png"]);
    expect(imageFiles(fixture.node.imgs)).toEqual(["two.png", "one.png"]);
    await vi.waitFor(() => {
      expect(fixture.app.nodeOutputs?.["7"]?.images).toEqual(
        references("two.png", "one.png")
      );
    });
  });
});
