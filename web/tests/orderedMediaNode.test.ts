// SimpleSyrup - workflow-focused ComfyUI extensions for image generation
// Copyright (C) 2026  Artificial Sweetener and contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

import { afterEach, describe, expect, it, vi } from "vitest";

import {
  configureOrderedMediaNode,
  registerOrderedMediaNode
} from "../src/orderedMediaNode";
import type {
  ComfyApi,
  ComfyApp,
  ComfyExtension,
  ComfyImageResult,
  ComfyNodeExecutionOutput
} from "../src/types";

const CONFIG = {
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

const activeNodes: Array<{ onRemoved: () => unknown }> = [];

afterEach(() => {
  for (const node of activeNodes.splice(0)) node.onRemoved();
  document.body.replaceChildren();
});

describe("ordered-media node integration", () => {
  it.each(["Nodes 1.0", "Nodes 2.0"])(
    "publishes through Comfy's native preview under %s",
    async () => {
      const fixture = createFixture(["one.png"]);
      configureOrderedMediaNode(
        fixture.node,
        fixture.app,
        fixture.api,
        CONFIG
      );

      expect(fixture.imageWidget.hidden).toBe(true);
      expect(fixture.uploadWidget.hidden).toBe(true);
      expect(fixture.imageWidget.computeSize?.()).toEqual([0, -4]);
      expect(
        fixture.node.widgets.some((candidate) =>
          [
            "simple_syrup_selected_media",
            "simple_syrup_move_earlier",
            "simple_syrup_move_later",
            "simple_syrup_remove_media"
          ].includes(candidate.name)
        )
      ).toBe(false);
      expect(widget(fixture, "simple_syrup_replace_media").label).toBe(
        "Replace images..."
      );
      expect(widget(fixture, "simple_syrup_add_media").label).toBe(
        "Add images..."
      );
      expect("addDOMWidget" in fixture.node).toBe(false);
      await vi.waitFor(() => {
        expect(fixture.app.nodeOutputs?.["7"]?.images).toEqual(
          references("one.png")
        );
      });
      expect(fixture.executed).toHaveBeenCalledWith(
        expect.objectContaining({ node: "7", display_node: "7" })
      );
    }
  );

  it("appends native multi-upload results and preserves duplicates", async () => {
    const fixture = configured(["one.png", "same.png"]);

    widget(fixture, "simple_syrup_add_media").callback?.();
    fixture.imageWidget.value = "two.png";
    fixture.imageWidget.callback?.("two.png");
    fixture.imageWidget.value = ["two.png", "same.png"];
    fixture.imageWidget.callback?.(["two.png", "same.png"]);

    expect(fixture.nativeUpload).toHaveBeenCalledOnce();
    expect(fixture.imageWidget.value).toEqual([
      "one.png",
      "same.png",
      "two.png",
      "same.png"
    ]);
    await vi.waitFor(() => {
      expect(fixture.app.nodeOutputs?.["7"]?.images).toEqual(
        references("one.png", "same.png", "two.png", "same.png")
      );
    });
  });

  it("treats pasted and dropped files as append operations", () => {
    const fixture = configured(["existing.png"]);

    fixture.node.onDragDrop();
    selectFiles(fixture, ["dropped-a.png", "dropped-b.png"]);
    expect(fixture.imageWidget.value).toEqual([
      "existing.png",
      "dropped-a.png",
      "dropped-b.png"
    ]);

    fixture.node.pasteFiles();
    selectFiles(fixture, ["pasted.png"]);
    expect(fixture.imageWidget.value).toEqual([
      "existing.png",
      "dropped-a.png",
      "dropped-b.png",
      "pasted.png"
    ]);
  });

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
    configureOrderedMediaNode(
      fixture.node,
      fixture.app,
      fixture.api,
      config
    );
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

  it("normalizes scalar workflows and survives Nodes 2.0 reactive assignments", async () => {
    const fixture = configured("saved.png", true);

    fixture.node.onGraphConfigured?.();

    expect(fixture.imageWidget.value).toEqual(["saved.png"]);
    await vi.waitFor(() => {
      expect(fixture.app.nodeOutputs?.["7"]?.images).toEqual(
        references("saved.png")
      );
    });
  });

  it("restores native handlers and clears preview on node removal", () => {
    const fixture = createFixture(["one.png"]);
    const paste = fixture.node.pasteFiles;
    const drop = fixture.node.onDragDrop;
    configureOrderedMediaNode(
      fixture.node,
      fixture.app,
      fixture.api,
      CONFIG
    );

    fixture.node.onRemoved();

    expect(fixture.node.pasteFiles).toBe(paste);
    expect(fixture.node.onDragDrop).toBe(drop);
    expect(fixture.originalRemoved).toHaveBeenCalledOnce();
    expect(fixture.app.nodeOutputs?.["7"]?.images).toEqual([]);
  });

  it("registers a guarded extension and ignores unrelated nodes", async () => {
    const fixture = createFixture([]);
    let extension: ComfyExtension | undefined;
    fixture.app.registerExtension = (value: ComfyExtension) => {
      extension = value;
    };
    registerOrderedMediaNode(
      fixture.app,
      fixture.api,
      "test.extension",
      CONFIG
    );
    fixture.node.constructor.comfyClass = "Other.Node";

    await extension?.nodeCreated?.(fixture.node);

    expect(extension?.name).toBe("test.extension");
    expect(fixture.node.addWidget).not.toHaveBeenCalled();
  });
});

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

function configured(value: unknown, reactive = false) {
  const fixture = createFixture(value, reactive);
  configureOrderedMediaNode(
    fixture.node,
    fixture.app,
    fixture.api,
    CONFIG
  );
  return fixture;
}

function createFixture(value: unknown, reactive = false) {
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

function nativePreviewImage(
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

function nativePreviewButton(filename: string, left: number): HTMLButtonElement {
  const button = document.createElement("button");
  button.append(nativePreviewImage(filename, left));
  mockRect(button, left, 0, 80, 80);
  return button;
}

function visiblePreviewFiles(root: HTMLElement): string[] {
  return Array.from(root.querySelectorAll<HTMLImageElement>("img"))
    .filter((image) => image.style.display !== "none")
    .map((image) => new URL(image.src).searchParams.get("filename") ?? "");
}

function imageFiles(images: HTMLImageElement[] | undefined): string[] {
  return (images ?? []).map(
    (image) => new URL(image.src).searchParams.get("filename") ?? ""
  );
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

function widget(
  fixture: ReturnType<typeof createFixture>,
  name: string
): TestWidget {
  const found = fixture.node.widgets.find((candidate) => candidate.name === name);
  if (!found) throw new Error(`Missing test widget ${name}.`);
  return found;
}

function selectFiles(
  fixture: ReturnType<typeof createFixture>,
  files: string[]
): void {
  fixture.imageWidget.value = files;
  fixture.imageWidget.callback?.(files);
}

function references(...filenames: string[]): ComfyImageResult[] {
  return filenames.map((filename) => ({
    filename,
    subfolder: "",
    type: "input"
  }));
}

function deferred<T>(): {
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
