// SimpleSyrup - workflow-focused ComfyUI extensions for image generation
// Copyright (C) 2026  Artificial Sweetener and contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

import { describe, expect, it, vi } from "vitest";

import {
  configureMaskBatchNode,
  registerMaskBatchUpload
} from "../src/maskBatchUpload";
import type { MaskBatchPreviewClient } from "../src/maskBatchUpload";
import type {
  ComfyApp,
  ComfyExtension,
  ComfyNodeExecutionOutput
} from "../src/types";

interface TestWidget {
  name: string;
  value: unknown;
  type?: string;
  label?: string;
  callback?: (value?: unknown) => unknown;
  computeSize?: (width?: number) => [number, number];
  hidden?: boolean;
  serializeValue?: () => unknown;
  disabled?: boolean;
  options?: {
    canvasOnly?: boolean;
    disabled?: boolean;
    serialize?: boolean;
    tooltip?: string;
    hidden?: boolean;
    values?: string[];
  };
}

const PREVIEW_OUTPUT: ComfyNodeExecutionOutput = {
  images: [
    {
      filename: "ComfyUI_temp_mask.png",
      subfolder: "",
      type: "temp"
    }
  ],
  animated: [false]
};

function createNode(value: unknown = []) {
  const nativeUploadCallback = vi.fn();
  const channelCallback = vi.fn();
  const onWidgetChanged = vi.fn();
  const setDirtyCanvas = vi.fn();
  const pasteFiles = vi.fn(() => true);
  const onDragDrop = vi.fn(() => Promise.resolve(true));
  const originalOnRemoved = vi.fn();
  const imageWidget: TestWidget = {
    name: "image",
    value,
    type: "combo",
    options: { canvasOnly: true }
  };
  const channelWidget: TestWidget = {
    name: "channel",
    value: "alpha",
    type: "combo",
    callback: channelCallback
  };
  const uploadWidget: TestWidget = {
    name: "upload",
    value: "image",
    type: "button",
    label: "choose file to upload",
    callback: nativeUploadCallback,
    options: { serialize: false, canvasOnly: true }
  };
  const widgets = [imageWidget, channelWidget, uploadWidget];
  const node = {
    constructor: { comfyClass: "SimpleSyrup.LoadMaskBatch" },
    id: 7,
    widgets,
    properties: {} as Record<string, unknown>,
    imageIndex: null as number | null,
    images: undefined as ComfyNodeExecutionOutput["images"],
    imgs: undefined as unknown[] | undefined,
    pasteFiles,
    onDragDrop,
    onRemoved: originalOnRemoved as (...args: unknown[]) => unknown,
    onGraphConfigured: undefined as ((...args: unknown[]) => unknown) | undefined,
    onWidgetChanged,
    graph: { setDirtyCanvas },
    addWidget(
      type: "button",
      name: string,
      widgetValue: string | undefined,
      callback: (value?: unknown) => void,
      options: TestWidget["options"]
    ): TestWidget {
      const widget: TestWidget = {
        type,
        name,
        value: widgetValue,
        callback
      };
      if (options) widget.options = options;
      widgets.push(widget);
      return widget;
    }
  };
  return {
    channelCallback,
    channelWidget,
    imageWidget,
    nativeUploadCallback,
    node,
    onDragDrop,
    onWidgetChanged,
    originalOnRemoved,
    pasteFiles,
    setDirtyCanvas,
    uploadWidget
  };
}

function createExecutionEvents(): EventTarget {
  return new EventTarget();
}

function previewClient(
  output: ComfyNodeExecutionOutput = PREVIEW_OUTPUT
): ReturnType<typeof vi.fn<MaskBatchPreviewClient>> {
  return vi.fn<MaskBatchPreviewClient>().mockResolvedValue(output);
}

function button(node: ReturnType<typeof createNode>["node"], name: string) {
  const widget = node.widgets.find((candidate) => candidate.name === name);
  if (!widget) throw new Error(`Missing test button ${name}.`);
  return widget;
}

function selectFiles(imageWidget: TestWidget, files: string[]): void {
  imageWidget.value = files;
  imageWidget.callback?.(files);
}

describe("Load Mask Batch native upload integration", () => {
  it("registers a node-created hook", () => {
    let extension: ComfyExtension | undefined;
    const app = {
      registerExtension(value: ComfyExtension) {
        extension = value;
      }
    } as ComfyApp;

    registerMaskBatchUpload(app, createExecutionEvents());

    expect(extension?.name).toBe("SimpleSyrup.LoadMaskBatchUpload");
    expect(typeof extension?.nodeCreated).toBe("function");
  });

  it("reports native node-configuration failures with context", () => {
    let extension: ComfyExtension | undefined;
    const logger = { warn: vi.fn() };
    const app = {
      registerExtension(value: ComfyExtension) {
        extension = value;
      }
    } as ComfyApp;
    registerMaskBatchUpload(app, createExecutionEvents(), undefined, logger);
    const { node } = createNode();
    node.addWidget = () => {
      throw new Error("native widget failed");
    };

    expect(() => extension?.nodeCreated?.(node)).toThrow(
      "native widget failed"
    );
    expect(logger.warn).toHaveBeenCalledWith(
      "Could not configure Load Mask Batch native controls: native widget failed",
      expect.any(Error)
    );
  });

  it("uses native cross-renderer controls and hides internal upload state", () => {
    const { imageWidget, node, uploadWidget } = createNode(["only.png"]);
    configureMaskBatchNode(node, createExecutionEvents());

    const replace = button(node, "simple_syrup_replace_masks");
    const add = button(node, "simple_syrup_add_masks");
    const selected = button(node, "simple_syrup_selected_mask");
    const remove = button(node, "simple_syrup_remove_mask");
    expect(uploadWidget.label).toBe("choose file to upload");
    expect(imageWidget.options).toMatchObject({
      canvasOnly: true,
      hidden: true
    });
    expect(uploadWidget.options).toMatchObject({
      canvasOnly: true,
      hidden: true
    });
    expect(imageWidget.computeSize?.()).toEqual([0, -4]);
    expect(imageWidget.hidden).toBe(true);
    expect(uploadWidget.computeSize?.()).toEqual([0, -4]);
    expect(replace.label).toBe("Replace masks...");
    expect(add.label).toBe("Add masks...");
    expect(selected.label).toBe("selected mask");
    expect(selected.value).toBe("1. only.png");
    expect(selected.options?.values).toEqual(["1. only.png"]);
    expect(remove.label).toBe("Remove selected mask");
    expect(add.type).toBe("button");
    expect(add.options).toMatchObject({ serialize: false });
    expect(add.options?.canvasOnly).toBeUndefined();
    expect(remove.disabled).toBe(false);
    expect(remove.options?.disabled).toBe(false);
  });

  it("shows an explicit disabled empty state instead of undefined", () => {
    const { node } = createNode([]);

    configureMaskBatchNode(node, createExecutionEvents());

    const selected = button(node, "simple_syrup_selected_mask");
    expect(selected.value).toBe("No masks loaded");
    expect(selected.options?.values).toEqual(["No masks loaded"]);
    expect(selected.disabled).toBe(true);
    expect(selected.options?.disabled).toBe(true);
    expect(button(node, "simple_syrup_remove_mask").disabled).toBe(true);
  });

  it("uses the native multi-value widget as the only selection owner", async () => {
    const { imageWidget, node } = createNode([]);
    const loadPreview = previewClient();
    configureMaskBatchNode(node, createExecutionEvents(), loadPreview);
    loadPreview.mockClear();

    button(node, "simple_syrup_add_masks").callback?.();
    const uploaded = ["one.png", "two.png", "three.png"];
    selectFiles(imageWidget, uploaded);

    expect(imageWidget.value).toEqual(uploaded);
    expect(imageWidget.serializeValue).toBeUndefined();
    expect(node.properties).toEqual({});
    await vi.waitFor(() => {
      expect(loadPreview).toHaveBeenCalledWith(uploaded, "alpha");
    });
  });

  it("accepts direct native list edits and clears without a backend request", async () => {
    const { imageWidget, node } = createNode(["one.png", "two.png"]);
    const loadPreview = previewClient();
    configureMaskBatchNode(node, createExecutionEvents(), loadPreview);
    loadPreview.mockClear();

    selectFiles(imageWidget, ["two.png"]);
    await vi.waitFor(() => {
      expect(loadPreview).toHaveBeenCalledWith(["two.png"], "alpha");
    });

    loadPreview.mockClear();
    selectFiles(imageWidget, []);

    expect(imageWidget.value).toEqual([]);
    expect(loadPreview).not.toHaveBeenCalled();
    expect(button(node, "simple_syrup_remove_mask").disabled).toBe(true);
  });

  it("normalizes a serialized scalar from an older workflow", async () => {
    const { imageWidget, node } = createNode("saved-mask.png");
    const loadPreview = previewClient();
    configureMaskBatchNode(node, createExecutionEvents(), loadPreview);
    loadPreview.mockClear();

    node.onGraphConfigured?.();

    expect(imageWidget.value).toEqual(["saved-mask.png"]);
    await vi.waitFor(() => {
      expect(loadPreview).toHaveBeenCalledWith(["saved-mask.png"], "alpha");
    });
  });

  it("replaces the ordered list through the explicit native control", async () => {
    const { imageWidget, nativeUploadCallback, node } = createNode([
      "old.png"
    ]);
    const loadPreview = previewClient();
    configureMaskBatchNode(node, createExecutionEvents(), loadPreview);
    loadPreview.mockClear();

    button(node, "simple_syrup_replace_masks").callback?.();
    const replacement = ["right.png", "left.png"];
    selectFiles(imageWidget, replacement);

    expect(nativeUploadCallback).toHaveBeenCalledOnce();
    expect(imageWidget.value).toEqual(replacement);
    expect(imageWidget.serializeValue).toBeUndefined();
    expect(node.properties).toEqual({});
    await vi.waitFor(() => {
      expect(loadPreview).toHaveBeenCalledWith(replacement, "alpha");
    });
  });

  it("appends native upload results in order and preserves duplicate paths", async () => {
    const { imageWidget, nativeUploadCallback, node } = createNode();
    const loadPreview = previewClient();
    configureMaskBatchNode(node, createExecutionEvents(), loadPreview);
    selectFiles(imageWidget, ["one.png", "duplicate.png"]);
    loadPreview.mockClear();

    button(node, "simple_syrup_add_masks").callback?.();
    imageWidget.value = "two.png";
    imageWidget.callback?.("two.png");
    const uploaded = ["two.png", "duplicate.png"];
    selectFiles(imageWidget, uploaded);

    const expected = [
      "one.png",
      "duplicate.png",
      "two.png",
      "duplicate.png"
    ];
    expect(nativeUploadCallback).toHaveBeenCalledOnce();
    expect(uploaded).toEqual(["two.png", "duplicate.png"]);
    expect(imageWidget.value).toEqual(expected);
    expect(imageWidget.serializeValue).toBeUndefined();
    expect(node.properties).toEqual({});
    await vi.waitFor(() => {
      expect(loadPreview).toHaveBeenCalledWith(expected, "alpha");
    });
  });

  it("avoids recursive updates from the Nodes 2.0 reactive multiselect", async () => {
    const { imageWidget, node } = createNode(["one.png"]);
    let storedValue = imageWidget.value;
    Object.defineProperty(imageWidget, "value", {
      configurable: true,
      get: () => storedValue,
      set: (value: unknown) => {
        storedValue = value;
        imageWidget.callback?.(value);
      }
    });
    const loadPreview = previewClient();
    configureMaskBatchNode(node, createExecutionEvents(), loadPreview);
    loadPreview.mockClear();

    button(node, "simple_syrup_add_masks").callback?.();
    const uploaded = ["two.png", "three.png"];
    imageWidget.value = uploaded;
    imageWidget.callback?.([...uploaded]);

    expect(imageWidget.value).toEqual([
      "one.png",
      "two.png",
      "three.png"
    ]);
    await vi.waitFor(() => {
      expect(loadPreview).toHaveBeenCalledTimes(1);
    });
  });

  it("cancelled Add cannot affect later replace, paste, or drop selections", () => {
    const { imageWidget, node, onDragDrop, pasteFiles, uploadWidget } =
      createNode();
    configureMaskBatchNode(node, createExecutionEvents());
    selectFiles(imageWidget, ["existing.png"]);

    button(node, "simple_syrup_add_masks").callback?.();
    uploadWidget.callback?.();
    selectFiles(imageWidget, ["replace.png"]);
    expect(imageWidget.value).toEqual(["replace.png"]);

    button(node, "simple_syrup_add_masks").callback?.();
    node.pasteFiles();
    selectFiles(imageWidget, ["paste.png"]);
    expect(imageWidget.value).toEqual(["paste.png"]);
    expect(pasteFiles).toHaveBeenCalledOnce();

    button(node, "simple_syrup_add_masks").callback?.();
    void node.onDragDrop();
    selectFiles(imageWidget, ["drop.png"]);
    expect(imageWidget.value).toEqual(["drop.png"]);
    expect(onDragDrop).toHaveBeenCalledOnce();
  });

  it("removes the active gallery position and never deletes by filename", async () => {
    const { imageWidget, node, onWidgetChanged } = createNode();
    const loadPreview = previewClient();
    configureMaskBatchNode(node, createExecutionEvents(), loadPreview);
    selectFiles(imageWidget, ["duplicate.png", "middle.png", "duplicate.png"]);
    loadPreview.mockClear();
    onWidgetChanged.mockClear();
    node.imageIndex = 2;

    button(node, "simple_syrup_remove_mask").callback?.();

    const remaining = ["duplicate.png", "middle.png"];
    expect(imageWidget.value).toEqual(remaining);
    expect(node.imageIndex).toBe(1);
    expect(onWidgetChanged).toHaveBeenCalledWith(
      "image",
      remaining,
      ["duplicate.png", "middle.png", "duplicate.png"],
      imageWidget
    );
    await vi.waitFor(() => {
      expect(loadPreview).toHaveBeenCalledWith(remaining, "alpha");
    });
  });

  it("removes the native list selection and permits clearing the final mask", () => {
    const { imageWidget, node, onWidgetChanged } = createNode();
    configureMaskBatchNode(node, createExecutionEvents());
    selectFiles(imageWidget, ["one.png", "two.png"]);
    const selected = button(node, "simple_syrup_selected_mask");
    const remove = button(node, "simple_syrup_remove_mask");

    selected.callback?.("2. two.png");
    remove.callback?.();
    expect(imageWidget.value).toEqual(["one.png"]);
    expect(selected.value).toBe("1. one.png");
    expect(remove.disabled).toBe(false);

    remove.callback?.();
    expect(imageWidget.value).toEqual([]);
    expect(selected.value).toBe("No masks loaded");
    expect(selected.options?.values).toEqual(["No masks loaded"]);
    expect(selected.disabled).toBe(true);
    expect(remove.disabled).toBe(true);
    expect(onWidgetChanged).toHaveBeenCalledTimes(2);
  });

  it("clears stale native preview state when the final mask is removed", () => {
    const { imageWidget, node } = createNode(["one.png"]);
    const loadPreview = previewClient();
    const app = {
      nodeOutputs: { "7": PREVIEW_OUTPUT },
      registerExtension: vi.fn(),
      ui: { settings: { addSetting: vi.fn() } }
    } as unknown as ComfyApp;
    configureMaskBatchNode(
      node,
      createExecutionEvents(),
      loadPreview,
      console,
      app
    );
    node.imgs = [{}];
    node.images = PREVIEW_OUTPUT.images;
    app.nodeOutputs = { "7": PREVIEW_OUTPUT };

    selectFiles(imageWidget, []);

    expect(node.imgs).toBeUndefined();
    expect(node.images).toEqual([]);
    expect(node.imageIndex).toBeNull();
    expect(app.nodeOutputs).toEqual({});
  });

  it("restores a persisted batch and refreshes the selected channel", async () => {
    const { channelWidget, imageWidget, node } = createNode([
      "one.png",
      "two.png"
    ]);
    const loadPreview = previewClient();
    configureMaskBatchNode(node, createExecutionEvents(), loadPreview);
    loadPreview.mockClear();
    channelWidget.value = "blue";

    node.onGraphConfigured?.();

    expect(imageWidget.value).toEqual(["one.png", "two.png"]);
    await vi.waitFor(() => {
      expect(loadPreview).toHaveBeenCalledWith(
        ["one.png", "two.png"],
        "blue"
      );
    });
  });

  it("restores native paste and drop handlers before native node cleanup", () => {
    const { node, onDragDrop, originalOnRemoved, pasteFiles } = createNode();
    configureMaskBatchNode(node, createExecutionEvents());

    node.onRemoved();

    expect(originalOnRemoved).toHaveBeenCalledOnce();
    expect(node.pasteFiles).toBe(pasteFiles);
    expect(node.onDragDrop).toBe(onDragDrop);
  });

  it("does not modify unrelated nodes", () => {
    const { node, uploadWidget } = createNode();
    node.constructor.comfyClass = "LoadImageMask";

    configureMaskBatchNode(node, createExecutionEvents());

    expect(uploadWidget.label).toBe("choose file to upload");
    expect(node.widgets).toHaveLength(3);
  });
});
