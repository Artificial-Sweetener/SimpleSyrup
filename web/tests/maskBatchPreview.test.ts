// SimpleSyrup - workflow-focused ComfyUI extensions for image generation
// Copyright (C) 2026  Artificial Sweetener and contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

import { describe, expect, it, vi } from "vitest";

import {
  MaskBatchPreviewController,
  type MaskBatchPreviewClient
} from "../src/maskBatchPreview";
import type {
  ComfyNodeExecutionOutput,
  Logger
} from "../src/types";

const OUTPUT: ComfyNodeExecutionOutput = {
  images: [{ filename: "mask.png", subfolder: "", type: "temp" }],
  animated: [false]
};

function deferred<T>(): {
  promise: Promise<T>;
  resolve: (value: T) => void;
} {
  let resolvePromise: ((value: T) => void) | undefined;
  const promise = new Promise<T>((resolve) => {
    resolvePromise = resolve;
  });
  return {
    promise,
    resolve(value: T) {
      if (!resolvePromise) throw new Error("Deferred promise was not initialized.");
      resolvePromise(value);
    }
  };
}

describe("MaskBatchPreviewController", () => {
  it("clears without requesting a preview for an empty native selection", () => {
    const executionEvents = new EventTarget();
    const published: ComfyNodeExecutionOutput[] = [];
    executionEvents.addEventListener("executed", (event) => {
      published.push(
        (event as CustomEvent<{ output: ComfyNodeExecutionOutput }>).detail
          .output
      );
    });
    const client = vi.fn<MaskBatchPreviewClient>();
    const clearNativePreview = vi.fn();
    const controller = new MaskBatchPreviewController(
      executionEvents,
      { id: 3 },
      client,
      console,
      clearNativePreview
    );

    controller.refresh([], "alpha");

    expect(published).toEqual([{ images: [], animated: [] }]);
    expect(clearNativePreview).toHaveBeenCalledOnce();
    expect(client).not.toHaveBeenCalled();
  });

  it("publishes through Comfy's native execution pipeline and clears first", async () => {
    const node = { id: 4 };
    const executionEvents = new EventTarget();
    const published: ComfyNodeExecutionOutput[] = [];
    executionEvents.addEventListener("executed", (event) => {
      published.push(
        (event as CustomEvent<{ output: ComfyNodeExecutionOutput }>).detail
          .output
      );
    });
    const client = vi.fn<MaskBatchPreviewClient>().mockResolvedValue(OUTPUT);
    const controller = new MaskBatchPreviewController(
      executionEvents,
      node,
      client
    );

    controller.refresh(["one.png"], "red");

    expect(published).toEqual([{ images: [], animated: [] }]);
    await vi.waitFor(() => {
      expect(published.at(-1)).toEqual(OUTPUT);
    });
    expect(client).toHaveBeenCalledWith(["one.png"], "red");
  });

  it("discards responses made stale by a newer channel request", async () => {
    const first = deferred<ComfyNodeExecutionOutput>();
    const second = deferred<ComfyNodeExecutionOutput>();
    const client = vi
      .fn<MaskBatchPreviewClient>()
      .mockReturnValueOnce(first.promise)
      .mockReturnValueOnce(second.promise);
    const executionEvents = new EventTarget();
    const published: ComfyNodeExecutionOutput[] = [];
    executionEvents.addEventListener("executed", (event) => {
      published.push(
        (event as CustomEvent<{ output: ComfyNodeExecutionOutput }>).detail
          .output
      );
    });
    const controller = new MaskBatchPreviewController(
      executionEvents,
      { id: 5 },
      client
    );
    const latest: ComfyNodeExecutionOutput = {
      images: [{ filename: "blue.png", subfolder: "", type: "temp" }],
      animated: [false]
    };

    controller.refresh(["mask.png"], "red");
    controller.refresh(["mask.png"], "blue");
    second.resolve(latest);
    await vi.waitFor(() => {
      expect(published.at(-1)).toEqual(latest);
    });
    first.resolve(OUTPUT);
    await Promise.resolve();

    expect(published.at(-1)).toEqual(latest);
  });

  it("keeps the cleared preview and reports current request failures", async () => {
    const error = new Error("preview failed");
    const client = vi.fn<MaskBatchPreviewClient>().mockRejectedValue(error);
    const logger: Logger = { warn: vi.fn() };
    const executionEvents = new EventTarget();
    const published: ComfyNodeExecutionOutput[] = [];
    executionEvents.addEventListener("executed", (event) => {
      published.push(
        (event as CustomEvent<{ output: ComfyNodeExecutionOutput }>).detail
          .output
      );
    });
    const controller = new MaskBatchPreviewController(
      executionEvents,
      { id: 6 },
      client,
      logger
    );

    controller.refresh(["mask.png"], "alpha");

    await vi.waitFor(() => {
      expect(logger.warn).toHaveBeenCalledWith(
        "Could not refresh Load Mask Batch preview.",
        error
      );
    });
    expect(published).toEqual([{ images: [], animated: [] }]);
  });
});
