// SimpleSyrup - workflow-focused ComfyUI extensions for image generation
// Copyright (C) 2026  Artificial Sweetener and contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

import { describe, expect, it, vi } from "vitest";

import { OrderedMediaPreviewController } from "../src/orderedMediaPreview";
import type { ComfyNodeExecutionOutput, Logger } from "../src/types";

const OUTPUT: ComfyNodeExecutionOutput = {
  images: [{ filename: "mask.png", subfolder: "", type: "temp" }],
  animated: [false]
};

describe("OrderedMediaPreviewController", () => {
  it("renders empty selections without making a preview request", () => {
    const target = fakeTarget();
    const client = vi.fn();
    const controller = new OrderedMediaPreviewController(target, "mask loader");

    controller.refresh([], client);

    expect(target.clear).toHaveBeenCalledOnce();
    expect(client).not.toHaveBeenCalled();
  });

  it("loads and publishes exact ordered references", async () => {
    const target = fakeTarget();
    const controller = new OrderedMediaPreviewController(target, "mask loader");

    controller.refresh(["mask.png"], () => Promise.resolve(OUTPUT));

    await vi.waitFor(() => {
      expect(target.publish).toHaveBeenCalledWith(OUTPUT);
    });
  });

  it("discards stale responses after a newer channel request", async () => {
    const first = deferred<ComfyNodeExecutionOutput>();
    const second = deferred<ComfyNodeExecutionOutput>();
    const target = fakeTarget();
    const controller = new OrderedMediaPreviewController(target, "mask loader");
    const latest: ComfyNodeExecutionOutput = {
      images: [{ filename: "blue.png", subfolder: "", type: "temp" }]
    };

    controller.refresh(["mask.png"], () => first.promise);
    controller.refresh(["mask.png"], () => second.promise);
    second.resolve(latest);
    await vi.waitFor(() => {
      expect(target.publish).toHaveBeenCalledWith(latest);
    });
    first.resolve(OUTPUT);
    await Promise.resolve();

    expect(target.publish).toHaveBeenCalledTimes(1);
  });

  it("keeps failures visible and invalidates pending work on dispose", async () => {
    const error = new Error("preview failed");
    const logger: Logger = { warn: vi.fn() };
    const target = fakeTarget();
    const controller = new OrderedMediaPreviewController(target, "mask loader", logger);

    controller.refresh(["mask.png"], () => Promise.reject(error));
    await vi.waitFor(() => {
      expect(target.clear).toHaveBeenCalledOnce();
    });
    expect(logger.warn).toHaveBeenCalledWith(
      "Could not refresh mask loader preview: preview failed",
      error
    );

    const pending = deferred<ComfyNodeExecutionOutput>();
    controller.refresh(["mask.png"], () => pending.promise);
    controller.dispose();
    pending.resolve(OUTPUT);
    await Promise.resolve();
    expect(target.publish).not.toHaveBeenCalled();
  });

  it("preserves the last native preview when a replacement fails", async () => {
    const error = new Error("replacement failed");
    const logger: Logger = { warn: vi.fn() };
    const target = fakeTarget();
    const controller = new OrderedMediaPreviewController(
      target,
      "image loader",
      logger
    );
    controller.refresh(["mask.png"], () => Promise.resolve(OUTPUT));
    await vi.waitFor(() => {
      expect(target.publish).toHaveBeenCalledWith(OUTPUT);
    });

    controller.refresh(["other.png"], () => Promise.reject(error));
    await vi.waitFor(() => {
      expect(logger.warn).toHaveBeenCalledWith(
        "Could not refresh image loader preview: replacement failed",
        error
      );
    });

    expect(target.clear).not.toHaveBeenCalled();
  });
});

function fakeTarget() {
  return {
    publish: vi.fn(),
    clear: vi.fn()
  };
}

function deferred<T>(): { promise: Promise<T>; resolve(value: T): void } {
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
