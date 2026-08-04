// SimpleSyrup - workflow-focused ComfyUI extensions for image generation
// Copyright (C) 2026  Artificial Sweetener and contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

import { afterEach, describe, expect, it, vi } from "vitest";

import { NativePreviewNavigator } from "../src/nativePreviewNavigator";
import type { ComfyApp } from "../src/types";
import { createFakeComfyApp } from "./testUtils";

describe("NativePreviewNavigator", () => {
  afterEach(() => {
    vi.useRealTimers();
    document.body.replaceChildren();
  });

  it("clicks the matching Nodes 2.0 native gallery item", () => {
    const app = previewApp();
    const node = { id: 7, imageIndex: null as number | null };
    const root = document.createElement("section");
    root.dataset.nodeId = "7";
    const first = nativeImageButton("first.png");
    const second = nativeImageButton("second.png");
    root.append(first.button, second.button);
    document.body.append(root);
    const clicked = vi.fn();
    second.button.addEventListener("click", clicked);

    const navigator = new NativePreviewNavigator(app, node);
    navigator.inspect(1);

    expect(clicked).toHaveBeenCalledOnce();
    expect(node.imageIndex).toBeNull();
    navigator.dispose();
  });

  it("waits for a restored Nodes 2.0 gallery before opening detail", async () => {
    const app = previewApp();
    const node = { id: 7, imageIndex: null as number | null };
    const root = document.createElement("section");
    root.dataset.nodeId = "7";
    document.body.append(root);
    const navigator = new NativePreviewNavigator(app, node);
    const second = nativeImageButton("second.png");
    const clicked = vi.fn();
    second.button.addEventListener("click", clicked);

    navigator.inspect(1);
    root.append(nativeImageButton("first.png").button, second.button);

    await vi.waitFor(() => {
      expect(clicked).toHaveBeenCalledOnce();
    });
    navigator.dispose();
  });

  it("uses Comfy's native return-to-grid control in Nodes 2.0", () => {
    const app = previewApp();
    const node = { id: 7, imageIndex: null as number | null };
    const root = document.createElement("section");
    root.dataset.nodeId = "7";
    const grid = document.createElement("button");
    grid.setAttribute("aria-label", "Grid view");
    const clicked = vi.fn();
    grid.addEventListener("click", clicked);
    root.append(grid);
    document.body.append(root);
    const navigator = new NativePreviewNavigator(app, node);

    navigator.showGrid();

    expect(clicked).toHaveBeenCalledOnce();
    navigator.dispose();
  });

  it("uses the Nodes 1.0 image index when no DOM gallery is mounted", async () => {
    const app = createFakeComfyApp();
    const dirty = vi.fn();
    const node = {
      id: 7,
      imageIndex: null as number | null,
      imageRects: [[0, 0, 10, 10]],
      imgs: [document.createElement("img"), document.createElement("img")],
      graph: { setDirtyCanvas: dirty }
    };

    const navigator = new NativePreviewNavigator(app, node);
    navigator.inspect(1);

    await vi.waitFor(() => {
      expect(node.imageIndex).toBe(1);
      expect(node.imageRects).toBeUndefined();
      expect(dirty).toHaveBeenCalledWith(true, true);
    });
    navigator.dispose();
  });

  it("uses Nodes 1.0 native pointer selection when the canvas is available", async () => {
    const app = createFakeComfyApp();
    app.canvas = { graph_mouse: [12, 34] } as unknown as NonNullable<
      ComfyApp["canvas"]
    >;
    const dirty = vi.fn<(foreground: boolean, background: boolean) => void>();
    const node: {
      id: number;
      imageIndex: number | null;
      imgs: HTMLImageElement[];
      pointerDown?: { index: number | null; pos: [number, number] };
      graph: {
        setDirtyCanvas: (foreground: boolean, background: boolean) => void;
      };
    } = {
      id: 7,
      imageIndex: null,
      imgs: [document.createElement("img"), document.createElement("img")],
      graph: { setDirtyCanvas: dirty }
    };

    const navigator = new NativePreviewNavigator(app, node);
    navigator.inspect(1);

    await vi.waitFor(() => {
      expect(node.pointerDown).toEqual({ index: 1, pos: [12, 34] });
      expect(node.imageIndex).toBe(1);
      expect(dirty).toHaveBeenCalledWith(true, true);
    });
    navigator.dispose();
  });

  it("waits for Nodes 1.0 images before selecting native detail", async () => {
    vi.useFakeTimers();
    vi.stubGlobal(
      "requestAnimationFrame",
      (callback: FrameRequestCallback): number =>
        window.setTimeout(() => {
          callback(performance.now());
        }, 16)
    );
    vi.stubGlobal("cancelAnimationFrame", (handle: number): void => {
      window.clearTimeout(handle);
    });
    const app = createFakeComfyApp();
    const node: {
      id: number;
      imageIndex: number | null;
      imgs?: HTMLImageElement[];
      graph: {
        setDirtyCanvas: ReturnType<
          typeof vi.fn<(foreground: boolean, background: boolean) => void>
        >;
      };
    } = {
      id: 7,
      imageIndex: null as number | null,
      graph: {
        setDirtyCanvas:
          vi.fn<(foreground: boolean, background: boolean) => void>()
      }
    };
    const navigator = new NativePreviewNavigator(app, node);

    navigator.inspect(1);
    await vi.advanceTimersByTimeAsync(16);

    expect(node.imageIndex).toBeNull();

    node.imgs = [document.createElement("img"), document.createElement("img")];
    await vi.advanceTimersByTimeAsync(16);

    expect(node.imageIndex).toBe(1);
    navigator.dispose();
  });

});

function previewApp(): ComfyApp {
  const app = createFakeComfyApp();
  app.nodeOutputs = {
    "7": {
      images: [reference("first.png"), reference("second.png")]
    }
  };
  return app;
}

function reference(filename: string) {
  return { filename, subfolder: "", type: "temp" as const };
}

function nativeImageButton(filename: string): {
  button: HTMLButtonElement;
  image: HTMLImageElement;
} {
  const button = document.createElement("button");
  const image = document.createElement("img");
  image.src = `/view?filename=${filename}&subfolder=&type=temp`;
  button.append(image);
  return { button, image };
}
