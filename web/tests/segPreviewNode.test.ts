// SimpleSyrup - workflow-focused ComfyUI extensions for image generation
// Copyright (C) 2026  Artificial Sweetener and contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

import { afterEach, describe, expect, it, vi } from "vitest";

import { registerSimplePreviewSEGS } from "../src/segPreviewNode";
import type {
  ComfyApi,
  ComfyNodeExecutionOutput,
  ComfyExtension
} from "../src/types";
import { createFakeComfyApp, installCanvasMock } from "./testUtils";

describe("Simple Preview SEGS node integration", () => {
  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it.each(["Nodes 1.0", "Nodes 2.0"])(
    "uses one native DOM widget under %s",
    async (renderer) => {
      installCanvasMock();
      const app = createFakeComfyApp();
      const api = new EventTarget() as ComfyApi;
      api.apiURL = (path) => `/base${path}`;
      const loadImage = vi.fn(() => Promise.resolve(document.createElement("img")));
      registerSimplePreviewSEGS(app, api, loadImage);
      const extension = requiredExtension(app.extensions);
      const node = fakeNode(renderer === "Nodes 2.0" ? 22 : 11);

      await extension.nodeCreated?.(node);

      expect(node.addDOMWidget).toHaveBeenCalledOnce();
      const widgetOptions = node.addDOMWidget.mock.calls[0]?.[3];
      expect(widgetOptions).toEqual({ serialize: false, canvasOnly: false });
      expect(node.setSize).toHaveBeenCalledWith([420, 520]);

      if (renderer === "Nodes 1.0") {
        node.onExecuted?.(executionOutput());
      } else {
        await extension.onNodeOutputsUpdated?.({
          [String(node.id)]: executionOutput()
        });
      }
      const root = node.previewRoot;
      await vi.waitFor(() => {
        expect(root.querySelector(".ss-segs-preview__canvas-stack")).not.toBeNull();
      });

      node.onRemoved?.();
      expect(root.querySelector(".ss-segs-preview__canvas-stack")).toBeNull();
    }
  );
});

interface FakeDomWidget {
  serialize?: boolean;
  computeSize?: (width?: number) => [number, number];
  options: { serialize?: boolean; canvasOnly?: boolean };
}

interface FakePreviewNode {
  constructor: { comfyClass: string };
  id: number;
  size: [number, number];
  previewRoot: HTMLElement;
  computeSize: ReturnType<typeof vi.fn<() => [number, number]>>;
  setSize: ReturnType<typeof vi.fn<(size: [number, number]) => void>>;
  addDOMWidget: ReturnType<
    typeof vi.fn<
      (
        name: string,
        type: string,
        element: HTMLElement,
        options?: { serialize?: boolean; canvasOnly?: boolean }
      ) => FakeDomWidget
    >
  >;
  onExecuted: ((output: unknown) => void) | undefined;
  onGraphConfigured: ((...args: unknown[]) => unknown) | undefined;
  onRemoved: ((...args: unknown[]) => unknown) | undefined;
}

function fakeNode(id: number): FakePreviewNode {
  const previewRoot: HTMLElement = document.createElement("div");
  const node = {
    constructor: { comfyClass: "SimpleSyrup.SimplePreviewSEGS" },
    id,
    size: [300, 200] as [number, number],
    previewRoot,
    computeSize: vi.fn((): [number, number] => [300, 240]),
    setSize: vi.fn<(size: [number, number]) => void>(),
    addDOMWidget: vi.fn(
      (
        _name: string,
        _type: string,
        element: HTMLElement,
        options?: { serialize?: boolean; canvasOnly?: boolean }
      ): FakeDomWidget => {
        void options;
        previewRoot.replaceWith(element);
        node.previewRoot = element;
        return { options: {} };
      }
    ),
    onExecuted: undefined as ((output: unknown) => void) | undefined,
    onGraphConfigured: undefined as ((...args: unknown[]) => unknown) | undefined,
    onRemoved: undefined as ((...args: unknown[]) => unknown) | undefined
  };
  return node;
}

function executionOutput(): ComfyNodeExecutionOutput & {
  simple_syrup_segs_preview: unknown[];
} {
  const image = { filename: "asset.png", subfolder: "", type: "temp" as const };
  return {
    images: [],
    simple_syrup_segs_preview: [
      {
        version: 1,
        source: { width: 4, height: 4 },
        preview: { width: 4, height: 4, image },
        atlas: { width: 1, height: 1, image },
        regions: []
      }
    ]
  };
}

function requiredExtension(
  extensions: ReturnType<typeof createFakeComfyApp>["extensions"]
): ComfyExtension {
  const extension = extensions.find(
    (candidate) => candidate.name === "SimpleSyrup.SimplePreviewSEGS"
  );
  if (!extension) throw new Error("Simple Preview SEGS extension was not registered.");
  return extension;
}
