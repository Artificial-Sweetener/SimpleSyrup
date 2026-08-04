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
    document.body.replaceChildren();
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
      if (renderer === "Nodes 2.0") {
        const nativeRoot = document.createElement("section");
        nativeRoot.dataset.nodeId = String(node.id);
        document.body.append(nativeRoot);
      }

      await extension.nodeCreated?.(node);

      expect(node.addDOMWidget).toHaveBeenCalledOnce();
      const widgetOptions = node.addDOMWidget.mock.calls[0]?.[3];
      expect(widgetOptions).toEqual({ serialize: false, canvasOnly: false });
      const previewWidget = node.addDOMWidget.mock.results[0]?.value as
        | FakeDomWidget
        | undefined;
      expect(previewWidget?.computeLayoutSize).toBeUndefined();
      expect(node.setSize).toHaveBeenCalledWith([420, 420]);

      const output = executionOutput();
      app.nodeOutputs ??= {};
      app.nodeOutputs[String(node.id)] = output;
      if (renderer === "Nodes 1.0") {
        node.onExecuted?.(output);
      } else {
        await extension.onNodeOutputsUpdated?.({
          [String(node.id)]: output
        });
      }
      const root = node.previewRoot;
      await vi.waitFor(() => {
        expect(root.querySelector(".ss-segs-preview__canvas-stack")).not.toBeNull();
      });
      expect(app.nodeOutputs[String(node.id)]?.images).toEqual(
        renderer === "Nodes 1.0"
          ? [{ filename: "region.png", subfolder: "", type: "temp" }]
          : []
      );
      expect(root.querySelector(".ss-segs-preview__grid")).toBeNull();

      modeButton(root, "Grid").click();

      expect(node.arrange).toHaveBeenCalled();
      expect(app.nodeOutputs[String(node.id)]?.images).toEqual([
        { filename: "region.png", subfolder: "", type: "temp" }
      ]);
      expect(root.querySelector(".ss-segs-preview__canvas-stack")).toBeNull();
      expect(root.querySelector(".ss-segs-preview__grid")).toBeNull();

      modeButton(root, "Overlay").click();

      expect(app.nodeOutputs[String(node.id)]?.images).toEqual(
        renderer === "Nodes 1.0"
          ? [{ filename: "region.png", subfolder: "", type: "temp" }]
          : []
      );
      expect(root.querySelector(".ss-segs-preview__canvas-stack")).not.toBeNull();

      node.onRemoved?.();
      expect(root.querySelector(".ss-segs-preview__canvas-stack")).toBeNull();
    }
  );
});

interface FakeDomWidget {
  serialize?: boolean;
  computeSize?: (width?: number) => [number, number];
  computeLayoutSize?: () => {
    minHeight: number;
    minWidth: number;
  };
  options: { serialize?: boolean; canvasOnly?: boolean };
}

interface FakePreviewNode {
  constructor: { comfyClass: string };
  id: number;
  size: [number, number];
  previewRoot: HTMLElement;
  computeSize: ReturnType<typeof vi.fn<() => [number, number]>>;
  setSize: ReturnType<typeof vi.fn<(size: [number, number]) => void>>;
  arrange: ReturnType<typeof vi.fn>;
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
  imageIndex: number | null;
  imageRects?: readonly unknown[];
  imgs?: HTMLImageElement[];
  graph: { setDirtyCanvas: ReturnType<typeof vi.fn> };
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
    arrange: vi.fn(),
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
        return {
          options: {},
          computeLayoutSize: () => ({ minHeight: 50, minWidth: 0 })
        };
      }
    ),
    onExecuted: undefined as ((output: unknown) => void) | undefined,
    onGraphConfigured: undefined as ((...args: unknown[]) => unknown) | undefined,
    onRemoved: undefined as ((...args: unknown[]) => unknown) | undefined,
    imageIndex: null,
    imgs: [],
    graph: { setDirtyCanvas: vi.fn() }
  };
  return node;
}

function executionOutput(): ComfyNodeExecutionOutput & {
  simple_syrup_segs_preview: unknown[];
} {
  const image = { filename: "asset.png", subfolder: "", type: "temp" as const };
  return {
    images: [
      { filename: "region.png", subfolder: "", type: "temp" as const }
    ],
    simple_syrup_segs_preview: [
      {
        version: 1,
        source: { width: 4, height: 4 },
        preview: { width: 4, height: 4, image },
        atlas: { width: 1, height: 1, image },
        regions: [
          {
            id: "seg-0001",
            index: 0,
            label: "subject",
            confidence: 0.9,
            area: 16,
            color: "#f24236",
            crop: { x: 0, y: 0, width: 4, height: 4 },
            atlas: { x: 0, y: 0, width: 1, height: 1 }
          }
        ]
      }
    ]
  };
}

function modeButton(root: ParentNode, label: string): HTMLButtonElement {
  const button = Array.from(root.querySelectorAll("button")).find(
    (candidate) => candidate.textContent === label
  );
  if (!button) throw new Error(`Missing ${label} mode button.`);
  return button;
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
