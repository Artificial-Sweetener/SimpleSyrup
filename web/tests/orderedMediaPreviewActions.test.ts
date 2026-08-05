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
  it("keeps Nodes 2.0 controls within the native preview stacking context", async () => {
    actionFixture(["one.png", "two.png"]);
    const root = document.createElement("section");
    root.dataset.nodeId = "7";
    const previewSurface = document.createElement("div");
    previewSurface.setAttribute("role", "region");
    previewSurface.setAttribute(
      "aria-label",
      "Image preview - Use arrow keys to navigate between images"
    );
    mockRect(previewSurface, 40, 60, 200, 100);
    const firstImage = previewImage("one.png", 50);
    const secondImage = previewImage("two.png", 130);
    mockRect(firstImage, 50, 70, 80, 80);
    mockRect(secondImage, 130, 70, 80, 80);
    previewSurface.append(firstImage, secondImage);
    root.append(previewSurface);
    document.body.append(root);

    await vi.waitFor(() => {
      expect(
        previewSurface.querySelectorAll(".ss-native-preview-affordance")
      ).toHaveLength(2);
    });
    const first = previewSurface.querySelector<HTMLElement>(
      ".ss-native-preview-affordance"
    );
    expect(first?.style.position).toBe("absolute");
    expect(first?.style.left).toBe("10px");
    expect(first?.style.top).toBe("10px");
    expect(first?.style.zIndex).toBe("1");
  });

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

  it("does not claim Nodes 2.0 DOM owned by another open workflow", async () => {
    const inactiveRoot = {};
    const fixture = actionFixture(["one.png", "two.png"], {
      graph: { _rootGraph: inactiveRoot }
    });
    fixture.app.rootGraph = {};
    const root = document.createElement("section");
    root.dataset.nodeId = "7";
    root.append(previewImage("one.png", 0), previewImage("two.png", 100));
    document.body.append(root);
    document.dispatchEvent(new MouseEvent("click"));

    await new Promise((resolve) => setTimeout(resolve, 100));

    expect(document.querySelectorAll(".ss-native-preview-affordance")).toHaveLength(0);
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

  it("anchors embedded Nodes 1.0 controls to SugarCubes' clipped face projection", async () => {
    const canvas = document.createElement("canvas");
    const container = document.createElement("div");
    mockRect(canvas, 0, 0, 800, 600);
    mockRect(container, 200, 300, 320, 240);
    const projectRect = vi.fn(
      ([left, top, width, height]: readonly [number, number, number, number]) => ({
        left: 200 + left,
        top: 300 + top,
        width,
        height
      })
    );
    const projectionSymbol = Symbol.for("sugarcubes.cube-face-projection.v1");
    const fixture = actionFixture(
      ["one.png", "two.png"],
      {
        pos: [0, 0],
        imageIndex: null,
        imageRects: [
          [10, 30, 80, 80],
          [90, 30, 80, 80]
        ],
        [projectionSymbol]: { container, projectRect }
      }
    );
    const nativeProjection = vi.fn(() => {
      throw new Error("Root graph projection must not run for an embedded node.");
    });
    fixture.app.canvas = {
      canvas,
      convertEventToCanvasOffset: (event) => [event.clientX, event.clientY],
      convertOffsetToCanvas: nativeProjection
    };
    document.body.append(canvas, container);

    await vi.waitFor(() => {
      expect(container.querySelectorAll(".ss-native-preview-affordance")).toHaveLength(2);
    });
    const first = container.querySelector<HTMLElement>(
      ".ss-native-preview-affordance"
    );
    expect(projectRect).toHaveBeenCalledWith([10, 30, 80, 80]);
    expect(nativeProjection).not.toHaveBeenCalled();
    expect(first?.style.position).toBe("absolute");
    expect(first?.style.left).toBe("10px");
    expect(first?.style.top).toBe("30px");
    expect(first?.style.width).toBe("80px");
  });

  it("positions selected-item controls in Nodes 1.0 detail view", async () => {
    const canvas = document.createElement("canvas");
    mockRect(canvas, 10, 20, 500, 500);
    const fixture = actionFixture(["one.png", "two.png"], {
      pos: [100, 200],
      size: [160, 260],
      imageIndex: 1,
      widgets: [
        {
          y: 30,
          computedHeight: 200,
          options: { canvasOnly: true }
        }
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
      expect(document.querySelectorAll(".ss-native-preview-affordance")).toHaveLength(1);
    });
    expect(
      document.querySelector('[aria-label="Move image 2 earlier"]')
    ).not.toBeNull();
    expect(
      document.querySelector<HTMLButtonElement>(
        '[aria-label="Move image 2 later"]'
      )?.disabled
    ).toBe(true);
    expect(
      document.querySelector<HTMLElement>(".ss-native-preview-affordance")
        ?.style.width
    ).toBe("160px");
  });

  it("reuses the Nodes 1.0 grid footprint after detail clears imageRects", async () => {
    const canvas = document.createElement("canvas");
    mockRect(canvas, 10, 20, 500, 500);
    const node = {
      id: 7,
      pos: [100, 200] as const,
      imageIndex: null as number | null,
      imageRects: [
        [0, 30, 80, 80],
        [80, 30, 80, 80]
      ] as Array<readonly [number, number, number, number]>
    };
    const fixture = actionFixture(["one.png", "two.png"], node);
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

    fixture.node.imageIndex = 1;
    delete (fixture.node as { imageRects?: unknown }).imageRects;
    document.dispatchEvent(new MouseEvent("click"));

    await vi.waitFor(() => {
      expect(document.querySelectorAll(".ss-native-preview-affordance")).toHaveLength(1);
    });
    expect(
      document.querySelector('[aria-label="Move image 2 earlier"]')
    ).not.toBeNull();
    expect(
      document.querySelector<HTMLElement>(".ss-native-preview-affordance")
        ?.style.width
    ).toBe("160px");
  });

  it("keeps selected-item controls while native detail view is open", async () => {
    const node = { id: 7 };
    const fixture = actionFixture(["one.png", "two.png"], node);
    const { root, select } = nativeDetailPreview();
    document.body.append(root);
    select(0);

    await vi.waitFor(() => {
      expect(document.querySelectorAll(".ss-native-preview-affordance")).toHaveLength(1);
    });
    document
      .querySelector<HTMLButtonElement>('[aria-label="Move image 1 later"]')
      ?.click();
    await vi.waitFor(() => {
      expect(
        document.querySelector('[aria-label="Remove image 2"]')
      ).not.toBeNull();
    });
    expect(fixture.moveLater).toHaveBeenCalledWith(0);
    expect(
      root.querySelector('[aria-label="View image 2 of 2"]')?.getAttribute(
        "aria-current"
      )
    ).toBe("true");

    select(-1);
    fixture.preview.publish({
      images: references("two.png", "one.png"),
      animated: [false, false]
    });
    await vi.waitFor(() => {
      expect(
        root.querySelector('[aria-label="View image 2 of 2"]')?.getAttribute(
          "aria-current"
        )
      ).toBe("true");
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
    imageRects?: Array<readonly [number, number, number, number]>;
    widgets?: Array<{
      y?: number;
      computedHeight?: number;
      options?: { canvasOnly?: boolean };
    }>;
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
  return { adapter, app, moveEarlier, moveLater, node, preview, remove };
}

function previewImage(filename: string, left: number): HTMLImageElement {
  const image = document.createElement("img");
  image.src = `/api/view?filename=${filename}&subfolder=&type=input&preview=webp`;
  mockRect(image, left, 0, 80, 80);
  return image;
}

function nativeDetailPreview(): {
  root: HTMLElement;
  select: (index: number) => void;
} {
  const root = document.createElement("section");
  root.dataset.nodeId = "7";
  const region = document.createElement("div");
  region.setAttribute("role", "region");
  region.setAttribute(
    "aria-label",
    "Image preview - Use arrow keys to navigate between images"
  );
  mockRect(region, 20, 30, 160, 200);
  const navigation = document.createElement("div");
  const select = (index: number): void => {
    for (const [buttonIndex, button] of buttons.entries()) {
      if (buttonIndex === index) button.setAttribute("aria-current", "true");
      else button.removeAttribute("aria-current");
    }
  };
  const buttons = [0, 1].map((index) => {
    const button = document.createElement("button");
    button.setAttribute(
      "aria-label",
      `View image ${String(index + 1)} of 2`
    );
    button.addEventListener("click", () => {
      select(index);
    });
    return button;
  });
  navigation.append(...buttons);
  root.append(region, navigation);
  return { root, select };
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
