// SimpleSyrup - workflow-focused ComfyUI extensions for image generation
// Copyright (C) 2026  Artificial Sweetener and contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

import { afterEach, describe, expect, it, vi } from "vitest";

import { SegPreviewInspector } from "../src/segPreviewInspector";
import type { SegPreviewDocument } from "../src/segPreviewTypes";
import { installCanvasMock } from "./testUtils";

describe("SegPreviewInspector", () => {
  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it("hands overlay selection to Comfy's native preview navigator", async () => {
    installCanvasMock();
    const loadImage = vi.fn(() => Promise.resolve(document.createElement("img")));
    const inspectRegion = vi.fn<(index: number) => void>();
    const changeMode = vi.fn<(mode: "overlay" | "grid") => void>();
    const inspector = new SegPreviewInspector(
      inspectRegion,
      changeMode,
      (path) => path,
      loadImage
    );
    const prepared = await inspector.prepare(previewDocument());
    prepared.commit();
    const highlight = requiredElement(
      inspector.element,
      ".ss-segs-preview__highlight"
    );
    if (!(highlight instanceof HTMLCanvasElement)) {
      throw new Error("Missing highlight canvas.");
    }
    highlight.getBoundingClientRect = () => new DOMRect(0, 0, 100, 100);

    highlight.dispatchEvent(
      new MouseEvent("pointermove", { clientX: 50, clientY: 50, bubbles: true })
    );
    expect(inspector.element.textContent).toContain("subject");
    highlight.click();

    expect(changeMode).not.toHaveBeenCalled();
    expect(inspectRegion).toHaveBeenCalledWith(0);
    expect(
      inspector.element.querySelector(".ss-segs-preview__canvas-stack")
    ).toBeNull();
  });

  it("switches to the native grid without implementing another gallery", async () => {
    installCanvasMock();
    const loadImage = vi.fn(() => Promise.resolve(document.createElement("img")));
    const changeMode = vi.fn<(mode: "overlay" | "grid") => void>();
    const inspector = new SegPreviewInspector(
      () => undefined,
      changeMode,
      (path) => path,
      loadImage
    );
    const prepared = await inspector.prepare(previewDocument());
    prepared.commit();

    expect(inspector.element.querySelector(".ss-segs-preview__grid")).toBeNull();
    expect(
      inspector.element.querySelector(".ss-segs-preview__inspection")
    ).toBeNull();
    const buttons = Array.from(inspector.element.querySelectorAll("button"));
    expect(buttons.map((button) => button.textContent)).toEqual([
      "Overlay",
      "Grid"
    ]);

    buttons[1]?.click();

    expect(changeMode).toHaveBeenCalledWith("grid");
    expect(inspector.element.dataset.mode).toBe("grid");
    expect(inspector.element.querySelector(".ss-segs-preview__body")?.childNodes)
      .toHaveLength(0);
    expect(inspector.preferredHeight()).toBe(38);
  });

  it("switches to an actionable error without throwing from the node lifecycle", () => {
    installCanvasMock();
    const inspector = new SegPreviewInspector();

    inspector.showError("Atlas could not be decoded.");

    expect(inspector.element.textContent).toContain("Atlas could not be decoded.");
    expect(
      inspector.element.querySelector(".ss-segs-preview__message")?.getAttribute(
        "data-error"
      )
    ).toBe("true");
  });
});

function previewDocument(): SegPreviewDocument {
  const image = { filename: "asset.png", subfolder: "", type: "temp" as const };
  return {
    version: 1,
    source: { width: 100, height: 100 },
    preview: { width: 100, height: 100, image },
    atlas: { width: 2, height: 2, image },
    regions: [
      {
        id: "seg-0001",
        index: 0,
        label: "subject",
        confidence: 0.95,
        area: 10000,
        color: "#f24236",
        crop: { x: 0, y: 0, width: 100, height: 100 },
        atlas: { x: 0, y: 0, width: 2, height: 2 }
      }
    ]
  };
}

function requiredElement(root: ParentNode, selector: string): Element {
  const element = root.querySelector(selector);
  if (!element) throw new Error(`Missing '${selector}' element.`);
  return element;
}
