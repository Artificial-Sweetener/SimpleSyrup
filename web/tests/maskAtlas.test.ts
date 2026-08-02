// SimpleSyrup - workflow-focused ComfyUI extensions for image generation
// Copyright (C) 2026  Artificial Sweetener and contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

import { describe, expect, it } from "vitest";

import { MaskAtlas } from "../src/maskAtlas";
import type { SegPreviewDocument } from "../src/segPreviewTypes";

describe("MaskAtlas", () => {
  it("returns nested hits from smallest to largest and respects mask holes", () => {
    const document = previewDocument();
    const pixels = new Uint8ClampedArray(4 * 2 * 4);
    setMaskPixel(pixels, 4, 0, 0, 255);
    setMaskPixel(pixels, 4, 1, 0, 255);
    setMaskPixel(pixels, 4, 2, 0, 255);
    setMaskPixel(pixels, 4, 3, 0, 255);
    setMaskPixel(pixels, 4, 0, 1, 0);
    setMaskPixel(pixels, 4, 1, 1, 255);
    setMaskPixel(pixels, 4, 2, 1, 255);
    setMaskPixel(pixels, 4, 3, 1, 255);
    const atlas = new MaskAtlas(document, pixels);

    expect(atlas.hitsAt(25, 25).map((region) => region.id)).toEqual([
      "small",
      "large"
    ]);
    expect(atlas.hitsAt(75, 25).map((region) => region.id)).toEqual(["large"]);
    expect(atlas.hitsAt(25, 75)).toEqual([]);
    expect(atlas.hitsAt(150, 50)).toEqual([]);
  });
});

function previewDocument(): SegPreviewDocument {
  const image = { filename: "asset.png", subfolder: "", type: "temp" as const };
  return {
    version: 1,
    source: { width: 200, height: 100 },
    preview: { width: 200, height: 100, image },
    atlas: { width: 4, height: 2, image },
    regions: [
      {
        id: "large",
        index: 0,
        label: "subject",
        confidence: 1,
        area: 10000,
        color: "#ff0000",
        crop: { x: 0, y: 0, width: 100, height: 100 },
        atlas: { x: 0, y: 0, width: 2, height: 2 }
      },
      {
        id: "small",
        index: 1,
        label: "detail",
        confidence: 1,
        area: 2500,
        color: "#00ff00",
        crop: { x: 0, y: 0, width: 50, height: 50 },
        atlas: { x: 2, y: 0, width: 2, height: 2 }
      }
    ]
  };
}

function setMaskPixel(
  pixels: Uint8ClampedArray,
  width: number,
  x: number,
  y: number,
  value: number
): void {
  pixels[(y * width + x) * 4] = value;
}
