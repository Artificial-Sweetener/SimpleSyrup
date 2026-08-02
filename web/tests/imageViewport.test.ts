// SimpleSyrup - workflow-focused ComfyUI extensions for image generation
// Copyright (C) 2026  Artificial Sweetener and contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

import { describe, expect, it } from "vitest";

import { ImageViewport } from "../src/imageViewport";

describe("ImageViewport", () => {
  it("maps displayed pointer and crop geometry to the source image", () => {
    const viewport = new ImageViewport(
      { width: 2000, height: 1000 },
      { width: 1000, height: 500 }
    );
    const bounds = new DOMRect(100, 50, 500, 250);

    expect(viewport.sourcePoint(350, 175, bounds)).toEqual({ x: 1000, y: 500 });
    expect(
      viewport.displayRectangle({ x: 200, y: 100, width: 600, height: 300 })
    ).toEqual({ x: 100, y: 50, width: 300, height: 150 });
  });
});
