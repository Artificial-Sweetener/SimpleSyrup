// SimpleSyrup - workflow-focused ComfyUI extensions for image generation
// Copyright (C) 2026  Artificial Sweetener and contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

import { afterEach, describe, expect, it, vi } from "vitest";

import { MaskedTextureRenderer } from "../src/maskedTextureRenderer";

describe("MaskedTextureRenderer", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("keeps source texture inside the mask and transparency outside it", () => {
    const events: string[] = [];
    const drawImage = vi.fn(() => {
      events.push("draw");
    });
    let compositeOperation = "source-over";
    const context = {
      drawImage,
      restore: vi.fn(),
      save: vi.fn()
    } as unknown as CanvasRenderingContext2D;
    Object.defineProperty(context, "globalCompositeOperation", {
      get: () => compositeOperation,
      set: (value: string) => {
        compositeOperation = value;
        events.push(value);
      }
    });
    vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(context);
    const image = document.createElement("img");
    const mask = document.createElement("canvas");

    const canvas = new MaskedTextureRenderer(image).render(
      { x: 10, y: 20, width: 200, height: 100 },
      mask,
      { width: 160, height: 160 }
    );

    expect({ width: canvas.width, height: canvas.height }).toEqual({
      width: 160,
      height: 80
    });
    expect(events).toEqual(["draw", "destination-in", "draw"]);
    expect(drawImage).toHaveBeenNthCalledWith(
      1,
      image,
      10,
      20,
      200,
      100,
      0,
      0,
      160,
      80
    );
    expect(drawImage).toHaveBeenNthCalledWith(2, mask, 0, 0, 160, 80);
  });
});
