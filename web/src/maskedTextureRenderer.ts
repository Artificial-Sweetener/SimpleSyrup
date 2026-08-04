// SimpleSyrup - workflow-focused ComfyUI extensions for image generation
// Copyright (C) 2026  Artificial Sweetener and contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

import type { Dimensions, Rectangle } from "./segPreviewTypes";

/** Render source-image texture through an alpha mask onto a transparent canvas. */
export class MaskedTextureRenderer {
  constructor(private readonly sourceImage: CanvasImageSource) {}

  /** Fit and render one source crop while making every non-mask pixel transparent. */
  render(
    sourceCrop: Rectangle,
    mask: CanvasImageSource,
    maximum: Dimensions
  ): HTMLCanvasElement {
    const size = fitWithin(sourceCrop, maximum);
    const canvas = document.createElement("canvas");
    canvas.width = size.width;
    canvas.height = size.height;
    const context = canvas.getContext("2d");
    if (!context) {
      throw new Error("Simple Preview SEGS requires canvas rendering.");
    }
    context.save();
    context.drawImage(
      this.sourceImage,
      sourceCrop.x,
      sourceCrop.y,
      sourceCrop.width,
      sourceCrop.height,
      0,
      0,
      size.width,
      size.height
    );
    context.globalCompositeOperation = "destination-in";
    context.drawImage(mask, 0, 0, size.width, size.height);
    context.restore();
    return canvas;
  }
}

/** Fit dimensions inside a bounding box without upscaling unavailable texture. */
function fitWithin(source: Dimensions, maximum: Dimensions): Dimensions {
  const scale = Math.min(
    1,
    maximum.width / source.width,
    maximum.height / source.height
  );
  return {
    width: Math.max(1, Math.round(source.width * scale)),
    height: Math.max(1, Math.round(source.height * scale))
  };
}
