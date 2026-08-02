// SimpleSyrup - workflow-focused ComfyUI extensions for image generation
// Copyright (C) 2026  Artificial Sweetener and contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

import type { Dimensions, Rectangle } from "./segPreviewTypes";

export interface Point {
  x: number;
  y: number;
}

/** Map pointer and source geometry into a displayed image viewport. */
export class ImageViewport {
  constructor(
    private readonly source: Dimensions,
    private readonly display: Dimensions
  ) {}

  /** Convert browser pointer coordinates into source-image coordinates. */
  sourcePoint(clientX: number, clientY: number, bounds: DOMRect): Point {
    const normalizedX = clamp((clientX - bounds.left) / Math.max(1, bounds.width));
    const normalizedY = clamp((clientY - bounds.top) / Math.max(1, bounds.height));
    return {
      x: normalizedX * this.source.width,
      y: normalizedY * this.source.height
    };
  }

  /** Project one source rectangle into the preview canvas. */
  displayRectangle(sourceRectangle: Rectangle): Rectangle {
    const scaleX = this.display.width / this.source.width;
    const scaleY = this.display.height / this.source.height;
    return {
      x: sourceRectangle.x * scaleX,
      y: sourceRectangle.y * scaleY,
      width: sourceRectangle.width * scaleX,
      height: sourceRectangle.height * scaleY
    };
  }
}

function clamp(value: number): number {
  return Math.max(0, Math.min(1, value));
}
