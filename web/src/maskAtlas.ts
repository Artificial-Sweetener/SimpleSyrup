// SimpleSyrup - workflow-focused ComfyUI extensions for image generation
// Copyright (C) 2026  Artificial Sweetener and contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

import type {
  SegPreviewDocument,
  SegPreviewRegion
} from "./segPreviewTypes";

/** Provide mask sampling and RGBA extraction from one packed atlas image. */
export class MaskAtlas {
  constructor(
    private readonly document: SegPreviewDocument,
    private readonly pixels: Uint8ClampedArray
  ) {
    const expected = document.atlas.width * document.atlas.height * 4;
    if (pixels.length !== expected) {
      throw new Error("Simple Preview SEGS atlas pixels do not match its dimensions.");
    }
  }

  /** Return every mask containing a source point, most specific first. */
  hitsAt(sourceX: number, sourceY: number): SegPreviewRegion[] {
    return this.document.regions
      .filter((region) => this.contains(region, sourceX, sourceY))
      .sort((left, right) => left.area - right.area || left.index - right.index);
  }

  /** Return one colored RGBA image for a packed region mask. */
  coloredMask(region: SegPreviewRegion): ImageData {
    const color = parseColor(region.color);
    const data = new Uint8ClampedArray(region.atlas.width * region.atlas.height * 4);
    for (let y = 0; y < region.atlas.height; y += 1) {
      for (let x = 0; x < region.atlas.width; x += 1) {
        const atlasOffset = this.offset(region.atlas.x + x, region.atlas.y + y);
        const mask = this.pixels[atlasOffset] ?? 0;
        const outputOffset = (y * region.atlas.width + x) * 4;
        data[outputOffset] = color[0];
        data[outputOffset + 1] = color[1];
        data[outputOffset + 2] = color[2];
        data[outputOffset + 3] = mask;
      }
    }
    return new ImageData(data, region.atlas.width, region.atlas.height);
  }

  private contains(
    region: SegPreviewRegion,
    sourceX: number,
    sourceY: number
  ): boolean {
    const crop = region.crop;
    if (
      sourceX < crop.x ||
      sourceY < crop.y ||
      sourceX >= crop.x + crop.width ||
      sourceY >= crop.y + crop.height
    ) {
      return false;
    }
    const localX = Math.min(
      region.atlas.width - 1,
      Math.floor(((sourceX - crop.x) / crop.width) * region.atlas.width)
    );
    const localY = Math.min(
      region.atlas.height - 1,
      Math.floor(((sourceY - crop.y) / crop.height) * region.atlas.height)
    );
    return (this.pixels[this.offset(region.atlas.x + localX, region.atlas.y + localY)] ?? 0) >= 128;
  }

  private offset(x: number, y: number): number {
    return (y * this.document.atlas.width + x) * 4;
  }
}

/** Read one loaded atlas image into deterministic hit-test pixels. */
export function maskAtlasFromImage(
  previewDocument: SegPreviewDocument,
  image: CanvasImageSource
): MaskAtlas {
  const canvas = globalThis.document.createElement("canvas");
  canvas.width = previewDocument.atlas.width;
  canvas.height = previewDocument.atlas.height;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) throw new Error("Simple Preview SEGS could not read its mask atlas.");
  context.drawImage(image, 0, 0, canvas.width, canvas.height);
  return new MaskAtlas(
    previewDocument,
    context.getImageData(0, 0, canvas.width, canvas.height).data
  );
}

function parseColor(value: string): [number, number, number] {
  return [
    Number.parseInt(value.slice(1, 3), 16),
    Number.parseInt(value.slice(3, 5), 16),
    Number.parseInt(value.slice(5, 7), 16)
  ];
}
