// SimpleSyrup - workflow-focused ComfyUI extensions for image generation
// Copyright (C) 2026  Artificial Sweetener and contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

/** Derive ordered-media preview items and renderer-neutral geometry. */

import type {
  NativePreviewActionSlot,
  NativePreviewSlot
} from "./orderedMediaPreviewAffordances";
import type { OrderedMediaPreviewItem } from "./orderedMediaPreviewTransaction";

export type NativeImageRect = readonly [number, number, number, number];

export function previewItems(
  images: HTMLImageElement[]
): OrderedMediaPreviewItem[] {
  return images.map((image) => ({
    sourceUrl: image.currentSrc || image.src,
    image
  }));
}

export function imageSlot(
  image: HTMLImageElement
): () => NativePreviewSlot {
  return () => elementSlot(image);
}

export function elementSlot(element: Element): NativePreviewSlot {
  const rect = element.getBoundingClientRect();
  return {
    left: rect.left,
    top: rect.top,
    width: rect.width,
    height: rect.height
  };
}

/** Reject malformed cross-extension projection geometry. */
export function validSlot(slot: NativePreviewSlot): boolean {
  return (
    Number.isFinite(slot.left) &&
    Number.isFinite(slot.top) &&
    Number.isFinite(slot.width) &&
    Number.isFinite(slot.height) &&
    slot.width >= 0 &&
    slot.height >= 0
  );
}

/** Measure a rendered image when native detail contains multiple candidates. */
export function imageArea(image: HTMLImageElement): number {
  const rect = image.getBoundingClientRect();
  return rect.width * rect.height;
}

/** Attach authoritative list positions to native preview footprints. */
export function indexedSlots(
  slots: NativePreviewSlot[],
  container: HTMLElement | null = null
): NativePreviewActionSlot[] {
  return slots.map((bounds, itemIndex) =>
    container ? { itemIndex, bounds, container } : { itemIndex, bounds }
  );
}

/** Return the smallest node-local rectangle containing every grid cell. */
export function unionImageRects(rects: NativeImageRect[]): NativeImageRect {
  const left = Math.min(...rects.map(([x]) => x));
  const top = Math.min(...rects.map(([, y]) => y));
  const right = Math.max(...rects.map(([x, , width]) => x + width));
  const bottom = Math.max(...rects.map(([, y, , height]) => y + height));
  return [left, top, right - left, bottom - top];
}
