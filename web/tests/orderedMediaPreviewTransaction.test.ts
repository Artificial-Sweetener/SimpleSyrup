// SimpleSyrup - workflow-focused ComfyUI extensions for image generation
// Copyright (C) 2026  Artificial Sweetener and contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

import { describe, expect, it, vi } from "vitest";

import {
  OrderedMediaPreviewTransaction,
  type OrderedMediaPreviewItem
} from "../src/orderedMediaPreviewTransaction";
import type { NativePreviewSlot } from "../src/orderedMediaPreviewAffordances";

describe("OrderedMediaPreviewTransaction", () => {
  it("reorders the captured native surface without creating DOM elements", async () => {
    let ready = false;
    let rendered = items("one.png", "two.png");
    const apply = vi.fn((next: OrderedMediaPreviewItem[]) => {
      rendered = [...next];
    });
    const release = vi.fn();
    const transaction = new OrderedMediaPreviewTransaction({
      captureSurface: () => ({
        items: rendered,
        slots: [() => slot(0), () => slot(100)],
        apply,
        release
      }),
      authoritativeReady: () => ready,
      stateChanged: vi.fn()
    });
    const childCount = document.body.childElementCount;

    expect(transaction.move(0, 1)).toBe(true);

    expect(rendered.map((item) => file(item.sourceUrl))).toEqual([
      "two.png",
      "one.png"
    ]);
    expect(document.body.childElementCount).toBe(childCount);
    transaction.authoritativePublished();
    await new Promise((resolve) => setTimeout(resolve, 30));
    expect(release).not.toHaveBeenCalled();
    ready = true;
    await vi.waitFor(() => {
      expect(release).toHaveBeenCalledOnce();
    });
    transaction.dispose();
  });

  it("compacts removals through the same native surface", () => {
    let rendered = items("one.png", "two.png", "three.png");
    const transaction = new OrderedMediaPreviewTransaction({
      captureSurface: () => ({
        items: rendered,
        slots: [() => slot(0), () => slot(100), () => slot(200)],
        apply: (next) => {
          rendered = [...next];
        },
        release: vi.fn()
      }),
      authoritativeReady: () => false,
      stateChanged: vi.fn()
    });

    expect(transaction.remove(1)).toBe(true);

    expect(rendered.map((item) => file(item.sourceUrl))).toEqual([
      "one.png",
      "three.png"
    ]);
    expect(transaction.activeSlots()).toEqual([slot(0), slot(100)]);
    transaction.dispose();
  });
});

function items(...files: string[]): OrderedMediaPreviewItem[] {
  return files.map((name) => {
    const image = document.createElement("img");
    image.src = `http://localhost/${name}`;
    return { sourceUrl: image.src, image };
  });
}

function slot(left: number): NativePreviewSlot {
  return { left, top: 20, width: 80, height: 90 };
}

function file(sourceUrl: string): string {
  return new URL(sourceUrl).pathname.slice(1);
}
