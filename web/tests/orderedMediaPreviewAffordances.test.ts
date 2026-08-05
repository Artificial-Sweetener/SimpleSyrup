// SimpleSyrup - workflow-focused ComfyUI extensions for image generation
// Copyright (C) 2026  Artificial Sweetener and contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

import { afterEach, describe, expect, it, vi } from "vitest";

import { OrderedMediaPreviewAffordances } from "../src/orderedMediaPreviewAffordances";

let affordances: OrderedMediaPreviewAffordances | undefined;

afterEach(() => {
  affordances?.dispose();
  affordances = undefined;
  document.body.replaceChildren();
});

describe("OrderedMediaPreviewAffordances", () => {
  it("mounts controls inside their native preview surface", async () => {
    const previewSurface = document.createElement("div");
    previewSurface.getBoundingClientRect = () => new DOMRect(100, 200, 240, 180);
    document.body.append(previewSurface);
    affordances = new OrderedMediaPreviewAffordances({
      itemLabel: "mask",
      getSlots: () => [
        {
          itemIndex: 0,
          bounds: { left: 110, top: 220, width: 80, height: 100 },
          container: previewSurface
        }
      ],
      getItemCount: () => 1,
      moveEarlier: vi.fn(),
      moveLater: vi.fn(),
      remove: vi.fn()
    });

    await vi.waitFor(() => {
      expect(previewSurface.querySelector(".ss-native-preview-affordance")).not.toBeNull();
    });
    const root = previewSurface.querySelector<HTMLElement>(
      ".ss-native-preview-affordance"
    );
    expect(previewSurface.style.position).toBe("relative");
    expect(root?.style.position).toBe("absolute");
    expect(root?.style.zIndex).toBe("1");
    expect(root?.style.left).toBe("10px");
    expect(root?.style.top).toBe("20px");
    expect(root?.style.borderRadius).toBe("");
    expect(root?.style.boxShadow).toBe("");

    affordances.dispose();
    affordances = undefined;
    expect(previewSurface.style.position).toBe("");
    expect(previewSurface.querySelector(".ss-native-preview-affordance")).toBeNull();
  });

  it("converts viewport geometry into a transformed preview surface", async () => {
    const previewSurface = document.createElement("div");
    Object.defineProperties(previewSurface, {
      offsetWidth: { value: 100 },
      offsetHeight: { value: 80 }
    });
    previewSurface.getBoundingClientRect = () => new DOMRect(100, 200, 200, 160);
    document.body.append(previewSurface);
    affordances = new OrderedMediaPreviewAffordances({
      itemLabel: "mask",
      getSlots: () => [
        {
          itemIndex: 0,
          bounds: { left: 120, top: 220, width: 80, height: 100 },
          container: previewSurface
        }
      ],
      getItemCount: () => 1,
      moveEarlier: vi.fn(),
      moveLater: vi.fn(),
      remove: vi.fn()
    });

    await vi.waitFor(() => {
      expect(previewSurface.querySelector(".ss-native-preview-affordance")).not.toBeNull();
    });
    const root = previewSurface.querySelector<HTMLElement>(
      ".ss-native-preview-affordance"
    );
    expect(root?.style.left).toBe("10px");
    expect(root?.style.top).toBe("10px");
    expect(root?.style.width).toBe("40px");
    expect(root?.style.height).toBe("18px");
  });

  it("positions native-icon list actions over every native slot", async () => {
    const moveEarlier = vi.fn();
    const moveLater = vi.fn();
    const remove = vi.fn();
    affordances = new OrderedMediaPreviewAffordances({
      itemLabel: "mask",
      getSlots: () => [
        {
          itemIndex: 0,
          bounds: { left: 10, top: 20, width: 100, height: 120 }
        },
        {
          itemIndex: 1,
          bounds: { left: 120, top: 20, width: 80, height: 120 }
        }
      ],
      getItemCount: () => 2,
      moveEarlier,
      moveLater,
      remove
    });
    await vi.waitFor(() => {
      expect(document.querySelectorAll(".ss-native-preview-affordance")).toHaveLength(2);
    });

    const earlierButtons = document.querySelectorAll<HTMLButtonElement>(
      "[data-ss-media-move-earlier]"
    );
    const laterButtons = document.querySelectorAll<HTMLButtonElement>(
      "[data-ss-media-move-later]"
    );
    const removeButtons = document.querySelectorAll<HTMLButtonElement>(
      "[data-ss-media-remove]"
    );
    expect(earlierButtons[0]?.getAttribute("aria-label")).toBe(
      "Move mask 1 earlier"
    );
    expect(laterButtons[1]?.getAttribute("aria-label")).toBe(
      "Move mask 2 later"
    );
    expect(removeButtons[1]?.getAttribute("aria-label")).toBe("Remove mask 2");
    expect(earlierButtons[0]?.disabled).toBe(true);
    expect(laterButtons[1]?.disabled).toBe(true);
    expect(earlierButtons[1]?.disabled).toBe(false);
    expect(laterButtons[0]?.disabled).toBe(false);
    expect(document.querySelectorAll(".pi-arrow-left")).toHaveLength(2);
    expect(document.querySelectorAll(".pi-arrow-right")).toHaveLength(2);
    expect(document.querySelectorAll(".pi-times")).toHaveLength(2);
    expect(earlierButtons[0]?.parentElement?.style.left).toBe("10px");

    earlierButtons[1]?.click();
    laterButtons[0]?.click();
    removeButtons[1]?.click();
    expect(moveEarlier).toHaveBeenCalledWith(1);
    expect(moveLater).toHaveBeenCalledWith(0);
    expect(remove).toHaveBeenCalledWith(1);
  });

  it("removes stale controls when the native preview shrinks", async () => {
    let count = 2;
    affordances = new OrderedMediaPreviewAffordances({
      itemLabel: "image",
      getSlots: () =>
        Array.from({ length: count }, (_, index) => ({
          itemIndex: index,
          bounds: {
            left: index * 100,
            top: 0,
            width: 90,
            height: 90
          }
        })),
      getItemCount: () => count,
      moveEarlier: vi.fn(),
      moveLater: vi.fn(),
      remove: vi.fn()
    });
    await vi.waitFor(() => {
      expect(document.querySelectorAll(".ss-native-preview-affordance")).toHaveLength(2);
    });

    count = 1;
    affordances.refresh();

    await vi.waitFor(() => {
      expect(document.querySelectorAll(".ss-native-preview-affordance")).toHaveLength(1);
    });
  });

  it("disappears without a native slot and returns after navigation", async () => {
    let visible = true;
    affordances = new OrderedMediaPreviewAffordances({
      itemLabel: "image",
      getSlots: () =>
        visible
          ? [
              {
                itemIndex: 0,
                bounds: { left: 10, top: 20, width: 100, height: 120 }
              }
            ]
          : [],
      getItemCount: () => 1,
      moveEarlier: vi.fn(),
      moveLater: vi.fn(),
      remove: vi.fn()
    });
    await vi.waitFor(() => {
      expect(document.querySelectorAll(".ss-native-preview-affordance")).toHaveLength(1);
    });

    visible = false;
    document.dispatchEvent(new MouseEvent("click"));
    await vi.waitFor(() => {
      expect(document.querySelectorAll(".ss-native-preview-affordance")).toHaveLength(0);
    });

    visible = true;
    document.dispatchEvent(new MouseEvent("click"));
    await vi.waitFor(() => {
      expect(document.querySelectorAll(".ss-native-preview-affordance")).toHaveLength(1);
    });
  });

  it("hides viewport controls while the canvas is being manipulated", async () => {
    const canvas = document.createElement("canvas");
    document.body.append(canvas);
    affordances = new OrderedMediaPreviewAffordances({
      itemLabel: "mask",
      getSlots: () => [
        {
          itemIndex: 0,
          bounds: { left: 10, top: 20, width: 100, height: 120 }
        }
      ],
      getItemCount: () => 1,
      moveEarlier: vi.fn(),
      moveLater: vi.fn(),
      remove: vi.fn()
    });
    await vi.waitFor(() => {
      expect(
        document.querySelector<HTMLElement>(".ss-native-preview-affordance")
          ?.hidden
      ).toBe(false);
    });

    canvas.dispatchEvent(new MouseEvent("pointerdown", { bubbles: true }));
    expect(
      document.querySelector<HTMLElement>(".ss-native-preview-affordance")
        ?.hidden
    ).toBe(true);

    document.dispatchEvent(new MouseEvent("pointerup", { bubbles: true }));
    await vi.waitFor(() => {
      expect(
        document.querySelector<HTMLElement>(".ss-native-preview-affordance")
          ?.hidden
      ).toBe(false);
    });
  });
});
