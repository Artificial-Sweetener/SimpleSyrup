// SimpleSyrup - workflow-focused ComfyUI extensions for image generation
// Copyright (C) 2026  Artificial Sweetener and contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

import type { NativeNodePreview } from "./nativeNodePreview";
import { subscribeNativePreviewLifecycle } from "./nativePreviewLifecycle";
import {
  OrderedMediaPreviewAffordances,
  type NativePreviewSlot
} from "./orderedMediaPreviewAffordances";
import {
  OrderedMediaPreviewTransaction,
  type OrderedMediaPreviewItem,
  type OrderedMediaPreviewSurface
} from "./orderedMediaPreviewTransaction";
import type { ComfyApp, ComfyImageResult } from "./types";
import {
  comfyImageReferenceKey,
  comfyImageSourceKey
} from "./comfyImageReference";

type NativeImageRect = readonly [number, number, number, number];

interface ActionPreviewNode {
  readonly id?: string | number;
  readonly pos?: readonly [number, number];
  readonly flags?: { readonly collapsed?: boolean };
  imageIndex?: number | null;
  imageRects?: NativeImageRect[];
  imgs?: HTMLImageElement[];
  graph?: { setDirtyCanvas?: (foreground: boolean, background: boolean) => void };
}

export interface OrderedMediaPreviewActionOptions {
  readonly app: ComfyApp;
  readonly node: ActionPreviewNode;
  readonly preview: NativeNodePreview;
  readonly itemLabel: string;
  readonly getFiles: () => string[];
  readonly moveEarlier: (index: number) => void;
  readonly moveLater: (index: number) => void;
  readonly remove: (index: number) => void;
}

/** Position deterministic list actions over Comfy's native preview cells. */
export class OrderedMediaPreviewActions {
  private readonly affordances: OrderedMediaPreviewAffordances;
  private readonly transaction: OrderedMediaPreviewTransaction;
  private readonly unsubscribePreview: () => void;
  private readonly unsubscribeLifecycle: () => void;

  constructor(private readonly options: OrderedMediaPreviewActionOptions) {
    this.transaction = new OrderedMediaPreviewTransaction({
      captureSurface: () => this.captureSurface(),
      authoritativeReady: () => this.authoritativeReady(),
      stateChanged: () => {
        this.affordances.refresh();
      }
    });
    this.affordances = new OrderedMediaPreviewAffordances({
      itemLabel: options.itemLabel,
      getSlots: () => this.nativeSlots(),
      moveEarlier: (index) => {
        this.transaction.move(index, index - 1);
        options.moveEarlier(index);
      },
      moveLater: (index) => {
        this.transaction.move(index, index + 1);
        options.moveLater(index);
      },
      remove: (index) => {
        this.transaction.remove(index);
        options.remove(index);
      }
    });
    this.unsubscribePreview = options.preview.subscribe(() => {
      this.affordances.refresh();
      this.transaction.authoritativePublished();
    });
    this.unsubscribeLifecycle = subscribeNativePreviewLifecycle(() => {
      this.affordances.refresh();
    });
  }

  /** Remove layout listeners and every loader-owned overlay control. */
  dispose(): void {
    this.unsubscribePreview();
    this.unsubscribeLifecycle();
    this.transaction.dispose();
    this.affordances.dispose();
  }

  private itemCount(): number {
    const files = this.options.getFiles();
    const images = this.currentImages();
    return images?.length === files.length ? files.length : 0;
  }

  private currentImages(): ComfyImageResult[] | undefined {
    const nodeId = this.nodeId();
    return nodeId ? this.options.app.nodeOutputs?.[nodeId]?.images : undefined;
  }

  private domImages(): HTMLImageElement[] {
    const root = this.domRoot();
    const references = this.currentImages();
    if (!root || !references || references.length !== this.itemCount()) return [];
    const remainingKeys = references.map(comfyImageReferenceKey);
    const matched: HTMLImageElement[] = [];
    for (const image of Array.from(
      root.querySelectorAll<HTMLImageElement>("img")
    )) {
      const key = comfyImageSourceKey(image.src);
      const referenceIndex = remainingKeys.indexOf(key);
      if (referenceIndex < 0) continue;
      matched.push(image);
      remainingKeys.splice(referenceIndex, 1);
    }
    return remainingKeys.length === 0 ? matched : [];
  }

  private domRoot(): HTMLElement | null {
    const nodeId = this.nodeId();
    if (!nodeId) return null;
    return (
      Array.from(document.querySelectorAll<HTMLElement>("[data-node-id]")).find(
        (element) => element.dataset.nodeId === nodeId
      ) ?? null
    );
  }

  private canvasSlots(): NativePreviewSlot[] {
    const canvasApi = this.options.app.canvas;
    const nodePosition = this.options.node.pos;
    const imageRects = this.options.node.imageRects;
    if (!canvasApi || !nodePosition || !imageRects) return [];
    const canvasRect = canvasApi.canvas.getBoundingClientRect();
    return imageRects.map(([x, y, width, height]) => {
      const topLeft = canvasApi.convertOffsetToCanvas([
        nodePosition[0] + x,
        nodePosition[1] + y
      ]);
      const bottomRight = canvasApi.convertOffsetToCanvas([
        nodePosition[0] + x + width,
        nodePosition[1] + y + height
      ]);
      return {
        left: canvasRect.left + topLeft[0],
        top: canvasRect.top + topLeft[1],
        width: bottomRight[0] - topLeft[0],
        height: bottomRight[1] - topLeft[1]
      };
    });
  }

  private nativeSlots(): NativePreviewSlot[] {
    if (
      this.options.node.flags?.collapsed ||
      this.options.node.imageIndex != null
    ) {
      return [];
    }
    const activeSlots = this.transaction.activeSlots();
    if (activeSlots) return activeSlots;
    if (!this.isGridVisible()) return [];
    const domSlots = this.domImages().map((image) => {
      const rect = image.getBoundingClientRect();
      return {
        left: rect.left,
        top: rect.top,
        width: rect.width,
        height: rect.height
      };
    });
    return domSlots.length === this.itemCount() ? domSlots : this.canvasSlots();
  }

  private captureSurface(): OrderedMediaPreviewSurface | null {
    const domImages = this.domImages();
    if (domImages.length > 0) return this.domSurface(domImages);
    const canvasImages = this.options.node.imgs;
    if (!canvasImages || canvasImages.length !== this.currentImages()?.length) {
      return null;
    }
    return {
      items: previewItems(canvasImages),
      slots: canvasImages.map((_, index) => () => this.canvasSlots()[index] as NativePreviewSlot),
      apply: (items) => {
        this.options.node.imgs = items.map((item) => item.image);
        delete this.options.node.imageRects;
        this.options.node.graph?.setDirtyCanvas?.(true, true);
      },
      release: () => undefined
    };
  }

  private domSurface(images: HTMLImageElement[]): OrderedMediaPreviewSurface {
    const slots = images.map((image) => imageSlot(image));
    const targets = images.map((image) => image.closest("button") ?? image);
    const originalDisplays = targets.map((target) => target.style.display);
    return {
      items: previewItems(images),
      slots,
      apply: (items) => {
        for (const [index, image] of images.entries()) {
          const item = items[index];
          const target = targets[index];
          if (!target) continue;
          if (!item) {
            target.style.display = "none";
            continue;
          }
          target.style.display = originalDisplays[index] ?? "";
          if (image.src !== item.sourceUrl) image.src = item.sourceUrl;
        }
      },
      release: () => {
        for (const [index, target] of targets.entries()) {
          target.style.display = originalDisplays[index] ?? "";
        }
      }
    };
  }

  private authoritativeReady(): boolean {
    const references = this.currentImages();
    if (!references || references.length !== this.options.getFiles().length) {
      return false;
    }
    const expectedKeys = references.map(comfyImageReferenceKey);
    const rendered = this.authoritativeImages(expectedKeys);
    return (
      rendered.length === expectedKeys.length &&
      rendered.every(
        (image, index) =>
          comfyImageSourceKey(image.src) === expectedKeys[index] &&
          image.complete &&
          image.naturalWidth > 0
      )
    );
  }

  private authoritativeImages(expectedKeys: string[]): HTMLImageElement[] {
    const root = this.domRoot();
    const domImages = root
      ? Array.from(root.querySelectorAll<HTMLImageElement>("img"))
      : [];
    if (domImages.length === expectedKeys.length) return domImages;
    return this.options.node.imgs ?? [];
  }

  private isGridVisible(): boolean {
    return (
      !this.options.node.flags?.collapsed &&
      this.options.node.imageIndex == null &&
      this.itemCount() > 1
    );
  }

  private nodeId(): string | undefined {
    const nodeId = this.options.node.id;
    return nodeId === undefined ? undefined : String(nodeId);
  }
}

function previewItems(images: HTMLImageElement[]): OrderedMediaPreviewItem[] {
  return images.map((image) => ({
    sourceUrl: image.currentSrc || image.src,
    image
  }));
}

function imageSlot(image: HTMLImageElement): () => NativePreviewSlot {
  return () => {
    const rect = image.getBoundingClientRect();
    return {
      left: rect.left,
      top: rect.top,
      width: rect.width,
      height: rect.height
    };
  };
}
