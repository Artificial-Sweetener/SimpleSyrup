// SimpleSyrup - workflow-focused ComfyUI extensions for image generation
// Copyright (C) 2026  Artificial Sweetener and contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

import type { NativeNodePreview } from "./nativeNodePreview";
import { subscribeNativePreviewLifecycle } from "./nativePreviewLifecycle";
import {
  OrderedMediaPreviewAffordances,
  type NativePreviewActionSlot,
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
const CUBE_FACE_PROJECTION_SYMBOL = Symbol.for(
  "sugarcubes.cube-face-projection.v1"
);

interface CubeFaceProjection {
  readonly container: HTMLElement;
  readonly projectRect: (rect: NativeImageRect) => NativePreviewSlot;
}

interface ActionPreviewNode {
  readonly id?: string | number;
  readonly pos?: readonly [number, number];
  readonly size?: readonly [number, number];
  readonly flags?: { readonly collapsed?: boolean };
  widgets?: ActionPreviewWidget[];
  imageIndex?: number | null;
  imageRects?: NativeImageRect[];
  imgs?: HTMLImageElement[];
  graph?: {
    _rootGraph?: object;
    setDirtyCanvas?: (foreground: boolean, background: boolean) => void;
  };
}

interface ActionPreviewWidget {
  readonly y?: number;
  readonly computedHeight?: number;
  readonly options?: { readonly canvasOnly?: boolean };
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
  private lastCanvasPreviewRect: NativeImageRect | null = null;
  private pendingDetailIndex: number | null = null;
  private detailRestoreFrame: number | null = null;

  constructor(private readonly options: OrderedMediaPreviewActionOptions) {
    this.transaction = new OrderedMediaPreviewTransaction({
      captureSurface: () => this.captureSurface(),
      authoritativeReady: () => this.authoritativeReady(),
      stateChanged: () => {
        this.affordances.refresh();
      }
    });
    const moveEarlier = (index: number): void => {
      const destination = index - 1;
      this.transaction.move(index, destination);
      this.followMovedDetail(index, destination);
      options.moveEarlier(index);
    };
    const moveLater = (index: number): void => {
      const destination = index + 1;
      this.transaction.move(index, destination);
      this.followMovedDetail(index, destination);
      options.moveLater(index);
    };
    const remove = (index: number): void => {
      const removedDetail = this.selectedItemIndex() === index;
      this.transaction.remove(index);
      options.remove(index);
      this.followRemovedDetail(index, removedDetail);
    };
    const actionOptions = {
      getItemCount: () => this.itemCount(),
      moveEarlier,
      moveLater,
      remove
    };
    this.affordances = new OrderedMediaPreviewAffordances({
      itemLabel: options.itemLabel,
      getSlots: () => this.nativeActionSlots(),
      ...actionOptions
    });
    this.unsubscribePreview = options.preview.subscribe(() => {
      this.affordances.refresh();
      this.transaction.authoritativePublished();
      this.restorePendingDetail();
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
    if (this.detailRestoreFrame !== null) {
      cancelAnimationFrame(this.detailRestoreFrame);
      this.detailRestoreFrame = null;
    }
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
    const imageRects = this.options.node.imageRects;
    if (!imageRects) return [];
    if (imageRects.length > 0) {
      this.lastCanvasPreviewRect = unionImageRects(imageRects);
    }
    const slots: NativePreviewSlot[] = [];
    for (const rect of imageRects) {
      const slot = this.canvasLocalSlot(rect);
      if (slot) slots.push(slot);
    }
    return slots;
  }

  private nativeActionSlots(): NativePreviewActionSlot[] {
    if (!this.belongsToActiveWorkflow()) return [];
    if (this.options.node.flags?.collapsed) return [];
    const selectedIndex = this.selectedItemIndex();
    if (selectedIndex !== null) {
      return this.detailActionSlots(selectedIndex);
    }
    const activeSlots = this.transaction.activeSlots();
    const domImages = this.domImages();
    if (activeSlots) {
      return indexedSlots(
        activeSlots,
        this.domPreviewContainer(domImages) ??
          this.cubeFaceProjection()?.container ??
          null
      );
    }
    if (!this.isGridVisible()) return [];
    const domSlots = domImages.map((image) => {
      const rect = image.getBoundingClientRect();
      return {
        left: rect.left,
        top: rect.top,
        width: rect.width,
        height: rect.height
      };
    });
    if (domSlots.length === this.itemCount()) {
      return indexedSlots(domSlots, this.domPreviewContainer(domImages));
    }
    return indexedSlots(
      this.canvasSlots(),
      this.cubeFaceProjection()?.container ?? null
    );
  }

  /** Prevent inactive workflow tabs with reused node IDs from claiming active Nodes 2 DOM. */
  private belongsToActiveWorkflow(): boolean {
    const nodeGraph = this.options.node.graph;
    const nodeRoot = nodeGraph?._rootGraph ?? nodeGraph;
    const activeRoot = this.options.app.rootGraph ?? this.options.app.canvas?.graph;
    return !nodeRoot || !activeRoot || nodeRoot === activeRoot;
  }

  /** Expose one selected-item control strip in native detail mode. */
  private detailActionSlots(selectedIndex: number): NativePreviewActionSlot[] {
    const domSlot = this.domDetailSlot(selectedIndex);
    if (domSlot) return [domSlot];
    const canvasSlot = this.canvasDetailSlot();
    if (!canvasSlot) return [];
    const container = this.cubeFaceProjection()?.container;
    return [
      container
        ? { itemIndex: selectedIndex, bounds: canvasSlot, container }
        : { itemIndex: selectedIndex, bounds: canvasSlot }
    ];
  }

  /** Locate the selected Nodes 2.0 preview without replacing native UI. */
  private domDetailSlot(selectedIndex: number): NativePreviewActionSlot | null {
    const root = this.domRoot();
    const reference = this.currentImages()?.[selectedIndex];
    if (!root || !reference) return null;
    const expectedKey = comfyImageReferenceKey(reference);
    const candidates = Array.from(
      root.querySelectorAll<HTMLImageElement>("img")
    ).filter((image) => {
      const rect = image.getBoundingClientRect();
      return (
        comfyImageSourceKey(image.src) === expectedKey &&
        rect.width > 0 &&
        rect.height > 0
      );
    });
    const image = candidates.reduce<HTMLImageElement | null>(
      (largest, candidate) =>
        !largest || imageArea(candidate) > imageArea(largest)
          ? candidate
          : largest,
      null
    );
    const previewRegion = root.querySelector<HTMLElement>(
      '[role="region"][aria-label^="Image preview"]'
    );
    const previewElement = image ?? previewRegion;
    if (!previewElement) return null;
    return {
      itemIndex: selectedIndex,
      bounds: elementSlot(previewElement),
      container:
        previewRegion ??
        this.domPreviewContainer(image ? [image] : []) ??
        root
    };
  }

  /** Return the closest native preview surface shared by rendered media. */
  private domPreviewContainer(images: HTMLImageElement[]): HTMLElement | null {
    const root = this.domRoot();
    if (!root || images.length === 0) return null;
    const previewRegion = root.querySelector<HTMLElement>(
      '[role="region"][aria-label^="Image preview"]'
    );
    if (previewRegion && images.every((image) => previewRegion.contains(image))) {
      return previewRegion;
    }
    let candidate = images[0]?.parentElement ?? null;
    while (candidate && candidate !== root) {
      if (images.every((image) => candidate?.contains(image) === true)) {
        return candidate;
      }
      candidate = candidate.parentElement;
    }
    return root;
  }

  /** Resolve selected preview geometry in node-local canvas coordinates. */
  private canvasDetailLocalSlot(): NativePreviewSlot | null {
    const previewWidget = this.options.node.widgets?.find(
      (widget) =>
        widget.options?.canvasOnly === true &&
        typeof widget.y === "number" &&
        typeof widget.computedHeight === "number" &&
        widget.computedHeight > 0
    );
    const nodeWidth = this.options.node.size?.[0];
    const previewY = previewWidget?.y;
    const previewHeight = previewWidget?.computedHeight;
    if (
      typeof nodeWidth !== "number" ||
      nodeWidth <= 0 ||
      typeof previewY !== "number" ||
      typeof previewHeight !== "number"
    ) {
      if (!this.lastCanvasPreviewRect) return null;
      const [left, top, width, height] = this.lastCanvasPreviewRect;
      return { left, top, width, height };
    }
    return { left: 0, top: previewY, width: nodeWidth, height: previewHeight };
  }

  /** Locate the selected Nodes 1.0 preview from its canvas widget geometry. */
  private canvasDetailSlot(): NativePreviewSlot | null {
    const localSlot = this.canvasDetailLocalSlot();
    return localSlot
      ? this.canvasLocalSlot([
          localSlot.left,
          localSlot.top,
          localSlot.width,
          localSlot.height
        ])
      : null;
  }

  /** Project one node-local canvas rectangle into viewport coordinates. */
  private canvasLocalSlot(rect: NativeImageRect): NativePreviewSlot | null {
    const projection = this.cubeFaceProjection();
    if (projection) {
      try {
        const slot = projection.projectRect(rect);
        if (validSlot(slot)) return slot;
      } catch {
        return null;
      }
    }
    const canvasApi = this.options.app.canvas;
    const nodePosition = this.options.node.pos;
    if (!canvasApi || !nodePosition) return null;
    const [x, y, width, height] = rect;
    const canvasRect = canvasApi.canvas.getBoundingClientRect();
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
  }

  /** Read SugarCubes' renderer-neutral projection contract when this node is embedded. */
  private cubeFaceProjection(): CubeFaceProjection | null {
    const value = Reflect.get(
      this.options.node,
      CUBE_FACE_PROJECTION_SYMBOL
    ) as unknown;
    if (typeof value !== "object" || value === null) return null;
    const candidate = value as Partial<CubeFaceProjection>;
    if (
      !(candidate.container instanceof HTMLElement) ||
      typeof candidate.projectRect !== "function"
    ) {
      return null;
    }
    return candidate as CubeFaceProjection;
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
      slots: canvasImages.map(
        (_, index) => () => this.canvasSlots()[index] as NativePreviewSlot
      ),
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

  /** Read detail selection from the renderer that currently owns it. */
  private selectedItemIndex(): number | null {
    const index = this.options.node.imageIndex;
    if (
      typeof index === "number" &&
      Number.isInteger(index) &&
      index >= 0 &&
      index < this.itemCount()
    ) {
      return index;
    }
    const currentButton = this.domDetailButtons().find(
      (button) => button.getAttribute("aria-current") === "true"
    );
    if (!currentButton) return null;
    const detailIndex = this.domDetailButtons().indexOf(currentButton);
    return detailIndex >= 0 && detailIndex < this.itemCount()
      ? detailIndex
      : null;
  }

  /** Keep the moved item selected across both native renderer state models. */
  private followMovedDetail(index: number, destination: number): void {
    if (this.selectedItemIndex() !== index) return;
    this.pendingDetailIndex = destination;
    if (this.options.node.imageIndex === index) {
      this.options.node.imageIndex = destination;
    } else {
      this.domDetailButtons()[destination]?.click();
    }
    this.affordances.refresh();
  }

  /** Select the nearest remaining item after removing an inspected item. */
  private followRemovedDetail(index: number, removedDetail: boolean): void {
    if (!removedDetail) return;
    const remaining = this.itemCount();
    const destination = remaining > 0 ? Math.min(index, remaining - 1) : null;
    this.pendingDetailIndex = destination;
    if (this.options.node.imageIndex === index) {
      this.options.node.imageIndex = destination;
    } else if (destination !== null) {
      this.domDetailButtons()[destination]?.click();
    }
    this.affordances.refresh();
  }

  /** Re-enter Nodes 2.0 detail mode after output publication resets its grid. */
  private restorePendingDetail(): void {
    if (this.pendingDetailIndex === null || this.detailRestoreFrame !== null) {
      return;
    }
    let stableFrames = 0;
    let remainingFrames = 12;
    const restore = (): void => {
      this.detailRestoreFrame = null;
      const destination = this.pendingDetailIndex;
      if (destination === null) return;
      const buttons = this.domDetailButtons();
      const selected = buttons.findIndex(
        (button) => button.getAttribute("aria-current") === "true"
      );
      if (selected === destination) {
        stableFrames += 1;
      } else {
        stableFrames = 0;
        buttons[destination]?.click();
      }
      remainingFrames -= 1;
      if (stableFrames >= 3 || remainingFrames <= 0) {
        this.pendingDetailIndex = null;
        this.affordances.refresh();
        return;
      }
      this.detailRestoreFrame = requestAnimationFrame(restore);
    };
    this.detailRestoreFrame = requestAnimationFrame(restore);
  }

  /** Return Comfy's ordered detail navigation controls for this node. */
  private domDetailButtons(): HTMLButtonElement[] {
    const root = this.domRoot();
    if (!root) return [];
    return Array.from(
      root.querySelectorAll<HTMLButtonElement>(
        'button[aria-label^="View image "]'
      )
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
  return () => elementSlot(image);
}

function elementSlot(element: Element): NativePreviewSlot {
  const rect = element.getBoundingClientRect();
  return {
    left: rect.left,
    top: rect.top,
    width: rect.width,
    height: rect.height
  };
}

/** Reject malformed cross-extension projection geometry. */
function validSlot(slot: NativePreviewSlot): boolean {
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
function imageArea(image: HTMLImageElement): number {
  const rect = image.getBoundingClientRect();
  return rect.width * rect.height;
}

/** Attach authoritative list positions to native preview footprints. */
function indexedSlots(
  slots: NativePreviewSlot[],
  container: HTMLElement | null = null
): NativePreviewActionSlot[] {
  return slots.map((bounds, itemIndex) =>
    container ? { itemIndex, bounds, container } : { itemIndex, bounds }
  );
}

/** Return the smallest node-local rectangle containing every grid cell. */
function unionImageRects(rects: NativeImageRect[]): NativeImageRect {
  const left = Math.min(...rects.map(([x]) => x));
  const top = Math.min(...rects.map(([, y]) => y));
  const right = Math.max(...rects.map(([x, , width]) => x + width));
  const bottom = Math.max(...rects.map(([, y, , height]) => y + height));
  return [left, top, right - left, bottom - top];
}
