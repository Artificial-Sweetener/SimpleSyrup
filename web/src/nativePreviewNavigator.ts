// SimpleSyrup - workflow-focused ComfyUI extensions for image generation
// Copyright (C) 2026  Artificial Sweetener and contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

import type { ComfyApp, ComfyImageResult } from "./types";
import {
  comfyImageReferenceKey,
  comfyImageSourceKey
} from "./comfyImageReference";
import { subscribeNativePreviewLifecycle } from "./nativePreviewLifecycle";

interface PreviewNavigationNode {
  readonly id?: string | number;
  imageIndex?: number | null;
  imageRects?: readonly unknown[];
  imgs?: readonly HTMLImageElement[];
  pointerDown?: { index: number | null; pos: [number, number] };
  graph?: { setDirtyCanvas?: (foreground: boolean, background: boolean) => void };
}

/** Open one output through Comfy's native image inspection behavior. */
export class NativePreviewNavigator {
  private pendingReference: ComfyImageResult | undefined;
  private legacyInspectionFrame: number | undefined;
  private unsubscribeLifecycle: (() => void) | undefined;

  constructor(
    private readonly app: ComfyApp,
    private readonly node: PreviewNavigationNode
  ) {}

  /** Report whether Comfy currently owns a DOM-backed preview surface. */
  usesDomPreview(): boolean {
    return this.domRoot() !== null;
  }

  /** Inspect one zero-based native preview item in either node renderer. */
  inspect(index: number): void {
    if (!Number.isInteger(index) || index < 0) return;
    if (this.domRoot()) {
      const images = this.images();
      if (!images || index >= images.length) return;
      const reference = images[index] as ComfyImageResult;
      if (this.clickDomImage(reference)) return;
      this.pendingReference = reference;
      this.unsubscribeLifecycle ??= subscribeNativePreviewLifecycle(() => {
        this.completePendingInspection();
      });
      return;
    }
    this.scheduleLegacyImage(index);
  }

  /** Wait for Comfy to restore canvas images before selecting native detail. */
  private scheduleLegacyImage(index: number): void {
    this.cancelLegacyInspection();
    const inspectWhenReady = (): void => {
      const images = this.node.imgs;
      if (images !== undefined && images.length > index) {
        this.legacyInspectionFrame = undefined;
        this.openLegacyImage(index);
        return;
      }
      this.legacyInspectionFrame = requestAnimationFrame(inspectWhenReady);
    };
    this.legacyInspectionFrame = requestAnimationFrame(inspectWhenReady);
  }

  /** Open one loaded image through Comfy's canvas detail state. */
  private openLegacyImage(index: number): void {
    this.node.imageIndex = index;
    const position = this.app.canvas?.graph_mouse;
    if (position) {
      this.node.pointerDown = { index, pos: [position[0], position[1]] };
    }
    delete this.node.imageRects;
    this.node.graph?.setDirtyCanvas?.(true, true);
  }

  /** Return the native preview to its one authoritative gallery. */
  showGrid(): void {
    this.cancelLegacyInspection();
    this.pendingReference = undefined;
    if (this.clickDomGrid()) return;
    this.node.imageIndex = null;
    delete this.node.imageRects;
    this.node.graph?.setDirtyCanvas?.(true, true);
  }

  /** Cancel pending navigation when native images are hidden. */
  hide(): void {
    this.cancelLegacyInspection();
    this.pendingReference = undefined;
    this.node.imageIndex = null;
    delete this.node.imageRects;
    this.node.graph?.setDirtyCanvas?.(true, true);
  }

  /** Release native-preview lifecycle observation. */
  dispose(): void {
    this.cancelLegacyInspection();
    this.pendingReference = undefined;
    this.unsubscribeLifecycle?.();
    this.unsubscribeLifecycle = undefined;
  }

  private cancelLegacyInspection(): void {
    if (this.legacyInspectionFrame === undefined) return;
    cancelAnimationFrame(this.legacyInspectionFrame);
    this.legacyInspectionFrame = undefined;
  }

  private completePendingInspection(): void {
    const reference = this.pendingReference;
    if (!reference || !this.clickDomImage(reference)) return;
    this.pendingReference = undefined;
  }

  private clickDomImage(reference: ComfyImageResult): boolean {
    const root = this.domRoot();
    if (!root) return false;
    const image = Array.from(root.querySelectorAll<HTMLImageElement>("img")).find(
      (candidate) =>
        comfyImageSourceKey(candidate.src) === comfyImageReferenceKey(reference)
    );
    const button = image?.closest("button");
    if (!button) return false;
    button.click();
    return true;
  }

  private clickDomGrid(): boolean {
    const root = this.domRoot();
    if (!root) return false;
    const button = Array.from(root.querySelectorAll<HTMLButtonElement>("button")).find(
      (candidate) =>
        candidate.getAttribute("aria-label") === "Grid view" ||
        candidate.title === "Grid view"
    );
    if (!button) return false;
    button.click();
    return true;
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

  private images(): ComfyImageResult[] | undefined {
    const nodeId = this.nodeId();
    return nodeId ? this.app.nodeOutputs?.[nodeId]?.images : undefined;
  }

  private nodeId(): string | undefined {
    return this.node.id === undefined ? undefined : String(this.node.id);
  }
}
