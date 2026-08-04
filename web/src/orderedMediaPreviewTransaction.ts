// SimpleSyrup - workflow-focused ComfyUI extensions for image generation
// Copyright (C) 2026  Artificial Sweetener and contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

import type { NativePreviewSlot } from "./orderedMediaPreviewAffordances";

export interface OrderedMediaPreviewItem {
  readonly sourceUrl: string;
  readonly image: HTMLImageElement;
}

export interface OrderedMediaPreviewSurface {
  readonly items: OrderedMediaPreviewItem[];
  readonly slots: Array<() => NativePreviewSlot>;
  readonly apply: (items: OrderedMediaPreviewItem[]) => void;
  readonly release: () => void;
}

export interface OrderedMediaPreviewTransactionOptions {
  readonly captureSurface: () => OrderedMediaPreviewSurface | null;
  readonly authoritativeReady: () => boolean;
  readonly stateChanged: () => void;
}

/** Reorder Comfy's loaded preview media without creating another drawing layer. */
export class OrderedMediaPreviewTransaction {
  private surface: OrderedMediaPreviewSurface | null = null;
  private items: OrderedMediaPreviewItem[] | null = null;
  private handoffFrame: number | null = null;

  constructor(private readonly options: OrderedMediaPreviewTransactionOptions) {}

  /** Move an already-loaded native item without rebuilding its gallery. */
  move(index: number, destination: number): boolean {
    if (!this.begin()) return false;
    const items = this.items;
    if (
      !items ||
      index < 0 ||
      index >= items.length ||
      destination < 0 ||
      destination >= items.length ||
      index === destination
    ) {
      return false;
    }
    const [moved] = items.splice(index, 1);
    if (!moved) return false;
    items.splice(destination, 0, moved);
    this.apply();
    return true;
  }

  /** Remove an already-loaded item and let the native gallery compact itself. */
  remove(index: number): boolean {
    if (!this.begin()) return false;
    const items = this.items;
    if (!items || index < 0 || index >= items.length) return false;
    items.splice(index, 1);
    this.apply();
    return true;
  }

  /** Return live native-cell geometry while an optimistic transaction is active. */
  activeSlots(): NativePreviewSlot[] | null {
    if (!this.surface || !this.items) return null;
    return this.surface.slots.slice(0, this.items.length).map((slot) => slot());
  }

  /** End the transaction after Comfy's authoritative native preview is ready. */
  authoritativePublished(): void {
    if (!this.surface || this.handoffFrame !== null) return;
    this.scheduleHandoff();
  }

  /** Release native surface changes and stop any pending handoff check. */
  dispose(): void {
    if (this.handoffFrame !== null) cancelAnimationFrame(this.handoffFrame);
    this.handoffFrame = null;
    this.finish();
  }

  private begin(): boolean {
    if (this.surface && this.items) return true;
    const surface = this.options.captureSurface();
    if (!surface || surface.items.length === 0) return false;
    this.surface = surface;
    this.items = [...surface.items];
    return true;
  }

  private apply(): void {
    if (!this.surface || !this.items) return;
    this.surface.apply(this.items);
    this.options.stateChanged();
  }

  private scheduleHandoff(): void {
    this.handoffFrame = requestAnimationFrame(() => {
      this.handoffFrame = null;
      if (!this.surface) return;
      if (this.options.authoritativeReady()) {
        this.finish();
        this.options.stateChanged();
        return;
      }
      this.scheduleHandoff();
    });
  }

  private finish(): void {
    this.surface?.release();
    this.surface = null;
    this.items = null;
  }
}
