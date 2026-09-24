// SimpleSyrup - workflow-focused ComfyUI extensions for image generation
// Copyright (C) 2026  Artificial Sweetener and contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

/** Preserve native detail selection across ordered-media list mutations. */

export interface OrderedMediaDetailSelectionOptions {
  readonly selectedIndex: () => number | null;
  readonly canvasIndex: () => number | null | undefined;
  readonly setCanvasIndex: (index: number | null) => void;
  readonly itemCount: () => number;
  readonly domSelectedIndex: () => number | null;
  readonly selectDomIndex: (index: number) => void;
  readonly refreshActions: () => void;
}

export class OrderedMediaDetailSelection {
  private pendingIndex: number | null = null;
  private restoreFrame: number | null = null;

  constructor(private readonly options: OrderedMediaDetailSelectionOptions) {}

  /** Keep the moved item selected across both native renderer state models. */
  followMoved(index: number, destination: number): void {
    if (this.options.selectedIndex() !== index) return;
    this.pendingIndex = destination;
    if (this.options.canvasIndex() === index) {
      this.options.setCanvasIndex(destination);
    } else {
      this.options.selectDomIndex(destination);
    }
    this.options.refreshActions();
  }

  /** Select the nearest remaining item after removing an inspected item. */
  followRemoved(index: number, removedDetail: boolean): void {
    if (!removedDetail) return;
    const remaining = this.options.itemCount();
    const destination = remaining > 0 ? Math.min(index, remaining - 1) : null;
    this.pendingIndex = destination;
    if (this.options.canvasIndex() === index) {
      this.options.setCanvasIndex(destination);
    } else if (destination !== null) {
      this.options.selectDomIndex(destination);
    }
    this.options.refreshActions();
  }

  /** Re-enter Nodes 2.0 detail mode after output publication resets its grid. */
  restorePending(): void {
    if (this.pendingIndex === null || this.restoreFrame !== null) return;
    let stableFrames = 0;
    let remainingFrames = 12;
    const restore = (): void => {
      this.restoreFrame = null;
      const destination = this.pendingIndex;
      if (destination === null) return;
      if (this.options.domSelectedIndex() === destination) {
        stableFrames += 1;
      } else {
        stableFrames = 0;
        this.options.selectDomIndex(destination);
      }
      remainingFrames -= 1;
      if (stableFrames >= 3 || remainingFrames <= 0) {
        this.pendingIndex = null;
        this.options.refreshActions();
        return;
      }
      this.restoreFrame = requestAnimationFrame(restore);
    };
    this.restoreFrame = requestAnimationFrame(restore);
  }

  /** Cancel any selection restoration still waiting for native publication. */
  dispose(): void {
    if (this.restoreFrame === null) return;
    cancelAnimationFrame(this.restoreFrame);
    this.restoreFrame = null;
  }
}
