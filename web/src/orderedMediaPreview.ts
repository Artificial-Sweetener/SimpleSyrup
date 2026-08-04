// SimpleSyrup - workflow-focused ComfyUI extensions for image generation
// Copyright (C) 2026  Artificial Sweetener and contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

import type { ComfyNodeExecutionOutput, Logger } from "./types";

export type OrderedMediaPreviewClient = () => Promise<ComfyNodeExecutionOutput>;

export interface OrderedMediaPreviewTarget {
  publish(output: ComfyNodeExecutionOutput): void;
  clear(): void;
}

/** Publish only the latest ordered-media result through Comfy's native preview. */
export class OrderedMediaPreviewController {
  private requestVersion = 0;
  private disposed = false;
  private hasPreview = false;

  constructor(
    private readonly target: OrderedMediaPreviewTarget,
    private readonly label: string,
    private readonly logger: Logger = console
  ) {}

  /** Resolve one ordered selection without allowing stale previews to win. */
  refresh(files: string[], loadPreview: OrderedMediaPreviewClient): void {
    if (this.disposed) return;
    const requestVersion = ++this.requestVersion;
    if (files.length === 0) {
      this.target.clear();
      this.hasPreview = false;
      return;
    }
    void loadPreview()
      .then(async (output) => {
        await nextTask();
        if (this.disposed || requestVersion !== this.requestVersion) return;
        if (!Array.isArray(output.images) || output.images.length !== files.length) {
          throw new Error(
            `${this.label} preview returned ${String(output.images?.length ?? 0)} images for ${String(files.length)} files.`
          );
        }
        this.target.publish(output);
        this.hasPreview = true;
      })
      .catch((error: unknown) => {
        if (this.disposed || requestVersion !== this.requestVersion) return;
        const message = error instanceof Error ? error.message : String(error);
        this.logger.warn(`Could not refresh ${this.label} preview: ${message}`, error);
        if (!this.hasPreview) this.target.clear();
      });
  }

  /** Clear native media and invalidate pending work owned by a removed node. */
  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.requestVersion += 1;
    this.target.clear();
  }
}

function nextTask(): Promise<void> {
  return new Promise((resolve) => {
    setTimeout(resolve, 0);
  });
}
