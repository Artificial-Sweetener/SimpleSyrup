// SimpleSyrup - workflow-focused ComfyUI extensions for image generation
// Copyright (C) 2026  Artificial Sweetener and contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

import { getMaskBatchPreview } from "./api";
import type {
  ComfyExecutionEvents,
  ComfyNodeExecutionOutput,
  Logger
} from "./types";

const EMPTY_PREVIEW: ComfyNodeExecutionOutput = {
  images: [],
  animated: []
};
const PREVIEW_PROMPT_ID = "simple-syrup-mask-batch-preview";

export type MaskBatchPreviewClient = (
  files: string[],
  channel: string
) => Promise<ComfyNodeExecutionOutput>;

export interface MaskBatchPreviewNode {
  id?: string | number;
  graph?: { setDirtyCanvas?: (foreground: boolean, background: boolean) => void };
}

/** Coordinate asynchronous native previews without publishing stale results. */
export class MaskBatchPreviewController {
  private requestVersion = 0;

  constructor(
    private readonly executionEvents: ComfyExecutionEvents,
    private readonly node: MaskBatchPreviewNode,
    private readonly loadPreview: MaskBatchPreviewClient = getMaskBatchPreview,
    private readonly logger: Logger = console,
    private readonly clearNativePreview: () => void = () => undefined
  ) {}

  /** Replace the visible preview with the exact selected-channel mask output. */
  refresh(files: string[], channel: string | undefined): void {
    if (this.node.id === undefined || channel === undefined) return;

    const requestVersion = ++this.requestVersion;
    this.clearNativePreview();
    this.publish(EMPTY_PREVIEW);
    if (files.length === 0) return;
    void this.loadPreview([...files], channel)
      .then((output) => {
        if (requestVersion === this.requestVersion) this.publish(output);
      })
      .catch((error: unknown) => {
        if (requestVersion !== this.requestVersion) return;
        this.logger.warn("Could not refresh Load Mask Batch preview.", error);
      });
  }

  /** Publish an execution-shaped result through Comfy's native preview. */
  private publish(output: ComfyNodeExecutionOutput): void {
    if (this.node.id === undefined) return;
    this.executionEvents.dispatchEvent(
      new CustomEvent("executed", {
        detail: {
          node: String(this.node.id),
          output,
          merge: false,
          prompt_id: PREVIEW_PROMPT_ID
        }
      })
    );
    this.node.graph?.setDirtyCanvas?.(true, true);
  }
}
