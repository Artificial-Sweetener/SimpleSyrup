// SimpleSyrup - workflow-focused ComfyUI extensions for image generation
// Copyright (C) 2026  Artificial Sweetener and contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

import type {
  ComfyApi,
  ComfyApp,
  ComfyNodeExecutionOutput
} from "./types";
import type { OrderedMediaPreviewTarget } from "./orderedMediaPreview";

interface PreviewNode {
  readonly id?: string | number;
}

/** Feed input-backed media through the same output path as Preview Image. */
export class NativeNodePreview implements OrderedMediaPreviewTarget {
  private readonly publishListeners = new Set<() => void>();

  constructor(
    private readonly app: ComfyApp,
    private readonly api: ComfyApi,
    private readonly node: PreviewNode
  ) {}

  /** Publish media as an execution-shaped output for both node renderers. */
  publish(output: ComfyNodeExecutionOutput): void {
    const nodeId = this.nodeId();
    if (!nodeId) return;
    this.app.nodeOutputs ??= {};
    this.app.nodeOutputs[nodeId] = output;
    this.api.dispatchEvent(
      new CustomEvent("executed", {
        detail: { node: nodeId, display_node: nodeId, output }
      })
    );
    for (const listener of this.publishListeners) listener();
  }

  /** Notify renderer adapters after native output changes. */
  subscribe(listener: () => void): () => void {
    this.publishListeners.add(listener);
    return () => this.publishListeners.delete(listener);
  }

  /** Remove loader-owned media from Comfy's native preview surface. */
  clear(): void {
    this.publish({ images: [], animated: [] });
  }

  private nodeId(): string | undefined {
    if (this.node.id === undefined) return undefined;
    return String(this.node.id);
  }
}
