// SimpleSyrup - workflow-focused ComfyUI extensions for image generation
// Copyright (C) 2026  Artificial Sweetener and contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

import { NativeNodePreview } from "./nativeNodePreview";
import { NativePreviewNavigator } from "./nativePreviewNavigator";
import { loadedImagesMatchReferences } from "./comfyImageReference";
import { parseSegPreviewDocument } from "./segPreviewTypes";
import type {
  ComfyApi,
  ComfyApp,
  ComfyImageResult,
  ComfyNodeExecutionOutput
} from "./types";

interface NativePreviewNode {
  readonly id?: string | number;
  imageIndex?: number | null;
  imageRects?: readonly unknown[];
  images?: ComfyImageResult[];
  imgs?: HTMLImageElement[];
  widgets?: Array<{
    name: string;
    hidden?: boolean;
    options?: { hidden?: boolean };
  }>;
  graph?: {
    id?: string | number;
    setDirtyCanvas?: (foreground: boolean, background: boolean) => void;
  };
}

type SegPreviewExecutionOutput = ComfyNodeExecutionOutput &
  Record<string, unknown> & {
    images: ComfyImageResult[];
  };

type SegPreviewSurfaceMode = "overlay" | "grid" | "detail";

/** Coordinate the overlay with one Comfy-owned grid and detail surface. */
export class SegPreviewNativeSurface {
  private readonly preview: NativeNodePreview;
  private readonly navigator: NativePreviewNavigator;
  private authoritativeOutput: SegPreviewExecutionOutput | undefined;
  private mode: SegPreviewSurfaceMode = "overlay";

  constructor(
    private readonly app: ComfyApp,
    api: ComfyApi,
    private readonly node: NativePreviewNode
  ) {
    this.preview = new NativeNodePreview(app, api, node);
    this.navigator = new NativePreviewNavigator(app, node);
  }

  /** Accept a complete backend result and apply the active surface mode. */
  update(value: unknown): void {
    const output = executionOutput(value);
    if (!output) return;
    const document = parseSegPreviewDocument(value);
    if (!document) return;
    if (output.images.length !== document.regions.length) return;
    this.authoritativeOutput = output;
    if (this.mode === "overlay") {
      this.hideNativePreview();
    } else if (this.mode === "grid") {
      this.navigator.showGrid();
    }
  }

  /** Show only the custom semantic overlay. */
  showOverlay(): void {
    this.mode = "overlay";
    this.hideNativePreview();
  }

  /** Show Comfy's native gallery and return it from native detail mode. */
  showGrid(): void {
    this.mode = "grid";
    this.setCanvasPreviewVisible(true);
    if (this.nativeImagesNeedPublishing()) {
      this.publishAuthoritativeOutput();
    }
    this.navigator.showGrid();
  }

  /** Show one SEG through Comfy's native detail viewer. */
  inspect(index: number): void {
    const output = this.authoritativeOutput;
    if (!output || index < 0 || index >= output.images.length) return;
    this.mode = "detail";
    this.setCanvasPreviewVisible(true);
    if (this.nativeImagesNeedPublishing()) {
      this.publishAuthoritativeOutput();
    }
    this.navigator.inspect(index);
  }

  /** Release native-preview observers owned by this node. */
  dispose(): void {
    this.navigator.dispose();
  }

  private hideNativePreview(): void {
    const usesDomPreview = this.navigator.usesDomPreview();
    this.navigator.hide();
    this.setCanvasPreviewVisible(false);
    if (!usesDomPreview) return;
    this.node.imgs = [];
    const output = this.authoritativeOutput;
    if (!output || output.images.length === 0) return;
    this.preview.publish({
      ...output,
      images: [],
      animated: []
    });
  }

  /** Hide the persistent Nodes 1.0 preview widget outside native modes. */
  private setCanvasPreviewVisible(visible: boolean): void {
    const widget = this.node.widgets?.find(
      (candidate) => candidate.name === "$$canvas-image-preview"
    );
    if (!widget) return;
    widget.hidden = !visible;
    widget.options ??= {};
    widget.options.hidden = !visible;
  }

  private publishAuthoritativeOutput(): void {
    const output = this.authoritativeOutput;
    if (!output) return;
    delete this.node.images;
    delete this.node.imgs;
    this.preview.publish({
      ...output,
      images: [...output.images]
    });
  }

  /** Check whether Comfy's current output still contains every SEG image. */
  private hasAuthoritativeImages(): boolean {
    const output = this.authoritativeOutput;
    if (!output || this.node.id === undefined) return false;
    return (
      this.app.nodeOutputs?.[String(this.node.id)]?.images?.length ===
      output.images.length
    );
  }

  /** Decide whether the active renderer needs its native images republished. */
  private nativeImagesNeedPublishing(): boolean {
    const output = this.authoritativeOutput;
    if (!output) return false;
    if (this.navigator.usesDomPreview()) return !this.hasAuthoritativeImages();
    return !loadedImagesMatchReferences(this.node.imgs, output.images);
  }
}

function executionOutput(value: unknown): SegPreviewExecutionOutput | undefined {
  if (typeof value !== "object" || value === null) return undefined;
  const output = value as Record<string, unknown>;
  if (!Array.isArray(output.images) || !output.images.every(isImageResult)) {
    return undefined;
  }
  return output as SegPreviewExecutionOutput;
}

function isImageResult(value: unknown): value is ComfyImageResult {
  if (typeof value !== "object" || value === null) return false;
  const image = value as Partial<ComfyImageResult>;
  return (
    typeof image.filename === "string" &&
    typeof image.subfolder === "string" &&
    (image.type === "input" || image.type === "output" || image.type === "temp")
  );
}
