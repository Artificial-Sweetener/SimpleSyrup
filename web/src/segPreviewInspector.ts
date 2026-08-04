// SimpleSyrup - workflow-focused ComfyUI extensions for image generation
// Copyright (C) 2026  Artificial Sweetener and contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

import {
  type AsyncInspectorView,
  type PreparedInspectorState
} from "./interactiveInspector";
import { ImageViewport } from "./imageViewport";
import { maskAtlasFromImage, type MaskAtlas } from "./maskAtlas";
import { comfyImageUrl } from "./comfyImageUrl";
import type {
  SegPreviewDocument,
  SegPreviewRegion
} from "./segPreviewTypes";
import { SelectionModel, type SelectionState } from "./selectionModel";

export type PreviewImageLoader = (url: string) => Promise<HTMLImageElement>;
export type SegPreviewMode = "overlay" | "grid";

interface CommittedOverlay {
  document: SegPreviewDocument;
  image: HTMLImageElement;
  atlas: MaskAtlas;
  viewport: ImageViewport;
  masks: Map<string, HTMLCanvasElement>;
  selection: SelectionModel<string>;
}

/** Render the SEGS overlay and switch visibility to Comfy's native gallery. */
export class SegPreviewInspector
  implements AsyncInspectorView<SegPreviewDocument>
{
  readonly element = document.createElement("section");
  private readonly overlayButton = modeButton("Overlay");
  private readonly gridButton = modeButton("Grid");
  private readonly body = document.createElement("div");
  private readonly status = document.createElement("div");
  private mode: SegPreviewMode = "overlay";
  private committed: CommittedOverlay | undefined;
  private unsubscribeSelection: (() => void) | undefined;
  private highlightCanvas: HTMLCanvasElement | undefined;

  constructor(
    private readonly inspectRegion: (index: number) => void = () => undefined,
    private readonly changeMode: (mode: SegPreviewMode) => void = () => undefined,
    private readonly apiURL: (path: string) => string = (path) => path,
    private readonly loadImage: PreviewImageLoader = loadPreviewImage
  ) {
    installStyles();
    this.element.className = "ss-segs-preview";
    this.body.className = "ss-segs-preview__body";
    this.status.className = "ss-segs-preview__status";
    this.status.setAttribute("aria-live", "polite");
    const toolbar = document.createElement("nav");
    toolbar.className = "ss-segs-preview__toolbar";
    toolbar.setAttribute("aria-label", "SEGS preview mode");
    toolbar.append(this.overlayButton, this.gridButton);
    this.overlayButton.addEventListener("click", () => {
      this.setMode("overlay");
    });
    this.gridButton.addEventListener("click", () => {
      this.setMode("grid");
    });
    this.element.append(toolbar, this.body, this.status);
    this.updateMode();
    this.showMessage("Run the workflow to inspect SEGS.");
  }

  /** Return the renderer height needed by the active preview surface. */
  preferredHeight(): number {
    return this.mode === "overlay" ? 360 : 38;
  }

  /** Select the overlay or native gallery and optionally notify its owner. */
  setMode(mode: SegPreviewMode, notifyOwner = true): void {
    if (mode === this.mode) {
      if (notifyOwner) this.changeMode(mode);
      return;
    }
    this.mode = mode;
    this.updateMode();
    this.renderMode();
    if (notifyOwner) this.changeMode(mode);
  }

  /** Show a loading state while execution assets are decoded. */
  setLoading(): void {
    this.showMessage("Loading SEGS preview…");
  }

  /** Load source and atlas assets without publishing partial state. */
  async prepare(document: SegPreviewDocument): Promise<PreparedInspectorState> {
    const [image, atlasImage] = await Promise.all([
      this.loadImage(comfyImageUrl(document.preview.image, this.apiURL)),
      this.loadImage(comfyImageUrl(document.atlas.image, this.apiURL))
    ]);
    const atlas = maskAtlasFromImage(document, atlasImage);
    let disposed = false;
    return {
      commit: () => {
        if (!disposed) this.commit(document, image, atlas);
      },
      dispose: () => {
        disposed = true;
      }
    };
  }

  /** Show an actionable error without breaking the node lifecycle. */
  showError(message: string): void {
    this.showMessage(message, true);
  }

  /** Release selection subscriptions and rendered state. */
  dispose(): void {
    this.unsubscribeSelection?.();
    this.unsubscribeSelection = undefined;
    this.committed = undefined;
    this.highlightCanvas = undefined;
    this.body.replaceChildren();
    this.status.textContent = "";
  }

  private commit(
    document: SegPreviewDocument,
    image: HTMLImageElement,
    atlas: MaskAtlas
  ): void {
    this.unsubscribeSelection?.();
    const selection = new SelectionModel<string>();
    const masks = new Map(
      document.regions.map((region) => [
        region.id,
        maskCanvas(atlas, region)
      ])
    );
    this.committed = {
      document,
      image,
      atlas,
      viewport: new ImageViewport(document.source, document.preview),
      masks,
      selection
    };
    this.unsubscribeSelection = selection.subscribe((state) => {
      this.renderSelection(state);
    });
    this.renderMode();
  }

  private renderMode(): void {
    const committed = this.committed;
    if (!committed) return;
    if (this.mode === "overlay") {
      this.renderOverlay();
      return;
    }
    this.highlightCanvas = undefined;
    this.body.replaceChildren();
    this.status.textContent = `${String(committed.document.regions.length)} regions`;
  }

  private renderOverlay(): void {
    const committed = required(this.committed);
    const stack = document.createElement("div");
    stack.className = "ss-segs-preview__canvas-stack";
    const base = sizedCanvas(committed.document.preview);
    const highlight = sizedCanvas(committed.document.preview);
    highlight.className = "ss-segs-preview__highlight";
    this.highlightCanvas = highlight;
    const context = context2d(base);
    context.drawImage(
      committed.image,
      0,
      0,
      committed.document.preview.width,
      committed.document.preview.height
    );
    context.globalAlpha = 0.38;
    for (const region of committed.document.regions) {
      drawRegionMask(context, committed, region);
    }
    context.globalAlpha = 1;
    highlight.addEventListener("pointermove", (event) => {
      const point = committed.viewport.sourcePoint(
        event.clientX,
        event.clientY,
        highlight.getBoundingClientRect()
      );
      const hits = committed.atlas.hitsAt(point.x, point.y);
      committed.selection.hover(hits.map((region) => region.id));
    });
    highlight.addEventListener("pointerleave", () => {
      committed.selection.clearHover();
    });
    highlight.addEventListener("click", () => {
      const selected = committed.selection.selectNextCandidate();
      const region = committed.document.regions.find(
        (candidate) => candidate.id === selected
      );
      if (region) {
        this.setMode("grid", false);
        this.inspectRegion(region.index);
      }
      committed.selection.select(null);
    });
    stack.append(base, highlight);
    this.body.replaceChildren(stack);
    this.renderSelection(committed.selection.state());
  }

  private renderSelection(state: SelectionState<string>): void {
    const committed = this.committed;
    const highlight = this.highlightCanvas;
    if (!committed || !highlight) return;
    const active = state.active
      ? committed.document.regions.find((region) => region.id === state.active)
      : undefined;
    const context = context2d(highlight);
    context.clearRect(0, 0, highlight.width, highlight.height);
    if (active) {
      context.save();
      context.globalAlpha = 0.9;
      context.shadowColor = "rgba(255, 255, 255, 0.95)";
      context.shadowBlur = 8;
      drawRegionMask(context, committed, active);
      context.restore();
    }
    if (!active) {
      this.status.textContent = `${String(committed.document.regions.length)} regions`;
    } else if (state.candidates.length > 1) {
      this.status.textContent = `${regionTitle(active)} · ${String(state.candidates.length)} overlapping regions · click to inspect`;
    } else {
      this.status.textContent = `${regionTitle(active)} · click to inspect`;
    }
  }

  private showMessage(message: string, error = false): void {
    const content = document.createElement("div");
    content.className = "ss-segs-preview__message";
    content.dataset.error = String(error);
    content.textContent = message;
    this.body.replaceChildren(content);
    this.status.textContent = "";
  }

  private updateMode(): void {
    this.element.dataset.mode = this.mode;
    this.overlayButton.setAttribute(
      "aria-pressed",
      String(this.mode === "overlay")
    );
    this.gridButton.setAttribute("aria-pressed", String(this.mode === "grid"));
  }
}

function modeButton(label: string): HTMLButtonElement {
  const button = document.createElement("button");
  button.type = "button";
  button.textContent = label;
  return button;
}

function drawRegionMask(
  context: CanvasRenderingContext2D,
  committed: CommittedOverlay,
  region: SegPreviewRegion
): void {
  const placement = committed.viewport.displayRectangle(region.crop);
  const mask = committed.masks.get(region.id);
  if (!mask) return;
  context.drawImage(
    mask,
    placement.x,
    placement.y,
    placement.width,
    placement.height
  );
}

function maskCanvas(atlas: MaskAtlas, region: SegPreviewRegion): HTMLCanvasElement {
  const canvas = sizedCanvas(region.atlas);
  context2d(canvas).putImageData(atlas.coloredMask(region), 0, 0);
  return canvas;
}

function sizedCanvas(size: { width: number; height: number }): HTMLCanvasElement {
  const canvas = document.createElement("canvas");
  canvas.width = size.width;
  canvas.height = size.height;
  return canvas;
}

function context2d(canvas: HTMLCanvasElement): CanvasRenderingContext2D {
  const context = canvas.getContext("2d");
  if (!context) throw new Error("Simple Preview SEGS requires canvas rendering.");
  return context;
}

function regionTitle(region: SegPreviewRegion): string {
  const label = region.label.trim();
  return `${String(region.index + 1)}. ${label || "region"}`;
}

function required<T>(value: T | undefined): T {
  if (!value) throw new Error("Simple Preview SEGS has no committed document.");
  return value;
}

function loadPreviewImage(url: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.addEventListener("load", () => {
      resolve(image);
    }, { once: true });
    image.addEventListener(
      "error",
      () => {
        reject(new Error("Simple Preview SEGS could not load a preview asset."));
      },
      { once: true }
    );
    image.src = url;
  });
}

let stylesInstalled = false;

function installStyles(): void {
  if (stylesInstalled) return;
  const style = document.createElement("style");
  style.dataset.simpleSyrupSegsPreview = "true";
  style.textContent = `
    .ss-segs-preview { box-sizing: border-box; width: 100%; color: var(--fg-color, #ddd); font: 12px sans-serif; }
    .ss-segs-preview * { box-sizing: border-box; }
    .ss-segs-preview__toolbar { display: flex; gap: 4px; margin: 0 0 6px; }
    .ss-segs-preview__toolbar button { flex: 1; min-height: 26px; border: 1px solid var(--border-color, #555); border-radius: 5px; color: inherit; background: var(--comfy-input-bg, #222); cursor: pointer; }
    .ss-segs-preview__toolbar button[aria-pressed="true"] { border-color: var(--p-primary-color, #6aa9ff); background: color-mix(in srgb, var(--p-primary-color, #6aa9ff) 28%, var(--comfy-input-bg, #222)); }
    .ss-segs-preview__body { overflow: hidden; border: 1px solid var(--border-color, #444); border-radius: 6px; background: var(--comfy-menu-bg, #181818); }
    .ss-segs-preview[data-mode="grid"] .ss-segs-preview__body { display: none; }
    .ss-segs-preview[data-mode="grid"] .ss-segs-preview__status { display: none; }
    .ss-segs-preview__canvas-stack { position: relative; line-height: 0; background: #111; }
    .ss-segs-preview__canvas-stack canvas { display: block; width: 100%; height: auto; }
    .ss-segs-preview__highlight { position: absolute; inset: 0; cursor: crosshair; }
    .ss-segs-preview__status { min-height: 22px; padding: 5px 2px 0; color: var(--descrip-text, #aaa); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .ss-segs-preview__message { display: grid; min-height: 160px; place-items: center; padding: 18px; color: var(--descrip-text, #aaa); text-align: center; }
    .ss-segs-preview__message[data-error="true"] { color: var(--error-text, #ff8a80); }
  `;
  document.head.append(style);
  stylesInstalled = true;
}
