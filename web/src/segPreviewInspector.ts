// SimpleSyrup - workflow-focused ComfyUI extensions for image generation
// Copyright (C) 2026  Artificial Sweetener and contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

import {
  type AsyncInspectorView,
  type PreparedInspectorState
} from "./interactiveInspector";
import { ImageViewport } from "./imageViewport";
import { maskAtlasFromImage } from "./maskAtlas";
import type { MaskAtlas } from "./maskAtlas";
import type {
  SegPreviewDocument,
  SegPreviewRegion
} from "./segPreviewTypes";
import { previewAssetUrl } from "./segPreviewTypes";
import { SelectionModel, type SelectionState } from "./selectionModel";

type PreviewMode = "overlay" | "grid";
export type PreviewImageLoader = (url: string) => Promise<HTMLImageElement>;

interface CommittedPreview {
  document: SegPreviewDocument;
  image: HTMLImageElement;
  atlas: MaskAtlas;
  viewport: ImageViewport;
  masks: Map<string, HTMLCanvasElement>;
  selection: SelectionModel<string>;
}

/** Render the interactive SEGS inspector inside Comfy's native DOM widget. */
export class SegPreviewInspector
  implements AsyncInspectorView<SegPreviewDocument>
{
  readonly element = document.createElement("section");
  private readonly body = document.createElement("div");
  private readonly status = document.createElement("div");
  private readonly overlayButton = modeButton("Overlay");
  private readonly gridButton = modeButton("Grid");
  private mode: PreviewMode = "overlay";
  private committed: CommittedPreview | undefined;
  private unsubscribeSelection: (() => void) | undefined;
  private highlightCanvas: HTMLCanvasElement | undefined;
  private focusContainer: HTMLElement | undefined;
  private gridButtons = new Map<string, HTMLButtonElement>();

  constructor(
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
    this.updateModeButtons();
    this.showMessage("Run the workflow to inspect SEGS.");
  }

  /** Show a native loading state while execution assets are decoded. */
  setLoading(): void {
    this.showMessage("Loading SEGS preview…");
  }

  /** Load source and atlas assets without publishing partial state. */
  async prepare(document: SegPreviewDocument): Promise<PreparedInspectorState> {
    const [image, atlasImage] = await Promise.all([
      this.loadImage(previewAssetUrl(document.preview.image, this.apiURL)),
      this.loadImage(previewAssetUrl(document.atlas.image, this.apiURL))
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

  /** Replace the inspector with an actionable failure message. */
  showError(message: string): void {
    this.showMessage(message, true);
  }

  /** Release node-owned DOM and interaction subscriptions. */
  dispose(): void {
    this.unsubscribeSelection?.();
    this.unsubscribeSelection = undefined;
    this.committed = undefined;
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

  private setMode(mode: PreviewMode): void {
    if (this.mode === mode) return;
    this.mode = mode;
    this.updateModeButtons();
    this.renderMode();
  }

  private updateModeButtons(): void {
    this.overlayButton.setAttribute(
      "aria-pressed",
      String(this.mode === "overlay")
    );
    this.gridButton.setAttribute("aria-pressed", String(this.mode === "grid"));
  }

  private renderMode(): void {
    this.highlightCanvas = undefined;
    this.focusContainer = undefined;
    this.gridButtons.clear();
    if (!this.committed) return;
    this.body.replaceChildren(
      this.mode === "overlay" ? this.overlayView() : this.gridView()
    );
    this.renderSelection(this.committed.selection.state());
  }

  private overlayView(): HTMLElement {
    const committed = required(this.committed);
    const container = document.createElement("div");
    container.className = "ss-segs-preview__overlay-view";
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
      committed.selection.selectNextCandidate();
    });
    stack.append(base, highlight);
    this.focusContainer = document.createElement("div");
    this.focusContainer.className = "ss-segs-preview__focus";
    container.append(stack, this.focusContainer);
    return container;
  }

  private gridView(): HTMLElement {
    const committed = required(this.committed);
    const grid = document.createElement("div");
    grid.className = "ss-segs-preview__grid";
    grid.setAttribute("role", "listbox");
    if (committed.document.regions.length === 0) {
      const empty = document.createElement("div");
      empty.className = "ss-segs-preview__empty";
      empty.textContent = "No SEGS to display.";
      grid.append(empty);
      return grid;
    }
    for (const region of committed.document.regions) {
      const card = document.createElement("button");
      card.type = "button";
      card.className = "ss-segs-preview__card";
      card.setAttribute("role", "option");
      card.dataset.regionId = region.id;
      const canvas = regionPreviewCanvas(committed, region, 160, 132);
      const label = document.createElement("span");
      label.textContent = regionTitle(region);
      card.append(canvas, label);
      card.addEventListener("pointerenter", () => {
        committed.selection.hover([region.id]);
      });
      card.addEventListener("pointerleave", () => {
        committed.selection.clearHover();
      });
      card.addEventListener("click", () => {
        committed.selection.select(region.id);
      });
      this.gridButtons.set(region.id, card);
      grid.append(card);
    }
    return grid;
  }

  private renderSelection(state: SelectionState<string>): void {
    const committed = this.committed;
    if (!committed) return;
    const active = state.active
      ? committed.document.regions.find((region) => region.id === state.active)
      : undefined;
    if (this.highlightCanvas) {
      const context = context2d(this.highlightCanvas);
      context.clearRect(
        0,
        0,
        this.highlightCanvas.width,
        this.highlightCanvas.height
      );
      if (active) {
        context.save();
        context.globalAlpha = 0.9;
        context.shadowColor = "rgba(255, 255, 255, 0.95)";
        context.shadowBlur = 8;
        drawRegionMask(context, committed, active);
        context.restore();
      }
    }
    for (const [id, button] of this.gridButtons) {
      const selected = id === state.selected;
      const highlighted = id === active?.id;
      button.setAttribute("aria-selected", String(selected));
      button.dataset.highlighted = String(highlighted);
    }
    this.renderFocus(active);
    if (!active) {
      this.status.textContent = `${String(committed.document.regions.length)} regions`;
    } else if (state.candidates.length > 1) {
      this.status.textContent = `${regionTitle(active)} · ${String(state.candidates.length)} overlapping regions · click to cycle`;
    } else {
      this.status.textContent = regionTitle(active);
    }
  }

  private renderFocus(region: SegPreviewRegion | undefined): void {
    if (!this.focusContainer || !this.committed) return;
    if (!region) {
      this.focusContainer.replaceChildren();
      return;
    }
    const title = document.createElement("strong");
    title.textContent = regionTitle(region);
    const details = document.createElement("span");
    details.textContent = `${String(Math.round(region.confidence * 100))}% confidence · ${region.area.toLocaleString()} px`;
    this.focusContainer.replaceChildren(
      regionPreviewCanvas(this.committed, region, 320, 220),
      title,
      details
    );
  }

  private showMessage(message: string, error = false): void {
    const content = document.createElement("div");
    content.className = "ss-segs-preview__message";
    content.dataset.error = String(error);
    content.textContent = message;
    this.body.replaceChildren(content);
    this.status.textContent = "";
  }
}

function drawRegionMask(
  context: CanvasRenderingContext2D,
  committed: CommittedPreview,
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

function regionPreviewCanvas(
  committed: CommittedPreview,
  region: SegPreviewRegion,
  maximumWidth: number,
  maximumHeight: number
): HTMLCanvasElement {
  const aspect = region.crop.width / region.crop.height;
  const width = Math.max(80, Math.min(maximumWidth, Math.round(maximumHeight * aspect)));
  const height = Math.max(64, Math.min(maximumHeight, Math.round(width / aspect)));
  const canvas = sizedCanvas({ width, height });
  const context = context2d(canvas);
  const source = committed.viewport.displayRectangle(region.crop);
  context.drawImage(
    committed.image,
    source.x,
    source.y,
    source.width,
    source.height,
    0,
    0,
    width,
    height
  );
  context.globalAlpha = 0.45;
  const mask = committed.masks.get(region.id);
  if (mask) context.drawImage(mask, 0, 0, width, height);
  context.globalAlpha = 1;
  return canvas;
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

function modeButton(label: string): HTMLButtonElement {
  const button = document.createElement("button");
  button.type = "button";
  button.textContent = label;
  return button;
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
    .ss-segs-preview { box-sizing: border-box; width: 100%; min-height: 360px; color: var(--fg-color, #ddd); font: 12px sans-serif; }
    .ss-segs-preview * { box-sizing: border-box; }
    .ss-segs-preview__toolbar { display: flex; gap: 4px; margin: 0 0 6px; }
    .ss-segs-preview__toolbar button { flex: 1; min-height: 26px; border: 1px solid var(--border-color, #555); border-radius: 5px; color: inherit; background: var(--comfy-input-bg, #222); cursor: pointer; }
    .ss-segs-preview__toolbar button[aria-pressed="true"] { border-color: var(--p-primary-color, #6aa9ff); background: color-mix(in srgb, var(--p-primary-color, #6aa9ff) 28%, var(--comfy-input-bg, #222)); }
    .ss-segs-preview__body { min-height: 320px; overflow: hidden; border: 1px solid var(--border-color, #444); border-radius: 6px; background: var(--comfy-menu-bg, #181818); }
    .ss-segs-preview__canvas-stack { position: relative; line-height: 0; background: #111; }
    .ss-segs-preview__canvas-stack canvas { display: block; width: 100%; height: auto; }
    .ss-segs-preview__highlight { position: absolute; inset: 0; cursor: crosshair; }
    .ss-segs-preview__status { min-height: 22px; padding: 5px 2px 0; color: var(--descrip-text, #aaa); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .ss-segs-preview__focus { display: grid; grid-template-columns: minmax(96px, 42%) 1fr; gap: 3px 9px; align-items: start; padding: 7px; border-top: 1px solid var(--border-color, #444); }
    .ss-segs-preview__focus canvas { grid-row: 1 / span 2; width: 100%; height: auto; border-radius: 4px; background: #111; }
    .ss-segs-preview__focus span { color: var(--descrip-text, #aaa); }
    .ss-segs-preview__grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(124px, 1fr)); gap: 6px; max-height: 430px; padding: 7px; overflow: auto; }
    .ss-segs-preview__card { min-width: 0; padding: 4px; border: 1px solid var(--border-color, #444); border-radius: 5px; color: inherit; background: var(--comfy-input-bg, #222); cursor: pointer; text-align: left; }
    .ss-segs-preview__card[data-highlighted="true"], .ss-segs-preview__card[aria-selected="true"] { border-color: var(--p-primary-color, #6aa9ff); box-shadow: 0 0 0 1px var(--p-primary-color, #6aa9ff); }
    .ss-segs-preview__card canvas { display: block; width: 100%; height: 94px; object-fit: contain; margin-bottom: 4px; background: #111; }
    .ss-segs-preview__card span { display: block; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .ss-segs-preview__message, .ss-segs-preview__empty { display: grid; min-height: 320px; place-items: center; padding: 18px; color: var(--descrip-text, #aaa); text-align: center; }
    .ss-segs-preview__message[data-error="true"] { color: var(--error-text, #ff8a80); }
  `;
  document.head.append(style);
  stylesInstalled = true;
}
