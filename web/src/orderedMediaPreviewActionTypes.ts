// SimpleSyrup - workflow-focused ComfyUI extensions for image generation
// Copyright (C) 2026  Artificial Sweetener and contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

/** Define the host-facing collaborators used by ordered-media preview actions. */

import type { NativeNodePreview } from "./nativeNodePreview";
import type { NativePreviewSlot } from "./orderedMediaPreviewAffordances";
import type { NativeImageRect } from "./orderedMediaPreviewGeometry";
import type { ComfyApp } from "./types";

export interface CubeFaceProjection {
  readonly container: HTMLElement;
  readonly projectRect: (rect: NativeImageRect) => NativePreviewSlot;
}

export interface ActionPreviewNode {
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
