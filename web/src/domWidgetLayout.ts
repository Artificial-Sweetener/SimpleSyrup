// SimpleSyrup - workflow-focused ComfyUI extensions for image generation
// Copyright (C) 2026  Artificial Sweetener and contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

interface FixedDomWidget {
  computeSize?: (width?: number) => [number, number];
  computeLayoutSize?:
    | ((node: unknown) => {
        minHeight: number;
        maxHeight?: number;
        minWidth: number;
        maxWidth?: number;
      })
    | undefined;
}

interface WidgetLayoutNode {
  arrange?: () => void;
  graph?: {
    setDirtyCanvas?: (foreground: boolean, background: boolean) => void;
  };
}

/** Keep a custom DOM widget content-sized in both Comfy node renderers. */
export class FixedDomWidgetLayout {
  constructor(
    private readonly node: WidgetLayoutNode,
    widget: FixedDomWidget,
    measure: (width?: number) => [number, number]
  ) {
    Object.defineProperty(widget, "computeLayoutSize", {
      configurable: true,
      value: undefined,
      writable: true
    });
    widget.computeSize = measure;
  }

  /** Recompute legacy widget allocation after the content mode changes. */
  reflow(): void {
    if (!this.node.graph) return;
    this.node.arrange?.();
    this.node.graph.setDirtyCanvas?.(true, true);
  }
}
