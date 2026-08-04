// SimpleSyrup - workflow-focused ComfyUI extensions for image generation
// Copyright (C) 2026  Artificial Sweetener and contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

import { InteractiveInspectorController } from "./interactiveInspector";
import { FixedDomWidgetLayout } from "./domWidgetLayout";
import { SegPreviewInspector, type PreviewImageLoader } from "./segPreviewInspector";
import { SegPreviewNativeSurface } from "./segPreviewNativeSurface";
import {
  parseSegPreviewDocument,
  type SegPreviewDocument
} from "./segPreviewTypes";
import type {
  ComfyApi,
  ComfyApp,
  ComfyExtension,
  ComfyNodeExecutionOutput,
  Logger
} from "./types";

const SIMPLE_PREVIEW_SEGS_NODE_ID = "SimpleSyrup.SimplePreviewSEGS";

interface DomWidget {
  serialize?: boolean;
  computeSize?: (width?: number) => [number, number];
  computeLayoutSize?:
    | ((node: unknown) => {
        minHeight: number;
        maxHeight?: number;
        minWidth: number;
        maxWidth?: number;
      })
    | undefined;
  options?: {
    serialize?: boolean;
    canvasOnly?: boolean;
  };
}

interface PreviewNode {
  constructor: { comfyClass?: string };
  id?: string | number;
  imageIndex?: number | null;
  imageRects?: readonly unknown[];
  imgs?: HTMLImageElement[];
  graph?: { setDirtyCanvas?: (foreground: boolean, background: boolean) => void };
  arrange?: () => void;
  size?: [number, number];
  onExecuted?: (output: unknown) => void;
  onGraphConfigured?: (...args: unknown[]) => unknown;
  onRemoved?: (...args: unknown[]) => unknown;
  computeSize?: () => [number, number];
  setSize?: (size: [number, number]) => void;
  addDOMWidget(
    name: string,
    type: string,
    element: HTMLElement,
    options?: DomWidget["options"]
  ): DomWidget;
}

/** Register the SEGS overlay while delegating browsing to Comfy's native preview. */
export function registerSimplePreviewSEGS(
  app: ComfyApp,
  api: ComfyApi,
  loadImage?: PreviewImageLoader,
  logger: Logger = console
): void {
  const controllers = new Map<
    string,
    InteractiveInspectorController<SegPreviewDocument>
  >();
  const nativeSurfaces = new Map<string, SegPreviewNativeSurface>();
  const widgetLayouts = new Map<string, FixedDomWidgetLayout>();

  const extension: ComfyExtension = {
    name: "SimpleSyrup.SimplePreviewSEGS",
    nodeCreated(candidate: unknown) {
      if (!isPreviewNode(candidate)) return;
      const nativeSurface = new SegPreviewNativeSurface(app, api, candidate);
      const layoutOwner: { current?: FixedDomWidgetLayout } = {};
      const inspector = new SegPreviewInspector(
        (index) => {
          nativeSurface.inspect(index);
          layoutOwner.current?.reflow();
        },
        (mode) => {
          if (mode === "overlay") nativeSurface.showOverlay();
          else nativeSurface.showGrid();
          layoutOwner.current?.reflow();
        },
        (path) => api.apiURL?.(path) ?? path,
        loadImage
      );
      const controller = new InteractiveInspectorController(
        inspector,
        parseSegPreviewDocument,
        logger
      );
      const widget = candidate.addDOMWidget(
        "simple_syrup_segs_preview",
        "simple_syrup_segs_preview",
        inspector.element,
        { serialize: false, canvasOnly: false }
      );
      widget.serialize = false;
      widget.options ??= {};
      widget.options.serialize = false;
      widget.options.canvasOnly = false;
      const widgetLayout = new FixedDomWidgetLayout(candidate, widget, (width = 420) => [
        Math.max(360, width),
        inspector.preferredHeight()
      ]);
      layoutOwner.current = widgetLayout;

      const registerId = (): void => {
        if (candidate.id !== undefined) {
          const nodeId = String(candidate.id);
          controllers.set(nodeId, controller);
          nativeSurfaces.set(nodeId, nativeSurface);
          widgetLayouts.set(nodeId, widgetLayout);
        }
      };
      registerId();

      const originalExecuted = candidate.onExecuted;
      candidate.onExecuted = function (output: unknown): void {
        originalExecuted?.call(this, output);
        controller.update(output);
        nativeSurface.update(output);
        widgetLayout.reflow();
      };

      const originalGraphConfigured = candidate.onGraphConfigured;
      candidate.onGraphConfigured = function (...args: unknown[]): unknown {
        const result = originalGraphConfigured?.apply(this, args);
        registerId();
        if (candidate.id !== undefined) {
          const output = app.nodeOutputs?.[String(candidate.id)];
          controller.update(output);
          nativeSurface.update(output);
          widgetLayout.reflow();
        }
        return result;
      };

      const originalRemoved = candidate.onRemoved;
      candidate.onRemoved = function (...args: unknown[]): unknown {
        if (candidate.id !== undefined) {
          const nodeId = String(candidate.id);
          controllers.delete(nodeId);
          nativeSurfaces.delete(nodeId);
          widgetLayouts.delete(nodeId);
        }
        controller.dispose();
        nativeSurface.dispose();
        return originalRemoved?.apply(this, args);
      };

      const computed = candidate.computeSize?.();
      const current = candidate.size ?? computed;
      if (current && candidate.setSize) {
        candidate.setSize([
          Math.max(420, current[0]),
          Math.max(420, computed?.[1] ?? current[1])
        ]);
      }
    },
    onNodeOutputsUpdated(outputs: Record<string, ComfyNodeExecutionOutput>) {
      for (const [nodeId, output] of Object.entries(outputs)) {
        controllers.get(nodeId)?.update(output);
        nativeSurfaces.get(nodeId)?.update(output);
        widgetLayouts.get(nodeId)?.reflow();
      }
    }
  };
  app.registerExtension(extension);
}

function isPreviewNode(value: unknown): value is PreviewNode {
  if (typeof value !== "object" || value === null) return false;
  const node = value as Partial<PreviewNode>;
  return (
    node.constructor?.comfyClass === SIMPLE_PREVIEW_SEGS_NODE_ID &&
    typeof node.addDOMWidget === "function"
  );
}
