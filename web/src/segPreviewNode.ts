// SimpleSyrup - workflow-focused ComfyUI extensions for image generation
// Copyright (C) 2026  Artificial Sweetener and contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

import { InteractiveInspectorController } from "./interactiveInspector";
import { SegPreviewInspector, type PreviewImageLoader } from "./segPreviewInspector";
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
  options?: {
    serialize?: boolean;
    canvasOnly?: boolean;
  };
}

interface PreviewNode {
  constructor: { comfyClass?: string };
  id?: string | number;
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

/** Register the renderer-neutral interactive SEGS inspector node widget. */
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

  const extension: ComfyExtension = {
    name: "SimpleSyrup.SimplePreviewSEGS",
    nodeCreated(candidate: unknown) {
      if (!isPreviewNode(candidate)) return;
      const inspector = new SegPreviewInspector(
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
      widget.computeSize = (width = 420) => [Math.max(360, width), 470];

      const registerId = (): void => {
        if (candidate.id !== undefined) {
          controllers.set(String(candidate.id), controller);
        }
      };
      registerId();

      const originalExecuted = candidate.onExecuted;
      candidate.onExecuted = function (output: unknown): void {
        originalExecuted?.call(this, output);
        controller.update(output);
      };

      const originalGraphConfigured = candidate.onGraphConfigured;
      candidate.onGraphConfigured = function (...args: unknown[]): unknown {
        const result = originalGraphConfigured?.apply(this, args);
        registerId();
        if (candidate.id !== undefined) {
          controller.update(app.nodeOutputs?.[String(candidate.id)]);
        }
        return result;
      };

      const originalRemoved = candidate.onRemoved;
      candidate.onRemoved = function (...args: unknown[]): unknown {
        if (candidate.id !== undefined) controllers.delete(String(candidate.id));
        controller.dispose();
        return originalRemoved?.apply(this, args);
      };

      const computed = candidate.computeSize?.();
      const current = candidate.size ?? computed;
      if (current && candidate.setSize) {
        candidate.setSize([
          Math.max(420, current[0]),
          Math.max(520, computed?.[1] ?? current[1])
        ]);
      }
    },
    onNodeOutputsUpdated(outputs: Record<string, ComfyNodeExecutionOutput>) {
      for (const [nodeId, output] of Object.entries(outputs)) {
        controllers.get(nodeId)?.update(output);
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
