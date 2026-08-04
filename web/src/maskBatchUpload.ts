// SimpleSyrup - workflow-focused ComfyUI extensions for image generation
// Copyright (C) 2026  Artificial Sweetener and contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

import { getMaskBatchPreview } from "./api";
import {
  configureOrderedMediaNode,
  registerOrderedMediaNode,
  type OrderedMediaNode
} from "./orderedMediaNode";
import type { ComfyApi, ComfyApp, Logger } from "./types";

const LOAD_MASK_BATCH_NODE_ID = "SimpleSyrup.LoadMaskBatch";

export type MaskBatchPreviewClient = typeof getMaskBatchPreview;

/** Register Load Mask Batch through the shared ordered-media interface. */
export function registerMaskBatchUpload(
  app: ComfyApp,
  api: ComfyApi,
  loadPreview: MaskBatchPreviewClient = getMaskBatchPreview,
  logger: Logger = console
): void {
  registerOrderedMediaNode(
    app,
    api,
    "SimpleSyrup.LoadMaskBatchUpload",
    maskConfig(loadPreview),
    logger
  );
}

/** Configure one mask loader while retaining selected-channel rendering. */
export function configureMaskBatchNode(
  candidate: unknown,
  app: ComfyApp,
  api: ComfyApi,
  loadPreview: MaskBatchPreviewClient = getMaskBatchPreview,
  logger: Logger = console
): void {
  configureOrderedMediaNode(
    candidate,
    app,
    api,
    {
      ...maskConfig(loadPreview),
      preview: (files, node) => loadPreview(files, selectedChannel(node))
    },
    logger
  );
}

function maskConfig(loadPreview: MaskBatchPreviewClient) {
  return {
    nodeId: LOAD_MASK_BATCH_NODE_ID,
    labels: {
      singular: "mask",
      plural: "masks",
      replace: "Replace masks...",
      add: "Add masks..."
    },
    previewWidgetNames: ["channel"],
    preview: (files: string[], node: OrderedMediaNode) =>
      loadPreview(files, selectedChannel(node))
  };
}

function selectedChannel(node: OrderedMediaNode): string {
  const value = node.widgets?.find((widget) => widget.name === "channel")?.value;
  return typeof value === "string" && value.length > 0 ? value : "alpha";
}
