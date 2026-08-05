// SimpleSyrup - workflow-focused ComfyUI extensions for image generation
// Copyright (C) 2026  Artificial Sweetener and contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

import { NativeNodePreview } from "./nativeNodePreview";
import { OrderedMediaPreviewController } from "./orderedMediaPreview";
import { OrderedMediaPreviewActions } from "./orderedMediaPreviewActions";
import {
  normalizeMediaFiles,
  OrderedMediaSelection
} from "./orderedMediaSelection";
import type {
  ComfyApi,
  ComfyApp,
  ComfyNodeExecutionOutput,
  Logger
} from "./types";

type SelectionIntent = "append" | "replace";
type NativeCallback = (...args: unknown[]) => unknown;
type WidgetType = "button";

export interface OrderedMediaLabels {
  readonly singular: string;
  readonly plural: string;
  readonly replace: string;
  readonly add: string;
}

export interface OrderedMediaNodeConfig {
  readonly nodeId: string;
  readonly labels: OrderedMediaLabels;
  readonly preview: (
    files: string[],
    node: OrderedMediaNode
  ) => Promise<ComfyNodeExecutionOutput>;
  readonly onSelectionChanged?: (files: string[]) => void;
  readonly previewWidgetNames?: readonly string[];
}

interface OrderedMediaWidgetOptions {
  canvasOnly?: boolean;
  serialize?: boolean;
  tooltip?: string;
  hidden?: boolean;
}

interface OrderedMediaWidget {
  name: string;
  value: unknown;
  type?: string;
  label?: string;
  callback?: (value?: unknown) => void;
  computeSize?: (width?: number) => [number, number];
  hidden?: boolean;
  options?: OrderedMediaWidgetOptions;
}

export interface OrderedMediaNode {
  constructor: { comfyClass?: string };
  id?: string | number;
  widgets?: OrderedMediaWidget[];
  widgets_values?: unknown[];
  pasteFiles?: NativeCallback;
  onDragDrop?: NativeCallback;
  onRemoved?: NativeCallback;
  onGraphConfigured?: NativeCallback;
  onWidgetChanged?: (
    name: string,
    value: unknown,
    previousValue: unknown,
    widget: OrderedMediaWidget
  ) => void;
  graph?: {
    id?: string | number;
    setDirtyCanvas?: (foreground: boolean, background: boolean) => void;
  };
  addWidget(
    type: WidgetType,
    name: string,
    value: string,
    callback: (value?: unknown) => void,
    options?: OrderedMediaWidgetOptions
  ): OrderedMediaWidget;
}

/** Configure one ordered loader around Comfy's native media preview. */
export function configureOrderedMediaNode(
  candidate: unknown,
  app: ComfyApp,
  api: ComfyApi,
  config: OrderedMediaNodeConfig,
  logger: Logger = console
): void {
  if (!isOrderedMediaNode(candidate, config.nodeId)) return;
  const imageWidget = findWidget(candidate, "image");
  const uploadWidget = findNativeUploadWidget(candidate);
  if (!imageWidget || !uploadWidget?.callback) return;

  hideInternalWidget(imageWidget);
  hideInternalWidget(uploadWidget);
  const selection = new OrderedMediaSelection(imageWidget.value);
  const nativePreview = new NativeNodePreview(app, api, candidate);
  const preview = new OrderedMediaPreviewController(
    nativePreview,
    `${config.labels.singular} loader`,
    logger
  );
  let programmaticSelectionUpdate = false;
  let selectionIntent: SelectionIntent = "replace";
  let appendBase: string[] = [];
  let uploadPending = false;
  let ignoredUploadCallbackFiles: string[] | undefined;
  const nativeUploadCallback = uploadWidget.callback;

  const setPersistedFiles = (files: string[]): void => {
    programmaticSelectionUpdate = true;
    try {
      imageWidget.value = [...files];
    } finally {
      programmaticSelectionUpdate = false;
    }
  };

  const refresh = (files: string[]): void => {
    config.onSelectionChanged?.([...files]);
    preview.refresh(files, () => config.preview([...files], candidate));
  };

  const commit = (current: string[], previous: string[]): void => {
    setPersistedFiles(current);
    candidate.onWidgetChanged?.(
      imageWidget.name,
      [...current],
      [...previous],
      imageWidget
    );
    candidate.graph?.setDirtyCanvas?.(true, true);
    refresh(current);
  };

  const removeAt = (index: number): void => {
    const previous = selection.snapshot();
    if (index < 0 || index >= previous.length) return;
    const current = selection.remove(index);
    commit(current, previous);
  };

  const moveTo = (index: number, destination: number): void => {
    const previous = selection.snapshot();
    if (
      index < 0 ||
      index >= previous.length ||
      destination < 0 ||
      destination >= previous.length ||
      index === destination
    ) {
      return;
    }
    const current = selection.move(index, destination);
    commit(current, previous);
  };

  const previewActions = new OrderedMediaPreviewActions({
    app,
    node: candidate,
    preview: nativePreview,
    itemLabel: config.labels.singular,
    getFiles: () => selection.snapshot(),
    moveEarlier: (index) => {
      moveTo(index, index - 1);
    },
    moveLater: (index) => {
      moveTo(index, index + 1);
    },
    remove: removeAt
  });

  const beginReplace = (): void => {
    selectionIntent = "replace";
    appendBase = [];
    uploadPending = true;
  };
  const beginAppend = (): void => {
    selectionIntent = "append";
    appendBase = selection.snapshot();
    uploadPending = true;
  };

  nativeButton(candidate, "simple_syrup_replace_media", config.labels.replace, () => {
    beginReplace();
    nativeUploadCallback.call(uploadWidget);
  });
  nativeButton(candidate, "simple_syrup_add_media", config.labels.add, () => {
    beginAppend();
    nativeUploadCallback.call(uploadWidget);
  });

  uploadWidget.callback = (value?: unknown) => {
    beginReplace();
    nativeUploadCallback.call(uploadWidget, value);
  };

  imageWidget.callback = (value?: unknown) => {
    if (programmaticSelectionUpdate) return;
    const callbackValue = value ?? imageWidget.value;
    if (uploadPending && !Array.isArray(callbackValue)) return;
    const incoming = normalizeMediaFiles(callbackValue);
    if (
      ignoredUploadCallbackFiles &&
      sameMediaFiles(incoming, ignoredUploadCallbackFiles)
    ) {
      ignoredUploadCallbackFiles = undefined;
      return;
    }
    ignoredUploadCallbackFiles = uploadPending ? [...incoming] : undefined;
    uploadPending = false;
    const current =
      selectionIntent === "append"
        ? selection.replace([...appendBase, ...incoming])
        : selection.replace(incoming);
    selectionIntent = "replace";
    appendBase = [];
    if (!sameMediaFiles(current, normalizeMediaFiles(imageWidget.value))) {
      setPersistedFiles(current);
    }
    candidate.graph?.setDirtyCanvas?.(true, true);
    refresh(current);
  };

  for (const widgetName of config.previewWidgetNames ?? []) {
    const widget = findWidget(candidate, widgetName);
    if (!widget) continue;
    const originalCallback = widget.callback;
    widget.callback = (value?: unknown) => {
      originalCallback?.call(widget, value);
      if (value !== undefined) widget.value = value;
      refresh(selection.snapshot());
    };
  }

  const restoreExternalUploads = wrapExternalAppendUploads(candidate, beginAppend);
  const originalOnGraphConfigured = candidate.onGraphConfigured;
  candidate.onGraphConfigured = function (...args: unknown[]): unknown {
    const configuredValue = imageWidget.value;
    const serializedValue = configuredWidgetValue(candidate, imageWidget);
    const result = originalOnGraphConfigured?.apply(this, args);
    const restored = selection.replace(
      Array.isArray(configuredValue)
        ? configuredValue
        : Array.isArray(serializedValue)
          ? serializedValue
          : imageWidget.value
    );
    if (!Array.isArray(imageWidget.value)) setPersistedFiles(restored);
    refresh(restored);
    return result;
  };

  const originalOnRemoved = candidate.onRemoved;
  candidate.onRemoved = function (...args: unknown[]): unknown {
    restoreExternalUploads();
    previewActions.dispose();
    preview.dispose();
    return originalOnRemoved?.apply(this, args);
  };

  const initial = selection.snapshot();
  if (!Array.isArray(imageWidget.value)) setPersistedFiles(initial);
  refresh(initial);
}

/** Read the persisted value aligned with one native widget when Comfy clears its live value. */
function configuredWidgetValue(
  node: OrderedMediaNode,
  widget: OrderedMediaWidget
): unknown {
  const index = node.widgets?.indexOf(widget) ?? -1;
  return index >= 0 ? node.widgets_values?.[index] : undefined;
}

/** Register one renderer-neutral ordered-media loader extension. */
export function registerOrderedMediaNode(
  app: ComfyApp,
  api: ComfyApi,
  extensionName: string,
  config: OrderedMediaNodeConfig,
  logger: Logger = console
): void {
  app.registerExtension({
    name: extensionName,
    nodeCreated(candidate: unknown) {
      try {
        configureOrderedMediaNode(candidate, app, api, config, logger);
      } catch (error: unknown) {
        const message = error instanceof Error ? error.message : String(error);
        logger.warn(
          `Could not configure ${config.labels.singular} loader: ${message}`,
          error
        );
        throw error;
      }
    }
  });
}

function nativeButton(
  node: OrderedMediaNode,
  name: string,
  label: string,
  callback: () => void
): OrderedMediaWidget {
  const widget = node.addWidget("button", name, "ordered_media", callback, {
    serialize: false,
    tooltip: label
  });
  widget.label = label;
  return widget;
}

function findWidget(
  node: OrderedMediaNode,
  name: string
): OrderedMediaWidget | undefined {
  return node.widgets?.find((widget) => widget.name === name);
}

function findNativeUploadWidget(
  node: OrderedMediaNode
): OrderedMediaWidget | undefined {
  return node.widgets?.find(
    (widget) =>
      widget.type === "button" &&
      widget.value === "image" &&
      widget.options?.serialize === false &&
      widget.options.canvasOnly === true
  );
}

function hideInternalWidget(widget: OrderedMediaWidget): void {
  widget.options ??= {};
  widget.options.hidden = true;
  widget.hidden = true;
  widget.computeSize = () => [0, -4];
}

function sameMediaFiles(left: string[], right: string[]): boolean {
  return (
    left.length === right.length &&
    left.every((file, index) => file === right[index])
  );
}

function wrapExternalAppendUploads(
  node: OrderedMediaNode,
  beginAppend: () => void
): () => void {
  const originalPasteFiles = node.pasteFiles;
  const originalOnDragDrop = node.onDragDrop;
  const wrappedPasteFiles: NativeCallback | undefined = originalPasteFiles
    ? (...args: unknown[]): unknown => {
        beginAppend();
        return originalPasteFiles.apply(node, args);
      }
    : undefined;
  const wrappedOnDragDrop: NativeCallback | undefined = originalOnDragDrop
    ? (...args: unknown[]): unknown => {
        beginAppend();
        return originalOnDragDrop.apply(node, args);
      }
    : undefined;
  if (wrappedPasteFiles) node.pasteFiles = wrappedPasteFiles;
  if (wrappedOnDragDrop) node.onDragDrop = wrappedOnDragDrop;
  return () => {
    if (node.pasteFiles === wrappedPasteFiles) {
      if (originalPasteFiles) node.pasteFiles = originalPasteFiles;
      else delete node.pasteFiles;
    }
    if (node.onDragDrop === wrappedOnDragDrop) {
      if (originalOnDragDrop) node.onDragDrop = originalOnDragDrop;
      else delete node.onDragDrop;
    }
  };
}

function isOrderedMediaNode(
  candidate: unknown,
  nodeId: string
): candidate is OrderedMediaNode {
  if (typeof candidate !== "object" || candidate === null) return false;
  const node = candidate as Partial<OrderedMediaNode>;
  return (
    node.constructor?.comfyClass === nodeId &&
    typeof node.addWidget === "function"
  );
}
