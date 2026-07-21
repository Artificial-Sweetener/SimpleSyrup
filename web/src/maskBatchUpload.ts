// SimpleSyrup - workflow-focused ComfyUI extensions for image generation
// Copyright (C) 2026  Artificial Sweetener and contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

import {
  MaskBatchPreviewController,
  type MaskBatchPreviewClient
} from "./maskBatchPreview";
import type {
  ComfyApp,
  ComfyExecutionEvents,
  ComfyExtension,
  ComfyImageResult,
  Logger
} from "./types";

const LOAD_MASK_BATCH_NODE_ID = "SimpleSyrup.LoadMaskBatch";
const REPLACE_MASKS_LABEL = "Replace masks...";
const ADD_MASKS_LABEL = "Add masks...";
const REMOVE_MASK_LABEL = "Remove selected mask";
const EMPTY_MASK_SELECTION_LABEL = "No masks loaded";

type SelectionIntent = "append" | "replace";
type NativeCallback = (...args: unknown[]) => unknown;

interface MaskBatchWidget {
  name: string;
  value: unknown;
  type?: string;
  label?: string;
  callback?: (value?: unknown) => void;
  computeSize?: (width?: number) => [number, number];
  hidden?: boolean;
  disabled?: boolean;
  options?: {
    canvasOnly?: boolean;
    disabled?: boolean;
    serialize?: boolean;
    tooltip?: string;
    hidden?: boolean;
    values?: string[];
  };
}

interface MaskBatchNode {
  constructor: { comfyClass?: string };
  id?: string | number;
  widgets?: MaskBatchWidget[];
  imageIndex?: number | null;
  images?: ComfyImageResult[];
  imgs?: unknown[] | undefined;
  pasteFiles?: NativeCallback;
  onDragDrop?: NativeCallback;
  onRemoved?: NativeCallback;
  onGraphConfigured?: NativeCallback;
  onExecuted?: NativeCallback;
  onWidgetChanged?: (
    name: string,
    value: unknown,
    previousValue: unknown,
    widget: MaskBatchWidget
  ) => void;
  graph?: { setDirtyCanvas?: (foreground: boolean, background: boolean) => void };
  addWidget(
    type: "button" | "combo",
    name: string,
    value: string | undefined,
    callback: (value?: unknown) => void,
    options: MaskBatchWidget["options"]
  ): MaskBatchWidget;
}

/** Register native upload-list controls for the mask batch loader. */
export function registerMaskBatchUpload(
  app: ComfyApp,
  executionEvents: ComfyExecutionEvents,
  loadPreview?: MaskBatchPreviewClient,
  logger: Logger = console
): void {
  const extension: ComfyExtension = {
    name: "SimpleSyrup.LoadMaskBatchUpload",
    nodeCreated(node: unknown) {
      try {
        configureMaskBatchNode(node, executionEvents, loadPreview, logger, app);
      } catch (error: unknown) {
        logger.warn(
          `Could not configure Load Mask Batch native controls: ${errorMessage(error)}`,
          error
        );
        throw error;
      }
    }
  };
  app.registerExtension(extension);
}

/** Return a stable diagnostic message for an unknown frontend failure. */
function errorMessage(error: unknown): string {
  return error instanceof Error && error.message
    ? error.message
    : String(error);
}

/** Wire native Comfy upload, append, remove, persistence, and preview behavior. */
export function configureMaskBatchNode(
  candidate: unknown,
  executionEvents: ComfyExecutionEvents,
  loadPreview?: MaskBatchPreviewClient,
  logger: Logger = console,
  app?: ComfyApp
): void {
  if (!isMaskBatchNode(candidate)) return;

  const imageWidget = findWidget(candidate, "image");
  const channelWidget = findWidget(candidate, "channel");
  const uploadWidget = findNativeUploadWidget(candidate);
  if (!imageWidget || !channelWidget || !uploadWidget?.callback) return;

  hideInternalWidget(imageWidget);
  hideInternalWidget(uploadWidget);

  const preview = new MaskBatchPreviewController(
    executionEvents,
    candidate,
    loadPreview,
    logger,
    () => {
      clearNativePreview(candidate, app);
    }
  );
  let selectionIntent: SelectionIntent = "replace";
  let appendBase: string[] = [];
  let uploadPending = false;
  let programmaticSelectionUpdate = false;
  let ignoredUploadCallbackFiles: string[] | undefined;
  const nativeUploadCallback = uploadWidget.callback;

  const replaceWidget = candidate.addWidget(
    "button",
    "simple_syrup_replace_masks",
    "image",
    () => {
      selectionIntent = "replace";
      appendBase = [];
      uploadPending = true;
      nativeUploadCallback.call(uploadWidget);
    },
    nativeButtonOptions(
      "Choose one or more masks and replace the current ordered list."
    )
  );
  replaceWidget.label = REPLACE_MASKS_LABEL;

  const setSelectedMasks = (files: string[]): void => {
    programmaticSelectionUpdate = true;
    try {
      imageWidget.value = [...files];
    } finally {
      programmaticSelectionUpdate = false;
    }
  };

  uploadWidget.callback = (value?: unknown) => {
    selectionIntent = "replace";
    appendBase = [];
    uploadPending = true;
    nativeUploadCallback.call(uploadWidget, value);
  };

  const addWidget = candidate.addWidget(
    "button",
    "simple_syrup_add_masks",
    "image",
    () => {
      selectionIntent = "append";
      appendBase = selectedMaskFiles(imageWidget);
      uploadPending = true;
      nativeUploadCallback.call(uploadWidget);
    },
    nativeButtonOptions(
      "Upload one or more masks and append them after the current ordered list."
    )
  );
  addWidget.label = ADD_MASKS_LABEL;

  const selectedMaskWidget = candidate.addWidget(
    "combo",
    "simple_syrup_selected_mask",
    EMPTY_MASK_SELECTION_LABEL,
    (value?: unknown) => {
      const selectedIndex = selectedMaskIndex(selectedMaskWidget, value);
      candidate.imageIndex = selectedIndex;
      candidate.graph?.setDirtyCanvas?.(true, true);
    },
    {
      serialize: false,
      tooltip:
        "Select one loaded mask by its ordered position for preview or removal.",
      values: []
    }
  );
  selectedMaskWidget.label = "selected mask";

  const removeWidget = candidate.addWidget(
    "button",
    "simple_syrup_remove_mask",
    "image",
    () => {
      const previous = selectedMaskFiles(imageWidget);
      const index = activeMaskIndex(candidate, selectedMaskWidget, previous);
      if (index === null) return;
      const current = previous.toSpliced(index, 1);

      setSelectedMasks(current);
      updateMaskSelection(selectedMaskWidget, current, index);
      candidate.imageIndex = current.length === 0 ? null : Math.min(index, current.length - 1);
      notifySelectionChanged(candidate, imageWidget, { current, previous });
      updateRemoveAvailability(removeWidget, current.length);
      preview.refresh(current, selectedChannel(channelWidget));
    },
    nativeButtonOptions(
      "Open a mask in the preview gallery, then remove that position from the loaded list."
    )
  );
  removeWidget.label = REMOVE_MASK_LABEL;

  imageWidget.callback = (value?: unknown) => {
    if (programmaticSelectionUpdate) return;
    const callbackValue = value ?? imageWidget.value;
    if (uploadPending && !Array.isArray(callbackValue)) return;
    const incoming = normalizeMaskFiles(callbackValue);
    if (
      ignoredUploadCallbackFiles &&
      sameMaskFiles(incoming, ignoredUploadCallbackFiles)
    ) {
      ignoredUploadCallbackFiles = undefined;
      return;
    }
    ignoredUploadCallbackFiles = uploadPending ? [...incoming] : undefined;
    uploadPending = false;

    const current =
      selectionIntent === "append" ? [...appendBase, ...incoming] : incoming;
    const appended = selectionIntent === "append";
    selectionIntent = "replace";
    appendBase = [];

    if (appended) setSelectedMasks(current);
    candidate.imageIndex = null;
    updateMaskSelection(selectedMaskWidget, current);
    updateRemoveAvailability(removeWidget, current.length);
    preview.refresh(current, selectedChannel(channelWidget));
  };

  const originalChannelCallback = channelWidget.callback;
  channelWidget.callback = (value?: unknown) => {
    originalChannelCallback?.call(channelWidget, value);
    const files = selectedMaskFiles(imageWidget);
    if (files.length > 0) {
      preview.refresh(files, selectedChannel(channelWidget, value));
    }
  };

  resetIntentForExternalUploads(candidate, () => {
    selectionIntent = "replace";
    appendBase = [];
    uploadPending = false;
  });

  const originalOnGraphConfigured = candidate.onGraphConfigured;
  candidate.onGraphConfigured = function (...args: unknown[]): unknown {
    const result = originalOnGraphConfigured?.apply(this, args);
    const restored = selectedMaskFiles(imageWidget);
    if (!Array.isArray(imageWidget.value)) setSelectedMasks(restored);
    updateMaskSelection(selectedMaskWidget, restored);
    updateRemoveAvailability(removeWidget, restored.length);
    if (restored.length > 0) {
      preview.refresh(restored, selectedChannel(channelWidget));
    }
    return result;
  };

  const initial = selectedMaskFiles(imageWidget);
  if (!Array.isArray(imageWidget.value)) setSelectedMasks(initial);
  updateMaskSelection(selectedMaskWidget, initial);
  updateRemoveAvailability(removeWidget, initial.length);
  if (initial.length > 0) {
    preview.refresh(initial, selectedChannel(channelWidget));
  }
}

/** Keep serialized helper widgets out of both native node renderers. */
function hideInternalWidget(widget: MaskBatchWidget): void {
  widget.options ??= {};
  widget.options.hidden = true;
  widget.hidden = true;
  widget.computeSize = () => [0, -4];
}

/** Replace the native selector choices with ordered, duplicate-safe labels. */
function updateMaskSelection(
  widget: MaskBatchWidget,
  files: string[],
  preferredIndex = 0
): void {
  const values = files.map(
    (file, index) => `${String(index + 1)}. ${file}`
  );
  const hasMasks = values.length > 0;
  widget.options ??= {};
  widget.options.values = hasMasks ? values : [EMPTY_MASK_SELECTION_LABEL];
  widget.disabled = !hasMasks;
  widget.options.disabled = !hasMasks;
  widget.value = hasMasks
    ? values[Math.min(Math.max(preferredIndex, 0), values.length - 1)]
    : EMPTY_MASK_SELECTION_LABEL;
}

/** Resolve the selected ordered position from the native list widget. */
function selectedMaskIndex(
  widget: MaskBatchWidget,
  callbackValue?: unknown
): number | null {
  if (widget.disabled) return null;
  const value = callbackValue ?? widget.value;
  const values = widget.options?.values ?? [];
  const index = typeof value === "string" ? values.indexOf(value) : -1;
  return index >= 0 ? index : null;
}

/** Clear classic and Nodes 2.0 preview state before publishing new output. */
function clearNativePreview(node: MaskBatchNode, app?: ComfyApp): void {
  node.imgs = undefined;
  node.images = [];
  if (node.id !== undefined) delete app?.nodeOutputs?.[String(node.id)];
  node.graph?.setDirtyCanvas?.(true, true);
}

/** Prefer Comfy's active gallery image, then the native list selection. */
function activeMaskIndex(
  node: MaskBatchNode,
  selectedMaskWidget: MaskBatchWidget,
  files: string[]
): number | null {
  const galleryIndex = node.imageIndex;
  if (
    galleryIndex !== null &&
    galleryIndex !== undefined &&
    galleryIndex >= 0 &&
    galleryIndex < files.length
  ) {
    return galleryIndex;
  }
  const selectedIndex = selectedMaskIndex(selectedMaskWidget);
  return selectedIndex !== null && selectedIndex < files.length
    ? selectedIndex
    : null;
}

/** Return one native widget by its stable input name. */
function findWidget(
  node: MaskBatchNode,
  name: string
): MaskBatchWidget | undefined {
  return node.widgets?.find((widget) => widget.name === name);
}

/** Find Comfy's image upload button without depending on localized labels. */
function findNativeUploadWidget(
  node: MaskBatchNode
): MaskBatchWidget | undefined {
  return node.widgets?.find(
    (widget) =>
      widget.type === "button" &&
      widget.value === "image" &&
      widget.options?.serialize === false &&
      widget.options.canvasOnly === true
  );
}

/** Return options for a native button rendered by classic and Nodes 2.0. */
function nativeButtonOptions(tooltip: string): MaskBatchWidget["options"] {
  return { serialize: false, tooltip };
}

/** Return the newly selected native channel when it is usable. */
function selectedChannel(
  channelWidget: MaskBatchWidget,
  callbackValue?: unknown
): string | undefined {
  const value = callbackValue ?? channelWidget.value;
  return typeof value === "string" && value.length > 0 ? value : undefined;
}

/** Return the native multi-value widget as an ordered file list. */
function selectedMaskFiles(widget: MaskBatchWidget): string[] {
  return normalizeMaskFiles(widget.value);
}

/** Narrow native scalar or multi-value widget state to valid file paths. */
function normalizeMaskFiles(value: unknown): string[] {
  const values = Array.isArray(value) ? value : [value];
  return values.filter(
    (item): item is string => typeof item === "string" && item.length > 0
  );
}

/** Compare ordered upload results without relying on reactive proxy identity. */
function sameMaskFiles(left: string[], right: string[]): boolean {
  return (
    left.length === right.length &&
    left.every((file, index) => file === right[index])
  );
}

/** Disable removal only when the ordered mask list is empty. */
function updateRemoveAvailability(widget: MaskBatchWidget, count: number): void {
  const disabled = count === 0;
  widget.disabled = disabled;
  widget.options ??= {};
  widget.options.disabled = disabled;
}

/** Notify Comfy when a non-upload control mutates persisted widget state. */
function notifySelectionChanged(
  node: MaskBatchNode,
  imageWidget: MaskBatchWidget,
  change: { current: string[]; previous: string[] }
): void {
  node.onWidgetChanged?.(
    imageWidget.name,
    [...change.current],
    [...change.previous],
    imageWidget
  );
  node.graph?.setDirtyCanvas?.(true, true);
}

/** Prevent a cancelled Add action from affecting later paste or drop uploads. */
function resetIntentForExternalUploads(
  node: MaskBatchNode,
  reset: () => void
): void {
  const originalPasteFiles = node.pasteFiles;
  const originalOnDragDrop = node.onDragDrop;
  const originalOnRemoved = node.onRemoved;

  const wrappedPasteFiles: NativeCallback | undefined = originalPasteFiles
    ? (...args: unknown[]): unknown => {
        reset();
        return originalPasteFiles.apply(node, args);
      }
    : undefined;
  const wrappedOnDragDrop: NativeCallback | undefined = originalOnDragDrop
    ? (...args: unknown[]): unknown => {
        reset();
        return originalOnDragDrop.apply(node, args);
      }
    : undefined;

  if (wrappedPasteFiles) node.pasteFiles = wrappedPasteFiles;
  if (wrappedOnDragDrop) node.onDragDrop = wrappedOnDragDrop;
  node.onRemoved = function (...args: unknown[]): unknown {
    if (node.pasteFiles === wrappedPasteFiles) {
      if (originalPasteFiles) node.pasteFiles = originalPasteFiles;
      else delete node.pasteFiles;
    }
    if (node.onDragDrop === wrappedOnDragDrop) {
      if (originalOnDragDrop) node.onDragDrop = originalOnDragDrop;
      else delete node.onDragDrop;
    }
    return originalOnRemoved?.apply(this, args);
  };
}

/** Return true when the candidate is the native-widget batch mask loader. */
function isMaskBatchNode(candidate: unknown): candidate is MaskBatchNode {
  if (typeof candidate !== "object" || candidate === null) return false;
  const node = candidate as Partial<MaskBatchNode>;
  return (
    node.constructor?.comfyClass === LOAD_MASK_BATCH_NODE_ID &&
    typeof node.addWidget === "function"
  );
}

export type { MaskBatchPreviewClient } from "./maskBatchPreview";
