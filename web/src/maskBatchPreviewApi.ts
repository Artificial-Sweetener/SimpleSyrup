// SimpleSyrup - workflow-focused ComfyUI extensions for image generation
// Copyright (C) 2026  Artificial Sweetener and contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

/** Mask-batch preview transport and native preview payload validation. */

import type { ComfyImageResult, ComfyNodeExecutionOutput } from "./types";
import { backendErrorMessage, type FetchLike } from "./apiTransport";

const MASK_BATCH_PREVIEW_ROUTE = "/simple-syrup/mask-batch/preview";

export async function getMaskBatchPreview(
  files: string[],
  channel: string,
  fetchImpl: FetchLike = fetch
): Promise<ComfyNodeExecutionOutput> {
  const response = await fetchImpl(MASK_BATCH_PREVIEW_ROUTE, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ files, channel })
  });
  if (!response.ok) {
    throw new Error(
      await backendErrorMessage(
        response,
        `Could not render Load Mask Batch preview. Backend returned ${String(response.status)}.`
      )
    );
  }
  return parseMaskBatchPreview(await response.json());
}

export function parseMaskBatchPreview(
  payload: unknown
): ComfyNodeExecutionOutput {
  if (!isMaskBatchPreviewPayload(payload)) {
    throw new Error(
      "SimpleSyrup mask batch preview payload is invalid. Expected native images and animation flags."
    );
  }
  return {
    images: payload.images.map((image) => ({ ...image })),
    animated: [...payload.animated]
  };
}

function isMaskBatchPreviewPayload(
  payload: unknown
): payload is Required<ComfyNodeExecutionOutput> {
  if (typeof payload !== "object" || payload === null) return false;
  const candidate = payload as Partial<ComfyNodeExecutionOutput>;
  return (
    Array.isArray(candidate.images) &&
    candidate.images.every(isComfyImageResult) &&
    Array.isArray(candidate.animated) &&
    candidate.animated.every((value) => typeof value === "boolean")
  );
}

function isComfyImageResult(value: unknown): value is ComfyImageResult {
  if (typeof value !== "object" || value === null) return false;
  const candidate = value as Partial<ComfyImageResult>;
  return (
    typeof candidate.filename === "string" &&
    typeof candidate.subfolder === "string" &&
    (candidate.type === "input" ||
      candidate.type === "output" ||
      candidate.type === "temp")
  );
}
