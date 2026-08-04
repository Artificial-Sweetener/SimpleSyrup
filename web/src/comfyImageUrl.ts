// SimpleSyrup - workflow-focused ComfyUI extensions for image generation
// Copyright (C) 2026  Artificial Sweetener and contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

import type { ComfyImageResult } from "./types";

/** Build a Comfy view URL without assuming origin-root hosting. */
export function comfyImageUrl(
  reference: ComfyImageResult,
  apiURL: (path: string) => string = (path) => path
): string {
  const query = new URLSearchParams({
    filename: reference.filename,
    subfolder: reference.subfolder,
    type: reference.type
  });
  return apiURL(`/view?${query.toString()}`);
}
/** Convert one persisted input-folder path into Comfy's preview reference. */
export function inputImageReference(path: string): ComfyImageResult {
  const annotated = path.trim().replace(/\s+\[input\]$/, "");
  const normalized = annotated.replaceAll("\\", "/");
  const separator = normalized.lastIndexOf("/");
  return {
    filename: normalized.slice(separator + 1),
    subfolder: separator >= 0 ? normalized.slice(0, separator) : "",
    type: "input"
  };
}
