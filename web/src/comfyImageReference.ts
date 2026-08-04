// SimpleSyrup - workflow-focused ComfyUI extensions for image generation
// Copyright (C) 2026  Artificial Sweetener and contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

import type { ComfyImageResult } from "./types";

/** Build the stable identity Comfy uses for an image result. */
export function comfyImageReferenceKey(reference: ComfyImageResult): string {
  return `${reference.type}\n${reference.subfolder}\n${reference.filename}`;
}

/** Read a Comfy image identity from its rendered URL. */
export function comfyImageSourceKey(sourceUrl: string): string {
  try {
    const url = new URL(sourceUrl, window.location.href);
    return `${url.searchParams.get("type") ?? ""}\n${
      url.searchParams.get("subfolder") ?? ""
    }\n${url.searchParams.get("filename") ?? ""}`;
  } catch {
    return "";
  }
}

/** Check that loaded native images exactly match an ordered Comfy result. */
export function loadedImagesMatchReferences(
  images: readonly HTMLImageElement[] | undefined,
  references: readonly ComfyImageResult[]
): boolean {
  return (
    images?.length === references.length &&
    images.every((image, index) => {
      const reference = references[index];
      return (
        reference !== undefined &&
        comfyImageSourceKey(image.currentSrc || image.src) ===
          comfyImageReferenceKey(reference)
      );
    })
  );
}
