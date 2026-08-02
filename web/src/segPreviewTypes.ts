// SimpleSyrup - workflow-focused ComfyUI extensions for image generation
// Copyright (C) 2026  Artificial Sweetener and contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

import type { ComfyImageResult, ComfyNodeExecutionOutput } from "./types";

export const SEG_PREVIEW_OUTPUT_KEY = "simple_syrup_segs_preview";

export interface Dimensions {
  width: number;
  height: number;
}

export interface Rectangle extends Dimensions {
  x: number;
  y: number;
}

export interface SegPreviewRegion {
  id: string;
  index: number;
  label: string;
  confidence: number;
  area: number;
  color: string;
  crop: Rectangle;
  atlas: Rectangle;
}

export interface SegPreviewDocument {
  version: 1;
  source: Dimensions;
  preview: Dimensions & { image: ComfyImageResult };
  atlas: Dimensions & { image: ComfyImageResult };
  regions: SegPreviewRegion[];
}

export interface SegPreviewExecutionOutput extends ComfyNodeExecutionOutput {
  [SEG_PREVIEW_OUTPUT_KEY]?: unknown;
}

/** Return the latest valid preview document from one execution payload. */
export function parseSegPreviewDocument(
  output: unknown
): SegPreviewDocument | undefined {
  if (!isRecord(output)) return undefined;
  const candidates = output[SEG_PREVIEW_OUTPUT_KEY];
  if (!Array.isArray(candidates) || candidates.length === 0) return undefined;
  const candidate: unknown = candidates[candidates.length - 1];
  if (!isRecord(candidate) || candidate.version !== 1) {
    throw new Error("Simple Preview SEGS received an unsupported preview payload.");
  }
  const source = parseDimensions(candidate.source, "source");
  const preview = parseAssetDimensions(candidate.preview, "preview");
  const atlas = parseAssetDimensions(candidate.atlas, "atlas");
  if (!Array.isArray(candidate.regions)) {
    throw new Error("Simple Preview SEGS regions must be a list.");
  }
  const regions = candidate.regions.map(parseRegion);
  return { version: 1, source, preview, atlas, regions };
}

/** Build a Comfy view URL without assuming it is hosted at the origin root. */
export function previewAssetUrl(
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

function parseRegion(value: unknown, index: number): SegPreviewRegion {
  if (!isRecord(value)) {
    throw new Error(
      `Simple Preview SEGS region ${String(index + 1)} must be an object.`
    );
  }
  const id = stringValue(value.id, "region id");
  const label = stringValue(value.label, "region label");
  const color = stringValue(value.color, "region color");
  if (!/^#[0-9a-f]{6}$/i.test(color)) {
    throw new Error(`Simple Preview SEGS region '${id}' has an invalid color.`);
  }
  return {
    id,
    index: integerValue(value.index, "region index", 0),
    label,
    confidence: numberValue(value.confidence, "region confidence", 0),
    area: integerValue(value.area, "region area", 0),
    color,
    crop: parseRectangle(value.crop, `region '${id}' crop`),
    atlas: parseRectangle(value.atlas, `region '${id}' atlas`)
  };
}

function parseAssetDimensions(
  value: unknown,
  name: string
): Dimensions & { image: ComfyImageResult } {
  if (!isRecord(value)) {
    throw new Error(`Simple Preview SEGS ${name} must be an object.`);
  }
  return {
    ...parseDimensions(value, name),
    image: parseImageReference(value.image, name)
  };
}

function parseDimensions(value: unknown, name: string): Dimensions {
  if (!isRecord(value)) {
    throw new Error(`Simple Preview SEGS ${name} dimensions must be an object.`);
  }
  return {
    width: integerValue(value.width, `${name} width`, 1),
    height: integerValue(value.height, `${name} height`, 1)
  };
}

function parseRectangle(value: unknown, name: string): Rectangle {
  if (!isRecord(value)) {
    throw new Error(`Simple Preview SEGS ${name} must be an object.`);
  }
  return {
    x: integerValue(value.x, `${name} x`, 0),
    y: integerValue(value.y, `${name} y`, 0),
    width: integerValue(value.width, `${name} width`, 1),
    height: integerValue(value.height, `${name} height`, 1)
  };
}

function parseImageReference(value: unknown, name: string): ComfyImageResult {
  if (!isRecord(value)) {
    throw new Error(`Simple Preview SEGS ${name} image reference is invalid.`);
  }
  const type = stringValue(value.type, `${name} image type`);
  if (type !== "input" && type !== "output" && type !== "temp") {
    throw new Error(`Simple Preview SEGS ${name} image type is invalid.`);
  }
  return {
    filename: stringValue(value.filename, `${name} image filename`),
    subfolder: stringValue(value.subfolder, `${name} image subfolder`),
    type
  };
}

function integerValue(
  value: unknown,
  name: string,
  minimum: number
): number {
  if (!Number.isInteger(value) || (value as number) < minimum) {
    throw new Error(
      `Simple Preview SEGS ${name} must be at least ${String(minimum)}.`
    );
  }
  return value as number;
}

function numberValue(value: unknown, name: string, minimum: number): number {
  if (typeof value !== "number" || !Number.isFinite(value) || value < minimum) {
    throw new Error(
      `Simple Preview SEGS ${name} must be at least ${String(minimum)}.`
    );
  }
  return value;
}

function stringValue(value: unknown, name: string): string {
  if (typeof value !== "string") {
    throw new Error(`Simple Preview SEGS ${name} must be text.`);
  }
  return value;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
