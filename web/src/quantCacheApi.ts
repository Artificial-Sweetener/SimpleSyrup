// SimpleSyrup - workflow-focused ComfyUI extensions for image generation
// Copyright (C) 2026  Artificial Sweetener and contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

/** Quantized-model cache transport and payload validation. */

import { backendErrorMessage, type FetchLike } from "./apiTransport";

export interface QuantCacheStatus {
  path: string;
  usage_bytes: number;
  limit_bytes: number;
  artifact_count: number;
  active_artifact_count: number;
  removed_artifacts?: number;
  removed_bytes?: number;
}

const QUANT_CACHE_ROUTE = "/simple-syrup/quant-cache";

export async function getQuantCacheStatus(
  fetchImpl: FetchLike = fetch
): Promise<QuantCacheStatus> {
  return requestQuantCache(fetchImpl);
}

export async function clearQuantCache(
  fetchImpl: FetchLike = fetch
): Promise<QuantCacheStatus> {
  return requestQuantCache(fetchImpl, "DELETE");
}

export async function enforceQuantCacheLimit(
  fetchImpl: FetchLike = fetch
): Promise<QuantCacheStatus> {
  return requestQuantCache(fetchImpl, "POST");
}

export function parseQuantCacheStatus(payload: unknown): QuantCacheStatus {
  if (!isQuantCacheStatusPayload(payload)) {
    throw new Error(
      "SimpleSyrup quant cache status is invalid. Expected path, byte usage, limit, and artifact counts."
    );
  }
  return { ...payload };
}

async function requestQuantCache(
  fetchImpl: FetchLike,
  method?: "POST" | "DELETE"
): Promise<QuantCacheStatus> {
  const response =
    method === undefined
      ? await fetchImpl(QUANT_CACHE_ROUTE)
      : await fetchImpl(QUANT_CACHE_ROUTE, { method });
  if (!response.ok) {
    const fallback =
      method === "DELETE"
        ? `Could not clear quant cache. Backend returned ${String(response.status)}.`
        : method === "POST"
          ? `Could not enforce quant cache limit. Backend returned ${String(response.status)}.`
          : `Could not load quant cache status. Backend returned ${String(response.status)}.`;
    throw new Error(
      await backendErrorMessage(response, fallback)
    );
  }
  return parseQuantCacheStatus(await response.json());
}

function isQuantCacheStatusPayload(
  payload: unknown
): payload is QuantCacheStatus {
  if (typeof payload !== "object" || payload === null) return false;
  const candidate = payload as Partial<QuantCacheStatus>;
  return (
    typeof candidate.path === "string" &&
    isNonNegativeInteger(candidate.usage_bytes) &&
    isNonNegativeInteger(candidate.limit_bytes) &&
    isNonNegativeInteger(candidate.artifact_count) &&
    isNonNegativeInteger(candidate.active_artifact_count) &&
    (candidate.removed_artifacts === undefined ||
      isNonNegativeInteger(candidate.removed_artifacts)) &&
    (candidate.removed_bytes === undefined ||
      isNonNegativeInteger(candidate.removed_bytes))
  );
}

function isNonNegativeInteger(value: unknown): value is number {
  return typeof value === "number" && Number.isInteger(value) && value >= 0;
}
