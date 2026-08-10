// SimpleSyrup - workflow-focused ComfyUI extensions for image generation
// Copyright (C) 2026  Artificial Sweetener and contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

import type { ComfyImageResult, ComfyNodeExecutionOutput } from "./types";

export interface SimpleSyrupSettings {
  show_downloadable_models: boolean;
  quant_cache_limit_gib: number;
}

export interface QuantCacheStatus {
  path: string;
  usage_bytes: number;
  limit_bytes: number;
  artifact_count: number;
  active_artifact_count: number;
  removed_artifacts?: number;
  removed_bytes?: number;
}

export interface ExternalLLMSettings {
  base_url: string;
  cached_models: string[];
  default_model: string;
  has_api_key: boolean;
}

export interface ExternalLLMSettingsUpdate {
  base_url: string;
  default_model: string;
}

export interface ExternalLLMApiKeyUpdate {
  api_key: string;
}

export type FetchLike = (
  input: RequestInfo | URL,
  init?: RequestInit
) => Promise<Response>;

const SETTINGS_ROUTE = "/simple-syrup/settings";
const QUANT_CACHE_ROUTE = "/simple-syrup/quant-cache";
const EXTERNAL_LLM_SETTINGS_ROUTE = "/simple-syrup/external-llm/settings";
const EXTERNAL_LLM_API_KEY_ROUTE = "/simple-syrup/external-llm/api-key";
const EXTERNAL_LLM_MODELS_REFRESH_ROUTE =
  "/simple-syrup/external-llm/models/refresh";
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

export async function getSettings(
  fetchImpl: FetchLike = fetch
): Promise<SimpleSyrupSettings> {
  const response = await fetchImpl(SETTINGS_ROUTE);
  if (!response.ok) {
    throw new Error(
      await backendErrorMessage(
        response,
        `Could not load SimpleSyrup settings. Backend returned ${String(response.status)}.`
      )
    );
  }
  return parseSettings(await response.json());
}

export async function saveSettings(
  settings: SimpleSyrupSettings,
  fetchImpl: FetchLike = fetch
): Promise<SimpleSyrupSettings> {
  const response = await fetchImpl(SETTINGS_ROUTE, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(settings)
  });
  if (!response.ok) {
    throw new Error(
      await backendErrorMessage(
        response,
        `Could not save SimpleSyrup settings. Backend returned ${String(response.status)}.`
      )
    );
  }
  return parseSettings(await response.json());
}

export function parseSettings(payload: unknown): SimpleSyrupSettings {
  if (!isSettingsPayload(payload)) {
    throw new Error(
      "SimpleSyrup settings payload is invalid. Expected show_downloadable_models to be a boolean."
    );
  }
  return {
    show_downloadable_models: payload.show_downloadable_models,
    quant_cache_limit_gib: payload.quant_cache_limit_gib
  };
}

export async function getQuantCacheStatus(
  fetchImpl: FetchLike = fetch
): Promise<QuantCacheStatus> {
  const response = await fetchImpl(QUANT_CACHE_ROUTE);
  if (!response.ok) {
    throw new Error(
      await backendErrorMessage(
        response,
        `Could not load quant cache status. Backend returned ${String(response.status)}.`
      )
    );
  }
  return parseQuantCacheStatus(await response.json());
}

export async function clearQuantCache(
  fetchImpl: FetchLike = fetch
): Promise<QuantCacheStatus> {
  const response = await fetchImpl(QUANT_CACHE_ROUTE, { method: "DELETE" });
  if (!response.ok) {
    throw new Error(
      await backendErrorMessage(
        response,
        `Could not clear quant cache. Backend returned ${String(response.status)}.`
      )
    );
  }
  return parseQuantCacheStatus(await response.json());
}

export async function enforceQuantCacheLimit(
  fetchImpl: FetchLike = fetch
): Promise<QuantCacheStatus> {
  const response = await fetchImpl(QUANT_CACHE_ROUTE, { method: "POST" });
  if (!response.ok) {
    throw new Error(
      await backendErrorMessage(
        response,
        `Could not enforce quant cache limit. Backend returned ${String(response.status)}.`
      )
    );
  }
  return parseQuantCacheStatus(await response.json());
}

export function parseQuantCacheStatus(payload: unknown): QuantCacheStatus {
  if (!isQuantCacheStatusPayload(payload)) {
    throw new Error(
      "SimpleSyrup quant cache status is invalid. Expected path, byte usage, limit, and artifact counts."
    );
  }
  return { ...payload };
}

export async function getExternalLLMSettings(
  fetchImpl: FetchLike = fetch
): Promise<ExternalLLMSettings> {
  const response = await fetchImpl(EXTERNAL_LLM_SETTINGS_ROUTE);
  if (!response.ok) {
    throw new Error(
      await backendErrorMessage(
        response,
        `Could not load external LLM settings. Backend returned ${String(response.status)}.`
      )
    );
  }
  return parseExternalLLMSettings(await response.json());
}

export async function saveExternalLLMSettings(
  settings: ExternalLLMSettingsUpdate,
  fetchImpl: FetchLike = fetch
): Promise<ExternalLLMSettings> {
  const response = await fetchImpl(EXTERNAL_LLM_SETTINGS_ROUTE, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(settings)
  });
  if (!response.ok) {
    throw new Error(
      await backendErrorMessage(
        response,
        `Could not save external LLM settings. Backend returned ${String(response.status)}.`
      )
    );
  }
  return parseExternalLLMSettings(await response.json());
}

export async function saveExternalLLMApiKey(
  payload: ExternalLLMApiKeyUpdate,
  fetchImpl: FetchLike = fetch
): Promise<ExternalLLMSettings> {
  const response = await fetchImpl(EXTERNAL_LLM_API_KEY_ROUTE, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  if (!response.ok) {
    throw new Error(
      await backendErrorMessage(
        response,
        `Could not save external LLM API key. Backend returned ${String(response.status)}.`
      )
    );
  }
  return parseExternalLLMSettings(await response.json());
}

export async function deleteExternalLLMApiKey(
  fetchImpl: FetchLike = fetch
): Promise<ExternalLLMSettings> {
  const response = await fetchImpl(EXTERNAL_LLM_API_KEY_ROUTE, {
    method: "DELETE"
  });
  if (!response.ok) {
    throw new Error(
      await backendErrorMessage(
        response,
        `Could not delete external LLM API key. Backend returned ${String(response.status)}.`
      )
    );
  }
  return parseExternalLLMSettings(await response.json());
}

export async function refreshExternalLLMModels(
  fetchImpl: FetchLike = fetch
): Promise<ExternalLLMSettings> {
  const response = await fetchImpl(EXTERNAL_LLM_MODELS_REFRESH_ROUTE, {
    method: "POST"
  });
  if (!response.ok) {
    throw new Error(
      await backendErrorMessage(
        response,
        `Could not refresh external LLM models. Backend returned ${String(response.status)}.`
      )
    );
  }
  return parseExternalLLMSettings(await response.json());
}

export function parseExternalLLMSettings(
  payload: unknown
): ExternalLLMSettings {
  if (!isExternalLLMSettingsPayload(payload)) {
    throw new Error(
      "External LLM settings payload is invalid. Expected base_url, cached_models, default_model, and has_api_key."
    );
  }
  return {
    base_url: payload.base_url,
    cached_models: [...payload.cached_models],
    default_model: payload.default_model,
    has_api_key: payload.has_api_key
  };
}

function isSettingsPayload(payload: unknown): payload is SimpleSyrupSettings {
  return (
    typeof payload === "object" &&
    payload !== null &&
    typeof (payload as Partial<SimpleSyrupSettings>).show_downloadable_models ===
      "boolean" &&
    Number.isInteger(
      (payload as Partial<SimpleSyrupSettings>).quant_cache_limit_gib
    ) &&
    Number((payload as Partial<SimpleSyrupSettings>).quant_cache_limit_gib) > 0
  );
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

function isExternalLLMSettingsPayload(
  payload: unknown
): payload is ExternalLLMSettings {
  return (
    typeof payload === "object" &&
    payload !== null &&
    typeof (payload as Partial<ExternalLLMSettings>).base_url === "string" &&
    Array.isArray((payload as Partial<ExternalLLMSettings>).cached_models) &&
    (payload as Partial<ExternalLLMSettings>).cached_models?.every(
      (model) => typeof model === "string"
    ) === true &&
    typeof (payload as Partial<ExternalLLMSettings>).default_model ===
      "string" &&
    typeof (payload as Partial<ExternalLLMSettings>).has_api_key === "boolean"
  );
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

async function backendErrorMessage(
  response: Response,
  fallback: string
): Promise<string> {
  try {
    const payload: unknown = await response.json();
    if (
      typeof payload === "object" &&
      payload !== null &&
      typeof (payload as { error?: unknown }).error === "string"
    ) {
      return (payload as { error: string }).error;
    }
  } catch {
    return fallback;
  }
  return fallback;
}
