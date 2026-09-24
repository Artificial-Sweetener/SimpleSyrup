// SimpleSyrup - workflow-focused ComfyUI extensions for image generation
// Copyright (C) 2026  Artificial Sweetener and contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

/** External LLM settings, credential, and model-refresh transport. */

import { backendErrorMessage, type FetchLike } from "./apiTransport";

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

const SETTINGS_ROUTE = "/simple-syrup/external-llm/settings";
const API_KEY_ROUTE = "/simple-syrup/external-llm/api-key";
const MODELS_REFRESH_ROUTE = "/simple-syrup/external-llm/models/refresh";

export async function getExternalLLMSettings(
  fetchImpl: FetchLike = fetch
): Promise<ExternalLLMSettings> {
  return requestSettings(
    fetchImpl,
    SETTINGS_ROUTE,
    undefined,
    "Could not load external LLM settings"
  );
}

export async function saveExternalLLMSettings(
  settings: ExternalLLMSettingsUpdate,
  fetchImpl: FetchLike = fetch
): Promise<ExternalLLMSettings> {
  return requestSettings(
    fetchImpl,
    SETTINGS_ROUTE,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(settings)
    },
    "Could not save external LLM settings"
  );
}

export async function saveExternalLLMApiKey(
  payload: ExternalLLMApiKeyUpdate,
  fetchImpl: FetchLike = fetch
): Promise<ExternalLLMSettings> {
  return requestSettings(
    fetchImpl,
    API_KEY_ROUTE,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    },
    "Could not save external LLM API key"
  );
}

export async function deleteExternalLLMApiKey(
  fetchImpl: FetchLike = fetch
): Promise<ExternalLLMSettings> {
  return requestSettings(
    fetchImpl,
    API_KEY_ROUTE,
    { method: "DELETE" },
    "Could not delete external LLM API key"
  );
}

export async function refreshExternalLLMModels(
  fetchImpl: FetchLike = fetch
): Promise<ExternalLLMSettings> {
  return requestSettings(
    fetchImpl,
    MODELS_REFRESH_ROUTE,
    { method: "POST" },
    "Could not refresh external LLM models"
  );
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

async function requestSettings(
  fetchImpl: FetchLike,
  route: string,
  init: RequestInit | undefined,
  errorPrefix: string
): Promise<ExternalLLMSettings> {
  const response =
    init === undefined ? await fetchImpl(route) : await fetchImpl(route, init);
  if (!response.ok) {
    throw new Error(
      await backendErrorMessage(
        response,
        `${errorPrefix}. Backend returned ${String(response.status)}.`
      )
    );
  }
  return parseExternalLLMSettings(await response.json());
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
    typeof (payload as Partial<ExternalLLMSettings>).default_model === "string" &&
    typeof (payload as Partial<ExternalLLMSettings>).has_api_key === "boolean"
  );
}
