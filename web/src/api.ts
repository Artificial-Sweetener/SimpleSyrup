// SimpleSyrup - workflow-focused ComfyUI extensions for image generation
// Copyright (C) 2026  Artificial Sweetener and contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

export interface SimpleSyrupSettings {
  show_downloadable_models: boolean;
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
const EXTERNAL_LLM_SETTINGS_ROUTE = "/simple-syrup/external-llm/settings";
const EXTERNAL_LLM_API_KEY_ROUTE = "/simple-syrup/external-llm/api-key";
const EXTERNAL_LLM_MODELS_REFRESH_ROUTE =
  "/simple-syrup/external-llm/models/refresh";

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
    show_downloadable_models: payload.show_downloadable_models
  };
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
      "boolean"
  );
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
