// SimpleSyrup - workflow-focused ComfyUI extensions for image generation
// Copyright (C) 2026  Artificial Sweetener and contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

/** General SimpleSyrup settings transport and payload validation. */

import { backendErrorMessage, type FetchLike } from "./apiTransport";

export interface SimpleSyrupSettings {
  show_downloadable_models: boolean;
  quant_cache_limit_gib: number;
}

const SETTINGS_ROUTE = "/simple-syrup/settings";

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
