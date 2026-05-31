// SimpleSyrup - workflow-focused ComfyUI extensions for image generation
// Copyright (C) 2026  Artificial Sweetener and contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

import { describe, expect, it, vi } from "vitest";

import {
  deleteExternalLLMApiKey,
  getExternalLLMSettings,
  getSettings,
  parseExternalLLMSettings,
  parseSettings,
  refreshExternalLLMModels,
  saveExternalLLMApiKey,
  saveExternalLLMSettings,
  saveSettings
} from "../src/api";
import type { FetchLike } from "../src/api";
import { createJsonResponse } from "./testUtils";

describe("settings API", () => {
  it("loads SimpleSyrup settings from the backend route", async () => {
    const fetchImpl = vi.fn<FetchLike>().mockResolvedValue(
      createJsonResponse({ show_downloadable_models: false })
    );

    await expect(getSettings(fetchImpl)).resolves.toEqual({
      show_downloadable_models: false
    });
    expect(fetchImpl).toHaveBeenCalledWith("/simple-syrup/settings");
  });

  it("saves SimpleSyrup settings to the backend route", async () => {
    const fetchImpl = vi.fn<FetchLike>().mockResolvedValue(
      createJsonResponse({ show_downloadable_models: true })
    );

    await expect(
      saveSettings({ show_downloadable_models: true }, fetchImpl)
    ).resolves.toEqual({ show_downloadable_models: true });
    expect(fetchImpl).toHaveBeenCalledWith(
      "/simple-syrup/settings",
      expect.objectContaining({
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ show_downloadable_models: true })
      })
    );
  });

  it("surfaces backend load errors when available", async () => {
    const fetchImpl = vi
      .fn<FetchLike>()
      .mockResolvedValue(createJsonResponse({ error: "nope" }, { status: 500 }));

    await expect(getSettings(fetchImpl)).rejects.toThrow("nope");
  });

  it("surfaces backend save errors when available", async () => {
    const fetchImpl = vi
      .fn<FetchLike>()
      .mockResolvedValue(createJsonResponse({ error: "nope" }, { status: 400 }));

    await expect(
      saveSettings({ show_downloadable_models: false }, fetchImpl)
    ).rejects.toThrow("nope");
  });

  it("rejects invalid response payloads conservatively", () => {
    expect(() => parseSettings({ show_downloadable_models: "false" })).toThrow(
      "SimpleSyrup settings payload is invalid"
    );
  });
});

describe("external LLM settings API", () => {
  const payload = {
    base_url: "https://provider.example/v1",
    cached_models: ["model-a"],
    default_model: "model-a",
    has_api_key: true
  };

  it("loads external LLM settings from the backend route", async () => {
    const fetchImpl = vi.fn<FetchLike>().mockResolvedValue(createJsonResponse(payload));

    await expect(getExternalLLMSettings(fetchImpl)).resolves.toEqual(payload);
    expect(fetchImpl).toHaveBeenCalledWith(
      "/simple-syrup/external-llm/settings"
    );
  });

  it("saves external LLM endpoint settings", async () => {
    const fetchImpl = vi.fn<FetchLike>().mockResolvedValue(createJsonResponse(payload));

    await expect(
      saveExternalLLMSettings(
        {
          base_url: "https://provider.example/v1",
          default_model: "model-a"
        },
        fetchImpl
      )
    ).resolves.toEqual(payload);
    expect(fetchImpl).toHaveBeenCalledWith(
      "/simple-syrup/external-llm/settings",
      expect.objectContaining({ method: "POST" })
    );
  });

  it("saves external LLM API keys without expecting the secret back", async () => {
    const fetchImpl = vi.fn<FetchLike>().mockResolvedValue(createJsonResponse(payload));

    await expect(
      saveExternalLLMApiKey({ api_key: "secret" }, fetchImpl)
    ).resolves.toEqual(payload);
    expect(fetchImpl).toHaveBeenCalledWith(
      "/simple-syrup/external-llm/api-key",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({ api_key: "secret" })
      })
    );
  });

  it("surfaces backend API key save errors", async () => {
    const fetchImpl = vi.fn<FetchLike>().mockResolvedValue(
      createJsonResponse(
        { error: "Configure an external LLM endpoint first." },
        { status: 400 }
      )
    );

    await expect(
      saveExternalLLMApiKey({ api_key: "secret" }, fetchImpl)
    ).rejects.toThrow("Configure an external LLM endpoint first.");
  });

  it("deletes external LLM API keys", async () => {
    const fetchImpl = vi.fn<FetchLike>().mockResolvedValue(
      createJsonResponse({ ...payload, has_api_key: false })
    );

    await expect(deleteExternalLLMApiKey(fetchImpl)).resolves.toMatchObject({
      has_api_key: false
    });
    expect(fetchImpl).toHaveBeenCalledWith(
      "/simple-syrup/external-llm/api-key",
      { method: "DELETE" }
    );
  });

  it("refreshes external LLM models", async () => {
    const fetchImpl = vi.fn<FetchLike>().mockResolvedValue(createJsonResponse(payload));

    await expect(refreshExternalLLMModels(fetchImpl)).resolves.toEqual(payload);
    expect(fetchImpl).toHaveBeenCalledWith(
      "/simple-syrup/external-llm/models/refresh",
      { method: "POST" }
    );
  });

  it("rejects malformed external LLM settings payloads", () => {
    expect(() =>
      parseExternalLLMSettings({
        base_url: "https://provider.example/v1",
        cached_models: [1],
        default_model: "model-a",
        has_api_key: true
      })
    ).toThrow("External LLM settings payload is invalid");
  });
});
