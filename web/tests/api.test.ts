// SimpleSyrup - workflow-focused ComfyUI extensions for image generation
// Copyright (C) 2026  Artificial Sweetener and contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

import { describe, expect, it, vi } from "vitest";

import {
  clearQuantCache,
  deleteExternalLLMApiKey,
  getExternalLLMSettings,
  getMaskBatchPreview,
  getQuantCacheStatus,
  getSettings,
  enforceQuantCacheLimit,
  parseExternalLLMSettings,
  parseMaskBatchPreview,
  parseQuantCacheStatus,
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
      createJsonResponse({
        show_downloadable_models: false,
        quant_cache_limit_gib: 20
      })
    );

    await expect(getSettings(fetchImpl)).resolves.toEqual({
      show_downloadable_models: false,
      quant_cache_limit_gib: 20
    });
    expect(fetchImpl).toHaveBeenCalledWith("/simple-syrup/settings");
  });

  it("saves SimpleSyrup settings to the backend route", async () => {
    const fetchImpl = vi.fn<FetchLike>().mockResolvedValue(
      createJsonResponse({
        show_downloadable_models: true,
        quant_cache_limit_gib: 30
      })
    );

    await expect(
      saveSettings(
        { show_downloadable_models: true, quant_cache_limit_gib: 30 },
        fetchImpl
      )
    ).resolves.toEqual({
      show_downloadable_models: true,
      quant_cache_limit_gib: 30
    });
    expect(fetchImpl).toHaveBeenCalledWith(
      "/simple-syrup/settings",
      expect.objectContaining({
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          show_downloadable_models: true,
          quant_cache_limit_gib: 30
        })
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
      saveSettings(
        { show_downloadable_models: false, quant_cache_limit_gib: 20 },
        fetchImpl
      )
    ).rejects.toThrow("nope");
  });

  it("rejects invalid response payloads conservatively", () => {
    expect(() => parseSettings({ show_downloadable_models: "false" })).toThrow(
      "SimpleSyrup settings payload is invalid"
    );
  });

  it("loads and clears the global quant cache through its dedicated route", async () => {
    const status = {
      path: "models/SyrupQuants",
      usage_bytes: 1024,
      limit_bytes: 20 * 1024 ** 3,
      artifact_count: 1,
      active_artifact_count: 0
    };
    const fetchImpl = vi
      .fn<FetchLike>()
      .mockImplementation(() => Promise.resolve(createJsonResponse(status)));

    await expect(getQuantCacheStatus(fetchImpl)).resolves.toEqual(status);
    await expect(enforceQuantCacheLimit(fetchImpl)).resolves.toEqual(status);
    await expect(clearQuantCache(fetchImpl)).resolves.toEqual(status);
    expect(fetchImpl).toHaveBeenNthCalledWith(1, "/simple-syrup/quant-cache");
    expect(fetchImpl).toHaveBeenNthCalledWith(2, "/simple-syrup/quant-cache", {
      method: "POST"
    });
    expect(fetchImpl).toHaveBeenNthCalledWith(3, "/simple-syrup/quant-cache", {
      method: "DELETE"
    });
  });

  it("rejects malformed quant cache status payloads", () => {
    expect(() =>
      parseQuantCacheStatus({ path: "models/SyrupQuants" })
    ).toThrow("quant cache status is invalid");
  });
});

describe("mask batch preview API", () => {
  const preview = {
    images: [
      {
        filename: "ComfyUI_temp_mask.png",
        subfolder: "",
        type: "temp" as const
      }
    ],
    animated: [false]
  };

  it("requests previews for the ordered files and selected channel", async () => {
    const fetchImpl = vi
      .fn<FetchLike>()
      .mockResolvedValue(createJsonResponse(preview));

    await expect(
      getMaskBatchPreview(["right.png", "left.png"], "blue", fetchImpl)
    ).resolves.toEqual(preview);
    expect(fetchImpl).toHaveBeenCalledWith(
      "/simple-syrup/mask-batch/preview",
      expect.objectContaining({
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          files: ["right.png", "left.png"],
          channel: "blue"
        })
      })
    );
  });

  it("surfaces preview validation errors", async () => {
    const fetchImpl = vi.fn<FetchLike>().mockResolvedValue(
      createJsonResponse(
        { error: "mask dimensions do not match" },
        { status: 400 }
      )
    );

    await expect(
      getMaskBatchPreview(["right.png", "left.png"], "alpha", fetchImpl)
    ).rejects.toThrow("mask dimensions do not match");
  });

  it("rejects malformed native preview responses", () => {
    expect(() =>
      parseMaskBatchPreview({
        images: [{ filename: "mask.png", subfolder: "", type: "other" }],
        animated: [false]
      })
    ).toThrow("mask batch preview payload is invalid");
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
