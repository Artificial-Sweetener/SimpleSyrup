// SimpleSyrup - workflow-focused ComfyUI extensions for image generation
// Copyright (C) 2026  Artificial Sweetener and contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

import { describe, expect, it, vi } from "vitest";

import { registerExternalLLMRefreshHook } from "../src/refresh";
import { createFakeComfyApp } from "./testUtils";

describe("external LLM refresh hook", () => {
  it("refreshes external models before Comfy refreshes node definitions", async () => {
    const app = createFakeComfyApp();
    const calls: string[] = [];
    app.refreshComboInNodes = vi.fn().mockImplementation(() => {
      calls.push("comfy");
      return Promise.resolve();
    });
    const api = {
      refreshExternalLLMModels: vi.fn().mockImplementation(() => {
        calls.push("external");
        return Promise.resolve();
      })
    };

    registerExternalLLMRefreshHook(app, api);
    await app.refreshComboInNodes();

    expect(calls).toEqual(["external", "comfy"]);
  });

  it("still runs the original refresh when external refresh fails", async () => {
    const app = createFakeComfyApp();
    const logger = { warn: vi.fn() };
    const originalRefresh = vi.fn().mockResolvedValue(undefined);
    app.refreshComboInNodes = originalRefresh;

    registerExternalLLMRefreshHook(
      app,
      { refreshExternalLLMModels: vi.fn().mockRejectedValue(new Error("offline")) },
      logger
    );
    await app.refreshComboInNodes();

    expect(originalRefresh).toHaveBeenCalledOnce();
    expect(logger.warn).toHaveBeenCalledWith(
      expect.stringContaining("Could not refresh"),
      expect.any(Error)
    );
  });

  it("wraps the refresh function only once", async () => {
    const app = createFakeComfyApp();
    const originalRefresh = vi.fn().mockResolvedValue(undefined);
    app.refreshComboInNodes = originalRefresh;
    const api = { refreshExternalLLMModels: vi.fn().mockResolvedValue(undefined) };

    registerExternalLLMRefreshHook(app, api);
    registerExternalLLMRefreshHook(app, api);
    await app.refreshComboInNodes();

    expect(api.refreshExternalLLMModels).toHaveBeenCalledOnce();
    expect(originalRefresh).toHaveBeenCalledOnce();
  });
});
