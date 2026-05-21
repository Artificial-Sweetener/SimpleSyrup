// SimpleSyrup - workflow-focused ComfyUI extensions for image generation
// Copyright (C) 2026  Artificial Sweetener and contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

import { describe, expect, it, vi } from "vitest";

import { getSettings, parseSettings, saveSettings } from "../src/api";
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

  it("rejects non-OK load responses with an actionable error", async () => {
    const fetchImpl = vi
      .fn<FetchLike>()
      .mockResolvedValue(createJsonResponse({ error: "nope" }, { status: 500 }));

    await expect(getSettings(fetchImpl)).rejects.toThrow(
      "Could not load SimpleSyrup settings"
    );
  });

  it("rejects non-OK save responses with an actionable error", async () => {
    const fetchImpl = vi
      .fn<FetchLike>()
      .mockResolvedValue(createJsonResponse({ error: "nope" }, { status: 400 }));

    await expect(
      saveSettings({ show_downloadable_models: false }, fetchImpl)
    ).rejects.toThrow("Could not save SimpleSyrup settings");
  });

  it("rejects invalid response payloads conservatively", () => {
    expect(() => parseSettings({ show_downloadable_models: "false" })).toThrow(
      "SimpleSyrup settings payload is invalid"
    );
  });
});
