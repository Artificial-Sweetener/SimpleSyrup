// SimpleSyrup - workflow-focused ComfyUI extensions for image generation
// Copyright (C) 2026  Artificial Sweetener and contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

import { describe, expect, it, vi } from "vitest";

import { NativeNodePreview } from "../src/nativeNodePreview";
import type { ComfyApi, ComfyApp } from "../src/types";

describe("NativeNodePreview", () => {
  it("publishes execution-shaped output through Comfy's native event path", () => {
    const app = { nodeOutputs: {} } as ComfyApp;
    const events = new EventTarget();
    const api = events as ComfyApi;
    const listener = vi.fn();
    events.addEventListener("executed", (event: Event) => {
      listener((event as CustomEvent).detail);
    });
    const preview = new NativeNodePreview(app, api, { id: 42 });
    const output = {
      images: [{ filename: "one.png", subfolder: "", type: "input" as const }]
    };

    preview.publish(output);

    expect(app.nodeOutputs?.["42"]).toBe(output);
    expect(listener).toHaveBeenCalledWith({
      node: "42",
      display_node: "42",
      output
    });
  });

  it("clears the same native output surface", () => {
    const app = { nodeOutputs: {} } as ComfyApp;
    const api = new EventTarget() as ComfyApi;
    const preview = new NativeNodePreview(app, api, { id: 42 });

    preview.clear();

    expect(app.nodeOutputs?.["42"]).toEqual({ images: [], animated: [] });
  });

  it("notifies renderer adapters after native output changes", () => {
    const app = { nodeOutputs: {} } as ComfyApp;
    const api = new EventTarget() as ComfyApi;
    const preview = new NativeNodePreview(app, api, { id: 42 });
    const listener = vi.fn();
    const unsubscribe = preview.subscribe(listener);

    preview.clear();
    unsubscribe();
    preview.clear();

    expect(listener).toHaveBeenCalledOnce();
  });

});
