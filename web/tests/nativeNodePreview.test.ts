// SimpleSyrup - workflow-focused ComfyUI extensions for image generation
// Copyright (C) 2026  Artificial Sweetener and contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

import { describe, expect, it, vi } from "vitest";

import { NativeNodePreview } from "../src/nativeNodePreview";
import type {
  ComfyApi,
  ComfyApp,
  ComfyGraph,
  ComfySetting,
  ComfySettingDefinition,
  SettingValue
} from "../src/types";

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

  it("publishes subgraph previews under the locator consumed by Nodes 2.0", () => {
    const subgraph = { id: "cube-definition" };
    const rootGraph = {
      id: "root",
      isRootGraph: true,
      nodes: [{ id: "cube-instance", subgraph }]
    };
    const app = previewApp(rootGraph);
    const events = new EventTarget();
    const api = events as ComfyApi;
    const listener = vi.fn();
    events.addEventListener("executed", (event: Event) => {
      listener((event as CustomEvent).detail);
    });
    const preview = new NativeNodePreview(app, api, {
      id: "mask-loader",
      graph: subgraph
    });
    const output = {
      images: [{ filename: "mask.png", subfolder: "", type: "temp" as const }]
    };

    preview.publish(output);

    expect(app.nodeOutputs?.["mask-loader"]).toBe(output);
    expect(app.nodeOutputs?.["cube-definition:mask-loader"]).toBe(output);
    expect(listener).toHaveBeenCalledWith({
      node: "cube-instance:mask-loader",
      display_node: "cube-instance:mask-loader",
      output
    });
  });

  it("publishes shared subgraph previews for every instance path", () => {
    const subgraph = { id: "cube-definition", nodes: [] };
    const nestedGraph = {
      id: "nested-definition",
      nodes: [{ id: "nested-cube", subgraph }]
    };
    const rootGraph = {
      id: "root",
      isRootGraph: true,
      nodes: [
        { id: "cube-a", subgraph },
        { id: "cube-b", subgraph },
        { id: "container", subgraph: nestedGraph }
      ]
    };
    const app = previewApp(rootGraph);
    const events = new EventTarget();
    const api = events as ComfyApi;
    const executionIds: string[] = [];
    const listener = vi.fn((executionId: string) => {
      executionIds.push(executionId);
    });
    events.addEventListener("executed", (event: Event) => {
      listener(executedNodeId(event));
    });
    const preview = new NativeNodePreview(app, api, {
      id: "mask-loader",
      graph: subgraph
    });

    preview.clear();

    expect(executionIds).toEqual([
      "cube-a:mask-loader",
      "cube-b:mask-loader",
      "container:nested-cube:mask-loader"
    ]);
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

function executedNodeId(event: Event): string {
  const detail: unknown = (event as CustomEvent<unknown>).detail;
  if (
    typeof detail !== "object" ||
    detail === null ||
    !("node" in detail) ||
    typeof detail.node !== "string"
  ) {
    throw new TypeError("Executed event did not include a node identity.");
  }
  return detail.node;
}

function previewApp(rootGraph: ComfyGraph): ComfyApp {
  return {
    nodeOutputs: {},
    rootGraph,
    ui: {
      settings: {
        addSetting<TValue extends SettingValue>(
          definition: ComfySettingDefinition<TValue>
        ): ComfySetting<TValue> {
          return { value: definition.defaultValue };
        }
      }
    },
    registerExtension: () => undefined
  };
}
