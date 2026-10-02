// SimpleSyrup - workflow-focused ComfyUI extensions for image generation
// Copyright (C) 2026  Artificial Sweetener and contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

import { describe, expect, it, vi } from "vitest";

import { registerSamplerSocketOrder } from "../../src/samplerSocketOrder";
import { createFakeComfyApp } from "../support/testUtils";

const correctOrder = [
  "model", "positive", "negative", "latent_image", "options", "segs", "region_masks"
];

/** Represent the host's required-first socket layout and saved connection slots. */
function fakeSampler() {
  return {
    id: 10,
    constructor: { comfyClass: "SimpleSyrup.KSampler" },
    inputs: [
      { name: "model", link: 1 },
      { name: "positive", link: 2 },
      { name: "latent_image", link: 3 },
      { name: "negative", link: 4 },
      { name: "options", link: null },
      { name: "segs", link: 5 },
      { name: "region_masks", link: 6 }
    ],
    widgets: [{ name: "seed", value: 17 }, { name: "denoise", value: 0.5 }],
    graph: {
      links: {
        1: { target_id: 10, target_slot: 0 },
        2: { target_id: 10, target_slot: 1 },
        3: { target_id: 10, target_slot: 2 },
        4: { target_id: 10, target_slot: 3 },
        5: { target_id: 10, target_slot: 5 },
        6: { target_id: 10, target_slot: 6 }
      },
      setDirtyCanvas: vi.fn()
    },
    onGraphConfigured: vi.fn(() => "preserved-result")
  };
}

describe("sampler socket ordering", () => {
  it("orders a new sampler without changing its inputs or widget values", async () => {
    const app = createFakeComfyApp();
    registerSamplerSocketOrder(app);
    const node = fakeSampler();
    const originalInputs = [...node.inputs];
    const originalArray = node.inputs;
    const widgets = [...node.widgets];
    await app.extensions[0]?.nodeCreated?.(node);
    expect(node.inputs.map((input) => input.name)).toEqual(correctOrder);
    expect(node.inputs).toBe(originalArray);
    expect(node.inputs).toEqual([
      originalInputs[0], originalInputs[1], originalInputs[3],
      originalInputs[2], originalInputs[4], originalInputs[5], originalInputs[6]
    ]);
    expect(node.widgets).toEqual(widgets);
    expect(node.graph.links[3].target_slot).toBe(3);
    expect(node.graph.links[4].target_slot).toBe(2);
    expect(node.inputs[5]?.link).toBe(5);
    expect(node.graph.links[5].target_slot).toBe(5);
    expect(node.inputs[6]?.link).toBe(6);
    expect(node.graph.links[6].target_slot).toBe(6);
  });

  it("repairs old workflow link indices by socket name after configuration", async () => {
    const app = createFakeComfyApp();
    registerSamplerSocketOrder(app);
    const node = fakeSampler();
    const originalCallback = node.onGraphConfigured;
    await app.extensions[0]?.nodeCreated?.(node);
    // Comfy restores the old saved sockets and link indices when loading a workflow.
    const restored = fakeSampler().inputs;
    node.inputs = restored;
    node.graph.links[3].target_slot = 2;
    node.graph.links[4].target_slot = 3;
    expect(node.onGraphConfigured()).toBe("preserved-result");
    expect(originalCallback).toHaveBeenCalledOnce();
    expect(node.inputs.map((input) => input.name)).toEqual(correctOrder);
    expect(node.inputs[2]?.link).toBe(4);
    expect(node.inputs[3]?.link).toBe(3);
    expect(node.graph.links[3].target_slot).toBe(3);
    expect(node.graph.links[4].target_slot).toBe(2);
    // Already ordered serialized sockets still need their graph link slots repaired.
    node.graph.links[3].target_slot = 2;
    node.onGraphConfigured();
    expect(node.graph.links[3].target_slot).toBe(3);
  });

  it("preserves added widget sockets and works without links or an existing callback", async () => {
    const app = createFakeComfyApp();
    registerSamplerSocketOrder(app);
    const node = {
      constructor: { comfyClass: "SimpleSyrup.KSampler" },
      inputs: [
        { name: "model", link: null }, { name: "seed", link: null },
        { name: "positive", link: null }, { name: "latent_image", link: null },
        { name: "negative", link: null }, { name: "options", link: null },
        { name: "segs", link: null },
        { name: "region_masks", link: null },
        { name: "denoise", link: null }
      ],
      onGraphConfigured: undefined as (() => unknown) | undefined
    };
    await app.extensions[0]?.nodeCreated?.(node);
    expect(node.inputs.map((input) => input.name)).toEqual([
      ...correctOrder, "seed", "denoise"
    ]);
    node.onGraphConfigured?.();
  });

  it.each<{ node: unknown }>([
    { node: null }, { node: {} },
    { node: { constructor: { comfyClass: "Other.KSampler" }, inputs: [] } }
  ])(
    "does not alter unrelated or malformed nodes: $node", async ({ node }) => {
      const app = createFakeComfyApp();
      registerSamplerSocketOrder(app);
      const snapshot = JSON.stringify(node);
      await app.extensions[0]?.nodeCreated?.(node);
      expect(JSON.stringify(node)).toBe(snapshot);
    }
  );
});
