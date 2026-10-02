// SimpleSyrup - workflow-focused ComfyUI extensions for image generation
// Copyright (C) 2026  Artificial Sweetener and contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

import type { ComfyApp } from "./types";

interface SamplerSocket {
  name: string;
  link?: string | number | null;
}

interface SamplerInputNode {
  constructor: { comfyClass?: string };
  inputs: SamplerSocket[];
  graph?: {
    links: Record<string | number, { target_slot: number } | undefined>;
    setDirtyCanvas?: (foreground: boolean, background: boolean) => void;
  };
  onGraphConfigured?: (...args: unknown[]) => unknown;
}

const SOCKET_ORDER = [
  "model", "positive", "negative", "latent_image", "options", "segs", "region_masks"
];

/** Keep optional negatives beside positives without changing backend validation. */
export function registerSamplerSocketOrder(app: ComfyApp): void {
  app.registerExtension({
    name: "SimpleSyrup.SamplerSocketOrder",
    nodeCreated(candidate: unknown) {
      if (!isSampler(candidate)) return;
      orderSockets(candidate);
      const configured = candidate.onGraphConfigured;
      candidate.onGraphConfigured = function (...args: unknown[]): unknown {
        const result = configured?.apply(this, args);
        orderSockets(this);
        return result;
      };
    }
  });
}

/** Preserve socket objects and repair saved links after the host restores a graph. */
function orderSockets(node: SamplerInputNode): void {
  const ranks = new Map(SOCKET_ORDER.map((name, index) => [name, index]));
  node.inputs.sort((left, right) =>
    (ranks.get(left.name) ?? SOCKET_ORDER.length) -
    (ranks.get(right.name) ?? SOCKET_ORDER.length)
  );
  for (const [index, input] of node.inputs.entries()) {
    if (input.link == null) continue;
    const link = node.graph?.links[input.link];
    if (link) link.target_slot = index;
  }
  node.graph?.setDirtyCanvas?.(true, true);
}

/** Narrow the host boundary to the one sampler whose socket order we own. */
function isSampler(value: unknown): value is SamplerInputNode {
  if (typeof value !== "object" || value === null) return false;
  const node = value as Partial<SamplerInputNode>;
  return node.constructor?.comfyClass === "SimpleSyrup.KSampler" &&
    Array.isArray(node.inputs) &&
    node.inputs.every((input) => typeof input.name === "string");
}
