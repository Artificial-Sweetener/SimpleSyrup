// SimpleSyrup - workflow-focused ComfyUI extensions for image generation
// Copyright (C) 2026  Artificial Sweetener and contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

import { describe, expect, it } from "vitest";

import {
  normalizeMediaFiles,
  OrderedMediaSelection
} from "../src/orderedMediaSelection";

describe("OrderedMediaSelection", () => {
  it("normalizes scalar and invalid persisted values", () => {
    expect(normalizeMediaFiles("one.png")).toEqual(["one.png"]);
    expect(normalizeMediaFiles(["one.png", "", 4, "two.png"])).toEqual([
      "one.png",
      "two.png"
    ]);
  });

  it("preserves duplicate positions through append, move, and remove", () => {
    const selection = new OrderedMediaSelection(["same.png", "middle.png"]);

    expect(selection.append(["same.png"])).toEqual([
      "same.png",
      "middle.png",
      "same.png"
    ]);
    expect(selection.move(2, 0)).toEqual([
      "same.png",
      "same.png",
      "middle.png"
    ]);
    expect(selection.remove(1)).toEqual(["same.png", "middle.png"]);
  });

  it("ignores invalid positional edits and returns defensive snapshots", () => {
    const selection = new OrderedMediaSelection(["one.png", "two.png"]);
    const snapshot = selection.move(-1, 7);
    snapshot.splice(0, 2);

    expect(selection.snapshot()).toEqual(["one.png", "two.png"]);
  });
});
