// SimpleSyrup - workflow-focused ComfyUI extensions for image generation
// Copyright (C) 2026  Artificial Sweetener and contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

import { describe, expect, it, vi } from "vitest";

import { SelectionModel } from "../src/selectionModel";

describe("SelectionModel", () => {
  it("shares hover and pinned selection across inspector views", () => {
    const model = new SelectionModel<string>();
    const subscriber = vi.fn();
    model.subscribe(subscriber);

    model.hover(["small", "large"]);
    model.selectNextCandidate();
    model.clearHover();

    expect(model.state()).toEqual({
      hovered: null,
      selected: "small",
      candidates: [],
      active: "small"
    });
    expect(subscriber).toHaveBeenCalledTimes(4);
  });

  it("cycles through overlapping candidates from the pinned region", () => {
    const model = new SelectionModel<string>();
    model.hover(["small", "large"]);

    expect(model.selectNextCandidate()).toBe("small");
    expect(model.selectNextCandidate()).toBe("large");
    expect(model.selectNextCandidate()).toBe("small");
  });
});
