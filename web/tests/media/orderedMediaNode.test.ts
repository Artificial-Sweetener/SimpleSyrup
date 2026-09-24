// SimpleSyrup - workflow-focused ComfyUI extensions for image generation
// Copyright (C) 2026  Artificial Sweetener and contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

import { afterEach, describe, expect, it, vi } from "vitest";

import {
  configureOrderedMediaNode,
  registerOrderedMediaNode
} from "../../src/orderedMediaNode";
import type { ComfyExtension } from "../../src/types";
import {
  CONFIG,
  configured,
  createFixture,
  references,
  resetOrderedMediaFixtures,
  selectFiles,
  widget
} from "./orderedMediaNodeTestSupport";

afterEach(resetOrderedMediaFixtures);

describe("ordered-media node integration", () => {
  it.each(["Nodes 1.0", "Nodes 2.0"])(
    "publishes through Comfy's native preview under %s",
    async () => {
      const fixture = createFixture(["one.png"]);
      configureOrderedMediaNode(fixture.node, fixture.app, fixture.api, CONFIG);

      expect(fixture.imageWidget.hidden).toBe(true);
      expect(fixture.uploadWidget.hidden).toBe(true);
      expect(fixture.imageWidget.computeSize?.()).toEqual([0, -4]);
      expect(
        fixture.node.widgets.some((candidate) =>
          [
            "simple_syrup_selected_media",
            "simple_syrup_move_earlier",
            "simple_syrup_move_later",
            "simple_syrup_remove_media"
          ].includes(candidate.name)
        )
      ).toBe(false);
      expect(widget(fixture, "simple_syrup_replace_media").label).toBe(
        "Replace images..."
      );
      expect(widget(fixture, "simple_syrup_add_media").label).toBe(
        "Add images..."
      );
      expect("addDOMWidget" in fixture.node).toBe(false);
      await vi.waitFor(() => {
        expect(fixture.app.nodeOutputs?.["7"]?.images).toEqual(
          references("one.png")
        );
      });
      expect(fixture.executed).toHaveBeenCalledWith(
        expect.objectContaining({ node: "7", display_node: "7" })
      );
    }
  );

  it("appends native multi-upload results and preserves duplicates", async () => {
    const fixture = configured(["one.png", "same.png"]);

    widget(fixture, "simple_syrup_add_media").callback?.();
    fixture.imageWidget.value = "two.png";
    fixture.imageWidget.callback?.("two.png");
    fixture.imageWidget.value = ["two.png", "same.png"];
    fixture.imageWidget.callback?.(["two.png", "same.png"]);

    expect(fixture.nativeUpload).toHaveBeenCalledOnce();
    expect(fixture.imageWidget.value).toEqual([
      "one.png",
      "same.png",
      "two.png",
      "same.png"
    ]);
    await vi.waitFor(() => {
      expect(fixture.app.nodeOutputs?.["7"]?.images).toEqual(
        references("one.png", "same.png", "two.png", "same.png")
      );
    });
  });

  it("treats pasted and dropped files as append operations", () => {
    const fixture = configured(["existing.png"]);

    fixture.node.onDragDrop();
    selectFiles(fixture, ["dropped-a.png", "dropped-b.png"]);
    expect(fixture.imageWidget.value).toEqual([
      "existing.png",
      "dropped-a.png",
      "dropped-b.png"
    ]);

    fixture.node.pasteFiles();
    selectFiles(fixture, ["pasted.png"]);
    expect(fixture.imageWidget.value).toEqual([
      "existing.png",
      "dropped-a.png",
      "dropped-b.png",
      "pasted.png"
    ]);
  });

  it("normalizes scalar workflows and survives Nodes 2.0 reactive assignments", async () => {
    const fixture = configured("saved.png", true);

    const onGraphConfigured = fixture.node.onGraphConfigured;
    if (!onGraphConfigured) throw new Error("Expected graph configuration callback.");
    onGraphConfigured();

    expect(fixture.imageWidget.value).toEqual(["saved.png"]);
    await vi.waitFor(() => {
      expect(fixture.app.nodeOutputs?.["7"]?.images).toEqual(
        references("saved.png")
      );
    });
  });

  it("retains array-valued media when the legacy graph lifecycle clears the widget", async () => {
    const fixture = createFixture(["one.png", "two.png"]);
    fixture.node.onGraphConfigured = () => {
      fixture.imageWidget.value = undefined;
    };
    configureOrderedMediaNode(fixture.node, fixture.app, fixture.api, CONFIG);
    fixture.node.onGraphConfigured();

    expect(fixture.imageWidget.value).toEqual(["one.png", "two.png"]);
    await vi.waitFor(() => {
      expect(fixture.app.nodeOutputs?.["7"]?.images).toEqual(
        references("one.png", "two.png")
      );
    });
  });

  it("restores persisted widget values when Comfy clears the live media widget", async () => {
    const fixture = createFixture(undefined);
    fixture.node.widgets_values = [
      ["one.png", "two.png"],
      "image",
      "ordered_media",
      "ordered_media"
    ];
    configureOrderedMediaNode(fixture.node, fixture.app, fixture.api, CONFIG);
    fixture.imageWidget.value = undefined;

    fixture.node.onGraphConfigured?.();

    expect(fixture.imageWidget.value).toEqual(["one.png", "two.png"]);
    await vi.waitFor(() => {
      expect(fixture.app.nodeOutputs?.["7"]?.images).toEqual(
        references("one.png", "two.png")
      );
    });
  });

  it("restores native handlers and clears preview on node removal", () => {
    const fixture = createFixture(["one.png"]);
    const paste = fixture.node.pasteFiles;
    const drop = fixture.node.onDragDrop;
    configureOrderedMediaNode(fixture.node, fixture.app, fixture.api, CONFIG);

    fixture.node.onRemoved();

    expect(fixture.node.pasteFiles).toBe(paste);
    expect(fixture.node.onDragDrop).toBe(drop);
    expect(fixture.originalRemoved).toHaveBeenCalledOnce();
    expect(fixture.app.nodeOutputs?.["7"]?.images).toEqual([]);
  });

  it("registers a guarded extension and ignores unrelated nodes", async () => {
    const fixture = createFixture([]);
    let extension: ComfyExtension | undefined;
    fixture.app.registerExtension = (value: ComfyExtension) => {
      extension = value;
    };
    registerOrderedMediaNode(fixture.app, fixture.api, "test.extension", CONFIG);
    fixture.node.constructor.comfyClass = "Other.Node";

    await extension?.nodeCreated?.(fixture.node);

    expect(extension?.name).toBe("test.extension");
    expect(fixture.node.addWidget).not.toHaveBeenCalled();
  });
});
