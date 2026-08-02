// SimpleSyrup - workflow-focused ComfyUI extensions for image generation
// Copyright (C) 2026  Artificial Sweetener and contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

import { describe, expect, it } from "vitest";

import {
  parseSegPreviewDocument,
  previewAssetUrl
} from "../src/segPreviewTypes";

describe("SEG preview transport", () => {
  it("validates execution payloads and uses the latest mapped document", () => {
    const document = validDocument();

    expect(
      parseSegPreviewDocument({ simple_syrup_segs_preview: [document] })
    ).toEqual(document);
    expect(parseSegPreviewDocument({ images: [] })).toBeUndefined();
    expect(() =>
      parseSegPreviewDocument({
        simple_syrup_segs_preview: [{ ...document, version: 2 }]
      })
    ).toThrow("unsupported preview payload");
  });

  it("builds an encoded view URL through Comfy's base-path resolver", () => {
    const url = previewAssetUrl(
      { filename: "one two.png", subfolder: "a/b", type: "temp" },
      (path) => `/comfy${path}`
    );

    expect(url).toBe(
      "/comfy/view?filename=one+two.png&subfolder=a%2Fb&type=temp"
    );
  });
});

function validDocument(): object {
  const image = { filename: "asset.png", subfolder: "", type: "temp" };
  return {
    version: 1,
    source: { width: 10, height: 8 },
    preview: { width: 10, height: 8, image },
    atlas: { width: 4, height: 4, image },
    regions: [
      {
        id: "seg-0001",
        index: 0,
        label: "subject",
        confidence: 0.9,
        area: 8,
        color: "#f24236",
        crop: { x: 1, y: 1, width: 4, height: 2 },
        atlas: { x: 0, y: 0, width: 4, height: 2 }
      }
    ]
  };
}
