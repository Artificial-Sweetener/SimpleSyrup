// SimpleSyrup - workflow-focused ComfyUI extensions for image generation
// Copyright (C) 2026  Artificial Sweetener and contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

import { describe, expect, it } from "vitest";

import { inputImageReference } from "../src/comfyImageUrl";
import { registerImageListUpload } from "../src/imageListUpload";
import type { ComfyApp, ComfyExtension } from "../src/types";

describe("Load Image List frontend integration", () => {
  it("registers its native ordered-preview extension", () => {
    let extension: ComfyExtension | undefined;
    const app = {
      registerExtension(value: ComfyExtension) {
        extension = value;
      }
    } as ComfyApp;
    registerImageListUpload(app, new EventTarget());

    expect(extension?.name).toBe("SimpleSyrup.LoadImageListUpload");
  });

  it("maps nested and annotated input paths to native preview references", () => {
    expect(inputImageReference("references\\subject.png [input]")).toEqual({
      filename: "subject.png",
      subfolder: "references",
      type: "input"
    });
  });
});
