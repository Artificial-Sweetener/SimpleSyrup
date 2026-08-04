// SimpleSyrup - workflow-focused ComfyUI extensions for image generation
// Copyright (C) 2026  Artificial Sweetener and contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

import { inputImageReference } from "./comfyImageUrl";
import { registerOrderedMediaNode } from "./orderedMediaNode";
import type { ComfyApi, ComfyApp, Logger } from "./types";

/** Register Load Image List through the shared ordered-media interface. */
export function registerImageListUpload(
  app: ComfyApp,
  api: ComfyApi,
  logger: Logger = console
): void {
  registerOrderedMediaNode(
    app,
    api,
    "SimpleSyrup.LoadImageListUpload",
    {
      nodeId: "SimpleSyrup.LoadImageList",
      labels: {
        singular: "image",
        plural: "images",
        replace: "Replace images...",
        add: "Add images..."
      },
      preview: (files) =>
        Promise.resolve({
          images: files.map(inputImageReference),
          animated: files.map(() => false)
        })
    },
    logger
  );
}
