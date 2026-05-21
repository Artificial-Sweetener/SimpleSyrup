// SimpleSyrup - workflow-focused ComfyUI extensions for image generation
// Copyright (C) 2026  Artificial Sweetener and contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

// @ts-expect-error ComfyUI serves this host module outside the extension package.
import { app } from "../../../scripts/app.js";

import { registerSimpleSyrupSettings } from "./settings";
import type { ComfyApp } from "./types";

const comfyApp = app as unknown as ComfyApp;

comfyApp.registerExtension({
  name: "SimpleSyrup.Settings",
  async setup(appInstance) {
    await registerSimpleSyrupSettings(appInstance);
  }
});
