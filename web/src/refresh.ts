// SimpleSyrup - workflow-focused ComfyUI extensions for image generation
// Copyright (C) 2026  Artificial Sweetener and contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

import { refreshExternalLLMModels } from "./api";
import type { ComfyApp, Logger } from "./types";

const REFRESH_WRAPPED = Symbol.for("SimpleSyrup.ExternalLLM.RefreshWrapped");

interface RefreshOwner {
  [REFRESH_WRAPPED]?: boolean;
}

export interface ExternalLLMRefreshApi {
  refreshExternalLLMModels(): Promise<unknown>;
}

export function registerExternalLLMRefreshHook(
  app: ComfyApp,
  api: ExternalLLMRefreshApi = { refreshExternalLLMModels },
  logger: Logger = console
): void {
  if (!app.refreshComboInNodes) {
    return;
  }

  const refreshOwner = app as RefreshOwner;
  if (refreshOwner[REFRESH_WRAPPED]) {
    return;
  }

  const originalRefresh = app.refreshComboInNodes.bind(app);
  refreshOwner[REFRESH_WRAPPED] = true;
  app.refreshComboInNodes = async () => {
    try {
      await api.refreshExternalLLMModels();
    } catch (error) {
      logger.warn("Could not refresh SimpleSyrup external LLM models.", error);
    }
    await originalRefresh();
  };
}
