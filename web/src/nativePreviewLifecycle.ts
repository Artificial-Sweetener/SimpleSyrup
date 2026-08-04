// SimpleSyrup - workflow-focused ComfyUI extensions for image generation
// Copyright (C) 2026  Artificial Sweetener and contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

const listeners = new Set<() => void>();
let observer: MutationObserver | null = null;

/** Subscribe to native preview mounts and source replacements across renderers. */
export function subscribeNativePreviewLifecycle(
  listener: () => void
): () => void {
  listeners.add(listener);
  ensureObserver();
  return () => {
    listeners.delete(listener);
    if (listeners.size === 0) {
      observer?.disconnect();
      observer = null;
    }
  };
}

function ensureObserver(): void {
  if (observer) return;
  observer = new MutationObserver(() => {
    for (const listener of listeners) listener();
  });
  observer.observe(document.body, {
    childList: true,
    subtree: true,
    attributes: true,
    attributeFilter: ["src"]
  });
}
