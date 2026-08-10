// SimpleSyrup - workflow-focused ComfyUI extensions for image generation
// Copyright (C) 2026  Artificial Sweetener and contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

export function installSimpleSyrupSettingsStyle(): void {
  if (document.getElementById("simple-syrup-settings-style")) return;

  const style = document.createElement("style");
  style.id = "simple-syrup-settings-style";
  style.textContent = `
    .simple-syrup-settings-row {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      min-width: min(38rem, 100%);
    }
    .simple-syrup-settings-row[data-pending="true"] {
      opacity: 0.75;
    }
    .simple-syrup-settings-input {
      min-width: 16rem;
      flex: 1 1 auto;
    }
    .simple-syrup-settings-button {
      flex: 0 0 auto;
      white-space: nowrap;
    }
    .simple-syrup-settings-status {
      color: var(--fg-color);
      opacity: 0.8;
      white-space: normal;
      overflow-wrap: anywhere;
    }
    .simple-syrup-dialog-backdrop {
      position: fixed;
      inset: 0;
      z-index: 2147483647;
      display: flex;
      align-items: center;
      justify-content: center;
      background: rgb(0 0 0 / 45%);
    }
    .simple-syrup-dialog {
      display: grid;
      gap: 0.75rem;
      min-width: min(28rem, calc(100vw - 2rem));
      padding: 1rem;
      background: var(--comfy-menu-bg);
      color: var(--fg-color);
    }
    .simple-syrup-dialog-title {
      margin: 0;
      font-size: 1rem;
    }
    .simple-syrup-dialog-actions {
      display: flex;
      justify-content: flex-end;
      gap: 0.5rem;
    }
  `;
  document.head.appendChild(style);
}

export function createElement<TTag extends keyof HTMLElementTagNameMap>(
  tagName: TTag,
  className: string
): HTMLElementTagNameMap[TTag] {
  const element = document.createElement(tagName);
  element.className = className;
  return element;
}

export function setPending(element: HTMLElement, pending: boolean): void {
  element.dataset.pending = pending ? "true" : "false";
  for (const control of Array.from(element.querySelectorAll("input, button"))) {
    if (
      control instanceof HTMLInputElement ||
      control instanceof HTMLButtonElement
    ) {
      control.disabled = pending;
    }
  }
}
