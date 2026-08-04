// SimpleSyrup - workflow-focused ComfyUI extensions for image generation
// Copyright (C) 2026  Artificial Sweetener and contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

export const MEDIA_MOVE_EARLIER_ATTRIBUTE = "data-ss-media-move-earlier";
export const MEDIA_MOVE_LATER_ATTRIBUTE = "data-ss-media-move-later";
export const MEDIA_REMOVE_ATTRIBUTE = "data-ss-media-remove";
export const MEDIA_INDEX_ATTRIBUTE = "data-ss-media-index";

export interface NativePreviewSlot {
  readonly left: number;
  readonly top: number;
  readonly width: number;
  readonly height: number;
}

export interface OrderedMediaPreviewAffordanceOptions {
  readonly itemLabel: string;
  readonly getSlots: () => NativePreviewSlot[];
  readonly moveEarlier: (index: number) => void;
  readonly moveLater: (index: number) => void;
  readonly remove: (index: number) => void;
}

interface AffordanceElements {
  readonly root: HTMLDivElement;
  readonly earlier: HTMLButtonElement;
  readonly later: HTMLButtonElement;
  readonly remove: HTMLButtonElement;
}

/** Overlay compact controls without replacing Comfy's native preview surface. */
export class OrderedMediaPreviewAffordances {
  private elements: AffordanceElements[] = [];
  private animationFrame: number | null = null;
  private remainingSyncFrames = 0;

  constructor(private readonly options: OrderedMediaPreviewAffordanceOptions) {
    window.addEventListener("resize", this.requestRefresh, true);
    document.addEventListener("click", this.requestRefresh, true);
    document.addEventListener("keydown", this.requestRefresh, true);
    document.addEventListener("pointermove", this.requestRefresh, true);
    document.addEventListener("pointerup", this.requestRefresh, true);
    document.addEventListener("wheel", this.requestRefresh, true);
    this.refresh();
  }

  /** Recheck native layout across several frames after Comfy rerenders output. */
  refresh(): void {
    this.remainingSyncFrames = Math.max(this.remainingSyncFrames, 4);
    this.scheduleSync();
  }

  /** Remove every overlay control and global layout listener. */
  dispose(): void {
    window.removeEventListener("resize", this.requestRefresh, true);
    document.removeEventListener("click", this.requestRefresh, true);
    document.removeEventListener("keydown", this.requestRefresh, true);
    document.removeEventListener("pointermove", this.requestRefresh, true);
    document.removeEventListener("pointerup", this.requestRefresh, true);
    document.removeEventListener("wheel", this.requestRefresh, true);
    if (this.animationFrame !== null) {
      cancelAnimationFrame(this.animationFrame);
      this.animationFrame = null;
    }
    for (const element of this.elements) element.root.remove();
    this.elements = [];
  }

  private readonly requestRefresh = (): void => {
    this.remainingSyncFrames = Math.max(this.remainingSyncFrames, 2);
    this.scheduleSync();
  };

  private scheduleSync(): void {
    if (this.animationFrame !== null) return;
    this.animationFrame = requestAnimationFrame(() => {
      this.animationFrame = null;
      this.sync();
      this.remainingSyncFrames -= 1;
      if (this.remainingSyncFrames > 0) this.scheduleSync();
    });
  }

  private sync(): void {
    const slots = this.options.getSlots();
    this.resizeElements(slots.length);
    for (const [index, elements] of this.elements.entries()) {
      const slot = slots[index];
      if (!slot || slot.width <= 0 || slot.height <= 0) {
        elements.root.hidden = true;
        continue;
      }
      elements.root.hidden = false;
      const stripHeight = Math.max(18, Math.min(26, slot.height * 0.16));
      Object.assign(elements.root.style, {
        left: `${String(slot.left)}px`,
        top: `${String(slot.top)}px`,
        width: `${String(slot.width)}px`,
        height: `${String(stripHeight)}px`
      });
      elements.root.dataset.ssMediaIndex = String(index);
      elements.earlier.dataset.ssMediaIndex = String(index);
      elements.later.dataset.ssMediaIndex = String(index);
      elements.remove.dataset.ssMediaIndex = String(index);
      setActionAvailability(elements.earlier, index > 0);
      setActionAvailability(
        elements.later,
        index < this.elements.length - 1
      );
      elements.earlier.setAttribute(
        "aria-label",
        `Move ${this.options.itemLabel} ${String(index + 1)} earlier`
      );
      elements.later.setAttribute(
        "aria-label",
        `Move ${this.options.itemLabel} ${String(index + 1)} later`
      );
      elements.remove.setAttribute(
        "aria-label",
        `Remove ${this.options.itemLabel} ${String(index + 1)}`
      );
    }
  }

  private resizeElements(count: number): void {
    while (this.elements.length > count) this.elements.pop()?.root.remove();
    while (this.elements.length < count) {
      this.elements.push(this.createElements());
    }
  }

  private createElements(): AffordanceElements {
    const root = document.createElement("div");
    root.className = "ss-native-preview-affordance";
    Object.assign(root.style, {
      position: "fixed",
      zIndex: "9990",
      display: "flex",
      alignItems: "stretch",
      pointerEvents: "none",
      overflow: "hidden",
      borderRadius: "4px 4px 0 0",
      background: "rgba(20, 20, 20, 0.72)",
      boxShadow: "inset 0 -1px 0 rgba(255, 255, 255, 0.16)"
    });
    const earlier = controlButton("pi-arrow-left", "Move earlier");
    earlier.setAttribute(MEDIA_MOVE_EARLIER_ATTRIBUTE, "true");
    const later = controlButton("pi-arrow-right", "Move later");
    later.setAttribute(MEDIA_MOVE_LATER_ATTRIBUTE, "true");
    const remove = controlButton("pi-times", "Remove");
    remove.setAttribute(MEDIA_REMOVE_ATTRIBUTE, "true");
    bindAction(earlier, (index) => {
      this.options.moveEarlier(index);
    });
    bindAction(later, (index) => {
      this.options.moveLater(index);
    });
    bindAction(remove, (index) => {
      this.options.remove(index);
    });
    root.append(earlier, later, remove);
    document.body.append(root);
    return { root, earlier, later, remove };
  }
}

/** Return the current positional index carried by one overlay control. */
export function mediaIndex(element: Element): number | null {
  const value = element.getAttribute(MEDIA_INDEX_ATTRIBUTE);
  if (value === null) return null;
  const index = Number(value);
  return Number.isInteger(index) && index >= 0 ? index : null;
}

function controlButton(iconClass: string, title: string): HTMLButtonElement {
  const button = document.createElement("button");
  button.type = "button";
  button.title = title;
  const icon = document.createElement("i");
  icon.className = `pi ${iconClass}`;
  icon.setAttribute("aria-hidden", "true");
  button.append(icon);
  Object.assign(button.style, {
    appearance: "none",
    border: "0",
    padding: "0",
    margin: "0",
    minWidth: "0",
    flex: "1 1 0",
    color: "rgba(255, 255, 255, 0.92)",
    background: "transparent",
    fontSize: "13px",
    cursor: "pointer",
    pointerEvents: "auto"
  });
  return button;
}

function bindAction(
  button: HTMLButtonElement,
  action: (index: number) => void
): void {
  button.addEventListener("pointerdown", stopControlPointerEvent);
  button.addEventListener("click", (event) => {
    event.preventDefault();
    event.stopPropagation();
    const index = mediaIndex(button);
    if (index !== null) action(index);
  });
}

function setActionAvailability(
  button: HTMLButtonElement,
  available: boolean
): void {
  button.disabled = !available;
  button.style.cursor = available ? "pointer" : "default";
  button.style.opacity = available ? "1" : "0.35";
}

function stopControlPointerEvent(event: PointerEvent): void {
  event.preventDefault();
  event.stopPropagation();
}
