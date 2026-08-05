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

/** Bind one native preview footprint to its authoritative media position. */
export interface NativePreviewActionSlot {
  readonly itemIndex: number;
  readonly bounds: NativePreviewSlot;
  readonly container?: HTMLElement;
}

export interface OrderedMediaPreviewAffordanceOptions {
  readonly itemLabel: string;
  readonly getSlots: () => NativePreviewActionSlot[];
  readonly getItemCount: () => number;
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
  private readonly positionedContainers = new Map<HTMLElement, string>();
  private animationFrame: number | null = null;
  private remainingSyncFrames = 0;
  private viewportControlsSuspended = false;

  constructor(private readonly options: OrderedMediaPreviewAffordanceOptions) {
    window.addEventListener("resize", this.requestRefresh, true);
    document.addEventListener("click", this.requestRefresh, true);
    document.addEventListener("keydown", this.requestRefresh, true);
    document.addEventListener("pointerdown", this.suspendViewportControls, true);
    document.addEventListener("pointermove", this.requestRefresh, true);
    document.addEventListener("pointerup", this.resumeViewportControls, true);
    document.addEventListener("pointercancel", this.resumeViewportControls, true);
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
    document.removeEventListener("pointerdown", this.suspendViewportControls, true);
    document.removeEventListener("pointermove", this.requestRefresh, true);
    document.removeEventListener("pointerup", this.resumeViewportControls, true);
    document.removeEventListener("pointercancel", this.resumeViewportControls, true);
    document.removeEventListener("wheel", this.requestRefresh, true);
    if (this.animationFrame !== null) {
      cancelAnimationFrame(this.animationFrame);
      this.animationFrame = null;
    }
    for (const element of this.elements) element.root.remove();
    this.elements = [];
    this.releasePositionedContainers(new Set());
  }

  private readonly requestRefresh = (): void => {
    this.remainingSyncFrames = Math.max(this.remainingSyncFrames, 2);
    this.scheduleSync();
  };

  private readonly suspendViewportControls = (event: Event): void => {
    if (!(event.target instanceof HTMLCanvasElement)) return;
    this.viewportControlsSuspended = true;
    for (const elements of this.elements) {
      if (elements.root.parentElement === document.body) elements.root.hidden = true;
    }
  };

  private readonly resumeViewportControls = (): void => {
    if (!this.viewportControlsSuspended) return;
    this.viewportControlsSuspended = false;
    this.refresh();
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
    const activeContainers = new Set<HTMLElement>();
    this.resizeElements(slots.length);
    for (const [index, elements] of this.elements.entries()) {
      const actionSlot = slots[index];
      const slot = actionSlot?.bounds;
      if (
        !actionSlot ||
        !slot ||
        slot.width <= 0 ||
        slot.height <= 0 ||
        (!actionSlot.container && this.viewportControlsSuspended)
      ) {
        elements.root.hidden = true;
        continue;
      }
      const itemIndex = actionSlot.itemIndex;
      elements.root.hidden = false;
      const position = this.mount(elements.root, actionSlot, activeContainers);
      const stripHeight = Math.max(18, Math.min(26, position.height * 0.16));
      Object.assign(elements.root.style, {
        left: `${String(position.left)}px`,
        top: `${String(position.top)}px`,
        width: `${String(position.width)}px`,
        height: `${String(stripHeight)}px`
      });
      elements.root.dataset.ssMediaIndex = String(itemIndex);
      elements.earlier.dataset.ssMediaIndex = String(itemIndex);
      elements.later.dataset.ssMediaIndex = String(itemIndex);
      elements.remove.dataset.ssMediaIndex = String(itemIndex);
      setActionAvailability(elements.earlier, itemIndex > 0);
      setActionAvailability(
        elements.later,
        itemIndex < this.options.getItemCount() - 1
      );
      elements.earlier.setAttribute(
        "aria-label",
        `Move ${this.options.itemLabel} ${String(itemIndex + 1)} earlier`
      );
      elements.later.setAttribute(
        "aria-label",
        `Move ${this.options.itemLabel} ${String(itemIndex + 1)} later`
      );
      elements.remove.setAttribute(
        "aria-label",
        `Remove ${this.options.itemLabel} ${String(itemIndex + 1)}`
      );
    }
    this.releasePositionedContainers(activeContainers);
  }

  /** Mount one control strip in the preview surface that owns its geometry. */
  private mount(
    root: HTMLElement,
    actionSlot: NativePreviewActionSlot,
    activeContainers: Set<HTMLElement>
  ): NativePreviewSlot {
    const container = actionSlot.container;
    if (!container) {
      if (root.parentElement !== document.body) document.body.append(root);
      root.style.position = "fixed";
      root.style.zIndex = "2";
      return actionSlot.bounds;
    }
    activeContainers.add(container);
    this.positionContainer(container);
    if (root.parentElement !== container) container.append(root);
    root.style.position = "absolute";
    root.style.zIndex = "1";
    const containerRect = container.getBoundingClientRect();
    const scaleX = containerScale(containerRect.width, container.offsetWidth);
    const scaleY = containerScale(containerRect.height, container.offsetHeight);
    return {
      left:
        (actionSlot.bounds.left - containerRect.left) / scaleX -
        container.clientLeft +
        container.scrollLeft,
      top:
        (actionSlot.bounds.top - containerRect.top) / scaleY -
        container.clientTop +
        container.scrollTop,
      width: actionSlot.bounds.width / scaleX,
      height: actionSlot.bounds.height / scaleY
    };
  }

  /** Establish a local containing block without overriding authored positioning. */
  private positionContainer(container: HTMLElement): void {
    if (this.positionedContainers.has(container)) return;
    const position = getComputedStyle(container).position;
    if (position !== "" && position !== "static") return;
    this.positionedContainers.set(container, container.style.position);
    container.style.position = "relative";
  }

  /** Restore preview surfaces that no longer contain loader controls. */
  private releasePositionedContainers(activeContainers: Set<HTMLElement>): void {
    for (const [container, originalPosition] of this.positionedContainers) {
      if (activeContainers.has(container)) continue;
      container.style.position = originalPosition;
      this.positionedContainers.delete(container);
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
      zIndex: "2",
      display: "flex",
      alignItems: "stretch",
      pointerEvents: "none",
      overflow: "hidden",
      background: "rgba(20, 20, 20, 0.72)"
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

/** Return a finite rendered-to-local scale for one preview axis. */
function containerScale(renderedSize: number, localSize: number): number {
  if (renderedSize <= 0 || localSize <= 0) return 1;
  const scale = renderedSize / localSize;
  return Number.isFinite(scale) && scale > 0 ? scale : 1;
}
