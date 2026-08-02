// SimpleSyrup - workflow-focused ComfyUI extensions for image generation
// Copyright (C) 2026  Artificial Sweetener and contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

export interface SelectionState<TId> {
  hovered: TId | null;
  selected: TId | null;
  candidates: readonly TId[];
  active: TId | null;
}

export type SelectionSubscriber<TId> = (state: SelectionState<TId>) => void;

/** Coordinate hover and pinned selection across inspector representations. */
export class SelectionModel<TId> {
  private hovered: TId | null = null;
  private selected: TId | null = null;
  private candidates: readonly TId[] = [];
  private readonly subscribers = new Set<SelectionSubscriber<TId>>();

  /** Subscribe to selection changes and receive the current state immediately. */
  subscribe(subscriber: SelectionSubscriber<TId>): () => void {
    this.subscribers.add(subscriber);
    subscriber(this.state());
    return () => this.subscribers.delete(subscriber);
  }

  /** Replace the hover target and all regions currently beneath the pointer. */
  hover(candidates: readonly TId[]): void {
    this.candidates = [...candidates];
    this.hovered = candidates[0] ?? null;
    this.publish();
  }

  /** Clear transient pointer state without changing the pinned region. */
  clearHover(): void {
    this.candidates = [];
    this.hovered = null;
    this.publish();
  }

  /** Pin one region, or clear the pinned selection with null. */
  select(id: TId | null): void {
    this.selected = id;
    this.publish();
  }

  /** Cycle through overlapping pointer candidates and pin the result. */
  selectNextCandidate(): TId | null {
    if (this.candidates.length === 0) return this.selected;
    const currentIndex =
      this.selected === null ? -1 : this.candidates.indexOf(this.selected);
    const next = this.candidates[(currentIndex + 1) % this.candidates.length] ?? null;
    this.select(next);
    return next;
  }

  /** Return immutable selection state for rendering or tests. */
  state(): SelectionState<TId> {
    return {
      hovered: this.hovered,
      selected: this.selected,
      candidates: [...this.candidates],
      active: this.hovered ?? this.selected
    };
  }

  private publish(): void {
    const state = this.state();
    for (const subscriber of this.subscribers) subscriber(state);
  }
}
