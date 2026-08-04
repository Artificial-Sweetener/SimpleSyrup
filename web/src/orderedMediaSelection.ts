// SimpleSyrup - workflow-focused ComfyUI extensions for image generation
// Copyright (C) 2026  Artificial Sweetener and contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

/** Own an ordered, duplicate-preserving collection of uploaded file references. */
export class OrderedMediaSelection {
  private files: string[];

  constructor(value: unknown = []) {
    this.files = normalizeMediaFiles(value);
  }

  /** Return a defensive snapshot in authored order. */
  snapshot(): string[] {
    return [...this.files];
  }

  /** Replace every position with a normalized persisted widget value. */
  replace(value: unknown): string[] {
    this.files = normalizeMediaFiles(value);
    return this.snapshot();
  }

  /** Append every incoming position without deduplicating filenames. */
  append(value: unknown): string[] {
    this.files.push(...normalizeMediaFiles(value));
    return this.snapshot();
  }

  /** Remove one exact position when it exists. */
  remove(index: number): string[] {
    if (validIndex(index, this.files.length)) this.files.splice(index, 1);
    return this.snapshot();
  }

  /** Move one exact position and retain all duplicate filenames. */
  move(from: number, to: number): string[] {
    if (!validIndex(from, this.files.length) || !validIndex(to, this.files.length)) {
      return this.snapshot();
    }
    const [moved] = this.files.splice(from, 1);
    if (moved !== undefined) this.files.splice(to, 0, moved);
    return this.snapshot();
  }
}
/** Narrow scalar or multiselect widget state to valid ordered file paths. */
export function normalizeMediaFiles(value: unknown): string[] {
  const values = Array.isArray(value) ? value : [value];
  return values.filter(
    (item): item is string => typeof item === "string" && item.length > 0
  );
}

function validIndex(index: number, length: number): boolean {
  return Number.isInteger(index) && index >= 0 && index < length;
}
