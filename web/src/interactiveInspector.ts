// SimpleSyrup - workflow-focused ComfyUI extensions for image generation
// Copyright (C) 2026  Artificial Sweetener and contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

import type { Logger } from "./types";

export interface PreparedInspectorState {
  commit(): void;
  dispose(): void;
}

export interface AsyncInspectorView<TDocument> {
  readonly element: HTMLElement;
  setLoading(): void;
  prepare(document: TDocument): Promise<PreparedInspectorState>;
  showError(message: string): void;
  dispose(): void;
}

/** Publish only the latest asynchronously prepared execution document. */
export class InteractiveInspectorController<TDocument> {
  private version = 0;
  private disposed = false;
  private committed: PreparedInspectorState | undefined;

  constructor(
    private readonly view: AsyncInspectorView<TDocument>,
    private readonly parse: (output: unknown) => TDocument | undefined,
    private readonly logger: Logger = console
  ) {}

  /** Parse and asynchronously display one execution output. */
  update(output: unknown): void {
    if (this.disposed) return;
    const version = ++this.version;
    let document: TDocument | undefined;
    try {
      document = this.parse(output);
    } catch (error: unknown) {
      this.fail(error);
      return;
    }
    if (!document) return;
    this.view.setLoading();
    void this.view
      .prepare(document)
      .then((prepared) => {
        if (this.disposed || version !== this.version) {
          prepared.dispose();
          return;
        }
        this.committed?.dispose();
        this.committed = prepared;
        prepared.commit();
      })
      .catch((error: unknown) => {
        if (!this.disposed && version === this.version) this.fail(error);
      });
  }

  /** Release committed state and prevent pending work from publishing. */
  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.version += 1;
    this.committed?.dispose();
    this.committed = undefined;
    this.view.dispose();
  }

  private fail(error: unknown): void {
    const message = error instanceof Error ? error.message : String(error);
    this.logger.warn(`Could not display Simple Preview SEGS: ${message}`, error);
    this.view.showError(message);
  }
}
