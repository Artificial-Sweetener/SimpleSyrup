// SimpleSyrup - workflow-focused ComfyUI extensions for image generation
// Copyright (C) 2026  Artificial Sweetener and contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

import { describe, expect, it, vi } from "vitest";

import {
  InteractiveInspectorController,
  type AsyncInspectorView,
  type PreparedInspectorState
} from "../src/interactiveInspector";

describe("InteractiveInspectorController", () => {
  it("discards stale prepared documents and commits only the latest", async () => {
    const first = deferred<PreparedInspectorState>();
    const second = deferred<PreparedInspectorState>();
    const view = fakeView([first.promise, second.promise]);
    const controller = new InteractiveInspectorController(view, (output) =>
      typeof output === "string" ? output : undefined
    );
    const stale = preparedState();
    const latest = preparedState();

    controller.update("first");
    controller.update("second");
    second.resolve(latest);
    await vi.waitFor(() => {
      expect(latest.commitSpy).toHaveBeenCalledOnce();
    });
    first.resolve(stale);
    await vi.waitFor(() => {
      expect(stale.disposeSpy).toHaveBeenCalledOnce();
    });

    expect(stale.commitSpy).not.toHaveBeenCalled();
  });

  it("disposes committed state and blocks pending publication", async () => {
    const pending = deferred<PreparedInspectorState>();
    const view = fakeView([pending.promise]);
    const controller = new InteractiveInspectorController(view, () => "document");
    const prepared = preparedState();

    controller.update({});
    controller.dispose();
    pending.resolve(prepared);
    await vi.waitFor(() => {
      expect(prepared.disposeSpy).toHaveBeenCalledOnce();
    });

    expect(prepared.commitSpy).not.toHaveBeenCalled();
    expect(view.disposeSpy).toHaveBeenCalledOnce();
  });
});

interface TestInspectorView extends AsyncInspectorView<string> {
  disposeSpy: ReturnType<typeof vi.fn<() => void>>;
}

function fakeView(
  preparations: Array<Promise<PreparedInspectorState>>
): TestInspectorView {
  let index = 0;
  const disposeSpy = vi.fn<() => void>();
  return {
    element: document.createElement("div"),
    setLoading: vi.fn(),
    prepare: vi.fn(() => {
      const preparation = preparations[index];
      index += 1;
      if (!preparation) throw new Error("Missing test preparation.");
      return preparation;
    }),
    showError: vi.fn(),
    dispose: () => {
      disposeSpy();
    },
    disposeSpy
  };
}

interface TestPreparedState extends PreparedInspectorState {
  commitSpy: ReturnType<typeof vi.fn<() => void>>;
  disposeSpy: ReturnType<typeof vi.fn<() => void>>;
}

function preparedState(): TestPreparedState {
  const commitSpy = vi.fn<() => void>();
  const disposeSpy = vi.fn<() => void>();
  return {
    commit: () => {
      commitSpy();
    },
    dispose: () => {
      disposeSpy();
    },
    commitSpy,
    disposeSpy
  };
}

function deferred<T>(): {
  promise: Promise<T>;
  resolve: (value: T) => void;
} {
  let resolver: ((value: T) => void) | undefined;
  const promise = new Promise<T>((resolve) => {
    resolver = resolve;
  });
  return {
    promise,
    resolve(value: T): void {
      if (!resolver) throw new Error("Deferred resolver was not initialized.");
      resolver(value);
    }
  };
}
