# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Test layer-rotating sampler cadence for attention capture."""

from __future__ import annotations

import torch

from simple_syrup.runtime.attention_region_cadence import AttentionCaptureCadence


class _Resolver:
    """Record exact progress resolutions through an injectable boundary."""

    def __init__(self) -> None:
        """Initialize an empty call log."""

        self.calls: list[tuple[torch.Tensor, object]] = []

    def resolve(self, sample_sigmas: torch.Tensor, current_sigma: object) -> float:
        """Record one request and return a stable progress value."""

        self.calls.append((sample_sigmas, current_sigma))
        return 0.5


def test_rotates_selected_layer_when_the_layer_sequence_restarts() -> None:
    """Select a different layer offset on each denoising step."""

    cadence = AttentionCaptureCadence(3)

    first = tuple(cadence.sample(layer) for layer in ("a", "b", "c"))
    second = tuple(cadence.sample(layer) for layer in ("a", "b", "c"))

    assert first == (0, None, None)
    assert second == (None, 1, None)


def test_resolves_inference_sigma_only_once_for_one_selected_step() -> None:
    """Avoid repeated GPU synchronization without tensor version counters."""

    resolver = _Resolver()
    cadence = AttentionCaptureCadence(1, resolver)
    with torch.inference_mode():
        options = {
            "sample_sigmas": torch.tensor([1.0, 0.5, 0.0]),
            "sigmas": torch.tensor([0.5]),
        }
    step = cadence.sample("a")
    assert step == 0

    values = tuple(cadence.progress(step, options) for _index in range(20))

    assert values == (0.5,) * 20
    assert len(resolver.calls) == 1


def test_resolves_again_after_the_layer_sequence_starts_a_new_step() -> None:
    """Keep progress exact when Comfy reuses inference tensors across steps."""

    resolver = _Resolver()
    cadence = AttentionCaptureCadence(1, resolver)
    options = {
        "sample_sigmas": torch.tensor([1.0, 0.5, 0.0]),
        "sigmas": torch.tensor([0.5]),
    }
    first_step = cadence.sample("a")
    cadence.sample("b")
    second_step = cadence.sample("a")
    assert first_step == 0
    assert second_step == 1

    cadence.progress(first_step, options)
    cadence.progress(second_step, options)

    assert len(resolver.calls) == 2
