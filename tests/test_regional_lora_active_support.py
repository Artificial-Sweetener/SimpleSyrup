# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Prove exact reusable sparse-support admission for regional LoRA."""

from __future__ import annotations

import torch

from simple_syrup.runtime.regional_lora.active_support import (
    RegionalLoraActiveSupportResolver,
)


def test_resolver_returns_union_indices_and_exact_fractional_values() -> None:
    """Retain every nonzero value across overlapping broadcast multipliers."""

    resolver = RegionalLoraActiveSupportResolver()
    first = torch.tensor([[0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    second = torch.tensor([[0.0, 0.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0]])

    support = resolver.resolve((first, second), leading_shape=(1, 8))
    repeated = resolver.resolve((first, second), leading_shape=(1, 8))

    assert support is not None
    assert repeated is support
    assert torch.equal(support.indices, torch.tensor([1, 2]))
    assert torch.equal(support.multiplier_values[0], torch.tensor([0.25, 0.0]))
    assert torch.equal(support.multiplier_values[1], torch.tensor([0.0, -0.5]))


def test_resolver_retains_support_only_when_sparse_projection_saves_rows() -> None:
    """Compact low-coverage rows and use dense projection at half coverage."""

    multiplier = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

    support = RegionalLoraActiveSupportResolver().resolve(
        (multiplier,),
        leading_shape=(1, 4),
    )

    assert support is not None
    assert torch.equal(support.indices, torch.tensor([0]))
    assert (
        RegionalLoraActiveSupportResolver().resolve(
            (torch.tensor([[1.0, 1.0, 0.0, 0.0]]),),
            leading_shape=(1, 4),
        )
        is None
    )
    assert (
        RegionalLoraActiveSupportResolver().resolve(
            (torch.ones((1, 4)),),
            leading_shape=(1, 4),
        )
        is None
    )


def test_resolver_retains_exact_empty_support() -> None:
    """Keep completely inactive adapters on the zero-work projection path."""

    support = RegionalLoraActiveSupportResolver().resolve(
        (torch.zeros((2, 4)),),
        leading_shape=(2, 4),
    )

    assert support is not None
    assert support.element_count == 8
    assert support.indices.numel() == 0
    assert support.multiplier_values[0].numel() == 0
