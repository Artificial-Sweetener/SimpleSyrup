# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify deterministic P10.1 contexts, masks, and tensor inputs."""

from __future__ import annotations

import torch

from tools.anima_regional_lora_performance.matrix_fixture import (
    build_matrix_attention_fixture,
    build_matrix_input_fixture,
)
from tools.anima_regional_lora_performance.matrix_manifest import (
    default_scaling_manifest,
)


def test_matrix_contexts_are_identical_across_every_region() -> None:
    """Ensure region scaling does not introduce prompt-content differences."""

    manifest = default_scaling_manifest()
    profile = manifest.profiles[7]
    attention = build_matrix_attention_fixture(
        manifest,
        profile,
        device=torch.device("cpu"),
    )

    assert len(attention.contexts.regions) == 4
    assert all(
        region.entries[0].context is attention.contexts.base_context
        for region in attention.contexts.regions
    )
    masks = attention.mask_bank.conditioning_masks
    assert tuple(masks.shape) == (4, 128, 128)
    assert torch.equal(masks.sum(dim=0), torch.ones((128, 128)))
    assert tuple(float(mask.sum().item()) for mask in masks) == (4096.0,) * 4


def test_matrix_half_full_and_zero_coverage_masks_are_exact() -> None:
    """Pin the control, all-mask, and pruning geometry independently."""

    manifest = default_scaling_manifest()
    half = build_matrix_attention_fixture(
        manifest,
        manifest.profiles[0],
        device=torch.device("cpu"),
    ).mask_bank.conditioning_masks
    full = build_matrix_attention_fixture(
        manifest,
        manifest.profiles[8],
        device=torch.device("cpu"),
    ).mask_bank.conditioning_masks
    pruned = build_matrix_attention_fixture(
        manifest,
        manifest.profiles[10],
        device=torch.device("cpu"),
    ).mask_bank.conditioning_masks

    assert float(half.sum().item()) == 8192.0
    assert torch.equal(full, torch.ones_like(full))
    assert torch.equal(pruned[0], torch.ones_like(pruned[0]))
    assert int(torch.count_nonzero(pruned[1:]).item()) == 0


def test_matrix_input_fixture_is_seed_stable_and_uses_thirty_calls() -> None:
    """Retain exact latent identity and trajectory independently of profiles."""

    manifest = default_scaling_manifest()
    attention = build_matrix_attention_fixture(
        manifest,
        manifest.profiles[0],
        device=torch.device("cpu"),
    )

    first = build_matrix_input_fixture(
        manifest,
        attention,
        device=torch.device("cpu"),
    )
    second = build_matrix_input_fixture(
        manifest,
        attention,
        device=torch.device("cpu"),
    )

    assert torch.equal(first[0], second[0])
    assert first[1] is attention.contexts.base_context
    assert tuple(first[2].shape) == (31,)
    assert float(first[2][0].item()) == 1.0
    assert float(first[2][-1].item()) == 0.0
