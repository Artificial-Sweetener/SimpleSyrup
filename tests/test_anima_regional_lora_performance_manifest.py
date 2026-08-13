# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Prove the immutable P5.7 zero/one/four benchmark contract."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

import pytest
import torch

from tools.anima_regional_lora_performance.execution_fixture import (
    build_attention_fixture,
    build_input_fixture,
)
from tools.anima_regional_lora_performance.manifest import default_manifest


def test_manifest_pins_full_resolution_calls_artifacts_and_limits() -> None:
    """Retain every non-negotiable performance gate dimension."""

    manifest = default_manifest(repeats=5)

    assert (manifest.width, manifest.height) == (1024, 1024)
    assert manifest.denoiser_calls == 30
    assert manifest.repeats == 5
    assert [profile.adapter_count for profile in manifest.profiles] == [0, 1, 4]
    assert [profile.maximum_overhead_percent for profile in manifest.profiles] == [
        0.0,
        15.0,
        35.0,
    ]
    assert [artifact.size_bytes for artifact in manifest.artifacts] == [
        4_182_218_328,
        138_663_768,
    ]
    assert all(len(artifact.sha256) == 64 for artifact in manifest.artifacts)


@pytest.mark.parametrize(
    "changes",
    [
        {"width": 512},
        {"denoiser_calls": 29},
        {"warmup_calls": 0},
        {"repeats": 2},
        {"profiles": default_manifest().profiles[:2]},
    ],
)
def test_manifest_rejects_reduced_or_incomplete_gates(
    changes: dict[str, object],
) -> None:
    """Prevent a cheaper matrix from being represented as P5.7 evidence."""

    with pytest.raises(ValueError):
        replace(default_manifest(), **cast(Any, changes))


def test_runtime_fixture_uses_normal_cfg_full_latents_and_one_hard_region() -> None:
    """Build the exact deterministic tensor geometry without requiring CUDA."""

    manifest = default_manifest()
    attention = build_attention_fixture(manifest, device=torch.device("cpu"))
    latent, context, sigmas = build_input_fixture(
        manifest,
        attention,
        device=torch.device("cpu"),
    )

    assert latent.shape == (2, 16, 1, 128, 128)
    assert context.shape == (2, 512, 1024)
    assert sigmas.shape == (31,)
    assert [chunk.branch.value for chunk in attention.contexts.chunks] == [
        "positive",
        "negative",
    ]
    assert attention.mask_bank.conditioning_masks[:, :, :64].eq(1.0).all()
    assert attention.mask_bank.conditioning_masks[:, :, 64:].eq(0.0).all()
