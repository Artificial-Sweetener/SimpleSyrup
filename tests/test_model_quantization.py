# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for reusable and Anima-specific quantization profiles."""

from __future__ import annotations

import pytest

from simple_syrup.domain.anima_quantization import (
    FP8_E4M3_PROFILE,
    FP8_E5M2_PROFILE,
    MXFP8_PROFILE,
    NVFP4_MIXED_PROFILE,
    ORIGINAL_PROFILE,
    AnimaQuantizationRecipe,
)
from simple_syrup.domain.model_quantization import (
    QuantizationFormat,
    QuantizationProfile,
    TensorDescriptor,
)


def test_anima_profiles_parse_labels_and_stable_ids() -> None:
    """Workflow labels and persisted identifiers resolve consistently."""

    recipe = AnimaQuantizationRecipe()

    assert recipe.profile_from_selection("Original") == ORIGINAL_PROFILE
    assert recipe.profile_from_selection("nvfp4-mixed") == NVFP4_MIXED_PROFILE
    assert QuantizationFormat.NVFP4.label == "NVFP4"
    with pytest.raises(ValueError, match="quantization profile must be one of"):
        recipe.profile_from_selection("unknown")


@pytest.mark.parametrize(
    "name",
    [
        "net.blocks.0.attn.q_proj.weight",
        "net.blocks.1.attn.q_proj.weight",
        "net.blocks.27.attn.q_proj.weight",
        "net.blocks.14.adaln_modulation.1.weight",
        "net.final_layer.linear.weight",
        "net.llm_adapter.proj.weight",
        "net.t_embedder.1.weight",
        "net.x_embedder.proj.weight",
        "some_other_model.blocks.14.attn.q_proj.weight",
    ],
)
@pytest.mark.parametrize(
    "profile",
    [FP8_E4M3_PROFILE, MXFP8_PROFILE, NVFP4_MIXED_PROFILE],
)
def test_anima_profiles_preserve_quality_sensitive_weights(
    name: str,
    profile: QuantizationProfile,
) -> None:
    """Every profile keeps known sensitive and out-of-envelope weights."""

    recipe = AnimaQuantizationRecipe()
    selected = recipe.profile_from_selection(profile.profile_id)

    assert recipe.policy_for(TensorDescriptor(name, (16, 16), "BF16"), selected) is None


def test_anima_mixed_profile_assigns_projection_specific_formats() -> None:
    """Recommended mixed precision follows the intended attention/MLP split."""

    recipe = AnimaQuantizationRecipe()

    def assigned(name: str) -> QuantizationFormat | None:
        return recipe.policy_for(
            TensorDescriptor(f"net.blocks.14.{name}.weight", (16, 16), "BF16"),
            NVFP4_MIXED_PROFILE,
        )

    assert assigned("attn.q_proj") is QuantizationFormat.NVFP4
    assert assigned("attn.k_proj") is QuantizationFormat.NVFP4
    assert assigned("attn.output_proj") is QuantizationFormat.NVFP4
    assert assigned("attn.v_proj") is QuantizationFormat.FP8_E4M3
    assert assigned("mlp.fc1") is QuantizationFormat.FP8_E4M3
    assert assigned("unmatched") is None


@pytest.mark.parametrize(
    ("profile", "expected"),
    [
        (FP8_E4M3_PROFILE, QuantizationFormat.FP8_E4M3),
        (FP8_E5M2_PROFILE, QuantizationFormat.FP8_E5M2),
        (MXFP8_PROFILE, QuantizationFormat.MXFP8),
    ],
)
def test_uniform_profiles_quantize_only_eligible_middle_block_matrices(
    profile: QuantizationProfile,
    expected: QuantizationFormat,
) -> None:
    """Uniform profiles share the safety envelope while selecting their format."""

    descriptor = TensorDescriptor(
        "diffusion_model.blocks.14.self_attn.q_proj.weight",
        (16, 16),
        "BF16",
    )

    assert AnimaQuantizationRecipe().policy_for(descriptor, profile) is expected
