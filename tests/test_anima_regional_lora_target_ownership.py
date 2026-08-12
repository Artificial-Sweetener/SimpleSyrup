"""Prove exact Anima regional-LoRA target ownership classification."""

from __future__ import annotations

from typing import cast

import pytest

from simple_syrup.runtime.regional_lora.anima_target_ownership import (
    AnimaRegionalLoraTargetOwner,
    AnimaRegionalLoraTargetOwnership,
    classify_anima_regional_lora_target_owner,
)


@pytest.mark.parametrize(
    ("target", "owner", "reason"),
    [
        (
            "diffusion_model.blocks.0.self_attn.q_proj",
            AnimaRegionalLoraTargetOwner.IMAGE_TRANSFORMER_BLOCK,
            None,
        ),
        (
            "diffusion_model.llm_adapter.blocks.0.cross_attn.q_proj",
            AnimaRegionalLoraTargetOwner.LLM_ADAPTER,
            "Anima LLM adapter target is outside the approved regional "
            "diffusion-LoRA contract",
        ),
        (
            "lora_te_model.layers.0.self_attn.q_proj",
            AnimaRegionalLoraTargetOwner.TEXT_ENCODER,
            "text-encoder target must be encoded into the supplied regional "
            "conditioning context",
        ),
        (
            "text_encoder.model.layers.0.mlp.down_proj",
            AnimaRegionalLoraTargetOwner.TEXT_ENCODER,
            "text-encoder target must be encoded into the supplied regional "
            "conditioning context",
        ),
        (
            "vae.decoder.conv_in",
            AnimaRegionalLoraTargetOwner.VAE,
            "VAE target is outside the approved regional diffusion-LoRA contract",
        ),
        (
            "diffusion_model.final_layer.linear",
            AnimaRegionalLoraTargetOwner.UNSUPPORTED,
            "unsupported Anima target path",
        ),
    ],
)
def test_target_owner_classifies_exact_path_authority(
    target: str,
    owner: AnimaRegionalLoraTargetOwner,
    reason: str | None,
) -> None:
    """Keep unsupported owners outside the image-transformer executor."""

    assert classify_anima_regional_lora_target_owner(target) == (
        AnimaRegionalLoraTargetOwnership(owner, reason)
    )


@pytest.mark.parametrize("target", ["", object()])
def test_target_owner_rejects_malformed_values(target: object) -> None:
    """Require a decoded nonempty target before lexical classification."""

    with pytest.raises(TypeError, match="nonempty string"):
        classify_anima_regional_lora_target_owner(cast(str, target))


def test_target_ownership_value_rejects_inconsistent_policy() -> None:
    """Prevent supported and rejected owner states from becoming ambiguous."""

    with pytest.raises(ValueError, match="inconsistent admission policy"):
        AnimaRegionalLoraTargetOwnership(
            AnimaRegionalLoraTargetOwner.LLM_ADAPTER,
            None,
        )
