# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Classify decoded Anima LoRA targets by their authoritative model owner."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class AnimaRegionalLoraTargetOwner(StrEnum):
    """Name target owners relevant to regional diffusion admission."""

    IMAGE_TRANSFORMER_BLOCK = "image_transformer_block"
    LLM_ADAPTER = "llm_adapter"
    TEXT_ENCODER = "text_encoder"
    VAE = "vae"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True, slots=True)
class AnimaRegionalLoraTargetOwnership:
    """Bind one target owner to its exact regional-admission policy."""

    owner: AnimaRegionalLoraTargetOwner
    rejection_reason: str | None

    def __post_init__(self) -> None:
        """Require a reason exactly when regional execution is unsupported."""

        supported = self.owner is AnimaRegionalLoraTargetOwner.IMAGE_TRANSFORMER_BLOCK
        if supported != (self.rejection_reason is None):
            raise ValueError(
                "Anima target ownership has inconsistent admission policy."
            )


def classify_anima_regional_lora_target_owner(
    target: str,
) -> AnimaRegionalLoraTargetOwnership:
    """Classify one decoded target without interpreting adapter tensors."""

    if not isinstance(target, str) or not target:
        raise TypeError("Anima regional LoRA target must be a nonempty string.")
    if target.startswith("diffusion_model.blocks."):
        return AnimaRegionalLoraTargetOwnership(
            AnimaRegionalLoraTargetOwner.IMAGE_TRANSFORMER_BLOCK,
            None,
        )
    if target.startswith("diffusion_model.llm_adapter."):
        return AnimaRegionalLoraTargetOwnership(
            AnimaRegionalLoraTargetOwner.LLM_ADAPTER,
            "Anima LLM adapter target is outside the approved regional "
            "diffusion-LoRA contract",
        )
    if target.startswith(
        (
            "clip.",
            "cond_stage_model.",
            "lora_te_",
            "lora_te1_",
            "lora_te2_",
            "text_encoder.",
        )
    ):
        return AnimaRegionalLoraTargetOwnership(
            AnimaRegionalLoraTargetOwner.TEXT_ENCODER,
            "text-encoder target must be encoded into the supplied regional "
            "conditioning context",
        )
    if target.startswith(("first_stage_model.", "vae.")):
        return AnimaRegionalLoraTargetOwnership(
            AnimaRegionalLoraTargetOwner.VAE,
            "VAE target is outside the approved regional diffusion-LoRA contract",
        )
    return AnimaRegionalLoraTargetOwnership(
        AnimaRegionalLoraTargetOwner.UNSUPPORTED,
        "unsupported Anima target path",
    )
