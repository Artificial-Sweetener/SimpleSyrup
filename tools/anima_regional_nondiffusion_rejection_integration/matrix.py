# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Define the closed P9.6 non-diffusion regional-target rejection matrix."""

from __future__ import annotations

from dataclasses import dataclass

from tools.anima_attention_coupling_prompts import PINNED_ADAPTER_A
from tools.anima_regional_lora_admission_integration.graph_contract import (
    CFG,
    PublicRegionalLoraHook,
    RegionalLoraAdmissionGraphCase,
)

TURBO_LORA_NAME = r"Anima\anima-turbo-lora-v0.2.safetensors"
TURBO_ADAPTER_IDENTITY = "anima-turbo-v0.2"
ADAPTER_A_ADAPTER_IDENTITY = "supported-adapter_a"
LLM_ADAPTER_TARGET_FIXTURE = "llm-adapter-target"
NON_DIFFUSION_OWNER_BUNDLE_FIXTURE = "non-diffusion-owner-bundle"


@dataclass(frozen=True, slots=True)
class PinnedTurboArtifactIdentity:
    """Record the exact public Turbo rejection fixture without a local path."""

    stable_name: str
    lora_name: str
    size_bytes: int
    sha256: str


PINNED_TURBO = PinnedTurboArtifactIdentity(
    stable_name="anima-turbo-lora-v0.2.safetensors",
    lora_name=TURBO_LORA_NAME,
    size_bytes=148_902_616,
    sha256="1b55e40bdb1d0e5a78cb498f245fccfdaae97823265db957d2aabdcf4cd3caf1",
)
TURBO_HOOK = PublicRegionalLoraHook(
    TURBO_LORA_NAME,
    1.0,
    TURBO_ADAPTER_IDENTITY,
)
ADAPTER_A_HOOK = PublicRegionalLoraHook(
    PINNED_ADAPTER_A,
    0.75,
    ADAPTER_A_ADAPTER_IDENTITY,
)


@dataclass(frozen=True, slots=True)
class AnimaNondiffusionRejectionCase(RegionalLoraAdmissionGraphCase):
    """Add exact owner-rejection expectations to one shared graph case."""

    expected_issue_count: int
    expected_issue_adapter_index: int
    expected_error_fragments: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate graph values and exact issue-cardinality policy."""

        RegionalLoraAdmissionGraphCase.__post_init__(self)
        if self.expected_issue_count < 1:
            raise ValueError("P9.6 rejection cases require at least one issue.")
        if self.expected_issue_adapter_index < 0:
            raise ValueError("P9.6 issue adapter indices must not be negative.")
        if not self.expected_error_fragments:
            raise ValueError("P9.6 rejection cases require exact error fragments.")


def cases() -> tuple[AnimaNondiffusionRejectionCase, ...]:
    """Return the exact real, isolated, aggregate, and mixed-owner matrix."""

    llm_reason = (
        "Anima LLM adapter target is outside the approved regional "
        "diffusion-LoRA contract"
    )
    text_reason = (
        "text-encoder target must be encoded into the supplied regional "
        "conditioning context"
    )
    vae_reason = "VAE target is outside the approved regional diffusion-LoRA contract"
    return (
        AnimaNondiffusionRejectionCase(
            "turbo-mixed-diffusion-llm",
            "Reject real Turbo diffusion plus LLM-adapter targets atomically",
            (TURBO_HOOK,),
            None,
            None,
            CFG,
            0,
            60,
            0,
            (
                "failed before sampling",
                TURBO_ADAPTER_IDENTITY,
                "diffusion_model.llm_adapter.blocks.0.cross_attn.k_proj",
                "diffusion_model.llm_adapter.blocks.5.self_attn.v_proj",
                llm_reason,
            ),
        ),
        AnimaNondiffusionRejectionCase(
            "llm-adapter-only",
            "Reject an isolated Anima LLM-adapter regional target",
            (),
            LLM_ADAPTER_TARGET_FIXTURE,
            LLM_ADAPTER_TARGET_FIXTURE,
            CFG,
            0,
            1,
            0,
            (
                "failed before sampling",
                LLM_ADAPTER_TARGET_FIXTURE,
                "diffusion_model.llm_adapter.blocks.0.cross_attn.q_proj",
                llm_reason,
            ),
        ),
        AnimaNondiffusionRejectionCase(
            "text-encoder-vae-bundle",
            "Reject text-encoder and VAE regional targets together",
            (),
            NON_DIFFUSION_OWNER_BUNDLE_FIXTURE,
            NON_DIFFUSION_OWNER_BUNDLE_FIXTURE,
            CFG,
            0,
            2,
            0,
            (
                "failed before sampling",
                "lora_te_model.layers.0.self_attn.q_proj",
                text_reason,
                "vae.decoder.conv_in",
                vae_reason,
            ),
        ),
        AnimaNondiffusionRejectionCase(
            "adapter_a-plus-nondiffusion-bundle",
            "Reject ADAPTER_A plus non-diffusion owners without partial sampling",
            (ADAPTER_A_HOOK,),
            NON_DIFFUSION_OWNER_BUNDLE_FIXTURE,
            NON_DIFFUSION_OWNER_BUNDLE_FIXTURE,
            CFG,
            0,
            2,
            1,
            (
                "failed before sampling",
                "adapter 1",
                NON_DIFFUSION_OWNER_BUNDLE_FIXTURE,
                "lora_te_model.layers.0.self_attn.q_proj",
                text_reason,
                "vae.decoder.conv_in",
                vae_reason,
            ),
        ),
    )
