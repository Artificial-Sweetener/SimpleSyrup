# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Declare checksum-pinned automatic artifacts for FLUX and FLUX.2."""

from __future__ import annotations

from ..domain.flux_profiles import Flux2TextEncoderProfile
from .auto_model_artifact import AutoModelArtifact

FLUX_CLIP_L = AutoModelArtifact(
    cache_id="flux_clip_l",
    filename="clip_l.safetensors",
    folder_name="text_encoders",
    canonical_subfolder="flux",
    source_url=(
        "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/"
        "6af2a98e3f615bdfa612fbd85da93d1ed5f69ef5/clip_l.safetensors"
    ),
    source_repo="comfyanonymous/flux_text_encoders",
    description="FLUX CLIP-L text encoder",
    sha256="660c6f5b1abae9dc498ac2d21e1347d2abdb0cf6c0c0c8576cd796491d9a6cdd",
)

FLUX_T5_XXL = AutoModelArtifact(
    cache_id="flux_t5_xxl",
    filename="t5xxl_fp16.safetensors",
    folder_name="text_encoders",
    canonical_subfolder="flux",
    source_url=(
        "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/"
        "6af2a98e3f615bdfa612fbd85da93d1ed5f69ef5/t5xxl_fp16.safetensors"
    ),
    source_repo="comfyanonymous/flux_text_encoders",
    description="FLUX T5-XXL FP16 text encoder",
    sha256="6e480b09fae049a72d2a8c5fbccb8d3e92febeb233bbe9dfe7256958a9167635",
)

FLUX_VAE = AutoModelArtifact(
    cache_id="flux_vae",
    filename="ae.safetensors",
    folder_name="vae",
    canonical_subfolder="flux",
    source_url=(
        "https://huggingface.co/Comfy-Org/Lumina_Image_2.0_Repackaged/resolve/"
        "22e393d707f2d13e736b1a461c958644258cd9d9/split_files/vae/"
        "ae.safetensors"
    ),
    source_repo="Comfy-Org/Lumina_Image_2.0_Repackaged",
    description="FLUX autoencoder VAE",
    sha256="afc8e28272cd15db3919bacdb6918ce9c1ed22e96cb12c4d5ed0fba823529e38",
)

FLUX2_DEV_TEXT_ENCODER = AutoModelArtifact(
    cache_id="flux2_mistral_3_small_text_encoder",
    filename="mistral_3_small_flux2_bf16.safetensors",
    folder_name="text_encoders",
    canonical_subfolder="flux2",
    source_url=(
        "https://huggingface.co/Comfy-Org/flux2-dev/resolve/"
        "03d6521e6f6a47396b3f951cbea50f7e6c2f482e/split_files/text_encoders/"
        "mistral_3_small_flux2_bf16.safetensors"
    ),
    source_repo="Comfy-Org/flux2-dev",
    description="FLUX.2 dev Mistral 3 Small text encoder",
    sha256="7d79902f60b1aeb3a6de2cfad02f4367b5e300a1387de3d03ac717cfa3df117c",
)

FLUX2_KLEIN_4B_TEXT_ENCODER = AutoModelArtifact(
    cache_id="flux2_klein_4b_qwen_text_encoder",
    filename="qwen_3_4b.safetensors",
    folder_name="text_encoders",
    canonical_subfolder="flux2",
    source_url=(
        "https://huggingface.co/Comfy-Org/vae-text-encorder-for-flux-klein-4b/"
        "resolve/a9e4ca87c16db4c4e1a16406a9ddb300ab0ae246/split_files/"
        "text_encoders/qwen_3_4b.safetensors"
    ),
    source_repo="Comfy-Org/vae-text-encorder-for-flux-klein-4b",
    description="FLUX.2 Klein 4B Qwen3 text encoder",
    sha256="6c671498573ac2f7a5501502ccce8d2b08ea6ca2f661c458e708f36b36edfc5a",
)

FLUX2_KLEIN_9B_TEXT_ENCODER = AutoModelArtifact(
    cache_id="flux2_klein_9b_qwen_text_encoder",
    filename="qwen_3_8b_fp8mixed.safetensors",
    folder_name="text_encoders",
    canonical_subfolder="flux2",
    source_url=(
        "https://huggingface.co/Comfy-Org/vae-text-encorder-for-flux-klein-9b/"
        "resolve/23fbc8aa8b621f29f2249cd1bd9c47e5d0eebd83/split_files/"
        "text_encoders/qwen_3_8b_fp8mixed.safetensors"
    ),
    source_repo="Comfy-Org/vae-text-encorder-for-flux-klein-9b",
    description="FLUX.2 Klein 9B Qwen3 8B FP8-mixed text encoder",
    sha256="abad16806e0cbabc54e0325d6565847443fe396d5f0be38bb3cd3fe75a1201d6",
)

FLUX2_VAE = AutoModelArtifact(
    cache_id="flux2_vae",
    filename="flux2-vae.safetensors",
    folder_name="vae",
    canonical_subfolder="flux2",
    source_url=(
        "https://huggingface.co/Comfy-Org/flux2-dev/resolve/"
        "03d6521e6f6a47396b3f951cbea50f7e6c2f482e/split_files/vae/"
        "flux2-vae.safetensors"
    ),
    source_repo="Comfy-Org/flux2-dev",
    description="FLUX.2 VAE",
    sha256="d64f3a68e1cc4f9f4e29b6e0da38a0204fe9a49f2d4053f0ec1fa1ca02f9c4b5",
)

FLUX2_TEXT_ENCODERS: dict[Flux2TextEncoderProfile, AutoModelArtifact] = {
    Flux2TextEncoderProfile.DEV: FLUX2_DEV_TEXT_ENCODER,
    Flux2TextEncoderProfile.KLEIN_4B: FLUX2_KLEIN_4B_TEXT_ENCODER,
    Flux2TextEncoderProfile.KLEIN_9B: FLUX2_KLEIN_9B_TEXT_ENCODER,
}
