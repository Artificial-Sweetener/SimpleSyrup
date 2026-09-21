# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Declare trusted automatic artifacts and selections for Krea 2."""

from __future__ import annotations

from .auto_model_artifact import AutoModelArtifact

KREA2_AUTO_TEXT_ENCODER = "auto"

KREA2_QWEN3_VL_4B_FP8 = AutoModelArtifact(
    cache_id="krea2_qwen3vl_4b_fp8_scaled",
    filename="qwen3vl_4b_fp8_scaled.safetensors",
    folder_name="text_encoders",
    canonical_subfolder="krea2",
    source_url=(
        "https://huggingface.co/Comfy-Org/Krea-2/resolve/"
        "e5ea8b4dd7f38f348b138eb0fe29f92c0e367e96/text_encoders/"
        "qwen3vl_4b_fp8_scaled.safetensors"
    ),
    source_repo="Comfy-Org/Krea-2",
    description="Krea 2 Qwen3-VL 4B FP8-scaled text encoder",
    sha256="54bd5144df0bbc25dd6ccadfcb826b521445a1b06ae5a42570bdd2974ca87094",
    file_size_bytes=5_242_467_968,
)

KREA2_QWEN3_VL_4B_BF16 = AutoModelArtifact(
    cache_id="krea2_qwen3vl_4b_bf16",
    filename="qwen3vl_4b_bf16.safetensors",
    folder_name="text_encoders",
    canonical_subfolder="krea2",
    source_url=(
        "https://huggingface.co/Comfy-Org/Krea-2/resolve/"
        "e5ea8b4dd7f38f348b138eb0fe29f92c0e367e96/text_encoders/"
        "qwen3vl_4b_bf16.safetensors"
    ),
    source_repo="Comfy-Org/Krea-2",
    description="Krea 2 Qwen3-VL 4B BF16 text encoder",
    sha256="36f3ff447ef59201722e8f9ce6020c9819fdcfba6aa2608c4e09b1c0ce114e34",
    file_size_bytes=8_875_719_384,
)

KREA2_TEXT_ENCODER_ARTIFACTS = {
    KREA2_QWEN3_VL_4B_FP8.filename: KREA2_QWEN3_VL_4B_FP8,
    KREA2_QWEN3_VL_4B_BF16.filename: KREA2_QWEN3_VL_4B_BF16,
}


__all__ = [
    "KREA2_AUTO_TEXT_ENCODER",
    "KREA2_QWEN3_VL_4B_BF16",
    "KREA2_QWEN3_VL_4B_FP8",
    "KREA2_TEXT_ENCODER_ARTIFACTS",
]
