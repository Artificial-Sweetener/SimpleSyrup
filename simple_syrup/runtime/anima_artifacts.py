# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Declare the trusted automatic artifacts used by Anima."""

from __future__ import annotations

from .auto_model_artifact import AutoModelArtifact
from .qwen_artifacts import QWEN_IMAGE_VAE

ANIMA_QWEN_TEXT_ENCODER = AutoModelArtifact(
    cache_id="anima_qwen_text_encoder",
    filename="qwen_3_06b_base.safetensors",
    folder_name="text_encoders",
    canonical_subfolder="qwen",
    source_url=(
        "https://huggingface.co/circlestone-labs/Anima/resolve/main/"
        "split_files/text_encoders/qwen_3_06b_base.safetensors"
    ),
    source_repo="circlestone-labs/Anima",
    description="Anima Qwen3 0.6B text encoder",
    sha256="cd2a512003e2f9f3cd3c32a9c3573f820bb28c940f73c57b1ddaa983d9223eba",
    file_size_bytes=1_192_135_096,
)

ANIMA_AUTO_ARTIFACTS = (ANIMA_QWEN_TEXT_ENCODER, QWEN_IMAGE_VAE)
