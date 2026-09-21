# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Declare shared checksum-pinned Qwen model artifacts."""

from __future__ import annotations

from .auto_model_artifact import AutoModelArtifact

QWEN_IMAGE_VAE = AutoModelArtifact(
    cache_id="qwen_image_vae",
    filename="qwen_image_vae.safetensors",
    folder_name="vae",
    canonical_subfolder="qwen",
    source_url=(
        "https://huggingface.co/Comfy-Org/Krea-2/resolve/"
        "e5ea8b4dd7f38f348b138eb0fe29f92c0e367e96/vae/"
        "qwen_image_vae.safetensors"
    ),
    source_repo="Comfy-Org/Krea-2",
    description="Qwen Image VAE",
    sha256="a70580f0213e67967ee9c95f05bb400e8fb08307e017a924bf3441223e023d1f",
    file_size_bytes=253_806_246,
)

__all__ = ["QWEN_IMAGE_VAE"]
