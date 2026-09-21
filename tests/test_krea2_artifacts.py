# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for trusted Krea 2 component metadata and choices."""

from __future__ import annotations

from simple_syrup.runtime.krea2_artifacts import (
    KREA2_QWEN3_VL_4B_BF16,
    KREA2_QWEN3_VL_4B_FP8,
    KREA2_TEXT_ENCODER_ARTIFACTS,
)
from simple_syrup.runtime.qwen_artifacts import QWEN_IMAGE_VAE


def test_krea2_artifacts_are_checksum_and_revision_pinned() -> None:
    """Every downloadable component uses an immutable URL and exact digest."""

    artifacts = (
        KREA2_QWEN3_VL_4B_FP8,
        KREA2_QWEN3_VL_4B_BF16,
        QWEN_IMAGE_VAE,
    )

    for artifact in artifacts:
        assert "/resolve/main/" not in artifact.source_url
        assert "e5ea8b4dd7f38f348b138eb0fe29f92c0e367e96" in artifact.source_url
        assert len(artifact.sha256) == 64
        assert set(artifact.sha256) <= set("0123456789abcdef")


def test_krea2_official_encoder_selections_map_to_downloadable_artifacts() -> None:
    """Both public precision choices resolve to their exact official files."""

    assert KREA2_TEXT_ENCODER_ARTIFACTS == {
        "qwen3vl_4b_fp8_scaled.safetensors": KREA2_QWEN3_VL_4B_FP8,
        "qwen3vl_4b_bf16.safetensors": KREA2_QWEN3_VL_4B_BF16,
    }
