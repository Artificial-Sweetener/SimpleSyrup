# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for external LLM image encoding."""

from __future__ import annotations

import base64

import torch

from simple_syrup.runtime.external_llm_images import ExternalLLMImageEncoder


def test_image_encoder_returns_png_data_url() -> None:
    """ComfyUI IMAGE tensors are encoded as PNG data URLs."""

    image = torch.zeros((1, 2, 2, 3), dtype=torch.float32)

    data_url = ExternalLLMImageEncoder().encode_first_image_as_data_url(image)

    assert data_url.startswith("data:image/png;base64,")
    payload = data_url.removeprefix("data:image/png;base64,")
    assert base64.b64decode(payload).startswith(b"\x89PNG")
