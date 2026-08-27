# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Test observational attention-logit calculation."""

from __future__ import annotations

import math

import pytest
import torch

from simple_syrup.runtime.attention_region_logits import scaled_attention_logits


def test_cpu_logits_use_stable_fp32_calculation() -> None:
    """Keep the portable CPU path numerically stable for tests and fallback use."""

    query = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=torch.float16)
    key = torch.tensor([[[[2.0, 1.0], [0.0, 1.0]]]], dtype=torch.float16)

    logits = scaled_attention_logits(query, key)

    expected = torch.einsum("bhqd,bhkd->bhqk", query.float(), key.float()) / math.sqrt(
        2.0
    )
    assert logits.dtype is torch.float32
    assert torch.allclose(logits, expected)


def test_rejects_misaligned_attention_axes() -> None:
    """Fail closed before observational matrix multiplication."""

    with pytest.raises(ValueError, match="axes must align"):
        scaled_attention_logits(
            torch.ones(1, 2, 3, 4),
            torch.ones(1, 3, 3, 4),
        )
