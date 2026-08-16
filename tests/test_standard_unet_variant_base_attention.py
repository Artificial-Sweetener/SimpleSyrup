# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Characterize graph-local Attention Couple options for uncovered base output."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from simple_syrup.runtime.attention_coupling.unet_attn2_execution import (
    UnetAttn2Execution,
)
from simple_syrup.runtime.regional_lora.standard_unet_variant_base_attention import (
    StandardUnetVariantBaseAttention,
)


class _Resolver:
    """Provide the structural attn2 resolver boundary without executing it."""

    def resolve(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        extra_options: dict[str, Any],
    ) -> UnetAttn2Execution:
        """Reject unexpected callback execution in option-construction tests."""

        del query, context, extra_options
        raise AssertionError("Characterization must not execute attention.")


def test_prepare_adds_paired_callbacks_without_mutating_source_options() -> None:
    """Preserve unrelated patches while isolating the base graph's callback pair."""

    unrelated = object()
    source: dict[str, object] = {
        "patches": {"attn1_patch": [unrelated]},
        "transformer_index": 7,
    }

    prepared = StandardUnetVariantBaseAttention(_Resolver()).prepare(source)

    assert prepared is not source
    assert prepared["transformer_index"] == 7
    patches = prepared["patches"]
    assert isinstance(patches, dict)
    assert patches is not source["patches"]
    assert patches["attn1_patch"] == [unrelated]
    assert len(patches["attn2_patch"]) == 1
    assert callable(patches["attn2_patch"][0])
    assert len(patches["attn2_output_patch"]) == 1
    assert callable(patches["attn2_output_patch"][0])
    assert source == {
        "patches": {"attn1_patch": [unrelated]},
        "transformer_index": 7,
    }


@pytest.mark.parametrize("key", ("attn2_patch", "attn2_output_patch"))
def test_prepare_rejects_any_preexisting_attn2_callback_surface(key: str) -> None:
    """Fail closed instead of composing an unproved foreign callback order."""

    with pytest.raises(ValueError, match="already contains"):
        StandardUnetVariantBaseAttention(_Resolver()).prepare({"patches": {key: []}})
