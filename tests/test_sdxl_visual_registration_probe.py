# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify U11 live registration probe metadata validation."""

from __future__ import annotations

import pytest

from tools.comfy_api import JsonObject
from tools.prove_sdxl_visual_registration import (
    _nested_text_values,
    _require_choice,
    _required_node_ids,
)
from tools.sdxl_attention_coupling_integration.visual_cases import CHECKPOINT_NAME


def test_required_nodes_cover_all_public_sampler_geometries() -> None:
    """Probe the same live registration surface as the 14-output matrix."""

    required = _required_node_ids("run")

    assert "SimpleSyrup.KSamplerAttentionCoupling" in required
    assert "SimpleSyrup.KSamplerAttentionCouplingTiled" in required
    assert "SimpleSyrup.KSamplerAttentionCouplingContextual" in required
    assert "CLIPTextEncodeSDXL" in required


def test_nested_choice_validation_is_layout_independent_and_exact() -> None:
    """Find exact Comfy choices without relying on one metadata nesting shape."""

    metadata: JsonObject = {
        "input": {"required": {"ckpt_name": [[CHECKPOINT_NAME], {}]}}
    }

    assert CHECKPOINT_NAME in _nested_text_values(metadata)
    _require_choice(metadata, CHECKPOINT_NAME)
    with pytest.raises(ValueError, match="omits model choice"):
        _require_choice(metadata, "missing.safetensors")
