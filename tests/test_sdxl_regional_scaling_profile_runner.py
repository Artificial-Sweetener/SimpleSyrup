# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify configurable indexed SDXL scaling profile construction."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.comfy_integration.artifacts import IntegrationArtifacts
from tools.sdxl_attention_coupling_integration.visual_lora_baseline_cases import (
    SdxlVisualPromptSet,
)
from tools.sdxl_regional_lora_performance.scaling_cases import (
    DeclaredSdxlRegionalScalingCase,
    regional_scaling_cases,
)
from tools.sdxl_regional_lora_performance.scaling_profile_runner import (
    _profile_workflows,
)


@pytest.mark.parametrize("call_index", (30, 15))
def test_profile_workflow_propagates_selected_call_index(
    tmp_path: Path,
    call_index: int,
) -> None:
    """Keep the historical final call and admit specialization diagnostics."""

    artifacts = IntegrationArtifacts(tmp_path)
    declared = _declared()

    _, _, profiled, _ = _profile_workflows(
        artifacts=artifacts,
        declared=declared,
        mask_names=("left.png", "right.png"),
        call_index=call_index,
    )

    inputs = profiled.prompt[profiled.profile_model_node_id]["inputs"]
    assert isinstance(inputs, dict)
    assert inputs["call_index"] == call_index


def test_profile_workflow_rejects_nonpositive_call_index(tmp_path: Path) -> None:
    """Fail before constructing a profile node for an impossible call."""

    with pytest.raises(ValueError, match="positive"):
        _profile_workflows(
            artifacts=IntegrationArtifacts(tmp_path),
            declared=_declared(),
            mask_names=("left.png", "right.png"),
            call_index=0,
        )


def _declared() -> DeclaredSdxlRegionalScalingCase:
    """Return the generic zero-use declaration required by graph construction."""

    return regional_scaling_cases(
        _prompts(),
        left_trigger_g="left trigger g",
        left_trigger_l="left trigger l",
        right_trigger_g="right trigger g",
        right_trigger_l="right trigger l",
        style_trigger_g="style trigger g",
        style_trigger_l="style trigger l",
    )[0]


def _prompts() -> SdxlVisualPromptSet:
    """Return one model-neutral dual-encoder prompt fixture."""

    return SdxlVisualPromptSet(
        base_positive_g="base g",
        base_positive_l="base l",
        base_negative_g="negative g",
        base_negative_l="negative l",
        left_positive_g="left g",
        left_positive_l="left l",
        right_positive_g="right g",
        right_positive_l="right l",
        left_negative_g="left negative g",
        left_negative_l="left negative l",
        right_negative_g="right negative g",
        right_negative_l="right negative l",
    )
