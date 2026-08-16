# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify persistent-variant SDXL visual sampler-work expectations."""

from __future__ import annotations

import pytest

from tools.sdxl_attention_coupling_integration.matrix import MODES
from tools.sdxl_attention_coupling_integration.visual_case_model import (
    GlobalVisualAdapter,
    RegionalVisualAdapter,
    SdxlVisualCase,
    VisualMaskProfile,
)
from tools.sdxl_attention_coupling_integration.visual_runtime_expectations import (
    SDXL_VISUAL_RUNTIME_EXPECTATIONS,
)

_MODE = MODES[0]
_REGIONAL = RegionalVisualAdapter("adapter-a", 0.5, 0.5)


@pytest.mark.parametrize(
    "case",
    [
        SdxlVisualCase("base", "Base"),
        SdxlVisualCase(
            "global",
            "Global",
            global_adapters=(GlobalVisualAdapter("adapter-global", 0.5),),
        ),
        SdxlVisualCase("left", "Left", left_adapters=(_REGIONAL,)),
        SdxlVisualCase("right", "Right", right_adapters=(_REGIONAL,)),
        SdxlVisualCase(
            "multiple-left",
            "Multiple left",
            left_adapters=(_REGIONAL, _REGIONAL),
        ),
        SdxlVisualCase(
            "both",
            "Both",
            left_adapters=(_REGIONAL,),
            right_adapters=(_REGIONAL,),
        ),
        SdxlVisualCase(
            "both-uncovered",
            "Both with uncovered center",
            left_adapters=(_REGIONAL,),
            right_adapters=(_REGIONAL,),
            mask_profile=VisualMaskProfile.UNCOVERED_CENTER,
        ),
    ],
)
def test_outer_model_calls_follow_native_sampler_topology(
    case: SdxlVisualCase,
) -> None:
    """Keep outer call count stable while regional variants execute internally."""

    assert (
        SDXL_VISUAL_RUNTIME_EXPECTATIONS.expected_model_calls(case, _MODE)
        == _MODE.expected_model_calls
    )


def test_native_model_calls_reject_foreign_values() -> None:
    """Fail closed before inferring work from untyped scenario data."""

    with pytest.raises(TypeError, match="case"):
        SDXL_VISUAL_RUNTIME_EXPECTATIONS.expected_model_calls(object(), _MODE)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="mode"):
        SDXL_VISUAL_RUNTIME_EXPECTATIONS.expected_model_calls(
            SdxlVisualCase("base", "Base"),
            object(),  # type: ignore[arg-type]
        )
