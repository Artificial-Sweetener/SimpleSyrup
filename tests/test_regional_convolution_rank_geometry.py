# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify exact generic convolution rank-space geometry resolution."""

from __future__ import annotations

import torch

from simple_syrup.domain.regional_lora_plan import RegionalLoraBranch
from simple_syrup.runtime.regional_lora.convolution_execution_plan import (
    RegionalConvolutionParameters,
    RegionalConvolutionTargetUse,
)
from simple_syrup.runtime.regional_lora.convolution_preparation import (
    RegionalConvolutionPreparation,
)
from simple_syrup.runtime.regional_lora.convolution_rank_geometry import (
    RegionalConvolutionRankGeometryResolver,
)


def test_rank_geometry_applies_down_and_middle_convolution_shapes() -> None:
    """Match rectangular stride/padding/dilation before the ordinary up path."""

    use = RegionalConvolutionTargetUse(
        0,
        0,
        RegionalLoraBranch.POSITIVE,
        (1, 2, 3),
        RegionalConvolutionPreparation(
            torch.ones((2, 1, 1, 1)),
            torch.ones((2, 2, 3, 3)),
            torch.ones((1, 2, 1, 1)),
        ),
        RegionalConvolutionParameters(
            2,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
            1,
            1,
            (3, 3),
        ),
        1.0,
    )

    assert RegionalConvolutionRankGeometryResolver().resolve((9, 13), use) == (4, 6)
