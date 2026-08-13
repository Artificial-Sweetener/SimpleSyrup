# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify separate Anima and standard-UNet processed-context policies."""

from __future__ import annotations

import pytest
import torch

from simple_syrup.runtime.attention_coupling.anima_context import (
    ANIMA_REGIONAL_CONTEXT_VALIDATOR,
)
from simple_syrup.runtime.attention_coupling.unet_context import (
    STANDARD_UNET_REGIONAL_CONTEXT_VALIDATOR,
)


def test_anima_context_validator_requires_exact_fixed_geometry() -> None:
    """Accept only the installed Anima semantic token and feature dimensions."""

    ANIMA_REGIONAL_CONTEXT_VALIDATOR.validate(
        torch.zeros(1, 512, 1024),
        prompt_type="positive",
        conditioning_index=0,
    )

    with pytest.raises(ValueError, match="exactly 512"):
        ANIMA_REGIONAL_CONTEXT_VALIDATOR.validate(
            torch.zeros(1, 77, 1024),
            prompt_type="positive",
            conditioning_index=0,
        )
    with pytest.raises(ValueError, match="feature width 1024"):
        ANIMA_REGIONAL_CONTEXT_VALIDATOR.validate(
            torch.zeros(1, 512, 2048),
            prompt_type="negative",
            conditioning_index=1,
        )


@pytest.mark.parametrize("shape", [(1, 77, 768), (1, 154, 768), (2, 77, 2048)])
def test_standard_unet_context_validator_accepts_model_consumed_bsd_geometry(
    shape: tuple[int, int, int],
) -> None:
    """Admit SD1 and SDXL token and feature dimensions without fixed constants."""

    STANDARD_UNET_REGIONAL_CONTEXT_VALIDATOR.validate(
        torch.zeros(shape),
        prompt_type="positive",
        conditioning_index=0,
    )
