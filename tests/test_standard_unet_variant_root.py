# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify managed persistent standard-UNet root isolation."""

from __future__ import annotations

from typing import cast

import torch
from torch import nn

from simple_syrup.runtime.regional_lora.standard_unet_variant_root import (
    StandardUnetVariantRootBuilder,
)


class _Diffusion(nn.Module):
    """Expose one ordinary child and forward implementation."""

    def __init__(self) -> None:
        """Create one deterministic child."""

        super().__init__()
        self.layer = nn.Linear(1, 1, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run the ordinary child."""

        return cast(torch.Tensor, self.layer(inputs))


def test_root_keeps_source_clean_and_registers_variants_for_device_movement() -> None:
    """Share base children while isolating the private variant registry."""

    source = _Diffusion()
    root = StandardUnetVariantRootBuilder.build(source)
    variant = nn.Linear(1, 1, bias=False)
    root.variants["variant_0000"] = variant

    assert root.module is not source
    assert root.module.layer is source.layer
    assert not hasattr(source, "simple_syrup_regional_variants")
    assert tuple(root.module.named_modules())[-1][1] is variant
