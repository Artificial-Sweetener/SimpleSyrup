# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify exact conventional persistent standard-UNet weight materialization."""

from __future__ import annotations

import torch
from comfy.model_patcher import ModelPatcher
from torch import nn

from simple_syrup.domain.regional_lora_plan import RegionalLoraScheduleBoundary
from simple_syrup.runtime.regional_lora.comfy_adapter_resolution import (
    ComfyAdapterTargetPath,
    ComfyNormalizedAdapterTarget,
)
from simple_syrup.runtime.regional_lora.standard_unet_variant_materialization import (
    StandardUnetVariantMaterializer,
)
from simple_syrup.runtime.regional_lora.standard_unet_variant_topology import (
    StandardUnetRegionalVariant,
    StandardUnetVariantAdapter,
)


class _Diffusion(nn.Module):
    """Expose one generic conventional target parameter."""

    def __init__(self) -> None:
        """Create one globally biased source weight."""

        super().__init__()
        self.layer = nn.Linear(2, 2, bias=False)
        self.layer.weight.data.fill_(2.0)


class _Root(nn.Module):
    """Expose the generic diffusion module through Comfy's normal path."""

    def __init__(self) -> None:
        """Create one globally biased source weight."""

        super().__init__()
        self.diffusion_model = _Diffusion()


def test_materialization_applies_global_then_ordered_regional_strengths() -> None:
    """Match conventional patch order without requiring a live patched weight."""

    root = _Root()
    patcher = ModelPatcher(root, torch.device("cpu"), torch.device("cpu"))
    patcher.patches["diffusion_model.layer.weight"] = [
        (1.0, ("diff", (torch.full((2, 2), 3.0),)), 1.0, None, None)
    ]
    variant = StandardUnetRegionalVariant(
        0,
        (
            _adapter(0, torch.ones((2, 2))),
            _adapter(1, torch.full((2, 2), 2.0)),
        ),
    )

    result = StandardUnetVariantMaterializer().materialize(
        patcher,
        variant,
        (0.5, 0.25),
    )

    assert result.region_index == 0
    assert result.parameters[0].path == "layer.weight"
    assert torch.equal(result.parameters[0].tensor, torch.full((2, 2), 6.0))
    assert torch.equal(root.diffusion_model.layer.weight, torch.full((2, 2), 2.0))


def _adapter(index: int, delta: torch.Tensor) -> StandardUnetVariantAdapter:
    """Return one generic legacy-diff operation accepted by Comfy."""

    target = ComfyNormalizedAdapterTarget(
        ComfyAdapterTargetPath("diffusion_model.layer.weight", None),
        ("diff", (delta,)),
        "LegacyDiff",
        (f"source-{index}",),
        False,
    )
    return StandardUnetVariantAdapter(
        index,
        0,
        1.0,
        (RegionalLoraScheduleBoundary(0.0, 1.0, 1.0, 0),),
        (target,),
    )
