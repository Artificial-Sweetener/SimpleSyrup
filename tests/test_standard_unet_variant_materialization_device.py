# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify capability-routed standard-UNet materialization placement."""

from __future__ import annotations

import pytest
import torch
from comfy.model_patcher import ModelPatcher
from torch import nn

from simple_syrup.runtime.regional_lora import (
    standard_unet_variant_materialization_device as materialization_device,
)


def test_materialization_device_uses_comfy_model_load_device() -> None:
    """Select the requested accelerator independently of source parameter device."""

    model = ModelPatcher(
        nn.Linear(2, 2, bias=False),
        torch.device("cuda"),
        torch.device("cpu"),
    )

    assert materialization_device.StandardUnetVariantMaterializationDevice.resolve(
        model
    ) == torch.device("cuda")
    assert next(model.model.parameters()).device == torch.device("cpu")


def test_materialization_device_preserves_cpu_execution() -> None:
    """Keep CPU-only Comfy models on their existing calculation device."""

    model = ModelPatcher(
        nn.Linear(2, 2, bias=False),
        torch.device("cpu"),
        torch.device("cpu"),
    )

    assert materialization_device.StandardUnetVariantMaterializationDevice.resolve(
        model
    ) == torch.device("cpu")


def test_materialization_device_rejects_invalid_model_contract() -> None:
    """Fail closed for absent, malformed, and non-materializable devices."""

    with pytest.raises(TypeError, match="requires a MODEL"):
        materialization_device.StandardUnetVariantMaterializationDevice.resolve(
            object()
        )

    model = ModelPatcher(
        nn.Linear(2, 2, bias=False),
        torch.device("cpu"),
        torch.device("cpu"),
    )
    model.load_device = "cuda"
    with pytest.raises(TypeError, match="must be a torch device"):
        materialization_device.StandardUnetVariantMaterializationDevice.resolve(model)

    model.load_device = torch.device("meta")
    with pytest.raises(ValueError, match="cannot materialize on meta"):
        materialization_device.StandardUnetVariantMaterializationDevice.resolve(model)
