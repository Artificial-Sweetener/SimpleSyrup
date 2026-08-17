# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Select the conventional Comfy device for standard-UNet patch calculation."""

from __future__ import annotations

import torch
from comfy.model_patcher import ModelPatcher


class StandardUnetVariantMaterializationDevice:
    """Resolve exact patch calculation placement from one Comfy MODEL contract."""

    @staticmethod
    def resolve(model: object) -> torch.device:
        """Return the validated device Comfy requests for loaded model weights."""

        if not isinstance(model, ModelPatcher):
            raise TypeError("Standard UNet materialization device requires a MODEL.")
        device = model.load_device
        if not isinstance(device, torch.device):
            raise TypeError("Standard UNet MODEL load device must be a torch device.")
        if device.type == "meta":
            raise ValueError("Standard UNet variants cannot materialize on meta.")
        return device


STANDARD_UNET_VARIANT_MATERIALIZATION_DEVICE = (
    StandardUnetVariantMaterializationDevice()
)
