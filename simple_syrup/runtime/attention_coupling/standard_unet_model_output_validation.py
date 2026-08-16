# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Validate one completed standard-UNet denoiser result."""

from __future__ import annotations

import torch


class StandardUnetModelOutputValidator:
    """Own the single recoverable device-value observation per model call."""

    @staticmethod
    def validate(output: object, *, model_input: torch.Tensor) -> torch.Tensor:
        """Return a finite shape-aligned result or fail after execution."""

        if not isinstance(model_input, torch.Tensor):
            raise TypeError("Standard UNet model input must be a tensor.")
        if not isinstance(output, torch.Tensor):
            raise TypeError("Standard UNet must return a tensor.")
        if not output.is_floating_point():
            raise TypeError("Standard UNet output must use floating point.")
        if output.device != model_input.device:
            raise ValueError("Standard UNet output must remain on the input device.")
        if not bool(torch.isfinite(output).all()):
            raise ValueError("Standard UNet output contains non-finite values.")
        return output


STANDARD_UNET_MODEL_OUTPUT_VALIDATOR = StandardUnetModelOutputValidator()
