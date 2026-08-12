# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Validate model-consumed standard-UNet cross-attention geometry."""

from __future__ import annotations

import torch


class StandardUnetRegionalContextValidator:
    """Admit positive B/S/D context geometry without Anima-specific constants."""

    def validate(
        self,
        context: torch.Tensor,
        *,
        prompt_type: str,
        conditioning_index: int,
    ) -> None:
        """Reject empty standard-UNet token or feature dimensions."""

        if int(context.shape[1]) < 1 or int(context.shape[2]) < 1:
            raise ValueError(
                f"{prompt_type} conditioning {conditioning_index} c_crossattn "
                "must contain positive token and feature dimensions."
            )


STANDARD_UNET_REGIONAL_CONTEXT_VALIDATOR = StandardUnetRegionalContextValidator()
