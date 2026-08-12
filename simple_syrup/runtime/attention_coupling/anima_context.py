# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Validate model-consumed Anima cross-attention context geometry."""

from __future__ import annotations

import torch

ANIMA_CONTEXT_SEQUENCE_LENGTH = 512
ANIMA_CONTEXT_FEATURE_WIDTH = 1024


class AnimaRegionalContextValidator:
    """Require Anima's exact semantic token count and feature width."""

    def validate(
        self,
        context: torch.Tensor,
        *,
        prompt_type: str,
        conditioning_index: int,
    ) -> None:
        """Reject incompatible Anima context output without token repetition."""

        if int(context.shape[1]) != ANIMA_CONTEXT_SEQUENCE_LENGTH:
            raise ValueError(
                f"{prompt_type} conditioning {conditioning_index} c_crossattn "
                f"must contain exactly {ANIMA_CONTEXT_SEQUENCE_LENGTH} semantic "
                f"tokens; observed {int(context.shape[1])}. Tokens are not repeated "
                "to force alignment."
            )
        if int(context.shape[2]) != ANIMA_CONTEXT_FEATURE_WIDTH:
            raise ValueError(
                f"{prompt_type} conditioning {conditioning_index} c_crossattn "
                f"must use feature width {ANIMA_CONTEXT_FEATURE_WIDTH}; observed "
                f"{int(context.shape[2])}."
            )


ANIMA_REGIONAL_CONTEXT_VALIDATOR = AnimaRegionalContextValidator()
