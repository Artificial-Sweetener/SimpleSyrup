# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Define the model-family processed-context validation boundary."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class RegionalContextValidator(Protocol):
    """Validate one model-consumed cross-attention tensor for a backend."""

    def validate(
        self,
        context: torch.Tensor,
        *,
        prompt_type: str,
        conditioning_index: int,
    ) -> None:
        """Reject context state incompatible with the selected backend."""
