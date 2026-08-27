# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Select cohesive concept support from continuous attention evidence."""

from __future__ import annotations

import torch

from ..masking.mask_components import connected_mask_components

WEAK_SUPPORT_RATIO = 0.1


class AttentionConceptSupportService:
    """Retain faint concept extent only when anchored to strong evidence."""

    def select(
        self,
        alpha: torch.Tensor,
        minimum_strength: float,
    ) -> torch.Tensor:
        """Return topology-gated support without changing retained alpha values."""

        if not isinstance(alpha, torch.Tensor) or alpha.ndim != 2:
            raise ValueError("Concept attention alpha must be an HW tensor.")
        if not 0.0 <= minimum_strength <= 1.0:
            raise ValueError("Concept minimum strength must be between 0 and 1.")
        positive = alpha > 0.0
        if minimum_strength <= 0.0:
            return positive
        strong = alpha >= minimum_strength
        if not strong.any().item():
            return torch.zeros_like(positive)
        weak = alpha >= minimum_strength * WEAK_SUPPORT_RATIO
        strong_cpu = strong.detach().to(device="cpu")
        retained = torch.zeros_like(strong_cpu)
        for component in connected_mask_components(weak):
            if (component.mask & strong_cpu).any().item():
                retained |= component.mask
        return retained.to(device=alpha.device)


ATTENTION_CONCEPT_SUPPORT_SERVICE = AttentionConceptSupportService()
