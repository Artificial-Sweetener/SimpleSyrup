# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Define rendered attention evidence before component and matte shaping."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class AttentionConceptEvidence:
    """Hold one concept's alpha, support, and confidence evidence."""

    label: str
    alpha: torch.Tensor
    support: torch.Tensor
    confidence: torch.Tensor
