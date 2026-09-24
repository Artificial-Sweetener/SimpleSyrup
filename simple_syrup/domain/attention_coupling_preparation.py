# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Define validated Attention Coupling preparation data."""

from __future__ import annotations

from dataclasses import dataclass

from .raw_regional_attention import RawRegionalAttentionPlan


@dataclass(frozen=True, slots=True)
class AttentionCouplingPreparation:
    """Retain the full raw plan and base-only ordinary sampler inputs."""

    plan: RawRegionalAttentionPlan
    positive: object
    negative: object
