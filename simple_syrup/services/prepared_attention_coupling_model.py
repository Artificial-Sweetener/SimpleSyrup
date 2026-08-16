# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Define the complete output of Attention Coupling model preparation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..domain.regional_mask_bank import RegionalMaskBank


@dataclass(frozen=True, slots=True)
class PreparedAttentionCouplingModel:
    """Retain a derived model, base conditioning pair, and canonical mask bank."""

    model: Any
    positive: object
    negative: object
    mask_bank: RegionalMaskBank
