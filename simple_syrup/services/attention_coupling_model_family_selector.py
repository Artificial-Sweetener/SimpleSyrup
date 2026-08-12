# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Select one model-family adapter from admitted Attention Coupling capabilities."""

from __future__ import annotations

from typing import ClassVar

from ..domain.regional_model_capabilities import (
    RegionalAttentionBackend,
    RegionalModelCapabilities,
)
from .anima_attention_coupling_model_family import (
    AnimaAttentionCouplingModelFamily,
)
from .attention_coupling_model_family import AttentionCouplingModelFamily
from .unet_attention_coupling_model_family import (
    StandardUnetAttentionCouplingModelFamily,
)


class AttentionCouplingModelFamilySelector:
    """Map one already-admitted backend enum to its focused family adapter."""

    anima_family_class: ClassVar[type[AnimaAttentionCouplingModelFamily]] = (
        AnimaAttentionCouplingModelFamily
    )
    unet_family_class: ClassVar[type[StandardUnetAttentionCouplingModelFamily]] = (
        StandardUnetAttentionCouplingModelFamily
    )

    def select(
        self,
        capabilities: RegionalModelCapabilities,
    ) -> AttentionCouplingModelFamily:
        """Return the exact family adapter named by central admission."""

        if not isinstance(capabilities, RegionalModelCapabilities):
            raise TypeError(
                "Attention Coupling family selection requires model capabilities."
            )
        if (
            capabilities.attention_backend
            is RegionalAttentionBackend.ANIMA_OBJECT_PATCH
        ):
            return self.anima_family_class()
        if capabilities.attention_backend is RegionalAttentionBackend.UNET_ATTN2_PATCH:
            return self.unet_family_class()
        raise ValueError(
            "Attention Coupling capabilities name an unsupported backend: "
            f"{capabilities.attention_backend!r}."
        )
