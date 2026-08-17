# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Route installed Comfy model objects to regional capability detectors."""

from __future__ import annotations

from typing import Protocol

import comfy.model_patcher

from ..domain.regional_model_capabilities import RegionalModelCapabilities
from .anima_model_capability import AnimaModelCapabilityDetector
from .standard_unet_model_capability import StandardUnetModelCapabilityDetector


class _CapabilityDetector(Protocol):
    """Define one read-only capability detector collaborator."""

    def detect(
        self,
        patcher: comfy.model_patcher.ModelPatcher,
    ) -> RegionalModelCapabilities | None:
        """Return capabilities only when this detector owns the live surface."""


class RegionalModelCapabilityRegistry:
    """Own ordered defensive capability detection for supported model families."""

    def __init__(self) -> None:
        """Install exactly one authoritative detector per supported family."""

        self._detectors: tuple[_CapabilityDetector, ...] = (
            AnimaModelCapabilityDetector(),
            StandardUnetModelCapabilityDetector(),
        )

    def capabilities_for(self, model: object) -> RegionalModelCapabilities:
        """Return capabilities or report every observed unsupported object type."""

        if not isinstance(model, comfy.model_patcher.ModelPatcher):
            raise TypeError(
                "Regional attention requires a Comfy ModelPatcher; received "
                f"{_type_name(model)}."
            )
        for detector in self._detectors:
            capabilities = detector.detect(model)
            if capabilities is not None:
                return capabilities
        base_model = model.model
        diffusion_model = getattr(base_model, "diffusion_model", None)
        latent_format = getattr(base_model, "latent_format", None)
        raise ValueError(
            "Regional attention does not support this model combination: "
            f"patcher={_type_name(model)}, base_model={_type_name(base_model)}, "
            f"diffusion_model={_type_name(diffusion_model)}, "
            f"latent_format={_type_name(latent_format)}."
        )


def _type_name(value: object) -> str:
    """Return a diagnostic fully qualified runtime type name."""

    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"
