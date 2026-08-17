# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Own exact size and safe residency for static standard-UNet variants."""

from __future__ import annotations

import logging
from collections.abc import Callable

import torch
from comfy import model_management
from comfy.model_patcher import ModelPatcher
from comfy.patcher_extension import WrappersMP
from torch import nn

from ..model_patcher_mutations import ModelKeyedWrapperMutation
from .standard_unet_cold_diagnostics import (
    STANDARD_UNET_COLD_PATH_DIAGNOSTICS,
    StandardUnetColdStage,
)
from .standard_unet_variant_residency_handoff import (
    STANDARD_UNET_VARIANT_RESIDENCY_HANDOFF,
    StandardUnetVariantResidencyHandoff,
)

_WRAPPER_KEY = "simple_syrup.standard_unet_static_variant_residency"


class StandardUnetStaticVariantResidency:
    """Account for an enlarged static root and select native full loading."""

    def __init__(
        self,
        model: object,
        source_diffusion: object,
        variant_root: object,
        residency_handoff: StandardUnetVariantResidencyHandoff = (
            STANDARD_UNET_VARIANT_RESIDENCY_HANDOFF
        ),
    ) -> None:
        """Measure the complete replacement while its source is still visible."""

        if not isinstance(model, ModelPatcher):
            raise TypeError("Static variant residency requires a MODEL.")
        if model.is_dynamic():
            raise ValueError("Static variant residency requires a static MODEL.")
        if not isinstance(source_diffusion, nn.Module) or not isinstance(
            variant_root, nn.Module
        ):
            raise TypeError("Static variant residency requires diffusion modules.")
        if not isinstance(
            residency_handoff,
            StandardUnetVariantResidencyHandoff,
        ):
            raise TypeError("Static variant residency requires a handoff owner.")
        required_bytes = (
            model.model_size()
            - model_management.module_size(source_diffusion)
            + model_management.module_size(variant_root)
        )
        if not isinstance(required_bytes, int) or required_bytes <= 0:
            raise ValueError("Static variant root has no measurable weights.")
        self._required_bytes = required_bytes
        self._residency_handoff = residency_handoff

    def apply(self, model: object) -> None:
        """Publish exact size and install one prepare-sampling residency wrapper."""

        if not isinstance(model, ModelPatcher):
            raise TypeError("Static variant residency requires a MODEL.")
        if model.is_dynamic():
            raise ValueError("Static variant residency requires a static MODEL.")
        model.size = self._required_bytes
        ModelKeyedWrapperMutation(
            WrappersMP.PREPARE_SAMPLING,
            _WRAPPER_KEY,
            self.prepare_sampling,
        ).apply(model)

    def prepare_sampling(
        self,
        executor: Callable[..., object],
        model: object,
        noise_shape: object,
        conds: object,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Request full static loading only within Comfy's device budget."""

        if not isinstance(model, ModelPatcher) or model.is_dynamic():
            raise TypeError("Static variant residency requires a static MODEL.")
        self._residency_handoff.activate(model)
        force_offload = kwargs.get("force_offload", False)
        force_full_load = kwargs.get("force_full_load", False)
        if not isinstance(force_offload, bool) or not isinstance(force_full_load, bool):
            raise TypeError("Comfy sampling residency flags must be bool values.")
        eligible = self._is_eligible(model, force_offload=force_offload)
        forwarded = kwargs.copy()
        forwarded["force_full_load"] = force_full_load or eligible
        if eligible and not force_full_load:
            logging.info(
                "Standard UNet regional variants requesting static full residency",
                extra={
                    "operation": "prepare_sampling",
                    "required_bytes": self._required_bytes,
                    "device": str(model.load_device),
                },
            )
        device = model.load_device
        measured_device = device if isinstance(device, torch.device) else None
        with STANDARD_UNET_COLD_PATH_DIAGNOSTICS.measure(
            StandardUnetColdStage.MODEL_RESIDENCY,
            device=measured_device,
        ) as metadata:
            result = executor(model, noise_shape, conds, *args, **forwarded)
            metadata["required_bytes"] = self._required_bytes
            metadata["force_full_load"] = force_full_load or eligible
            metadata["force_offload"] = force_offload
        return result

    def _is_eligible(
        self,
        model: ModelPatcher,
        *,
        force_offload: bool,
    ) -> bool:
        """Return whether the exact enlarged root fits Comfy's safe capacity."""

        device = model.load_device
        if force_offload or not isinstance(device, torch.device):
            return False
        if device.type == "cpu":
            return False
        capacity = model_management.maximum_vram_for_weights(device)
        return isinstance(capacity, int | float) and self._required_bytes <= capacity

    @property
    def required_bytes(self) -> int:
        """Return the exact patcher size published to Comfy."""

        return self._required_bytes
