# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Coordinate device residency between distinct standard-UNet variant roots."""

from __future__ import annotations

import logging
from collections.abc import Callable
from threading import Lock
from weakref import ReferenceType, ref

import torch
from comfy import model_management
from comfy.model_patcher import ModelPatcher

ModelUnloader = Callable[..., None]


class StandardUnetVariantResidencyHandoff:
    """Keep only one separately allocated regional root active per device."""

    def __init__(
        self,
        unload_model: ModelUnloader = model_management.unload_model_and_clones,
    ) -> None:
        """Retain the host unload boundary and an empty device registry."""

        if not callable(unload_model):
            raise TypeError("Standard UNet residency handoff requires an unloader.")
        self._unload_model = unload_model
        self._active: dict[torch.device, ReferenceType[ModelPatcher]] = {}
        self._lock = Lock()

    def activate(self, model: object) -> bool:
        """Unload a different retained root before activating this allocation."""

        if not isinstance(model, ModelPatcher):
            raise TypeError("Standard UNet residency handoff requires a MODEL.")
        if model.is_dynamic():
            raise ValueError("Standard UNet residency handoff requires a static MODEL.")
        device = model.load_device
        if not isinstance(device, torch.device):
            raise TypeError("Standard UNet residency handoff requires a torch device.")
        if device.type == "cpu":
            return False

        with self._lock:
            previous_reference = self._active.get(device)
            previous = None if previous_reference is None else previous_reference()
            self._active[device] = ref(model)
            if previous is None or previous.model is model.model:
                return False
            self._unload_model(
                previous,
                unload_additional_models=False,
                all_devices=False,
            )

        logging.info(
            "Standard UNet regional variant residency handed off",
            extra={
                "operation": "standard_unet_variant_residency_handoff",
                "device": str(device),
            },
        )
        return True

    def clear(self) -> None:
        """Forget retained identities without unloading host-owned models."""

        with self._lock:
            self._active.clear()


STANDARD_UNET_VARIANT_RESIDENCY_HANDOFF = StandardUnetVariantResidencyHandoff()
