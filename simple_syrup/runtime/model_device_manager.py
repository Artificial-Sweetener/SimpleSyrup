# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Comfy-aware device management for SimpleSyrup-owned torch models."""

from __future__ import annotations

import importlib
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any, cast

import torch

from ..shared.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass
class ManagedTorchModel:
    """Track one SimpleSyrup-owned model under Comfy's memory manager."""

    model: object
    model_id: str
    source: str
    patcher: object | None = field(default=None, init=False, repr=False)
    load_device: torch.device | None = field(default=None, init=False)
    offload_device: torch.device | None = field(default=None, init=False)


@dataclass(frozen=True)
class LoadedManagedModel:
    """Expose a model loaded for one inference call."""

    model: object
    device: torch.device
    policy: str


class TorchModelDeviceManager:
    """Load SimpleSyrup-owned torch models through Comfy's VRAM policy."""

    def manage(self, model: object, model_id: str, source: str) -> ManagedTorchModel:
        """Return a managed handle for a raw model without moving it to CUDA."""

        _eval_model(model)
        return ManagedTorchModel(model=model, model_id=model_id, source=source)

    @contextmanager
    def inference(
        self,
        managed_model: ManagedTorchModel,
        execution_device: str,
    ) -> Iterator[LoadedManagedModel]:
        """Yield the model loaded on the device selected by `execution_device`."""

        policy = _validate_execution_device(execution_device)
        if policy == "cpu":
            device = torch.device("cpu")
            _move_model(managed_model.model, device)
            _eval_model(managed_model.model)
            LOGGER.debug(
                "Torch model prepared for CPU inference",
                extra={
                    "operation": "torch_model_device_manager",
                    "model": managed_model.model_id,
                    "policy": policy,
                    "device": str(device),
                    "source": managed_model.source,
                },
            )
            yield LoadedManagedModel(
                model=managed_model.model,
                device=device,
                policy=policy,
            )
            return

        comfy_model_management, comfy_model_patcher = _comfy_modules()
        load_device = torch.device(comfy_model_management.get_torch_device())
        if load_device.type == "cpu":
            _move_model(managed_model.model, load_device)
            _eval_model(managed_model.model)
            yield LoadedManagedModel(
                model=managed_model.model,
                device=load_device,
                policy=policy,
            )
            return

        offload_device = _resolve_offload_device(comfy_model_management)
        if not _supports_comfy_model_patcher(managed_model.model):
            LOGGER.debug(
                "Torch model uses bounded device movement outside ModelPatcher",
                extra={
                    "operation": "torch_model_device_manager",
                    "model": managed_model.model_id,
                    "policy": policy,
                    "load_device": str(load_device),
                    "offload_device": str(offload_device),
                    "source": managed_model.source,
                },
            )
            _move_model(managed_model.model, load_device)
            _eval_model(managed_model.model)
            try:
                yield LoadedManagedModel(
                    model=managed_model.model,
                    device=load_device,
                    policy=policy,
                )
            finally:
                _move_model(managed_model.model, offload_device)
                _soft_empty_cache(comfy_model_management)
            return

        patcher = self._patcher_for(
            managed_model,
            comfy_model_patcher,
            load_device,
            offload_device,
        )
        LOGGER.debug(
            "Loading torch model through Comfy model manager",
            extra={
                "operation": "torch_model_device_manager",
                "model": managed_model.model_id,
                "policy": policy,
                "load_device": str(load_device),
                "offload_device": str(offload_device),
                "source": managed_model.source,
            },
        )
        comfy_model_management.load_model_gpu(patcher)
        _eval_model(managed_model.model)
        yield LoadedManagedModel(
            model=managed_model.model,
            device=load_device,
            policy=policy,
        )

    def _patcher_for(
        self,
        managed_model: ManagedTorchModel,
        comfy_model_patcher: ModuleType,
        load_device: torch.device,
        offload_device: torch.device,
    ) -> object:
        """Return a reusable `ModelPatcher` for the requested devices."""

        if (
            managed_model.patcher is not None
            and managed_model.load_device == load_device
            and managed_model.offload_device == offload_device
        ):
            return managed_model.patcher

        patcher_class = cast(Any, comfy_model_patcher).ModelPatcher
        managed_model.patcher = patcher_class(
            managed_model.model,
            load_device,
            offload_device,
        )
        managed_model.load_device = load_device
        managed_model.offload_device = offload_device
        return managed_model.patcher


def resolve_execution_device(execution_device: str) -> torch.device:
    """Resolve a public SimpleSyrup execution device policy."""

    policy = _validate_execution_device(execution_device)
    if policy == "cpu":
        return torch.device("cpu")
    comfy_model_management = importlib.import_module("comfy.model_management")
    return torch.device(comfy_model_management.get_torch_device())


@contextmanager
def external_model_inference(
    model: object,
    execution_device: str,
) -> Iterator[LoadedManagedModel]:
    """Move a compatible external raw model for one bounded inference call."""

    device = resolve_execution_device(execution_device)
    original_device = _model_device(model)
    _move_model(model, device)
    _eval_model(model)
    try:
        yield LoadedManagedModel(model=model, device=device, policy=execution_device)
    finally:
        if original_device is not None and original_device != device:
            _move_model(model, original_device)


def _validate_execution_device(execution_device: str) -> str:
    """Return a normalized execution policy or fail clearly."""

    if execution_device in {"auto", "cpu"}:
        return execution_device
    raise ValueError("execution_device must be 'auto' or 'cpu'.")


def _comfy_modules() -> tuple[ModuleType, ModuleType]:
    """Import Comfy model-management modules lazily."""

    return (
        importlib.import_module("comfy.model_management"),
        importlib.import_module("comfy.model_patcher"),
    )


def _resolve_offload_device(comfy_model_management: ModuleType) -> torch.device:
    """Return Comfy's preferred offload device for auxiliary torch modules."""

    offload_device = cast(Any, comfy_model_management).text_encoder_offload_device()
    return torch.device(offload_device)


def _supports_comfy_model_patcher(model: object) -> bool:
    """Return whether Comfy's patcher can assign `model.device` safely."""

    class_device = getattr(type(model), "device", None)
    if isinstance(class_device, property) and class_device.fset is None:
        return False
    return True


def _soft_empty_cache(comfy_model_management: ModuleType) -> None:
    """Ask Comfy to release cached memory after bounded manual offload."""

    soft_empty_cache = getattr(comfy_model_management, "soft_empty_cache", None)
    if callable(soft_empty_cache):
        soft_empty_cache()


def _move_model(model: object, device: torch.device) -> None:
    """Move a PyTorch-style model when it exposes `.to(...)`."""

    to_method = getattr(model, "to", None)
    if callable(to_method):
        to_method(device)


def _eval_model(model: object) -> None:
    """Set eval mode when the model exposes `.eval()`."""

    eval_method = getattr(model, "eval", None)
    if callable(eval_method):
        eval_method()


def _model_device(model: object) -> torch.device | None:
    """Return the first known torch device for a raw model, if any."""

    device = getattr(model, "device", None)
    if device is not None:
        return torch.device(device)
    parameters = getattr(model, "parameters", None)
    if callable(parameters):
        try:
            first_parameter = next(iter(parameters()))
        except StopIteration:
            return None
        except TypeError:
            return None
        if isinstance(first_parameter, torch.Tensor):
            return first_parameter.device
        parameter_device = getattr(first_parameter, "device", None)
        if parameter_device is not None:
            return torch.device(parameter_device)
    return None
