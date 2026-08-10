# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Discover quantization profiles executable and reloadable by active ComfyUI."""

from __future__ import annotations

import importlib
from collections.abc import Iterable
from types import ModuleType
from typing import Any

from ..domain.model_quantization import QuantizationFormat, QuantizationProfile
from ..shared.logging import get_logger
from .comfy_safetensors_dtypes import ComfySafetensorsDtypeRegistry

LOGGER = get_logger(__name__)


class QuantizationCapabilityCatalog:
    """Intersect profile requirements with GPU, kernels, and checkpoint loading."""

    def __init__(
        self,
        quant_ops_module: ModuleType | None = None,
        model_management_module: ModuleType | None = None,
        dtype_registry: ComfySafetensorsDtypeRegistry | None = None,
    ) -> None:
        """Create a catalog with injectable ComfyUI runtime boundaries."""

        self._quant_ops_module = quant_ops_module
        self._model_management_module = model_management_module
        self._dtype_registry = dtype_registry or ComfySafetensorsDtypeRegistry()
        self._cached_formats: frozenset[QuantizationFormat] | None = None

    def available_formats(self) -> frozenset[QuantizationFormat]:
        """Return formats supported through compute and checkpoint reload."""

        if self._cached_formats is not None:
            return self._cached_formats
        formats: set[QuantizationFormat] = set()
        try:
            quant_ops = self._quant_ops()
            model_management = self._model_management()
            registered = quant_ops.QUANT_ALGOS
            device = model_management.get_torch_device()
            if model_management.supports_fp8_compute(device):
                self._append_registered(
                    formats, registered, QuantizationFormat.FP8_E4M3
                )
                self._append_registered(
                    formats, registered, QuantizationFormat.FP8_E5M2
                )
            if model_management.supports_nvfp4_compute(device):
                self._append_registered(formats, registered, QuantizationFormat.NVFP4)
            if model_management.supports_mxfp8_compute(device):
                self._append_registered(formats, registered, QuantizationFormat.MXFP8)
        except (
            AttributeError,
            ImportError,
            OSError,
            RuntimeError,
            ValueError,
        ) as error:
            LOGGER.warning(
                "quantization capability discovery failed",
                extra={"reason": str(error)},
            )
        self._cached_formats = frozenset(formats)
        return self._cached_formats

    def available_profiles(
        self,
        profiles: Iterable[QuantizationProfile],
    ) -> tuple[QuantizationProfile, ...]:
        """Return profiles whose complete format set is available."""

        available = self.available_formats()
        return tuple(
            profile
            for profile in profiles
            if profile.is_original or profile.required_formats.issubset(available)
        )

    def selection_labels(self, profiles: Iterable[QuantizationProfile]) -> list[str]:
        """Return workflow labels in recipe-defined preference order."""

        return [profile.label for profile in self.available_profiles(profiles)]

    def require_available(self, profile: QuantizationProfile) -> None:
        """Reject a profile unavailable to the current ComfyUI runtime."""

        missing = profile.required_formats.difference(self.available_formats())
        if not missing:
            return
        missing_labels = ", ".join(sorted(item.label for item in missing))
        raise ValueError(
            f"{profile.label} is unavailable on the current ComfyUI GPU/runtime. "
            f"Missing format support: {missing_labels}."
        )

    def _quant_ops(self) -> Any:
        """Return ComfyUI's quantization operation module."""

        return self._quant_ops_module or importlib.import_module("comfy.quant_ops")

    def _model_management(self) -> Any:
        """Return ComfyUI's device capability module."""

        return self._model_management_module or importlib.import_module(
            "comfy.model_management"
        )

    def _append_registered(
        self,
        formats: set[QuantizationFormat],
        registered: object,
        candidate: QuantizationFormat,
    ) -> None:
        """Append only when Comfy registers, computes, and reloads the format."""

        if (
            isinstance(registered, dict)
            and candidate.value in registered
            and self._dtype_registry.supports_format(candidate)
        ):
            formats.add(candidate)
