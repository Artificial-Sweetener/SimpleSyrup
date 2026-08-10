# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for GPU-aware ComfyUI quantization capability discovery."""

from __future__ import annotations

from types import ModuleType

import pytest

from simple_syrup.domain.anima_quantization import (
    NVFP4_MIXED_PROFILE,
    AnimaQuantizationRecipe,
)
from simple_syrup.domain.model_quantization import QuantizationFormat
from simple_syrup.runtime.comfy_safetensors_dtypes import (
    ComfySafetensorsDtypeRegistry,
)
from simple_syrup.runtime.quantization_capabilities import (
    QuantizationCapabilityCatalog,
)


class FakeQuantOps(ModuleType):
    """Expose an injectable subset of ComfyUI's quantization registry."""

    def __init__(self, formats: tuple[str, ...]) -> None:
        """Register the requested fake formats."""

        super().__init__("comfy.quant_ops")
        self.QUANT_ALGOS = {name: object() for name in formats}


class FakeModelManagement(ModuleType):
    """Expose deterministic device capability decisions."""

    def __init__(self, *, fp8: bool, nvfp4: bool, mxfp8: bool) -> None:
        """Create a fake with explicit format support."""

        super().__init__("comfy.model_management")
        self._fp8 = fp8
        self._nvfp4 = nvfp4
        self._mxfp8 = mxfp8

    def get_torch_device(self) -> str:
        """Return a stable fake device."""

        return "cuda:0"

    def supports_fp8_compute(self, device: object) -> bool:
        """Return configured FP8 support."""

        del device
        return self._fp8

    def supports_nvfp4_compute(self, device: object) -> bool:
        """Return configured NVFP4 support."""

        del device
        return self._nvfp4

    def supports_mxfp8_compute(self, device: object) -> bool:
        """Return configured MXFP8 support."""

        del device
        return self._mxfp8


class FakeDtypeRegistry(ComfySafetensorsDtypeRegistry):
    """Expose deterministic loader compatibility decisions."""

    def __init__(
        self, unsupported: frozenset[QuantizationFormat] = frozenset()
    ) -> None:
        """Store formats that the fake loader cannot deserialize."""

        self._unsupported = unsupported

    def supports_format(self, quantization_format: QuantizationFormat) -> bool:
        """Return whether the fake loader supports one format."""

        return quantization_format not in self._unsupported


def test_catalog_intersects_gpu_support_with_comfy_registry() -> None:
    """Dropdown choices require both device support and a registered layout."""

    catalog = QuantizationCapabilityCatalog(
        FakeQuantOps(("float8_e4m3fn", "float8_e5m2", "nvfp4")),
        FakeModelManagement(fp8=True, nvfp4=True, mxfp8=True),
        FakeDtypeRegistry(),
    )

    assert catalog.available_formats() == frozenset(
        {
            QuantizationFormat.FP8_E4M3,
            QuantizationFormat.FP8_E5M2,
            QuantizationFormat.NVFP4,
        }
    )


def test_catalog_always_keeps_original_and_rejects_unavailable_format() -> None:
    """Users can always load the source model even without quant GPU support."""

    catalog = QuantizationCapabilityCatalog(
        FakeQuantOps(("nvfp4",)),
        FakeModelManagement(fp8=False, nvfp4=False, mxfp8=False),
        FakeDtypeRegistry(),
    )

    profiles = AnimaQuantizationRecipe().profiles
    assert catalog.selection_labels(profiles) == ["Original"]
    with pytest.raises(ValueError, match="unavailable on the current"):
        catalog.require_available(NVFP4_MIXED_PROFILE)


def test_catalog_hides_profile_when_loader_cannot_read_required_dtype() -> None:
    """Compute support does not advertise an unloadable MXFP8 checkpoint."""

    catalog = QuantizationCapabilityCatalog(
        FakeQuantOps(("mxfp8",)),
        FakeModelManagement(fp8=False, nvfp4=False, mxfp8=True),
        FakeDtypeRegistry(frozenset({QuantizationFormat.MXFP8})),
    )

    assert catalog.selection_labels(AnimaQuantizationRecipe().profiles) == ["Original"]
