# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for guarded Comfy safetensors dtype compatibility."""

from __future__ import annotations

from types import ModuleType

import torch

from simple_syrup.domain.model_quantization import QuantizationFormat
from simple_syrup.runtime.comfy_safetensors_dtypes import (
    ComfySafetensorsDtypeRegistry,
)


def _comfy_types(types: dict[str, object]) -> ModuleType:
    """Create a fake Comfy utils module with a private dtype registry."""

    module = ModuleType("comfy.utils")
    module._TYPES = types  # type: ignore[attr-defined]
    return module


def test_registry_adds_e8m0_when_host_supports_it() -> None:
    """SimpleSyrup fills the missing mapping without replacing host entries."""

    types: dict[str, object] = {"F8_E4M3": torch.float8_e4m3fn}
    registry = ComfySafetensorsDtypeRegistry(_comfy_types(types))

    assert registry.register_extension_dtypes()
    assert types["F8_E8M0"] == torch.float8_e8m0fnu
    assert registry.supports_format(QuantizationFormat.MXFP8)


def test_registry_accepts_identical_existing_mapping() -> None:
    """A host-provided identical mapping is an idempotent success."""

    types: dict[str, object] = {
        "F8_E4M3": torch.float8_e4m3fn,
        "F8_E8M0": torch.float8_e8m0fnu,
    }

    assert ComfySafetensorsDtypeRegistry(
        _comfy_types(types)
    ).register_extension_dtypes()


def test_registry_rejects_conflict_without_overwriting_it() -> None:
    """A conflicting host mapping fails closed and remains untouched."""

    conflict = object()
    types = {"F8_E4M3": torch.float8_e4m3fn, "F8_E8M0": conflict}
    registry = ComfySafetensorsDtypeRegistry(_comfy_types(types))

    assert not registry.register_extension_dtypes()
    assert not registry.supports_format(QuantizationFormat.MXFP8)
    assert types["F8_E8M0"] is conflict
