# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Define model-independent checkpoint quantization profile contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol


class QuantizationFormat(StrEnum):
    """Identify one reusable ComfyUI tensor quantization format."""

    FP8_E4M3 = "float8_e4m3fn"
    FP8_E5M2 = "float8_e5m2"
    NVFP4 = "nvfp4"
    MXFP8 = "mxfp8"

    @property
    def label(self) -> str:
        """Return the concise format label used in diagnostics."""

        return {
            QuantizationFormat.FP8_E4M3: "FP8 E4M3",
            QuantizationFormat.FP8_E5M2: "FP8 E5M2",
            QuantizationFormat.NVFP4: "NVFP4",
            QuantizationFormat.MXFP8: "MXFP8",
        }[self]


@dataclass(frozen=True)
class QuantizationProfile:
    """Describe one workflow-facing, versioned per-tensor policy profile."""

    profile_id: str
    label: str
    version: int
    required_formats: frozenset[QuantizationFormat]

    @property
    def is_original(self) -> bool:
        """Return whether this profile loads the source checkpoint unchanged."""

        return not self.required_formats


@dataclass(frozen=True)
class TensorDescriptor:
    """Describe a checkpoint tensor without coupling policy to PyTorch."""

    name: str
    shape: tuple[int, ...]
    dtype_name: str


class ModelQuantizationRecipe(Protocol):
    """Assign model-specific per-tensor formats for named profiles."""

    @property
    def model_family(self) -> str:
        """Return the stable family identifier used in cache identity."""

    @property
    def version(self) -> int:
        """Return the recipe version used in cache invalidation."""

    @property
    def profiles(self) -> tuple[QuantizationProfile, ...]:
        """Return deterministic workflow profiles owned by this recipe."""

    def profile_from_selection(self, selection: str) -> QuantizationProfile:
        """Parse a workflow selection into a recipe-owned profile."""

    def policy_for(
        self,
        tensor: TensorDescriptor,
        profile: QuantizationProfile,
    ) -> QuantizationFormat | None:
        """Return the tensor format or ``None`` to preserve source precision."""
