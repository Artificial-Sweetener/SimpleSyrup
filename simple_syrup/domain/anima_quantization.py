# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Define versioned, quality-aware Anima quantization profiles."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .model_quantization import (
    QuantizationFormat,
    QuantizationProfile,
    TensorDescriptor,
)

ORIGINAL_PROFILE = QuantizationProfile("original", "Original", 2, frozenset())
FP8_E4M3_PROFILE = QuantizationProfile(
    "fp8-e4m3",
    "FP8 E4M3",
    2,
    frozenset({QuantizationFormat.FP8_E4M3}),
)
FP8_E5M2_PROFILE = QuantizationProfile(
    "fp8-e5m2",
    "FP8 E5M2",
    2,
    frozenset({QuantizationFormat.FP8_E5M2}),
)
MXFP8_PROFILE = QuantizationProfile(
    "mxfp8",
    "MXFP8",
    2,
    frozenset({QuantizationFormat.MXFP8}),
)
NVFP4_MIXED_PROFILE = QuantizationProfile(
    "nvfp4-mixed",
    "NVFP4 (Mixed)",
    3,
    frozenset({QuantizationFormat.FP8_E4M3, QuantizationFormat.NVFP4}),
)
_PROFILES = (
    ORIGINAL_PROFILE,
    FP8_E4M3_PROFILE,
    FP8_E5M2_PROFILE,
    MXFP8_PROFILE,
    NVFP4_MIXED_PROFILE,
)
_MAIN_BLOCK_PATTERN = re.compile(
    r"(?:^|\.)(?:net|diffusion_model)\.blocks\.(?P<index>\d+)\."
)
_PROTECTED_BLOCKS = {0, 1, 27}
_FLOAT_DTYPES = {"F16", "BF16", "F32", "F64"}


@dataclass(frozen=True)
class AnimaQuantizationRecipe:
    """Assign formats only within Anima's quality-safe DiT block envelope."""

    model_family: str = "Anima"
    version: int = 2

    @property
    def profiles(self) -> tuple[QuantizationProfile, ...]:
        """Return Anima's stable workflow-facing profile order."""

        return _PROFILES

    def profile_from_selection(self, selection: str) -> QuantizationProfile:
        """Parse one current workflow selection into its profile."""

        for profile in self.profiles:
            if selection in (profile.label, profile.profile_id):
                return profile
        valid = ", ".join(profile.label for profile in self.profiles)
        raise ValueError(f"quantization profile must be one of: {valid}.")

    def policy_for(
        self,
        tensor: TensorDescriptor,
        profile: QuantizationProfile,
    ) -> QuantizationFormat | None:
        """Return Anima's per-tensor format while preserving sensitive layers."""

        if profile not in self.profiles:
            raise ValueError(
                f"Unknown Anima quantization profile '{profile.profile_id}'."
            )
        if profile.is_original or not _is_matrix_weight(tensor):
            return None
        if "llm_adapter" in tensor.name or "adaln_modulation" in tensor.name:
            return None
        block_match = _MAIN_BLOCK_PATTERN.search(tensor.name)
        if block_match is None:
            return None
        if int(block_match.group("index")) in _PROTECTED_BLOCKS:
            return None
        if profile.profile_id == NVFP4_MIXED_PROFILE.profile_id:
            if "v_proj" in tensor.name or ".mlp." in tensor.name:
                return QuantizationFormat.FP8_E4M3
            if any(
                projection in tensor.name
                for projection in ("q_proj", "k_proj", "output_proj")
            ):
                return QuantizationFormat.NVFP4
            return None
        return next(iter(profile.required_formats))


def _is_matrix_weight(tensor: TensorDescriptor) -> bool:
    """Return whether a tensor is an eligible floating-point matrix weight."""

    return (
        tensor.dtype_name in _FLOAT_DTYPES
        and len(tensor.shape) == 2
        and tensor.name.endswith(".weight")
    )
