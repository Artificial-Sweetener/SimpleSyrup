# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Reject exact static-global and admitted-regional Anima LoRA overlap."""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from comfy.weight_adapter.lora import LoRAAdapter

from .anima_plan_admission import (
    AnimaRegionalLoraAdapterAdmission,
    AnimaRegionalLoraPlanAdmission,
)
from .standard_adapter import StandardLoraTarget


class AnimaGlobalRegionalLoraOverlapError(ValueError):
    """Report regional adapters already present in the static global MODEL."""


class AnimaGlobalRegionalLoraOverlapValidator:
    """Compare exact admitted regional A/B tensors with static global patches."""

    def validate(
        self,
        model: object,
        admission: AnimaRegionalLoraPlanAdmission,
    ) -> None:
        """Reject every regional adapter whose complete content is global."""

        if not isinstance(admission, AnimaRegionalLoraPlanAdmission):
            raise TypeError("Anima global LoRA overlap requires an admitted plan.")
        patches = getattr(model, "patches", None)
        if not isinstance(patches, Mapping):
            raise TypeError("Anima global LoRA overlap requires MODEL patches.")
        duplicates = tuple(
            adapter.adapter_plan.adapter_identity.value
            for adapter in admission.adapters
            if self._duplicates_global_content(patches, adapter)
        )
        unique_duplicates = tuple(dict.fromkeys(duplicates))
        if unique_duplicates:
            identities = ", ".join(repr(value) for value in unique_duplicates)
            raise AnimaGlobalRegionalLoraOverlapError(
                "Regional Anima LoRA content is already applied globally to the "
                f"input MODEL: {identities}. Remove either the global or regional "
                "application before sampling."
            )

    def _duplicates_global_content(
        self,
        patches: Mapping[object, object],
        adapter: AnimaRegionalLoraAdapterAdmission,
    ) -> bool:
        """Return whether every admitted regional target has an exact global pair."""

        targets = adapter.admission.targets
        return bool(targets) and all(
            self._target_matches(patches, target.adapter) for target in targets
        )

    def _target_matches(
        self,
        patches: Mapping[object, object],
        regional: StandardLoraTarget,
    ) -> bool:
        """Match one regional target against nonzero comparable static patches."""

        key = f"{regional.target}.weight"
        entries = patches.get(key, ())
        if entries == ():
            return False
        if not isinstance(entries, list):
            raise TypeError(f"MODEL patches[{key!r}] must be a list.")
        for index, entry in enumerate(entries):
            if not isinstance(entry, tuple) or len(entry) < 3:
                raise TypeError(
                    f"MODEL patches[{key!r}][{index}] must be a Comfy patch tuple."
                )
            if _nonzero_strength(entry[0], key=key, index=index) and _matches_pair(
                entry[1],
                regional,
            ):
                return True
        return False


def _nonzero_strength(value: object, *, key: str, index: int) -> bool:
    """Validate one installed static patch strength and report its activity."""

    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"MODEL patches[{key!r}][{index}] strength must be numeric.")
    strength = float(value)
    if not math.isfinite(strength):
        raise ValueError(f"MODEL patches[{key!r}][{index}] strength must be finite.")
    return strength != 0.0


def _matches_pair(value: object, regional: StandardLoraTarget) -> bool:
    """Compare one installed standard LoRA patch without copies or transfers."""

    if not isinstance(value, LoRAAdapter):
        return False
    weights = value.weights
    if not isinstance(weights, tuple) or len(weights) != 6:
        return False
    up, down, alpha, mid, dora_scale, reshape = weights
    if any(item is not None for item in (alpha, mid, dora_scale, reshape)):
        return False
    if not isinstance(down, torch.Tensor) or not isinstance(up, torch.Tensor):
        return False
    return _same_tensor(down, regional.down) and _same_tensor(up, regional.up)


def _same_tensor(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Use an identity fast path before exact same-residency tensor equality."""

    if left is right:
        return True
    if (
        left.shape != right.shape
        or left.dtype != right.dtype
        or left.device != right.device
    ):
        return False
    return bool(torch.equal(left, right))


ANIMA_GLOBAL_REGIONAL_LORA_OVERLAP_VALIDATOR = AnimaGlobalRegionalLoraOverlapValidator()
