# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Own ordered derivation of the regional Attention Coupling model stack."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Generic, TypeVar

from .patcher_lifecycle import PATCHER_LIFECYCLE, ModelMutation

PatcherValue = TypeVar("PatcherValue")


@dataclass(frozen=True)
class RegionalModelPatchStack(Generic[PatcherValue]):
    """Retain the user, Attention Coupling, and spatial MODEL generations."""

    user_model: PatcherValue
    attention_model: PatcherValue
    sampling_model: PatcherValue


class RegionalModelPatchStackBuilder:
    """Build the fixed two-derivation regional MODEL patch order."""

    def build(
        self,
        user_model: PatcherValue,
        *,
        attention_mutations: Iterable[ModelMutation],
        spatial_mutations: Iterable[ModelMutation],
    ) -> RegionalModelPatchStack[PatcherValue]:
        """Derive Attention Coupling first and spatial sampling second."""

        attention_steps = tuple(attention_mutations)
        spatial_steps = tuple(spatial_mutations)
        if not attention_steps:
            raise ValueError(
                "Regional model patch stack requires Attention Coupling mutations."
            )
        if not spatial_steps:
            raise ValueError("Regional model patch stack requires spatial mutations.")

        attention_model = PATCHER_LIFECYCLE.derive_model(
            user_model,
            attention_steps,
            operation="SimpleSyrup Attention Coupling patch stack",
        )
        sampling_model = PATCHER_LIFECYCLE.derive_model(
            attention_model,
            spatial_steps,
            operation="SimpleSyrup spatial regional patch stack",
        )
        return RegionalModelPatchStack(
            user_model=user_model,
            attention_model=attention_model,
            sampling_model=sampling_model,
        )


REGIONAL_MODEL_PATCH_STACK_BUILDER = RegionalModelPatchStackBuilder()
