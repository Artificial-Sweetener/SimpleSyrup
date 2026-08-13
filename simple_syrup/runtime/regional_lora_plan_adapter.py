# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Adapt ordered Comfy HookGroups into immutable regional LoRA plans."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

from comfy.hooks import (
    HookKeyframe,
    HookKeyframeGroup,
    WeightHook,
)

from ..domain.regional_lora_plan import (
    EMPTY_REGIONAL_LORA_PLAN,
    RegionalLoraAdapterIdentity,
    RegionalLoraAdapterPlan,
    RegionalLoraBranch,
    RegionalLoraPlan,
    RegionalLoraScheduleBoundary,
)
from .regional_lora_host_payload import RegionalLoraHostPayload
from .regional_model_hook_selection import (
    REGIONAL_MODEL_HOOK_SELECTOR,
    RegionalModelHookSelector,
)


@dataclass(frozen=True)
class RegionalLoraHookSource:
    """Bind explicit adapter identities and regional ownership to one HookGroup."""

    region_index: int
    branch: RegionalLoraBranch
    hooks: object
    adapter_identities: tuple[RegionalLoraAdapterIdentity, ...]

    def __post_init__(self) -> None:
        """Validate source-owned values before inspecting Comfy hook state."""

        if isinstance(self.region_index, bool) or not isinstance(
            self.region_index, int
        ):
            raise TypeError("Regional LoRA source region_index must be an integer.")
        if self.region_index < 0:
            raise ValueError("Regional LoRA source region_index must be non-negative.")
        if not isinstance(self.branch, RegionalLoraBranch):
            raise TypeError("Regional LoRA source branch has an invalid type.")
        if not isinstance(self.adapter_identities, tuple):
            raise TypeError("Regional LoRA source adapter identities must be a tuple.")
        if any(
            not isinstance(identity, RegionalLoraAdapterIdentity)
            for identity in self.adapter_identities
        ):
            raise TypeError(
                "Regional LoRA source contains an invalid adapter identity."
            )


@dataclass(frozen=True, slots=True)
class RegionalLoraPlanAdaptation:
    """Retain one immutable plan with its exact ordered hook weight payloads."""

    plan: RegionalLoraPlan
    adapter_payloads: tuple[RegionalLoraHostPayload, ...]

    def __post_init__(self) -> None:
        """Require exact plan types and one payload per canonical adapter."""

        if not isinstance(self.plan, RegionalLoraPlan):
            raise TypeError("Regional LoRA adaptation requires a plan.")
        if not isinstance(self.adapter_payloads, tuple):
            raise TypeError("Regional LoRA adaptation payloads must be a tuple.")
        if any(
            not isinstance(payload, RegionalLoraHostPayload)
            for payload in self.adapter_payloads
        ):
            raise TypeError("Regional LoRA adaptation contains an invalid payload.")
        if len(self.adapter_payloads) != len(self.plan.adapters):
            raise ValueError(
                "Regional LoRA adaptation requires one host payload per adapter."
            )


class RegionalLoraPlanAdapter:
    """Adapt ordered regional HookGroups without changing supplied hooks."""

    def __init__(
        self,
        hook_selector: RegionalModelHookSelector = REGIONAL_MODEL_HOOK_SELECTOR,
    ) -> None:
        """Retain the sole WeightHook model-participation authority."""

        self._hook_selector = hook_selector

    def adapt(
        self,
        sources: Iterable[RegionalLoraHookSource],
        *,
        model: object,
    ) -> RegionalLoraPlanAdaptation:
        """Preserve source order, plan values, and exact hook weight payloads."""

        source_values = tuple(sources)
        if not source_values:
            return RegionalLoraPlanAdaptation(EMPTY_REGIONAL_LORA_PLAN, ())

        adapters: list[RegionalLoraAdapterPlan] = []
        adapter_payloads: list[RegionalLoraHostPayload] = []
        for source_index, source in enumerate(source_values):
            if not isinstance(source, RegionalLoraHookSource):
                raise TypeError(
                    f"Regional LoRA source {source_index} has an invalid type."
                )
            source_adapters, source_payloads = self._adapt_source(
                source,
                model=model,
                first_composition_index=len(adapters),
                source_index=source_index,
            )
            adapters.extend(source_adapters)
            adapter_payloads.extend(source_payloads)
        return RegionalLoraPlanAdaptation(
            RegionalLoraPlan(adapters=tuple(adapters)),
            tuple(adapter_payloads),
        )

    def _adapt_source(
        self,
        source: RegionalLoraHookSource,
        *,
        model: object,
        first_composition_index: int,
        source_index: int,
    ) -> tuple[
        tuple[RegionalLoraAdapterPlan, ...],
        tuple[RegionalLoraHostPayload, ...],
    ]:
        """Adapt one exact HookGroup while retaining hook order and payloads."""

        selection = self._hook_selector.select(
            source.hooks,
            source_label=f"Regional LoRA source {source_index}",
        )
        if len(source.adapter_identities) != len(selection.model_hooks):
            raise ValueError(
                f"Regional LoRA source {source_index} supplies "
                f"{len(source.adapter_identities)} adapter identities for "
                f"{len(selection.model_hooks)} ordered model-active WeightHooks."
            )

        adapters = tuple(
            RegionalLoraAdapterPlan(
                adapter_identity=source.adapter_identities[participant_index],
                composition_index=first_composition_index + participant_index,
                region_index=source.region_index,
                branch=source.branch,
                model_strength=participant.model_strength,
                schedule=self._schedule(
                    participant.hook,
                    model=model,
                    source_index=source_index,
                    hook_index=participant.hook_index,
                ),
            )
            for participant_index, participant in enumerate(selection.model_hooks)
        )
        return adapters, tuple(
            RegionalLoraHostPayload.from_hook(participant.hook)
            for participant in selection.model_hooks
        )

    def _schedule(
        self,
        hook: WeightHook,
        *,
        model: object,
        source_index: int,
        hook_index: int,
    ) -> tuple[RegionalLoraScheduleBoundary, ...]:
        """Copy every ordered HookKeyframe field into immutable boundaries."""

        keyframe_group = getattr(hook, "hook_keyframe", None)
        if not isinstance(keyframe_group, HookKeyframeGroup):
            raise TypeError(
                f"Regional LoRA source {source_index} WeightHook {hook_index} "
                "must expose a Comfy HookKeyframeGroup."
            )
        keyframes = keyframe_group.keyframes
        if not isinstance(keyframes, list):
            raise TypeError(
                f"Regional LoRA source {source_index} WeightHook {hook_index} "
                "keyframes must be a list."
            )
        if not keyframes:
            return (
                RegionalLoraScheduleBoundary(
                    start_percent=0.0,
                    start_sigma=_percent_to_sigma(model, 0.0),
                    strength_multiplier=1.0,
                    guarantee_steps=0,
                ),
            )

        boundaries: list[RegionalLoraScheduleBoundary] = []
        for boundary_index, keyframe in enumerate(keyframes):
            if not isinstance(keyframe, HookKeyframe):
                raise TypeError(
                    f"Regional LoRA source {source_index} WeightHook {hook_index} "
                    f"keyframe {boundary_index} has an invalid type."
                )
            guarantee_steps = keyframe.guarantee_steps
            if isinstance(guarantee_steps, bool) or not isinstance(
                guarantee_steps, int
            ):
                raise TypeError(
                    f"Regional LoRA source {source_index} WeightHook {hook_index} "
                    f"keyframe {boundary_index} guarantee_steps must be an integer."
                )
            start_percent = _required_finite_number(
                keyframe.start_percent,
                field_name=(
                    f"source {source_index} WeightHook {hook_index} "
                    f"keyframe {boundary_index} start_percent"
                ),
            )
            boundaries.append(
                RegionalLoraScheduleBoundary(
                    start_percent=start_percent,
                    start_sigma=_percent_to_sigma(model, start_percent),
                    strength_multiplier=_required_finite_number(
                        keyframe.strength,
                        field_name=(
                            f"source {source_index} WeightHook {hook_index} "
                            f"keyframe {boundary_index} strength"
                        ),
                    ),
                    guarantee_steps=guarantee_steps,
                )
            )
        return tuple(boundaries)


def _required_finite_number(value: object, *, field_name: str) -> float:
    """Narrow one external Comfy numeric field to a finite float."""

    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
    ):
        raise TypeError(f"Regional LoRA {field_name} must be finite numeric data.")
    return float(value)


def _percent_to_sigma(model: object, percent: float) -> float:
    """Convert one authored boundary through the selected model schedule."""

    model_sampling = getattr(model, "model_sampling", None)
    converter = getattr(model_sampling, "percent_to_sigma", None)
    if not callable(converter):
        raise TypeError(
            "Regional LoRA model must expose model_sampling.percent_to_sigma."
        )
    return _required_finite_number(
        converter(percent),
        field_name=f"converted start sigma for percent {percent}",
    )


REGIONAL_LORA_PLAN_ADAPTER = RegionalLoraPlanAdapter()
