# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify conditioning-owned HookGroup adaptation for Attention Coupling."""

from __future__ import annotations

from typing import Any

import comfy.hooks
import pytest
import torch

from simple_syrup.domain.conditioning_batch import ConditioningBatch
from simple_syrup.domain.raw_regional_attention import (
    build_raw_regional_attention_plan,
)
from simple_syrup.domain.regional_lora_plan import RegionalLoraBranch
from simple_syrup.domain.regional_mask_bank import RegionalMaskBank
from simple_syrup.masking.regional_prompt_masks import build_regional_mask_bank
from simple_syrup.runtime.regional_lora_conditioning_adapter import (
    RegionalLoraConditioningAdapter,
)


class _Sampling:
    """Convert percentages into descending deterministic sigmas."""

    @staticmethod
    def percent_to_sigma(percent: float) -> float:
        """Return one linear descending sigma."""

        return 100.0 * (1.0 - percent)


class _Model:
    """Expose the model sampling boundary required by HookGroup adaptation."""

    model_sampling = _Sampling()


def test_adapter_extracts_ordered_positive_and_negative_regional_stacks() -> None:
    """Preserve region, branch, hook order, identity, strength, and schedules."""

    positive_hooks = _hooks(("pc-PRIMARY_ADAPTER-0.8-0.0", "pc-detail-0.4-0.0"))
    negative_hooks = _hooks(("pc-SECONDARY_ADAPTER-0.6-0.0",))
    positive = ConditioningBatch(
        (
            _conditioning(),
            _conditioning(positive_hooks),
            _conditioning(),
        )
    )
    negative = ConditioningBatch(
        (
            _conditioning(),
            _conditioning(),
            _conditioning(negative_hooks),
        )
    )
    plan = build_raw_regional_attention_plan(
        positive=positive,
        negative=negative,
        mask_bank=_mask_bank(2),
    )

    adaptation = RegionalLoraConditioningAdapter().adapt(plan, model=_Model())

    assert [item.adapter_identity.value for item in adaptation.plan.adapters] == [
        "pc-PRIMARY_ADAPTER-0.8-0.0",
        "pc-detail-0.4-0.0",
        "pc-SECONDARY_ADAPTER-0.6-0.0",
    ]
    assert [item.region_index for item in adaptation.plan.adapters] == [0, 0, 1]
    assert [item.branch for item in adaptation.plan.adapters] == [
        RegionalLoraBranch.POSITIVE,
        RegionalLoraBranch.POSITIVE,
        RegionalLoraBranch.NEGATIVE,
    ]
    expected_weights = tuple(
        hook.weights
        for group in (positive_hooks, negative_hooks)
        for hook in group.get_type(comfy.hooks.EnumHookType.Weight)
    )
    assert (
        tuple(payload.raw_weights for payload in adaptation.adapter_payloads)
        == expected_weights
    )
    assert all(payload.needs_resolution for payload in adaptation.adapter_payloads)


def test_adapter_accepts_cloned_schedule_entries_and_rejects_mixed_groups() -> None:
    """Use one shared model schedule owner across every text-schedule entry."""

    hooks = _hooks(("pc-PRIMARY_ADAPTER-0.8-0.0",))
    cloned = hooks.clone()
    scheduled_conditioning: list[list[Any]] = [
        [torch.ones((1, 2, 3)), {"hooks": hooks}],
        [torch.ones((1, 2, 3)), {"hooks": cloned}],
    ]
    plan = build_raw_regional_attention_plan(
        positive=ConditioningBatch((_conditioning(), scheduled_conditioning)),
        negative=ConditioningBatch((_conditioning(), _conditioning())),
        mask_bank=_mask_bank(1),
    )
    assert (
        len(RegionalLoraConditioningAdapter().adapt(plan, model=_Model()).plan.adapters)
        == 1
    )

    other = _hooks(("pc-other-0.8-0.0",))
    scheduled_conditioning[1][1]["hooks"] = other
    mixed = build_raw_regional_attention_plan(
        positive=ConditioningBatch((_conditioning(), scheduled_conditioning)),
        negative=ConditioningBatch((_conditioning(), _conditioning())),
        mask_bank=_mask_bank(1),
    )
    with pytest.raises(ValueError, match="different model HookGroups"):
        RegionalLoraConditioningAdapter().adapt(mixed, model=_Model())


def test_adapter_retains_encoded_text_loras_without_model_admission() -> None:
    """Ignore text-only hooks and retain only mixed-stack model participants."""

    global_text_hooks = _weight_hook(
        identity=None,
        model_strength=0.0,
        clip_strength=0.75,
        key="lora_te1_global.lora_down.weight",
    )
    regional_text_hooks = _weight_hook(
        identity=None,
        model_strength=0.0,
        clip_strength=0.75,
        key="lora_te1_regional.lora_down.weight",
    )
    regional_model_hooks = _weight_hook(
        identity="pc-PRIMARY_ADAPTER-0.8-0.0",
        model_strength=0.8,
        clip_strength=0.0,
        key="diffusion_model.blocks.0.lora_A.weight",
    )
    mixed_hooks = regional_text_hooks.clone_and_combine(regional_model_hooks)
    plan = build_raw_regional_attention_plan(
        positive=ConditioningBatch(
            (_conditioning(global_text_hooks), _conditioning(mixed_hooks))
        ),
        negative=ConditioningBatch((_conditioning(), _conditioning())),
        mask_bank=_mask_bank(1),
    )

    adaptation = RegionalLoraConditioningAdapter().adapt(plan, model=_Model())

    assert [item.composition_index for item in adaptation.plan.adapters] == [0]
    assert [item.adapter_identity.value for item in adaptation.plan.adapters] == [
        "pc-PRIMARY_ADAPTER-0.8-0.0"
    ]
    model_hook = regional_model_hooks.get_type(comfy.hooks.EnumHookType.Weight)[0]
    assert len(adaptation.adapter_payloads) == 1
    assert adaptation.adapter_payloads[0].needs_resolution is True
    assert adaptation.adapter_payloads[0].raw_weights is model_hook.weights
    assert (
        global_text_hooks.get_type(comfy.hooks.EnumHookType.Weight)[0]._strength_model
        == 0.0
    )
    assert (
        regional_text_hooks.get_type(comfy.hooks.EnumHookType.Weight)[0].hook_ref
        is None
    )


def test_adapter_accepts_different_text_only_hooks_across_schedule_entries() -> None:
    """Treat already-encoded text-only schedule metadata as context-owned."""

    first = _weight_hook(
        identity=None,
        model_strength=0.0,
        clip_strength=0.5,
        key="lora_te1_first.lora_down.weight",
    )
    second = _weight_hook(
        identity=None,
        model_strength=0.0,
        clip_strength=0.9,
        key="lora_te1_second.lora_down.weight",
    )
    scheduled = [
        [torch.ones((1, 2, 3)), {"hooks": first}],
        [torch.full((1, 2, 3), 2.0), {"hooks": second}],
    ]
    plan = build_raw_regional_attention_plan(
        positive=ConditioningBatch((_conditioning(), scheduled)),
        negative=ConditioningBatch((_conditioning(), _conditioning())),
        mask_bank=_mask_bank(1),
    )

    adaptation = RegionalLoraConditioningAdapter().adapt(plan, model=_Model())

    assert adaptation.plan.adapters == ()
    assert adaptation.adapter_payloads == ()


def test_adapter_rejects_global_hooks_and_opaque_regional_identity() -> None:
    """Require global MODEL ownership and a stable regional adapter identity."""

    hooks = _hooks(("pc-PRIMARY_ADAPTER-0.8-0.0",))
    global_plan = build_raw_regional_attention_plan(
        positive=ConditioningBatch((_conditioning(hooks), _conditioning())),
        negative=ConditioningBatch((_conditioning(), _conditioning())),
        mask_bank=_mask_bank(1),
    )
    with pytest.raises(ValueError, match="Apply global LoRAs to the input MODEL"):
        RegionalLoraConditioningAdapter().adapt(global_plan, model=_Model())

    opaque = comfy.hooks.create_hook_lora({}, 1.0, 0.0)
    opaque_plan = build_raw_regional_attention_plan(
        positive=ConditioningBatch((_conditioning(), _conditioning(opaque))),
        negative=ConditioningBatch((_conditioning(), _conditioning())),
        mask_bank=_mask_bank(1),
    )
    with pytest.raises(ValueError, match="no stable string adapter identity"):
        RegionalLoraConditioningAdapter().adapt(opaque_plan, model=_Model())


def _hooks(identities: tuple[str, ...]) -> comfy.hooks.HookGroup:
    """Return ordered WeightHooks with Prompt Control-style stable refs."""

    groups: list[comfy.hooks.HookGroup] = []
    for index, identity in enumerate(identities):
        group = comfy.hooks.create_hook_lora({}, 0.8 - index * 0.1, 0.0)
        hook = group.get_type(comfy.hooks.EnumHookType.Weight)[0]
        hook.hook_ref = identity
        groups.append(group)
    combined = comfy.hooks.HookGroup.combine_all_hooks(groups)
    assert combined is not None
    return combined


def _weight_hook(
    *,
    identity: str | None,
    model_strength: float,
    clip_strength: float,
    key: str,
) -> comfy.hooks.HookGroup:
    """Return one raw WeightHook with explicit model/CLIP participation."""

    group = comfy.hooks.create_hook_lora(
        {key: object()},
        strength_model=model_strength,
        strength_clip=clip_strength,
    )
    group.get_type(comfy.hooks.EnumHookType.Weight)[0].hook_ref = identity
    return group


def _conditioning(
    hooks: comfy.hooks.HookGroup | None = None,
) -> list[list[object]]:
    """Return one small valid conditioning value."""

    metadata: dict[str, object] = {}
    if hooks is not None:
        metadata["hooks"] = hooks
    return [[torch.ones((1, 2, 3)), metadata]]


def _mask_bank(region_count: int) -> RegionalMaskBank:
    """Return a canonical one-row mask bank."""

    return build_regional_mask_bank(
        torch.ones((region_count, 1, 1)),
        feather=0,
        canvas_height=1,
        canvas_width=1,
    )
