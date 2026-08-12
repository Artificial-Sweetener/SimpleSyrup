"""Prove all-or-nothing regional Anima adapter-plan admission."""

from __future__ import annotations

from typing import Any, cast

import comfy.hooks
import pytest
import torch

from simple_syrup.domain.regional_lora_plan import (
    RegionalLoraAdapterIdentity,
    RegionalLoraAdapterPlan,
    RegionalLoraBranch,
    RegionalLoraPlan,
    RegionalLoraScheduleBoundary,
)
from simple_syrup.runtime.regional_lora.anima_plan_admission import (
    ANIMA_REGIONAL_LORA_PLAN_ADMISSION_SERVICE,
    AnimaRegionalLoraPlanAdmissionError,
)
from simple_syrup.runtime.regional_lora_plan_adapter import (
    REGIONAL_LORA_PLAN_ADAPTER,
    RegionalLoraHookSource,
    RegionalLoraPlanAdaptation,
)


def test_plan_admission_preserves_supported_static_and_scheduled_adapters() -> None:
    """Admit exact standard tensors across both supported schedule forms."""

    first = _weights("self_attn.q_proj")
    second = _weights("self_attn.k_proj")
    hooks = comfy.hooks.create_hook_lora(first, 0.75, 0.0).clone_and_combine(
        comfy.hooks.create_hook_lora(second, -0.5, 0.0)
    )
    scheduled = comfy.hooks.HookKeyframeGroup()
    scheduled.add(comfy.hooks.HookKeyframe(0.0, 0.0, 0))
    scheduled.add(comfy.hooks.HookKeyframe(1.0, 0.5, 0))
    hooks.get_type(comfy.hooks.EnumHookType.Weight)[1].hook_keyframe = scheduled
    adaptation = REGIONAL_LORA_PLAN_ADAPTER.adapt(
        (
            RegionalLoraHookSource(
                region_index=0,
                branch=RegionalLoraBranch.POSITIVE,
                hooks=hooks,
                adapter_identities=(
                    RegionalLoraAdapterIdentity("static.safetensors"),
                    RegionalLoraAdapterIdentity("scheduled.safetensors"),
                ),
            ),
        ),
        model=_Model(),
    )

    result = ANIMA_REGIONAL_LORA_PLAN_ADMISSION_SERVICE.admit(adaptation)

    assert result.plan is adaptation.plan
    assert (
        tuple(item.adapter_plan for item in result.adapters) == adaptation.plan.adapters
    )
    assert result.plan.adapters[1].schedule == (
        RegionalLoraScheduleBoundary(0.0, 100.0, 0.0, 0),
        RegionalLoraScheduleBoundary(0.5, 50.0, 1.0, 0),
    )
    assert [item.admission.targets[0].family.value for item in result.adapters] == [
        "self_attn.q_proj",
        "self_attn.k_proj",
    ]
    assert result.adapters[0].admission.targets[0].adapter.down is next(
        value for key, value in first.items() if key.endswith("lora_A.weight")
    )
    assert result.adapters[1].admission.targets[0].adapter.up is next(
        value for key, value in second.items() if key.endswith("lora_B.weight")
    )


def test_plan_admission_aggregates_every_adapter_format_and_target_failure() -> None:
    """Reject empty, converted, and unsupported-target adapters as one failure."""

    plan = _plan(
        (RegionalLoraScheduleBoundary(0.0, 100.0, 1.0, 0),),
        (RegionalLoraScheduleBoundary(0.0, 100.0, 1.0, 0),),
        (RegionalLoraScheduleBoundary(0.0, 100.0, 1.0, 0),),
    )
    converted_model_patch = {
        "diffusion_model.blocks.0.self_attn.q_proj.weight": (
            "model_as_lora",
            (torch.ones((1,)),),
        )
    }
    unsupported_target = _weights("unknown_projection")

    with pytest.raises(AnimaRegionalLoraPlanAdmissionError) as captured:
        ANIMA_REGIONAL_LORA_PLAN_ADMISSION_SERVICE.admit(
            RegionalLoraPlanAdaptation(
                plan,
                ({}, converted_model_patch, unsupported_target),
            )
        )

    issues = captured.value.issues
    assert [issue.composition_index for issue in issues] == [0, 1, 2]
    assert [issue.adapter_identity for issue in issues] == [
        "adapter-0.safetensors",
        "adapter-1.safetensors",
        "adapter-2.safetensors",
    ]
    assert "at least one complete supported target" in issues[0].reason
    assert "unsupported adapter format" in issues[1].reason
    assert "unsupported Anima target family" in issues[2].reason
    assert "failed before sampling" in str(captured.value)


def test_plan_admission_requires_one_weight_mapping_per_ordered_adapter() -> None:
    """Reject an ambiguous plan-to-artifact axis before target inspection."""

    plan = _plan((RegionalLoraScheduleBoundary(0.0, 100.0, 1.0, 0),))

    with pytest.raises(ValueError, match="one weight payload"):
        RegionalLoraPlanAdaptation(plan, ())
    with pytest.raises(TypeError, match="requires an adaptation"):
        ANIMA_REGIONAL_LORA_PLAN_ADMISSION_SERVICE.admit(cast(Any, plan))


def test_empty_plan_admission_is_exact_and_empty() -> None:
    """Retain the supported no-regional-LoRA path without synthetic adapters."""

    plan = RegionalLoraPlan(())

    result = ANIMA_REGIONAL_LORA_PLAN_ADMISSION_SERVICE.admit(
        RegionalLoraPlanAdaptation(plan, ())
    )

    assert result.plan is plan
    assert result.adapters == ()


def _plan(
    *schedules: tuple[RegionalLoraScheduleBoundary, ...],
) -> RegionalLoraPlan:
    """Build one canonical ordered plan for admission tests."""

    return RegionalLoraPlan(
        tuple(
            RegionalLoraAdapterPlan(
                adapter_identity=RegionalLoraAdapterIdentity(
                    f"adapter-{index}.safetensors"
                ),
                composition_index=index,
                region_index=0,
                branch=RegionalLoraBranch.POSITIVE,
                model_strength=1.0,
                schedule=schedule,
            )
            for index, schedule in enumerate(schedules)
        )
    )


def _weights(family: str) -> dict[str, torch.Tensor]:
    """Build one standard rank-one Anima adapter target."""

    target = f"diffusion_model.blocks.0.{family}"
    return {
        f"{target}.lora_A.weight": torch.ones((1, 2048)),
        f"{target}.lora_B.weight": torch.ones((2048, 1)),
    }


class _LinearSampling:
    """Convert percentages into deterministic descending admission sigmas."""

    @staticmethod
    def percent_to_sigma(percent: float) -> float:
        """Return one exact synthetic schedule boundary."""

        return 100.0 * (1.0 - percent)


class _Model:
    """Expose the model-sampling boundary required by plan adaptation."""

    model_sampling = _LinearSampling()
