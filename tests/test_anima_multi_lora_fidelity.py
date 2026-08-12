# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Prove ordered multi-LoRA fidelity, invariants, and pinned target execution."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import torch
from anima_branch_test_values import uniform_branch_invocation
from comfy.hooks import HookKeyframe, HookKeyframeGroup, WeightHook
from regional_attention_test_values import single_entry_regions
from regional_lora_test_values import static_lora_schedule
from safetensors.torch import load_file
from torch import nn

from simple_syrup.domain.regional_attention import RegionalAttentionBranch
from simple_syrup.domain.regional_attention_batch import (
    BatchedRegionalAttentionContexts,
    RegionalAttentionChunkBatch,
)
from simple_syrup.domain.regional_lora_plan import (
    RegionalLoraAdapterIdentity,
    RegionalLoraAdapterPlan,
    RegionalLoraBranch,
    RegionalLoraScheduleBoundary,
)
from simple_syrup.domain.regional_mask_bank import RegionalMaskBank
from simple_syrup.runtime.regional_lora.anima_attention_execution import (
    AnimaRegionalAttentionExecution,
)
from simple_syrup.runtime.regional_lora.anima_composition import (
    AnimaRegionalLoraComposition,
)
from simple_syrup.runtime.regional_lora.anima_cross_attention_context import (
    AnimaCrossAttentionInvocationContext,
)
from simple_syrup.runtime.regional_lora.anima_execution_scope import (
    AnimaLoraBranchInvocationContext,
    AnimaLoraSpatialInvocation,
    AnimaLoraSpatialInvocationContext,
    AnimaRegionalLoraAdapterExecution,
)
from simple_syrup.runtime.regional_lora.anima_linear_execution import (
    AnimaRegionalLoraCompositionLinearPatch,
)
from simple_syrup.runtime.regional_lora.anima_lora_weights import (
    AnimaLoraWeightResolver,
)
from simple_syrup.runtime.regional_lora.anima_query_masks import AnimaQueryMaskBatch
from simple_syrup.runtime.regional_lora.anima_schedule_context import (
    AnimaRegionalLoraScheduleContext,
)
from simple_syrup.runtime.regional_lora.anima_targets import (
    ANIMA_LORA_TARGET_CLASSIFIER,
    AnimaLoraAdmission,
    AnimaLoraTarget,
    AnimaLoraTargetFamily,
)
from simple_syrup.runtime.regional_lora.execution_cache import (
    ModelCloneLineage,
    RegionalLoraExecutionCache,
)
from simple_syrup.runtime.regional_lora.standard_adapter import StandardLoraTarget
from simple_syrup.runtime.regional_lora_schedule_resolution import (
    RegionalLoraScheduleSession,
)

_PINNED_ADAPTER_A_PATH = Path(r"<MODEL_ROOT>\Loras\Anima\style\adapter-a.safetensors")
_SELF_FAMILIES = frozenset(
    family for family in AnimaLoraTargetFamily if family.value.startswith("self_attn")
)
_MLP_FAMILIES = frozenset(
    family for family in AnimaLoraTargetFamily if family.value.startswith("mlp")
)
_CROSS_FAMILIES = frozenset(
    family for family in AnimaLoraTargetFamily if family.value.startswith("cross_attn")
)


class _CountingZeroLinear(nn.Module):
    """Return a zero original output and retain exact call cardinality."""

    def __init__(self, output_features: int) -> None:
        """Retain the output width and start with no calls."""

        super().__init__()
        self.output_features = output_features
        self._calls = [0]

    @property
    def calls(self) -> int:
        """Return the shared shell-visible call count."""

        return self._calls[0]

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return one zero output over all leading dimensions."""

        self._calls[0] += 1
        return torch.zeros(
            (*inputs.shape[:-1], self.output_features),
            device=inputs.device,
            dtype=inputs.dtype,
        )


def test_all_one_stack_and_partitioned_repeat_match_global_lora_math() -> None:
    """Match ordinary global composition for all-one and complete partitions."""

    first = _tiny_target(down=2.0, up=3.0)
    second = _tiny_target(down=-1.0, up=0.5)
    inputs = torch.tensor([[[1.0], [2.0]]])

    all_one = _run_spatial(
        inputs,
        masks=torch.ones((1, 1, 1, 1, 2)),
        uses=((first, "first", 0, 0.25), (second, "second", 0, -0.5)),
    )
    expected_stack = torch.zeros_like(inputs)
    for target, strength in ((first, 0.25), (second, -0.5)):
        delta = (inputs @ target.adapter.down.T) @ target.adapter.up.T
        expected_stack = expected_stack + delta * strength
    torch.testing.assert_close(all_one, expected_stack)

    partitioned = _run_spatial(
        inputs,
        masks=torch.tensor([[[[[1.0, 0.0]]]], [[[[0.0, 1.0]]]]]),
        uses=((first, "first", 0, 0.75), (first, "first", 1, 0.75)),
    )
    global_first = ((inputs @ first.adapter.down.T) @ first.adapter.up.T) * 0.75
    torch.testing.assert_close(partitioned, global_first)


def test_scheduled_single_stacked_adjacent_and_overlapping_match_global_hooks() -> None:
    """Match ordinary global WeightHook output at every interval and boundary."""

    profiles = (
        (((0.0, 0.0, 0), (0.25, 1.0, 0), (0.75, 0.5, 0)),),
        (
            ((0.0, 1.0, 0), (0.5, 0.25, 0)),
            ((0.0, 0.0, 0), (0.25, 1.0, 0), (0.75, -0.5, 0)),
        ),
        (
            ((0.0, 1.0, 0), (0.5, 0.0, 0)),
            ((0.0, 0.0, 0), (0.5, 1.0, 0)),
        ),
    )
    targets = (
        _tiny_target(down=1.25, up=0.5),
        _tiny_target(down=-0.75, up=2.0),
    )

    for schedules in profiles:
        selected_targets = targets[: len(schedules)]
        observed, expected = _scheduled_outputs(
            selected_targets,
            schedules,
            regions=(0,) * len(schedules),
            masks=torch.ones((1, 1, 1, 1, 2)),
        )
        for actual, reference in zip(observed, expected, strict=True):
            torch.testing.assert_close(actual, reference)


def test_different_region_schedules_remain_spatially_independent() -> None:
    """Apply independent WeightHook transitions only to each owned region."""

    schedules = (
        ((0.0, 1.0, 0), (0.5, 0.0, 0)),
        ((0.0, 0.0, 0), (0.25, 0.5, 0), (0.75, 1.0, 0)),
    )
    observed, expected = _scheduled_outputs(
        (
            _tiny_target(down=1.0, up=2.0),
            _tiny_target(down=1.0, up=-3.0),
        ),
        schedules,
        regions=(0, 1),
        masks=torch.tensor([[[[[1.0, 0.0]]]], [[[[0.0, 1.0]]]]]),
    )

    for actual, reference in zip(observed, expected, strict=True):
        torch.testing.assert_close(actual, reference)


def test_scheduled_transition_preserves_declared_non_associative_order() -> None:
    """Retain declared adapter addition order when a later schedule activates."""

    always_active = ((0.0, 1.0, 0),)
    activates_at_half = ((0.0, 0.0, 0), (0.5, 1.0, 0))
    observed, expected = _scheduled_outputs(
        (
            _tiny_target(down=1.0, up=1.0e20),
            _tiny_target(down=1.0, up=1.5e20),
            _tiny_target(down=1.0, up=12.0),
        ),
        (always_active, always_active, activates_at_half),
        regions=(0, 0, 0),
        masks=torch.ones((1, 1, 1, 1, 2)),
    )

    for actual, reference in zip(observed, expected, strict=True):
        torch.testing.assert_close(actual, reference)
    torch.testing.assert_close(observed[1], torch.zeros_like(observed[1]))
    torch.testing.assert_close(observed[2], torch.tensor([[[3.0], [6.0]]]))


def test_overlaps_uncovered_pixels_and_both_branches_match_ordered_reference() -> None:
    """Retain overlap sums, base complement, and independent CFG branch ownership."""

    first = _tiny_target(down=1.25, up=0.5)
    second = _tiny_target(down=-0.75, up=2.0)
    masks = torch.tensor(
        [
            [[[[0.8, 0.4, 0.0]]], [[[0.8, 0.4, 0.0]]]],
            [[[[0.3, 0.9, 0.0]]], [[[0.3, 0.9, 0.0]]]],
        ]
    )
    inputs = torch.tensor([[[1.0], [2.0], [3.0]], [[1.0], [2.0], [3.0]]])
    attention = _attention(masks[:, 0, 0], include_negative=True)
    model = ModelCloneLineage(uuid4(), uuid4())
    cache = RegionalLoraExecutionCache()
    executions = (
        _execution(0, first, "first", 0, 0.7, attention, model, cache),
        _execution(1, second, "second", 0, -0.25, attention, model, cache),
        _execution(2, first, "first", 1, 0.2, attention, model, cache),
        _execution(
            3,
            second,
            "second",
            1,
            0.5,
            attention,
            model,
            cache,
            branch=RegionalLoraBranch.NEGATIVE,
        ),
    )
    output = _execute_spatial(inputs, masks, executions)
    expected = _spatial_reference(inputs, masks, executions)

    torch.testing.assert_close(output, expected)
    assert torch.equal(output[:, 2], torch.zeros_like(output[:, 2]))
    assert not torch.equal(output[0, :2], output[1, :2])


def test_declared_adapter_order_is_observable_and_preserved() -> None:
    """Expose float32 non-associativity so a reordered stack cannot pass silently."""

    huge = _tiny_target(down=1.0, up=1.0e20)
    negative_huge = _tiny_target(down=1.0, up=-1.0e20)
    small = _tiny_target(down=1.0, up=3.0)
    inputs = torch.ones((1, 1, 1))
    masks = torch.ones((1, 1, 1, 1, 1))

    first_order = _run_spatial(
        inputs,
        masks=masks,
        uses=(
            (huge, "huge", 0, 1.0),
            (negative_huge, "negative", 0, 1.0),
            (small, "small", 0, 1.0),
        ),
    )
    second_order = _run_spatial(
        inputs,
        masks=masks,
        uses=(
            (huge, "huge", 0, 1.0),
            (small, "small", 0, 1.0),
            (negative_huge, "negative", 0, 1.0),
        ),
    )

    torch.testing.assert_close(first_order, torch.tensor([[[3.0]]]))
    torch.testing.assert_close(second_order, torch.tensor([[[0.0]]]))
    assert not torch.equal(first_order, second_order)


def test_region_and_branch_permutations_change_only_their_owned_positions() -> None:
    """Prove region and CFG ownership are positional rather than incidental."""

    first = _tiny_target(down=1.0, up=2.0)
    second = _tiny_target(down=1.0, up=-3.0)
    inputs = torch.ones((1, 2, 1))
    partition = torch.tensor([[[[[1.0, 0.0]]]], [[[[0.0, 1.0]]]]])
    direct = _run_spatial(
        inputs,
        masks=partition,
        uses=((first, "first", 0, 1.0), (second, "second", 1, 1.0)),
    )
    region_swapped = _run_spatial(
        inputs,
        masks=partition,
        uses=((first, "first", 1, 1.0), (second, "second", 0, 1.0)),
    )
    torch.testing.assert_close(direct.flatten(), torch.tensor([2.0, -3.0]))
    torch.testing.assert_close(region_swapped.flatten(), torch.tensor([-3.0, 2.0]))

    branch_masks = torch.ones((1, 2, 1, 1, 1))
    branch_inputs = torch.ones((2, 1, 1))
    attention = _attention(branch_masks[:, 0, 0], include_negative=True)
    model = ModelCloneLineage(uuid4(), uuid4())
    cache = RegionalLoraExecutionCache()
    positive = _execute_spatial(
        branch_inputs,
        branch_masks,
        (
            _execution(
                0,
                first,
                "first",
                0,
                1.0,
                attention,
                model,
                cache,
            ),
        ),
    )
    negative = _execute_spatial(
        branch_inputs,
        branch_masks,
        (
            _execution(
                0,
                first,
                "first",
                0,
                1.0,
                attention,
                model,
                cache,
                branch=RegionalLoraBranch.NEGATIVE,
            ),
        ),
    )
    torch.testing.assert_close(positive.flatten(), torch.tensor([2.0, 0.0]))
    torch.testing.assert_close(negative.flatten(), torch.tensor([0.0, 2.0]))


def test_pinned_adapter_a_executes_every_target_through_composition_path() -> None:
    """Execute all 448 pinned rank-32 targets without ignored-key allowances."""

    weights = load_file(_PINNED_ADAPTER_A_PATH, device="cpu")
    admission = ANIMA_LORA_TARGET_CLASSIFIER.admit(weights)
    source_ids = tuple(
        (id(target.adapter.down), id(target.adapter.up)) for target in admission.targets
    )
    masks = torch.tensor([[[[[0.25]]]], [[[[0.75]]]]], dtype=torch.bfloat16)
    attention = _attention(masks[:, 0, 0].float(), include_negative=False)
    model = ModelCloneLineage(uuid4(), uuid4())
    cache = RegionalLoraExecutionCache()
    executions = (
        _execution(
            0, None, "adapter_a", 0, 0.35, attention, model, cache, admission=admission
        ),
        _execution(
            1, None, "adapter_a", 1, -0.2, attention, model, cache, admission=admission
        ),
    )
    composition = AnimaRegionalLoraComposition(executions)
    executed: set[str] = set()

    for target in admission.targets:
        output, original_calls = _execute_pinned_target(
            target,
            composition,
            masks,
        )
        assert output.shape[-1] == target.adapter.output_features
        assert bool(torch.isfinite(output).all().item())
        assert original_calls == 1
        executed.add(target.adapter.target)

    assert len(executed) == 448
    assert cache.size == 448
    assert (
        tuple(
            (id(target.adapter.down), id(target.adapter.up))
            for target in admission.targets
        )
        == source_ids
    )


def _execute_pinned_target(
    target: AnimaLoraTarget,
    composition: AnimaRegionalLoraComposition,
    masks: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """Run one real pinned target under its exact family execution scope."""

    cross = AnimaCrossAttentionInvocationContext()
    branch = AnimaLoraBranchInvocationContext()
    spatial = AnimaLoraSpatialInvocationContext()
    schedule, resolution = static_lora_schedule(composition)
    original = _CountingZeroLinear(target.adapter.output_features)
    patch = AnimaRegionalLoraCompositionLinearPatch(
        original,
        composition.groups_for_target(target.adapter.target),
        weight_resolver=AnimaLoraWeightResolver(
            cross_attention_context=cross,
            branch_context=branch,
            spatial_context=spatial,
        ),
        schedule_context=schedule,
    )
    features = target.adapter.input_features
    vector = torch.linspace(-0.01, 0.01, features, dtype=torch.bfloat16)
    if target.family in _SELF_FAMILIES:
        inputs = vector.reshape(1, 1, features)
        with (
            schedule.activate(resolution),
            spatial.activate(AnimaLoraSpatialInvocation(AnimaQueryMaskBatch(masks))),
        ):
            output = patch(inputs)
    elif target.family in _MLP_FAMILIES:
        inputs = vector.reshape(1, 1, 1, 1, features)
        with (
            schedule.activate(resolution),
            spatial.activate(AnimaLoraSpatialInvocation(AnimaQueryMaskBatch(masks))),
        ):
            output = patch(inputs)
    else:
        inputs = vector.reshape(1, 1, features).expand(3, 1, features)
        invocation = uniform_branch_invocation(1, (None, 0, 1))
        if target.family in _CROSS_FAMILIES:
            with (
                schedule.activate(resolution),
                cross.activate(uniform_branch_invocation(1, (None, 0, 1))),
            ):
                output = patch(inputs)
        else:
            with schedule.activate(resolution), branch.activate(invocation):
                output = patch(inputs)
    expected = _pinned_reference(target, inputs, masks)
    torch.testing.assert_close(output, expected)
    return output, original.calls


def _pinned_reference(
    target: AnimaLoraTarget,
    inputs: torch.Tensor,
    masks: torch.Tensor,
) -> torch.Tensor:
    """Return explicit ordered per-use pinned full-rank delta output."""

    delta = (inputs @ target.adapter.down.T) @ target.adapter.up.T
    result = torch.zeros_like(delta)
    if target.family in _SELF_FAMILIES:
        weights = (masks[0, 0, 0] * 0.35, masks[1, 0, 0] * -0.2)
        return result + delta * (weights[0] + weights[1]).reshape(1, 1, 1)
    if target.family in _MLP_FAMILIES:
        weight = masks[0, 0, 0, 0, 0] * 0.35 + masks[1, 0, 0, 0, 0] * -0.2
        return result + delta * weight
    result[1] = result[1] + delta[1] * 0.35
    result[2] = result[2] + delta[2] * -0.2
    return result


def _run_spatial(
    inputs: torch.Tensor,
    *,
    masks: torch.Tensor,
    uses: tuple[tuple[AnimaLoraTarget, str, int, float], ...],
) -> torch.Tensor:
    """Build and execute one positive-branch spatial composition."""

    attention = _attention(masks[:, 0, 0], include_negative=False)
    model = ModelCloneLineage(uuid4(), uuid4())
    cache = RegionalLoraExecutionCache()
    executions = tuple(
        _execution(index, target, identity, region, strength, attention, model, cache)
        for index, (target, identity, region, strength) in enumerate(uses)
    )
    return _execute_spatial(inputs, masks, executions)


def _scheduled_outputs(
    targets: tuple[AnimaLoraTarget, ...],
    schedules: tuple[tuple[tuple[float, float, int], ...], ...],
    *,
    regions: tuple[int, ...],
    masks: torch.Tensor,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    """Execute scheduled regional deltas beside installed global WeightHooks."""

    if not (len(targets) == len(schedules) == len(regions)):
        raise AssertionError("Scheduled fidelity axes must align.")
    strengths = (0.75, -0.5, 0.25)[: len(targets)]
    attention = _attention(masks[:, 0, 0], include_negative=False)
    model = ModelCloneLineage(uuid4(), uuid4())
    cache = RegionalLoraExecutionCache()
    hooks: list[WeightHook] = []
    executions: list[AnimaRegionalLoraAdapterExecution] = []
    for index, (target, values, region, strength) in enumerate(
        zip(targets, schedules, regions, strengths, strict=True)
    ):
        hook = WeightHook(strength_model=strength)
        group = HookKeyframeGroup()
        boundaries = []
        for start_percent, multiplier, guarantee_steps in values:
            group.add(HookKeyframe(multiplier, start_percent, guarantee_steps))
            boundaries.append(
                RegionalLoraScheduleBoundary(
                    start_percent,
                    _LinearSampling.percent_to_sigma(start_percent),
                    multiplier,
                    guarantee_steps,
                )
            )
        hook.hook_keyframe = group
        group.initialize_timesteps(_ScheduleModel())
        group.reset()
        hooks.append(hook)
        executions.append(
            _execution(
                index,
                target,
                f"scheduled-{index}",
                region,
                strength,
                attention,
                model,
                cache,
                schedule=tuple(boundaries),
            )
        )

    composition = AnimaRegionalLoraComposition(tuple(executions))
    spatial_context = AnimaLoraSpatialInvocationContext()
    schedule_context = AnimaRegionalLoraScheduleContext()
    session = RegionalLoraScheduleSession(
        tuple(execution.adapter_plan for execution in executions),
        maximum_sigma=100.0,
    )
    target_name = targets[0].adapter.target
    patch = AnimaRegionalLoraCompositionLinearPatch(
        _CountingZeroLinear(1),
        composition.groups_for_target(target_name),
        weight_resolver=AnimaLoraWeightResolver(
            cross_attention_context=AnimaCrossAttentionInvocationContext(),
            branch_context=AnimaLoraBranchInvocationContext(),
            spatial_context=spatial_context,
        ),
        schedule_context=schedule_context,
    )
    inputs = torch.tensor([[[1.0], [2.0]]])
    options = {"sample_sigmas": torch.tensor([100.0, 75.0, 50.0, 25.0, 0.0])}
    observed: list[torch.Tensor] = []
    expected: list[torch.Tensor] = []
    for sigma in (100.0, 75.0, 50.0, 25.0, 0.0):
        for hook in hooks:
            hook.hook_keyframe.prepare_current_keyframe(sigma, options)
        resolution = session.resolve(sigma)
        assert resolution.effective_strengths == tuple(
            hook.strength_model for hook in hooks
        )
        with (
            schedule_context.activate(resolution),
            spatial_context.activate(
                AnimaLoraSpatialInvocation(AnimaQueryMaskBatch(masks))
            ),
        ):
            actual = patch(inputs)
        if not isinstance(actual, torch.Tensor):
            raise AssertionError("Scheduled composition patch must return a tensor.")
        reference = torch.zeros_like(inputs)
        for target, region, hook in zip(targets, regions, hooks, strict=True):
            delta = (inputs @ target.adapter.down.T) @ target.adapter.up.T
            region_mask = masks[region].flatten(start_dim=1)
            reference = reference + (
                delta * region_mask.unsqueeze(-1) * hook.strength_model
            )
        observed.append(actual)
        expected.append(reference)
    return tuple(observed), tuple(expected)


def _execute_spatial(
    inputs: torch.Tensor,
    masks: torch.Tensor,
    executions: tuple[AnimaRegionalLoraAdapterExecution, ...],
) -> torch.Tensor:
    """Execute one composition under a published pointwise mask scope."""

    composition = AnimaRegionalLoraComposition(executions)
    spatial = AnimaLoraSpatialInvocationContext()
    schedule, resolution = static_lora_schedule(composition)
    target_name = executions[0].admission.targets[0].adapter.target
    patch = AnimaRegionalLoraCompositionLinearPatch(
        _CountingZeroLinear(1),
        composition.groups_for_target(target_name),
        weight_resolver=AnimaLoraWeightResolver(
            cross_attention_context=AnimaCrossAttentionInvocationContext(),
            branch_context=AnimaLoraBranchInvocationContext(),
            spatial_context=spatial,
        ),
        schedule_context=schedule,
    )
    with (
        schedule.activate(resolution),
        spatial.activate(AnimaLoraSpatialInvocation(AnimaQueryMaskBatch(masks))),
    ):
        output = patch(inputs)
    if not isinstance(output, torch.Tensor):
        raise AssertionError("Composition patch must return a tensor.")
    return output


def _spatial_reference(
    inputs: torch.Tensor,
    masks: torch.Tensor,
    executions: tuple[AnimaRegionalLoraAdapterExecution, ...],
) -> torch.Tensor:
    """Return explicit ordered full-rank output for both aligned CFG chunks."""

    result = torch.zeros_like(inputs)
    for execution in executions:
        target = execution.admission.targets[0].adapter
        delta = (inputs @ target.down.T) @ target.up.T
        region = masks[execution.adapter_plan.region_index].flatten(start_dim=1)
        expected_branch = execution.adapter_plan.branch
        gate = torch.tensor(
            [
                1.0 if expected_branch is RegionalLoraBranch.POSITIVE else 0.0,
                1.0 if expected_branch is RegionalLoraBranch.NEGATIVE else 0.0,
            ]
        )[:, None]
        result = result + delta * (region * gate).unsqueeze(-1) * float(
            execution.adapter_plan.model_strength
        )
    return result


def _tiny_target(*, down: float, up: float) -> AnimaLoraTarget:
    """Build one scalar self-attention target with distinct immutable weights."""

    name = "diffusion_model.blocks.0.self_attn.q_proj"
    adapter = StandardLoraTarget(
        name,
        torch.tensor([[down]]),
        torch.tensor([[up]]),
        rank=1,
        input_features=1,
        output_features=1,
    )
    return AnimaLoraTarget(0, AnimaLoraTargetFamily.SELF_ATTN_Q, adapter)


def _execution(
    index: int,
    target: AnimaLoraTarget | None,
    identity: str,
    region: int,
    strength: float,
    attention: AnimaRegionalAttentionExecution,
    model: object,
    cache: RegionalLoraExecutionCache,
    *,
    branch: RegionalLoraBranch = RegionalLoraBranch.POSITIVE,
    admission: AnimaLoraAdmission | None = None,
    schedule: tuple[RegionalLoraScheduleBoundary, ...] | None = None,
) -> AnimaRegionalLoraAdapterExecution:
    """Build one ordered use under shared composition authorities."""

    selected_admission = admission
    if selected_admission is None:
        if target is None:
            raise AssertionError("Synthetic execution requires one target.")
        selected_admission = AnimaLoraAdmission((target,))
    plan = RegionalLoraAdapterPlan(
        RegionalLoraAdapterIdentity(identity),
        composition_index=index,
        region_index=region,
        branch=branch,
        model_strength=strength,
        schedule=schedule
        if schedule is not None
        else (RegionalLoraScheduleBoundary(0.0, 100.0, 1.0, 0),),
    )
    return AnimaRegionalLoraAdapterExecution(
        plan,
        selected_admission,
        attention,
        ModelCloneLineage.from_model(model),
        cache,
    )


class _LinearSampling:
    """Convert authored schedule percentages to deterministic descending sigmas."""

    @staticmethod
    def percent_to_sigma(percent: float) -> float:
        """Return the exact synthetic sigma for one authored percentage."""

        return 100.0 * (1.0 - percent)


class _ScheduleModel:
    """Expose the model sampling boundary required by installed keyframes."""

    model_sampling = _LinearSampling()


def _attention(
    masks: torch.Tensor,
    *,
    include_negative: bool,
) -> AnimaRegionalAttentionExecution:
    """Build aligned contexts and canonical masks for one fidelity scenario."""

    batch = 2 if include_negative else 1
    context = torch.zeros((batch, 1, 1))
    chunks = [RegionalAttentionChunkBatch(0, RegionalAttentionBranch.POSITIVE, 0, 1)]
    if include_negative:
        chunks.append(
            RegionalAttentionChunkBatch(1, RegionalAttentionBranch.NEGATIVE, 1, 2)
        )
    if masks.ndim != 3:
        raise AssertionError("Fidelity masks must use R/H/W layout.")
    canonical = masks.float()
    return AnimaRegionalAttentionExecution(
        BatchedRegionalAttentionContexts(
            1,
            tuple(chunks),
            context,
            single_entry_regions(
                tuple(context.clone() for _ in range(int(canonical.shape[0])))
            ),
        ),
        RegionalMaskBank(
            canonical.clone(),
            canonical.clone(),
            int(canonical.shape[-1]),
            int(canonical.shape[-2]),
        ),
        (1.0,) * int(canonical.shape[0]),
    )
