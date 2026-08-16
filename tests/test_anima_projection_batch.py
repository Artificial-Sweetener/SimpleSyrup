# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Prove exact leader/follower batching across shared Anima projection inputs."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

import pytest
import torch

from simple_syrup.domain.regional_lora_plan import RegionalLoraAdapterIdentity
from simple_syrup.runtime.regional_lora.anima_projection_batch import (
    AnimaProjectionBatchRegistry,
    AnimaProjectionBatchRequest,
)
from simple_syrup.runtime.regional_lora.anima_targets import (
    AnimaLoraTargetFamily,
    expected_anima_lora_features,
)
from simple_syrup.runtime.regional_lora.delta_execution import (
    RegionalLoraDeltaExecutor,
)
from simple_syrup.runtime.regional_lora.execution_cache import (
    ModelCloneLineage,
    RegionalLoraExecutionCache,
)
from simple_syrup.runtime.regional_lora.preparation import (
    RegionalLoraCompatibleBatchPreparation,
    RegionalLoraTargetPreparation,
)
from simple_syrup.runtime.regional_lora.standard_adapter import StandardLoraTarget
from simple_syrup.runtime.regional_lora_schedule_resolution import (
    RegionalLoraScheduleResolution,
)


@dataclass
class _Participant:
    """Return one fixed request while recording leader coordination calls."""

    request: AnimaProjectionBatchRequest | None
    calls: int = 0

    def projection_batch_request(
        self,
        inputs: torch.Tensor,
        resolution: RegionalLoraScheduleResolution,
    ) -> AnimaProjectionBatchRequest | None:
        """Record the exact shared input and return configured work."""

        del inputs, resolution
        self.calls += 1
        return self.request


class _NoMaterializedDeltaExecutor(RegionalLoraDeltaExecutor):
    """Fail if coordinated execution recreates complete output deltas."""

    def compatible_deltas(
        self,
        inputs: torch.Tensor,
        *,
        preparation: RegionalLoraCompatibleBatchPreparation,
        multipliers: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        """Reject the superseded full-delta coordination path."""

        del inputs, preparation, multipliers
        raise AssertionError("coordinated execution must retain rank values only")


def test_registry_batches_leader_and_reuses_follower_delta_for_same_input() -> None:
    """Match independent full-rank math and avoid follower recomputation."""

    inputs = torch.tensor([[[1.0, -0.5], [0.25, 2.0]]])
    multiplier = torch.tensor([[1.0, 0.25]])
    first = _participant(
        torch.tensor([[1.0, 0.5], [-0.5, 1.0]]),
        torch.tensor([[1.0, 0.25], [0.5, -1.0]]),
        multiplier,
    )
    second = _participant(
        torch.tensor([[0.25, 1.0], [1.5, -0.5]]),
        torch.tensor([[0.75, -0.25], [1.0, 0.5]]),
        multiplier,
    )
    registry = AnimaProjectionBatchRegistry(_NoMaterializedDeltaExecutor())
    registry.register("shared", (first, second))
    resolution = RegionalLoraScheduleResolution((1.0,), (1.0,))

    first_execution = registry.resolve_execution(first, inputs, resolution)
    second_execution = registry.resolve_execution(second, inputs, resolution)

    assert first_execution is not None
    assert second_execution is not None
    first_delta = registry.accumulate(torch.zeros_like(inputs), first_execution)
    second_delta = registry.accumulate(torch.zeros_like(inputs), second_execution)
    first_target = first.request
    second_target = second.request
    assert first_target is not None
    assert second_target is not None
    expected_first = (
        (inputs @ first_target.preparation.target.down.T) * multiplier.unsqueeze(-1)
    ) @ first_target.preparation.target.up.T
    expected_second = (
        (inputs @ second_target.preparation.target.down.T) * multiplier.unsqueeze(-1)
    ) @ second_target.preparation.target.up.T
    torch.testing.assert_close(first_delta, expected_first)
    torch.testing.assert_close(second_delta, expected_second)
    assert first.calls == 1
    assert second.calls == 1

    assert registry.resolve_execution(second, inputs.clone(), resolution) is None
    assert second.calls == 1


def test_registry_declines_whole_group_when_one_member_is_inactive() -> None:
    """Keep all members on their independent path for partial group activity."""

    active = _participant(
        torch.ones((1, 2)),
        torch.ones((2, 1)),
        torch.ones((1, 2)),
    )
    inactive = _Participant(None)
    registry = AnimaProjectionBatchRegistry()
    registry.register("partial", (active, inactive))

    assert (
        registry.resolve_execution(
            active,
            torch.ones((1, 2, 2)),
            RegionalLoraScheduleResolution((1.0,), (1.0,)),
        )
        is None
    )
    assert active.calls == 1
    assert inactive.calls == 1


@pytest.mark.parametrize(
    "families",
    (
        ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
        (
            "adaln_modulation_self_attn.1",
            "adaln_modulation_cross_attn.1",
            "adaln_modulation_mlp.1",
        ),
    ),
)
def test_bf16_projection_batch_retains_reference_lora_output_quality(
    families: tuple[str, str, str],
) -> None:
    """Bound rank-32 batching error against independent generated projections."""

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the BF16 projection comparison.")
    selected_families = tuple(AnimaLoraTargetFamily(family) for family in families)
    generator_cpu = torch.Generator().manual_seed(9157)
    adapters = tuple(
        _generated_adapter(family, generator=generator_cpu)
        for family in selected_families
    )
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(9157)
    inputs = torch.randn(
        (2, 257, 2048),
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    multiplier = torch.zeros((2, 257), device=device, dtype=torch.bfloat16)
    multiplier[:, :129] = 1.0
    participants = tuple(
        _participant(
            adapter.down,
            adapter.up,
            multiplier,
        )
        for adapter in adapters
    )
    registry = AnimaProjectionBatchRegistry()
    registry.register("pinned-qkv", participants)
    resolution = RegionalLoraScheduleResolution((1.0,), (1.0,))

    observed_values: list[torch.Tensor] = []
    for participant, adapter in zip(participants, adapters, strict=True):
        execution = registry.resolve_execution(participant, inputs, resolution)
        if execution is not None:
            original_output = inputs.new_zeros(
                (*inputs.shape[:-1], adapter.output_features)
            )
            observed_values.append(registry.accumulate(original_output, execution))
    observed = tuple(observed_values)
    expected = tuple(
        (
            (inputs @ adapter.down.to(device=device, dtype=inputs.dtype).T)
            * multiplier.unsqueeze(-1)
        )
        @ adapter.up.to(device=device, dtype=inputs.dtype).T
        for adapter in adapters
    )
    torch.cuda.synchronize(device)

    assert len(observed) == len(expected) == 3
    for actual, reference in zip(observed, expected, strict=True):
        difference = (actual.float() - reference.float()).flatten()
        relative_l2 = torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(
            reference.float().flatten()
        )
        cosine = torch.nn.functional.cosine_similarity(
            actual.float().flatten(),
            reference.float().flatten(),
            dim=0,
        )
        assert float(relative_l2.item()) <= 0.005
        assert float(cosine.item()) >= 0.99999


def _generated_adapter(
    family: AnimaLoraTargetFamily,
    *,
    generator: torch.Generator,
) -> StandardLoraTarget:
    """Build one deterministic rank-32 adapter for an architecture-owned family."""

    input_features, output_features = expected_anima_lora_features(family)
    rank = 32
    return StandardLoraTarget(
        f"diffusion_model.blocks.0.{family.value}",
        torch.randn((rank, input_features), generator=generator) * 0.01,
        torch.randn((output_features, rank), generator=generator) * 0.01,
        rank,
        input_features,
        output_features,
    )


def _participant(
    down: torch.Tensor,
    up: torch.Tensor,
    multiplier: torch.Tensor,
) -> _Participant:
    """Build one compatible prepared target and fixed batch request."""

    return _Participant(_request(down, up, multiplier))


def _request(
    down: torch.Tensor,
    up: torch.Tensor,
    multiplier: torch.Tensor,
) -> AnimaProjectionBatchRequest:
    """Build one compatible prepared target and batch request."""

    target = StandardLoraTarget(
        "target",
        down,
        up,
        int(down.shape[0]),
        int(down.shape[1]),
        int(up.shape[0]),
    )
    preparation = RegionalLoraTargetPreparation(
        RegionalLoraAdapterIdentity(str(uuid4())),
        ModelCloneLineage(uuid4(), uuid4()),
        target,
        RegionalLoraExecutionCache(),
    )
    return AnimaProjectionBatchRequest(preparation, multiplier)
