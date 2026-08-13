# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify exact model-neutral regional Linear execution semantics."""

from __future__ import annotations

from typing import cast
from uuid import uuid4

import pytest
import torch
import torch.nn.functional as functional
from comfy.weight_adapter.lora import LoRAAdapter
from torch import nn

from simple_syrup.domain.regional_activation_geometry import (
    RegionalActivationBatchAlignment,
    RegionalActivationGeometry,
    RegionalActivationLayout,
)
from simple_syrup.domain.regional_lora_plan import RegionalLoraAdapterIdentity
from simple_syrup.masking.regional_activation_mask_projection import (
    RegionalActivationMaskBatch,
)
from simple_syrup.runtime.regional_lora.execution_cache import (
    ModelCloneLineage,
    RegionalLoraExecutionCache,
)
from simple_syrup.runtime.regional_lora.linear_execution import RegionalLinearExecutor
from simple_syrup.runtime.regional_lora.linear_execution_plan import (
    RegionalLinearExecutionPlan,
    RegionalLinearOperationKey,
    RegionalLinearTargetUse,
)
from simple_syrup.runtime.regional_lora.preparation import (
    RegionalLoraTargetPreparation,
)
from simple_syrup.runtime.regional_lora.standard_adapter import StandardLoraTarget


def test_zero_mask_returns_exact_original_and_performs_no_projection() -> None:
    """Call the original once and avoid preparing inactive adapter weights."""

    fixture = _fixture()
    plan = _plan((_use(fixture, composition=0, region=0),))
    original = _Original(fixture.base)
    masks = _masks(torch.zeros((1, 2, 6)), features=4)

    result = RegionalLinearExecutor().execute(
        original,
        fixture.inputs,
        "marker",
        plan=plan,
        masks=masks,
        schedule_strengths=(1.0,),
        expected="value",
    )

    expected = fixture.base(fixture.inputs)
    assert torch.equal(result, expected)
    assert original.calls == [("marker", "value")]
    assert fixture.cache.size == 0


def test_all_one_mask_matches_ordinary_global_lora_math() -> None:
    """Match the unmerged full-rank additive LoRA reference over the whole image."""

    fixture = _fixture()
    plan = _plan((_use(fixture, composition=0, region=0, strength=0.75),))

    result = _execute(fixture, plan, torch.ones((1, 2, 6)), (0.5,))

    scale = 0.75 * 0.5 * fixture.target.intrinsic_scale
    expected = fixture.base(fixture.inputs) + _delta(fixture, fixture.inputs) * scale
    torch.testing.assert_close(result, expected, rtol=1e-6, atol=1e-7)


def test_all_one_mask_matches_comfy_calculated_weight() -> None:
    """Match Comfy's installed ordinary-LoRA weight calculation at full coverage."""

    fixture = _fixture()
    strength = 0.75
    schedule = 0.5
    plan = _plan((_use(fixture, composition=0, region=0, strength=strength),))

    result = _execute(fixture, plan, torch.ones((1, 2, 6)), (schedule,))

    adapter = LoRAAdapter(
        {"up", "down"},
        (
            fixture.target.up,
            fixture.target.down,
            fixture.target.rank * fixture.target.intrinsic_scale,
            None,
            None,
            None,
        ),
    )
    patched_weight = adapter.calculate_weight(
        fixture.base.weight.detach().clone(),
        "diffusion_model.layer.weight",
        strength * schedule,
        1.0,
        None,
        lambda value: value,
        intermediate_dtype=torch.float32,
    )
    expected = functional.linear(fixture.inputs, patched_weight, fixture.base.bias)
    torch.testing.assert_close(result, expected, rtol=1e-6, atol=1e-7)


def test_complementary_regions_reconstruct_global_application() -> None:
    """Partition one adapter spatially without changing its complete-image result."""

    fixture = _fixture()
    uses = (
        _use(fixture, composition=0, region=0),
        _use(fixture, composition=1, region=1),
    )
    plan = _plan(uses)
    left = torch.zeros((2, 6))
    left[:, :3] = 1.0
    masks = torch.stack((left, 1.0 - left))

    result = _execute(fixture, plan, masks, (1.0, 1.0))

    expected = fixture.base(fixture.inputs) + _delta(fixture, fixture.inputs) * (
        fixture.target.intrinsic_scale
    )
    torch.testing.assert_close(result, expected, rtol=1e-6, atol=1e-7)
    assert len(plan.groups) == 1


def test_fractional_schedule_matches_rank_space_reference() -> None:
    """Apply authored soft coverage and schedule scale before the up projection."""

    fixture = _fixture()
    plan = _plan((_use(fixture, composition=0, region=0, strength=0.8),))
    mask = torch.linspace(0.0, 1.0, 12).reshape(1, 2, 6)

    result = _execute(fixture, plan, mask, (0.25,))

    rank = functional.linear(fixture.inputs, fixture.target.down)
    multiplier = mask[0] * 0.8 * 0.25 * fixture.target.intrinsic_scale
    expected = fixture.base(fixture.inputs) + functional.linear(
        rank * multiplier.unsqueeze(-1),
        fixture.target.up,
    )
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


def test_different_adapters_preserve_declared_addition_order() -> None:
    """Accumulate non-contiguous targets in exact composition order."""

    fixture = _fixture()
    second = _target(
        "second",
        torch.tensor([[0.3, -0.2, 0.1, 0.4], [-0.1, 0.2, 0.5, -0.3]]),
        torch.tensor([[0.2, 0.1], [-0.4, 0.2], [0.3, -0.2]]),
    )
    uses = (
        _use(fixture, composition=0, region=0),
        _use(fixture, composition=1, region=1, target=second),
        _use(fixture, composition=2, region=0),
    )
    plan = _plan(uses)
    masks = torch.stack(
        (
            torch.full((2, 6), 0.25),
            torch.full((2, 6), 0.75),
        )
    )

    result = _execute(fixture, plan, masks, (1.0, 0.5, 0.25))

    expected = fixture.base(fixture.inputs)
    for target, mask, schedule in (
        (fixture.target, masks[0], 1.0),
        (second, masks[1], 0.5),
        (fixture.target, masks[0], 0.25),
    ):
        rank = functional.linear(fixture.inputs, target.down)
        expected = expected + functional.linear(
            rank * (mask * schedule * target.intrinsic_scale).unsqueeze(-1),
            target.up,
        )
    torch.testing.assert_close(result, expected, rtol=1e-6, atol=1e-7)
    assert len(plan.groups) == 3


def test_repeated_adapter_combination_matches_separate_ordered_uses() -> None:
    """Combine adjacent exact targets without changing fractional spatial math."""

    fixture = _fixture()
    uses = (
        _use(fixture, composition=0, region=0, strength=0.5),
        _use(fixture, composition=1, region=1, strength=0.25),
    )
    plan = _plan(uses)
    masks = torch.stack((torch.full((2, 6), 0.4), torch.full((2, 6), 0.7)))

    result = _execute(fixture, plan, masks, (0.8, 0.6))

    rank = functional.linear(fixture.inputs, fixture.target.down)
    combined = (
        masks[0] * 0.5 * 0.8 + masks[1] * 0.25 * 0.6
    ) * fixture.target.intrinsic_scale
    expected = fixture.base(fixture.inputs) + functional.linear(
        rank * combined.unsqueeze(-1),
        fixture.target.up,
    )
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    assert len(plan.groups) == 1
    assert fixture.cache.size == 1


def test_failure_and_clear_release_local_preparation_state() -> None:
    """Validate before preparation and release successful preparation explicitly."""

    fixture = _fixture()
    plan = _plan((_use(fixture, composition=0, region=0),))
    wrong = _masks(torch.ones((1, 1, 6)), features=4)
    with pytest.raises(ValueError, match="input must match"):
        RegionalLinearExecutor().execute(
            fixture.base,
            fixture.inputs,
            plan=plan,
            masks=wrong,
            schedule_strengths=(1.0,),
        )
    assert fixture.cache.size == 0

    _execute(fixture, plan, torch.ones((1, 2, 6)), (1.0,))
    assert fixture.cache.size == 1
    plan.clear()
    assert all(not preparation._prepared for preparation in _preparations(plan))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cuda_low_precision_matches_explicit_execution_dtype_reference(
    dtype: torch.dtype,
) -> None:
    """Preserve full-rank FP16/BF16 adapter math on the installed GPU."""

    fixture = _fixture(device=torch.device("cuda"), dtype=dtype)
    plan = _plan((_use(fixture, composition=0, region=0),))
    mask = torch.linspace(0.0, 1.0, 12, device="cuda", dtype=dtype).reshape(1, 2, 6)

    result = _execute(fixture, plan, mask, (1.0,))

    rank = functional.linear(fixture.inputs, fixture.target.down.to("cuda", dtype))
    expected = fixture.base(fixture.inputs) + functional.linear(
        rank * (mask[0] * fixture.target.intrinsic_scale).unsqueeze(-1),
        fixture.target.up.to("cuda", dtype),
    )
    torch.testing.assert_close(result, expected, rtol=0.01, atol=0.01)


class _Fixture:
    """Retain deterministic Linear execution values for one device and dtype."""

    def __init__(self, device: torch.device, dtype: torch.dtype) -> None:
        """Create base, input, adapter, lineage, and cache state."""

        self.base = nn.Linear(4, 3, bias=True, device=device, dtype=dtype)
        with torch.no_grad():
            self.base.weight.copy_(
                torch.tensor(
                    [
                        [0.1, 0.2, -0.3, 0.4],
                        [0.5, -0.2, 0.1, 0.3],
                        [-0.4, 0.2, 0.6, -0.1],
                    ],
                    device=device,
                    dtype=dtype,
                )
            )
            self.base.bias.copy_(
                torch.tensor([0.1, -0.2, 0.3], device=device, dtype=dtype)
            )
        self.inputs = torch.linspace(
            -1.0,
            1.0,
            48,
            device=device,
            dtype=dtype,
        ).reshape(2, 6, 4)
        self.target = _target(
            "first",
            torch.tensor([[0.2, -0.1, 0.3, 0.4], [-0.3, 0.5, 0.2, -0.2]]),
            torch.tensor([[0.4, -0.2], [0.1, 0.3], [-0.5, 0.2]]),
        )
        self.lineage = ModelCloneLineage(uuid4(), uuid4())
        self.cache = RegionalLoraExecutionCache()


class _Original:
    """Record forwarded arguments around one real base Linear operation."""

    def __init__(self, base: nn.Linear) -> None:
        """Retain the base and an empty call journal."""

        self._base = base
        self.calls: list[tuple[object, object]] = []

    def __call__(
        self,
        inputs: torch.Tensor,
        marker: object,
        *,
        expected: object,
    ) -> torch.Tensor:
        """Record positional/keyword forwarding and execute the base."""

        self.calls.append((marker, expected))
        return cast(torch.Tensor, self._base(inputs))


def _fixture(
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> _Fixture:
    """Return one deterministic execution fixture."""

    return _Fixture(device or torch.device("cpu"), dtype)


def _target(
    name: str,
    down: torch.Tensor,
    up: torch.Tensor,
) -> StandardLoraTarget:
    """Return one immutable CPU ordinary-LoRA target."""

    return StandardLoraTarget(name, down, up, 2, 4, 3, 0.5)


def _use(
    fixture: _Fixture,
    *,
    composition: int,
    region: int,
    strength: float = 1.0,
    target: StandardLoraTarget | None = None,
) -> RegionalLinearTargetUse:
    """Return one ordered use sharing exact tensor identities where requested."""

    selected = target or fixture.target
    identity = RegionalLoraAdapterIdentity(f"{selected.target}.safetensors")
    return RegionalLinearTargetUse(
        composition,
        region,
        RegionalLinearOperationKey(
            identity,
            "diffusion_model.layer.weight",
            id(selected.down),
            id(selected.up),
        ),
        RegionalLoraTargetPreparation(
            identity,
            fixture.lineage,
            selected,
            fixture.cache,
        ),
        strength * selected.intrinsic_scale,
    )


def _plan(uses: tuple[RegionalLinearTargetUse, ...]) -> RegionalLinearExecutionPlan:
    """Build one generic execution plan from ordered uses."""

    return RegionalLinearExecutionPlan(uses)


def _masks(values: torch.Tensor, *, features: int) -> RegionalActivationMaskBatch:
    """Wrap region/B/S values in explicit consumer-spatialized geometry."""

    _, batch, tokens = values.shape
    geometry = RegionalActivationGeometry(
        RegionalActivationLayout.CONSUMER_SPATIALIZED,
        (batch, tokens, features),
        2,
        2,
        tokens // 2,
        RegionalActivationBatchAlignment(batch, 1),
    )
    return RegionalActivationMaskBatch(values.unsqueeze(-1), geometry)


def _execute(
    fixture: _Fixture,
    plan: RegionalLinearExecutionPlan,
    masks: torch.Tensor,
    schedules: tuple[float, ...],
) -> torch.Tensor:
    """Run one complete generic Linear execution case."""

    return RegionalLinearExecutor().execute(
        fixture.base,
        fixture.inputs,
        plan=plan,
        masks=_masks(masks.to(fixture.inputs), features=4),
        schedule_strengths=schedules,
    )


def _delta(fixture: _Fixture, inputs: torch.Tensor) -> torch.Tensor:
    """Return the explicit unscaled ordinary-LoRA delta."""

    down = fixture.target.down.to(inputs)
    up = fixture.target.up.to(inputs)
    return functional.linear(functional.linear(inputs, down), up)


def _preparations(
    plan: RegionalLinearExecutionPlan,
) -> tuple[RegionalLoraTargetPreparation, ...]:
    """Return every group-local preparation for lifecycle assertions."""

    return tuple(group.preparation for group in plan.groups)
