"""Verify shared CFG and schedule context publication for standard UNet."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

import pytest
import torch

from simple_syrup.domain.conditioning_schedule import ConditioningScheduleRange
from simple_syrup.domain.processed_regional_attention import (
    ProcessedRegionalAttentionBranch,
    ProcessedRegionalAttentionContext,
    ProcessedRegionalAttentionEntry,
    ProcessedRegionalAttentionPlan,
)
from simple_syrup.domain.regional_attention import RegionalAttentionBranch
from simple_syrup.domain.regional_attention_batch import (
    BatchedRegionalAttentionContexts,
)
from simple_syrup.domain.regional_lora_plan import EMPTY_REGIONAL_LORA_PLAN
from simple_syrup.domain.regional_mask_bank import RegionalMaskBank
from simple_syrup.runtime.attention_coupling.unet_attention_context_wrapper import (
    StandardUnetAttentionContextDiffusionWrapper,
)
from simple_syrup.runtime.attention_coupling.unet_attention_state import (
    StandardUnetAttentionState,
)
from simple_syrup.runtime.regional_attention_diagnostics import (
    RegionalAttentionDiagnosticsBuilder,
)


class _DiffusionModel(torch.nn.Module):
    """Provide exact weak-referenceable standard-UNet model identity."""


@dataclass
class _Executor:
    """Capture task-local state from inside one nested diffusion call."""

    class_obj: object
    state: StandardUnetAttentionState
    fail: bool = False
    observed: BatchedRegionalAttentionContexts | None = None
    forwarded_context: torch.Tensor | None = None
    cache_size: int | None = None

    def __call__(self, *args: object, **__: object) -> torch.Tensor:
        """Capture current contexts and optionally fail inside the wrapper scope."""

        self.observed = self.state.execution_context.require_current()
        if len(args) < 3 or not isinstance(args[2], torch.Tensor):
            raise AssertionError("UNet executor requires a tensor base context.")
        self.forwarded_context = args[2]
        self.cache_size = self.state.resolution_cache.size
        if self.fail:
            raise RuntimeError("nested UNet failure")
        return torch.ones(2, 4, 2, 2)


def test_unet_context_wrapper_publishes_reversed_cfg_and_restores_state() -> None:
    """Reuse the shared resolver for exact current standard-UNet CFG order."""

    state = _state()
    model = _DiffusionModel()
    wrapper = StandardUnetAttentionContextDiffusionWrapper(model, state)
    executor = _Executor(model, state)
    negative = state.plan.negative.base_context.entries[0].cross_attention
    positive = state.plan.positive.base_context.entries[0].cross_attention

    output = wrapper(
        executor,
        torch.zeros(2, 4, 2, 2),
        torch.tensor([0.5, 0.5]),
        torch.cat((negative, positive)),
        None,
        None,
        {
            "cond_or_uncond": [1, 0],
            "sigmas": torch.tensor([0.5, 0.5]),
        },
    )

    assert isinstance(output, torch.Tensor)
    assert tuple(output.shape) == (2, 4, 2, 2)
    assert executor.observed is not None
    assert executor.forwarded_context is executor.observed.base_context
    assert executor.cache_size == 0
    assert [chunk.branch for chunk in executor.observed.chunks] == [
        RegionalAttentionBranch.NEGATIVE,
        RegionalAttentionBranch.POSITIVE,
    ]
    assert executor.observed.regions[0].entries[0].context[:, 0, 0].tolist() == [
        -2.0,
        2.0,
    ]
    with pytest.raises(RuntimeError, match="outside"):
        state.execution_context.require_current()
    with pytest.raises(RuntimeError, match="outside"):
        _ = state.resolution_cache.size


def test_unet_context_wrapper_clears_state_after_nested_failure() -> None:
    """Prevent failed UNet diffusion calls from leaking active plan state."""

    state = _state()
    model = _DiffusionModel()
    wrapper = StandardUnetAttentionContextDiffusionWrapper(model, state)
    positive = state.plan.positive.base_context.entries[0].cross_attention

    executor = _Executor(model, state, fail=True)
    with pytest.raises(RuntimeError, match="nested UNet failure"):
        wrapper(
            executor,
            torch.zeros(1, 4, 2, 2),
            torch.tensor([0.5]),
            positive,
            None,
            None,
            {
                "cond_or_uncond": [0],
                "sigmas": torch.tensor([0.5]),
            },
        )

    assert executor.observed is not None
    assert executor.forwarded_context is executor.observed.base_context
    assert executor.cache_size == 0
    with pytest.raises(RuntimeError, match="outside"):
        state.execution_context.require_current()
    with pytest.raises(RuntimeError, match="outside"):
        _ = state.resolution_cache.size


def test_unet_context_wrapper_rejects_model_identity_and_positional_drift() -> None:
    """Fail before plan resolution when Comfy's bound UNet contract changes."""

    state = _state()
    model = _DiffusionModel()
    wrapper = StandardUnetAttentionContextDiffusionWrapper(model, state)

    with pytest.raises(ValueError, match="does not own"):
        wrapper(
            _Executor(_DiffusionModel(), state),
            torch.zeros(1, 4, 2, 2),
            torch.tensor([0.5]),
            state.plan.positive.base_context.entries[0].cross_attention,
            None,
            None,
            {
                "cond_or_uncond": [0],
                "sigmas": torch.tensor([0.5]),
            },
        )
    with pytest.raises(TypeError, match="third positional"):
        wrapper(
            _Executor(model, state),
            torch.zeros(1, 4, 2, 2),
            torch.tensor([0.5]),
            None,
            None,
            {
                "cond_or_uncond": [0],
                "sigmas": torch.tensor([0.5]),
            },
        )
    with pytest.raises(TypeError, match="sixth positional"):
        wrapper(
            _Executor(model, state),
            torch.zeros(1, 4, 2, 2),
            torch.tensor([0.5]),
            state.plan.positive.base_context.entries[0].cross_attention,
            None,
            None,
            transformer_options={
                "cond_or_uncond": [0],
                "sigmas": torch.tensor([0.5]),
            },
        )


def test_unet_attention_state_rejects_strength_authority_mismatch() -> None:
    """Require one immutable region-strength value per canonical region."""

    plan = _plan()
    with pytest.raises(ValueError, match="count must match"):
        StandardUnetAttentionState(plan, (), _diagnostics(plan))


def test_unet_attention_state_rejects_foreign_diagnostic_mask_authority() -> None:
    """Prevent UNet diagnostics from observing a different canonical mask bank."""

    plan = _plan()
    foreign_plan = _plan()

    with pytest.raises(ValueError, match="mask-bank identity"):
        StandardUnetAttentionState(plan, (1.0,), _diagnostics(foreign_plan))


def _state() -> StandardUnetAttentionState:
    """Return one standard-UNet plan with distinct branch contexts."""

    plan = _plan()
    return StandardUnetAttentionState(plan, (1.0,), _diagnostics(plan))


def _diagnostics(
    plan: ProcessedRegionalAttentionPlan,
) -> RegionalAttentionDiagnosticsBuilder:
    """Bind test diagnostics to the plan's exact canonical mask authority."""

    return RegionalAttentionDiagnosticsBuilder(plan.mask_bank, backend="test.unet")


def _plan() -> ProcessedRegionalAttentionPlan:
    """Return one always-active positive/negative regional plan."""

    masks = torch.ones(1, 1, 1)
    return ProcessedRegionalAttentionPlan(
        ProcessedRegionalAttentionBranch(
            _context(0, None, 1.0),
            (_context(1, 0, 2.0),),
        ),
        ProcessedRegionalAttentionBranch(
            _context(0, None, -1.0),
            (_context(1, 0, -2.0),),
        ),
        RegionalMaskBank(masks, masks.clone(), 1, 1),
        EMPTY_REGIONAL_LORA_PLAN,
    )


def _context(
    conditioning_index: int,
    region_index: int | None,
    value: float,
) -> ProcessedRegionalAttentionContext:
    """Return one always-active model-consumed context."""

    return ProcessedRegionalAttentionContext(
        conditioning_index,
        region_index,
        (
            ProcessedRegionalAttentionEntry(
                0,
                uuid4(),
                ConditioningScheduleRange(None, None, None, None),
                torch.full((1, 2, 3), value),
                1.0,
            ),
        ),
    )
