"""Verify dynamic regional state reaches the installed UNet attn2 callbacks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

import pytest
import torch
from comfy.ldm.modules.attention import BasicTransformerBlock
from torch import nn

from simple_syrup.domain.conditioning_schedule import ConditioningScheduleRange
from simple_syrup.domain.processed_regional_attention import (
    ProcessedRegionalAttentionBranch,
    ProcessedRegionalAttentionContext,
    ProcessedRegionalAttentionEntry,
    ProcessedRegionalAttentionPlan,
)
from simple_syrup.domain.regional_attention_batch import (
    BatchedRegionalAttentionContexts,
)
from simple_syrup.domain.regional_lora_plan import EMPTY_REGIONAL_LORA_PLAN
from simple_syrup.domain.regional_mask_bank import RegionalMaskBank
from simple_syrup.domain.spatial_views import (
    SpatialBatchLayout,
    SpatialView,
    SpatialViewKind,
)
from simple_syrup.runtime.attention_coupling.unet_attention_context_wrapper import (
    StandardUnetAttentionContextDiffusionWrapper,
)
from simple_syrup.runtime.attention_coupling.unet_attention_state import (
    StandardUnetAttentionState,
)
from simple_syrup.runtime.attention_coupling.unet_attn2_execution import (
    UnetAttn2Execution,
)
from simple_syrup.runtime.attention_coupling.unet_attn2_patch import (
    UnetAttn2PatchPair,
)
from simple_syrup.runtime.regional_attention_diagnostic_values import (
    RegionalAttentionExecutionDiagnostics,
    RegionalAttentionQueryGeometry,
)
from simple_syrup.runtime.regional_attention_diagnostics import (
    RegionalAttentionDiagnosticsBuilder,
)


class _DiffusionModel(torch.nn.Module):
    """Provide exact weak-referenceable standard-UNet model identity."""


class _ZeroAttention(nn.Module):
    """Retain one self-attention trajectory without changing its input."""

    def __init__(self) -> None:
        """Initialize the invocation count."""

        super().__init__()
        self.calls = 0

    def forward(
        self,
        query: torch.Tensor,
        *,
        context: torch.Tensor | None,
        value: torch.Tensor | None,
        transformer_options: dict[str, Any],
    ) -> torch.Tensor:
        """Return zero after observing one installed self-attention call."""

        del context, value, transformer_options
        self.calls += 1
        return torch.zeros_like(query)


class _ContextAttention(nn.Module):
    """Return each compact branch context through one native attention call."""

    def __init__(self) -> None:
        """Initialize the invocation count."""

        super().__init__()
        self.calls = 0

    def forward(
        self,
        query: torch.Tensor,
        *,
        context: torch.Tensor | None,
        value: torch.Tensor | None,
        transformer_options: dict[str, Any],
    ) -> torch.Tensor:
        """Broadcast the model-consumed context value over query tokens."""

        del value, transformer_options
        self.calls += 1
        if context is None:
            raise AssertionError("Dynamic UNet test requires cross-attention context.")
        return context.mean(dim=1, keepdim=True).expand_as(query)


class _ZeroFeedForward(nn.Module):
    """Retain one feed-forward trajectory without changing its input."""

    def __init__(self) -> None:
        """Initialize the invocation count."""

        super().__init__()
        self.calls = 0

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Return zero after observing one installed feed-forward call."""

        self.calls += 1
        return torch.zeros_like(value)


@dataclass
class _ContextBoundResolver:
    """Resolve callbacks only from the wrapper-published active call state."""

    state: StandardUnetAttentionState
    observed: list[BatchedRegionalAttentionContexts] = field(default_factory=list)
    diagnostics: list[RegionalAttentionExecutionDiagnostics] = field(
        default_factory=list
    )

    def resolve(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        extra_options: dict[str, Any],
    ) -> UnetAttn2Execution:
        """Build one execution from current contexts without schedule logic."""

        del query, context, extra_options
        contexts = self.state.execution_context.require_current()
        self.observed.append(contexts)
        self.diagnostics.append(
            self.state.diagnostics.build(
                contexts,
                RegionalAttentionQueryGeometry(2, 1, 1, 2, None),
                _layout(),
            )
        )
        return UnetAttn2Execution(
            contexts,
            torch.tensor([[[1.0, 0.0], [1.0, 0.0]]]),
            self.state.region_strengths,
        )


@dataclass
class _NestedUnetExecutor:
    """Run the installed transformer block inside one diffusion wrapper call."""

    class_obj: object
    block: BasicTransformerBlock
    patches: UnetAttn2PatchPair
    query: torch.Tensor

    def __call__(self, *args: object, **kwargs: object) -> torch.Tensor:
        """Forward the wrapper-resolved context through the paired callbacks."""

        if len(args) < 3 or not isinstance(args[2], torch.Tensor):
            raise AssertionError("Nested UNet executor requires positional context.")
        del kwargs
        options = args[5] if len(args) > 5 else None
        if not isinstance(options, dict):
            raise AssertionError("Nested UNet executor requires transformer options.")
        result = self.block(
            self.query,
            context=args[2],
            transformer_options={
                **options,
                "patches": {
                    "attn2_patch": [self.patches.input_patch],
                    "attn2_output_patch": [self.patches.output_patch],
                },
            },
        )
        if not isinstance(result, torch.Tensor):
            raise AssertionError("Installed UNet block returned a non-tensor output.")
        return result


def test_dynamic_unet_callbacks_consume_current_schedule_and_cfg_state() -> None:
    """Resolve schedules once per call and retain one transformer trajectory."""

    plan = _plan()
    state = StandardUnetAttentionState(
        plan,
        (1.0,),
        RegionalAttentionDiagnosticsBuilder(plan.mask_bank, backend="test.unet"),
    )
    resolver = _ContextBoundResolver(state)
    patches = UnetAttn2PatchPair(resolver)
    block, self_attention, cross_attention, feed_forward = _block()
    model = _DiffusionModel()
    wrapper = StandardUnetAttentionContextDiffusionWrapper(model, state)
    executor = _NestedUnetExecutor(
        model,
        block,
        patches,
        torch.tensor([[[10.0], [20.0]], [[10.0], [20.0]]]),
    )

    high_result = _invoke(
        wrapper,
        executor,
        sigma=0.75,
        selectors=[1, 0],
        base_values=(-1.0, 1.0),
    )
    low_result = _invoke(
        wrapper,
        executor,
        sigma=0.25,
        selectors=[0, 1],
        base_values=(1.0, -1.0),
    )

    assert torch.equal(
        high_result,
        torch.tensor([[[8.0], [19.0]], [[12.0], [21.0]]]),
    )
    assert torch.equal(
        low_result,
        torch.tensor([[[14.0], [21.0]], [[6.0], [19.0]]]),
    )
    assert [
        entry.context[:, 0, 0].tolist()
        for entry in resolver.observed[0].regions[0].entries
    ] == [[-2.0, 2.0]]
    assert [
        entry.context[:, 0, 0].tolist()
        for entry in resolver.observed[1].regions[0].entries
    ] == [[4.0, -4.0]]
    assert self_attention.calls == 2
    assert cross_attention.calls == 2
    assert feed_forward.calls == 2
    assert [
        snapshot.cross_attention_branch_multiplier for snapshot in resolver.diagnostics
    ] == [
        2.0,
        2.0,
    ]
    assert all(
        snapshot.denoiser_call_multiplier == 1.0 for snapshot in resolver.diagnostics
    )
    with pytest.raises(RuntimeError, match="outside"):
        state.execution_context.require_current()


def _invoke(
    wrapper: StandardUnetAttentionContextDiffusionWrapper,
    executor: _NestedUnetExecutor,
    *,
    sigma: float,
    selectors: list[int],
    base_values: tuple[float, float],
) -> torch.Tensor:
    """Invoke one dynamic diffusion call with exact CFG and sigma metadata."""

    batch = len(selectors)
    output = wrapper(
        executor,
        torch.zeros(batch, 4, 2, 2),
        torch.full((batch,), sigma),
        torch.tensor(base_values).reshape(batch, 1, 1),
        None,
        None,
        {
            "cond_or_uncond": selectors,
            "sigmas": torch.full((batch,), sigma),
        },
    )
    if not isinstance(output, torch.Tensor):
        raise AssertionError("Dynamic UNet wrapper returned a non-tensor output.")
    return output


def _block() -> tuple[
    BasicTransformerBlock,
    _ZeroAttention,
    _ContextAttention,
    _ZeroFeedForward,
]:
    """Build one installed block with observable retained trajectories."""

    block = BasicTransformerBlock(
        dim=1,
        n_heads=1,
        d_head=1,
        context_dim=1,
        checkpoint=False,
    )
    self_attention = _ZeroAttention()
    cross_attention = _ContextAttention()
    feed_forward = _ZeroFeedForward()
    block.norm1 = nn.Identity()
    block.attn1 = self_attention
    block.norm2 = nn.Identity()
    block.attn2 = cross_attention
    block.norm3 = nn.Identity()
    block.ff = feed_forward
    return block, self_attention, cross_attention, feed_forward


def _plan() -> ProcessedRegionalAttentionPlan:
    """Return one plan whose regional context changes at sigma 0.5."""

    masks = torch.ones(1, 1, 2)
    return ProcessedRegionalAttentionPlan(
        ProcessedRegionalAttentionBranch(
            _context(0, None, ((1.0, _unbounded_schedule()),)),
            (
                _context(
                    1,
                    0,
                    ((2.0, _schedule(1.0, 0.5)), (4.0, _schedule(0.5, 0.0))),
                ),
            ),
        ),
        ProcessedRegionalAttentionBranch(
            _context(0, None, ((-1.0, _unbounded_schedule()),)),
            (
                _context(
                    1,
                    0,
                    ((-2.0, _schedule(1.0, 0.5)), (-4.0, _schedule(0.5, 0.0))),
                ),
            ),
        ),
        RegionalMaskBank(masks, masks.clone(), 2, 1),
        EMPTY_REGIONAL_LORA_PLAN,
    )


def _layout() -> SpatialBatchLayout:
    """Return the exact full-canvas layout for the test query grid."""

    return SpatialBatchLayout(
        2,
        1,
        (SpatialView(SpatialViewKind.FULL, 0, 0, 2, 1, 2, 1),),
        1,
    )


def _context(
    conditioning_index: int,
    region_index: int | None,
    values: tuple[tuple[float, ConditioningScheduleRange], ...],
) -> ProcessedRegionalAttentionContext:
    """Return one scheduled processed context with stable entry order."""

    return ProcessedRegionalAttentionContext(
        conditioning_index,
        region_index,
        tuple(
            ProcessedRegionalAttentionEntry(
                index,
                uuid4(),
                schedule,
                torch.tensor([[[value]]]),
                1.0,
            )
            for index, (value, schedule) in enumerate(values)
        ),
    )


def _schedule(start: float, end: float) -> ConditioningScheduleRange:
    """Return one converted-sigma schedule interval."""

    return ConditioningScheduleRange(None, None, start, end)


def _unbounded_schedule() -> ConditioningScheduleRange:
    """Return one always-active conditioning schedule."""

    return ConditioningScheduleRange(None, None, None, None)
