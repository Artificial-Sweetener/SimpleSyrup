"""Verify dynamic per-resolution standard-UNet attn2 execution resolution."""

from __future__ import annotations

import json
import logging
from typing import Any
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
    BatchedRegionalAttentionEntry,
    BatchedRegionalAttentionRegion,
    RegionalAttentionChunkBatch,
)
from simple_syrup.domain.regional_lora_plan import EMPTY_REGIONAL_LORA_PLAN
from simple_syrup.domain.regional_mask_bank import RegionalMaskBank
from simple_syrup.domain.spatial_views import SpatialBatchLayout
from simple_syrup.masking.regional_mask_projection import (
    RegionalMaskForm,
    RegionalMaskProjectionMode,
)
from simple_syrup.runtime.attention_coupling.unet_attention_diagnostics import (
    StandardUnetAttentionDiagnosticsEmitter,
)
from simple_syrup.runtime.attention_coupling.unet_attention_state import (
    StandardUnetAttentionState,
)
from simple_syrup.runtime.attention_coupling.unet_attn2_execution_resolver import (
    StandardUnetAttn2ExecutionResolver,
)
from simple_syrup.runtime.regional_attention_diagnostic_values import (
    RegionalAttentionQueryGeometry,
)
from simple_syrup.runtime.regional_attention_diagnostics import (
    RegionalAttentionDiagnosticsBuilder,
)
from simple_syrup.runtime.regional_attention_query_masks import (
    RegionalAttentionQueryMaskBatch,
    RegionalAttentionQueryMaskProjector,
)


class _RecordHandler(logging.Handler):
    """Retain emitted log records without formatting side effects."""

    def __init__(self) -> None:
        """Initialize one empty record sink."""

        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        """Append one emitted record."""

        self.records.append(record)


class _CountingQueryMaskProjector(RegionalAttentionQueryMaskProjector):
    """Count real projection misses while delegating exact mask behavior."""

    def __init__(self) -> None:
        """Initialize an empty ordered query-grid record."""

        super().__init__()
        self.query_grids: list[tuple[int, int]] = []

    def project(
        self,
        *,
        bank: RegionalMaskBank,
        query_geometry: RegionalAttentionQueryGeometry,
        layout: SpatialBatchLayout,
        form: RegionalMaskForm,
        mode: RegionalMaskProjectionMode,
        device: torch.device,
        dtype: torch.dtype,
    ) -> RegionalAttentionQueryMaskBatch:
        """Record the query grid before performing the canonical projection."""

        self.query_grids.append(
            (query_geometry.query_height, query_geometry.query_width)
        )
        return super().project(
            bank=bank,
            query_geometry=query_geometry,
            layout=layout,
            form=form,
            mode=mode,
            device=device,
            dtype=dtype,
        )


def test_dynamic_resolver_projects_each_resolution_once_and_emits_diagnostics() -> None:
    """Reuse repeated blocks while resolving distinct rectangular grids."""

    state = _state()
    contexts = _contexts()
    handler = _RecordHandler()
    logger = logging.Logger("tests.standard_unet_diagnostics", level=logging.INFO)
    logger.addHandler(handler)
    query_masks = _CountingQueryMaskProjector()
    resolver = StandardUnetAttn2ExecutionResolver(
        state,
        query_masks=query_masks,
        diagnostics=StandardUnetAttentionDiagnosticsEmitter(logger),
    )

    with (
        state.execution_context.activate(contexts),
        state.resolution_cache.activate(),
    ):
        first = resolver.resolve(
            torch.zeros(2, 24, 32),
            contexts.base_context,
            _options([2, 32, 4, 6], block_index=0),
        )
        repeated = resolver.resolve(
            torch.ones(2, 24, 32),
            contexts.base_context,
            _options([2, 32, 4, 6], block_index=1),
        )
        reduced = resolver.resolve(
            torch.zeros(2, 6, 32),
            contexts.base_context,
            _options([2, 32, 2, 3], block_index=0),
        )

        assert repeated is first
        assert reduced is not first
        assert state.resolution_cache.size == 2

    assert first.query_masks.shape == (1, 2, 24)
    assert reduced.query_masks.shape == (1, 2, 6)
    assert first.contexts is contexts
    assert query_masks.query_grids == [(4, 6), (2, 3)]
    assert len(handler.records) == 2
    first_layer = handler.records[0].__dict__.get("unet_layer")
    first_snapshot = handler.records[0].__dict__.get("regional_diagnostics")
    assert isinstance(first_layer, dict)
    assert isinstance(first_snapshot, dict)
    assert first_layer["query_height"] == 4
    assert first_layer["query_width"] == 6
    assert first_snapshot["estimated_work"] == {
        "cross_attention_branch_multiplier": 2.0,
        "cross_attention_formula": "base_plus_region_count",
        "denoiser_call_multiplier": 1.0,
    }
    json.dumps(first_snapshot)
    with pytest.raises(RuntimeError, match="outside"):
        resolver.resolve(
            torch.zeros(2, 24, 32),
            contexts.base_context,
            _options([2, 32, 4, 6], block_index=0),
        )


def test_dynamic_resolver_requires_exact_wrapper_published_context_identity() -> None:
    """Reject equal copied context without a device-synchronizing comparison."""

    state = _state()
    contexts = _contexts()
    resolver = StandardUnetAttn2ExecutionResolver(state)

    with (
        state.execution_context.activate(contexts),
        state.resolution_cache.activate(),
    ):
        with pytest.raises(ValueError, match="active base authority"):
            resolver.resolve(
                torch.zeros(2, 24, 32),
                contexts.base_context.clone(),
                _options([2, 32, 4, 6], block_index=0),
            )

    assert torch.equal(contexts.base_context, contexts.base_context.clone())


def _options(
    activations_shape: list[int],
    *,
    block_index: int,
) -> dict[str, Any]:
    """Return complete installed callback metadata for one full model call."""

    return {
        "activations_shape": activations_shape,
        "original_shape": [2, 4, 8, 12],
        "block": ("input", 4),
        "block_index": block_index,
        "transformer_index": 2,
        "cond_or_uncond": [1, 0],
    }


def _state() -> StandardUnetAttentionState:
    """Return one processed plan and bound common diagnostics authority."""

    masks = torch.tensor(
        [
            [
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                for _ in range(8)
            ]
        ]
    )
    mask_bank = RegionalMaskBank(masks, masks.clone(), 12, 8)
    plan = ProcessedRegionalAttentionPlan(
        ProcessedRegionalAttentionBranch(
            _processed_context(0, None, 1.0),
            (_processed_context(1, 0, 2.0),),
        ),
        ProcessedRegionalAttentionBranch(
            _processed_context(0, None, -1.0),
            (_processed_context(1, 0, -2.0),),
        ),
        mask_bank,
        EMPTY_REGIONAL_LORA_PLAN,
    )
    return StandardUnetAttentionState(
        plan,
        (1.0,),
        RegionalAttentionDiagnosticsBuilder(
            mask_bank,
            backend="test.standard_unet",
        ),
    )


def _processed_context(
    conditioning_index: int,
    region_index: int | None,
    value: float,
) -> ProcessedRegionalAttentionContext:
    """Return one always-active processed context."""

    return ProcessedRegionalAttentionContext(
        conditioning_index,
        region_index,
        (
            ProcessedRegionalAttentionEntry(
                0,
                uuid4(),
                ConditioningScheduleRange(None, None, None, None),
                torch.full((1, 3, 16), value),
                1.0,
            ),
        ),
    )


def _contexts() -> BatchedRegionalAttentionContexts:
    """Return reversed CFG contexts aligned to the resolver plan."""

    return BatchedRegionalAttentionContexts(
        1,
        (
            RegionalAttentionChunkBatch(0, RegionalAttentionBranch.NEGATIVE, 0, 1),
            RegionalAttentionChunkBatch(1, RegionalAttentionBranch.POSITIVE, 1, 2),
        ),
        torch.cat(
            (
                torch.full((1, 3, 16), -1.0),
                torch.full((1, 3, 16), 1.0),
            )
        ),
        (
            BatchedRegionalAttentionRegion(
                0,
                (
                    BatchedRegionalAttentionEntry(
                        0,
                        torch.cat(
                            (
                                torch.full((1, 3, 16), -2.0),
                                torch.full((1, 3, 16), 2.0),
                            )
                        ),
                        (1.0, 1.0),
                    ),
                ),
            ),
        ),
    )
