# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Prove compact Anima predictions preserve both tiled fusion policies."""

from __future__ import annotations

from typing import Any, cast

import pytest
import torch
from regional_attention_test_values import single_entry_regions
from torch import nn

from simple_syrup.domain.regional_attention import RegionalAttentionBranch
from simple_syrup.domain.regional_attention_batch import (
    BatchedRegionalAttentionContexts,
    RegionalAttentionChunkBatch,
)
from simple_syrup.domain.regional_mask_bank import RegionalMaskBank
from simple_syrup.domain.spatial_views import SpatialBatchLayout
from simple_syrup.domain.tiled_diffusion import build_tiled_diffusion_plan
from simple_syrup.masking.regional_mask_projection import (
    RegionalMaskForm,
    RegionalMaskProjectionMode,
)
from simple_syrup.runtime.regional_lora.anima_activation_context import (
    AnimaActivationContext,
    AnimaActivationGeometry,
)
from simple_syrup.runtime.regional_lora.anima_attention_execution import (
    AnimaRegionalAttentionExecution,
)
from simple_syrup.runtime.regional_lora.anima_branch_batch import (
    AnimaRegionalBranchInvocation,
)
from simple_syrup.runtime.regional_lora.anima_composition_phase import (
    AnimaCompositionPhase,
    AnimaCompositionStage,
)
from simple_syrup.runtime.regional_lora.anima_composition_phase_context import (
    AnimaCompositionPhaseContext,
)
from simple_syrup.runtime.regional_lora.anima_cross_attention import (
    AnimaRegionalCrossAttentionPatch,
)
from simple_syrup.runtime.regional_lora.anima_cross_attention_context import (
    AnimaCrossAttentionInvocationContext,
)
from simple_syrup.runtime.regional_lora.anima_cross_attention_weights import (
    ANIMA_CROSS_ATTENTION_WEIGHTING_POLICY,
)
from simple_syrup.runtime.regional_lora.anima_query_masks import (
    AnimaQueryMaskProjector,
)
from simple_syrup.runtime.spatial_model_arguments import (
    SIMPLE_SYRUP_TRANSFORMER_NAMESPACE,
    SPATIAL_BATCH_LAYOUT_KEY,
)
from simple_syrup.runtime.tile_prediction_accumulation import (
    TilePredictionAccumulator,
)


class _ContextValueAttention(nn.Module):
    """Return each compact row's context scalar and record branch identity."""

    def __init__(self, invocation: AnimaCrossAttentionInvocationContext) -> None:
        """Expose the installed child surface and initialize call records."""

        super().__init__()
        for name in (
            "q_proj",
            "q_norm",
            "k_proj",
            "k_norm",
            "v_proj",
            "output_proj",
            "output_dropout",
        ):
            setattr(self, name, nn.Identity())
        self._invocation = invocation
        self.invocations: list[AnimaRegionalBranchInvocation] = []

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        **_kwargs: object,
    ) -> torch.Tensor:
        """Expand the branch context value across its compact query grid."""

        self.invocations.append(self._invocation.require_current())
        return context[:, :1].expand(-1, int(query.shape[1]), -1)


class _FullRegionalPhaseContext(AnimaCompositionPhaseContext):
    """Publish full regional influence for tiled accumulation tests."""

    def require_current(self) -> AnimaCompositionPhase:
        """Return one stable specialization phase."""

        return AnimaCompositionPhase(
            AnimaCompositionStage.SPECIALIZATION, 0.5, True, 1.0
        )


@pytest.mark.parametrize("tile_batch_size", [1, 2, 4, 8])
@pytest.mark.parametrize(
    "diffusion_mode",
    ["multidiffusion", "mixture_of_diffusers"],
)
@pytest.mark.parametrize("layout", ["bchw", "bcdhw"])
def test_compact_regional_predictions_leave_outer_accumulation_unchanged(
    tile_batch_size: int,
    diffusion_mode: str,
    layout: str,
) -> None:
    """Match dense tiled fusion with one compact attention call per tile batch."""

    plan = build_tiled_diffusion_plan(16, 4, 6, 4, 2, tile_batch_size)
    bank = _mask_bank()
    execution = AnimaRegionalAttentionExecution(
        _contexts(1),
        bank,
        (1.0, 1.0),
        dynamic_contexts=True,
    )
    activation = AnimaActivationContext()
    invocation = AnimaCrossAttentionInvocationContext()
    original = _ContextValueAttention(invocation)
    patch = AnimaRegionalCrossAttentionPatch(
        original,
        execution,
        activation_context=activation,
        invocation_context=invocation,
        phase_context=_FullRegionalPhaseContext(),
    )
    bchw_canvas = torch.zeros((1, 1, 4, 16))
    canvas = bchw_canvas if layout == "bchw" else bchw_canvas.unsqueeze(2)
    args: dict[str, Any] = {
        "input": canvas,
        "timestep": torch.ones((1,)),
        "c": {"transformer_options": {}},
    }

    compact = TilePredictionAccumulator(
        plan,
        diffusion_mode=diffusion_mode,
    ).predict(
        args=args,
        x=canvas,
        evaluate=lambda tiled_args: _compact_prediction(
            tiled_args,
            patch=patch,
            execution=execution,
            activation=activation,
        ),
    )
    dense = TilePredictionAccumulator(
        plan,
        diffusion_mode=diffusion_mode,
    ).predict(
        args=args,
        x=canvas,
        evaluate=lambda tiled_args: _dense_prediction(tiled_args, bank=bank),
    )

    torch.testing.assert_close(compact, dense)
    assert compact.shape == canvas.shape
    if layout == "bcdhw":
        assert int(compact.shape[2]) == 1
    assert len(original.invocations) == len(plan.batches)
    assert sum(call.packed_batch_size for call in original.invocations) == 6
    assert sum(call.source_batch_size for call in original.invocations) == len(
        plan.tiles
    )


def _compact_prediction(
    tiled_args: dict[str, Any],
    *,
    patch: AnimaRegionalCrossAttentionPatch,
    execution: AnimaRegionalAttentionExecution,
    activation: AnimaActivationContext,
) -> torch.Tensor:
    """Run one real compact regional attention call and restore its tile rows."""

    tiled_input = cast(torch.Tensor, tiled_args["input"])
    layout = _layout(tiled_args)
    contexts = _contexts(layout.view_count)
    geometry = _geometry(layout)
    query = tiled_input.new_zeros(
        (layout.expanded_batch_size, geometry.query_token_count, 1)
    )
    with execution.execution_context.activate(contexts), activation.activate(geometry):
        prediction = patch(query, contexts.base_context)
    if not isinstance(prediction, torch.Tensor):
        raise TypeError("Compact Anima test prediction must be a tensor.")
    return prediction.transpose(1, 2).reshape_as(tiled_input)


def _dense_prediction(
    tiled_args: dict[str, Any],
    *,
    bank: RegionalMaskBank,
) -> torch.Tensor:
    """Return the complete unpruned regional blend for the same tiled call."""

    tiled_input = cast(torch.Tensor, tiled_args["input"])
    layout = _layout(tiled_args)
    geometry = _geometry(layout)
    masks = AnimaQueryMaskProjector().project(
        bank=bank,
        geometry=geometry,
        latent_batch_size=1,
        form=RegionalMaskForm.CONDITIONING,
        mode=RegionalMaskProjectionMode.CONTINUOUS_COVERAGE,
        device=tiled_input.device,
        dtype=tiled_input.dtype,
    )
    weights = ANIMA_CROSS_ATTENTION_WEIGHTING_POLICY
    normalized = weights.weights(
        masks.flattened,
        region_strengths=(1.0, 1.0),
    )
    batch = layout.expanded_batch_size
    query_count = geometry.query_token_count
    base = tiled_input.new_full((batch, query_count, 1), 10.0)
    regions = torch.stack(
        (
            tiled_input.new_full((batch, query_count, 1), 20.0),
            tiled_input.new_full((batch, query_count, 1), 30.0),
        )
    )
    prediction = weights.blend(
        weights=normalized,
        base_output=base,
        regional_outputs=regions,
    )
    return prediction.transpose(1, 2).reshape_as(tiled_input)


def _layout(tiled_args: dict[str, Any]) -> SpatialBatchLayout:
    """Return the authoritative spatial layout published by tile batching."""

    conditioning = cast(dict[str, Any], tiled_args["c"])
    options = cast(dict[str, Any], conditioning["transformer_options"])
    namespace = cast(dict[str, Any], options[SIMPLE_SYRUP_TRANSFORMER_NAMESPACE])
    layout = namespace[SPATIAL_BATCH_LAYOUT_KEY]
    if not isinstance(layout, SpatialBatchLayout):
        raise TypeError("Tiled test call requires a spatial batch layout.")
    return layout


def _geometry(layout: SpatialBatchLayout) -> AnimaActivationGeometry:
    """Map the latent-sized synthetic tile directly to Anima query geometry."""

    view = layout.views[0]
    return AnimaActivationGeometry(
        layout.expanded_batch_size,
        1,
        view.model_height,
        view.model_width,
        1,
        1,
        1,
        view.model_height,
        view.model_width,
        layout,
    )


def _contexts(batch_size: int) -> BatchedRegionalAttentionContexts:
    """Return recognizable base and region contexts in view-major order."""

    base = torch.full((batch_size, 1, 1), 10.0)
    region_a = torch.full((batch_size, 1, 1), 20.0)
    region_b = torch.full((batch_size, 1, 1), 30.0)
    return BatchedRegionalAttentionContexts(
        1,
        tuple(
            RegionalAttentionChunkBatch(
                index,
                RegionalAttentionBranch.POSITIVE,
                index,
                index + 1,
            )
            for index in range(batch_size)
        ),
        base,
        single_entry_regions((region_a, region_b)),
    )


def _mask_bank() -> RegionalMaskBank:
    """Return separated A and B regions with boundaries and uncovered tiles."""

    masks = torch.zeros((2, 4, 16))
    masks[0, :, 0:4] = 1.0
    masks[1, :, 5:10] = 1.0
    return RegionalMaskBank(masks.clone(), masks, 16, 4)
