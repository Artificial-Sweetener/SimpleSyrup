# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Prove mixed tiled Anima calls evaluate only supported attention rows."""

from __future__ import annotations

import torch
from regional_attention_test_values import single_entry_regions
from torch import nn

from simple_syrup.domain.regional_attention import RegionalAttentionBranch
from simple_syrup.domain.regional_attention_batch import (
    BatchedRegionalAttentionContexts,
    RegionalAttentionChunkBatch,
)
from simple_syrup.domain.regional_mask_bank import RegionalMaskBank
from simple_syrup.domain.spatial_views import (
    SpatialBatchLayout,
    SpatialView,
    SpatialViewKind,
)
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


class _ContextValueAttention(nn.Module):
    """Return one context scalar per row and record compact branch identity."""

    def __init__(self, context: AnimaCrossAttentionInvocationContext) -> None:
        """Install host children and initialize one empty call record."""

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
        self._context = context
        self.calls: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.invocations: list[AnimaRegionalBranchInvocation] = []

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        **_kwargs: object,
    ) -> torch.Tensor:
        """Expand each compact context value over its query tokens."""

        self.calls.append((query, context))
        self.invocations.append(self._context.require_current())
        return context[:, :1].expand(-1, int(query.shape[1]), -1)


class _FullRegionalPhaseContext(AnimaCompositionPhaseContext):
    """Publish full regional influence for tiled pruning tests."""

    def require_current(self) -> AnimaCompositionPhase:
        """Return one stable specialization phase."""

        return AnimaCompositionPhase(
            AnimaCompositionStage.SPECIALIZATION, 0.5, True, 1.0
        )


def test_mixed_tile_batch_prunes_only_unsupported_rows_without_output_drift() -> None:
    """Retain global rows and compact supported regional rows into one call."""

    masks = torch.zeros((2, 8, 10), dtype=torch.float32)
    masks[0, 0:4, 0:4] = 1.0
    masks[1, 0:4, 6:10] = 1.0
    bank = RegionalMaskBank(masks.clone(), masks.clone(), 10, 8)
    layout = SpatialBatchLayout(
        10,
        8,
        (
            _tile(0, 0),
            _tile(6, 0),
            _tile(3, 0),
            _tile(6, 4),
            _tile(2, 4),
        ),
        input_batch_size=1,
    )
    geometry = AnimaActivationGeometry(
        input_batch_size=5,
        activation_time=1,
        activation_height=4,
        activation_width=4,
        patch_temporal=1,
        patch_spatial=2,
        query_time=1,
        query_height=2,
        query_width=2,
        spatial_layout=layout,
    )
    base = _contexts(10.0)
    region_a = _contexts(20.0)
    region_b = _contexts(30.0)
    contexts = BatchedRegionalAttentionContexts(
        latent_batch_size=1,
        chunks=tuple(
            RegionalAttentionChunkBatch(
                index,
                RegionalAttentionBranch.POSITIVE,
                index,
                index + 1,
            )
            for index in range(5)
        ),
        base_context=base,
        regions=single_entry_regions((region_a, region_b)),
    )
    execution = AnimaRegionalAttentionExecution(contexts, bank, (1.0, 1.0))
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
    query = torch.zeros((5, 4, 1))

    with activation.activate(geometry):
        output = patch(query, base)

    assert len(original.calls) == 1
    packed_query, packed_context = original.calls[0]
    assert packed_query.shape == (9, 4, 1)
    assert packed_context[:, 0, 0].tolist() == [
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        20.0,
        22.0,
        31.0,
        32.0,
    ]
    assert original.invocations == [
        AnimaRegionalBranchInvocation(
            5,
            (0, 1, 2, 3, 4, 0, 2, 1, 2),
            (None, None, None, None, None, 0, 0, 1, 1),
        )
    ]

    projected = (
        AnimaQueryMaskProjector()
        .project(
            bank=bank,
            geometry=geometry,
            latent_batch_size=1,
            form=RegionalMaskForm.CONDITIONING,
            mode=RegionalMaskProjectionMode.CONTINUOUS_COVERAGE,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        .flattened
    )
    weighting = ANIMA_CROSS_ATTENTION_WEIGHTING_POLICY
    weights = weighting.weights(
        projected,
        region_strengths=(1.0, 1.0),
    )
    expected = weighting.blend(
        weights=weights,
        base_output=base[:, :1].expand(-1, 4, -1),
        regional_outputs=torch.stack(
            (
                region_a[:, :1].expand(-1, 4, -1),
                region_b[:, :1].expand(-1, 4, -1),
            )
        ),
    )
    torch.testing.assert_close(output, expected)


def _tile(x: int, y: int) -> SpatialView:
    """Return one identity-sized four-by-four tile."""

    return SpatialView(SpatialViewKind.TILE, x, y, 4, 4, 4, 4)


def _contexts(offset: float) -> torch.Tensor:
    """Return five recognizable one-token context rows."""

    return torch.arange(offset, offset + 5.0).reshape(5, 1, 1)
